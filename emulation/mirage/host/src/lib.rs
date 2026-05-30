//! `mirage_host`: the per-session host process.
//!
//! See `host/src/main.rs` for the CLI entry point. The interesting
//! logic lives in this library so that integration tests can drive
//! the host directly without spawning a subprocess.
//!
//! # Overview
//!
//! A host owns exactly one session directory. While running, it:
//!
//! 1. Writes `host.pid` (its own pid) and an initial `health.json`
//!    (`healthy=true`, `state="ready"`) once startup is complete.
//! 2. Polls `exec/` for new exec definitions (i.e. directories that
//!    have a `def.json` but no `status.json` yet).
//! 3. For each new exec, creates per-node directories with a stdin
//!    FIFO, then spawns one child process per node wired to those
//!    files. Writes per-node `pid` and, on exit, `exit_code`.
//! 4. Aggregates per-node exits into `status.json` (`ended=true` +
//!    overall `exit_code`).
//! 5. On `SIGTERM`/`SIGINT`, marks the session unhealthy, signals all
//!    in-flight execs, and exits.
//!
//! Polling is used (rather than `inotify`) because filesystem
//! notifications are notoriously platform-dependent; a 50ms cadence
//! gives interactive-feeling latency without burning CPU.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, NodeStatus};
use mirage_core::paths::{ExecLayout, SessionLayout};
use mirage_core::session::{SessionHealth, SessionId};
use mirage_core::state::{read_json, read_small_str, write_bytes, write_json};
use nix::sys::stat::Mode;
use nix::unistd::mkfifo;
use tokio::sync::Notify;

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Configuration for running a host.
#[derive(Debug, Clone)]
pub struct HostConfig {
    pub session: SessionId,
}

/// Run the host for `session_id` until shutdown is signalled.
///
/// `shutdown` lets callers (such as tests) ask the host to exit
/// cleanly. SIGTERM/SIGINT also trigger shutdown automatically.
pub async fn run(config: HostConfig, shutdown: Arc<Notify>) -> Result<()> {
    let layout = SessionLayout::for_id(&config.session);
    if !layout.def().exists() {
        return Err(MirageError::SessionNotFound(config.session.to_string()));
    }

    // Publish pid + initial health.
    write_bytes(
        &layout.host_pid(),
        std::process::id().to_string().as_bytes(),
    )?;
    publish_health(&layout, true, "ready", None)?;

    // Install signal handlers.
    let sig_shutdown = shutdown.clone();
    tokio::spawn(async move {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM");
        let mut intr = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("install SIGINT");
        tokio::select! {
            _ = term.recv() => {}
            _ = intr.recv() => {}
        }
        sig_shutdown.notify_waiters();
    });

    let mut seen: HashSet<ExecId> = HashSet::new();
    let mut tasks: Vec<tokio::task::JoinHandle<()>> = Vec::new();

    loop {
        // Discover new execs.
        if let Ok(rd) = std::fs::read_dir(layout.exec_root()) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let Ok(eid) = ExecId::new(name.clone()) else {
                    continue;
                };
                if seen.contains(&eid) {
                    continue;
                }
                let exec_layout = layout.exec(&eid);
                if !exec_layout.def().exists() {
                    continue;
                }
                if exec_layout.status().exists() {
                    // already handled (possibly by a previous host run)
                    seen.insert(eid);
                    continue;
                }
                seen.insert(eid.clone());
                let exec_layout_clone = exec_layout.clone();
                tasks.push(tokio::spawn(async move {
                    if let Err(err) = run_exec(exec_layout_clone).await {
                        tracing::error!("exec failed: {err}");
                    }
                }));
            }
        }

        // Wait for either shutdown or the next poll tick.
        let woke = tokio::select! {
            _ = shutdown.notified() => true,
            _ = tokio::time::sleep(POLL_INTERVAL) => false,
        };
        if woke {
            break;
        }
    }

    // Shutdown path: mark unhealthy and signal in-flight execs.
    publish_health(&layout, false, "stopping", None).ok();
    signal_all_execs(&layout, nix::sys::signal::Signal::SIGTERM);

    // Give children a moment to exit, then cancel polling tasks.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    for t in tasks.drain(..) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let _ = tokio::time::timeout(remaining.max(Duration::from_millis(10)), t).await;
    }
    signal_all_execs(&layout, nix::sys::signal::Signal::SIGKILL);
    publish_health(&layout, false, "stopped", None).ok();
    Ok(())
}

fn publish_health(
    layout: &SessionLayout,
    healthy: bool,
    state: &str,
    message: Option<String>,
) -> Result<()> {
    let h = SessionHealth {
        timestamp: Utc::now(),
        healthy,
        state: Some(state.to_string()),
        terminal: false,
        message,
    };
    write_json(&layout.health(), &h)
}

fn signal_all_execs(layout: &SessionLayout, sig: nix::sys::signal::Signal) {
    let Ok(rd) = std::fs::read_dir(layout.exec_root()) else {
        return;
    };
    for e in rd.flatten() {
        let exec_layout = ExecLayout::for_root(e.path());
        let Ok(nodes) = std::fs::read_dir(exec_layout.node_root()) else {
            continue;
        };
        for n in nodes.flatten() {
            let pid_path = n.path().join("pid");
            if let Ok(Some(s)) = read_small_str(&pid_path)
                && let Ok(pid) = s.parse::<i32>()
            {
                let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), sig);
            }
        }
    }
}

async fn run_exec(layout: ExecLayout) -> Result<()> {
    let def: ExecDef = read_json(&layout.def())?;
    let mut status = ExecStatus {
        started: false,
        ended: false,
        exit_code: None,
        started_at: Some(Utc::now()),
        ended_at: None,
        nodes: Default::default(),
    };
    write_json(&layout.status(), &status)?;

    // Determine node count. For now: head is node 0; if worker_exec is
    // set, spawn one worker on nodes 1..N where N comes from the
    // session's profile. We don't have direct access to the profile
    // here without a parse, so just spawn head-only for now and let
    // workers be added when we wire emulator-level injection.
    let mut handles = Vec::new();
    let head_layout = layout.node(0);
    std::fs::create_dir_all(&head_layout.root).map_err(|e| MirageError::Io {
        path: head_layout.root.clone(),
        source: e,
    })?;
    let head = spawn_node(&def, 0, head_layout.clone())?;
    status.started = true;
    status.nodes.insert(
        0,
        NodeStatus {
            pid: Some(head.pid),
            exit_code: None,
        },
    );
    write_json(&layout.status(), &status)?;
    handles.push((0u32, head, head_layout));

    let mut overall_code: i32 = 0;
    for (node, child, nlayout) in handles {
        let code = wait_node(child).await;
        write_bytes(&nlayout.exit_code(), code.to_string().as_bytes())?;
        status.nodes.insert(
            node,
            NodeStatus {
                pid: status.nodes.get(&node).and_then(|s| s.pid),
                exit_code: Some(code),
            },
        );
        if code.abs() > overall_code.abs() {
            overall_code = code;
        }
    }
    status.ended = true;
    status.ended_at = Some(Utc::now());
    status.exit_code = Some(overall_code);
    write_json(&layout.status(), &status)?;

    // If the exec is keep=false, remove the directory.
    if !def.keep {
        // best effort; if attached clients are still tailing they'll
        // gracefully shutdown when the dir disappears.
        let _ = std::fs::remove_dir_all(&layout.root);
    }
    Ok(())
}

struct SpawnedNode {
    child: tokio::process::Child,
    pid: u32,
    #[allow(dead_code)]
    stdin_path: PathBuf,
}

fn spawn_node(
    def: &ExecDef,
    node: u32,
    nlayout: mirage_core::paths::NodeLayout,
) -> Result<SpawnedNode> {
    // Pick the args for this node.
    let args = if node == 0 {
        &def.exec
    } else if let Some(w) = &def.worker_exec {
        w
    } else {
        return Err(MirageError::other("no worker exec for non-head node"));
    };

    // Create FIFO for stdin.
    let stdin_path = nlayout.stdin();
    if !stdin_path.exists() {
        mkfifo(&stdin_path, Mode::S_IRUSR | Mode::S_IWUSR)
            .map_err(|e| MirageError::other(format!("mkfifo {stdin_path:?}: {e}")))?;
    }
    // Open stdout/stderr files.
    let stdout_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(nlayout.stdout())
        .map_err(|e| MirageError::Io {
            path: nlayout.stdout(),
            source: e,
        })?;
    let stderr_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(nlayout.stderr())
        .map_err(|e| MirageError::Io {
            path: nlayout.stderr(),
            source: e,
        })?;

    // Open the FIFO for reading on the child side. We use a writer fd
    // briefly to ensure read end doesn't return EOF immediately.
    // tokio's Command will dup this fd for the child.
    use std::os::fd::OwnedFd;
    let read_fd: OwnedFd = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&stdin_path)
        .map_err(|e| MirageError::Io {
            path: stdin_path.clone(),
            source: e,
        })?
        .into();
    // Clear the O_NONBLOCK we used to avoid blocking on open()-for-read.
    unsafe {
        let raw = std::os::fd::AsRawFd::as_raw_fd(&read_fd);
        let flags = libc::fcntl(raw, libc::F_GETFL);
        libc::fcntl(raw, libc::F_SETFL, flags & !libc::O_NONBLOCK);
    }

    let mut cmd = tokio::process::Command::new(&args.command);
    cmd.args(&args.args)
        .stdin(Stdio::from(read_fd))
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .env_clear()
        // inherit a minimal environment by default
        .envs(std::env::vars().filter(|(k, _)| {
            matches!(
                k.as_str(),
                "PATH" | "HOME" | "USER" | "LANG" | "LC_ALL" | "TERM" | "TMPDIR"
            )
        }))
        .envs(&args.env);
    if let Some(wd) = &args.workdir {
        cmd.current_dir(wd);
    }

    let child = cmd.spawn().map_err(|e| MirageError::Io {
        path: PathBuf::from(&args.command),
        source: e,
    })?;
    let pid = child.id().unwrap_or(0);
    write_bytes(&nlayout.pid(), pid.to_string().as_bytes())?;
    Ok(SpawnedNode {
        child,
        pid,
        stdin_path,
    })
}

async fn wait_node(node: SpawnedNode) -> i32 {
    let SpawnedNode { mut child, .. } = node;
    match child.wait().await {
        Ok(s) => {
            if let Some(c) = s.code() {
                c
            } else if let Some(sig) = std::os::unix::process::ExitStatusExt::signal(&s) {
                128 + sig
            } else {
                -1
            }
        }
        Err(_) => -1,
    }
}

// silence unused-import in case OS-specific items get conditionally compiled.
use std::os::unix::fs::OpenOptionsExt;
