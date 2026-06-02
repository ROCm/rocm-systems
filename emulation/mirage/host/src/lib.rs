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
//!    FIFO, then spawns one child process per node attached to a
//!    pseudo-terminal (PTY). The child's stdin/stdout/stderr are wired
//!    to the PTY slave so interactive programs (a shell, a REPL) get a
//!    real terminal: line editing, prompts, and input echo all work.
//!    The host bridges the PTY: it pumps the stdin FIFO into the PTY
//!    master and the PTY master's output into the node's `stdout` file
//!    (which attached clients tail). Writes per-node `pid` and, on
//!    exit, `exit_code`.
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
use mirage_core::common::MaybeRef;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, InjectionDef, NodeStatus};
use mirage_core::paths::{ExecLayout, SessionLayout};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionDef, SessionHealth, SessionId};
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

        // Handle pending signal requests across all execs.
        process_signal_requests(&layout);

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
        signal_exec_nodes(&exec_layout, sig);
    }
}

/// Forward `sig` to every node process that has published a pid file.
fn signal_exec_nodes(exec_layout: &ExecLayout, sig: nix::sys::signal::Signal) {
    let Ok(nodes) = std::fs::read_dir(exec_layout.node_root()) else {
        return;
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

/// Scan every exec directory for a pending `signal` request file. Each
/// file holds a single signal number as text; the host parses it,
/// forwards the signal to every node pid in that exec, and removes
/// the file. Removal is the consumed-acknowledgement and is what
/// distinguishes "signal handled" from "signal still pending" to
/// outside observers.
fn process_signal_requests(layout: &SessionLayout) {
    let Ok(rd) = std::fs::read_dir(layout.exec_root()) else {
        return;
    };
    for e in rd.flatten() {
        let exec_layout = ExecLayout::for_root(e.path());
        let signal_path = exec_layout.signal();
        let Ok(Some(raw)) = read_small_str(&signal_path) else {
            continue;
        };
        let Ok(num) = raw.parse::<i32>() else {
            // bogus payload: drop it so we don't loop forever.
            let _ = std::fs::remove_file(&signal_path);
            continue;
        };
        match nix::sys::signal::Signal::try_from(num) {
            Ok(sig) => signal_exec_nodes(&exec_layout, sig),
            Err(_) => {
                tracing::warn!("ignoring invalid signal request {num}");
            }
        }
        let _ = std::fs::remove_file(&signal_path);
    }
}

/// Resolve the emulator-level injection for a session by reading its
/// on-disk definition, resolving its profile, and computing the env
/// vars / `LD_PRELOAD` the configured emulator backend needs.
///
/// Returns an empty [`InjectionDef`] for emulators that need no
/// injection (e.g. `noop`). Errors only when a configured emulator
/// cannot produce its required assets (e.g. rocjitsu can't build its
/// simulation config), so a misconfigured session fails loudly instead
/// of silently running unemulated.
fn resolve_injection(session: &SessionId) -> Result<InjectionDef> {
    let session_def: SessionDef = read_json(&SessionLayout::for_id(session).def())?;
    let profile: ProfileDef = match session_def.profile {
        MaybeRef::Owned(p) => p,
        MaybeRef::Ref(name) => {
            let p = mirage_core::paths::profile_path(&name);
            if !p.exists() {
                return Err(MirageError::ProfileNotFound(name));
            }
            read_json(&p)?
        }
    };
    let emulator = &profile.emulator;

    match emulator.emulator.as_str() {
        "rocjitsu" => {
            // Extract embedded assets if they aren't on disk yet, then
            // compute the preload + simulation config for this profile.
            if let Err(e) = mirage_rocjitsu::ensure_assets(false) {
                tracing::warn!("rocjitsu: failed to extract assets: {e:#}");
            }
            let (config, schema) = match mirage_rocjitsu::kmd_config(emulator) {
                Ok(paths) => paths,
                Err(e) => {
                    // Can't build the sim config (e.g. rocjitsu assets not
                    // available on this machine). Warn and run unemulated
                    // rather than failing the exec outright.
                    tracing::warn!(
                        "rocjitsu: could not build simulation config; \
                         running without emulation env: {e:#}"
                    );
                    return Ok(InjectionDef::default());
                }
            };
            let mut env = std::collections::BTreeMap::new();
            env.insert("RJ_CONFIG".to_string(), config.display().to_string());
            env.insert("RJ_SCHEMA".to_string(), schema.display().to_string());
            let ld_preload = mirage_rocjitsu::kmd_preload().map(|p| p.display().to_string());
            if ld_preload.is_none() {
                tracing::warn!(
                    "rocjitsu: no KMD preload library found; workload will not be emulated"
                );
            }
            Ok(InjectionDef {
                wrapper: None,
                ld_preload,
                files: Default::default(),
                env,
            })
        }
        // `noop` and any other emulator currently need no injection.
        _ => Ok(InjectionDef::default()),
    }
}

async fn run_exec(layout: ExecLayout) -> Result<()> {
    let def: ExecDef = read_json(&layout.def())?;
    // Resolve the emulator-level injection (env vars, LD_PRELOAD, ...)
    // for this exec's session. For a rocjitsu session this is what wires
    // `LD_PRELOAD=librocjitsu_kmd.so` plus `RJ_CONFIG`/`RJ_SCHEMA` into
    // every spawned child so the workload actually runs under emulation.
    let injection = resolve_injection(&def.session)?;
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
    let head = match spawn_node(&def, 0, head_layout.clone(), &injection) {
        Ok(head) => head,
        Err(e) => {
            // Spawning the node failed (e.g. the command doesn't exist).
            // Rather than leaving the exec stuck in a perpetual "started
            // but never ended" state, surface the reason on the node's
            // stderr (which attach clients tail) and finish the exec with
            // the conventional 127 "command not found" exit code.
            let msg = format!("mirage: {e}\n");
            let _ = std::fs::write(head_layout.stderr(), msg.as_bytes());
            let _ = write_bytes(&head_layout.exit_code(), b"127");
            status.started = true;
            status.ended = true;
            status.ended_at = Some(Utc::now());
            status.exit_code = Some(127);
            status.nodes.insert(
                0,
                NodeStatus {
                    pid: None,
                    exit_code: Some(127),
                },
            );
            write_json(&layout.status(), &status)?;
            if !def.keep {
                let _ = std::fs::remove_dir_all(&layout.root);
            }
            return Ok(());
        }
    };
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
    /// Background task that bridges the node's PTY: stdin FIFO -> PTY
    /// master, and PTY master -> the node's `stdout` file. It owns the
    /// master fd and a keepalive writer on the FIFO; it finishes on its
    /// own when the child closes the slave (EOF on the master), and is
    /// aborted by [`wait_node`] as a backstop once the child has exited.
    bridge: tokio::task::JoinHandle<()>,
}

fn spawn_node(
    def: &ExecDef,
    node: u32,
    nlayout: mirage_core::paths::NodeLayout,
    injection: &InjectionDef,
) -> Result<SpawnedNode> {
    // Pick the args for this node.
    let args = if node == 0 {
        &def.exec
    } else if let Some(w) = &def.worker_exec {
        w
    } else {
        return Err(MirageError::other("no worker exec for non-head node"));
    };

    // Create FIFO for stdin. The control plane (`session_stdin`) writes
    // user keystrokes here; the bridge task forwards them into the PTY.
    let stdin_path = nlayout.stdin();
    if !stdin_path.exists() {
        mkfifo(&stdin_path, Mode::S_IRUSR | Mode::S_IWUSR)
            .map_err(|e| MirageError::other(format!("mkfifo {stdin_path:?}: {e}")))?;
    }
    // Create the stdout file (the bridge appends merged PTY output here)
    // and an empty stderr file (the PTY merges stderr into stdout, but
    // attach clients still tail the stderr path, so it must exist).
    let stdout_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(nlayout.stdout())
        .map_err(|e| MirageError::Io {
            path: nlayout.stdout(),
            source: e,
        })?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(nlayout.stderr())
        .map_err(|e| MirageError::Io {
            path: nlayout.stderr(),
            source: e,
        })?;

    // Allocate a pseudo-terminal. The child runs on the slave side so it
    // sees a real TTY (echo, line discipline, job control); the host
    // keeps the master side to pump bytes in and out.
    use std::os::fd::{AsRawFd, OwnedFd};
    let winsize = nix::pty::Winsize {
        ws_row: 24,
        ws_col: 80,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let pty = nix::pty::openpty(Some(&winsize), None)
        .map_err(|e| MirageError::other(format!("openpty: {e}")))?;
    let master: OwnedFd = pty.master;
    let slave: OwnedFd = pty.slave;

    // Wire all three child standard streams to the slave end.
    let slave_in = slave.try_clone().map_err(|e| MirageError::Io {
        path: stdin_path.clone(),
        source: e,
    })?;
    let slave_out = slave.try_clone().map_err(|e| MirageError::Io {
        path: nlayout.stdout(),
        source: e,
    })?;
    let slave_err = slave.try_clone().map_err(|e| MirageError::Io {
        path: nlayout.stderr(),
        source: e,
    })?;

    let mut cmd = tokio::process::Command::new(&args.command);
    cmd.args(&args.args)
        .stdin(Stdio::from(slave_in))
        .stdout(Stdio::from(slave_out))
        .stderr(Stdio::from(slave_err))
        .env_clear()
        // inherit a minimal environment by default
        .envs(std::env::vars().filter(|(k, _)| {
            matches!(
                k.as_str(),
                "PATH" | "HOME" | "USER" | "LANG" | "LC_ALL" | "TERM" | "TMPDIR"
            )
        }))
        // Make sure programs that consult $TERM behave like a real
        // terminal even if the host's own environment has none set.
        .env("TERM", std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()))
        // Emulator-provided env (e.g. rocjitsu's RJ_CONFIG / RJ_SCHEMA)
        // is applied before the user's env so an explicit exec env can
        // still override it if needed.
        .envs(&injection.env)
        .envs(&args.env);
    // `LD_PRELOAD` is special: the emulator's interposer must stay on the
    // preload list, so combine it with any user-supplied value rather than
    // letting one clobber the other.
    if let Some(preload) = &injection.ld_preload {
        let combined = match args.env.get("LD_PRELOAD") {
            Some(user) if !user.is_empty() => format!("{preload}:{user}"),
            _ => preload.clone(),
        };
        cmd.env("LD_PRELOAD", combined);
    }
    if let Some(wd) = &args.workdir {
        cmd.current_dir(wd);
    }

    // After fork, in the child: start a new session and make the PTY our
    // controlling terminal. By the time these closures run, the standard
    // streams (fd 0/1/2) already point at the slave, so the ioctl on fd 0
    // claims the right TTY.
    unsafe {
        cmd.pre_exec(|| {
            nix::unistd::setsid().map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
            if libc::ioctl(0, libc::TIOCSCTTY as libc::c_ulong, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = cmd.spawn().map_err(|e| match e.kind() {
        // Translate the common spawn failures into a clear, actionable
        // message instead of a raw OS error. argv[0] not existing is by
        // far the most common ("mirage run -- typo").
        std::io::ErrorKind::NotFound => {
            MirageError::other(format!("command not found: {}", args.command))
        }
        std::io::ErrorKind::PermissionDenied => {
            MirageError::other(format!("permission denied: {}", args.command))
        }
        _ => MirageError::Io {
            path: PathBuf::from(&args.command),
            source: e,
        },
    })?;
    // The host no longer needs the slave: drop every host-side copy so the
    // master observes EOF once the child exits and closes its own copies.
    drop(slave);
    let pid = child.id().unwrap_or(0);
    write_bytes(&nlayout.pid(), pid.to_string().as_bytes())?;

    // Open the stdin FIFO read end (non-blocking so open() doesn't block
    // waiting for a writer) plus a keepalive writer so the read end never
    // hits EOF when no client is currently sending input.
    let fifo_read: OwnedFd = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&stdin_path)
        .map_err(|e| MirageError::Io {
            path: stdin_path.clone(),
            source: e,
        })?
        .into();
    let fifo_keepalive: OwnedFd = std::fs::OpenOptions::new()
        .write(true)
        .open(&stdin_path)
        .map_err(|e| MirageError::Io {
            path: stdin_path.clone(),
            source: e,
        })?
        .into();

    // Make the master and FIFO read end non-blocking for AsyncFd.
    for fd in [master.as_raw_fd(), fifo_read.as_raw_fd()] {
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }

    let bridge = tokio::spawn(async move {
        // Keep the keepalive writer alive for the whole bridge lifetime.
        let _keepalive = fifo_keepalive;
        if let Err(err) = pump_pty(master, fifo_read, stdout_file).await {
            tracing::debug!("pty bridge ended: {err}");
        }
    });

    Ok(SpawnedNode { child, pid, bridge })
}

/// Read a fd into `buf`, mapping a negative return into the last OS error.
fn raw_read(fd: std::os::fd::RawFd, buf: &mut [u8]) -> std::io::Result<usize> {
    let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// Write `buf` to a fd, mapping a negative return into the last OS error.
fn raw_write(fd: std::os::fd::RawFd, buf: &[u8]) -> std::io::Result<usize> {
    let n = unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, buf.len()) };
    if n < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(n as usize)
    }
}

/// Bridge a node's PTY for its whole lifetime:
///   * PTY master -> the node's `stdout` file (so attach clients see
///     program output *and* the terminal's echo of typed input);
///   * stdin FIFO -> PTY master (so forwarded keystrokes reach the
///     program through the terminal line discipline).
///
/// Returns when the master reports EOF, i.e. the child has exited and
/// closed the slave.
async fn pump_pty(
    master: std::os::fd::OwnedFd,
    fifo_read: std::os::fd::OwnedFd,
    mut stdout_file: std::fs::File,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use tokio::io::unix::AsyncFd;

    let master = AsyncFd::new(master)?;
    let fifo = AsyncFd::new(fifo_read)?;
    let mut obuf = [0u8; 8192];
    let mut ibuf = [0u8; 8192];

    loop {
        tokio::select! {
            // PTY output -> stdout file.
            guard = master.readable() => {
                let mut guard = guard?;
                match guard.try_io(|inner| raw_read(inner.get_ref().as_raw_fd(), &mut obuf)) {
                    Ok(Ok(0)) => break, // child exited; slave closed
                    Ok(Ok(n)) => {
                        stdout_file.write_all(&obuf[..n])?;
                        stdout_file.flush()?;
                    }
                    // EIO on the master means the slave is gone (child exited).
                    Ok(Err(e)) => {
                        if e.raw_os_error() == Some(libc::EIO) {
                            break;
                        }
                        return Err(e);
                    }
                    Err(_would_block) => {}
                }
            }
            // stdin FIFO -> PTY master.
            guard = fifo.readable() => {
                let mut guard = guard?;
                match guard.try_io(|inner| raw_read(inner.get_ref().as_raw_fd(), &mut ibuf)) {
                    Ok(Ok(0)) => {} // no writers momentarily; keepalive keeps us open
                    Ok(Ok(n)) => write_all_to_master(&master, &ibuf[..n]).await?,
                    Ok(Err(e)) => return Err(e),
                    Err(_would_block) => {}
                }
            }
        }
    }
    Ok(())
}

/// Write every byte of `data` to the PTY master, honoring non-blocking
/// back-pressure via the `AsyncFd` writable readiness.
async fn write_all_to_master(
    master: &tokio::io::unix::AsyncFd<std::os::fd::OwnedFd>,
    mut data: &[u8],
) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    while !data.is_empty() {
        let mut guard = master.writable().await?;
        match guard.try_io(|inner| raw_write(inner.get_ref().as_raw_fd(), data)) {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => data = &data[n..],
            Ok(Err(e)) => return Err(e),
            Err(_would_block) => continue,
        }
    }
    Ok(())
}

async fn wait_node(node: SpawnedNode) -> i32 {
    let SpawnedNode {
        mut child, bridge, ..
    } = node;
    let code = match child.wait().await {
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
    };
    // The bridge normally finishes on its own when the master hits EOF;
    // give it a brief moment to flush trailing output, then abort.
    let _ = tokio::time::timeout(Duration::from_millis(200), &mut { bridge }).await;
    code
}

// silence unused-import in case OS-specific items get conditionally compiled.
use std::os::unix::fs::OpenOptionsExt;
