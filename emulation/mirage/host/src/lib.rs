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
//! 1. Writes `node/0/pid` (its own pid) and an initial `health.json`
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

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use mirage_core::common::MaybeRef;
use mirage_core::container::{
    ContainerState, ENV_HEAD_ADDR, ENV_HEAD_PORT, ENV_RANK, container_name,
};
use mirage_core::emulator::{Emulator, EmulatorKind};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, InjectionDef, NodeStatus};
use mirage_core::paths::{ExecLayout, SessionLayout};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionDef, SessionHealth, SessionId};
use mirage_core::state::{read_json, read_json_opt, read_small_str, write_bytes, write_json};
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

    // Publish pid + initial health. The session host is node 0's host:
    // its pid and log live under `node/0`.
    let node0 = layout.node(0);
    std::fs::create_dir_all(&node0.root).map_err(|e| MirageError::Io {
        path: node0.root.clone(),
        source: e,
    })?;
    write_bytes(&node0.pid(), std::process::id().to_string().as_bytes())?;

    // If the session's profile is containerised, bring up one container
    // per node on a shared per-session network *before* declaring the
    // session ready. This pulls the image, creates the network, starts
    // the containers, and persists the runtime record + per-node cids so
    // execs run inside the right container and teardown can find them.
    if let Err(e) = maybe_bring_up_containers(&config.session, &layout) {
        // A containerised session that cannot start is fatal: surface it
        // as terminal health so clients stop waiting, and abort.
        let h = SessionHealth {
            timestamp: Utc::now(),
            healthy: false,
            state: Some("failed".to_string()),
            terminal: true,
            message: Some(format!("container bring-up failed: {e}")),
        };
        let _ = write_json(&layout.health(), &h);
        return Err(e);
    }

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
    // Tear down any per-node containers and the per-session network.
    // Idempotent and a no-op for non-containerised sessions; the control
    // plane also calls this on `session destroy`, so a crashed host never
    // leaks containers.
    mirage_core::container::teardown(&layout.container_json());
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

/// Read a session's on-disk definition and resolve its profile (whether
/// stored inline or referenced by name).
fn resolve_profile(session: &SessionId) -> Result<ProfileDef> {
    let session_def: SessionDef = read_json(&SessionLayout::for_id(session).def())?;
    match session_def.profile {
        MaybeRef::Owned(p) => Ok(p),
        MaybeRef::Ref(name) => {
            let p = mirage_core::paths::profile_path(&name);
            if !p.exists() {
                return Err(MirageError::ProfileNotFound(name));
            }
            read_json(&p)
        }
    }
}

/// Resolve the number of nodes a profile's topology describes (defaults
/// to 1 for the common single-node case).
fn resolve_node_count(profile: &ProfileDef) -> Result<u32> {
    let topology = match &profile.emulator.topology {
        MaybeRef::Owned(t) => t.clone(),
        MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
    };
    Ok(topology.total_nodes().max(1))
}

/// Pick an ephemeral TCP port by binding to `127.0.0.1:0` and reading
/// back the assigned port. Used as the head node's advertised port.
fn pick_head_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
        // 0 is a harmless fallback: it just means no port was reserved.
        .unwrap_or(0)
}

/// Bring up the per-node containers + network for a containerised
/// session and persist the runtime record. A no-op when the profile is
/// not containerised.
fn maybe_bring_up_containers(session: &SessionId, layout: &SessionLayout) -> Result<()> {
    let profile = resolve_profile(session)?;
    let Some(def) = profile.containerize.clone() else {
        return Ok(());
    };

    publish_health(
        layout,
        true,
        "pulling",
        Some(format!("pulling image {}", def.image)),
    )?;

    let engine =
        mirage_container::Engine::resolve(&def).map_err(|e| MirageError::other(format!("{e}")))?;
    let node_count = resolve_node_count(&profile)?;
    let head_port = pick_head_port();
    let head_addr = container_name(session, 0);

    let (state, cids) = engine
        .bring_up(session, &def, node_count, head_port, |rank| {
            node_mirage_env(rank, &head_addr, head_port)
        })
        .map_err(|e| MirageError::other(format!("{e}")))?;

    // Persist the runtime record (used by execs and teardown) and the
    // per-node container ids under each node's runtime directory.
    write_json(&layout.container_json(), &state)?;
    for (rank, cid) in &cids {
        let nlayout = layout.node(*rank);
        std::fs::create_dir_all(&nlayout.root).map_err(|e| MirageError::Io {
            path: nlayout.root.clone(),
            source: e,
        })?;
        write_bytes(&nlayout.cid(), cid.as_bytes())?;
    }
    Ok(())
}

/// Build the always-present mirage environment for a node of `rank`.
///
/// `MIRAGE_RANK` and `MIRAGE_HEAD_PORT` are set on every node.
/// `MIRAGE_HEAD_ADDR` is also set on every node: worker nodes get the
/// head's address, while the head (rank 0) gets `localhost` since it is
/// itself the head. These are injected whether or not the session is
/// containerised.
fn node_mirage_env(rank: u32, head_addr: &str, head_port: u16) -> Vec<(String, String)> {
    let head = if rank == 0 { "localhost" } else { head_addr };
    vec![
        (ENV_RANK.to_string(), rank.to_string()),
        (ENV_HEAD_PORT.to_string(), head_port.to_string()),
        (ENV_HEAD_ADDR.to_string(), head.to_string()),
    ]
}

/// Resolve the emulator-level injection for a session by reading its
/// on-disk definition, resolving its profile, and dispatching to the
/// configured [`EmulatorKind`]'s [`EmulatorBackend`] implementation to
/// compute the env vars / `LD_PRELOAD` it needs.
///
/// Returns an empty [`InjectionDef`] for emulators that need no
/// injection (`noop`). Errors when a configured emulator cannot produce
/// its required assets or its runtime library is missing, so a
/// misconfigured session fails loudly instead of silently running
/// unemulated.
fn resolve_injection(session: &SessionId) -> Result<InjectionDef> {
    let profile = resolve_profile(session)?;
    match profile.emulator.emulator {
        EmulatorKind::Rocjitsu => mirage_rocjitsu::Rocjitsu::new(profile).injection_def(),
        EmulatorKind::Hotswap => mirage_hotswap::Hotswap::new(profile).injection_def(),
        EmulatorKind::Noop => mirage_core::emulator::Noop::new(profile).injection_def(),
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

    // Resolve how many nodes to spawn and how they reach the head.
    //
    // * Containerised sessions: the container bring-up already decided
    //   the node count and head port and recorded them, so we read that
    //   back and run each node's command *inside* its container via the
    //   provider's `exec`. The head is reachable by its container name.
    // * Non-containerised sessions: the node count comes from the
    //   profile's topology (defaulting to 1), the head listens on
    //   loopback, and we pick an ephemeral port per exec.
    let session_layout = SessionLayout::for_id(&def.session);
    let container_state: Option<ContainerState> = read_json_opt(&session_layout.container_json())?;
    let (node_count, head_port, head_addr) = match &container_state {
        Some(state) => (
            state.nodes.len().max(1) as u32,
            state.head_port,
            container_name(&def.session, 0),
        ),
        None => {
            let profile = resolve_profile(&def.session)?;
            (
                resolve_node_count(&profile)?,
                pick_head_port(),
                "127.0.0.1".to_string(),
            )
        }
    };

    let mut handles = Vec::new();
    for rank in 0..node_count {
        let nlayout = layout.node(rank);
        std::fs::create_dir_all(&nlayout.root).map_err(|e| MirageError::Io {
            path: nlayout.root.clone(),
            source: e,
        })?;
        // Per-node container target, if containerised.
        let container = container_state.as_ref().and_then(|state| {
            state
                .nodes
                .iter()
                .find(|n| n.rank == rank)
                .map(|n| ContainerExec {
                    provider: state.provider.clone(),
                    name: n.name.clone(),
                })
        });
        let mirage_env = node_mirage_env(rank, &head_addr, head_port);
        match spawn_node(
            &def,
            rank,
            nlayout.clone(),
            &injection,
            &mirage_env,
            container,
        ) {
            Ok(node) => {
                status.started = true;
                status.nodes.insert(
                    rank,
                    NodeStatus {
                        pid: Some(node.pid),
                        exit_code: None,
                    },
                );
                handles.push((rank, node, nlayout));
            }
            Err(e) => {
                // Spawning the node failed (e.g. the command doesn't
                // exist). Rather than leaving the exec stuck in a
                // perpetual "started but never ended" state, surface the
                // reason on the node's stderr (which attach clients tail)
                // and record the conventional 127 "command not found"
                // exit code for this node.
                let msg = format!("mirage: {e}\n");
                let _ = std::fs::write(nlayout.stderr(), msg.as_bytes());
                let _ = write_bytes(&nlayout.exit_code(), b"127");
                status.started = true;
                status.nodes.insert(
                    rank,
                    NodeStatus {
                        pid: None,
                        exit_code: Some(127),
                    },
                );
            }
        }
    }
    write_json(&layout.status(), &status)?;

    let mut overall_code: i32 = status
        .nodes
        .values()
        .filter_map(|n| n.exit_code)
        .fold(0, |acc, c| if c.abs() > acc.abs() { c } else { acc });
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

/// Target container for running a node's command, when the session is
/// containerised: the resolved provider binary plus the node's
/// container name.
struct ContainerExec {
    provider: String,
    name: String,
}

fn spawn_node(
    def: &ExecDef,
    node: u32,
    nlayout: mirage_core::paths::NodeLayout,
    injection: &InjectionDef,
    mirage_env: &[(String, String)],
    container: Option<ContainerExec>,
) -> Result<SpawnedNode> {
    // Pick the args for this node. The head (rank 0) runs `def.exec`;
    // workers run `worker_exec` when set, otherwise they reuse the head's
    // command so a multi-node session runs the same workload everywhere
    // unless a distinct worker command was provided.
    let args = if node == 0 {
        &def.exec
    } else {
        def.worker_exec.as_ref().unwrap_or(&def.exec)
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

    // Build the combined environment that the workload should see. The
    // emulator injection is applied first, then the user's exec env, then
    // the always-present mirage variables (rank/head addr/port), so the
    // mirage variables can't be accidentally clobbered. `LD_PRELOAD` is
    // special-cased so the emulator's interposer is preserved alongside
    // any user-supplied value rather than one clobbering the other.
    let mut workload_env: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in &injection.env {
        workload_env.insert(k.clone(), v.clone());
    }
    for (k, v) in &args.env {
        workload_env.insert(k.clone(), v.clone());
    }
    for (k, v) in mirage_env {
        workload_env.insert(k.clone(), v.clone());
    }
    if let Some(preload) = &injection.ld_preload {
        let combined = match args.env.get("LD_PRELOAD") {
            Some(user) if !user.is_empty() => format!("{preload}:{user}"),
            _ => preload.clone(),
        };
        workload_env.insert("LD_PRELOAD".to_string(), combined);
    }

    let mut cmd = if let Some(c) = &container {
        // Containerised: run the workload *inside* the node's container
        // via the provider's `exec`. The environment is passed explicitly
        // with `-e` (so it lands inside the container) rather than
        // inherited from the host, and the working directory is resolved
        // inside the container with `-w`.
        let env_pairs: Vec<(String, String)> = workload_env.into_iter().collect();
        let argv = mirage_container::Engine::exec_argv(
            &c.name,
            args.workdir.as_deref(),
            &env_pairs,
            &args.command,
            &args.args,
        );
        let mut cmd = tokio::process::Command::new(&c.provider);
        cmd.args(&argv)
            .stdin(Stdio::from(slave_in))
            .stdout(Stdio::from(slave_out))
            .stderr(Stdio::from(slave_err))
            .env_clear()
            // The provider binary itself only needs a minimal host env to
            // find its config/socket; the workload's env travels in argv.
            .envs(std::env::vars().filter(|(k, _)| {
                matches!(
                    k.as_str(),
                    "PATH" | "HOME" | "USER" | "XDG_RUNTIME_DIR" | "DOCKER_HOST" | "CONTAINER_HOST"
                )
            }))
            .env(
                "TERM",
                std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()),
            );
        cmd
    } else {
        // Non-containerised: run the workload directly on the host with a
        // minimal inherited environment plus the computed workload env.
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
            .env(
                "TERM",
                std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()),
            )
            .envs(&workload_env);
        if let Some(wd) = &args.workdir {
            cmd.current_dir(wd);
        }
        cmd
    };

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
