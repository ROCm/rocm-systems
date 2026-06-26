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
    ContainerState, ENV_HEAD_ADDR, ENV_HEAD_PORT, ENV_LOCAL_RANK, ENV_MASTER_ADDR,
    ENV_MASTER_PORT, ENV_NCCL_HOSTID, ENV_RANK, ENV_TORCH_RANK, ENV_WORLD_SIZE, container_name,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecStatus, InjectionDef, NodeStatus};
use mirage_core::paths::{ExecLayout, SessionLayout};
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{
    HEALTH_HEARTBEAT_INTERVAL, SessionDef, SessionHealth, SessionId, session_uses_daemon,
};
use mirage_core::state::{read_json, read_json_opt, read_small_str, write_bytes, write_json};
use nix::sys::stat::Mode;
use nix::unistd::mkfifo;
use tokio::sync::Notify;

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Path the mirage binary is bind-mounted at inside every node
/// container; also the container's entrypoint (`mirage host`).
const CONTAINER_MIRAGE_BIN: &str = "/mnt/mirage/bin/mirage";

/// Mirage runtime root inside every node container. The session's
/// runtime directory is bind-mounted under here and `MIRAGE_RUNTIME`
/// points at it so the in-container host and the orchestrator share the
/// same on-disk session state.
const CONTAINER_RUNTIME_DIR: &str = "/mnt/mirage/runtime";

/// Mirage config root inside every node container. The orchestrator's
/// config directory is bind-mounted here read-only and `MIRAGE_CONFIG`
/// points at it so the in-container host can resolve profiles and
/// topologies that are stored *by reference* (rather than inline) in the
/// session definition.
const CONTAINER_CONFIG_DIR: &str = "/mnt/mirage/config";

/// Mirage state root inside every node container. Set via `MIRAGE_STATE`
/// so the in-container host resolves state at a deterministic location
/// instead of falling back to a non-existent in-container XDG path.
const CONTAINER_STATE_DIR: &str = "/mnt/mirage/state";

/// Directory inside every node container where host shared libraries are
/// bind-mounted (the emulator's interposer/`libraries` plus the host
/// `libc`/`libstdc++` the mirage binary was built against). Added to
/// `LD_LIBRARY_PATH` so the loader prefers them over the image's own,
/// possibly older, system libraries.
const CONTAINER_LIB_DIR: &str = "/mnt/mirage/lib";

/// Configuration for running a host.
#[derive(Debug, Clone)]
pub struct HostConfig {
    pub session: SessionId,
    /// Node rank this host serves, or `None` for the orchestrator host.
    ///
    /// * `None` — the orchestrator host, run on the real host by the
    ///   control plane. It brings up the per-node containers (for a
    ///   containerised session) and, for a *non*-containerised session,
    ///   runs every node's exec directly. For a containerised session it
    ///   delegates exec execution to the per-node hosts and just waits,
    ///   then tears the containers down.
    /// * `Some(rank)` — a per-node host, run *inside* a node's container
    ///   (the container's entrypoint is `mirage host --rank <rank>`). It
    ///   runs only its own node's execs as direct local children and
    ///   never manages containers.
    pub rank: Option<u32>,
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
    tracing::info!(session = %config.session, rank = ?config.rank, "host starting");

    // A per-node host runs inside a container and owns only its rank; the
    // orchestrator host runs on the real host and owns containers and
    // session-level health.
    let is_node_host = config.rank.is_some();
    let manages_session = !is_node_host;

    // Publish this host's pid. The orchestrator is node 0's host; a
    // per-node host records its own node's pid.
    let pid_rank = config.rank.unwrap_or(0);
    let pid_node = layout.node(pid_rank);
    std::fs::create_dir_all(&pid_node.root).map_err(|e| MirageError::Io {
        path: pid_node.root.clone(),
        source: e,
    })?;
    write_bytes(&pid_node.pid(), std::process::id().to_string().as_bytes())?;

    // Only the orchestrator brings up containers: it pulls the image,
    // creates the per-session network, starts one container per node
    // (each running its own `mirage host --rank <n>`), and persists the
    // runtime record + per-node cids. A per-node host is already inside
    // its container, so it skips this entirely.
    if manages_session {
        if let Err(e) = maybe_bring_up_containers(&config.session, &layout) {
            // A containerised session that cannot start is fatal: surface
            // it as terminal health so clients stop waiting, and abort.
            // `e` already names the phase that failed (e.g. "pulling
            // image … failed: <provider error>").
            let h = SessionHealth {
                timestamp: Utc::now(),
                healthy: false,
                state: Some("failed".to_string()),
                terminal: true,
                message: Some(e.to_string()),
            };
            let _ = write_json(&layout.health(), &h);
            return Err(e);
        }
        publish_health(&layout, true, "ready", None)?;
        tracing::info!(session = %config.session, "host ready");
    }

    // Whether *this* host runs execs locally. The orchestrator runs them
    // only for a non-containerised session (a containerised session's
    // execs are run by the per-node hosts inside the containers). A
    // per-node host always runs its own rank's execs.
    let containerized = layout.container_json().exists();
    let run_execs_here = is_node_host || !containerized;
    let host_rank = config.rank;

    // Host the emulator daemon for the node this host serves. A backend
    // that emulates the GPU out-of-process (rocjitsu) stands up its
    // daemon here — one per node host — and we keep the handle alive for
    // the host's lifetime; dropping it on shutdown tears the daemon down.
    // Only a host that actually runs execs (a per-node in-container host,
    // or the orchestrator of a non-containerised session) hosts a daemon,
    // because that is where the workload — and thus the
    // `ROCJITSU_RUNTIME_DIR` the daemon binds its socket under — lives.
    //
    // Daemon mode is the default per session; pass `mirage run
    // --in-process` to opt into in-process (local mode) instead, where
    // the interposer reads the session's local config directly and no
    // daemon socket is bound. Daemon mode is required for cross-process
    // GPU memory sharing (e.g. multi-GPU RCCL collectives).
    let mut emulator_daemon = if run_execs_here && session_uses_daemon(&config.session) {
        start_emulator_daemon(&config.session)
    } else {
        None
    };

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

    // Heartbeat: the orchestrator re-stamps session health periodically
    // so readers can tell a live host from a crashed one. A host that
    // stops beating is reported as `stalled` and then terminally `dead`
    // (see `SessionHealth::escalate_if_stale`). Per-node hosts do not own
    // session-level health, so only the orchestrator beats.
    let mut last_heartbeat = tokio::time::Instant::now();

    loop {
        // Detect external destruction. When `session destroy` removes our
        // session directory out from under us, our `def.json` disappears.
        // Exit immediately and *without* re-publishing health, because a
        // heartbeat write would recreate the directory (its parent is
        // created on demand) and the "stopped" session would silently
        // reappear in `session list`. Only the session-owning orchestrator
        // makes this decision; a per-node host follows the orchestrator.
        if manages_session && !layout.def().exists() {
            tracing::info!(
                session = %config.session,
                "session directory removed; host exiting"
            );
            return Ok(());
        }

        // Discover new execs. The orchestrator host of a containerised
        // session does not run execs itself (the per-node hosts do), so
        // it skips discovery and merely waits for shutdown.
        if run_execs_here {
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
                    // Skip execs this host has already handled (e.g. across
                    // a host restart). A per-node host keys this off *its
                    // own* node result, not the exec-wide `status.json`: in
                    // a multi-node containerised session that shared status
                    // is written by the rank-0 aggregator, so keying off it
                    // would make a worker (rank > 0) treat the exec as done
                    // before running its own rank — leaving its `exit_code`
                    // unwritten and the aggregator waiting on it forever.
                    let already_handled = match host_rank {
                        Some(rank) => exec_layout.node(rank).exit_code().exists(),
                        None => exec_layout.status().exists(),
                    };
                    if already_handled {
                        seen.insert(eid);
                        continue;
                    }
                    seen.insert(eid.clone());
                    tracing::info!(exec = %eid, "discovered new exec");
                    let exec_layout_clone = exec_layout.clone();
                    tasks.push(tokio::spawn(async move {
                        if let Err(err) = run_exec(exec_layout_clone, host_rank).await {
                            tracing::error!("exec failed: {err}");
                        }
                    }));
                }
            }
        }

        // Handle pending signal requests across all execs.
        process_signal_requests(&layout);

        // Re-stamp the heartbeat so readers know the host is still alive.
        if manages_session && last_heartbeat.elapsed() >= HEALTH_HEARTBEAT_INTERVAL {
            publish_health(&layout, true, "ready", None).ok();
            last_heartbeat = tokio::time::Instant::now();
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

    // Shutdown path: mark unhealthy and signal in-flight execs. Only
    // re-publish health while the session directory still exists; if it
    // was destroyed under us, writing health would recreate the directory
    // and resurrect the session in listings.
    tracing::info!(session = %config.session, "host shutting down");
    if manages_session && layout.def().exists() {
        publish_health(&layout, false, "stopping", None).ok();
    }
    signal_all_execs(&layout, nix::sys::signal::Signal::SIGTERM);

    // Give children a moment to exit, then cancel polling tasks.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    for t in tasks.drain(..) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let _ = tokio::time::timeout(remaining.max(Duration::from_millis(10)), t).await;
    }
    signal_all_execs(&layout, nix::sys::signal::Signal::SIGKILL);
    // Stop the emulator daemon after the workload children are gone, so
    // the simulated device outlives every process that talks to it.
    if let Some(daemon) = emulator_daemon.take() {
        daemon.stop();
    }
    // Tear down any per-node containers and the per-session network.
    // Only the orchestrator owns the containers, so only it tears them
    // down. Idempotent and a no-op for non-containerised sessions; the
    // control plane also calls this on `session destroy`, so a crashed
    // host never leaks containers.
    if manages_session {
        mirage_core::container::teardown(&layout.container_json());
        if layout.def().exists() {
            publish_health(&layout, false, "stopped", None).ok();
        }
    }
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
    let Some(mut def) = profile.containerize.clone() else {
        return Ok(());
    };

    // Merge the emulator-declared mounts (e.g. HotSwap's runtime + cache
    // trees, bind-mounted under `/mnt/mirage`) and GPU device/group
    // passthrough into the node containers so the injected
    // `LD_PRELOAD`/env paths resolve and the workload can reach the GPU.
    let injection = resolve_injection(session)?;
    def.mounts.extend(injection.mounts.iter().cloned());

    // The emulator decides whether its workload needs host GPU access
    // (e.g. HotSwap runs the retargeted code on the real GPU). The
    // provider-specific group handling lives in the container engine.
    let host_gpus = injection.host_gpus;

    // Each container hosts itself: its entrypoint is `mirage host`, run
    // from the mirage binary bind-mounted in read-only, against the
    // session's runtime directory bind-mounted read-write. Mounting only
    // this one session's directory (not the whole runtime root) keeps
    // containers isolated from other sessions.
    let mirage_bin = std::env::current_exe().map_err(|e| MirageError::other(format!("{e}")))?;
    def.mounts.push(FileMount {
        host_path: mirage_bin.to_string_lossy().into_owned(),
        container_path: CONTAINER_MIRAGE_BIN.to_string(),
        read_only: true,
    });
    let host_session_dir = mirage_core::paths::session_dir(session);
    let container_session_dir = format!("{CONTAINER_RUNTIME_DIR}/session/{session}");
    def.mounts.push(FileMount {
        host_path: host_session_dir.to_string_lossy().into_owned(),
        container_path: container_session_dir,
        read_only: false,
    });

    // Mount the orchestrator's config directory read-only so the
    // in-container host can resolve profiles and topologies that are
    // referenced by name (not stored inline) in the session definition.
    // Without this, `resolve_profile`/`resolve_node_count` inside the
    // container would look in a non-existent XDG path and the per-node
    // host would silently fail to run requested execs.
    let host_config_dir = mirage_core::paths::mirage_config_dir();
    if host_config_dir.exists() {
        def.mounts.push(FileMount {
            host_path: host_config_dir.to_string_lossy().into_owned(),
            container_path: CONTAINER_CONFIG_DIR.to_string(),
            read_only: true,
        });
    }

    // Expose the emulator's shared libraries inside each node container
    // under `CONTAINER_LIB_DIR` (`/mnt/mirage/lib`). The per-node host
    // *inside* the container re-resolves its injection against this
    // directory, and the workload's `LD_PRELOAD` interposer is loaded
    // from here, so we bind-mount, in priority order:
    //   1. the emulator's declared `libraries`,
    //   2. the emulator's `LD_PRELOAD` interposer itself,
    // each preserving its file name, and put `CONTAINER_LIB_DIR` on
    // `LD_LIBRARY_PATH` (below) so the loader prefers them.
    let mut libraries: Vec<String> = Vec::new();
    libraries.extend(injection.libraries.iter().cloned());
    if let Some(preload) = &injection.ld_preload {
        libraries.push(preload.clone());
    }

    // `LD_PRELOAD` is resolved against the orchestrator's *host*
    // filesystem; once mounted under `CONTAINER_LIB_DIR` we point the
    // container's `LD_PRELOAD` at that in-container path (the host path
    // does not exist inside the container, so passing it verbatim makes
    // `ld.so` fail with "cannot be preloaded").
    let container_ld_preload = injection.ld_preload.as_deref().and_then(|p| {
        std::path::Path::new(p)
            .file_name()
            .map(|n| format!("{CONTAINER_LIB_DIR}/{}", n.to_string_lossy()))
    });

    let mut mounted_libs: HashSet<String> = HashSet::new();
    for lib in &libraries {
        let Some(name) = std::path::Path::new(lib)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        else {
            continue;
        };
        // Mount only the first library seen for each file name; later
        // duplicates (e.g. the `LD_PRELOAD` interposer also listed in
        // `libraries`) would collide on the same container path.
        if !mounted_libs.insert(name.clone()) {
            continue;
        }
        def.mounts.push(FileMount {
            host_path: lib.clone(),
            container_path: format!("{CONTAINER_LIB_DIR}/{name}"),
            read_only: true,
        });
    }

    // Environment every node container inherits: the emulator's injected
    // env (already remapped to container paths), its `LD_PRELOAD`, and
    // the `MIRAGE_*` directory overrides so the in-container host
    // resolves every mirage directory at its mounted (or deterministic
    // in-container) location instead of a non-existent XDG fallback. The
    // in-container host forwards this environment to the workload child,
    // so no separate per-exec injection happens inside the container.
    let mut injected_env: Vec<(String, String)> = injection
        .env
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let Some(preload) = &container_ld_preload {
        injected_env.push(("LD_PRELOAD".to_string(), preload.clone()));
    }
    // Prepend the mirage library mount dir to `LD_LIBRARY_PATH` so the
    // loader finds the emulator's `libraries`/interposer mounted there
    // ahead of the image's own copies.
    {
        let value = match injection.env.get("LD_LIBRARY_PATH") {
            Some(existing) if !existing.is_empty() => {
                format!("{CONTAINER_LIB_DIR}:{existing}")
            }
            _ => CONTAINER_LIB_DIR.to_string(),
        };
        injected_env.push(("LD_LIBRARY_PATH".to_string(), value));
    }
    injected_env.push((
        "MIRAGE_RUNTIME".to_string(),
        CONTAINER_RUNTIME_DIR.to_string(),
    ));
    injected_env.push((
        "MIRAGE_CONFIG".to_string(),
        CONTAINER_CONFIG_DIR.to_string(),
    ));
    injected_env.push(("MIRAGE_STATE".to_string(), CONTAINER_STATE_DIR.to_string()));

    // Propagate the orchestrator's log level into each node container so
    // the per-node `mirage host` (and thus its exec/node events) logs at
    // the verbosity the user asked for via `-v`/`-vv`. Without this the
    // in-container host defaults to `warn` and its events never appear in
    // `podman logs`.
    if let Ok(log) = std::env::var("MIRAGE_LOG") {
        injected_env.push(("MIRAGE_LOG".to_string(), log));
    }

    publish_health(
        layout,
        false,
        "preparing",
        Some(format!(
            "resolving container provider for image {}",
            def.image
        )),
    )?;

    let engine =
        mirage_container::Engine::resolve(&def).map_err(|e| MirageError::other(format!("{e}")))?;

    // Apply any profile hacks by building a derivative image from the
    // base image and running that in place of it. The derived tag is a
    // pure function of the base image + hack set, so a previously built
    // image is reused rather than rebuilt. On failure the bring-up aborts
    // before any container is started.
    if let (Some(tag), Some(dockerfile)) = (
        mirage_core::profile::hacks_image_tag(&def.image, &def.hacks),
        mirage_core::profile::hacks_dockerfile(&def.image, &def.hacks),
    ) {
        if engine.image_present(&tag) {
            let msg = format!("derived image {tag} already built; skipping build");
            tracing::info!(image = %tag, "{msg}");
            publish_health(layout, false, "building", Some(msg))?;
        } else {
            let msg = format!(
                "building derived image {tag} from {} (this can take a while)…",
                def.image
            );
            tracing::info!(image = %tag, base = %def.image, hacks = ?def.hacks, "{msg}");
            publish_health(layout, false, "building", Some(msg))?;
            engine.build_image(&tag, &dockerfile).map_err(|e| {
                MirageError::other(format!("building derived image {tag} failed: {e}"))
            })?;
            tracing::info!(image = %tag, "derived image built");
        }
        def.image = tag;
    }

    let node_count = resolve_node_count(&profile)?;
    let head_port = pick_head_port();
    let head_addr = container_name(session, 0);
    let session_str = session.to_string();

    // Track the most recent bring-up phase so that, if a step fails, the
    // error we surface names exactly what mirage was doing (e.g. "while
    // pulling image …") rather than a bare provider error.
    let last_phase: std::cell::RefCell<Option<mirage_container::BringUpPhase>> =
        std::cell::RefCell::new(None);

    let (state, cids) = engine
        .bring_up(
            session,
            &def,
            host_gpus,
            node_count,
            head_port,
            |rank| {
                let mut env = node_mirage_env(rank, node_count, &head_addr, head_port);
                env.extend(injected_env.iter().cloned());
                env
            },
            |rank| {
                vec![
                    CONTAINER_MIRAGE_BIN.to_string(),
                    "host".to_string(),
                    "--session".to_string(),
                    session_str.clone(),
                    "--rank".to_string(),
                    rank.to_string(),
                ]
            },
            |phase| {
                // Mirror each bring-up phase into session health so
                // clients see live, detailed progress (pulling the image,
                // creating the network, starting each node), and log it
                // at INFO so it's visible in the host log too.
                let (state, message) = phase.health();
                tracing::info!(state, "{message}");
                let _ = publish_health(layout, false, state, Some(message));
                *last_phase.borrow_mut() = Some(phase);
            },
        )
        .map_err(|e| {
            // Name the phase that was in flight so the failure is
            // actionable (e.g. a registry auth error while pulling, or a
            // device-permission error while starting a node).
            let context = match last_phase.into_inner() {
                Some(p) => format!(
                    "{} failed: {e}",
                    p.message().trim_end_matches('…').trim_end()
                ),
                None => format!("container bring-up failed: {e}"),
            };
            MirageError::other(context)
        })?;

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

/// Build the always-present mirage environment for one workload process.
///
/// A process is identified by three ranks:
///
/// * `node_rank` — which node it runs on (`MIRAGE_RANK`). The head node
///   is rank 0.
/// * `global_rank` — its index across the whole job (`RANK`, in
///   `0..world_size`). With `nproc_per_node > 1` several processes share
///   a node (and thus a `MIRAGE_RANK`) but each gets a unique `RANK`.
/// * `local_rank` — its index within the node (`LOCAL_RANK`, in
///   `0..nproc_per_node`), which the workload typically uses to pin a
///   GPU.
///
/// `MASTER_ADDR`/`MASTER_PORT` point at the head node so
/// `torch.distributed` (and `torchrun`) can rendezvous without the
/// workload translating mirage's own variables or needing a launcher;
/// the head's own processes reach it via loopback. These are injected
/// whether or not the session is containerised. With the default of one
/// process per node `global_rank == node_rank` and `local_rank == 0`, so
/// the historical one-process-per-node environment is unchanged.
fn proc_mirage_env(
    global_rank: u32,
    node_rank: u32,
    local_rank: u32,
    world_size: u32,
    head_addr: &str,
    head_port: u16,
) -> Vec<(String, String)> {
    // The head node hosts the rendezvous: processes on it reach it via
    // loopback, processes on other nodes via the head's address.
    let head = if node_rank == 0 { "localhost" } else { head_addr };
    vec![
        (ENV_RANK.to_string(), node_rank.to_string()),
        (ENV_TORCH_RANK.to_string(), global_rank.to_string()),
        (ENV_HEAD_PORT.to_string(), head_port.to_string()),
        (ENV_HEAD_ADDR.to_string(), head.to_string()),
        (ENV_MASTER_ADDR.to_string(), head.to_string()),
        (ENV_MASTER_PORT.to_string(), head_port.to_string()),
        (ENV_WORLD_SIZE.to_string(), world_size.to_string()),
        (ENV_LOCAL_RANK.to_string(), local_rank.to_string()),
        // A distinct host id per emulated node so RCCL/NCCL does not see
        // ranks on different nodes (whose identical per-node config gives
        // the same GPU `location_id`) as duplicate GPUs on one host.
        (
            ENV_NCCL_HOSTID.to_string(),
            format!("mirage-node-{node_rank}"),
        ),
    ]
}

/// Node-level baseline environment for a node of `rank`, used as the
/// container's env at launch (before any exec, so the process grid isn't
/// known yet). It is the one-process-per-node case of [`proc_mirage_env`]
/// (`global_rank == node_rank`, `local_rank == 0`); the authoritative
/// per-process values are layered on by [`spawn_node`] at exec time.
fn node_mirage_env(
    rank: u32,
    world_size: u32,
    head_addr: &str,
    head_port: u16,
) -> Vec<(String, String)> {
    proc_mirage_env(rank, rank, 0, world_size, head_addr, head_port)
}

/// Resolve the emulator-level injection for a session by reading its
/// on-disk definition, resolving its profile, and dispatching to the
/// configured emulator backend (looked up in the registry by its
/// [`mirage_core::emulator::EmulatorKind`] name) to compute the env
/// vars / `LD_PRELOAD` it needs.
///
/// Returns an empty [`InjectionDef`] for emulators that need no
/// injection (`noop`). Errors when a configured emulator cannot produce
/// its required assets, its runtime library is missing, or no backend
/// with that name was compiled in, so a misconfigured session fails
/// loudly instead of silently running unemulated.
fn resolve_injection(session: &SessionId) -> Result<InjectionDef> {
    let profile = resolve_profile(session)?;
    let kind = &profile.emulator.emulator;
    match mirage_core::emulator::get_emulator_backend(kind) {
        Some(backend) => backend.injection_def(session),
        None => Err(MirageError::Other(format!("unknown emulator `{kind}`"))),
    }
}

/// Start the emulator daemon for `session`, if its backend hosts one.
///
/// Dispatches to the configured emulator backend's
/// [`mirage_core::emulator::EmulatorBackend::start_daemon`]. A failure
/// to start the daemon is *not* fatal: the rocjitsu interposer falls
/// back to in-process emulation when no daemon socket is present, so we
/// log and continue rather than aborting the host. Returns `None` for
/// emulators that host no daemon (`noop`, `hotswap`) or when the backend
/// is unknown.
fn start_emulator_daemon(
    session: &SessionId,
) -> Option<Box<dyn mirage_core::emulator::EmulatorDaemon>> {
    let profile = resolve_profile(session).ok()?;
    let kind = &profile.emulator.emulator;
    let backend = mirage_core::emulator::get_emulator_backend(kind)?;
    match backend.start_daemon(session) {
        Ok(daemon) => {
            if daemon.is_some() {
                tracing::info!(session = %session, emulator = %kind, "emulator daemon hosted");
            }
            daemon
        }
        Err(e) => {
            tracing::warn!(
                session = %session,
                "emulator daemon failed to start ({e}); \
                 falling back to in-process emulation"
            );
            None
        }
    }
}

/// Run a single exec.
///
/// `host_rank` scopes which node(s) this host is responsible for:
///
/// * `None` — the orchestrator host of a *non*-containerised session:
///   it spawns every node directly on the real host and aggregates the
///   exec status. (The orchestrator of a containerised session never
///   calls this; its per-node hosts do.)
/// * `Some(rank)` — a per-node host running *inside* its container:
///   it spawns only its own `rank` as a direct child of itself,
///   inheriting the container's environment (which already carries the
///   emulator injection). Rank 0 additionally aggregates every node's
///   status; other ranks only publish their own node files.
async fn run_exec(layout: ExecLayout, host_rank: Option<u32>) -> Result<()> {
    let def: ExecDef = read_json(&layout.def())?;
    // Only the aggregator (rank 0, or the non-containerised orchestrator)
    // publishes the exec-wide `status.json`; other per-node hosts only
    // write their own node files so concurrent hosts never race on it.
    let is_aggregator = matches!(host_rank, None | Some(0));
    let mut status = ExecStatus {
        started: false,
        ended: false,
        exit_code: None,
        started_at: Some(Utc::now()),
        ended_at: None,
        nodes: Default::default(),
    };
    if is_aggregator {
        write_json(&layout.status(), &status)?;
    }

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

    // This host runs only the node(s) it owns: a per-node in-container
    // host runs just its own rank; the non-containerised orchestrator
    // runs every node.
    let in_container = host_rank.is_some();
    let owned_nodes: Vec<u32> = match host_rank {
        Some(r) => vec![r],
        None => (0..node_count).collect(),
    };

    // Expand each owned node into `nproc_per_node` workload processes.
    // Each process gets a slot keyed by its global rank
    // (`node * nproc + local`), which is also the directory it streams
    // through (`node/<global>`) and the key under which its status is
    // published. With the default `nproc == 1` the global rank equals the
    // node rank, so the on-disk layout is identical to the historical
    // one-process-per-node case. `world_size` spans the whole job.
    let nproc = def.nproc_per_node.max(1);
    let world_size = node_count * nproc;
    let procs: Vec<ProcSpec> = owned_nodes
        .iter()
        .flat_map(|&n| {
            (0..nproc).map(move |local| ProcSpec {
                global: n * nproc + local,
                node: n,
                local,
            })
        })
        .collect();

    tracing::info!(
        session = %def.session,
        command = %def.exec.command,
        nodes = owned_nodes.len(),
        procs = procs.len(),
        world_size,
        "running exec"
    );

    // Resolve the emulator-level injection (env vars, LD_PRELOAD, ...)
    // for this exec's session. For a rocjitsu session this is what wires
    // `LD_PRELOAD=librocjitsu_kmd.so` plus `ROCJITSU_RUNTIME_DIR` into
    // every spawned child so the workload actually runs under emulation.
    //
    // Crucially, this happens *after* the initial `status.json` and node
    // ranks are known: when injection resolution fails (e.g. the
    // emulator's runtime library is missing inside a node container) we
    // must record a terminal failure for every process we own. Otherwise
    // the exec would have no `status.json` reporting `ended`, and an
    // attached client (`mirage run`) would block forever waiting for an
    // exit that never comes.
    let injection = match resolve_injection(&def.session) {
        Ok(injection) => injection,
        Err(e) => {
            tracing::error!(session = %def.session, "exec setup failed: {e}");
            for p in &procs {
                let nlayout = layout.node(p.global);
                let _ = std::fs::create_dir_all(&nlayout.root);
                let _ = std::fs::write(nlayout.stdout(), format!("mirage: {e}\n").as_bytes());
                let _ = write_bytes(&nlayout.exit_code(), b"127");
                status.nodes.insert(
                    p.global,
                    NodeStatus {
                        pid: None,
                        exit_code: Some(127),
                    },
                );
            }
            status.started = true;
            status.ended = true;
            status.ended_at = Some(Utc::now());
            status.exit_code = Some(127);
            if is_aggregator {
                write_json(&layout.status(), &status)?;
            }
            return Ok(());
        }
    };

    let mut handles = Vec::new();
    for p in procs {
        let nlayout = layout.node(p.global);
        std::fs::create_dir_all(&nlayout.root).map_err(|e| MirageError::Io {
            path: nlayout.root.clone(),
            source: e,
        })?;
        let mirage_env =
            proc_mirage_env(p.global, p.node, p.local, world_size, &head_addr, head_port);
        match spawn_node(
            &def,
            p.global,
            nlayout.clone(),
            &injection,
            &mirage_env,
            in_container,
        ) {
            Ok(node) => {
                status.started = true;
                tracing::debug!(
                    global = p.global,
                    node = p.node,
                    local = p.local,
                    pid = node.pid,
                    "spawned process"
                );
                status.nodes.insert(
                    p.global,
                    NodeStatus {
                        pid: Some(node.pid),
                        exit_code: None,
                    },
                );
                handles.push((p.global, node, nlayout));
            }
            Err(e) => {
                // Spawning the process failed (e.g. the command doesn't
                // exist). Rather than leaving the exec stuck in a
                // perpetual "started but never ended" state, surface the
                // reason on the process's stdout (which attach clients
                // tail) and record the conventional 127 "command not
                // found" exit code for this slot.
                let msg = format!("mirage: {e}\n");
                let _ = std::fs::write(nlayout.stdout(), msg.as_bytes());
                let _ = write_bytes(&nlayout.exit_code(), b"127");
                status.started = true;
                status.nodes.insert(
                    p.global,
                    NodeStatus {
                        pid: None,
                        exit_code: Some(127),
                    },
                );
            }
        }
    }
    if is_aggregator {
        write_json(&layout.status(), &status)?;
    }

    let mut overall_code: i32 = status
        .nodes
        .values()
        .filter_map(|n| n.exit_code)
        .fold(0, |acc, c| if c.abs() > acc.abs() { c } else { acc });
    for (node, child, nlayout) in handles {
        let code = wait_node(child).await;
        write_bytes(&nlayout.exit_code(), code.to_string().as_bytes())?;
        tracing::debug!(node, code, "node exited");
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

    // Rank 0 of a multi-node containerised session aggregates the other
    // ranks' processes: each per-node host writes an `exit_code` file for
    // every process it owns (keyed by global rank), so we wait for those
    // to appear and fold them into the exec-wide status. (For a
    // single-node session or the non-containerised orchestrator this loop
    // has nothing to wait on, since this host already owns every global
    // rank in `0..world_size`.)
    if is_aggregator {
        for global in 0..world_size {
            if status.nodes.get(&global).and_then(|n| n.exit_code).is_some() {
                continue;
            }
            let nlayout = layout.node(global);
            let code = await_node_exit_code(&nlayout).await;
            status.nodes.insert(
                global,
                NodeStatus {
                    pid: status.nodes.get(&global).and_then(|s| s.pid),
                    exit_code: Some(code),
                },
            );
            if code.abs() > overall_code.abs() {
                overall_code = code;
            }
        }
    }

    status.ended = true;
    status.ended_at = Some(Utc::now());
    status.exit_code = Some(overall_code);
    if is_aggregator {
        write_json(&layout.status(), &status)?;
    }
    tracing::info!(session = %def.session, exit_code = overall_code, "exec finished");

    // If the exec is keep=false, remove the directory.
    if !def.keep {
        // best effort; if attached clients are still tailing they'll
        // gracefully shutdown when the dir disappears.
        let _ = std::fs::remove_dir_all(&layout.root);
    }
    Ok(())
}

/// One workload process in the exec's process grid.
///
/// With `--nproc-per-node N` each node hosts `N` processes; this records
/// the three ranks that identify one of them: its `global` rank across
/// the whole job (also the `node/<global>` directory it streams
/// through), the `node` it runs on, and its `local` index within that
/// node.
struct ProcSpec {
    global: u32,
    node: u32,
    local: u32,
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

/// Wait for a node's `exit_code` file (written by that node's own host)
/// to appear and return the code it records. Used by rank 0 to collect
/// the other ranks' results in a multi-node containerised session.
async fn await_node_exit_code(nlayout: &mirage_core::paths::NodeLayout) -> i32 {
    loop {
        if let Ok(s) = std::fs::read_to_string(nlayout.exit_code()) {
            if let Ok(code) = s.trim().parse::<i32>() {
                return code;
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn spawn_node(
    def: &ExecDef,
    node: u32,
    nlayout: mirage_core::paths::NodeLayout,
    injection: &InjectionDef,
    mirage_env: &[(String, String)],
    in_container: bool,
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
    // Create the stdout file (the bridge appends merged PTY output here;
    // the child's stderr is merged into stdout by the PTY).
    let stdout_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(nlayout.stdout())
        .map_err(|e| MirageError::Io {
            path: nlayout.stdout(),
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
        path: nlayout.stdout(),
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

    let mut cmd = if in_container {
        // Per-node host running *inside* its container: spawn the workload
        // as a direct child of this host. We inherit the container's full
        // environment (which the orchestrator populated with the emulator
        // injection at `podman run -e`) and layer the per-exec workload
        // env on top, so `LD_PRELOAD`/library paths resolve at their
        // mounted container locations.
        let mut cmd = tokio::process::Command::new(&args.command);
        cmd.args(&args.args)
            .stdin(Stdio::from(slave_in))
            .stdout(Stdio::from(slave_out))
            .stderr(Stdio::from(slave_err))
            // Inherit the container environment as-is, then apply the
            // workload env (rank/head/user/injection) over it.
            .env(
                "TERM",
                std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()),
            )
            .envs(&workload_env);
        if let Some(wd) = &args.workdir {
            cmd.current_dir(wd);
        }
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
        let flags = nix::fcntl::fcntl(fd, nix::fcntl::FcntlArg::F_GETFL)
            .map(nix::fcntl::OFlag::from_bits_truncate)
            .unwrap_or(nix::fcntl::OFlag::empty());
        let _ = nix::fcntl::fcntl(
            fd,
            nix::fcntl::FcntlArg::F_SETFL(flags | nix::fcntl::OFlag::O_NONBLOCK),
        );
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

/// Read a fd into `buf`, mapping a nix error into the last OS error.
fn raw_read(fd: std::os::fd::RawFd, buf: &mut [u8]) -> std::io::Result<usize> {
    nix::unistd::read(fd, buf).map_err(std::io::Error::from)
}

/// Write `buf` to a fd, mapping a nix error into the last OS error.
fn raw_write<Fd: std::os::fd::AsFd>(fd: Fd, buf: &[u8]) -> std::io::Result<usize> {
    nix::unistd::write(fd, buf).map_err(std::io::Error::from)
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
    while !data.is_empty() {
        let mut guard = master.writable().await?;
        match guard.try_io(|inner| raw_write(inner.get_ref(), data)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_node_env_uses_localhost_and_aliases_master() {
        let env: std::collections::BTreeMap<_, _> =
            node_mirage_env(0, 4, "mirage-sess-node-0", 29500)
                .into_iter()
                .collect();
        assert_eq!(env[ENV_RANK], "0");
        // torch.distributed reads the global rank under its standard name.
        assert_eq!(env[ENV_TORCH_RANK], "0");
        assert_eq!(env[ENV_HEAD_ADDR], "localhost");
        assert_eq!(env[ENV_HEAD_PORT], "29500");
        // torch.distributed rendezvous vars alias the head addr/port.
        assert_eq!(env[ENV_MASTER_ADDR], "localhost");
        assert_eq!(env[ENV_MASTER_PORT], "29500");
        // One workload process per node: world size is the node count
        // and the local rank is always 0.
        assert_eq!(env[ENV_WORLD_SIZE], "4");
        assert_eq!(env[ENV_LOCAL_RANK], "0");
    }

    #[test]
    fn worker_node_env_points_master_at_head() {
        let env: std::collections::BTreeMap<_, _> =
            node_mirage_env(2, 4, "mirage-sess-node-0", 29500)
                .into_iter()
                .collect();
        assert_eq!(env[ENV_RANK], "2");
        // torch.distributed reads the global rank under its standard name.
        assert_eq!(env[ENV_TORCH_RANK], "2");
        // Workers see the head's address, not localhost.
        assert_eq!(env[ENV_HEAD_ADDR], "mirage-sess-node-0");
        assert_eq!(env[ENV_MASTER_ADDR], "mirage-sess-node-0");
        assert_eq!(env[ENV_MASTER_PORT], "29500");
        assert_eq!(env[ENV_WORLD_SIZE], "4");
        assert_eq!(env[ENV_LOCAL_RANK], "0");
    }

    #[test]
    fn proc_env_distinguishes_local_and_global_rank() {
        // 2 nodes x 2 procs/node => world size 4. Inspect the second
        // process on node 1: global rank 3, local rank 1.
        let env: std::collections::BTreeMap<_, _> =
            proc_mirage_env(3, 1, 1, 4, "mirage-sess-node-0", 29500)
                .into_iter()
                .collect();
        // MIRAGE_RANK identifies the node; RANK is the global process rank.
        assert_eq!(env[ENV_RANK], "1");
        assert_eq!(env[ENV_TORCH_RANK], "3");
        assert_eq!(env[ENV_LOCAL_RANK], "1");
        assert_eq!(env[ENV_WORLD_SIZE], "4");
        // A non-head node reaches the rendezvous via the head's address.
        assert_eq!(env[ENV_MASTER_ADDR], "mirage-sess-node-0");
        assert_eq!(env[ENV_MASTER_PORT], "29500");
        // NCCL_HOSTID is keyed by node so cross-node ranks (which share an
        // identical emulated GPU location_id) are not flagged as duplicate.
        assert_eq!(env[ENV_NCCL_HOSTID], "mirage-node-1");
    }

    #[test]
    fn proc_env_head_node_extra_proc_uses_localhost() {
        // The second process on the head node (node 0, local 1, global 1)
        // shares the node with the rendezvous, so it reaches it on
        // loopback even though its global rank is non-zero.
        let env: std::collections::BTreeMap<_, _> =
            proc_mirage_env(1, 0, 1, 4, "mirage-sess-node-0", 29500)
                .into_iter()
                .collect();
        assert_eq!(env[ENV_RANK], "0");
        assert_eq!(env[ENV_TORCH_RANK], "1");
        assert_eq!(env[ENV_LOCAL_RANK], "1");
        assert_eq!(env[ENV_MASTER_ADDR], "localhost");
        // Both processes share node 0, so they share a host id and
        // disambiguate by local GPU index instead.
        assert_eq!(env[ENV_NCCL_HOSTID], "mirage-node-0");
    }
}
