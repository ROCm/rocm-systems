//! A live session: its profile, emulator runtime, containers and execs.
//!
//! Bring-up runs in the background so `session_create` returns promptly
//! and the caller can watch progress through health, rather than blocking
//! for however long an image pull takes. Teardown is the mirror image and
//! is deliberately *not* backgrounded: `session_destroy` returns only
//! once every process is reaped, every container removed and the scratch
//! directory deleted, because a caller that has been told a session is
//! destroyed must be able to rely on that.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use chrono::Utc;
use mirage_core::common::MaybeRef;
use mirage_core::container::{
    ContainerState, ENV_HEAD_ADDR, ENV_HEAD_PORT, ENV_LOCAL_RANK, ENV_MASTER_ADDR,
    ENV_MASTER_PORT, ENV_NCCL_HOSTID, ENV_RANK, ENV_TORCH_RANK, ENV_WORLD_SIZE, container_name,
};
use mirage_core::emulator::EmulatorDaemon;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, InjectionDef};
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionContext, SessionDef, SessionHealth, SessionId, state};
use tokio::sync::watch;

use crate::exec::Exec;
use crate::process::{SpawnSpec, StdioMode};

/// Path the mirage binary is bind-mounted at inside every node container.
const CONTAINER_MIRAGE_BIN: &str = "/mnt/mirage/bin/mirage";

/// Mirage config root inside every node container, bind-mounted read-only
/// so an in-container process can resolve by-name profile/topology
/// references.
const CONTAINER_CONFIG_DIR: &str = "/mnt/mirage/config";

/// Directory inside every node container where host shared libraries are
/// bind-mounted (the emulator's interposer and its declared libraries).
/// Prepended to `LD_LIBRARY_PATH` so the loader prefers them over the
/// image's own, possibly older, copies.
const CONTAINER_LIB_DIR: &str = "/mnt/mirage/lib";

/// Per-session scratch directory inside every node container.
const CONTAINER_RUNTIME_DIR: &str = "/mnt/mirage/runtime";

/// Foreground process of a node container.
///
/// The container needs a PID 1 that simply stays alive: with the host
/// process gone, workloads are started from outside with `provider exec`,
/// so the container's own entrypoint has no work to do. An earlier design
/// ran `mirage host --rank N` here, which meant every container had a
/// second mirage process inside it polling a bind-mounted directory.
const CONTAINER_IDLE_COMMAND: &[&str] = &["sleep", "infinity"];

/// A live session.
#[derive(Debug)]
pub struct Session {
    /// The definition this session was created from.
    pub def: SessionDef,
    /// Resolved profile, emulator context and scratch directory.
    pub ctx: SessionContext,
    /// Health, published so waiters can react without polling.
    health: watch::Sender<SessionHealth>,
    /// Runtime state, guarded together because teardown must observe a
    /// consistent view of "what exists to clean up".
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    /// Execs by id, live and finished alike.
    execs: BTreeMap<ExecId, Arc<Exec>>,
    /// Counter for the next exec id.
    next_exec: u32,
    /// Containers backing this session, once brought up.
    containers: Option<ContainerState>,
    /// The emulator's out-of-process daemon, if it hosts one.
    emulator_daemon: Option<Box<dyn EmulatorDaemon>>,
    /// The injection to apply to every workload process.
    injection: InjectionDef,
    /// Set once teardown has begun, so no new exec can be started into a
    /// session that is being destroyed — otherwise a process could be
    /// spawned after teardown had already collected the list to kill, and
    /// would survive it.
    tearing_down: bool,
}

impl Session {
    /// Register a new session in its `starting` state.
    ///
    /// `runtime_dir` is created here so a backend can rely on it existing.
    ///
    /// # Errors
    ///
    /// Returns an error if the scratch directory cannot be created.
    pub fn new(def: SessionDef, profile: ProfileDef) -> Result<Arc<Self>> {
        let runtime_dir = mirage_core::paths::session_runtime_dir(&def.id);
        std::fs::create_dir_all(&runtime_dir)
            .map_err(|e| MirageError::io(runtime_dir.clone(), e))?;
        let ctx = SessionContext {
            id: def.id.clone(),
            profile,
            runtime_dir,
            daemon: def.daemon,
        };
        let (health, _) = watch::channel(SessionHealth::phase(false, state::STARTING, None));
        Ok(Arc::new(Self {
            def,
            ctx,
            health,
            inner: Mutex::new(Inner::default()),
        }))
    }

    /// The session's id.
    #[must_use]
    pub fn id(&self) -> &SessionId {
        &self.def.id
    }

    /// Current health snapshot.
    #[must_use]
    pub fn health(&self) -> SessionHealth {
        self.health.borrow().clone()
    }

    /// Subscribe to health changes.
    #[must_use]
    pub fn watch_health(&self) -> watch::Receiver<SessionHealth> {
        self.health.subscribe()
    }

    /// Publish a new health snapshot.
    pub fn set_health(&self, health: SessionHealth) {
        // `send_replace`, not `send`. `watch::Sender::send` *fails and
        // leaves the value unchanged* when there are no receivers, so a
        // session that finished bring-up before anyone subscribed would
        // stay stuck at its initial `starting` forever — and the next
        // caller to wait on it would time out against a session that had
        // been ready all along. That is exactly the case for a warm
        // daemon, where bring-up completes in well under the round trip
        // it takes a client to ask.
        //
        // `send_replace` updates the value unconditionally and notifies
        // whoever happens to be listening.
        let _previous = self.health.send_replace(health);
    }

    /// Publish a non-terminal phase.
    pub fn set_phase(&self, healthy: bool, phase: &str, message: Option<String>) {
        self.set_health(SessionHealth::phase(healthy, phase, message));
    }

    /// The container record, once bring-up has produced one.
    #[must_use]
    pub fn containers(&self) -> Option<ContainerState> {
        self.lock().containers.clone()
    }

    /// Record the container state produced by bring-up.
    pub fn set_containers(&self, state: ContainerState) {
        self.lock().containers = Some(state);
    }

    /// Record the emulator injection to apply to every workload.
    pub fn set_injection(&self, injection: InjectionDef) {
        self.lock().injection = injection;
    }

    /// Record the emulator's daemon handle so teardown can stop it.
    pub fn set_emulator_daemon(&self, daemon: Option<Box<dyn EmulatorDaemon>>) {
        self.lock().emulator_daemon = daemon;
    }

    /// Whether teardown has begun.
    #[must_use]
    pub fn is_tearing_down(&self) -> bool {
        self.lock().tearing_down
    }

    /// Ids of every exec, sorted.
    #[must_use]
    pub fn exec_ids(&self) -> Vec<ExecId> {
        self.lock().execs.keys().cloned().collect()
    }

    /// Look up one exec.
    #[must_use]
    pub fn exec(&self, id: &ExecId) -> Option<Arc<Exec>> {
        self.lock().execs.get(id).cloned()
    }

    /// Every exec, live and finished.
    #[must_use]
    pub fn execs(&self) -> Vec<Arc<Exec>> {
        self.lock().execs.values().cloned().collect()
    }

    /// Synchronously `SIGKILL` every process in every exec.
    ///
    /// See [`Exec::kill_now`]: this is the no-runtime, no-grace-period
    /// backstop, not the normal teardown path.
    pub fn kill_now(&self) {
        for exec in self.execs() {
            exec.kill_now();
        }
    }

    /// Start an exec in this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the session is being torn down, or if the
    /// process grid cannot be described (an unresolvable topology).
    pub fn start_exec(
        &self,
        def: &ExecDef,
        replay_bytes: usize,
        max_finished: usize,
    ) -> Result<Arc<Exec>> {
        let specs = self.build_specs(def)?;

        let mut inner = self.lock();
        if inner.tearing_down {
            return Err(MirageError::SessionNotFound(format!(
                "{} (session is shutting down)",
                self.def.id
            )));
        }

        // Forget the oldest finished execs beyond the cap.
        //
        // The daemon is long-lived and each finished exec retains its
        // output for replay, so a session that runs execs in a loop would
        // otherwise grow without bound — a slow leak that only shows up
        // after hours. Evicting here, as new execs arrive, keeps it
        // deterministic: no timer, and no window in which an exec is
        // dropped out from under a client that just asked for it.
        //
        // Only *finished* execs are candidates. A running exec is never
        // evicted, however old.
        Self::evict_finished(&mut inner, max_finished);

        let id = ExecId::from_counter(inner.next_exec);
        inner.next_exec += 1;
        // Start the exec while holding the lock. `Exec::start` spawns
        // processes and returns immediately without awaiting, and doing it
        // under the lock is what closes the race with teardown: either
        // teardown sees this exec in the map and kills it, or it set
        // `tearing_down` first and we refused above. A process cannot slip
        // between the two.
        let exec = Exec::start(id.clone(), def.clone(), specs, replay_bytes);
        inner.execs.insert(id, exec.clone());
        Ok(exec)
    }

    /// Drop the oldest finished execs until at most `max_finished`
    /// remain. Running execs are never dropped.
    fn evict_finished(inner: &mut Inner, max_finished: usize) {
        let finished: Vec<ExecId> = inner
            .execs
            .iter()
            .filter(|(_, exec)| exec.is_ended())
            .map(|(id, _)| id.clone())
            .collect();
        // `execs` is a BTreeMap keyed by a zero-padded counter, so
        // iteration order is creation order and the front is the oldest.
        let excess = finished.len().saturating_sub(max_finished);
        for id in finished.into_iter().take(excess) {
            inner.execs.remove(&id);
        }
    }

    /// Remove an exec, terminating it first if it is still running.
    ///
    /// # Errors
    ///
    /// Returns [`MirageError::ExecNotFound`] if there is no such exec.
    pub async fn remove_exec(&self, id: &ExecId) -> Result<()> {
        let exec = self
            .lock()
            .execs
            .remove(id)
            .ok_or_else(|| MirageError::ExecNotFound(id.to_string()))?;
        // Removed from the map first, so a concurrent lookup cannot hand
        // out an exec that is being killed; then terminated to completion,
        // so `remove` never leaves orphans behind.
        exec.terminate().await;
        Ok(())
    }

    /// Tear the session down completely.
    ///
    /// Ordering matters and is the whole point of doing it here rather
    /// than letting `Drop` handle pieces of it:
    ///
    /// 1. mark the session as tearing down, so no new exec can start;
    /// 2. terminate and reap every workload process;
    /// 3. only then stop the emulator daemon — the simulated device has
    ///    to outlive every process that might still be talking to it, or
    ///    a workload gets an I/O error on the way out instead of a clean
    ///    exit;
    /// 4. remove the containers and network;
    /// 5. delete the scratch directory.
    ///
    /// Every step is best-effort past the first: a failure in one must not
    /// strand the ones after it, because those are what release the
    /// resources that actually matter.
    pub async fn teardown(&self) {
        let (execs, containers, emulator_daemon) = {
            let mut inner = self.lock();
            if inner.tearing_down {
                // Another teardown is already in flight. Returning here
                // rather than racing it keeps the ordering guarantees
                // above meaningful.
                return;
            }
            inner.tearing_down = true;
            (
                inner.execs.values().cloned().collect::<Vec<_>>(),
                inner.containers.take(),
                inner.emulator_daemon.take(),
            )
        };

        self.set_phase(false, state::STOPPING, None);

        // 2. Every exec, concurrently: with many execs, terminating them
        // in sequence would multiply the SIGTERM grace period by their
        // count and make teardown take minutes.
        let terminations = execs.iter().map(|e| e.terminate());
        futures::future::join_all(terminations).await;
        {
            let mut inner = self.lock();
            inner.execs.clear();
        }

        // 3. The emulator daemon, now that nothing is talking to it.
        // `stop` is blocking (it joins the emulator's own threads), so it
        // runs on a blocking thread rather than stalling the runtime.
        if let Some(daemon) = emulator_daemon
            && let Err(e) = tokio::task::spawn_blocking(move || daemon.stop()).await
        {
            tracing::warn!(session = %self.def.id, "emulator daemon shutdown failed: {e}");
        }
        if let Some(backend) =
            mirage_core::emulator::get_emulator_backend(&self.ctx.profile.emulator.emulator)
        {
            let ctx = self.ctx.clone();
            if let Err(e) = tokio::task::spawn_blocking(move || backend.shutdown(&ctx)).await {
                tracing::warn!(session = %self.def.id, "emulator shutdown hook failed: {e}");
            }
        }

        // 4. Containers and the per-session network.
        if let Some(state) = containers
            && let Err(e) =
                tokio::task::spawn_blocking(move || mirage_core::container::teardown(&state)).await
        {
            tracing::warn!(session = %self.def.id, "container teardown failed: {e}");
        }

        // 5. Scratch. Anything the emulator wrote here is dead with the
        // session; leaving it would accumulate a directory per session.
        let runtime_dir = self.ctx.runtime_dir.clone();
        if runtime_dir.exists()
            && let Err(e) = tokio::fs::remove_dir_all(&runtime_dir).await
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                session = %self.def.id,
                path = %runtime_dir.display(),
                "could not remove session scratch directory: {e}"
            );
        }

        self.set_phase(false, state::STOPPED, None);
        tracing::info!(session = %self.def.id, "session destroyed");
    }

    /// Build the per-process spawn specs for an exec.
    fn build_specs(&self, def: &ExecDef) -> Result<Vec<SpawnSpec>> {
        let (injection, containers) = {
            let inner = self.lock();
            (inner.injection.clone(), inner.containers.clone())
        };

        let node_count = resolve_node_count(&self.ctx.profile)?;
        let nproc = def.nproc_per_node.max(1);

        // A terminal is allocated per process when the exec asked for one.
        // Only rank 0 is interactive in practice, but giving every rank a
        // terminal keeps `isatty` consistent across the job — a workload
        // that branches on it would otherwise behave differently on rank 0
        // than on its workers.
        let stdio = if def.tty {
            StdioMode::Tty {
                rows: def.tty_rows,
                cols: def.tty_cols,
            }
        } else {
            StdioMode::Pipes
        };
        let world_size = node_count * nproc;
        let (head_addr, head_port) = match &containers {
            Some(state) => (container_name(&self.def.id, 0), state.head_port),
            None => ("127.0.0.1".to_string(), pick_head_port()),
        };

        let mut specs = Vec::with_capacity(world_size as usize);
        for node in 0..node_count {
            for local in 0..nproc {
                let global = node * nproc + local;
                let args = if node == 0 {
                    &def.exec
                } else {
                    def.worker_exec.as_ref().unwrap_or(&def.exec)
                };

                // Layering order: the emulator's injection first, then the
                // user's per-exec environment, then mirage's own rank
                // variables. Mirage's go last so a workload cannot
                // accidentally break its own rendezvous by exporting
                // `RANK` or `WORLD_SIZE`.
                //
                // For a containerised session the emulator's env is
                // remapped onto the in-container mounts first: it was
                // computed against the host filesystem, and those paths do
                // not exist inside the container.
                let mut env: BTreeMap<String, String> = if containers.is_some() {
                    remap_env_for_container(&injection.env, &self.ctx.runtime_dir)
                } else {
                    injection.env.clone()
                };
                env.extend(args.env.clone());
                env.extend(proc_env(global, node, local, world_size, &head_addr, head_port));

                // `LD_PRELOAD` is the exception: the emulator's interposer
                // and a user-supplied preload must coexist, so they are
                // concatenated rather than one clobbering the other. In a
                // container the interposer is named by its mount.
                if let Some(preload) = &injection.ld_preload {
                    let resolved = if containers.is_some() {
                        library_in_container(preload).unwrap_or_else(|| preload.clone())
                    } else {
                        preload.clone()
                    };
                    let combined = match args.env.get("LD_PRELOAD") {
                        Some(user) if !user.is_empty() => format!("{resolved}:{user}"),
                        _ => resolved,
                    };
                    env.insert("LD_PRELOAD".to_string(), combined);
                }

                let workdir = args
                    .workdir
                    .clone()
                    .or_else(|| Some(self.def.workdir.clone()));

                specs.push(match &containers {
                    // Containerised: run the workload inside the node's
                    // container via the provider, as a direct child of the
                    // daemon. The provider process is what we supervise,
                    // and killing its group stops the exec.
                    Some(state) => {
                        let container = state
                            .nodes
                            .iter()
                            .find(|n| n.rank == node)
                            .map(|n| n.name.clone())
                            .unwrap_or_else(|| container_name(&self.def.id, node));
                        let engine = mirage_container::Engine::with_provider(&state.provider);
                        let env_pairs: Vec<(String, String)> = env.into_iter().collect();
                        let argv = engine.exec_command_line(
                            &container,
                            workdir.as_deref(),
                            &env_pairs,
                            &args.command,
                            &args.args,
                            def.tty,
                        );
                        let (command, rest) = argv
                            .split_first()
                            .map(|(c, r)| (c.clone(), r.to_vec()))
                            .unwrap_or_else(|| (state.provider.clone(), Vec::new()));
                        SpawnSpec {
                            node: global,
                            command,
                            args: rest,
                            env: BTreeMap::new(),
                            workdir: None,
                            // The provider is given `-t` when a terminal
                            // was asked for, and we run *it* on our own
                            // pty so the two ends line up.
                            stdio,
                            // The provider CLI needs its own environment
                            // to locate its socket and configuration.
                            inherit_env: true,
                        }
                    }
                    None => SpawnSpec {
                        node: global,
                        command: args.command.clone(),
                        args: args.args.clone(),
                        env,
                        workdir,
                        stdio,
                        inherit_env: false,
                    },
                });
            }
        }
        Ok(specs)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Resolve how many nodes a profile's topology describes.
///
/// # Errors
///
/// Returns an error if the topology is referenced by name and no such
/// topology exists.
pub fn resolve_node_count(profile: &ProfileDef) -> Result<u32> {
    let topology = match &profile.emulator.topology {
        MaybeRef::Owned(t) => t.clone(),
        MaybeRef::Ref(name) => mirage_core::topology::store::get(name)?,
    };
    Ok(topology.total_nodes().max(1))
}

/// Fully resolve a profile reference into an owned profile.
///
/// # Errors
///
/// Returns [`MirageError::ProfileNotFound`] if a by-name reference cannot
/// be resolved.
pub fn resolve_profile(reference: &MaybeRef<ProfileDef>) -> Result<ProfileDef> {
    match reference {
        MaybeRef::Owned(p) => Ok(p.clone()),
        MaybeRef::Ref(name) => {
            let path = mirage_core::paths::profile_path(name);
            if !path.exists() {
                return Err(MirageError::ProfileNotFound(name.clone()));
            }
            mirage_core::state::read_json(&path)
        }
    }
}

/// Reserve an ephemeral TCP port for the head node's rendezvous.
///
/// Binding to port 0 and reading back the assignment is a hint, not a
/// reservation: the socket is closed immediately, so the port is free
/// again by the time the workload binds it. That is the standard trick
/// and its standard caveat; a collision is possible but vanishingly rare
/// and self-evident when it happens.
fn pick_head_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map_or(0, |a| a.port())
}

/// The mirage/`torch.distributed` environment for one workload process.
///
/// Three ranks identify a process: `node` (which emulated node it runs
/// on), `global` (its index across the whole job) and `local` (its index
/// within the node, which a workload typically uses to pin a GPU). With
/// the default of one process per node, `global == node` and `local == 0`.
fn proc_env(
    global: u32,
    node: u32,
    local: u32,
    world_size: u32,
    head_addr: &str,
    head_port: u16,
) -> Vec<(String, String)> {
    // Processes on the head node reach the rendezvous over loopback;
    // everyone else needs the head's address.
    let head = if node == 0 { "localhost" } else { head_addr };
    vec![
        (ENV_RANK.to_string(), node.to_string()),
        (ENV_TORCH_RANK.to_string(), global.to_string()),
        (ENV_HEAD_ADDR.to_string(), head.to_string()),
        (ENV_HEAD_PORT.to_string(), head_port.to_string()),
        (ENV_MASTER_ADDR.to_string(), head.to_string()),
        (ENV_MASTER_PORT.to_string(), head_port.to_string()),
        (ENV_WORLD_SIZE.to_string(), world_size.to_string()),
        (ENV_LOCAL_RANK.to_string(), local.to_string()),
        // Every emulated node runs on the same real host and is
        // synthesised from an identical config, so their GPUs report the
        // same location id and RCCL would reject them as duplicates. A
        // distinct host id per node is the correct model: one emulated
        // GPU per node.
        (ENV_NCCL_HOSTID.to_string(), format!("mirage-node-{node}")),
    ]
}

/// Build the mounts, environment and command a node container is launched
/// with.
///
/// Split out from bring-up so it can be tested without a container
/// runtime: the mapping from "host paths the emulator needs" to
/// "in-container paths plus `LD_LIBRARY_PATH`" is the part that goes
/// wrong, and it is pure.
#[derive(Debug, Clone)]
pub struct ContainerPlan {
    /// Mounts to add to the profile's own.
    pub mounts: Vec<FileMount>,
    /// Environment every node container is launched with.
    pub env: Vec<(String, String)>,
    /// The container's foreground process.
    pub command: Vec<String>,
}

/// Rewrite a host path into its in-container location.
///
/// The emulator computes its injection against the *host* filesystem —
/// `ROCJITSU_RUNTIME_DIR` points at the session scratch directory, for
/// instance. Inside a node container that path does not exist, so any
/// value naming it has to be remapped to the mount it appears at.
///
/// Previously this was handled by a second mirage process running inside
/// the container, which re-resolved the whole injection against its own
/// filesystem. With workloads launched from outside via `provider exec`
/// there is no such process, so the translation happens here.
fn to_container_path(value: &str, host_dir: &std::path::Path, container_dir: &str) -> String {
    let host = host_dir.to_string_lossy();
    if host.is_empty() {
        return value.to_string();
    }
    match value.strip_prefix(host.as_ref()) {
        // An exact match, or a path beneath it. The separator check stops
        // `/run/session/s1` from matching `/run/session/s10`.
        Some("") => container_dir.to_string(),
        Some(rest) if rest.starts_with('/') => format!("{container_dir}{rest}"),
        _ => value.to_string(),
    }
}

/// Remap every value in an environment that names a path under the
/// session's scratch directory.
fn remap_env_for_container(
    env: &BTreeMap<String, String>,
    runtime_dir: &std::path::Path,
) -> BTreeMap<String, String> {
    env.iter()
        .map(|(k, v)| {
            (
                k.clone(),
                to_container_path(v, runtime_dir, CONTAINER_RUNTIME_DIR),
            )
        })
        .collect()
}

/// The in-container path a host library is bind-mounted at.
fn library_in_container(host_path: &str) -> Option<String> {
    std::path::Path::new(host_path)
        .file_name()
        .map(|name| format!("{CONTAINER_LIB_DIR}/{}", name.to_string_lossy()))
}

/// Plan the container-side layout for a session.
#[must_use]
pub fn plan_container(ctx: &SessionContext, injection: &InjectionDef) -> ContainerPlan {
    let mut mounts: Vec<FileMount> = injection.mounts.clone();

    // The mirage binary itself, so anything in-container that needs it can
    // find it at a stable path.
    if let Ok(bin) = std::env::current_exe() {
        mounts.push(FileMount {
            host_path: bin.to_string_lossy().into_owned(),
            container_path: CONTAINER_MIRAGE_BIN.to_string(),
            read_only: true,
        });
    }

    // The session scratch directory, so an in-container emulator runtime
    // sees the same config files the supervisor wrote.
    mounts.push(FileMount {
        host_path: ctx.runtime_dir.to_string_lossy().into_owned(),
        container_path: CONTAINER_RUNTIME_DIR.to_string(),
        read_only: false,
    });

    // Config, read-only, so by-name profile/topology references resolve.
    let config_dir = mirage_core::paths::mirage_config_dir();
    if config_dir.exists() {
        mounts.push(FileMount {
            host_path: config_dir.to_string_lossy().into_owned(),
            container_path: CONTAINER_CONFIG_DIR.to_string(),
            read_only: true,
        });
    }

    // The emulator's libraries and its interposer, each mounted under
    // `CONTAINER_LIB_DIR` keeping its file name. Duplicates are skipped:
    // the interposer is commonly listed in `libraries` as well, and two
    // mounts on the same container path is an error.
    let mut libraries: Vec<String> = injection.libraries.clone();
    if let Some(preload) = &injection.ld_preload {
        libraries.push(preload.clone());
    }
    let mut seen = std::collections::HashSet::new();
    for lib in &libraries {
        let Some(name) = std::path::Path::new(lib)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        else {
            continue;
        };
        if !seen.insert(name.clone()) {
            continue;
        }
        mounts.push(FileMount {
            host_path: lib.clone(),
            container_path: format!("{CONTAINER_LIB_DIR}/{name}"),
            read_only: true,
        });
    }

    // The emulator's env is computed against the host filesystem, so
    // remap anything naming the session scratch directory to its mount.
    let mut env: Vec<(String, String)> =
        remap_env_for_container(&injection.env, &ctx.runtime_dir)
            .into_iter()
            .collect();

    // `LD_PRELOAD` must name the *in-container* path: the host path does
    // not exist inside the container, and `ld.so` fails the whole process
    // with "cannot be preloaded" rather than skipping it.
    if let Some(preload) = &injection.ld_preload
        && let Some(in_container) = library_in_container(preload)
    {
        env.push(("LD_PRELOAD".to_string(), in_container));
    }

    let library_path = match injection.env.get("LD_LIBRARY_PATH") {
        Some(existing) if !existing.is_empty() => format!("{CONTAINER_LIB_DIR}:{existing}"),
        _ => CONTAINER_LIB_DIR.to_string(),
    };
    env.push(("LD_LIBRARY_PATH".to_string(), library_path));
    env.push((
        "MIRAGE_RUNTIME".to_string(),
        CONTAINER_RUNTIME_DIR.to_string(),
    ));
    env.push((
        "MIRAGE_CONFIG".to_string(),
        CONTAINER_CONFIG_DIR.to_string(),
    ));

    ContainerPlan {
        mounts,
        env,
        command: CONTAINER_IDLE_COMMAND
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
    }
}

/// The scratch directory a session's emulator assets live in.
#[must_use]
pub fn scratch_dir(id: &SessionId) -> PathBuf {
    mirage_core::paths::session_runtime_dir(id)
}

/// Build the [`SessionDef`] for a create request.
#[must_use]
pub fn make_def(
    id: SessionId,
    profile: MaybeRef<ProfileDef>,
    workdir: String,
    daemon: bool,
) -> SessionDef {
    SessionDef {
        id,
        profile,
        workdir,
        daemon,
        created_at: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use mirage_core::emulator::{EmulatorDef, ExecMode};
    use mirage_core::topology::TopologyDef;

    fn ctx(runtime_dir: PathBuf) -> SessionContext {
        SessionContext {
            id: SessionId::new("s").unwrap(),
            profile: profile(1, 1),
            runtime_dir,
            daemon: false,
        }
    }

    fn profile(num_nodes: u32, gpus_per_node: u32) -> ProfileDef {
        ProfileDef {
            name: "p".to_string(),
            description: None,
            emulator: EmulatorDef {
                emulator: "noop".to_string(),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: Default::default(),
                topology: MaybeRef::Owned(TopologyDef {
                    num_nodes,
                    gpus_per_node,
                    agent: MaybeRef::Ref("MI350X".to_string()),
                }),
            },
            containerize: None,
        }
    }

    fn env_map(pairs: Vec<(String, String)>) -> BTreeMap<String, String> {
        pairs.into_iter().collect()
    }

    #[test]
    fn head_process_reaches_the_rendezvous_over_loopback() {
        let env = env_map(proc_env(0, 0, 0, 4, "mirage-s-node-0", 29500));
        assert_eq!(env[ENV_RANK], "0");
        assert_eq!(env[ENV_TORCH_RANK], "0");
        assert_eq!(env[ENV_MASTER_ADDR], "localhost");
        assert_eq!(env[ENV_MASTER_PORT], "29500");
        assert_eq!(env[ENV_WORLD_SIZE], "4");
        assert_eq!(env[ENV_LOCAL_RANK], "0");
    }

    #[test]
    fn worker_process_reaches_the_rendezvous_by_head_address() {
        let env = env_map(proc_env(2, 2, 0, 4, "mirage-s-node-0", 29500));
        assert_eq!(env[ENV_RANK], "2");
        assert_eq!(env[ENV_MASTER_ADDR], "mirage-s-node-0");
        assert_eq!(env[ENV_NCCL_HOSTID], "mirage-node-2");
    }

    #[test]
    fn local_and_global_ranks_are_distinct_with_multiple_procs_per_node() {
        // 2 nodes x 2 procs. The second process on node 1 is global 3,
        // local 1.
        let env = env_map(proc_env(3, 1, 1, 4, "mirage-s-node-0", 29500));
        assert_eq!(env[ENV_RANK], "1", "MIRAGE_RANK identifies the node");
        assert_eq!(env[ENV_TORCH_RANK], "3", "RANK is the global process rank");
        assert_eq!(env[ENV_LOCAL_RANK], "1");
        assert_eq!(env[ENV_WORLD_SIZE], "4");
    }

    #[test]
    fn processes_sharing_a_node_share_a_host_id() {
        let a = env_map(proc_env(0, 0, 0, 4, "h", 1));
        let b = env_map(proc_env(1, 0, 1, 4, "h", 1));
        assert_eq!(a[ENV_NCCL_HOSTID], b[ENV_NCCL_HOSTID]);
        let other_node = env_map(proc_env(2, 1, 0, 4, "h", 1));
        assert_ne!(a[ENV_NCCL_HOSTID], other_node[ENV_NCCL_HOSTID]);
    }

    #[test]
    fn node_count_comes_from_the_topology_and_is_never_zero() {
        assert_eq!(resolve_node_count(&profile(3, 8)).unwrap(), 3);
        assert_eq!(
            resolve_node_count(&profile(0, 1)).unwrap(),
            1,
            "a zero-node topology must still run one node"
        );
    }

    #[test]
    fn container_plan_mounts_the_interposer_and_puts_it_on_the_library_path() {
        let dir = tempfile::tempdir().unwrap();
        let injection = InjectionDef {
            ld_preload: Some("/opt/rocm/lib/librocjitsu.so".to_string()),
            libraries: vec!["/opt/rocm/lib/libextra.so".to_string()],
            ..Default::default()
        };
        let plan = plan_container(&ctx(dir.path().to_path_buf()), &injection);

        let paths: Vec<&str> = plan
            .mounts
            .iter()
            .map(|m| m.container_path.as_str())
            .collect();
        assert!(paths.contains(&"/mnt/mirage/lib/librocjitsu.so"), "{paths:?}");
        assert!(paths.contains(&"/mnt/mirage/lib/libextra.so"), "{paths:?}");

        let env: BTreeMap<_, _> = plan.env.iter().cloned().collect();
        // The preload must be the container path; the host path does not
        // exist inside the container and ld.so would fail the process.
        assert_eq!(env["LD_PRELOAD"], "/mnt/mirage/lib/librocjitsu.so");
        assert!(env["LD_LIBRARY_PATH"].starts_with("/mnt/mirage/lib"));
    }

    #[test]
    fn container_plan_does_not_mount_the_same_path_twice() {
        // The interposer is commonly listed in `libraries` as well; two
        // mounts on one container path is an error the provider rejects.
        let dir = tempfile::tempdir().unwrap();
        let lib = "/opt/rocm/lib/librocjitsu.so".to_string();
        let injection = InjectionDef {
            ld_preload: Some(lib.clone()),
            libraries: vec![lib],
            ..Default::default()
        };
        let plan = plan_container(&ctx(dir.path().to_path_buf()), &injection);
        let lib_mounts: Vec<_> = plan
            .mounts
            .iter()
            .filter(|m| m.container_path.starts_with("/mnt/mirage/lib/"))
            .collect();
        assert_eq!(lib_mounts.len(), 1, "{lib_mounts:?}");
    }

    #[test]
    fn container_plan_preserves_an_emulator_supplied_library_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut injection = InjectionDef::default();
        injection
            .env
            .insert("LD_LIBRARY_PATH".to_string(), "/existing".to_string());
        let plan = plan_container(&ctx(dir.path().to_path_buf()), &injection);
        let env: BTreeMap<_, _> = plan.env.iter().cloned().collect();
        assert_eq!(env["LD_LIBRARY_PATH"], "/mnt/mirage/lib:/existing");
    }

    #[test]
    fn host_paths_under_the_scratch_directory_are_remapped_into_the_container() {
        let dir = tempfile::tempdir().unwrap();
        let scratch = dir.path().to_path_buf();
        let mut injection = InjectionDef::default();
        // What rocjitsu actually injects: a runtime directory the
        // interposer probes for its config and daemon socket.
        injection.env.insert(
            "ROCJITSU_RUNTIME_DIR".to_string(),
            scratch.join("rocjitsu").display().to_string(),
        );
        injection
            .env
            .insert("UNRELATED".to_string(), "/opt/elsewhere".to_string());

        let plan = plan_container(&ctx(scratch), &injection);
        let env: BTreeMap<_, _> = plan.env.iter().cloned().collect();

        assert_eq!(
            env["ROCJITSU_RUNTIME_DIR"], "/mnt/mirage/runtime/rocjitsu",
            "a host path that does not exist inside the container would \
             leave the interposer unable to find its config"
        );
        // Paths outside the scratch directory are left alone.
        assert_eq!(env["UNRELATED"], "/opt/elsewhere");
    }

    #[test]
    fn remapping_does_not_match_a_sibling_with_a_shared_prefix() {
        // `/run/session/s1` must not match `/run/session/s10`.
        let host = std::path::Path::new("/run/session/s1");
        assert_eq!(
            to_container_path("/run/session/s10/config", host, "/mnt"),
            "/run/session/s10/config"
        );
        assert_eq!(
            to_container_path("/run/session/s1/config", host, "/mnt"),
            "/mnt/config"
        );
        assert_eq!(to_container_path("/run/session/s1", host, "/mnt"), "/mnt");
    }

    #[test]
    fn container_entrypoint_just_idles() {
        // With the per-session host process gone, a node container has no
        // mirage process inside it: workloads arrive via `provider exec`.
        let dir = tempfile::tempdir().unwrap();
        let plan = plan_container(&ctx(dir.path().to_path_buf()), &InjectionDef::default());
        assert_eq!(plan.command, vec!["sleep", "infinity"]);
    }
}
