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
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use mirage_core::common::MaybeRef;
use mirage_core::container::{ContainerState, container_name};
use mirage_core::emulator::EmulatorDaemon;
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, InjectionDef};
use mirage_core::profile::{FileMount, ProfileDef};
use mirage_core::session::{SessionContext, SessionDef, SessionHealth, SessionId, state};
use tokio::sync::watch;

use crate::exec::Exec;
use mirage_container::NodeClient;
use mirage_core::proto::{ContainerTargets, SessionDescription};

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
    /// `true` while no `start_exec` call is between reserving an id and
    /// registering the exec it produced.
    ///
    /// Spawning happens outside the lock (see [`Session::start_exec`]),
    /// which opens a window in which processes exist but are in nobody's
    /// map. Teardown waits for this before collecting what to kill, so
    /// that window can never hide a workload from it.
    quiescent: watch::Sender<bool>,
}

#[derive(Debug, Default)]
struct Inner {
    /// Execs by id, live and finished alike.
    execs: BTreeMap<ExecId, Arc<Exec>>,
    /// Counter for the next exec id.
    next_exec: u32,
    /// Containers backing this session, once brought up.
    containers: Option<ContainerState>,
    /// The provider clients running those containers. Each one *is* its
    /// container's lifetime; dropping them stops the containers.
    container_clients: Vec<NodeClient>,
    /// The emulator's out-of-process daemon, if it hosts one.
    emulator_daemon: Option<Box<dyn EmulatorDaemon>>,
    /// The injection to apply to every workload process.
    injection: InjectionDef,
    /// Set once teardown has begun, so no new exec can be started into a
    /// session that is being destroyed — otherwise a process could be
    /// spawned after teardown had already collected the list to kill, and
    /// would survive it.
    tearing_down: bool,
    /// How many `start_exec` calls have reserved an id and not yet
    /// registered their exec. See [`Session::quiescent`].
    starting: usize,
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
            quiescent: watch::channel(true).0,
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

    /// Publish a lifecycle phase.
    ///
    /// `stopped` is marked terminal, because it is: a torn-down session
    /// will never become healthy, and anyone waiting on it has their
    /// answer. Publishing it as non-terminal makes
    /// [`SessionHealth::is_settled`] false for a session that is provably
    /// finished, so a concurrent `mirage session wait` blocks for its
    /// entire timeout and then reports a timeout for a session that was
    /// stopped seconds earlier.
    pub fn set_phase(&self, healthy: bool, phase: &str, message: Option<String>) {
        let mut health = SessionHealth::phase(healthy, phase, message);
        health.terminal = phase == state::STOPPED;
        self.set_health(health);
    }

    /// The container record, once bring-up has produced one.
    #[must_use]
    pub fn containers(&self) -> Option<ContainerState> {
        self.lock().containers.clone()
    }

    /// Record the containers produced by bring-up, and take ownership of
    /// the provider clients running them.
    ///
    /// Holding the clients here is what ties the containers' lifetime to
    /// the session's: they are killed by [`Session::teardown`], and by
    /// their own `Drop` if the process dies some other way.
    pub fn set_containers(&self, state: ContainerState, clients: Vec<NodeClient>) {
        let mut inner = self.lock();
        inner.containers = Some(state);
        inner.container_clients = clients;
    }

    /// Whether every container backing this session is still running.
    ///
    /// `true` for a non-containerised session, which has none. A client
    /// that exited on its own means the container died underneath the
    /// session — an OOM kill, an external `podman stop`, a crashed engine
    /// — and the caller would otherwise only find out through an exec
    /// failing with "no such container".
    #[must_use]
    pub fn containers_alive(&self) -> bool {
        let mut inner = self.lock();
        inner.container_clients.iter_mut().all(NodeClient::alive)
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
    /// Returns an error if the session is not ready, if it is being torn
    /// down, or if the process grid cannot be described (an unresolvable
    /// topology).
    pub async fn start_exec(
        &self,
        def: &ExecDef,
    ) -> Result<(Arc<Exec>, tokio::sync::mpsc::Receiver<crate::process::OutputChunk>)> {
        // Refuse until bring-up has finished. This is a correctness
        // guard, not politeness: the emulator injection and the container
        // record are both written by bring-up, and `build_specs` reads
        // whatever is there *now*. Run before they are set and it happily
        // produces a spec with an empty `InjectionDef` (no `LD_PRELOAD`,
        // no runtime directory) and no container — so the workload runs
        // directly on the real host, unemulated, touching whatever GPU is
        // actually installed, and exits 0. Mirage reports success for a
        // job that never went near the emulator.
        //
        let health = self.health();
        if !health.healthy {
            return Err(MirageError::other(format!(
                "session {} is not ready ({})",
                self.def.id,
                health.state.as_deref().unwrap_or("unknown"),
            )));
        }

        // Reserve an id and register the intent to start, under the lock.
        //
        // The lock is released before anything is spawned. Holding it
        // across `Exec::start` used to be how the teardown race was
        // closed, but `Exec::start` forks and execs one process per rank
        // — up to `MAX_WORLD_SIZE` of them — and doing that under a
        // `std::sync::Mutex` on a runtime thread blocks the executor and
        // every other operation on this session for the duration. Worse,
        // the same lock is taken by `kill_now`, the synchronous backstop
        // that must work from a panic handler.
        //
        // `starting` replaces it: teardown waits for the count to reach
        // zero before collecting what to kill, so an exec spawned in this
        // window is still guaranteed to be visible to it.
        let id = {
            let mut inner = self.lock();
            if inner.tearing_down {
                return Err(MirageError::SessionNotFound(format!(
                    "{} (session is shutting down)",
                    self.def.id
                )));
            }

            let id = ExecId::from_counter(inner.next_exec);
            inner.next_exec += 1;
            inner.starting += 1;
            self.quiescent.send_replace(false);
            id
        };

        // Everything from here to the re-lock runs without the lock held.
        let started = self.spawn_exec(def, &id);

        let tearing_down = {
            let mut inner = self.lock();
            if let Ok((exec, _)) = &started {
                inner.execs.insert(id.clone(), exec.clone());
            }
            inner.starting -= 1;
            if inner.starting == 0 {
                self.quiescent.send_replace(true);
            }
            inner.tearing_down
        };

        let (exec, output) = started?;
        if tearing_down {
            // Teardown began while we were spawning. It waits for
            // quiescence, so it will find this exec in the map and kill
            // it — but the caller must not be handed an exec belonging to
            // a session that is going away.
            exec.terminate().await;
            self.lock().execs.remove(&id);
            return Err(MirageError::SessionNotFound(format!(
                "{} (session is shutting down)",
                self.def.id
            )));
        }
        Ok((exec, output))
    }

    /// Build the specs for `id` and spawn its processes.
    ///
    /// Split out so the caller can keep this off the critical section:
    /// it materialises the pid-file directory, forks once per rank and
    /// wires up the output pumps.
    fn spawn_exec(
        &self,
        def: &ExecDef,
        id: &ExecId,
    ) -> Result<(Arc<Exec>, tokio::sync::mpsc::Receiver<crate::process::OutputChunk>)> {
        let specs = crate::spec::build_specs(&self.describe()?, def, id)?;
        Ok(Exec::start(id.clone(), def.clone(), specs))
    }

    /// Describe this session for a client that wants to start processes
    /// in it.
    ///
    /// The same description `mirage run` builds its own specs from, so an
    /// exec started from another terminal lands in exactly the same
    /// environment. For a containerised session the emulator's paths are
    /// remapped onto the in-container mounts here, once, rather than at
    /// every call site.
    ///
    /// # Errors
    ///
    /// Returns an error if the topology cannot be resolved.
    pub fn describe(&self) -> Result<SessionDescription> {
        let (injection, containers) = {
            let inner = self.lock();
            (inner.injection.clone(), inner.containers.clone())
        };
        let node_count = resolve_node_count(&self.ctx.profile)?;

        let (head_addr, head_port) = match &containers {
            Some(state) => (container_name(&self.def.id, 0), state.head_port),
            None => ("127.0.0.1".to_string(), pick_head_port()),
        };

        // The emulator's environment was computed against the host
        // filesystem; inside a container those paths do not exist.
        let (env, ld_preload) = match &containers {
            Some(_) => (
                remap_env_for_container(&injection.env, &self.ctx.runtime_dir),
                injection
                    .ld_preload
                    .as_ref()
                    .map(|p| library_in_container(p).unwrap_or_else(|| p.clone())),
            ),
            None => (injection.env.clone(), injection.ld_preload.clone()),
        };

        Ok(SessionDescription {
            session: self.def.id.clone(),
            node_count,
            workdir: self.def.workdir.clone(),
            containers: containers.map(|state| ContainerTargets {
                provider: state.provider.clone(),
                names: (0..node_count)
                    .map(|rank| {
                        state
                            .nodes
                            .iter()
                            .find(|n| n.rank == rank)
                            .map_or_else(|| container_name(&self.def.id, rank), |n| n.name.clone())
                    })
                    .collect(),
                scratch: self.ctx.runtime_dir.clone(),
            }),
            env,
            ld_preload,
            head_addr,
            head_port,
        })
    }

    /// Resolve once no `start_exec` call is mid-spawn.
    async fn await_quiescent(&self) {
        // Bounded: a spawn that wedges must not make teardown unkillable,
        // and the backstop sweep at the end of teardown still runs.
        const QUIESCE: Duration = Duration::from_secs(10);
        let mut rx = self.quiescent.subscribe();
        // `wait_for` inspects the current value before suspending, so the
        // common case — nothing in flight — returns without yielding.
        if tokio::time::timeout(QUIESCE, rx.wait_for(|quiet| *quiet))
            .await
            .is_err()
        {
            tracing::warn!(
                session = %self.def.id,
                "an exec was still starting when teardown began; \
                 proceeding without it"
            );
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
        {
            let mut inner = self.lock();
            if inner.tearing_down {
                // Another teardown is already in flight. Returning here
                // rather than racing it keeps the ordering guarantees
                // above meaningful.
                return;
            }
            inner.tearing_down = true;
        }

        self.set_phase(false, state::STOPPING, None);

        // Let any `start_exec` that is mid-spawn finish registering.
        //
        // The flag above stops new ones, but a call that had already
        // passed that check is spawning processes right now with the lock
        // released. Collecting the exec list before it registers would
        // leave those processes running with nothing left that knows
        // about them — the precise leak this whole design exists to
        // prevent.
        self.await_quiescent().await;

        let (execs, containers, mut clients, emulator_daemon) = {
            let mut inner = self.lock();
            (
                inner.execs.values().cloned().collect::<Vec<_>>(),
                inner.containers.take(),
                std::mem::take(&mut inner.container_clients),
                inner.emulator_daemon.take(),
            )
        };

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
        //
        // Killing the provider clients is what actually stops the
        // containers, and `--rm` then removes them. The explicit teardown
        // that follows is the belt to that braces: it removes the network
        // (which no client owns) and force-removes any container the
        // provider has not finished reaping, so `Ok` from here means the
        // resources are gone rather than scheduled to go.
        if !clients.is_empty() {
            let kills = tokio::task::spawn_blocking(move || {
                for client in &mut clients {
                    client.kill();
                }
            })
            .await;
            if let Err(e) = kills {
                tracing::warn!(session = %self.def.id, "stopping container clients failed: {e}");
            }
        }
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


    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Host path of the file rank `global` of `exec` records its
/// in-container pid to.
///
/// It lives under the session scratch directory, which is already
/// bind-mounted read-write into every node container at
/// [`CONTAINER_RUNTIME_DIR`] — so the container writes it and the
/// supervisor reads it off the host filesystem, with no provider round
/// trip and nothing to clean up separately (teardown removes the whole
/// directory).
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
pub(crate) fn pick_head_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map_or(0, |a| a.port())
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
    use std::path::PathBuf;
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
                emulator: "rocjitsu".to_string(),
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
