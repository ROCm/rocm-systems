//! [`Run`]: one session, owned by the process that started it.
//!
//! A `Run` is created by `mirage run` and lives exactly as long as that
//! process does. It brings a session up, hands out
//! [`SessionDescription`]s so other terminals can start processes in it,
//! and tears everything down on the way out.
//!
//! # Why there is no session manager
//!
//! There used to be one: a `SessionManager` holding a map of sessions
//! inside a long-lived daemon, because sessions outlived the commands
//! that created them and something had to own them in between. Every
//! awkward part of that design followed from the map — sessions had to be
//! looked up by id and could be missing; creation raced shutdown; a
//! shutdown flag had to be re-checked under a write lock; finished execs
//! had to be evicted because the daemon never exited and would otherwise
//! grow without bound.
//!
//! Making `mirage run` the owner removes the map, and with it every one
//! of those problems. There is one session, it is right here, and it
//! cannot outlive the process holding this value.

use std::sync::Arc;
use std::time::Duration;

use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, InjectionDef};
use mirage_core::proto::SessionDescription;
use mirage_core::session::{
    CreateSessionRequest, SessionContext, SessionDef, SessionHealth, SessionId, SessionState, state,
};

use tokio::sync::mpsc;

use crate::process::OutputChunk;
use crate::session::{Session, make_def, resolve_profile};

/// One session and everything it owns.
///
/// Dropping a `Run` without [`Run::destroy`] falls back to a synchronous
/// `SIGKILL` sweep: it cannot await, so it cannot remove containers, but
/// it does guarantee no workload process outlives the run.
#[derive(Debug)]
pub struct Run {
    session: Arc<Session>,
}

impl Run {
    /// Create a session and start bringing it up.
    ///
    /// Returns as soon as the session exists. Container pulls, image
    /// builds and emulator daemon startup happen in the background; use
    /// [`Run::wait_ready`] to wait for them and to learn why if they
    /// fail.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile cannot be resolved, which is
    /// deliberately checked here rather than during bring-up: a session
    /// naming a profile that does not exist should fail at creation, not
    /// come up and fail every exec.
    pub fn start(req: CreateSessionRequest) -> Result<Self> {
        let profile = resolve_profile(&req.profile)?;
        let id = req.id.unwrap_or_else(SessionId::generate);
        let def = make_def(id.clone(), req.profile, req.workdir, req.daemon);
        let session = Session::new(def, profile)?;

        tracing::info!(session = %id, "session created");
        tokio::spawn(Self::bring_up(session.clone()));
        Ok(Self { session })
    }

    /// The session's id.
    #[must_use]
    pub fn id(&self) -> &SessionId {
        self.session.id()
    }

    /// The definition this session was created from.
    #[must_use]
    pub fn def(&self) -> &SessionDef {
        &self.session.def
    }

    /// The session's current health.
    #[must_use]
    pub fn health(&self) -> SessionHealth {
        self.session.health()
    }

    /// Definition, health and container record together.
    #[must_use]
    pub fn state(&self) -> SessionState {
        SessionState {
            def: self.session.def.clone(),
            health: self.session.health(),
            container: self.session.containers(),
        }
    }

    /// Everything another process needs to start a workload in this
    /// session.
    ///
    /// Meaningful only once the session is ready: it snapshots the
    /// emulator's injection and the container names, neither of which
    /// exists before bring-up finishes.
    ///
    /// # Errors
    ///
    /// Returns an error if the session has not resolved its emulator
    /// injection yet, or if its topology cannot be read.
    pub fn describe(&self) -> Result<SessionDescription> {
        self.session.describe()
    }

    /// Wait until the session is healthy, terminally failed, or `timeout`
    /// elapses.
    ///
    /// Time spent pulling or building a container image does not count
    /// against the timeout: those are bounded by a registry and a network
    /// rather than by mirage, and the timeout exists to catch a session
    /// that is *stuck*, not one that is slow.
    ///
    /// # Errors
    ///
    /// Returns [`MirageError::Timeout`] if the session is still not
    /// settled when the deadline passes.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<SessionHealth> {
        let mut watch = self.session.watch_health();
        let id = self.session.id().clone();

        // Suspending the clock — rather than restarting it each time round
        // the loop — is what actually implements "slow is not stuck". Each
        // phase reports itself exactly once and the work then happens
        // inside a single blocking call, so a multi-gigabyte pull produces
        // one health event and then silence: a deadline merely *reset* on
        // that one event still expires mid-pull, and `mirage run` tears
        // down a session whose image was downloading normally.
        //
        // Waiting unbounded is safe because bring-up always publishes
        // again: it records a terminal `failed` health on any error, and
        // the phase callback fires on the way out of every step.
        let mut deadline = Some(tokio::time::Instant::now() + timeout);
        loop {
            {
                let health = watch.borrow_and_update().clone();
                if health.is_settled() {
                    return Ok(health);
                }
                deadline = if state::is_externally_bounded(health.state.as_deref()) {
                    None
                } else {
                    // Leaving an externally-bounded phase restarts the
                    // clock: time spent pulling must not be charged to the
                    // steps after it either.
                    Some(deadline.unwrap_or_else(|| tokio::time::Instant::now() + timeout))
                };
            }
            let changed = match deadline {
                Some(deadline) => tokio::time::timeout_at(deadline, watch.changed()).await,
                None => Ok(watch.changed().await),
            };
            match changed {
                // Health changed; loop and re-inspect.
                Ok(Ok(())) => {}
                // The session was dropped while we waited.
                Ok(Err(_)) => return Err(MirageError::SessionNotFound(id.to_string())),
                Err(_elapsed) => {
                    return Err(MirageError::Timeout(format!(
                        "session {id} did not become ready within {timeout:?} \
                         (last state: {})",
                        watch
                            .borrow()
                            .state
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string())
                    )));
                }
            }
        }
    }

    /// Start a workload in this session and return the running exec.
    ///
    /// # Errors
    ///
    /// Returns an error if the process grid cannot be built (an
    /// unreadable topology, an impossible world size).
    pub async fn exec(
        &self,
        def: &ExecDef,
    ) -> Result<(Arc<crate::exec::Exec>, mpsc::Receiver<OutputChunk>)> {
        self.session.start_exec(def).await
    }

    /// Whether every container backing this session is still running.
    ///
    /// `true` for a non-containerised session, which has none.
    #[must_use]
    pub fn containers_alive(&self) -> bool {
        self.session.containers_alive()
    }

    /// Tear the session down and wait until it is really gone.
    ///
    /// Terminates every exec and its process tree, stops the emulator
    /// daemon, and removes the containers and network. Returns only once
    /// all of it has happened, so a caller that awaits this can state —
    /// not hope — that the run left nothing behind.
    pub async fn destroy(&self) {
        self.session.teardown().await;
    }

    /// Synchronously `SIGKILL` every process in the session.
    ///
    /// A backstop for contexts that cannot await: a `Drop`, a panic
    /// handler, or a test cleaning up after a failed assertion. It does
    /// not remove containers — use [`Run::destroy`] for that — it only
    /// guarantees no workload process outlives this one.
    pub fn kill_now(&self) {
        self.session.kill_now();
    }

    // ---- bring-up -------------------------------------------------------

    /// Resolve the emulator injection, start any containers, and start
    /// the emulator daemon.
    ///
    /// Failure is recorded as terminal health rather than thrown away:
    /// the session stays alive so the caller can read *why* it failed and
    /// [`Run::wait_ready`] resolves instead of timing out.
    async fn bring_up(session: Arc<Session>) {
        match Self::bring_up_inner(&session).await {
            Ok(()) => {
                session.set_phase(true, state::READY, None);
                tracing::info!(session = %session.id(), "session ready");
            }
            Err(e) => {
                tracing::warn!(session = %session.id(), "session bring-up failed: {e}");
                session.set_health(SessionHealth::failed(e.to_string()));
                // Release whatever the failed bring-up did manage to
                // create. Without this a session that failed halfway
                // leaves containers and a network behind.
                session.teardown().await;
                // Teardown ends in `stopped`; restore the terminal failure
                // so the reason survives for the caller to read.
                session.set_health(SessionHealth::failed(e.to_string()));
            }
        }
    }

    async fn bring_up_inner(session: &Arc<Session>) -> Result<()> {
        session.set_phase(false, state::PREPARING, None);

        // Resolve the emulator injection. This is where a missing runtime
        // library or an unresolvable agent surfaces, and it must surface
        // now rather than at first exec: a session that reports ready and
        // then fails every exec is far harder to diagnose.
        let ctx = session.ctx.clone();
        let injection = Self::resolve_injection(&ctx).await?;
        session.set_injection(injection.clone());

        if session.ctx.profile.containerize.is_some() {
            Self::bring_up_containers(session, &injection).await?;
        }

        if session.ctx.daemon {
            let ctx = session.ctx.clone();
            let daemon = tokio::task::spawn_blocking(move || {
                let Some(backend) =
                    mirage_core::emulator::get_emulator_backend(&ctx.profile.emulator.emulator)
                else {
                    return Ok(None);
                };
                backend.start_daemon(&ctx)
            })
            .await
            .map_err(|e| MirageError::other(format!("emulator daemon task failed: {e}")))?;
            match daemon {
                Ok(handle) => {
                    if handle.is_some() {
                        tracing::info!(session = %session.id(), "emulator daemon hosted");
                    }
                    session.set_emulator_daemon(handle);
                }
                Err(e) => {
                    // Not fatal: an emulator whose daemon will not start
                    // may still run in-process, and the per-exec injection
                    // fails loudly if it genuinely cannot.
                    tracing::warn!(
                        session = %session.id(),
                        "emulator daemon failed to start ({e}); continuing in-process"
                    );
                }
            }
        }

        Ok(())
    }

    /// Compute the emulator injection for a session.
    ///
    /// Backends do blocking filesystem work (probing for libraries,
    /// writing config), so this runs on a blocking thread.
    async fn resolve_injection(ctx: &SessionContext) -> Result<InjectionDef> {
        let ctx = ctx.clone();
        tokio::task::spawn_blocking(move || {
            let kind = &ctx.profile.emulator.emulator;
            match mirage_core::emulator::get_emulator_backend(kind) {
                Some(backend) => backend.injection_def(&ctx),
                None => Err(MirageError::other(format!("unknown emulator `{kind}`"))),
            }
        })
        .await
        .map_err(|e| MirageError::other(format!("emulator injection task failed: {e}")))?
    }

    /// Start the per-node containers and network for a containerised
    /// session.
    async fn bring_up_containers(session: &Arc<Session>, injection: &InjectionDef) -> Result<()> {
        let Some(mut def) = session.ctx.profile.containerize.clone() else {
            return Ok(());
        };
        let plan = crate::session::plan_container(&session.ctx, injection);
        def.mounts.extend(plan.mounts);

        let node_count = crate::session::resolve_node_count(&session.ctx.profile)?;
        let host_gpus = injection.host_gpus;
        let id = session.id().clone();
        let watcher = session.clone();

        // The container engine is entirely blocking (it shells out to
        // podman/docker and waits), so all of it runs on a blocking
        // thread. Progress is reported back through session health, which
        // is a watch channel and safe to publish to from anywhere.
        let result = tokio::task::spawn_blocking(move || -> Result<_> {
            let engine = mirage_container::Engine::resolve(&def)
                .map_err(|e| MirageError::other(e.to_string()))?;

            // Profile hacks build a derived image once, keyed by the base
            // image plus the hack set, and run that instead.
            if let (Some(tag), Some(dockerfile)) = (
                mirage_core::profile::hacks_image_tag(&def.image, &def.hacks),
                mirage_core::profile::hacks_dockerfile(&def.image, &def.hacks),
            ) {
                if engine.image_present(&tag) {
                    watcher.set_phase(
                        false,
                        state::BUILDING,
                        Some(format!("derived image {tag} already built")),
                    );
                } else {
                    watcher.set_phase(
                        false,
                        state::BUILDING,
                        Some(format!(
                            "building derived image {tag} from {} (this can take a while)…",
                            def.image
                        )),
                    );
                    engine.build_image(&tag, &dockerfile).map_err(|e| {
                        MirageError::other(format!("building derived image {tag} failed: {e}"))
                    })?;
                }
                def.image = tag;
            }

            let head_port = crate::session::pick_head_port();
            let node_env = plan.env.clone();

            // Name the phase that was in flight when a failure happened,
            // so the error says "pulling image X failed: ..." rather than
            // just relaying an opaque provider error.
            let mut last_phase: Option<mirage_container::BringUpPhase> = None;
            let outcome = engine.bring_up(
                &id,
                &def,
                host_gpus,
                node_count,
                head_port,
                |_rank| node_env.clone(),
                |phase| {
                    let (state, message) = phase.health();
                    tracing::info!(state, "{message}");
                    watcher.set_phase(false, state, Some(message));
                    last_phase = Some(phase);
                },
            );

            outcome.map_err(|e| {
                let context = last_phase.map_or_else(
                    || format!("container bring-up failed: {e}"),
                    |p| format!("{} failed: {e}", p.message().trim_end_matches('…').trim_end()),
                );
                MirageError::other(context)
            })
        })
        .await
        .map_err(|e| MirageError::other(format!("container bring-up task failed: {e}")))??;

        let (containers, clients) = result;
        // The clients are the containers' lifetime: the session holds
        // them so that dropping the session stops the containers.
        //
        // Unless teardown got there first, which it can: all of the above
        // ran on a blocking thread, and a `wait_ready` timeout or a Ctrl-C
        // tears the session down without waiting for it. The session hands
        // the containers back in that case and they are ours to remove —
        // nothing else knows they exist.
        if let Some((containers, mut clients)) = session.set_containers(containers, clients) {
            tracing::warn!(
                session = %session.id(),
                "container bring-up finished after teardown; removing what it created"
            );
            let _ = tokio::task::spawn_blocking(move || {
                for client in &mut clients {
                    client.kill();
                }
                mirage_core::container::teardown(&containers);
            })
            .await;
        }
        Ok(())
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.session.kill_now();
    }
}
