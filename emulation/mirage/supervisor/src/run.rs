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
use mirage_core::exec::{ExecDef, ExecId, InjectionDef};
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
    /// Returns an error if the profile or its topology cannot be
    /// resolved, which is deliberately checked here rather than during
    /// bring-up: a session naming a profile that does not exist should
    /// fail at creation, not come up and fail every exec.
    pub fn start(req: CreateSessionRequest) -> Result<Self> {
        let profile = resolve_profile(&req.profile)?;
        let id = req.id.unwrap_or_else(SessionId::generate);
        let def = make_def(id.clone(), req.profile, req.workdir, req.daemon);
        let session = Session::new(def, profile)?;

        tracing::info!(session = %id, "session created");
        // Declared before the task exists, so there is no instant in
        // which teardown could look and see no bring-up in flight. The
        // `JoinHandle` is dropped on purpose: aborting this task would
        // not stop the blocking `podman` call inside it, so what teardown
        // waits on is the session's own signal rather than the handle —
        // see [`Session::await_bring_up`].
        session.begin_bring_up();
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

    /// Declare how many processes per node this run's job has.
    ///
    /// See [`Session::set_job_shape`](crate::session::Session::set_job_shape):
    /// call it before serving the control socket, so that a `mirage exec`
    /// is never handed a description of a differently-shaped job than the
    /// one this run's own ranks are in.
    pub fn set_job_shape(&self, nproc_per_node: u32) {
        self.session.set_job_shape(nproc_per_node);
    }

    /// The session's current health.
    #[must_use]
    pub fn health(&self) -> SessionHealth {
        self.session.health()
    }

    /// Subscribe to health changes.
    ///
    /// How `mirage run` shows bring-up progress: every phase publishes
    /// here, and a client that only polled `health()` would miss the ones
    /// between polls — which for a pull is most of them.
    #[must_use]
    pub fn watch_health(&self) -> tokio::sync::watch::Receiver<SessionHealth> {
        self.session.watch_health()
    }

    /// Take a borrower's lease on this session, unless it is tearing down.
    ///
    /// `borrower` is the pid of the process taking it, where the caller
    /// can say — the control socket asks the kernel for its client's
    /// credentials. `exec` is the id that client will stamp into its
    /// workload's environment, which is how the session recognises that
    /// workload after it has stopped descending from the client. See
    /// [`Session::attach`](crate::session::Session::attach).
    #[must_use]
    pub fn attach(
        &self,
        borrower: Option<u32>,
        exec: Option<ExecId>,
    ) -> Option<crate::session::SessionLease> {
        self.session.attach(borrower, exec)
    }

    /// Stop what a borrower that has just gone away left running in the
    /// session.
    ///
    /// See
    /// [`Session::reap_departed_borrowers`](crate::session::Session::reap_departed_borrowers):
    /// called by the control socket when a lease ends because its client
    /// did, which is the moment the run learns that a `mirage exec` is
    /// not coming back.
    pub async fn reap_departed_borrowers(&self) {
        self.session.reap_departed_borrowers().await;
    }

    /// How many `mirage exec` clients are borrowing this session.
    #[must_use]
    pub fn borrowers(&self) -> usize {
        self.session.borrowers()
    }

    /// Resolve once no borrower is left.
    pub async fn wait_for_borrowers(&self) {
        self.session.wait_for_borrowers().await;
    }

    /// Resolve once teardown has asked borrowers to let go.
    pub async fn wait_closing(&self) {
        self.session.wait_closing().await;
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

    /// Wait until the session is healthy, terminally failed, or bring-up
    /// stops making progress for `timeout`.
    ///
    /// `timeout` bounds the *gap between phases*, not the total. Bring-up
    /// is a sequence of steps whose count depends on the session — one
    /// container launch per node, launched serially — and a budget for
    /// the whole thing is therefore a budget that a large session
    /// exceeds by being large rather than by being broken. A four-node
    /// session taking twenty seconds a node is healthy, and charging it
    /// against a single sixty-second deadline tore it down.
    ///
    /// Time spent pulling or building a container image does not count at
    /// all: those are bounded by a registry and a network rather than by
    /// mirage. Progress-based timing alone would not cover them, because
    /// a pull reports itself once and then goes quiet for however long it
    /// takes — so the clock is *suspended* for those phases and merely
    /// *restarted* for the rest.
    ///
    /// # Errors
    ///
    /// Returns [`MirageError::Timeout`] if the session is still not
    /// settled when the deadline passes.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<SessionHealth> {
        let mut watch = self.session.watch_health();
        let id = self.session.id().clone();

        // Two mechanisms, because there are two ways to be slow without
        // being stuck, and neither one covers the other.
        //
        // *Suspending* the clock covers a phase whose duration mirage does
        // not control. A pull reports itself exactly once and the work
        // then happens inside a single blocking call, so a multi-gigabyte
        // download produces one health event and then silence — a
        // deadline merely restarted on that event still expires mid-pull,
        // and `mirage run` tears down a session whose image was
        // downloading normally.
        //
        // *Restarting* the clock on every event covers a phase made of
        // many steps. Node containers are launched one at a time, each
        // with its own bounded wait, and all of them publish under the
        // same `starting` state — so a single deadline spanning the whole
        // phase is a budget for the session's *size*, and a four-node
        // session at twenty seconds a node was timed out for being big.
        // Restarting turns the deadline into what it is documented to be:
        // a detector for bring-up that has stopped moving.
        //
        // Waiting unbounded during a suspended phase is safe because
        // bring-up always publishes again: it records a terminal `failed`
        // health on any error, and the phase callback fires on the way out
        // of every step.
        loop {
            // Recomputed every time round, which is the restart: the
            // deadline belongs to the wait for the *next* event, not to
            // bring-up as a whole.
            let deadline = {
                let health = watch.borrow_and_update().clone();
                if health.is_settled() {
                    return Ok(health);
                }
                if state::is_externally_bounded(health.state.as_deref()) {
                    None
                } else {
                    Some(tokio::time::Instant::now() + timeout)
                }
            };
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
                        "session {id} made no progress for {timeout:?} \
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
    ///
    /// Both arms end the bring-up before doing anything that waits, and
    /// they end it *after* publishing, so that a teardown blocked on this
    /// bring-up (a Ctrl-C during an image pull) is released only once the
    /// session's health is the truth and everything created is either the
    /// session's or already removed.
    async fn bring_up(session: Arc<Session>) {
        match Self::bring_up_inner(&session).await {
            Ok(()) => {
                // Refused if teardown got here first, which it can: this
                // is the instant after a Ctrl-C during a slow pull.
                session.set_phase(true, state::READY, None);
                session.finish_bring_up();
                tracing::info!(session = %session.id(), "session ready");
            }
            Err(e) => {
                // `debug`, not `warn`. The reason is published as the
                // session's health a line below, which is where every
                // caller reads it from and how `mirage run` renders the
                // fatal error — so logging it at a level that is on by
                // default printed the same paragraph twice, and for a
                // `--hack` build failure that paragraph is hundreds of
                // lines of provider output. One path reports an error;
                // this one only says when it happened.
                tracing::debug!(session = %session.id(), "session bring-up failed: {e}");
                // Published before the teardown below rather than after
                // it, and it stays published: `set_phase` will not let
                // `stopping` overwrite a terminal failure. Republishing
                // afterwards instead left whoever was waiting to read the
                // last value in between, which is `stopped` — a session
                // that "failed to start (stopped)", with the reason lost.
                session.set_health(SessionHealth::failed(e.to_string()));
                // Nothing further will be created, so a teardown waiting
                // on this bring-up may go ahead — including the one
                // below, which is this failure's own rollback and would
                // otherwise be waiting for itself.
                session.finish_bring_up();
                // Release whatever the failed bring-up did manage to
                // create. Without this a session that failed halfway
                // leaves containers and a network behind.
                session.teardown().await;
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

        // And whether the backend could host the daemon this session
        // needs — before anything is created, not after everything is.
        // `start_daemon` is the last step below, so a runtime that simply
        // cannot host one used to fail the run at the very end: image
        // pulled, network up, every container created and then removed
        // again, for a fact about a file that was knowable here. The
        // check is cheap and the message is the same one, arriving in a
        // second instead of a few minutes.
        if session.ctx.daemon {
            Self::check_daemon_capability(&ctx).await?;
        }

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
                    // Refused when teardown got here first; the daemon is
                    // then ours to stop, and nothing else knows it exists.
                    //
                    // `debug`, not `warn`. Nothing has gone wrong here:
                    // this is the design working — a Ctrl-C landed while
                    // a blocking start was in flight, and the thing it
                    // produced is being handed back and stopped. As a
                    // warning it was the only output an interrupted
                    // bring-up produced, so the user's Ctrl-C was
                    // answered with an internal tracing line about
                    // mirage's own bookkeeping and nothing else. What
                    // they should see instead is printed by `mirage run`.
                    if let Some(orphan) = session.set_emulator_daemon(handle) {
                        tracing::debug!(
                            session = %session.id(),
                            "emulator daemon started after teardown; stopping it"
                        );
                        let _ = tokio::task::spawn_blocking(move || orphan.stop()).await;
                    }
                }
                Err(e) => {
                    // Fatal, and the reasoning that said otherwise was
                    // the trap. "It may still run in-process" is true and
                    // is exactly the problem: in-process emulation cannot
                    // share GPU memory between processes, so a multi-GPU
                    // collective silently computes something else. The
                    // run was not asked for "emulation, any kind" — this
                    // arm is only reachable when the user did not pass
                    // `--in-process`, and `start_daemon` is documented to
                    // return `Err` only when a daemon was expected and
                    // could not be started (a backend that needs none
                    // returns `Ok(None)` and never lands here).
                    //
                    // Failing loudly at the first exec, the other half of
                    // the old reasoning, does not happen either: the
                    // workload runs, it just runs on one GPU's worth of
                    // memory and returns a plausible wrong number.
                    return Err(MirageError::other(format!(
                        "the {} emulator daemon could not be started: {e}\n\
                         The daemon is what lets several processes share emulated GPU \
                         memory, so multi-GPU collectives need it. Pass `--in-process` \
                         to run without it — results from a single process are still \
                         correct.",
                        session.ctx.profile.emulator.emulator
                    )));
                }
            }
        }

        Ok(())
    }

    /// Fail now if the backend could not host this session's daemon.
    ///
    /// Blocking for the same reason [`Self::resolve_injection`] is: the
    /// answer comes from the filesystem, and for a backend that hosts a
    /// daemon by `dlopen`ing a library it comes from loading one.
    ///
    /// An unknown backend is not this check's business — `resolve_injection`
    /// has already refused the session by the time anything calls here —
    /// so it passes rather than inventing a second phrasing of that error.
    async fn check_daemon_capability(ctx: &SessionContext) -> Result<()> {
        let kind = ctx.profile.emulator.emulator.clone();
        tokio::task::spawn_blocking(move || {
            mirage_core::emulator::get_emulator_backend(&kind)
                .map_or(Ok(()), |backend| backend.daemon_capability())
        })
        .await
        .map_err(|e| MirageError::other(format!("emulator capability task failed: {e}")))?
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
        // Checked here, before the two lists become one, because after
        // that there is nothing left to tell them apart.
        //
        // `resolve_mount` runs on the combined list and can only ask
        // `covers_mirage_dir`, which is true for the reserved directory
        // and its ancestors and false for everything *below* it — it has
        // to be, because mirage's own mounts are all down there and a
        // check that caught them would refuse every containerised
        // session. So a user mount at `/mnt/mirage/runtime/rj_config.json`
        // passed it, overlaid the emulator's config inside the scratch
        // mount, and the workload ran unemulated against the real device
        // with nothing said. Provenance is what the later check lacks,
        // and at this line it is still free: everything in `def.mounts`
        // is the user's and everything in `plan.mounts` is mirage's.
        reject_mounts_over_mirages_own(&def.mounts, &plan.mounts)?;
        def.mounts.extend(plan.mounts);

        // The shape the session was created with, not a fresh read of the
        // topology: bring-up must start containers for exactly the nodes
        // every later `describe` will name.
        let node_count = session.node_count();
        let host_gpus = injection.host_gpus;
        let id = session.id().clone();
        let watcher = session.clone();
        // The switch teardown flips to end this bring-up rather than wait
        // it out. Taken here, where the engine is built, because that is
        // the only place an engine and the session that owns it are both
        // in scope.
        let cancel = session.cancel_switch();

        // The container engine is entirely blocking (it shells out to
        // podman/docker and waits), so all of it runs on a blocking
        // thread. Progress is reported back through session health, which
        // is a watch channel and safe to publish to from anywhere.
        let result = tokio::task::spawn_blocking(move || -> Result<_> {
            let engine = mirage_container::Engine::resolve(&def)
                .map_err(|e| MirageError::other(e.to_string()))?
                .with_cancel(cancel);

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
                    |p| {
                        format!(
                            "{} failed: {e}",
                            p.message().trim_end_matches('…').trim_end()
                        )
                    },
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
            // `debug`, for the reason given where the emulator daemon is
            // handed back: this is the interrupt path behaving correctly,
            // and a warning about it is mirage narrating its own
            // internals over the top of the answer the user asked for.
            tracing::debug!(
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

/// Refuse a user mount that overlaps one of mirage's own.
///
/// `mirage` bind-mounts its binary, the session scratch directory, the
/// config directory and the emulator's libraries into every node
/// container. A user `--mount` landing on any of those destinations —
/// at it, above it, or inside it — replaces something the session needs,
/// and the failure is silent in the worst case: overlaying the scratch
/// mount hides the emulator's `rj_config.json`, so the interposer finds
/// no simulation config and the workload runs unemulated against the
/// real device, at full speed, producing results that look fine.
///
/// # Errors
///
/// Returns an error naming both the user's mount and the mirage
/// destination it lands on.
fn reject_mounts_over_mirages_own(
    user: &[mirage_core::profile::FileMount],
    ours: &[mirage_core::profile::FileMount],
) -> Result<()> {
    for mount in user {
        let Some(hit) = ours.iter().find(|ours| {
            mirage_core::container::container_paths_overlap(
                &mount.container_path,
                &ours.container_path,
            )
        }) else {
            continue;
        };
        return Err(MirageError::other(format!(
            "--mount {}: `{}` inside the container is mirage's own — it is where mirage \
             mounts {} for this session, and a mount there would replace it. A session whose \
             scratch or library mounts are covered runs without emulation rather than \
             failing, so this is refused before anything is created. Choose another \
             destination; every path outside `{}` is yours.",
            mount.to_volume_arg(),
            hit.container_path,
            hit.host_path,
            mirage_core::container::CONTAINER_MIRAGE_DIR,
        )));
    }
    Ok(())
}

impl Drop for Run {
    fn drop(&mut self) {
        self.session.kill_now();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::common::MaybeRef;
    use mirage_core::emulator::{EmulatorDef, ExecMode};
    use mirage_core::profile::ProfileDef;
    use mirage_core::topology::TopologyDef;

    const TIMEOUT: Duration = Duration::from_secs(60);

    /// A `Run` over a session nothing is bringing up, so a test can drive
    /// its health by hand.
    ///
    /// The real bring-up path is covered end to end in
    /// `supervisor/tests/run.rs`; what these need is the opposite — a
    /// session that publishes exactly the phases the test says it does,
    /// at exactly the times it says, which no backend can be persuaded to
    /// do.
    ///
    /// The path override is installed and removed inside this function,
    /// under the shared lock, and neither outlives it. `TEST_ROOT` is
    /// process-wide, so a test that held it across its `await`s would
    /// redirect every other test in this binary for as long as it ran —
    /// and holding a `std::sync::Mutex` across an await is denied by the
    /// workspace lints for exactly that class of reason. Nothing after
    /// construction consults `paths`: the session captured its scratch
    /// directory, and `wait_ready` only reads a watch channel.
    fn stalled_run(dir: &std::path::Path) -> Arc<Run> {
        let _guard = mirage_core::paths::test_env_lock();
        mirage_core::paths::set_test_root(dir);
        let id = SessionId::new("waitready").unwrap();
        let profile = stub_profile();
        let def = make_def(id, MaybeRef::Owned(profile.clone()), "/".to_string(), false);
        let run = Arc::new(Run {
            session: Session::new(def, profile).unwrap(),
        });
        mirage_core::paths::clear_test_root();
        run
    }

    /// A one-node profile naming an emulator no build registers.
    ///
    /// Nothing brings a session on it up by accident, which is what
    /// [`stalled_run`] needs — and when one *is* brought up, bring-up
    /// fails at its first step with a message worth asserting on.
    fn stub_profile() -> ProfileDef {
        ProfileDef {
            name: "p".to_string(),
            description: None,
            emulator: EmulatorDef {
                emulator: "stub".to_string(),
                plugins: Default::default(),
                exec_mode: ExecMode::Functional,
                options: Default::default(),
                topology: MaybeRef::Owned(TopologyDef {
                    num_nodes: 1,
                    gpus_per_node: 1,
                    agent: MaybeRef::Ref("MI350X".to_string()),
                }),
            },
            containerize: None,
        }
    }

    #[tokio::test]
    async fn a_waiter_that_arrives_late_still_learns_why_bring_up_failed() {
        // `wait_ready` reads a `watch`, and a `watch` keeps only the last
        // value published. Bring-up records its failure and then tears
        // the session down to roll back what it created, so `stopping`
        // and `stopped` land immediately behind the reason — and whoever
        // was not already awake for the failure read `stopped` instead.
        // `mirage run` then said "session failed to start (stopped)",
        // which names nothing the user can fix.
        //
        // Deterministic where the race is not: this waiter arrives after
        // the whole sequence, which is the case a watch channel cannot
        // replay.
        let dir = tempfile::tempdir().unwrap();
        let run = stalled_run(dir.path());

        run.session.set_health(SessionHealth::failed(
            "pulling image quay.io/nope:1 failed: not found",
        ));
        run.session.teardown().await;

        let health = run
            .wait_ready(TIMEOUT)
            .await
            .expect("a settled session must not be timed out");
        assert!(!health.healthy, "{health:?}");
        assert_eq!(health.state.as_deref(), Some(state::FAILED), "{health:?}");
        assert!(
            health
                .message
                .as_deref()
                .unwrap_or_default()
                .contains("quay.io/nope:1"),
            "the reason is what the user acts on: {health:?}"
        );
    }

    #[tokio::test]
    async fn a_bring_up_that_cannot_start_reports_its_reason_and_tears_itself_down() {
        // The real path, end to end: `Run::start` spawns bring-up,
        // bring-up fails at its first step, and it calls `teardown`
        // itself to roll back. That teardown waits for the bring-up that
        // is in flight — which is this very task — so an ordering mistake
        // there is not a leak but a hang, and this is the test that would
        // catch it.
        let dir = tempfile::tempdir().unwrap();
        let run = {
            let _guard = mirage_core::paths::test_env_lock();
            mirage_core::paths::set_test_root(dir.path());
            let run = Run::start(CreateSessionRequest {
                id: Some(SessionId::new("no-backend").unwrap()),
                profile: MaybeRef::Owned(stub_profile()),
                workdir: "/".to_string(),
                daemon: false,
            })
            .expect("a session with a resolvable profile is created");
            mirage_core::paths::clear_test_root();
            run
        };

        let health = tokio::time::timeout(TIMEOUT, run.wait_ready(TIMEOUT))
            .await
            .expect("bring-up and its rollback must not deadlock")
            .expect("a session that failed to start is settled, not timed out");
        assert!(!health.healthy, "{health:?}");
        assert!(
            health
                .message
                .as_deref()
                .unwrap_or_default()
                .contains("unknown emulator `stub`"),
            "the caller must be told what is wrong with the profile: {health:?}"
        );

        // And the run can still be destroyed, without waiting for a
        // bring-up that has already finished.
        tokio::time::timeout(TIMEOUT, run.destroy())
            .await
            .expect("teardown of a failed session must not hang");
    }

    #[tokio::test(start_paused = true)]
    async fn bring_up_that_keeps_moving_is_never_timed_out() {
        // The regression: the deadline used to span the whole of
        // bring-up, and node containers are launched one at a time under
        // a single `starting` state. Four nodes at twenty seconds each is
        // a healthy session that exceeded a sixty-second budget by being
        // four nodes, and `mirage run` tore it down.
        let dir = tempfile::tempdir().unwrap();
        let run = stalled_run(dir.path());

        // Eight steps, each comfortably inside the timeout, adding up to
        // well over it.
        tokio::spawn({
            let run = run.clone();
            async move {
                for step in 0..8 {
                    tokio::time::sleep(TIMEOUT / 2).await;
                    run.session
                        .set_phase(false, state::STARTING, Some(format!("node {step}")));
                }
                tokio::time::sleep(TIMEOUT / 2).await;
                run.session.set_phase(true, state::READY, None);
            }
        });

        let health = run
            .wait_ready(TIMEOUT)
            .await
            .expect("a session that is still making progress must not be timed out");
        assert!(health.healthy);
    }

    #[tokio::test(start_paused = true)]
    async fn bring_up_that_stops_moving_is_timed_out() {
        // The other half: "restart the clock on progress" must not become
        // "never time out". A session that publishes nothing further is
        // exactly what the deadline exists to catch.
        let dir = tempfile::tempdir().unwrap();
        let run = stalled_run(dir.path());
        run.session.set_phase(false, state::STARTING, None);

        let err = run.wait_ready(TIMEOUT).await.unwrap_err();
        assert!(
            matches!(err, MirageError::Timeout(_)),
            "expected a timeout, got {err}"
        );
        assert!(err.to_string().contains("made no progress"), "{err}");
    }

    #[tokio::test(start_paused = true)]
    async fn an_image_pull_is_not_charged_against_the_clock_at_all() {
        // A pull reports itself once and then goes quiet for however long
        // the registry and the network take. Restarting the clock on
        // progress does not help when there is no further progress to
        // report, so the externally-bounded phases suspend it outright.
        let dir = tempfile::tempdir().unwrap();
        let run = stalled_run(dir.path());
        run.session.set_phase(
            false,
            state::PULLING,
            Some("pulling image big:latest".to_string()),
        );

        tokio::spawn({
            let run = run.clone();
            async move {
                // Twenty times the timeout, in one silent stretch.
                tokio::time::sleep(TIMEOUT * 20).await;
                run.session.set_phase(true, state::READY, None);
            }
        });

        let health = run
            .wait_ready(TIMEOUT)
            .await
            .expect("time spent pulling must not count against readiness");
        assert!(health.healthy);
    }
}
