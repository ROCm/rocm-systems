//! [`SessionManager`]: the in-process implementation of
//! [`MirageCtl`](mirage_core::ctl::MirageCtl).
//!
//! One manager owns every session in the daemon. Configuration
//! (profiles, topologies, agents) is still read from and written to disk,
//! because those are user-authored documents; everything about a *running*
//! session lives here in memory.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mirage_core::agent::AgentDef;
use mirage_core::ctl::{
    CreateSessionRequest, MirageCtl, StreamPacket, StreamPacketStream,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecDef, ExecId, ExecRef, ExecStatus, InjectionDef};
use mirage_core::profile::ProfileDef;
use mirage_core::session::{SessionContext, SessionDef, SessionHealth, SessionId, SessionState, state};
use mirage_core::topology::TopologyDef;
use tokio::sync::watch;

use crate::output::DEFAULT_REPLAY_BYTES;
use crate::session::{Session, make_def, resolve_profile};

/// Tuning knobs for a manager.
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// Output retained per exec for replay, in bytes.
    pub replay_bytes: usize,
    /// How many *finished* execs a session keeps before the oldest are
    /// forgotten. Running execs are never forgotten.
    ///
    /// The daemon outlives any individual command, so unbounded history
    /// is a slow memory leak: a session running execs in a loop would
    /// retain every one of them, with its output, forever.
    pub max_finished_execs: usize,
}

/// Default cap on remembered finished execs per session.
pub const DEFAULT_MAX_FINISHED_EXECS: usize = 200;

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            replay_bytes: DEFAULT_REPLAY_BYTES,
            max_finished_execs: DEFAULT_MAX_FINISHED_EXECS,
        }
    }
}

/// Owns every live session.
#[derive(Debug)]
pub struct SessionManager {
    // A std lock rather than an async one: every critical section here is
    // a map lookup or insert, and none of them awaits. An async lock
    // would add a suspension point to operations that never block, and
    // would make the synchronous [`SessionManager::kill_all_now`]
    // backstop impossible to write.
    sessions: std::sync::RwLock<BTreeMap<SessionId, Arc<Session>>>,
    config: ManagerConfig,
    /// Set when the daemon has been asked to exit.
    ///
    /// A level-triggered `watch` rather than a `Notify` for the same
    /// reason as [`crate::exec::Exec`]'s completion signal: `Notify` only
    /// registers a waiter when its future is first polled, so a shutdown
    /// request landing between a waiter's flag check and its `await`
    /// would be lost — and the daemon would ignore `mirage daemon stop`.
    shutdown: watch::Sender<bool>,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(ManagerConfig::default())
    }
}

impl SessionManager {
    /// Build an empty manager.
    #[must_use]
    pub fn new(config: ManagerConfig) -> Self {
        Self {
            sessions: std::sync::RwLock::new(BTreeMap::new()),
            config,
            shutdown: watch::channel(false).0,
        }
    }

    /// Resolves when someone has requested a daemon shutdown.
    pub async fn wait_for_shutdown(&self) {
        let mut rx = self.shutdown.subscribe();
        // `wait_for` inspects the current value before suspending, so a
        // request that already arrived is not missed.
        if rx.wait_for(|requested| *requested).await.is_err() {
            // The manager is being dropped; treat that as a shutdown.
        }
    }

    /// Whether a shutdown has been requested.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        *self.shutdown.borrow()
    }

    /// How many sessions are live.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.read().len()
    }

    /// Synchronously `SIGKILL` every process in every session.
    ///
    /// A backstop for contexts that cannot await: a `Drop`, a panic
    /// handler, or a test harness cleaning up after a failed assertion.
    /// It does not tear sessions down or remove containers — use
    /// [`SessionManager::shutdown_all`] for that — it only guarantees
    /// that no workload process outlives the daemon.
    pub fn kill_all_now(&self) {
        for session in self.read().values() {
            session.kill_now();
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<SessionId, Arc<Session>>> {
        self.sessions.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<SessionId, Arc<Session>>> {
        self.sessions.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Destroy every session and wait for teardown to finish.
    ///
    /// The daemon calls this on its way out, so that terminating the
    /// daemon terminates every workload it started rather than orphaning
    /// them to init. Sessions are torn down concurrently: doing it in
    /// sequence would make shutdown take the SIGTERM grace period times
    /// the number of sessions.
    pub async fn shutdown_all(&self) {
        // Latch the shutdown flag first, whatever brought us here.
        //
        // Only the `daemon_shutdown` RPC used to set it, so a `SIGTERM`,
        // an idle timeout or a deleted socket left `session_create` still
        // accepting work — and a request already in flight could insert
        // its session after the map below had been taken, producing a
        // session the daemon reports as created and then never tears
        // down. Every shutdown path runs through here, so this is the
        // place the guard belongs.
        self.shutdown.send_replace(true);

        let sessions: Vec<Arc<Session>> = {
            let mut guard = self.write();
            std::mem::take(&mut *guard).into_values().collect()
        };
        if sessions.is_empty() {
            return;
        }
        tracing::info!(count = sessions.len(), "tearing down all sessions");
        futures::future::join_all(sessions.iter().map(|s| s.teardown())).await;

        // Last-resort sweep over the sessions we just took. `teardown` is
        // the correct path and normally leaves nothing, but a process
        // that survived it (a container exec whose provider hung, say)
        // must still not outlive the daemon — and by this point these
        // sessions are out of the map, so `kill_all_now` can no longer
        // see them.
        for session in &sessions {
            session.kill_now();
        }
    }

    /// Look up a session, or report it missing.
    fn get_sync(&self, id: &SessionId) -> Result<Arc<Session>> {
        self.read()
            .get(id)
            .cloned()
            .ok_or_else(|| MirageError::SessionNotFound(id.to_string()))
    }

    async fn get(&self, id: &SessionId) -> Result<Arc<Session>> {
        self.get_sync(id)
    }

    /// Look up an exec, or report it (or its session) missing.
    async fn get_exec(&self, r: &ExecRef) -> Result<Arc<crate::exec::Exec>> {
        self.get(&r.session)
            .await?
            .exec(&r.exec)
            .ok_or_else(|| MirageError::ExecNotFound(r.exec.to_string()))
    }

    /// Bring a session up: resolve its emulator injection, start any
    /// containers, and start the emulator daemon.
    ///
    /// Runs in the background so `session_create` returns promptly.
    /// Failure is recorded as terminal health rather than thrown away:
    /// the session stays registered so a client can read *why* it failed,
    /// and `session_wait_ready` resolves instead of timing out.
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
                // leaves containers and a network behind, and the user
                // has no handle to remove them with.
                session.teardown().await;
                // Teardown ends in `stopped`; restore the terminal failure
                // so the reason survives for the client to read.
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
            let command = plan.command.clone();

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
                |_rank| command.clone(),
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

        let (containers, _cids) = result;
        session.set_containers(containers);
        Ok(())
    }
}

/// A mirage-generated line on an attached exec's stderr.
///
/// Used for things the client has to know but the workload did not say —
/// that its output was truncated, or that frames were skipped. It goes on
/// stderr, tagged, so it is distinguishable from the workload's own
/// output and does not corrupt a piped stdout.
fn notice(message: &str) -> StreamPacket {
    StreamPacket::Output {
        node: 0,
        stream: mirage_core::ctl::StdStream::Stderr,
        data: format!("mirage: {message}\n").into_bytes(),
    }
}

#[async_trait]
impl MirageCtl for SessionManager {
    // ---- Profiles -------------------------------------------------------
    //
    // Configuration is plain filesystem work with no session state
    // involved, so it delegates to the shared store. The CLI answers the
    // same queries from the same place without a daemon, which is why
    // `mirage profile list` does not start one.

    async fn profile_list(&self) -> Result<Vec<String>> {
        mirage_core::store::profile_list()
    }

    async fn profile_get(&self, name: &str) -> Result<ProfileDef> {
        mirage_core::store::profile_get(name)
    }

    async fn profile_put(&self, profile: &ProfileDef) -> Result<()> {
        mirage_core::store::profile_put(profile)
    }

    async fn profile_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::profile_delete(name)
    }

    // ---- Topologies -----------------------------------------------------

    async fn topology_list(&self) -> Result<Vec<String>> {
        mirage_core::store::topology_list()
    }

    async fn topology_get(&self, name: &str) -> Result<TopologyDef> {
        mirage_core::store::topology_get(name)
    }

    async fn topology_put(&self, name: &str, topology: &TopologyDef) -> Result<()> {
        mirage_core::store::topology_put(name, topology)
    }

    async fn topology_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::topology_delete(name)
    }

    // ---- Agents ---------------------------------------------------------

    async fn agent_list(&self) -> Result<Vec<String>> {
        mirage_core::store::agent_list()
    }

    async fn agent_get(&self, name: &str) -> Result<AgentDef> {
        mirage_core::store::agent_get(name)
    }

    async fn agent_put(&self, name: &str, agent: &AgentDef) -> Result<()> {
        mirage_core::store::agent_put(name, agent)
    }

    async fn agent_delete(&self, name: &str) -> Result<()> {
        mirage_core::store::agent_delete(name)
    }

    // ---- Sessions -------------------------------------------------------

    async fn session_list(&self) -> Result<Vec<SessionId>> {
        Ok(self.read().keys().cloned().collect())
    }

    async fn session_state(&self, id: &SessionId) -> Result<SessionState> {
        let session = self.get(id).await?;
        Ok(SessionState {
            def: session.def.clone(),
            health: session.health(),
            container: session.containers(),
        })
    }

    async fn session_health(&self, id: &SessionId) -> Result<SessionHealth> {
        Ok(self.get(id).await?.health())
    }

    async fn session_create(&self, req: CreateSessionRequest) -> Result<SessionDef> {
        if self.is_shutting_down() {
            return Err(MirageError::daemon("daemon is shutting down"));
        }
        // Resolve the profile up front: a session referring to a profile
        // that does not exist should fail at creation, not silently come
        // up and fail later.
        let profile = resolve_profile(&req.profile)?;
        let id = req.id.unwrap_or_else(SessionId::generate);
        let def = make_def(id.clone(), req.profile, req.workdir, req.daemon);
        let session = Session::new(def.clone(), profile)?;

        {
            let mut sessions = self.write();
            // Re-check under the write lock. `shutdown_all` latches the
            // flag and *then* takes the map, both of which race the
            // check above; re-reading here means either we insert before
            // the map is taken (and the session is torn down with the
            // rest) or we refuse. A session inserted after the take would
            // be one nothing ever destroys.
            if self.is_shutting_down() {
                return Err(MirageError::daemon("daemon is shutting down"));
            }
            if sessions.contains_key(&id) {
                return Err(MirageError::SessionExists(id.to_string()));
            }
            sessions.insert(id.clone(), session.clone());
        }

        tracing::info!(session = %id, "session created");
        tokio::spawn(Self::bring_up(session));
        Ok(def)
    }

    async fn session_wait_ready(
        &self,
        id: &SessionId,
        timeout: Duration,
    ) -> Result<SessionHealth> {
        let session = self.get(id).await?;
        let mut watch = session.watch_health();

        // The timeout is there to catch a *stuck* session, and a slow
        // container registry is not stuck. While the session is in a
        // phase whose duration is bounded by something outside mirage
        // (pulling or building an image), the clock is suspended
        // entirely.
        //
        // Suspending it — rather than restarting it each time round the
        // loop — is what actually implements that. Each phase reports
        // itself exactly once and the work then happens inside a single
        // blocking call, so a multi-gigabyte pull produces one health
        // event and then silence: a deadline merely *reset* on that one
        // event still expires mid-pull, and `mirage run` tears down a
        // session whose image was downloading normally.
        //
        // Waiting unbounded here is safe because bring-up always
        // publishes again: `bring_up` records a terminal `failed` health
        // on any error, and the phase callback fires on the way out of
        // every step.
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
                    // clock: the time spent pulling must not be charged
                    // to the steps after it either.
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
                Ok(Err(_)) => {
                    return Err(MirageError::SessionNotFound(id.to_string()));
                }
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

    async fn session_destroy(&self, id: &SessionId) -> Result<()> {
        // Remove first so the session is invisible to new requests while
        // it is being dismantled, then tear it down to completion. A
        // caller that gets `Ok` can rely on every process being reaped.
        let session = self
            .write()
            .remove(id)
            .ok_or_else(|| MirageError::SessionNotFound(id.to_string()))?;
        session.teardown().await;
        Ok(())
    }

    // ---- Execs ----------------------------------------------------------

    async fn exec_list(&self, session: &SessionId) -> Result<Vec<ExecId>> {
        Ok(self.get(session).await?.exec_ids())
    }

    async fn exec_status(&self, r: &ExecRef) -> Result<ExecStatus> {
        Ok(self.get_exec(r).await?.status())
    }

    async fn exec_get(&self, r: &ExecRef) -> Result<ExecDef> {
        Ok(self.get_exec(r).await?.def.clone())
    }

    async fn session_exec(&self, exec: &ExecDef) -> Result<ExecRef> {
        let session = self.get(&exec.session).await?;
        let started = session
            .start_exec(
                exec,
                self.config.replay_bytes,
                self.config.max_finished_execs,
            )
            .await?;
        Ok(ExecRef {
            session: exec.session.clone(),
            exec: started.id.clone(),
        })
    }

    async fn session_attach(&self, exec: &ExecRef) -> Result<StreamPacketStream> {
        let handle = self.get_exec(exec).await?;
        let sub = handle.hub.subscribe();

        // Replay first, then follow. Building the stream this way (rather
        // than handing back the broadcast receiver alone) is what makes
        // attaching to an already-finished exec return its whole output
        // and its exit code instead of nothing.
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        tokio::spawn(async move {
            // Say so when the replay is incomplete. The hub counts what
            // it evicted precisely so this is not guesswork, and a
            // truncated log that looks whole is worse than no log: a user
            // reads the surviving tail and concludes the error they are
            // hunting never happened.
            if sub.dropped_bytes > 0 {
                let note = notice(&format!(
                    "earlier output was dropped ({} bytes) to stay within \
                     the retained-output limit",
                    sub.dropped_bytes
                ));
                if tx.send(note).await.is_err() {
                    return;
                }
            }
            for packet in sub.replay {
                let terminal = matches!(packet, StreamPacket::ExecExit { .. });
                if tx.send(packet).await.is_err() || terminal {
                    return;
                }
            }
            let Some(mut live) = sub.live else {
                // Finished before we subscribed and the replay did not
                // include the exit (it was evicted). Synthesise it so the
                // client still terminates.
                if let Some(exit_code) = sub.finished {
                    let _ = tx.send(StreamPacket::ExecExit { exit_code }).await;
                }
                return;
            };
            loop {
                match live.recv().await {
                    Ok(packet) => {
                        let terminal = matches!(packet, StreamPacket::ExecExit { .. });
                        if tx.send(packet).await.is_err() || terminal {
                            return;
                        }
                    }
                    // This client fell far enough behind that the channel
                    // dropped packets. Keep going rather than
                    // disconnecting: a gap in output is bad, but silently
                    // losing the exit code would hang the client.
                    //
                    // Tell the client about the gap, though. Logging it
                    // in the daemon puts the one record of it somewhere
                    // the person reading the output will never look.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(dropped = n, "attach client fell behind; output skipped");
                        let note =
                            notice(&format!("{n} output frames were skipped (client too slow)"));
                        if tx.send(note).await.is_err() {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                }
            }
        });
        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn session_stdin(&self, exec: &ExecRef, data: &[u8]) -> Result<()> {
        self.get_exec(exec).await?.write_stdin(data).await
    }

    async fn session_stdin_close(&self, exec: &ExecRef) -> Result<()> {
        self.get_exec(exec).await?.close_stdin().await;
        Ok(())
    }

    async fn exec_resize(&self, exec: &ExecRef, rows: u16, cols: u16) -> Result<()> {
        self.get_exec(exec).await?.resize(rows, cols).await
    }

    async fn exec_signal(&self, exec: &ExecRef, sig: i32) -> Result<()> {
        self.get_exec(exec).await?.signal(sig).await
    }

    async fn exec_remove(&self, exec: &ExecRef) -> Result<()> {
        self.get(&exec.session)
            .await?
            .remove_exec(&exec.exec)
            .await
    }

    // ---- Daemon ---------------------------------------------------------

    async fn daemon_shutdown(&self) -> Result<()> {
        // `send_replace` records the request even with nothing listening,
        // so a shutdown asked for before the daemon started waiting is
        // still honoured.
        self.shutdown.send_replace(true);
        Ok(())
    }
}
