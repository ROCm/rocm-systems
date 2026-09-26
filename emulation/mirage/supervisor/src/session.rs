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
///
/// Shared with [`crate::spec`] rather than declared twice: this is the
/// mount target, and `spec` builds the in-container pid-file path *under*
/// it that each workload's wrapper writes to. Two copies that drifted
/// would leave the wrapper redirecting into a path that is not a mount —
/// a write that fails silently, and every containerised signal then
/// reaching nothing.
use crate::spec::CONTAINER_RUNTIME_DIR;

/// Foreground process of a node container.
///
/// The container needs a PID 1 that simply stays alive: with the host
/// process gone, workloads are started from outside with `provider exec`,
/// so the container's own entrypoint has no work to do. An earlier design
/// ran `mirage host --rank N` here, which meant every container had a
/// second mirage process inside it polling a bind-mounted directory.
///
/// Taken from `mirage_container`, which is what actually launches the
/// containers, so the plan and the launch cannot disagree.
use mirage_container::CONTAINER_IDLE_COMMAND;

/// A live session.
#[derive(Debug)]
pub struct Session {
    /// The definition this session was created from.
    pub def: SessionDef,
    /// Resolved profile, emulator context and scratch directory.
    pub ctx: SessionContext,
    /// How many nodes this session runs on, resolved once here.
    ///
    /// A session's shape is fixed when it is created, and this field is
    /// what fixes it. It used to be re-read from disk on every
    /// [`Session::describe`] — once per exec and once per `Describe` or
    /// `Attach` — so editing a by-name topology while a run was live
    /// reshaped that live session underneath it: a one-node run started
    /// fanning out to three ranks, with three different `WORLD_SIZE`s in
    /// play, and containers for two of the nodes that were never brought
    /// up.
    node_count: u32,
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
    /// Set once [`Session::teardown`] has run to completion.
    ///
    /// A second caller has to *wait* for the first rather than return, so
    /// the field is a level-triggered `watch` and not just the
    /// `tearing_down` flag: see the head of `teardown` for what returning
    /// early costs.
    torn_down: watch::Sender<bool>,
    /// How many `mirage exec` clients are currently borrowing this
    /// session. See [`Session::attach`].
    ///
    /// A `watch` so a waiter can be woken rather than poll, and so
    /// [`Session::borrowers`] can be read without taking `inner` — it is
    /// consulted from the control socket's accept path, which must not
    /// contend with process bookkeeping.
    leases: watch::Sender<usize>,
    /// Who holds each live lease, by the entry number the lease was
    /// given.
    ///
    /// The only question ever asked of this is "is this leftover process
    /// one a *live* borrower is still using?" — see
    /// [`Session::reap_departed_borrowers`] — and it is answered two
    /// ways, because neither alone is enough. Either handle on its own is
    /// worth recording, so a borrower is listed when it has one: a pid
    /// with no exec id still spares everything descended from it, and an
    /// exec id with no pid still spares everything carrying the mark.
    /// Only a borrower with *neither* — no credentials from the kernel
    /// and no exec id from the client — is absent, because there is
    /// nothing about it to record that would not be a placeholder, and a
    /// placeholder would either protect nothing or protect everything.
    borrowers: Arc<Mutex<BTreeMap<u64, Borrower>>>,
    /// Set when teardown wants its borrowers to let go.
    ///
    /// Published rather than inferred from `tearing_down` because the
    /// party that needs it is a per-connection task on the control
    /// socket, which has no other reason to look at `inner` at all.
    closing: watch::Sender<bool>,
    /// `true` while a bring-up may still be creating things this session
    /// does not know about.
    ///
    /// Bring-up runs in a task nobody holds a handle to, and most of its
    /// work happens inside blocking calls that cannot be cancelled — a
    /// `podman run` creating containers and a network. Teardown therefore
    /// waits for it (see [`Session::await_bring_up`]) rather than racing
    /// it: the alternative, and what used to happen, is a Ctrl-C during
    /// bring-up returning promptly and the runtime shutting down on top
    /// of a half-built session, cancelling the very rollback that would
    /// have removed it.
    bringing_up: watch::Sender<bool>,
    /// The switch that ends whatever provider the bring-up is waiting on.
    ///
    /// The companion to `bringing_up`, and the reason waiting on it is
    /// affordable. The wait has to be unbounded — see
    /// [`Session::await_bring_up`] — but "unbounded" only has to mean
    /// "as long as the rollback needs", not "as long as a registry
    /// takes". Teardown flips this immediately before it waits, so the
    /// `podman pull` in flight is killed, bring-up fails at its next
    /// step, and the wait ends in a poll interval rather than in
    /// minutes. Handed to the engine where the engine is built, in
    /// [`Run::bring_up_containers`](crate::Run).
    cancel: mirage_container::Cancel,
}

/// A `mirage exec` client's claim on a session.
///
/// Held for as long as the borrowed workload runs, and released by being
/// dropped — which is what makes it survive the borrower crashing: on the
/// server side the lease is owned by the task serving that client's
/// socket, and the kernel ends that connection however the client ends.
///
/// Deliberately not `Clone`: one connection, one claim.
#[derive(Debug)]
pub struct SessionLease {
    leases: watch::Sender<usize>,
    /// The register this lease's borrower is listed in, and its entry.
    ///
    /// Shared with the session rather than reached through it, so that
    /// dropping the lease needs nothing of the session but this: a lease
    /// is held by a per-connection task, and giving that task an `Arc` to
    /// the whole session to satisfy a `Drop` would make the session's
    /// lifetime depend on its clients'.
    borrowers: Arc<Mutex<BTreeMap<u64, Borrower>>>,
    entry: u64,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        lock(&self.borrowers).remove(&self.entry);
        self.leases.send_modify(|n| *n = n.saturating_sub(1));
    }
}

#[derive(Debug, Default)]
struct Inner {
    /// Execs by id, live and finished alike.
    execs: BTreeMap<ExecId, Arc<Exec>>,
    /// Counter for the next exec id.
    next_exec: u32,
    /// Counter for the next lease entry. Entries are numbered rather
    /// than keyed by pid: two `mirage exec`s could in principle come
    /// from one process, and a lease must remove its own claim and not
    /// somebody else's.
    next_lease: u64,
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
    /// Processes per node in the job this session was created to run.
    ///
    /// Declared by the owning run before it serves its socket (see
    /// [`Session::set_job_shape`]) and reported in every
    /// [`Session::describe`], because it is part of the session's shape
    /// rather than of one command: every rank variable is numbered in
    /// this grid, and a `mirage exec` that had to guess it handed its
    /// workload a different `WORLD_SIZE` from the one the run's own
    /// ranks have.
    ///
    /// Zero until the run says otherwise, and read as one: a session
    /// nobody declared a shape for runs one process per node, which is
    /// what `mirage run` without `--nproc-per-node` asks for.
    job_nproc: u32,
    /// Rendezvous port for a non-containerised session, chosen on the
    /// first [`Session::describe`] and then fixed.
    ///
    /// It has to be fixed. `describe` is called once per exec *and* once
    /// per `Describe` request off the control socket, so re-picking here
    /// would hand `mirage run`'s ranks one `MASTER_PORT` and a
    /// `mirage exec` in another terminal a different one — and the two
    /// halves of the job would never find each other. A containerised
    /// session already gets this right by recording the port in its
    /// [`ContainerState`] at bring-up.
    head_port: Option<u16>,
}

impl Session {
    /// Register a new session in its `starting` state.
    ///
    /// `runtime_dir` is created here so a backend can rely on it existing.
    ///
    /// The session's node count is resolved here too, once and for all:
    /// see [`Session::node_count`].
    ///
    /// # Errors
    ///
    /// Returns an error if the scratch directory cannot be created, or if
    /// the profile names a topology that does not exist.
    pub fn new(def: SessionDef, profile: ProfileDef) -> Result<Arc<Self>> {
        let runtime_dir = mirage_core::paths::session_runtime_dir(&def.id);
        std::fs::create_dir_all(&runtime_dir)
            .map_err(|e| MirageError::io(runtime_dir.clone(), e))?;
        let node_count = resolve_node_count(&profile)?;
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
            node_count,
            health,
            inner: Mutex::new(Inner::default()),
            quiescent: watch::channel(true).0,
            torn_down: watch::channel(false).0,
            leases: watch::channel(0).0,
            borrowers: Arc::new(Mutex::new(BTreeMap::new())),
            closing: watch::channel(false).0,
            bringing_up: watch::channel(false).0,
            cancel: mirage_container::Cancel::new(),
        }))
    }

    /// The switch teardown uses to end a bring-up that is in flight.
    ///
    /// Handed to the container engine so that a `podman pull` or a
    /// `podman run` is a thing that can be *stopped* rather than only
    /// waited out. See the `cancel` field.
    #[must_use]
    pub fn cancel_switch(&self) -> mirage_container::Cancel {
        self.cancel.clone()
    }

    /// Declare how many processes per node the job in this session runs.
    ///
    /// Called by the owning run before it serves its control socket, so
    /// that no borrower can ever be handed a description of a
    /// differently-shaped job than the one the run's own ranks are in.
    /// Doing it at creation would be better still, and is not possible:
    /// the count is a property of the command `mirage run` was given,
    /// which is not known to [`Session::new`].
    pub fn set_job_shape(&self, nproc_per_node: u32) {
        self.lock().job_nproc = nproc_per_node.max(1);
    }

    /// How many nodes this session runs on.
    ///
    /// Answered from the count resolved at creation, never from the
    /// topology store: a live session's shape must not change because
    /// somebody edited the topology it was created from.
    #[must_use]
    pub fn node_count(&self) -> u32 {
        self.node_count
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

    /// Publish a lifecycle phase, unless it would overwrite an answer.
    ///
    /// `stopped` is marked terminal, because it is: a torn-down session
    /// will never become healthy, and anyone waiting on it has their
    /// answer. Publishing it as non-terminal makes
    /// [`SessionHealth::is_settled`] false for a session that is provably
    /// finished, so a concurrent `mirage session wait` blocks for its
    /// entire timeout and then reports a timeout for a session that was
    /// stopped seconds earlier.
    ///
    /// Two publications are refused here rather than at the call sites,
    /// because a `watch` channel is last-value-wins and both would
    /// silently replace something a waiter is owed.
    ///
    /// A **terminal failure** is the reason bring-up gave, and bring-up
    /// tears the session down itself to roll back what it created — so
    /// `failed` is immediately followed by `stopping` and then `stopped`,
    /// with nothing in between for a waiter to wake on. Whoever lost that
    /// race read `stopped`, waited out the teardown, and reported
    /// `session failed to start (stopped)` with the reason gone; with a
    /// slow containerised teardown it reported a timeout instead.
    ///
    /// A **healthy** phase published after teardown has begun is the
    /// mirror image: it would tell a `mirage exec` in another terminal
    /// that a session whose containers are being removed is open for
    /// business. Bring-up finishing just after a Ctrl-C is exactly that
    /// case, and it is why the check is made under the lock that
    /// `tearing_down` is set under.
    pub fn set_phase(&self, healthy: bool, phase: &str, message: Option<String>) {
        // The lock is held across the publish, not merely across the
        // check: released in between, a `ready` cleared by teardown
        // starting a microsecond later would still reach the channel.
        let inner = self.lock();
        if self.has_failed() || (healthy && inner.tearing_down) {
            return;
        }
        let mut health = SessionHealth::phase(healthy, phase, message);
        health.terminal = phase == state::STOPPED;
        self.set_health(health);
        drop(inner);
    }

    /// Whether bring-up has recorded a terminal failure.
    fn has_failed(&self) -> bool {
        let health = self.health.borrow();
        health.terminal && health.state.as_deref() == Some(state::FAILED)
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
    ///
    /// Returns `None` on success. If teardown has already run nothing is
    /// recorded and the arguments are handed straight back, because the
    /// caller is now the only thing that knows these containers exist.
    ///
    /// Bring-up happens on a blocking thread and can finish *after*
    /// `teardown` has collected what to kill — a `wait_ready` timeout or a
    /// Ctrl-C during a slow pull is enough. Storing the containers then
    /// would hand them to a session nobody will ever tear down again, and
    /// they would outlive the run holding their ports and devices.
    #[must_use = "containers refused by a session that is tearing down must be removed by the caller"]
    pub fn set_containers(
        &self,
        state: ContainerState,
        clients: Vec<NodeClient>,
    ) -> Option<(ContainerState, Vec<NodeClient>)> {
        let mut inner = self.lock();
        if inner.tearing_down {
            return Some((state, clients));
        }
        inner.containers = Some(state);
        inner.container_clients = clients;
        None
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
        // Every client, not `all`. `NodeClient::alive` is a `try_wait`,
        // so asking it is also what *reaps* a provider client that exited
        // on its own — and `Iterator::all` short-circuits on the first
        // `false`, which meant the moment one container died none of the
        // ones after it were ever polled again. Their clients then stayed
        // zombies for the rest of the run, which is the whole thing this
        // method's second job exists to prevent.
        // An explicit loop, not `all`/`fold`: clippy rewrites both back to
        // the short-circuiting form, and its own note for
        // `unnecessary_fold` names the hazard — "`all` is short
        // circuiting and may change the program semantics if the iterator
        // has side effects". This iterator has one, and it is the point.
        let mut alive = true;
        for client in &mut inner.container_clients {
            alive &= client.alive();
        }
        alive
    }

    /// Record the emulator injection to apply to every workload.
    pub fn set_injection(&self, injection: InjectionDef) {
        self.lock().injection = injection;
    }

    /// Record the emulator's daemon handle so teardown can stop it.
    ///
    /// Returns the handle straight back if teardown has already run, for
    /// the same reason [`Session::set_containers`] does: bring-up happens
    /// in a background task that a `wait_ready` timeout or a Ctrl-C does
    /// not wait for, so it can finish starting a daemon *after* teardown
    /// has stopped the one it found (none) and deleted the scratch
    /// directory the daemon's socket lives in. Storing it then would hand
    /// a live emulator to a session nothing will ever tear down again.
    #[must_use = "an emulator daemon refused by a session that is tearing down must be stopped by the caller"]
    pub fn set_emulator_daemon(
        &self,
        daemon: Option<Box<dyn EmulatorDaemon>>,
    ) -> Option<Box<dyn EmulatorDaemon>> {
        let mut inner = self.lock();
        if inner.tearing_down {
            return daemon;
        }
        inner.emulator_daemon = daemon;
        None
    }

    /// Whether teardown has begun.
    #[must_use]
    pub fn is_tearing_down(&self) -> bool {
        self.lock().tearing_down
    }

    /// Declare that a bring-up is about to start, so [`Session::teardown`]
    /// waits for it.
    ///
    /// Called by the owner *before* the bring-up task is spawned: a flag
    /// the task sets for itself would leave a window in which teardown
    /// sees no bring-up and a bring-up is nonetheless about to create
    /// containers.
    pub fn begin_bring_up(&self) {
        self.bringing_up.send_replace(true);
    }

    /// Declare the in-flight bring-up finished.
    ///
    /// "Finished" means it will create nothing further and has cleaned up
    /// anything the session refused to take (see
    /// [`Session::set_containers`]) — not that it succeeded. A bring-up
    /// that fails calls [`Session::teardown`] itself to roll back, so it
    /// must call this *first*: teardown waits here, and a task waiting on
    /// its own completion would wait forever.
    pub fn finish_bring_up(&self) {
        self.bringing_up.send_replace(false);
    }

    /// Resolve once no bring-up is in flight.
    ///
    /// Deliberately unbounded, and cheap anyway. What is being waited for
    /// is a `podman pull` or a `podman run` on a blocking thread, and
    /// every deadline short enough to be worth having is one that expires
    /// mid-pull — at which point the run exits, the runtime drops the
    /// bring-up task, and the containers and the per-session network it
    /// had created are left behind. Networks are not `--rm`; nothing
    /// removes them but mirage.
    ///
    /// What makes the unbounded wait affordable is that the caller ends
    /// the bring-up rather than merely outlasting it: [`Session::teardown`]
    /// flips the session's [`Cancel`](mirage_container::Cancel)
    /// immediately before waiting here, which kills the provider in
    /// flight, so bring-up fails at its next step and rolls back. The
    /// wait is therefore as long as the rollback needs and no longer —
    /// a Ctrl-C during a ten-minute pull comes back in a poll interval,
    /// with nothing left behind.
    ///
    /// A second interrupt still does not shorten it, and deliberately so:
    /// the rollback is what "nothing is left behind" is made of, and the
    /// only way past it is to kill the run outright, which is the case
    /// `mirage cleanup` exists for.
    async fn await_bring_up(&self) {
        let mut rx = self.bringing_up.subscribe();
        // `wait_for` inspects the current value before suspending, so the
        // common case — teardown of a session that came up long ago —
        // returns without yielding.
        let in_flight = *rx.borrow();
        if in_flight {
            tracing::info!(
                session = %self.def.id,
                "waiting for bring-up to finish before tearing the session down"
            );
        }
        let _ = rx.wait_for(|in_flight| !*in_flight).await;
    }

    /// Take a borrower's lease on this session.
    ///
    /// Returns `None` once teardown has begun — the same answer, and for
    /// the same reason, that [`Session::start_exec`] gives: a borrower
    /// admitted now would build its process grid from a description of
    /// containers that are being removed.
    ///
    /// Holding a lease is what a borrower has instead of being visible: a
    /// `mirage exec` starts its processes in *its own* terminal, in its
    /// own process, so the session has no other way to know they exist —
    /// and tearing down under them stops the emulator daemon, runs the
    /// backend's shutdown hook and deletes the scratch directory those
    /// processes are actively using.
    ///
    /// The lease does not itself hold teardown off, and reading it that
    /// way is how the property gets attributed to the wrong layer.
    /// [`Session::teardown`] never looks at the count; what waits is
    /// `mirage run`, which calls [`Session::wait_for_borrowers`] before
    /// it destroys the run — and which deliberately stops waiting when
    /// the user interrupts it. Teardown's own duty to a borrower is to
    /// *tell* it: it publishes `closing`, which drops the connection the
    /// lease is made of.
    ///
    /// `borrower` is the pid of the process taking the lease, where the
    /// connection could say, and `exec` the id it stamped into its
    /// workload's environment, where the client said. Either one is
    /// enough to be listed as a live borrower and neither is required —
    /// see the `borrowers` field.
    #[must_use]
    pub fn attach(&self, borrower: Option<u32>, exec: Option<ExecId>) -> Option<SessionLease> {
        let mut inner = self.lock();
        if inner.tearing_down {
            return None;
        }
        let entry = inner.next_lease;
        inner.next_lease += 1;
        // Recorded on either handle. Requiring the pid meant a lease
        // taken over a connection whose credentials the kernel would not
        // give up was registered as nothing at all, so the exec id it
        // *had* supplied protected nothing and the next borrower to
        // disconnect swept its live workload.
        //
        // Recorded unconditionally, including the borrower with neither
        // handle. Skipping that one saved a map node and cost the map its
        // meaning: the lease count below is incremented for every
        // borrower, so a borrower absent from the map was one `mirage
        // run` waited for and the sweep could not see. One entry per
        // lease, and the two reads below — both `filter_map` — already
        // ignore a handle that is not there.
        lock(&self.borrowers).insert(
            entry,
            Borrower {
                pid: borrower,
                exec,
            },
        );
        // Under the lock, so the increment cannot land between another
        // thread setting `tearing_down` and teardown reading the count.
        self.leases.send_modify(|n| *n += 1);
        drop(inner);
        Some(SessionLease {
            leases: self.leases.clone(),
            borrowers: Arc::clone(&self.borrowers),
            entry,
        })
    }

    /// How many `mirage exec` clients are borrowing this session.
    #[must_use]
    pub fn borrowers(&self) -> usize {
        *self.leases.borrow()
    }

    /// Resolve once no borrower is left.
    ///
    /// Unbounded on purpose. A borrowed workload is a job the user
    /// started and is watching in another terminal, and no timeout mirage
    /// could pick would be right for it. `mirage run` races this against
    /// an interrupt instead, so the user decides when they have waited
    /// long enough.
    pub async fn wait_for_borrowers(&self) {
        let mut rx = self.leases.subscribe();
        // `wait_for` inspects the current value before suspending, so the
        // common case — nobody attached — returns without yielding.
        let _ = rx.wait_for(|n| *n == 0).await;
    }

    /// Resolve once teardown has asked borrowers to let go.
    ///
    /// Awaited by the control socket's per-connection task, which then
    /// closes the connection so the borrower learns the session is going
    /// away rather than finding out when its container disappears.
    pub async fn wait_closing(&self) {
        let mut rx = self.closing.subscribe();
        let _ = rx.wait_for(|closing| *closing).await;
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
    /// down, or if the process grid cannot be built.
    pub async fn start_exec(
        &self,
        def: &ExecDef,
    ) -> Result<(
        Arc<Exec>,
        tokio::sync::mpsc::Receiver<crate::process::OutputChunk>,
    )> {
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

        // Health is a record of how bring-up went, not a probe: nothing
        // re-publishes it if a node container dies afterwards. Asking the
        // containers directly is what turns "the provider says no such
        // container", once per rank, into one error that names the cause —
        // and it is also the only thing that reaps a provider client which
        // exited on its own, rather than leaving it a zombie for the rest
        // of the run.
        if !self.containers_alive() {
            return Err(MirageError::other(format!(
                "session {} is no longer running: a node container has exited",
                self.def.id,
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
        let started = self.spawn_exec(def, &id).await;

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
    async fn spawn_exec(
        &self,
        def: &ExecDef,
        id: &ExecId,
    ) -> Result<(
        Arc<Exec>,
        tokio::sync::mpsc::Receiver<crate::process::OutputChunk>,
    )> {
        // Described here, on the runtime, because it needs `self` and is
        // a few field reads. Everything after it is owned, which is what
        // lets the rest move to a blocking thread.
        let desc = self.describe()?;
        let def = def.clone();
        let id = id.clone();
        // The run process is the one the user is sitting in front of, so
        // its own streams are the ones a workload would be given; see
        // [`crate::spec::CallerStreams`].
        let caller = crate::spec::CallerStreams::probe();

        // On a blocking thread, because both halves of this genuinely
        // block and neither is anything a runtime thread should be doing.
        //
        // `build_specs` runs the container `--workdir` probe, which is a
        // provider round trip -- a `podman exec` into the node container,
        // for as long as that takes. `Exec::start` forks and execs one
        // process per rank, up to `MAX_WORLD_SIZE` of them. Held on the
        // runtime, either one stalls every other task in the process, and
        // the visible cost was the control socket: `mirage run` polls its
        // `serving` future from a `select!`, and a task that blocks in
        // `poll` blocks its siblings -- so for the whole of a slow probe
        // the socket was bound, accepting into the backlog, and answering
        // nobody. A `mirage exec` in another terminal sat there and was
        // eventually told the run "is either still starting up, or
        // shutting down", which it was not.
        tokio::task::spawn_blocking(move || {
            let specs = crate::spec::build_specs(&desc, &def, &id, caller)?;
            Ok(Exec::start(id, def, specs))
        })
        .await
        .map_err(|e| MirageError::other(format!("starting an exec: {e}")))?
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
    /// Every call describes the same session: the shape is the one fixed
    /// at creation, and the rendezvous port is picked once and kept, so a
    /// `mirage exec` in another terminal joins the job rather than
    /// starting a differently-shaped one beside it.
    ///
    /// # Errors
    ///
    /// Returns an error if the session is not ready.
    pub fn describe(&self) -> Result<SessionDescription> {
        // Refuse until bring-up has finished, for the reason spelled out
        // on [`Session::start_exec`]: the injection and the container
        // record are written by bring-up, and a description snapshotted
        // before them names no containers and carries no `LD_PRELOAD`. A
        // `mirage exec` handed one would run its workload directly on the
        // host, unemulated, against whatever GPU is really installed, and
        // exit 0. The control socket is bound before bring-up starts (so
        // this run is visible to `mirage state purge` from the first
        // instant), which makes this the gate that keeps a client from
        // being answered too early.
        let health = self.health();
        if !health.healthy {
            return Err(MirageError::other(format!(
                "session {} is not ready ({})",
                self.def.id,
                health.state.as_deref().unwrap_or("unknown"),
            )));
        }
        let (injection, containers, head_port, job_nproc) = {
            let mut inner = self.lock();
            // Chosen once and remembered, so every caller — this run's own
            // exec and any `mirage exec` in another terminal — rendezvous
            // on the same port.
            let head_port = *inner.head_port.get_or_insert_with(pick_head_port);
            (
                inner.injection.clone(),
                inner.containers.clone(),
                head_port,
                inner.job_nproc.max(1),
            )
        };
        // Fixed at creation, not re-read here: see [`Session::node_count`].
        let node_count = self.node_count;

        let (head_addr, head_port) = match &containers {
            Some(state) => (container_name(&self.def.id, 0), state.head_port),
            None => ("127.0.0.1".to_string(), head_port),
        };

        // The emulator's environment was computed against the host
        // filesystem; inside a container those paths do not exist.
        let (env, ld_preload) = match &containers {
            Some(_) => {
                let mut env = remap_env_for_container(&injection.env, &self.ctx.runtime_dir);
                // `provider exec -e KEY=V` overrides the container's own
                // value for that process, so this has to carry the same
                // `CONTAINER_LIB_DIR` prefix `plan_container` gave the
                // container — see [`container_library_path`]. Without it
                // every exec silently dropped the directory holding the
                // emulator's libraries.
                env.insert(
                    "LD_LIBRARY_PATH".to_string(),
                    container_library_path(&injection.env, &self.ctx.runtime_dir),
                );
                (
                    env,
                    injection
                        .ld_preload
                        .as_ref()
                        .map(|p| libraries_in_container(p).unwrap_or_else(|| p.clone())),
                )
            }
            None => (injection.env.clone(), injection.ld_preload.clone()),
        };

        Ok(SessionDescription {
            session: self.def.id.clone(),
            node_count,
            nproc_per_node: job_nproc,
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
    /// 1. mark the session as tearing down, so no new exec can start and
    ///    no new borrower can attach, and tell any borrower still holding
    ///    a lease that the session is going away;
    /// 2. wait for any bring-up still in flight, so that what is torn
    ///    down is everything the session has, rather than everything it
    ///    had when the user pressed Ctrl-C;
    /// 3. terminate and reap every workload process;
    /// 4. only then stop the emulator daemon — the simulated device has
    ///    to outlive every process that might still be talking to it, or
    ///    a workload gets an I/O error on the way out instead of a clean
    ///    exit;
    /// 5. remove the containers and network;
    /// 6. delete the scratch directory.
    ///
    /// Every step is best-effort past the first: a failure in one must not
    /// strand the ones after it, because those are what release the
    /// resources that actually matter.
    pub async fn teardown(&self) {
        {
            let already = {
                let mut inner = self.lock();
                let already = inner.tearing_down;
                inner.tearing_down = true;
                already
            };
            if already {
                // Another teardown is already in flight. Racing it would
                // break the ordering above — but *returning* breaks
                // something worse. Bring-up tears the session down itself
                // when it fails, and it publishes the terminal health
                // first, so `wait_ready` unblocks and `mirage run` reaches
                // `Run::destroy` while that teardown is still mid-flight.
                // Returning here would let `destroy` report success, `main`
                // return, and the tokio runtime drop — cancelling the
                // unfinished teardown at its next await and leaving the
                // containers, the network and the scratch directory
                // behind. Waiting is what makes `Run::destroy`'s "returns
                // only once all of it has happened" true for the second
                // caller as well as the first.
                let mut done = self.torn_down.subscribe();
                let _ = done.wait_for(|done| *done).await;
                return;
            }
        }

        self.set_phase(false, state::STOPPING, None);

        // Tell the borrowers. `mirage run` normally waits for them to
        // finish before getting here, so usually there are none — but
        // teardown also runs when the user declines to wait, and when
        // bring-up fails, and a borrower that only found out by having
        // its container removed underneath it gets no chance to stop
        // cleanly. Each one's connection is closed on the way out; see
        // [`Session::wait_closing`].
        self.closing.send_replace(true);

        // Wait for a bring-up that is still running.
        //
        // Everything it creates from here on is refused by the session —
        // `tearing_down` is set above — and handed straight back to it to
        // remove, which it does before returning. Waiting is what makes
        // that rollback actually happen: without it `mirage run` returns,
        // `main` returns, the tokio runtime drops, and the task is
        // cancelled at its next await with a network and a container per
        // node already created. That is the leak a Ctrl-C during bring-up
        // used to produce, and `--rm` does not cover it: a container is
        // removed when it stops, a network is removed by nobody.
        //
        // Ended, not merely waited out. The provider running right now is
        // a blocking child process, so the only way to hurry it is to
        // kill it, and this is the switch that does — see
        // [`Session::await_bring_up`]. Flipped *before* the wait rather
        // than after it, which is the whole point: after it there is
        // nothing left to cancel.
        self.cancel.cancel();
        self.await_bring_up().await;

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

        // 3. Every exec, concurrently: with many execs, terminating them
        // in sequence would multiply the SIGTERM grace period by their
        // count and make teardown take minutes.
        let terminations = execs.iter().map(|e| e.terminate());
        futures::future::join_all(terminations).await;
        {
            let mut inner = self.lock();
            inner.execs.clear();
        }

        // And whatever a borrower left in the session, which the step
        // above cannot reach: a `mirage exec` starts its processes in its
        // own process, as its own children, so they are not in any exec
        // of ours to terminate.
        self.reap_leftovers().await;

        // 4. The emulator daemon, now that nothing is talking to it.
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

        // 5. Containers and the per-session network.
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

        // 6. Scratch. Anything the emulator wrote here is dead with the
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
        // Release anyone who arrived while this was running. `send_replace`
        // rather than `send`, for the reason given on `set_health`: the
        // value has to be recorded even with nothing currently waiting.
        self.torn_down.send_replace(true);
        tracing::info!(session = %self.def.id, "session destroyed");
    }

    /// Stop anything still running in this session that no exec owns.
    ///
    /// Normally there is nothing: every process the run started is in an
    /// exec, and every process a borrower started is reaped by the
    /// borrower. The case this exists for is the borrower that cannot do
    /// that — a `mirage exec` that was `SIGKILL`ed. Its workload is
    /// reparented to init, still tagged with this session, still holding
    /// the emulated device; `mirage cleanup` will not touch it, correctly,
    /// because the session is *live*, and that is exactly why the run has
    /// to be the one to reap it. The run owns the session, so nothing may
    /// be left in it when the session goes.
    ///
    /// Found by the tag rather than by bookkeeping, for the reason
    /// [`mirage_core::reclaim`] gives at length: the borrower's own record
    /// of its processes died with the borrower, and the tag did not. The
    /// scan is filtered to *this* session — every other session under
    /// this runtime directory is somebody's live job, and a run tearing
    /// itself down has no business reclaiming those.
    ///
    /// `SIGTERM` first and `SIGKILL` after the same grace period a
    /// workload of our own gets: a borrower that is still alive and
    /// winding down (teardown with a lease still held — the Ctrl-C path)
    /// deserves the chance to exit on its own terms, and a stranded one
    /// has nobody left to care either way.
    ///
    /// The pid alone is signalled, not `kill(-pid)`, for the reason
    /// [`mirage_core::reclaim::reap`] gives: the scan already returns
    /// every descendant that kept the tag, and the group form would
    /// additionally reach processes that had *dropped* it by joining a
    /// mirage-led group — a wider claim than the tag supports.
    async fn reap_leftovers(&self) {
        let id = self.def.id.clone();
        let leftovers: Vec<u32> = tokio::task::spawn_blocking(move || {
            // Nothing is "live" for this purpose: the question is not
            // which sessions still have an owner but which processes are
            // in *this* session, which is about to have none.
            processes_in_session(&id)
        })
        .await
        .unwrap_or_default();
        self.stop_leftovers(leftovers).await;
    }

    /// Stop what a borrower that has just gone away left running, while
    /// the run carries on.
    ///
    /// The same leak as the sweep teardown runs, and the same evidence,
    /// caught at the other end of it. Teardown reaps a dead borrower's
    /// processes because nothing may be left in a session that is going
    /// away; this reaps them because nothing may be left in a session that
    /// is *staying*. Without it, `mirage exec --help`'s promise that the
    /// workload "dies with it" held only for a borrower that outlived the
    /// run — `SIGKILL` a `mirage exec` while its run is still up and its
    /// workload carried on inside the session indefinitely, holding the
    /// emulated device and invisible to `mirage cleanup`, which is right
    /// not to touch a live session.
    ///
    /// # Whose processes these are
    ///
    /// Everything in the session carries the same session and runtime
    /// marks — that is what being in one session means — so the tag pair
    /// finds this run's own live workloads just as readily as a departed
    /// borrower's leftovers, and killing those would be very much worse
    /// than the leak. Two things narrow the set, and both are needed:
    ///
    /// * [`ENV_EXEC`](crate::spec::ENV_EXEC) says which exec started a
    ///   process. Every exec of this run's is in its own map, so a
    ///   process whose exec is one of ours is ours however far it has
    ///   been reparented — a workload that double-forked into the
    ///   background is still the run's business and not a borrower's
    ///   leftover. A process with no exec mark at all is left alone:
    ///   unattributable is not the same as unowned, which is the rule
    ///   [`mirage_core::reclaim`] states for the same reason.
    /// * A borrower still holding a lease owns whatever descends from it.
    ///   Two `mirage exec`s at once is ordinary — one finishing must not
    ///   take the other's workload with it — and the first's departure is
    ///   exactly when this runs.
    ///
    /// What is left after both is a process started by a `mirage exec`
    /// that no longer holds a lease, which is the definition of the leak.
    ///
    /// A containerised borrower is reached only as far as its provider
    /// client, which is the host process the mark is on; the workload
    /// inside the container is the container's until the session's
    /// teardown removes it. That is the same reach teardown's own sweep
    /// has.
    pub async fn reap_departed_borrowers(&self) {
        if self.is_tearing_down() {
            // Teardown is already doing this, and doing more of it: it
            // stops *everything* in the session, ours included, and
            // waits for the result. Racing it would only mean two
            // signals arriving at one process.
            return;
        }
        let id = self.def.id.clone();
        let ours: std::collections::BTreeSet<String> = self
            .lock()
            .execs
            .keys()
            .map(|id| id.as_str().to_string())
            .collect();
        // Both handles on every live borrower; see [`Borrower`] for why
        // neither is sufficient alone.
        let (live_pids, live_execs) = self.live_borrower_handles();

        let leftovers: Vec<u32> = tokio::task::spawn_blocking(move || {
            processes_in_session(&id)
                .into_iter()
                // Not ours, and not a live borrower's. The second half is
                // the one that was missing: a client-side exec id is
                // never in `inner.execs` — the run did not start it and
                // has no record of it — so a live borrower's workload got
                // past this filter and was spared only by the ancestry
                // check below. A workload that had reparented (a
                // `setsid`, a double fork, anything daemonised) failed
                // that too, and the next borrower to disconnect normally
                // sent it `SIGTERM` and then `SIGKILL` while its own
                // borrower was still sitting there holding the lease.
                .filter(|pid| {
                    exec_mark_of(*pid)
                        .is_some_and(|exec| !ours.contains(&exec) && !live_execs.contains(&exec))
                })
                .filter(|pid| !descends_from_any(*pid, &live_pids))
                .collect()
        })
        .await
        .unwrap_or_default();

        // Asked again, against the borrowers there are *now*.
        //
        // The scan above is a full walk of `/proc` reading every
        // candidate's `environ`, and it runs on the blocking pool, so an
        // appreciable time passes between the snapshot the filters used
        // and the answer arriving. A `mirage exec` that attached inside
        // that window is in neither list: its pid is not an ancestor
        // anybody knew about and its exec mark was not in `live_execs`,
        // so its workload — started moments before, very much alive —
        // comes back in `leftovers` and would be sent `SIGTERM` and then
        // `SIGKILL`. Two `mirage exec`s at once is ordinary, and one
        // finishing must not take the other's workload with it.
        //
        // Re-checking cannot be exact — there is no instant at which the
        // set is fixed — but the two mistakes are not equals. Sparing a
        // borrower that left during the scan costs nothing: it is still
        // leftover, and the next disconnect (or teardown, which stops
        // everything) collects it. Killing a live borrower's workload is
        // not recoverable. So the later, narrower answer wins.
        let leftovers = self.still_stranded(leftovers).await;
        self.stop_leftovers(leftovers).await;
    }

    /// Both handles on every borrower holding a lease right now.
    ///
    /// See [`Borrower`]: a borrower may have either, both or neither, and
    /// each handle protects on its own.
    fn live_borrower_handles(&self) -> (Vec<u32>, std::collections::BTreeSet<String>) {
        let borrowers = lock(&self.borrowers);
        (
            borrowers.values().filter_map(|b| b.pid).collect(),
            borrowers
                .values()
                .filter_map(|b| b.exec.as_ref())
                .map(|id| id.as_str().to_string())
                .collect(),
        )
    }

    /// Those of `candidates` that no borrower arriving since the scan has
    /// claimed.
    ///
    /// The same two filters the scan applied, re-applied to the borrowers
    /// of this moment — see the call site for why the answer is taken
    /// twice. Cheap where it matters: `candidates` is what a scan of the
    /// whole session narrowed down to, usually empty and never large,
    /// whereas the scan itself walked every process on the machine.
    async fn still_stranded(&self, candidates: Vec<u32>) -> Vec<u32> {
        if candidates.is_empty() {
            return candidates;
        }
        let ours: std::collections::BTreeSet<String> = self
            .lock()
            .execs
            .keys()
            .map(|id| id.as_str().to_string())
            .collect();
        let (live_pids, live_execs) = self.live_borrower_handles();
        tokio::task::spawn_blocking(move || {
            candidates
                .into_iter()
                .filter(|pid| {
                    exec_mark_of(*pid)
                        .is_some_and(|exec| !ours.contains(&exec) && !live_execs.contains(&exec))
                })
                .filter(|pid| !descends_from_any(*pid, &live_pids))
                .collect()
        })
        .await
        .unwrap_or_default()
    }

    /// `SIGTERM` each of `leftovers`, and `SIGKILL` whatever is still
    /// there after the grace period.
    ///
    /// The same escalation a workload of our own gets: a borrower that is
    /// still alive and winding down deserves the chance to exit on its
    /// own terms, and a stranded one has nobody left to care either way.
    ///
    /// The pid alone is signalled, not `kill(-pid)`, for the reason
    /// [`mirage_core::reclaim::reap`] gives: the scan already returns
    /// every descendant that kept the tag, and the group form would
    /// additionally reach processes that had *dropped* it by joining a
    /// mirage-led group — a wider claim than the tag supports.
    async fn stop_leftovers(&self, leftovers: Vec<u32>) {
        if leftovers.is_empty() {
            return;
        }
        tracing::info!(
            session = %self.def.id,
            "reaping {} process(es) left in the session by a borrower that did not",
            leftovers.len()
        );
        for pid in &leftovers {
            signal_pid(*pid, nix::sys::signal::Signal::SIGTERM);
        }
        let deadline = tokio::time::Instant::now() + crate::process::TERM_GRACE;
        for pid in &leftovers {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if crate::process::wait_gone(*pid, remaining).await {
                continue;
            }
            signal_pid(*pid, nix::sys::signal::Signal::SIGKILL);
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        lock(&self.inner)
    }
}

/// What is known about the holder of one live lease.
///
/// Two independent handles on the same borrower, because a workload can
/// slip out of either one.
///
/// `pid` is the client's, from `SO_PEERCRED`, and it is unforgeable — but
/// it only reaches the workload through ancestry, and a workload that
/// calls `setsid`, double-forks or is otherwise reparented stops
/// descending from it while remaining very much alive.
///
/// `exec` is the id the client stamped into its workload's environment as
/// `MIRAGE_EXEC`. That mark is inherited by everything the workload forks
/// and survives reparenting, which is exactly the case ancestry loses. It
/// is spoofable in the sense that a client names its own id, but the only
/// thing a false one buys is that the sweep spares somebody else's
/// leftovers — and the socket already grants `exec` into the session,
/// which is arbitrary code execution as its owner.
///
/// Both are optional and for the same reason: the two protections are
/// independent, so the absence of one must not withdraw the other. A
/// `getsockopt(SO_PEERCRED)` that fails — or a peer in a namespace this
/// process cannot resolve, which the kernel answers with a pid of `0`
/// that the socket discards rather than believing — leaves a borrower
/// with only its exec id, and that borrower's workload is no less live
/// for it. Both may be absent, and such a borrower is still recorded — one
/// entry per lease, so the map and the lease count cannot disagree about
/// who is here; see [`Session::attach`]. It simply protects nothing,
/// which the `filter_map`s in [`Session::reap_departed_borrowers`]
/// express without needing a special case.
#[derive(Debug, Clone)]
struct Borrower {
    /// The client process holding the lease, when the kernel said.
    pid: Option<u32>,
    /// The exec id its workload carries, when the client said.
    exec: Option<ExecId>,
}

/// Take a mutex, treating a poisoned one as merely locked.
///
/// A panic while one of these is held leaves the data it guards
/// consistent — every critical section in this file is a few field
/// assignments — and refusing to lock afterwards would turn a survivable
/// panic into a teardown that cannot run.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Every process on this machine tagged as belonging to `session`.
///
/// [`mirage_core::reclaim::stranded_workloads`] with no live session
/// excluded, filtered to one session: the question here is never "which
/// sessions still have an owner" — this run is the owner and is asking —
/// but "what is running in mine". Every other session under this runtime
/// directory is somebody's live job.
fn processes_in_session(session: &SessionId) -> Vec<u32> {
    mirage_core::reclaim::stranded_workloads(&[])
        .into_iter()
        .filter(|s| &s.session == session)
        .map(|s| s.pid)
        .collect()
}

/// The exec a process was started by, if it says.
///
/// Read from `/proc/<pid>/environ`, which is the same place and the same
/// evidence [`mirage_core::reclaim`] reads the session mark from, and for
/// the same reason: it is inherited by everything the process forks and
/// it cannot go stale, because it is gone the moment the process is.
///
/// `None` for a process that carries no [`ENV_EXEC`] — one started by a
/// mirage older than the mark, or one that scrubbed its own environment.
/// The caller must read that as "not attributable" rather than as "not
/// mine".
fn exec_mark_of(pid: u32) -> Option<String> {
    // Not `read_to_string`: an environment is arbitrary bytes and need
    // not be UTF-8, and one invalid byte in an unrelated variable must
    // not hide the mark.
    let raw = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let prefix = format!("{}=", crate::spec::ENV_EXEC);
    raw.split(|b| *b == 0)
        .filter_map(|entry| std::str::from_utf8(entry).ok())
        .find_map(|entry| entry.strip_prefix(&prefix))
        .map(str::to_string)
}

/// Whether `pid` is one of `ancestors`, or descends from one of them.
///
/// Walks `/proc/<pid>/stat` upwards. An empty `ancestors` — the ordinary
/// case, one borrower and it has just left — costs nothing at all, which
/// is why the check is written to be asked before the walk rather than
/// during it.
///
/// The walk stops at init, at a pid it cannot read, and at a hard bound
/// on its length: a `/proc` that reported a cycle would otherwise be a
/// loop that never ends, and this runs on a blocking thread pool that
/// teardown later waits on.
fn descends_from_any(pid: u32, ancestors: &[u32]) -> bool {
    const MAX_DEPTH: usize = 64;
    if ancestors.is_empty() {
        return false;
    }
    let mut current = pid;
    for _ in 0..MAX_DEPTH {
        if ancestors.contains(&current) {
            return true;
        }
        match parent_of(current) {
            Some(parent) if parent > 1 => current = parent,
            _ => return false,
        }
    }
    false
}

/// The parent of `pid`, as `/proc` reports it.
///
/// The comm field is parenthesised and may itself contain spaces and
/// brackets, so the fields are counted from after the *last* `)`, which
/// is the parse `proc(5)` documents: state first, then the parent's pid.
fn parent_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let close = stat.rfind(')')?;
    stat.get(close + 1..)?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Send `sig` to one process and to nothing else.
///
/// Non-positive pids are rejected here as they are in
/// [`crate::process::signal_group`], and for the same reason: `kill(0,
/// sig)` addresses the caller's own process group and `kill(-1, sig)`
/// every process the caller may signal, so a stray zero would turn a
/// cleanup into ending the user's session. `read_dir` on `/proc` cannot
/// produce one, which is what makes the guard free.
fn signal_pid(pid: u32, sig: nix::sys::signal::Signal) {
    let Ok(raw) = i32::try_from(pid) else {
        return;
    };
    if raw <= 0 {
        return;
    }
    let _ = nix::sys::signal::kill(nix::unistd::Pid::from_raw(raw), sig);
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
        // Through the store rather than by hand: `profile_get` validates
        // the name before joining it to the config directory, and a name
        // is only a filename by convention — `paths::profile_path` joins
        // whatever it is given, so `../../etc/something` would resolve
        // outside the config root. Today `mirage run` happens to have
        // validated already, but `Run::start` is public API and this is
        // the one place the name becomes a path.
        MaybeRef::Ref(name) => mirage_core::store::profile_get(name),
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

/// The `LD_LIBRARY_PATH` a containerised process must run with.
///
/// [`CONTAINER_LIB_DIR`] first, because that is the only place the
/// emulator's libraries and its interposer exist inside the container,
/// followed by whatever the emulator asked for — remapped, like every
/// other value, so an entry naming the session scratch directory points
/// at its mount rather than at a host path that does not exist there.
///
/// Computed here so that both places that build a container environment
/// agree. They did not: `plan_container` prepended the directory when it
/// launched the container, and `Session::describe` did not, so every
/// `provider exec` passed `-e LD_LIBRARY_PATH=<emulator's value>` and
/// overrode the container's own — deleting the one directory the
/// interposer's sibling libraries resolve from, for every workload.
fn container_library_path(env: &BTreeMap<String, String>, runtime_dir: &std::path::Path) -> String {
    match env.get("LD_LIBRARY_PATH") {
        Some(existing) if !existing.is_empty() => {
            let remapped = to_container_path(existing, runtime_dir, CONTAINER_RUNTIME_DIR);
            format!("{CONTAINER_LIB_DIR}:{remapped}")
        }
        _ => CONTAINER_LIB_DIR.to_string(),
    }
}

/// The in-container path a host library is bind-mounted at.
///
/// `LD_PRELOAD` is a `:`-separated *list*, and at least one backend uses
/// it as one: hotswap preloads its patched ROCR and then its intercept,
/// as `"<dir>/libhsa-runtime64.so:<dir>/libhsa_intercept.so"`. Treating
/// that as a single path made `Path::file_name` return only the last
/// component, so the patched ROCR silently vanished from the preload and
/// the mount built from the same string named a file with a `:` in it —
/// which `-v host:container:ro` cannot even express. Each entry is
/// mapped separately.
fn libraries_in_container(host_paths: &str) -> Option<String> {
    let mapped: Vec<String> = split_library_list(host_paths)
        .filter_map(library_in_container)
        .collect();
    (!mapped.is_empty()).then(|| mapped.join(":"))
}

/// The individual library paths in a `:`-separated search list.
fn split_library_list(value: &str) -> impl Iterator<Item = &str> {
    value.split(':').map(str::trim).filter(|s| !s.is_empty())
}

/// The in-container path one host library is bind-mounted at.
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
    let scratch = ctx.runtime_dir.to_string_lossy().into_owned();
    mounts.push(FileMount {
        host_path: scratch.clone(),
        container_path: CONTAINER_RUNTIME_DIR.to_string(),
        read_only: false,
    });

    // And again at its *host* path.
    //
    // Only environment *values* are remapped onto the mount above
    // (`remap_env_for_container`); nothing rewrites the contents of the
    // files the emulator wrote, and those hold absolute host paths.
    // rocjitsu's `config_path` discovery file is exactly that — it names
    // `<scratch>/rj_config.json` — so an interposer inside the container
    // reads a path that does not exist there, finds no simulation config,
    // and the workload runs unemulated against the real device. The
    // per-node `mirage host` process used to re-resolve the whole
    // injection against the container's own filesystem; with it gone,
    // making the host path resolve in both places is what replaces it.
    if scratch != CONTAINER_RUNTIME_DIR {
        mounts.push(FileMount {
            host_path: scratch.clone(),
            container_path: scratch,
            read_only: false,
        });
    }

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
        // A preload is a `:`-separated list; every entry needs its own
        // mount, or the ones after the first are named in `LD_PRELOAD`
        // and never actually present in the container.
        libraries.extend(split_library_list(preload).map(str::to_string));
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
    let mut env: Vec<(String, String)> = remap_env_for_container(&injection.env, &ctx.runtime_dir)
        .into_iter()
        .collect();

    // `LD_PRELOAD` must name the *in-container* path: the host path does
    // not exist inside the container, and `ld.so` fails the whole process
    // with "cannot be preloaded" rather than skipping it.
    if let Some(preload) = &injection.ld_preload
        && let Some(in_container) = libraries_in_container(preload)
    {
        env.push(("LD_PRELOAD".to_string(), in_container));
    }

    env.push((
        "LD_LIBRARY_PATH".to_string(),
        container_library_path(&injection.env, &ctx.runtime_dir),
    ));
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::emulator::{EmulatorDef, ExecMode};
    use mirage_core::topology::TopologyDef;
    use std::path::PathBuf;

    fn ctx(runtime_dir: PathBuf) -> SessionContext {
        SessionContext {
            id: SessionId::new("s").unwrap(),
            profile: profile(1, 1),
            runtime_dir,
            daemon: false,
        }
    }

    /// A session with a scratch directory under `dir` and nothing
    /// bringing it up, so a test can drive its lifecycle by hand.
    ///
    /// The path override is installed and removed inside this function,
    /// under the shared lock: `TEST_ROOT` is process-wide, and a test
    /// that held it across its `await`s would redirect every other test
    /// in this binary for as long as it ran. Nothing after construction
    /// consults it — the session captured its scratch directory.
    fn a_session(dir: &std::path::Path, profile: ProfileDef) -> Arc<Session> {
        let _guard = mirage_core::paths::test_env_lock();
        mirage_core::paths::set_test_root(dir);
        let def = make_def(
            SessionId::new("lifecycle").unwrap(),
            MaybeRef::Owned(profile.clone()),
            "/".to_string(),
            false,
        );
        let session = Session::new(def, profile).unwrap();
        mirage_core::paths::clear_test_root();
        session
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
    fn a_multi_entry_preload_keeps_every_entry() {
        // hotswap preloads its patched ROCR *and* its intercept, as one
        // `:`-separated value. Treating that as a single path kept only
        // the last component — so the patched ROCR silently vanished from
        // `LD_PRELOAD` and the workload ran against the unpatched runtime
        // while mirage reported success.
        assert_eq!(
            libraries_in_container("/opt/hs/libhsa-runtime64.so:/opt/hs/libhsa_intercept.so"),
            Some(format!(
                "{CONTAINER_LIB_DIR}/libhsa-runtime64.so:{CONTAINER_LIB_DIR}/libhsa_intercept.so"
            ))
        );
        assert_eq!(
            libraries_in_container("/opt/x/libone.so"),
            Some(format!("{CONTAINER_LIB_DIR}/libone.so"))
        );
        assert_eq!(libraries_in_container(""), None);
    }

    #[test]
    fn every_preloaded_library_is_mounted_not_just_the_last() {
        // A path named in `LD_PRELOAD` but never bind-mounted is a file
        // that does not exist inside the container.
        let dir = tempfile::tempdir().unwrap();
        let injection = InjectionDef {
            ld_preload: Some("/opt/hs/libhsa-runtime64.so:/opt/hs/libhsa_intercept.so".to_string()),
            ..Default::default()
        };
        let plan = plan_container(&ctx(dir.path().to_path_buf()), &injection);
        for name in ["libhsa-runtime64.so", "libhsa_intercept.so"] {
            assert!(
                plan.mounts
                    .iter()
                    .any(|m| m.container_path == format!("{CONTAINER_LIB_DIR}/{name}")),
                "{name} is preloaded but never mounted: {:?}",
                plan.mounts
            );
        }
    }

    #[test]
    fn the_container_library_path_always_leads_with_the_mount() {
        // `CONTAINER_LIB_DIR` is the only place the emulator's libraries
        // exist inside the container, so it has to win the search whether
        // or not the backend asked for a path of its own.
        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path();

        let empty = BTreeMap::new();
        assert_eq!(container_library_path(&empty, runtime), CONTAINER_LIB_DIR);

        let mut env = BTreeMap::new();
        env.insert("LD_LIBRARY_PATH".to_string(), "/opt/hs/lib".to_string());
        assert_eq!(
            container_library_path(&env, runtime),
            format!("{CONTAINER_LIB_DIR}:/opt/hs/lib")
        );

        // And a value naming the session scratch is remapped onto its
        // mount, like every other value in the injection.
        let mut env = BTreeMap::new();
        env.insert(
            "LD_LIBRARY_PATH".to_string(),
            runtime.join("lib").display().to_string(),
        );
        assert_eq!(
            container_library_path(&env, runtime),
            format!("{CONTAINER_LIB_DIR}:{CONTAINER_RUNTIME_DIR}/lib")
        );
    }

    #[test]
    fn a_live_sessions_shape_is_the_one_it_was_created_with() {
        // A by-name topology is a file the user can edit, and they do —
        // while a run is up, to set up the next one. `describe` used to
        // re-read it on every call, once per exec and once per
        // `Describe`, so a one-node session started handing out
        // `WORLD_SIZE=3` and fanning execs out to two nodes it had never
        // brought containers up for.
        let dir = tempfile::tempdir().unwrap();
        let _guard = mirage_core::paths::test_env_lock();
        mirage_core::paths::set_test_root(dir.path());

        let mut topology = TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref("MI350X".to_string()),
        };
        mirage_core::topology::store::put("shared", &topology).unwrap();

        let mut p = profile(1, 1);
        p.emulator.topology = MaybeRef::Ref("shared".to_string());
        let def = make_def(
            SessionId::new("shape").unwrap(),
            MaybeRef::Owned(p.clone()),
            "/".to_string(),
            false,
        );
        let session = Session::new(def, p).unwrap();
        session.set_phase(true, state::READY, None);
        let before = session.describe().unwrap().node_count;

        // The user grows the topology for their next run.
        topology.num_nodes = 3;
        mirage_core::topology::store::put("shared", &topology).unwrap();
        let after = session.describe().unwrap().node_count;

        mirage_core::paths::clear_test_root();

        assert_eq!(before, 1);
        assert_eq!(
            after, 1,
            "editing a topology reshaped a session that was already running"
        );
    }

    #[tokio::test]
    async fn teardown_waits_for_a_bring_up_that_is_still_creating_things() {
        // Ctrl-C during bring-up. The bring-up task is not one anybody
        // holds a handle to, and the work inside it is a blocking
        // `podman run` that no cancellation reaches — so a teardown that
        // returned early let `mirage run` exit, the runtime drop the
        // task, and the containers and per-session network it had just
        // created stay behind. `--rm` does not cover that: a network is
        // removed by nobody.
        use std::sync::atomic::{AtomicBool, Ordering};

        let dir = tempfile::tempdir().unwrap();
        let session = a_session(dir.path(), profile(1, 1));
        session.begin_bring_up();

        let rolled_back = Arc::new(AtomicBool::new(false));
        // The stand-in bring-up asserts nothing and answers instead.
        // An assertion inside a spawned task is swallowed — a panic there
        // fails no test — and, worse, it skips the `finish_bring_up()`
        // below it, so the deliberately unbounded `await_bring_up()` in
        // teardown waits forever. A wrong answer has to come back as a
        // hung test rather than as a failing one exactly once before that
        // is worth writing down.
        let refused = Arc::new(AtomicBool::new(false));
        let bringing_up = tokio::spawn({
            let session = session.clone();
            let rolled_back = Arc::clone(&rolled_back);
            let refused = Arc::clone(&refused);
            async move {
                // What an interrupted bring-up does: it finishes creating
                // what it was creating, is handed it straight back by a
                // session that is tearing down, and removes it itself.
                tokio::time::sleep(Duration::from_millis(50)).await;
                let handed_back = session.set_containers(ContainerState::default(), Vec::new());
                refused.store(handed_back.is_some(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                rolled_back.store(true, Ordering::SeqCst);
                session.finish_bring_up();
            }
        });

        // Bounded, so a teardown that does not come back is a failure and
        // not a suite that never finishes. Generously bounded: what is
        // under test is "did it wait at all", which the two sleeps above
        // decide, not how fast it is.
        tokio::time::timeout(Duration::from_secs(30), session.teardown())
            .await
            .expect("teardown must not wait on a bring-up that has finished");
        bringing_up.await.expect("the stand-in bring-up finishes");

        assert!(
            refused.load(Ordering::SeqCst),
            "a session that is tearing down must hand its containers back"
        );
        assert!(
            rolled_back.load(Ordering::SeqCst),
            "teardown returned while bring-up was still rolling back; \
             the runtime shutting down next would have cancelled it"
        );
    }

    #[tokio::test]
    async fn teardown_cancels_the_bring_up_it_is_about_to_wait_for() {
        // The wait is unbounded on purpose, and that is only affordable
        // because teardown ends the bring-up rather than outlasting it.
        // With the switch unwired — which it was, a complete
        // cancellation mechanism in `mirage_container` that no production
        // caller ever constructed — a Ctrl-C during a ten-minute image
        // pull sat in `await_bring_up` for the rest of that pull, and
        // `mirage run` looked like it had ignored the signal.
        let dir = tempfile::tempdir().unwrap();
        let session = a_session(dir.path(), profile(1, 1));
        let cancel = session.cancel_switch();
        assert!(!cancel.is_cancelled(), "nothing has asked it to stop yet");
        session.begin_bring_up();

        // A bring-up that only ends when it is told to, which is what a
        // provider under a `Cancel` does.
        tokio::spawn({
            let session = session.clone();
            let cancel = cancel.clone();
            async move {
                while !cancel.is_cancelled() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                session.finish_bring_up();
            }
        });

        tokio::time::timeout(Duration::from_secs(30), session.teardown())
            .await
            .expect("teardown must end the bring-up it waits for, not wait it out");
        assert!(cancel.is_cancelled());
    }

    #[tokio::test]
    async fn a_terminal_failure_survives_the_teardown_it_triggers() {
        // Bring-up tears the session down itself to roll back what it
        // created, so `failed` is followed at once by `stopping` and
        // `stopped`. A `watch` is last-value-wins, and the reason is the
        // only thing the user can act on.
        let dir = tempfile::tempdir().unwrap();
        let session = a_session(dir.path(), profile(1, 1));

        session.set_health(SessionHealth::failed(
            "pulling image quay.io/nope:1 failed: not found",
        ));
        session.teardown().await;

        let health = session.health();
        assert_eq!(health.state.as_deref(), Some(state::FAILED), "{health:?}");
        assert!(
            health
                .message
                .as_deref()
                .unwrap_or_default()
                .contains("quay.io/nope:1"),
            "the reason bring-up gave was lost: {health:?}"
        );
        assert!(health.is_settled(), "{health:?}");
    }

    #[tokio::test]
    async fn a_session_that_is_tearing_down_never_reports_itself_ready() {
        // The other end of the same race: bring-up finishing an instant
        // after a Ctrl-C. `ready` published then would tell a
        // `mirage exec` in another terminal that a session whose
        // containers are being removed is open for business.
        let dir = tempfile::tempdir().unwrap();
        let session = a_session(dir.path(), profile(1, 1));

        session.teardown().await;
        session.set_phase(true, state::READY, None);

        assert!(!session.health().healthy, "{:?}", session.health());
        assert!(
            session.describe().is_err(),
            "a torn-down session described itself to a borrower"
        );
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
        assert!(
            paths.contains(&"/mnt/mirage/lib/librocjitsu.so"),
            "{paths:?}"
        );
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
