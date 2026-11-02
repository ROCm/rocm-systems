// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process lifecycle and in-flight native call admission.

use super::*;

/// Sticky uncertainty shared by native work in one HSA generation.
pub(crate) struct GenerationHealth {
    /// Any cleanup result that could leave native users of frontend owners.
    uncertain: AtomicBool,
}

impl GenerationHealth {
    /// Starts a generation without uncertain native owners.
    pub(crate) fn new() -> Self {
        Self {
            uncertain: AtomicBool::new(false),
        }
    }

    /// Retains the generation when native retirement cannot be proved.
    pub(crate) fn quarantine(&self) {
        self.uncertain.store(true, Ordering::Release);
    }

    /// Reports whether any participant retained an uncertain owner.
    pub(crate) fn is_quarantined(&self) -> bool {
        self.uncertain.load(Ordering::Acquire)
    }
}

thread_local! {
    static LIFECYCLE_ENTRY: Cell<bool> = const { Cell::new(false) };
}

/// Rejects recursive public lifecycle calls from tool callbacks while a
/// transition runs on this thread.
pub(crate) struct LifecycleCallScope {
    /// Whether this scope set and must clear the thread-local entry flag.
    owns_flag: bool,
}

impl LifecycleCallScope {
    /// Enters one public lifecycle call, rejecting same-thread recursion.
    pub(crate) fn enter() -> Result<Self, Status> {
        LIFECYCLE_ENTRY.with(|active| {
            if active.get() {
                Err(INVALID_RUNTIME_STATE)
            } else {
                active.set(true);
                Ok(Self { owns_flag: true })
            }
        })
    }

    /// A shutdown helper inherits the public call's transition when it runs
    /// on the same thread and establishes one when dispatched elsewhere.
    pub(crate) fn enter_helper() -> Self {
        let owns_flag = LIFECYCLE_ENTRY.with(|active| {
            if active.get() {
                false
            } else {
                active.set(true);
                true
            }
        });
        Self { owns_flag }
    }
}

impl Drop for LifecycleCallScope {
    fn drop(&mut self) {
        if self.owns_flag {
            LIFECYCLE_ENTRY.with(|active| active.set(false));
        }
    }
}

/// Keeps an unlocked native operation in one HSA generation until its owner
/// is either published back into that generation or safely retired.
pub(crate) struct InFlightGeneration {
    /// Lifecycle generation in which this token was acquired.
    generation: u64,
    /// In-flight reference preventing that generation from retiring.
    _token: InFlightToken,
    /// Generation owner retained if native cleanup becomes uncertain.
    health: Arc<GenerationHealth>,
}

impl InFlightGeneration {
    /// Enters the active generation and retains its in-flight token.
    pub(crate) fn enter(
        generation: u64,
        tracker: &Arc<InFlightTracker>,
        health: &Arc<GenerationHealth>,
    ) -> Result<Self, Status> {
        let token = tracker.enter().ok_or(OUT_OF_RESOURCES)?;
        Ok(Self {
            generation,
            _token: token,
            health: health.clone(),
        })
    }

    /// Checks whether this operation still belongs to the active generation.
    pub(crate) fn is_current(&self, lifecycle: &HsaLifecycle) -> bool {
        lifecycle.active_generation() == Some(self.generation)
    }

    /// Prevents shutdown from dropping dependencies when an unpublished owner
    /// cannot be returned to its original generation.
    pub(crate) fn quarantine(&self) {
        self.health.quarantine();
    }
}
/// Retains an owner unless native cleanup has proved it safe to drop.
pub(crate) struct RetainOnDrop<T> {
    /// Deliberately retained on error or unwind.
    value: std::mem::ManuallyDrop<T>,
}

impl<T> RetainOnDrop<T> {
    /// Takes responsibility for retaining an owner through uncertain cleanup.
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: std::mem::ManuallyDrop::new(value),
        }
    }

    /// Releases the guard only after cleanup proves the owner may be dropped.
    pub(crate) fn into_inner(self) -> T {
        std::mem::ManuallyDrop::into_inner(self.value)
    }
}

impl<T> std::ops::Deref for RetainOnDrop<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for RetainOnDrop<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// Owner retained when an HSA generation cannot be safely recreated.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing HSA state would allocate on an uncertain cleanup path"
)]
enum Quarantine {
    /// HSA objects retired; only the native session still needs cleanup.
    NativeRetry(SessionRetirement),
    /// An uncertain frontend cleanup retains the entire HSA generation.
    RetainedGeneration {
        /// Owner intentionally held without an automatic destructor.
        _state: RetainOnDrop<HsaState>,
    },
    /// Rollback or unwind retained an owner outside the published registry.
    Unrecoverable,
}

/// Process registry phase and generation during initialization or teardown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LifecyclePhase {
    /// No HSA runtime is published.
    Inactive,
    /// A new generation is acquiring native resources.
    Starting(u64),
    /// The generation is invoking tool initialization callbacks.
    LoadingTools(u64),
    /// The generation is available to HSA callers.
    Active(u64),
    /// Final shutdown is waiting for registrar setters with state still live.
    Preparing(u64),
    /// Final shutdown is retiring the generation.
    Stopping(u64),
    /// Failed native cleanup prevents a safe new generation.
    Quarantined,
}

/// Work detached from the registry for one shutdown call.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing HSA state would allocate while shutting down a runtime"
)]
pub(crate) enum ShutdownWork {
    /// An initialization reference was released; the generation remains live.
    Reference,
    /// The final reference must first quiesce registered tools.
    Prepare(u64),
    /// An earlier native shutdown failed after HSA state was retired.
    NativeRetry(SessionRetirement, u64),
}

/// Process-wide gate for HSA generation transitions.
pub(crate) struct HsaLifecycle {
    /// Published HSA adapter state for the active generation.
    state: Option<HsaState>,
    /// Successful `hsa_init` calls awaiting shutdown for the active generation.
    references: u32,
    /// Last generation number assigned by initialization.
    generation: u64,
    /// Current transition phase under the registry mutex.
    phase: LifecyclePhase,
    /// Distinguishes native retry from terminal retained ownership.
    quarantine: Option<Quarantine>,
}

impl HsaLifecycle {
    /// Initializes the unstarted process-global HSA lifecycle.
    pub(super) const fn new() -> Self {
        Self {
            state: None,
            references: 0,
            generation: 0,
            phase: LifecyclePhase::Inactive,
            quarantine: None,
        }
    }

    /// Returns this value as ref.
    pub(crate) fn as_ref(&self) -> Option<&HsaState> {
        match self.phase {
            LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Active(_)
            | LifecyclePhase::Preparing(_) => self.state.as_ref(),
            _ => None,
        }
    }

    /// Returns this value as mut.
    pub(crate) fn as_mut(&mut self) -> Option<&mut HsaState> {
        match self.phase {
            LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Active(_)
            | LifecyclePhase::Preparing(_) => self.state.as_mut(),
            _ => None,
        }
    }

    /// Returns the generation of the currently published runtime.
    pub(crate) fn active_generation(&self) -> Option<u64> {
        match (self.phase, self.state.as_ref()) {
            (
                LifecyclePhase::LoadingTools(generation)
                | LifecyclePhase::Active(generation)
                | LifecyclePhase::Preparing(generation),
                Some(_),
            ) => Some(generation),
            _ => None,
        }
    }

    /// Returns None after adding a reference to an already active runtime.
    pub(crate) fn begin_init(&mut self) -> Result<Option<u64>, Status> {
        match self.phase {
            LifecyclePhase::Active(_) => {
                if self.state.is_none() {
                    return Err(ERROR);
                }
                if self.references == i32::MAX as u32 {
                    return Err(REFCOUNT_OVERFLOW);
                }
                self.references += 1;
                Ok(None)
            }
            LifecyclePhase::Inactive => {
                self.generation = self.generation.checked_add(1).ok_or(ERROR)?;
                self.phase = LifecyclePhase::Starting(self.generation);
                Ok(Some(self.generation))
            }
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Preparing(_)
            | LifecyclePhase::Stopping(_)
            | LifecyclePhase::Quarantined => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Reports a transition that another `hsa_init` caller must await.
    fn init_transition_in_progress(&self) -> bool {
        matches!(
            self.phase,
            LifecyclePhase::Starting(_)
                | LifecyclePhase::LoadingTools(_)
                | LifecyclePhase::Preparing(_)
                | LifecyclePhase::Stopping(_)
        )
    }

    /// Publishes fully initialized HSA state for the active generation.
    pub(crate) fn publish_init(
        &mut self,
        generation: u64,
        runtime: HsaState,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.state = Some(runtime);
        self.references = 1;
        process_admission().open(generation);
        // HSA queries may run from a tool's OnLoad, but recursive init and
        // shutdown must remain rejected until the tool table is published.
        self.phase = LifecyclePhase::LoadingTools(generation);
        Ok(())
    }

    /// Completes init after its pending work has retired.
    pub(crate) fn finish_init(&mut self, generation: u64) -> Result<(), Status> {
        if self.phase != LifecyclePhase::LoadingTools(generation) || self.state.is_none() {
            return Err(ERROR);
        }
        self.phase = LifecyclePhase::Active(generation);
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }

    /// Cancels an incomplete initialization and advances the lifecycle.
    pub(crate) fn cancel_init(
        &mut self,
        generation: u64,
        cleanup_failed: bool,
    ) -> Result<(), Status> {
        self.cancel_init_with_notification(generation, cleanup_failed, &LIFECYCLE_CONDVAR)
    }

    /// Cancels initialization and wakes callers waiting on this lifecycle.
    pub(super) fn cancel_init_with_notification(
        &mut self,
        generation: u64,
        cleanup_failed: bool,
        changed: &Condvar,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.phase = if cleanup_failed {
            self.quarantine = Some(Quarantine::Unrecoverable);
            LifecyclePhase::Quarantined
        } else {
            LifecyclePhase::Inactive
        };
        process_admission().close();
        self.references = 0;
        changed.notify_all();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn prepare_without_runtime_for_test(&mut self, generation: u64) {
        assert_eq!(self.phase, LifecyclePhase::Starting(generation));
        self.phase = LifecyclePhase::Preparing(generation);
    }

    /// Reports whether final shutdown must prepare this active generation.
    pub(crate) fn shutdown_will_prepare(&self) -> Result<bool, Status> {
        match self.phase {
            LifecyclePhase::Active(_) if self.state.is_some() => Ok(self.references == 1),
            LifecyclePhase::Active(_) => Err(ERROR),
            _ => Ok(false),
        }
    }

    /// Detaches final teardown work, or releases one of several references.
    pub(crate) fn begin_shutdown(&mut self) -> Result<ShutdownWork, Status> {
        match self.phase {
            LifecyclePhase::Active(generation) => {
                if self.state.is_none() {
                    return Err(ERROR);
                }
                if self.references > 1 {
                    self.references -= 1;
                    return Ok(ShutdownWork::Reference);
                }
                // This is the single public admission boundary for final
                // shutdown. Existing leases finish before state is detached.
                let runtime = self.state.as_ref().ok_or(ERROR)?;
                runtime.request_stop();
                self.references = 0;
                self.phase = LifecyclePhase::Preparing(generation);
                Ok(ShutdownWork::Prepare(generation))
            }
            LifecyclePhase::Quarantined => match self.quarantine.take() {
                Some(Quarantine::NativeRetry(session)) => {
                    self.phase = LifecyclePhase::Stopping(self.generation);
                    Ok(ShutdownWork::NativeRetry(session, self.generation))
                }
                other => {
                    self.quarantine = other;
                    Err(INVALID_RUNTIME_STATE)
                }
            },
            LifecyclePhase::Inactive => Err(NOT_INITIALIZED),
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Preparing(_)
            | LifecyclePhase::Stopping(_) => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Detaches a prepared generation after all tool setters have quiesced.
    pub(crate) fn finish_prepare_shutdown(&mut self, generation: u64) -> Result<HsaState, Status> {
        if self.phase != LifecyclePhase::Preparing(generation) {
            return Err(ERROR);
        }
        // The caller closed and drained process admission before detaching
        // the generation. Keep that wait outside the lifecycle mutex.
        debug_assert!(process_admission().is_closed_and_idle());
        let runtime = self.state.take().ok_or(ERROR)?;
        self.phase = LifecyclePhase::Stopping(generation);
        Ok(runtime)
    }

    /// Completes shutdown after its pending work has retired.
    pub(crate) fn finish_shutdown(
        &mut self,
        generation: u64,
        outcome: ShutdownOutcome,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Stopping(generation) {
            return Err(ERROR);
        }
        // A failed stop can leave native resources alive. Do not create a new
        // runtime generation over their process-wide KFD state.
        self.quarantine = match outcome {
            ShutdownOutcome::Complete => None,
            ShutdownOutcome::NativeRetry { session, .. } => Some(Quarantine::NativeRetry(session)),
            ShutdownOutcome::RetainedGeneration { state, .. } => {
                Some(Quarantine::RetainedGeneration { _state: state })
            }
        };
        self.phase = if self.quarantine.is_some() {
            LifecyclePhase::Quarantined
        } else {
            LifecyclePhase::Inactive
        };
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }
}

/// Quarantines a generation if initialization or shutdown unwinds.
pub(crate) struct LifecycleTransition {
    /// Generation phase that will be quarantined if this guard unwinds.
    phase: LifecyclePhase,
    /// Cleared once the transition finishes successfully.
    armed: bool,
}

impl LifecycleTransition {
    /// Constructs a transition guard for initialization.
    pub(crate) fn starting(generation: u64) -> Self {
        Self {
            phase: LifecyclePhase::Starting(generation),
            armed: true,
        }
    }

    /// Constructs a transition guard for shutdown.
    pub(crate) fn stopping(generation: u64) -> Self {
        Self {
            phase: LifecyclePhase::Stopping(generation),
            armed: true,
        }
    }

    /// Guards the preparation phase before native state is detached.
    pub(crate) fn preparing(generation: u64) -> Self {
        Self {
            phase: LifecyclePhase::Preparing(generation),
            armed: true,
        }
    }

    /// Advances an armed shutdown guard after the state is detached.
    pub(crate) fn advance_to_stopping(&mut self, generation: u64) {
        self.phase = LifecyclePhase::Stopping(generation);
    }

    /// Prevents a completed transition guard from rolling back state.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LifecycleTransition {
    /// Rolls back an incomplete generation transition.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut registry = LIFECYCLE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if registry.phase == self.phase
            || matches!(
                (self.phase, registry.phase),
                (LifecyclePhase::Starting(expected), LifecyclePhase::LoadingTools(current))
                    if expected == current
            )
        {
            // Some handle APIs intentionally bypass the registry mutex. Fence
            // those entry points when a failed transition retains live state.
            process_admission().close();
            registry.quarantine = Some(match registry.state.take() {
                Some(state) => Quarantine::RetainedGeneration {
                    _state: RetainOnDrop::new(state),
                },
                None => Quarantine::Unrecoverable,
            });
            registry.phase = LifecyclePhase::Quarantined;
            LIFECYCLE_CONDVAR.notify_all();
        }
    }
}

/// Process-global gate owning the active HSA generation.
pub(crate) static LIFECYCLE: Mutex<HsaLifecycle> = Mutex::new(HsaLifecycle::new());
/// Wakes concurrent init callers after an initialization or final shutdown.
pub(crate) static LIFECYCLE_CONDVAR: Condvar = Condvar::new();

/// Locks the process-global lifecycle state, handling poisoned locks.
pub(crate) fn lock() -> Result<MutexGuard<'static, HsaLifecycle>, Status> {
    LIFECYCLE.lock().map_err(|_| ERROR)
}

/// Waits for another thread's lifecycle transition before adding an init
/// reference or starting a new generation.
pub(crate) fn begin_init_waiting(
    mut guard: MutexGuard<'_, HsaLifecycle>,
    changed: &Condvar,
) -> Result<Option<u64>, Status> {
    while guard.init_transition_in_progress() {
        // A deferred final shutdown must join callbacks and workers before
        // it can leave Preparing. A borrowed logging write may also hold the
        // config mutex that shutdown needs, so none of these callers can wait.
        if ApiLeaseScope::current()
            || CallbackScope::active()
            || LogWriteScope::active()
            || guard.as_ref().is_some_and(HsaState::running_on_worker)
        {
            return Err(INVALID_RUNTIME_STATE);
        }
        guard = changed.wait(guard).map_err(|_| ERROR)?;
    }
    guard.begin_init()
}

/// Returns mutable initialized state or an HSA lifecycle error.
pub(crate) fn initialized_mut(runtime: &mut HsaLifecycle) -> Result<&mut HsaState, Status> {
    runtime.as_mut().ok_or(NOT_INITIALIZED)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn detached_retirement_keeps_owners_on_unwind() {
        struct Owner(Arc<AtomicUsize>);

        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let owner = Owner(drops.clone());
        let result = std::panic::catch_unwind(move || {
            let retained = RetainOnDrop::new(owner);
            assert_eq!(retained.0.load(Ordering::Relaxed), 0);
            panic!("injected retirement unwind");
        });
        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        let clean = RetainOnDrop::new(Owner(drops.clone()));
        drop(clean.into_inner());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_native_shutdown_retains_session_until_retry_succeeds() {
        let session = Session::new(DriverContextLifetime::Session).unwrap();
        let clone = session.clone();
        let mut retirement = session.into_retirement();
        let error = retirement.try_destroy().unwrap_err();
        let outcome = ShutdownOutcome::NativeRetry {
            status: map_error(error),
            session: retirement,
        };
        let mut lifecycle = HsaLifecycle::new();
        lifecycle.generation = 1;
        lifecycle.phase = LifecyclePhase::Stopping(1);
        lifecycle.finish_shutdown(1, outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Quarantined);
        assert!(matches!(
            lifecycle.quarantine.as_ref(),
            Some(Quarantine::NativeRetry(_))
        ));

        let ShutdownWork::NativeRetry(retirement, generation) = lifecycle.begin_shutdown().unwrap()
        else {
            panic!("failed session was not offered for retry");
        };
        assert_eq!(generation, 1);
        let outcome = ShutdownOutcome::retry(retirement);
        assert_eq!(outcome.status(), RESOURCE_BUSY);
        lifecycle.finish_shutdown(generation, outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Quarantined);
        assert!(matches!(
            lifecycle.quarantine.as_ref(),
            Some(Quarantine::NativeRetry(_))
        ));

        let ShutdownWork::NativeRetry(retirement, generation) = lifecycle.begin_shutdown().unwrap()
        else {
            panic!("failed session was not kept for another retry");
        };
        drop(clone);
        let outcome = ShutdownOutcome::retry(retirement);
        assert_eq!(outcome.status(), SUCCESS);
        lifecycle.finish_shutdown(generation, outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Inactive);
        assert!(lifecycle.quarantine.is_none());
        assert_eq!(lifecycle.begin_init().unwrap(), Some(2));
    }
}
