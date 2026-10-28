// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process lifecycle and in-flight native call admission.

use super::*;

/// Counts calls that must finish before shutdown releases runtime storage.
pub(crate) struct InFlightTracker {
    /// Number of native operations currently holding a token.
    active: Mutex<usize>,
    /// Final shutdown has stopped admitting new native operations.
    closing: AtomicBool,
    /// Wakes shutdown when the final token is dropped.
    idle: Condvar,
    /// Native cleanup failed after the generation was detached for shutdown.
    cleanup_uncertain: AtomicBool,
}

impl InFlightTracker {
    /// Initializes the count and closing flag for in-flight operations.
    pub(crate) fn new() -> Self {
        Self {
            active: Mutex::new(0),
            closing: AtomicBool::new(false),
            idle: Condvar::new(),
            cleanup_uncertain: AtomicBool::new(false),
        }
    }

    /// Adds an in-flight operation unless this generation is closing.
    pub(crate) fn enter(self: &Arc<Self>) -> Option<InFlightToken> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.closing.load(Ordering::Acquire) {
            return None;
        }
        *active = active.checked_add(1)?;
        Some(InFlightToken {
            tracker: self.clone(),
        })
    }

    /// Excludes new native work before shutdown waits for existing tokens.
    pub(crate) fn close(&self) {
        let _active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.closing.store(true, Ordering::Release);
    }

    /// Waits for idle to complete.
    pub(crate) fn wait_idle(&self) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *active != 0 {
            active = self
                .idle
                .wait(active)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Retains the whole generation when an unpublished owner cannot be
    /// returned to its registry after native cleanup fails.
    pub(crate) fn quarantine(&self) {
        self.cleanup_uncertain.store(true, Ordering::Release);
    }

    /// Reports whether uncertain native cleanup prevents reuse of this state.
    pub(crate) fn is_quarantined(&self) -> bool {
        self.cleanup_uncertain.load(Ordering::Acquire)
    }
}

/// One registered operation that must finish before runtime teardown.
pub(crate) struct InFlightToken {
    /// Counter decremented when this operation leaves its native work.
    tracker: Arc<InFlightTracker>,
}

impl Drop for InFlightToken {
    /// Decrements the active count and wakes teardown when it reaches zero.
    fn drop(&mut self) {
        let mut active = self
            .tracker
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *active -= 1;
        if *active == 0 {
            self.tracker.idle.notify_all();
        }
    }
}

/// Keeps an unlocked native operation in one HSA generation until its owner
/// is either published back into that generation or safely retired.
pub(crate) struct InFlightGeneration {
    /// Lifecycle generation in which this token was acquired.
    generation: u64,
    /// In-flight reference preventing that generation from retiring.
    token: InFlightToken,
}

impl InFlightGeneration {
    /// Enters the active generation and retains its in-flight token.
    pub(crate) fn enter(generation: u64, tracker: &Arc<InFlightTracker>) -> Result<Self, Status> {
        let token = tracker.enter().ok_or(OUT_OF_RESOURCES)?;
        Ok(Self { generation, token })
    }

    /// Checks whether this operation still belongs to the active generation.
    pub(crate) fn is_current(&self, lifecycle: &HsaLifecycle) -> bool {
        lifecycle.active_generation() == Some(self.generation)
    }

    /// Prevents shutdown from dropping dependencies when an unpublished owner
    /// cannot be returned to its original generation.
    pub(crate) fn quarantine(&self) {
        self.token.tracker.quarantine();
    }
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
    /// Native session retained for retry after HSA objects have been retired.
    retired_session: Option<SessionRetirement>,
}

impl HsaLifecycle {
    /// Initializes the unstarted process-global HSA lifecycle.
    pub(super) const fn new() -> Self {
        Self {
            state: None,
            references: 0,
            generation: 0,
            phase: LifecyclePhase::Inactive,
            retired_session: None,
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
                    return Err(0x100c);
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
        PUBLIC_RAW_HANDLES.open();
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
        Ok(())
    }

    /// Cancels an incomplete initialization and advances the lifecycle.
    pub(crate) fn cancel_init(
        &mut self,
        generation: u64,
        cleanup_failed: bool,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.phase = if cleanup_failed {
            LifecyclePhase::Quarantined
        } else {
            LifecyclePhase::Inactive
        };
        PUBLIC_RAW_HANDLES.close();
        self.references = 0;
        Ok(())
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
                self.references = 0;
                self.phase = LifecyclePhase::Preparing(generation);
                Ok(ShutdownWork::Prepare(generation))
            }
            LifecyclePhase::Quarantined => {
                let session = self.retired_session.take().ok_or(INVALID_RUNTIME_STATE)?;
                self.phase = LifecyclePhase::Stopping(self.generation);
                Ok(ShutdownWork::NativeRetry(session, self.generation))
            }
            LifecyclePhase::Inactive => Err(NOT_INITIALIZED),
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Preparing(_)
            | LifecyclePhase::Stopping(_) => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Restores the active generation if tool suspension cannot finish.
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn cancel_shutdown(&mut self, generation: u64) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Preparing(generation) || self.state.is_none() {
            return Err(ERROR);
        }
        self.references = 1;
        self.phase = LifecyclePhase::Active(generation);
        Ok(())
    }

    /// Detaches a prepared generation after all tool setters have quiesced.
    pub(crate) fn finish_prepare_shutdown(&mut self, generation: u64) -> Result<HsaState, Status> {
        if self.phase != LifecyclePhase::Preparing(generation) {
            return Err(ERROR);
        }
        // The caller closed and drained raw-handle entry before detaching
        // the generation. Keep that ordering outside the lifecycle mutex.
        debug_assert!(PUBLIC_RAW_HANDLES.is_closed_and_idle());
        let runtime = self.state.take().ok_or(ERROR)?;
        self.phase = LifecyclePhase::Stopping(generation);
        Ok(runtime)
    }

    /// Completes shutdown after its pending work has retired.
    pub(crate) fn finish_shutdown(
        &mut self,
        generation: u64,
        outcome: &mut ShutdownOutcome,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Stopping(generation) {
            return Err(ERROR);
        }
        // A failed stop can leave native resources alive. Do not create a new
        // runtime generation over their process-wide KFD state.
        self.phase = if outcome.status == SUCCESS {
            LifecyclePhase::Inactive
        } else {
            LifecyclePhase::Quarantined
        };
        if outcome.status != SUCCESS {
            self.retired_session = outcome.pending_session.take();
        }
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
            PUBLIC_RAW_HANDLES.close();
            registry.phase = LifecyclePhase::Quarantined;
        }
    }
}

/// The high bit rejects new raw-handle calls; the remaining bits count calls
/// that may still be reading signal or queue storage during final shutdown.
const RAW_HANDLES_CLOSED: usize = 1 << (usize::BITS - 1);

/// Atomic admission state for raw public operations in one process generation.
struct RawHandleGate {
    /// Closing bit and number of admitted readers.
    state: AtomicUsize,
}

impl RawHandleGate {
    /// Initializes a closed gate with no admitted readers.
    const fn new() -> Self {
        Self {
            state: AtomicUsize::new(RAW_HANDLES_CLOSED),
        }
    }

    /// Opens admission after a fully initialized generation is published.
    fn open(&self) {
        debug_assert!(self.is_closed_and_idle());
        self.state.store(0, Ordering::Release);
    }

    /// Holds one reader through its entire operation if admission is open.
    fn enter(&self) -> Option<RawHandleAdmission<'_>> {
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state < RAW_HANDLES_CLOSED - 1).then_some(state + 1)
            })
            .ok()
            .map(|_| RawHandleAdmission { gate: self })
    }

    /// Rejects new readers while allowing admitted operations to finish.
    fn close(&self) {
        self.state.fetch_or(RAW_HANDLES_CLOSED, Ordering::AcqRel);
    }

    /// Waits for all admitted operations after closing admission.
    fn wait_idle(&self) {
        while !self.is_closed_and_idle() {
            // This runs only at final shutdown. A signal wait observes the
            // closing bit and returns so it cannot hold the gate forever.
            std::thread::yield_now();
        }
    }

    /// Checks that storage can be safely detached.
    fn is_closed_and_idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == RAW_HANDLES_CLOSED
    }
}

/// Keeps public signal and queue storage alive for a complete raw operation.
pub(crate) struct RawHandleAdmission<'a> {
    /// Gate whose reader count this token releases.
    gate: &'a RawHandleGate,
}

impl Drop for RawHandleAdmission<'_> {
    fn drop(&mut self) {
        self.gate.state.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Process gate used by signal and queue index C entry points.
static PUBLIC_RAW_HANDLES: RawHandleGate = RawHandleGate::new();

/// Closes raw entry before the prepared generation can be detached.
pub(crate) fn close_public_raw_handles() {
    PUBLIC_RAW_HANDLES.close();
}

/// Waits without the lifecycle mutex for admitted raw operations to finish.
pub(crate) fn wait_public_raw_handles_idle() {
    PUBLIC_RAW_HANDLES.wait_idle();
}

/// Admits one raw-handle operation for its full callback duration.
pub(crate) fn enter_public_raw_handle() -> Option<RawHandleAdmission<'static>> {
    #[cfg(test)]
    return TEST_RAW_HANDLES.enter();
    #[cfg(not(test))]
    PUBLIC_RAW_HANDLES.enter()
}

#[cfg(test)]
static TEST_RAW_HANDLES: RawHandleGate = RawHandleGate {
    state: AtomicUsize::new(0),
};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod raw_handle_tests {
    use super::*;

    #[test]
    fn shutdown_waits_for_admitted_raw_calls() {
        let gate = Arc::new(RawHandleGate::new());
        gate.open();
        let admitted = gate.enter().unwrap();
        let (closed_tx, closed_rx) = std::sync::mpsc::channel();
        let (idle_tx, idle_rx) = std::sync::mpsc::channel();
        let shutting_down = gate.clone();
        let worker = std::thread::spawn(move || {
            shutting_down.close();
            closed_tx.send(()).unwrap();
            shutting_down.wait_idle();
            idle_tx.send(()).unwrap();
        });
        closed_rx.recv().unwrap();
        assert!(gate.enter().is_none());
        assert!(idle_rx.try_recv().is_err());
        drop(admitted);
        idle_rx.recv().unwrap();
        worker.join().unwrap();
    }
}

/// Process-global gate owning the active HSA generation.
pub(crate) static LIFECYCLE: Mutex<HsaLifecycle> = Mutex::new(HsaLifecycle::new());
/// Wakes callers waiting for a VM fault or lifecycle transition.
pub(crate) static VM_FAULT_CONDVAR: Condvar = Condvar::new();

/// Locks the process-global lifecycle state, handling poisoned locks.
pub(crate) fn lock() -> Result<MutexGuard<'static, HsaLifecycle>, Status> {
    LIFECYCLE.lock().map_err(|_| ERROR)
}

#[allow(clippy::needless_pass_by_value)]
/// Maps error to the corresponding HSA representation.
pub(crate) fn map_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
        // Busy describes a live native dependency. Only queue-handle
        // validation in the queue adapter may return INVALID_QUEUE.
        rocddi::ErrorKind::Busy => RESOURCE_BUSY,
        _ => ERROR,
    }
}

/// Runs an HSA ABI operation behind the panic-to-status boundary.
pub(crate) fn boundary(operation: impl FnOnce() -> Status) -> Status {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(ERROR)
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
    fn failed_native_shutdown_retains_session_until_retry_succeeds() {
        let session = Session::new(DriverContextLifetime::Session).unwrap();
        let clone = session.clone();
        let mut retirement = session.into_retirement();
        let error = retirement.try_destroy().unwrap_err();
        let mut outcome = ShutdownOutcome {
            status: map_error(error),
            pending_session: Some(retirement),
        };
        let mut lifecycle = HsaLifecycle::new();
        lifecycle.generation = 1;
        lifecycle.phase = LifecyclePhase::Stopping(1);
        lifecycle.finish_shutdown(1, &mut outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Quarantined);

        let ShutdownWork::NativeRetry(retirement, generation) = lifecycle.begin_shutdown().unwrap()
        else {
            panic!("failed session was not offered for retry");
        };
        assert_eq!(generation, 1);
        let mut outcome = ShutdownOutcome::retry(retirement);
        assert_eq!(outcome.status, RESOURCE_BUSY);
        lifecycle.finish_shutdown(generation, &mut outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Quarantined);

        let ShutdownWork::NativeRetry(retirement, generation) = lifecycle.begin_shutdown().unwrap()
        else {
            panic!("failed session was not kept for another retry");
        };
        drop(clone);
        let mut outcome = ShutdownOutcome::retry(retirement);
        assert_eq!(outcome.status, SUCCESS);
        lifecycle.finish_shutdown(generation, &mut outcome).unwrap();
        assert_eq!(lifecycle.phase, LifecyclePhase::Inactive);
        assert_eq!(lifecycle.begin_init().unwrap(), Some(2));
    }
}
