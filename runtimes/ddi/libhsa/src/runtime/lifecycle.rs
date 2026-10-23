// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process lifecycle and in-flight native call admission.

use super::*;

/// Counts calls that must finish before shutdown releases runtime storage.
pub(crate) struct InFlightTracker {
    /// Number of native operations currently holding a token.
    active: Mutex<usize>,
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
        *active = active.checked_add(1)?;
        Some(InFlightToken {
            tracker: self.clone(),
        })
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
    /// Final shutdown is retiring the generation.
    Stopping(u64),
    /// Failed native cleanup prevents a safe new generation.
    Quarantined,
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
}

impl HsaLifecycle {
    /// Initializes the unstarted process-global HSA lifecycle.
    pub(super) const fn new() -> Self {
        Self {
            state: None,
            references: 0,
            generation: 0,
            phase: LifecyclePhase::Inactive,
        }
    }

    /// Returns this value as ref.
    pub(crate) fn as_ref(&self) -> Option<&HsaState> {
        self.state.as_ref()
    }

    /// Returns this value as mut.
    pub(crate) fn as_mut(&mut self) -> Option<&mut HsaState> {
        self.state.as_mut()
    }

    /// Returns the generation of the currently published runtime.
    pub(crate) fn active_generation(&self) -> Option<u64> {
        match (self.phase, self.state.as_ref()) {
            (
                LifecyclePhase::LoadingTools(generation) | LifecyclePhase::Active(generation),
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
        self.references = 0;
        Ok(())
    }

    /// Returns None after releasing one of several active references.
    pub(crate) fn begin_shutdown(&mut self) -> Result<Option<(HsaState, u64)>, Status> {
        match self.phase {
            LifecyclePhase::Active(generation) => {
                if self.state.is_none() {
                    return Err(ERROR);
                }
                if self.references > 1 {
                    self.references -= 1;
                    return Ok(None);
                }
                let runtime = self.state.take().ok_or(ERROR)?;
                self.references = 0;
                self.phase = LifecyclePhase::Stopping(generation);
                Ok(Some((runtime, generation)))
            }
            LifecyclePhase::Inactive => Err(NOT_INITIALIZED),
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Stopping(_)
            | LifecyclePhase::Quarantined => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Completes shutdown after its pending work has retired.
    pub(crate) fn finish_shutdown(
        &mut self,
        generation: u64,
        status: Status,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Stopping(generation) {
            return Err(ERROR);
        }
        // A failed stop can leave native resources alive. Do not create a new
        // runtime generation over their process-wide KFD state.
        self.phase = if status == SUCCESS {
            LifecyclePhase::Inactive
        } else {
            LifecyclePhase::Quarantined
        };
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
            registry.phase = LifecyclePhase::Quarantined;
        }
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
