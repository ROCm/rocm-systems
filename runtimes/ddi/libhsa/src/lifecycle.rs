// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process lifecycle and in-flight native call admission.

#[cfg(all(hsa_tool_dispatch, not(test)))]
use crate::admission;
use crate::admission::{ApiLeaseScope, InFlightToken, InFlightTracker, process_admission};
use crate::events::CallbackScope;
use crate::ffi::*;
use crate::logging::LogWriteScope;
use crate::session::{HsaSession, InitRollback, SessionHealth, ShutdownOutcome, defer_cleanup};
use crate::status::plain_boundary;
use rocddi::memory::storage::RetainOnDrop;
use rocddi::session::SessionRetirement;
use std::cell::Cell;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

thread_local! {
    /// Rejects recursive initialization or shutdown on the transition thread.
    static LIFECYCLE_ENTRY: Cell<bool> = const { Cell::new(false) };
}

/// Rejects recursive public lifecycle calls from tool callbacks while a
/// transition runs on this thread.
struct LifecycleCallScope {
    /// Whether this scope set and must clear the thread-local entry flag.
    owns_flag: bool,
    /// The thread-local entry flag must be cleared on its entering thread.
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl LifecycleCallScope {
    /// Enters one public lifecycle call, rejecting same-thread recursion.
    pub(crate) fn enter() -> Result<Self, Status> {
        LIFECYCLE_ENTRY.with(|active| {
            if active.get() {
                Err(INVALID_RUNTIME_STATE)
            } else {
                active.set(true);
                Ok(Self {
                    owns_flag: true,
                    _thread_bound: std::marker::PhantomData,
                })
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
        Self {
            owns_flag,
            _thread_bound: std::marker::PhantomData,
        }
    }
}

impl Drop for LifecycleCallScope {
    fn drop(&mut self) {
        if self.owns_flag {
            LIFECYCLE_ENTRY.with(|active| active.set(false));
        }
    }
}

/// Keeps an unlocked native operation in its HSA session until its owner
/// is either published back into that session or safely retired.
pub(crate) struct InFlightSession {
    /// Lifecycle generation in which this token was acquired.
    generation: u64,
    /// In-flight reference preventing that session from retiring.
    _token: InFlightToken,
    /// Session health retained if native cleanup becomes uncertain.
    health: Arc<SessionHealth>,
}

impl InFlightSession {
    /// Retains the active session under its generation identity.
    fn enter(
        generation: u64,
        tracker: &Arc<InFlightTracker>,
        health: &Arc<SessionHealth>,
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
    /// cannot be returned to its original session.
    pub(crate) fn quarantine(&self) {
        self.health.quarantine();
    }
}
/// Published frontend ownership, retained until explicit retirement succeeds.
struct PublishedSession {
    /// Identity shared by public admission and detached generation leases.
    number: u64,
    /// Object graph released only after checked retirement succeeds.
    state: RetainOnDrop<HsaSession>,
}

/// Checkpoints whose only incomplete work is a counted operation drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShutdownStage {
    /// Ordinary callers must finish before tools prepare shutdown.
    Calls,
    /// Tools prepared successfully; their admitted descendants must finish.
    Cleanup,
}

/// Phase and its ownership are changed together under the lifecycle mutex.
#[allow(
    clippy::large_enum_variant,
    reason = "retirement moves existing owners without allocating"
)]
enum LifecycleState {
    /// No HSA objects or pending native retirement remain.
    Inactive,
    /// One executor constructs the identified HSA session outside the registry.
    Starting(u64),
    /// Objects are published for tool loading before the first reference commits.
    LoadingTools(PublishedSession),
    /// Public initialization references keep this session available.
    Active {
        /// Objects reachable through admitted HSA calls.
        session: PublishedSession,
        /// Successful initializations not yet matched by shutdown calls.
        references: std::num::NonZeroU32,
    },
    /// One executor drains callers and prepares tools for final retirement.
    Preparing {
        /// Objects retained for admitted calls and tool cleanup.
        session: PublishedSession,
        /// Earliest preparation work that the executor still needs to complete.
        stage: ShutdownStage,
    },
    /// No executor is active; a later shutdown resumes this exact checkpoint.
    PendingShutdown {
        /// Objects still reachable by known outstanding work.
        session: PublishedSession,
        /// Incomplete drain to resume without replaying successful preparation.
        stage: ShutdownStage,
    },
    /// One executor owns detached frontend or native retirement for this identity.
    Stopping(u64),
    /// Startup frontend rollback completed; only native retirement remains.
    InitRetirement {
        /// Failed startup identity whose native ownership is still pending.
        number: u64,
        /// Exact native retirement owner retried by the next initialization.
        session: SessionRetirement,
    },
    /// Final frontend retirement completed; only native retirement remains.
    NativeRetirement {
        /// Retired HSA identity whose native ownership is still pending.
        number: u64,
        /// Exact native retirement owner retried by the next shutdown.
        session: SessionRetirement,
    },
    /// Native reachability is uncertain; the full HSA session remains owned.
    Retained {
        /// Dependencies kept alive because their last native user is uncertain.
        _session: PublishedSession,
    },
    /// An abnormal exit retained dependencies outside the registry.
    Unrecoverable,
}

/// Work owned by one initialization call after releasing the registry lock.
pub(crate) enum InitWork {
    /// A reference was acquired from the active session; no startup is needed.
    Reference,
    /// Construct and publish the newly reserved generation identity.
    Create(u64),
    /// Retry failed startup retirement before attempting a replacement generation.
    NativeRetry(SessionRetirement, u64),
}

#[cfg(test)]
impl InitWork {
    /// Extracts the new generation expected by a startup transition test.
    pub(crate) fn created(self) -> Option<u64> {
        match self {
            Self::Create(number) => Some(number),
            _ => None,
        }
    }
}

/// Work owned by one shutdown call after releasing the registry lock.
pub(crate) enum ShutdownWork {
    /// A reference was released while another initialization still owns the session.
    Reference,
    /// Prepare the identified session, resuming any recorded drain checkpoint.
    Prepare(u64),
    /// Retry native retirement after all HSA objects have already been released.
    NativeRetry(SessionRetirement, u64),
}

/// Process-wide HSA session coordinator, independent of native session policy.
pub(crate) struct HsaLifecycle {
    /// Monotonic identity source, including generations that failed startup.
    last_generation: u64,
    /// Transition phase together with the owner required to complete or retry it.
    state: LifecycleState,
}

impl HsaLifecycle {
    /// Retains the identity and lifetime controls from one published owner.
    pub(crate) fn retain_session(&self) -> Result<InFlightSession, Status> {
        let session = self.published().ok_or(NOT_INITIALIZED)?;
        InFlightSession::enter(
            session.number,
            &session.state.control.inflight,
            &session.state.control.health,
        )
    }

    /// Starts with no published session, references, or pending retirement.
    pub(crate) const fn new() -> Self {
        Self {
            last_generation: 0,
            state: LifecycleState::Inactive,
        }
    }

    /// Finds objects that remain available to admitted callers and tool cleanup.
    fn published(&self) -> Option<&PublishedSession> {
        match &self.state {
            LifecycleState::LoadingTools(session)
            | LifecycleState::Active { session, .. }
            | LifecycleState::Preparing { session, .. }
            | LifecycleState::PendingShutdown { session, .. } => Some(session),
            _ => None,
        }
    }

    /// Borrows the session while admitted calls or tool cleanup may use it.
    pub(crate) fn as_ref(&self) -> Option<&HsaSession> {
        self.published().map(|session| &*session.state)
    }

    /// Mutates published objects while the caller holds the registry lock.
    pub(crate) fn as_mut(&mut self) -> Option<&mut HsaSession> {
        match &mut self.state {
            LifecycleState::LoadingTools(session)
            | LifecycleState::Active { session, .. }
            | LifecycleState::Preparing { session, .. }
            | LifecycleState::PendingShutdown { session, .. } => Some(&mut session.state),
            _ => None,
        }
    }

    /// Identifies the generation retained by admitted operations.
    pub(crate) fn active_generation(&self) -> Option<u64> {
        self.published().map(|session| session.number)
    }

    /// Acquires an initialization reference or one exclusive startup action.
    pub(crate) fn begin_init(&mut self) -> Result<InitWork, Status> {
        match &mut self.state {
            LifecycleState::Active { references, .. } => {
                let count = references.get();
                if count == i32::MAX as u32 {
                    return Err(REFCOUNT_OVERFLOW);
                }
                *references = std::num::NonZeroU32::new(count + 1).ok_or(ERROR)?;
                Ok(InitWork::Reference)
            }
            LifecycleState::Inactive => {
                self.last_generation = self.last_generation.checked_add(1).ok_or(ERROR)?;
                self.state = LifecycleState::Starting(self.last_generation);
                Ok(InitWork::Create(self.last_generation))
            }
            LifecycleState::InitRetirement { number, .. } => {
                let number = *number;
                let old = std::mem::replace(&mut self.state, LifecycleState::Stopping(number));
                let LifecycleState::InitRetirement { session, .. } = old else {
                    return Err(ERROR);
                };
                Ok(InitWork::NativeRetry(session, number))
            }
            _ => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Distinguishes an active executor from a checkpoint awaiting an explicit retry.
    fn init_transition_in_progress(&self) -> bool {
        matches!(
            self.state,
            LifecycleState::Starting(_)
                | LifecycleState::LoadingTools(_)
                | LifecycleState::Preparing { .. }
                | LifecycleState::Stopping(_)
        )
    }

    /// Publishes owned state before tool callbacks may access HSA services.
    pub(crate) fn publish_init(&mut self, number: u64, state: HsaSession) -> Result<(), Status> {
        let session = PublishedSession {
            number,
            state: RetainOnDrop::new(state),
        };
        if !matches!(self.state, LifecycleState::Starting(current) if current == number) {
            return Err(ERROR);
        }
        self.state = LifecycleState::LoadingTools(session);
        process_admission().open(number);
        Ok(())
    }

    /// Commits the first reference only after tool loading has completed.
    pub(crate) fn finish_init(&mut self, number: u64) -> Result<(), Status> {
        if !matches!(&self.state, LifecycleState::LoadingTools(session) if session.number == number)
        {
            return Err(ERROR);
        }
        let old = std::mem::replace(&mut self.state, LifecycleState::Unrecoverable);
        let LifecycleState::LoadingTools(session) = old else {
            return Err(ERROR);
        };
        self.state = LifecycleState::Active {
            session,
            references: std::num::NonZeroU32::MIN,
        };
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }

    /// Records startup rollback and wakes callers waiting on the process transition.
    pub(crate) fn cancel_init(
        &mut self,
        number: u64,
        rollback: InitRollback,
    ) -> Result<(), Status> {
        self.cancel_init_with_notification(number, rollback, &LIFECYCLE_CONDVAR)
    }

    /// Transfers failed startup ownership into its retry or terminal state.
    pub(crate) fn cancel_init_with_notification(
        &mut self,
        number: u64,
        rollback: InitRollback,
        changed: &Condvar,
    ) -> Result<(), Status> {
        if !matches!(self.state, LifecycleState::Starting(current) if current == number) {
            return Err(ERROR);
        }
        self.state = match rollback {
            InitRollback::Complete => LifecycleState::Inactive,
            InitRollback::NativeRetry(session) => {
                LifecycleState::InitRetirement { number, session }
            }
            InitRollback::Uncertain => LifecycleState::Unrecoverable,
        };
        process_admission().close();
        changed.notify_all();
        Ok(())
    }

    /// Completes one retry without reconstructing already retired frontend state.
    fn finish_init_retirement(
        &mut self,
        number: u64,
        outcome: ShutdownOutcome,
    ) -> Result<(), Status> {
        if !matches!(self.state, LifecycleState::Stopping(current) if current == number) {
            return Err(ERROR);
        }
        self.state = match outcome {
            ShutdownOutcome::Complete => LifecycleState::Inactive,
            ShutdownOutcome::NativeRetry { session, .. } => {
                LifecycleState::InitRetirement { number, session }
            }
            ShutdownOutcome::RetainedSession { state, .. } => LifecycleState::Retained {
                _session: PublishedSession { number, state },
            },
        };
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }

    #[cfg(test)]
    /// Models an active retirement executor without constructing HSA objects.
    pub(crate) fn stop_without_runtime_for_test(&mut self, number: u64) {
        assert!(matches!(self.state, LifecycleState::Starting(current) if current == number));
        self.state = LifecycleState::Stopping(number);
    }

    /// Reports whether shutdown needs a preparation executor, possibly a helper.
    fn shutdown_will_prepare(&self) -> bool {
        match &self.state {
            LifecycleState::Active { references, .. } => references.get() == 1,
            LifecycleState::PendingShutdown { .. } => true,
            _ => false,
        }
    }

    /// Releases a reference or grants one executor exclusive shutdown progress.
    pub(crate) fn begin_shutdown(&mut self) -> Result<ShutdownWork, Status> {
        if let LifecycleState::Active { references, .. } = &mut self.state {
            if references.get() > 1 {
                *references = std::num::NonZeroU32::new(references.get() - 1).ok_or(ERROR)?;
                return Ok(ShutdownWork::Reference);
            }
        }
        let old = std::mem::replace(&mut self.state, LifecycleState::Unrecoverable);
        match old {
            LifecycleState::Active { session, .. } => {
                session.state.control.request_shutdown();
                let number = session.number;
                self.state = LifecycleState::Preparing {
                    session,
                    stage: ShutdownStage::Calls,
                };
                Ok(ShutdownWork::Prepare(number))
            }
            LifecycleState::PendingShutdown { session, stage } => {
                let number = session.number;
                self.state = LifecycleState::Preparing { session, stage };
                Ok(ShutdownWork::Prepare(number))
            }
            LifecycleState::NativeRetirement { number, session } => {
                self.state = LifecycleState::Stopping(number);
                Ok(ShutdownWork::NativeRetry(session, number))
            }
            other => {
                let status = if matches!(other, LifecycleState::Inactive) {
                    NOT_INITIALIZED
                } else {
                    INVALID_RUNTIME_STATE
                };
                self.state = other;
                Err(status)
            }
        }
    }

    /// Checks executor identity before exposing its current preparation checkpoint.
    fn preparation_stage(&self, number: u64) -> Result<ShutdownStage, Status> {
        match &self.state {
            LifecycleState::Preparing { session, stage } if session.number == number => Ok(*stage),
            _ => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Records completion of tool preparation before waiting for descendants.
    fn finish_tool_preparation(&mut self, number: u64) -> Result<(), Status> {
        match &mut self.state {
            LifecycleState::Preparing { session, stage }
                if session.number == number && *stage == ShutdownStage::Calls =>
            {
                *stage = ShutdownStage::Cleanup;
                Ok(())
            }
            _ => Err(INVALID_RUNTIME_STATE),
        }
    }

    /// Retains known outstanding work without turning a timeout into uncertainty.
    fn pause_shutdown(&mut self, number: u64) -> Result<(), Status> {
        self.preparation_stage(number)?;
        let old = std::mem::replace(&mut self.state, LifecycleState::Unrecoverable);
        let LifecycleState::Preparing { session, stage } = old else {
            return Err(ERROR);
        };
        self.state = LifecycleState::PendingShutdown { session, stage };
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }

    /// Detaches only a fully prepared HSA session into an explicit retirement owner.
    fn finish_prepare_shutdown(&mut self, number: u64) -> Result<PreparedSession, Status> {
        if self.preparation_stage(number)? != ShutdownStage::Cleanup {
            return Err(INVALID_RUNTIME_STATE);
        }
        let session = self.published().ok_or(ERROR)?;
        if !session.state.control.inflight.is_closed_and_idle() {
            return Err(RESOURCE_BUSY);
        }
        session.state.begin_native_retirement()?;
        let old = std::mem::replace(&mut self.state, LifecycleState::Stopping(number));
        let LifecycleState::Preparing { session, .. } = old else {
            return Err(ERROR);
        };
        Ok(PreparedSession(session.state))
    }

    /// Publishes retirement completion or retains the owner needed after failure.
    pub(crate) fn finish_shutdown(
        &mut self,
        number: u64,
        outcome: ShutdownOutcome,
    ) -> Result<(), Status> {
        if !matches!(self.state, LifecycleState::Stopping(current) if current == number) {
            return Err(ERROR);
        }
        self.state = match outcome {
            ShutdownOutcome::Complete => LifecycleState::Inactive,
            ShutdownOutcome::NativeRetry { session, .. } => {
                LifecycleState::NativeRetirement { number, session }
            }
            ShutdownOutcome::RetainedSession { state, .. } => LifecycleState::Retained {
                _session: PublishedSession { number, state },
            },
        };
        LIFECYCLE_CONDVAR.notify_all();
        Ok(())
    }

    /// Prevents replacement when a matching executor exits without committing progress.
    fn retain_failed_transition(&mut self, phase: TransitionPhase, number: u64) {
        let matches = match &self.state {
            LifecycleState::Starting(current) => {
                phase == TransitionPhase::Starting && *current == number
            }
            LifecycleState::LoadingTools(session) => {
                phase == TransitionPhase::Starting && session.number == number
            }
            LifecycleState::Preparing { session, .. } => {
                phase == TransitionPhase::Preparing && session.number == number
            }
            LifecycleState::Stopping(current) => {
                phase == TransitionPhase::Stopping && *current == number
            }
            _ => false,
        };
        if !matches {
            return;
        }
        process_admission().close();
        let old = std::mem::replace(&mut self.state, LifecycleState::Unrecoverable);
        if let LifecycleState::LoadingTools(session) | LifecycleState::Preparing { session, .. } =
            old
        {
            session.state.control.request_stop();
            self.state = LifecycleState::Retained { _session: session };
        }
        LIFECYCLE_CONDVAR.notify_all();
    }
}

/// Proof that producers, callbacks and tools have completed their cleanup window.
struct PreparedSession(RetainOnDrop<HsaSession>);

impl PreparedSession {
    /// Transfers the prepared owner into ordered frontend and native retirement.
    fn retire(self) -> ShutdownOutcome {
        self.0.into_inner().retire_prepared()
    }
}

/// Executor phase whose abnormal exit a transition guard must retain.
#[derive(Clone, Copy, Eq, PartialEq)]
enum TransitionPhase {
    /// Construction or tool loading before the first reference commits.
    Starting,
    /// Caller drain and tool cleanup while HSA objects remain published.
    Preparing,
    /// Detached frontend retirement or a retry of remaining native retirement.
    Stopping,
}

/// Retains incomplete ownership on every abnormal transition exit.
struct LifecycleTransition {
    /// Phase still owned by this executor.
    phase: TransitionPhase,
    /// Identity used to avoid changing an unrelated lifecycle transition.
    number: u64,
    /// Whether scope exit must retain the incomplete transition.
    armed: bool,
}

impl LifecycleTransition {
    /// Protects construction and tool loading until initialization commits.
    fn starting(number: u64) -> Self {
        Self {
            phase: TransitionPhase::Starting,
            number,
            armed: true,
        }
    }

    /// Protects published owners during call drains and tool cleanup.
    fn preparing(number: u64) -> Self {
        Self {
            phase: TransitionPhase::Preparing,
            number,
            armed: true,
        }
    }

    /// Protects a retirement executor after its owner leaves the registry.
    fn stopping(number: u64) -> Self {
        Self {
            phase: TransitionPhase::Stopping,
            number,
            armed: true,
        }
    }

    /// Follows the ownership transfer from published preparation to retirement.
    fn advance_to_stopping(&mut self) {
        self.phase = TransitionPhase::Stopping;
    }

    /// Marks progress committed to a complete or explicitly retryable state.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LifecycleTransition {
    fn drop(&mut self) {
        if self.armed && crate::platform::process_is_current() {
            LIFECYCLE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain_failed_transition(self.phase, self.number);
        }
    }
}

/// Serializes HSA publication, registry access, and lifecycle ownership changes.
pub(crate) static LIFECYCLE: Mutex<HsaLifecycle> = Mutex::new(HsaLifecycle::new());
/// Wakes initialization callers when another transition executor relinquishes ownership.
pub(crate) static LIFECYCLE_CONDVAR: Condvar = Condvar::new();

/// Locks the process registry after rejecting inherited post-fork state.
pub(crate) fn lock() -> Result<MutexGuard<'static, HsaLifecycle>, Status> {
    if !crate::platform::process_is_current() {
        return Err(NOT_INITIALIZED);
    }
    LIFECYCLE.lock().map_err(|_| ERROR)
}

/// Waits only while another thread actually owns a transition executor.
pub(crate) fn begin_init_waiting(
    mut guard: MutexGuard<'_, HsaLifecycle>,
    changed: &Condvar,
) -> Result<InitWork, Status> {
    while guard.init_transition_in_progress() {
        if ApiLeaseScope::current()
            || CallbackScope::active()
            || LogWriteScope::active()
            || guard.as_ref().is_some_and(HsaSession::running_on_worker)
        {
            return Err(INVALID_RUNTIME_STATE);
        }
        guard = changed.wait(guard).map_err(|_| ERROR)?;
    }
    guard.begin_init()
}

/// Borrows still-published objects after the caller has established admission.
pub(crate) fn initialized_mut(runtime: &mut HsaLifecycle) -> Result<&mut HsaSession, Status> {
    runtime.as_mut().ok_or(NOT_INITIALIZED)
}

#[cfg(all(hsa_tool_dispatch, not(test)))]
/// Acquires the coordinator's lease for synchronous tool retirement.
pub(crate) fn enter_tool_cleanup() -> Result<InFlightToken, Status> {
    let registry = lock()?;
    let LifecycleState::Preparing { session, .. } = &registry.state else {
        return Err(INVALID_RUNTIME_STATE);
    };
    process_admission()
        .enter_cleanup(session.number)
        .ok_or(INVALID_RUNTIME_STATE)
}

#[cfg(all(hsa_tool_dispatch, not(test)))]
/// Opens scoped cleanup delegation while the ordinary-call checkpoint is owned.
pub(crate) fn begin_tool_preparation() -> Result<admission::ToolCleanupScope, Status> {
    let registry = lock()?;
    let LifecycleState::Preparing {
        session,
        stage: ShutdownStage::Calls,
    } = &registry.state
    else {
        return Err(INVALID_RUNTIME_STATE);
    };
    if session.state.control.health.is_quarantined() {
        return Err(INVALID_RUNTIME_STATE);
    }
    process_admission()
        .begin_cleanup(session.number)
        .ok_or(INVALID_RUNTIME_STATE)
}

#[cfg(all(hsa_tool_dispatch, not(test)))]
/// Admits tool work through the active session or its live cleanup authority.
pub(crate) fn enter_delegated_tool_cleanup(number: u64) -> Result<InFlightToken, Status> {
    let registry = lock()?;
    let session = registry.published().ok_or(INVALID_RUNTIME_STATE)?;
    if session.number != number || session.state.control.health.is_quarantined() {
        return Err(INVALID_RUNTIME_STATE);
    }
    match &registry.state {
        LifecycleState::Active { .. } | LifecycleState::LoadingTools(_) => {
            process_admission().enter_tool_work(number)
        }
        LifecycleState::Preparing { .. } if ApiLeaseScope::cleanup() => {
            process_admission().enter_tool_work(number)
        }
        LifecycleState::Preparing { .. } => process_admission().enter_delegated_cleanup(number),
        // A timed-out cleanup descendant may continue nested work under its
        // existing lease, but never opens delegation to another worker.
        LifecycleState::PendingShutdown { .. } if ApiLeaseScope::cleanup() => {
            process_admission().enter_tool_work(number)
        }
        _ => None,
    }
    .ok_or(INVALID_RUNTIME_STATE)
}

mod operations;
pub(crate) use operations::{initialize, shutdown};

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests;
