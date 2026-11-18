// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process admission and object-local rundown protection.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// One bit closes admission; lower bits count work in this gate.
const ADMISSION_CLOSED: usize = 1 << (usize::BITS - 1);

/// Process admission shared by public and unlocked HSA work. Queue-local
/// trackers remain separate because queue destruction has its own lifetime.
static PROCESS_ADMISSION: std::sync::LazyLock<Arc<InFlightTracker>> =
    std::sync::LazyLock::new(|| Arc::new(InFlightTracker::new_process()));

thread_local! {
    /// Generation retained by this thread's current call or callback.
    static API_LEASE_GENERATION: Cell<u64> = const { Cell::new(0) };
    /// Cleanup work uses the worker stop flag instead of public cancellation.
    static API_CLEANUP_LEASE: Cell<bool> = const { Cell::new(false) };
}

/// Executes work under a retained generation without exposing the TLS guard.
pub(crate) struct ApiLeaseScope;

/// Restores TLS before releasing the token that justified its contents.
/// This guard never leaves `ApiLeaseScope::run`, so destruction is lexical.
struct LeaseGuard {
    /// Lease generation restored when this scope exits.
    previous: u64,
    /// Cleanup classification restored together with the generation.
    previous_cleanup: bool,
    /// The parent count stays live until the thread-local marker is cleared.
    _token: InFlightToken,
    /// A lease scope must be dropped on the thread whose marker it changed.
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl ApiLeaseScope {
    /// Runs one operation with its owned lease available to nested calls.
    pub(crate) fn run<R>(token: InFlightToken, operation: impl FnOnce() -> R) -> R {
        let previous = API_LEASE_GENERATION.with(|slot| slot.replace(token.generation()));
        let previous_cleanup = API_CLEANUP_LEASE.with(|slot| slot.replace(token.cleanup));
        let _guard = LeaseGuard {
            previous,
            previous_cleanup,
            _token: token,
            _thread_bound: std::marker::PhantomData,
        };
        operation()
    }

    /// Executes a worker body with an inherited cleanup lease when supplied.
    pub(crate) fn run_optional<R>(
        token: Option<InFlightToken>,
        operation: impl FnOnce() -> R,
    ) -> R {
        match token {
            Some(token) => Self::run(token, operation),
            None => operation(),
        }
    }

    /// Reports whether this thread holds the current generation's lease.
    pub(crate) fn current() -> bool {
        API_LEASE_GENERATION.with(|slot| slot.get() != 0)
    }

    /// Whether the current call is part of explicit generation cleanup.
    pub(crate) fn cleanup() -> bool {
        API_CLEANUP_LEASE.with(Cell::get)
    }

    /// Allows nested work on this thread to inherit its live parent lease.
    fn retained_generation(generation: u64) -> bool {
        API_LEASE_GENERATION.with(|slot| slot.get() == generation)
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        API_LEASE_GENERATION.with(|slot| slot.set(self.previous));
        API_CLEANUP_LEASE.with(|slot| slot.set(self.previous_cleanup));
    }
}

/// Returns the single process admission gate for one HSA generation.
pub(crate) fn process_admission() -> &'static Arc<InFlightTracker> {
    &PROCESS_ADMISSION
}

/// Counts calls that must finish before shutdown releases runtime storage.
pub(crate) struct InFlightTracker {
    /// Process admission permits callback reentry; object drains do not.
    process: bool,
    /// Closing bit and number of admitted operations.
    state: AtomicUsize,
    /// Generation served by the process admission gate, or zero for a local gate.
    generation: AtomicU64,
    /// Serializes delegated cleanup admission against the end of preparation.
    cleanup_generation: Mutex<u64>,
    /// Wakes shutdown when the final token is dropped.
    idle: Condvar,
    /// Prevents a notification from racing the waiter's predicate check.
    idle_lock: Mutex<()>,
}

impl InFlightTracker {
    /// Initializes the count and closing flag for in-flight operations.
    pub(crate) fn new() -> Self {
        Self {
            process: false,
            state: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            cleanup_generation: Mutex::new(0),
            idle: Condvar::new(),
            idle_lock: Mutex::new(()),
        }
    }

    /// Initializes the closed process gate before its first HSA generation.
    pub(crate) fn new_process() -> Self {
        Self {
            process: true,
            state: AtomicUsize::new(ADMISSION_CLOSED),
            generation: AtomicU64::new(0),
            cleanup_generation: Mutex::new(0),
            idle: Condvar::new(),
            idle_lock: Mutex::new(()),
        }
    }

    /// Publishes one generation after its HSA state is ready for tool `OnLoad`.
    pub(crate) fn open(&self, generation: u64) {
        assert_eq!(self.state.load(Ordering::Acquire), ADMISSION_CLOSED);
        self.generation.store(generation, Ordering::Release);
        self.state.store(0, Ordering::Release);
    }

    /// Proves that an old generation has no admitted work before detachment.
    pub(crate) fn is_closed_and_idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == ADMISSION_CLOSED
    }

    /// Adds an in-flight operation unless this generation is closing.
    pub(crate) fn enter(self: &Arc<Self>) -> Option<InFlightToken> {
        if !crate::platform::process_is_current() {
            return None;
        }
        let admitted = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state < ADMISSION_CLOSED - 1).then_some(state + 1)
            });
        if admitted.is_err() {
            // An admitted call or callback may perform nested HSA work after
            // close. Its same-thread parent lease keeps the generation alive.
            let generation = self.generation.load(Ordering::Acquire);
            if generation == 0 || !ApiLeaseScope::retained_generation(generation) || !self.process {
                return None;
            }
            self.state
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                    (ADMISSION_CLOSED + 1..usize::MAX)
                        .contains(&state)
                        .then_some(state + 1)
                })
                .ok()?;
        }
        Some(InFlightToken {
            tracker: Some(self.clone()),
            generation: self.generation.load(Ordering::Acquire),
            cleanup: ApiLeaseScope::cleanup(),
        })
    }

    /// Grants one internal teardown lease only after a matching process
    /// generation is closed and idle. The lifecycle owner retains its state.
    #[cfg(any(test, hsa_tool_dispatch))]
    pub(crate) fn enter_cleanup(self: &Arc<Self>, generation: u64) -> Option<InFlightToken> {
        if !crate::platform::process_is_current()
            || !self.process
            || generation == 0
            || self.generation.load(Ordering::Acquire) != generation
        {
            return None;
        }
        self.state
            .compare_exchange(
                ADMISSION_CLOSED,
                ADMISSION_CLOSED + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()?;
        Some(InFlightToken {
            tracker: Some(self.clone()),
            generation,
            cleanup: true,
        })
    }

    /// Opens a cleanup window after ordinary admission has drained. The
    /// anchor remains live until the owner closes delegation on every exit.
    #[cfg(any(test, hsa_tool_dispatch))]
    pub(crate) fn begin_cleanup(self: &Arc<Self>, generation: u64) -> Option<ToolCleanupScope> {
        let mut active = self.cleanup_generation.lock().ok()?;
        if *active != 0 {
            return None;
        }
        let token = self.enter_cleanup(generation)?;
        *active = generation;
        Some(ToolCleanupScope {
            window: CleanupWindow(self.clone()),
            token,
        })
    }

    /// Delegates one bounded cleanup operation to a known worker. The mutex
    /// makes closing the cleanup window atomic with acquiring its last lease.
    pub(crate) fn enter_delegated_cleanup(
        self: &Arc<Self>,
        generation: u64,
    ) -> Option<InFlightToken> {
        let active = self.cleanup_generation.lock().ok()?;
        if generation == 0 || *active != generation || !crate::platform::process_is_current() {
            return None;
        }
        self.state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (ADMISSION_CLOSED + 1..usize::MAX)
                    .contains(&state)
                    .then_some(state + 1)
            })
            .ok()?;
        Some(InFlightToken {
            tracker: Some(self.clone()),
            generation,
            cleanup: true,
        })
    }

    /// Retains an already admitted internal operation through public closure.
    /// Unlike ordinary work, its bounded native stop/drain sequence is not
    /// canceled by closing public admission.
    #[cfg(all(hsa_tool_dispatch, not(test)))]
    pub(crate) fn enter_tool_work(self: &Arc<Self>, generation: u64) -> Option<InFlightToken> {
        if generation == 0 || self.generation.load(Ordering::Acquire) != generation {
            return None;
        }
        let mut token = self.enter()?;
        token.cleanup = true;
        Some(token)
    }

    /// Internal callback workers can drain during preparation, but cannot
    /// keep admitting work after the owner closes the delegated window.
    fn enter_service(self: &Arc<Self>) -> Option<InFlightToken> {
        self.enter()
            .or_else(|| self.enter_delegated_cleanup(self.generation.load(Ordering::Acquire)))
    }

    /// Excludes new native work before shutdown waits for existing tokens.
    pub(crate) fn close(&self) {
        self.state.fetch_or(ADMISSION_CLOSED, Ordering::AcqRel);
    }

    /// Waits for idle to complete.
    pub(crate) fn wait_idle(&self) {
        if !crate::platform::process_is_current() {
            return;
        }
        debug_assert!(self.state.load(Ordering::Acquire) & ADMISSION_CLOSED != 0);
        let mut idle = self
            .idle_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while self.state.load(Ordering::Acquire) & !ADMISSION_CLOSED != 0 {
            idle = self
                .idle
                .wait(idle)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Bounds preparation when an external tool holds an admitted native
    /// operation. A timeout requires retaining the generation, not detachment.
    pub(crate) fn wait_idle_for(&self, timeout: Duration) -> bool {
        if !crate::platform::process_is_current() {
            return false;
        }
        debug_assert!(self.state.load(Ordering::Acquire) & ADMISSION_CLOSED != 0);
        let idle = self
            .idle_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (_idle, _) = self
            .idle
            .wait_timeout_while(idle, timeout, |()| {
                self.state.load(Ordering::Acquire) & !ADMISSION_CLOSED != 0
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.is_closed_and_idle()
    }
}

/// One registered operation that must finish before runtime teardown.
pub(crate) struct InFlightToken {
    /// Counter decremented when this operation leaves its native work.
    tracker: Option<Arc<InFlightTracker>>,
    /// Generation retained by this token, or zero for an object-local drain.
    generation: u64,
    /// Selects cleanup cancellation for nested HSA calls.
    cleanup: bool,
}

/// Owns permission to execute preparation with delegated worker admission.
/// Merely holding this value never installs a thread-local execution scope.
#[cfg(any(test, hsa_tool_dispatch))]
pub(crate) struct ToolCleanupScope {
    /// Delegation closes before the anchor is released, including unused permits.
    window: CleanupWindow,
    /// Counted anchor retained until all delegated entry has been fenced.
    token: InFlightToken,
}

#[cfg(any(test, hsa_tool_dispatch))]
impl ToolCleanupScope {
    /// Executes preparation and fences new delegates before restoring TLS.
    pub(crate) fn run<R>(self, operation: impl FnOnce() -> R) -> R {
        let Self { window, token } = self;
        ApiLeaseScope::run(token, || {
            let _window = window;
            operation()
        })
    }
}

/// Closes delegation on every exit before its preparation lease is released.
#[cfg(any(test, hsa_tool_dispatch))]
struct CleanupWindow(Arc<InFlightTracker>);

#[cfg(any(test, hsa_tool_dispatch))]
impl Drop for CleanupWindow {
    fn drop(&mut self) {
        if !crate::platform::process_is_current() {
            return;
        }
        *self
            .0
            .cleanup_generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = 0;
    }
}

/// Acquires a generation lease for an internal callback iteration.
pub(crate) fn enter_runtime_service() -> Option<InFlightToken> {
    #[cfg(test)]
    let token = enter_public_raw_handle();
    #[cfg(not(test))]
    let token = PROCESS_ADMISSION.enter_service();
    token
}

impl InFlightToken {
    /// Returns the generation retained by an admitted process call.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for InFlightToken {
    /// Decrements the active count and wakes teardown when it reaches zero.
    fn drop(&mut self) {
        if !crate::platform::process_is_current() {
            if let Some(tracker) = self.tracker.take() {
                std::mem::forget(tracker);
            }
            return;
        }
        let Some(tracker) = self.tracker.as_ref() else {
            return;
        };
        let previous = tracker.state.fetch_sub(1, Ordering::AcqRel);
        if previous == ADMISSION_CLOSED + 1 {
            let _idle = tracker
                .idle_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            tracker.idle.notify_all();
        }
    }
}

/// Admits one raw-handle operation for its full callback duration.
pub(crate) fn enter_public_raw_handle() -> Option<InFlightToken> {
    if !crate::platform::process_is_current() {
        return None;
    }
    #[cfg(test)]
    return TEST_PUBLIC_RAW_GATE
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(|| TEST_RAW_HANDLES.clone())
        .enter();
    #[cfg(not(test))]
    PROCESS_ADMISSION.enter()
}

#[cfg(test)]
thread_local! {
    /// Lets one test thread race a process gate without closing other tests' gate.
    static TEST_PUBLIC_RAW_GATE: std::cell::RefCell<Option<Arc<InFlightTracker>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn install_test_public_raw_gate(gate: Arc<InFlightTracker>) -> impl Drop {
    /// Restores the test thread's previous raw-handle gate on scope exit.
    struct RestoreGate(Option<Arc<InFlightTracker>>);

    impl Drop for RestoreGate {
        fn drop(&mut self) {
            TEST_PUBLIC_RAW_GATE.with(|slot| {
                slot.replace(self.0.take());
            });
        }
    }

    RestoreGate(TEST_PUBLIC_RAW_GATE.with(|slot| slot.replace(Some(gate))))
}

#[cfg(test)]
static TEST_RAW_HANDLES: std::sync::LazyLock<Arc<InFlightTracker>> =
    std::sync::LazyLock::new(|| Arc::new(InFlightTracker::new()));

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod raw_handle_tests {
    use super::*;

    #[test]
    fn shutdown_waits_for_admitted_raw_calls() {
        let gate = Arc::new(InFlightTracker::new());
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
        closed_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(gate.enter().is_none());
        assert!(idle_rx.try_recv().is_err());
        drop(admitted);
        idle_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn shutdown_without_readers_can_reopen_admission() {
        let gate = Arc::new(InFlightTracker::new());
        gate.close();
        gate.wait_idle();
        gate.open(1);
        assert!(gate.enter().is_some());
    }

    #[test]
    fn closed_gate_admits_only_work_inheriting_a_live_lease() {
        let gate = Arc::new(InFlightTracker::new_process());
        gate.open(7);
        let parent = gate.enter().unwrap();
        ApiLeaseScope::run(parent, || {
            gate.close();
            let outsider = gate.clone();
            assert!(
                std::thread::spawn(move || outsider.enter().is_none())
                    .join()
                    .unwrap()
            );
            let nested = gate.enter().unwrap();
            assert_eq!(nested.generation(), 7);
            ApiLeaseScope::run(nested, || assert!(ApiLeaseScope::current()));
            assert!(ApiLeaseScope::current());
        });
        assert!(!ApiLeaseScope::current());
        gate.wait_idle();
        gate.open(8);
        assert_eq!(gate.enter().unwrap().generation(), 8);
    }

    #[test]
    fn stale_execution_marker_cannot_admit_work_after_the_last_lease() {
        let gate = Arc::new(InFlightTracker::new_process());
        gate.open(7);
        gate.close();
        // Defense in depth: TLS alone never proves retained ownership. The
        // public scoped API cannot produce this stale marker on normal exits.
        API_LEASE_GENERATION.with(|slot| slot.set(7));
        assert!(gate.is_closed_and_idle());
        let admitted = gate.enter();
        API_LEASE_GENERATION.with(|slot| slot.set(0));
        assert!(admitted.is_none());
    }

    #[test]
    #[allow(clippy::panic)]
    fn unwinding_nested_execution_restores_tls_before_draining() {
        let gate = Arc::new(InFlightTracker::new_process());
        gate.open(7);
        let token = gate.enter().unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ApiLeaseScope::run(token, || {
                gate.close();
                ApiLeaseScope::run(gate.enter().unwrap(), || panic!("injected unwind"));
            });
        }));
        assert!(result.is_err());
        assert!(!ApiLeaseScope::current());
        assert!(!ApiLeaseScope::cleanup());
        assert!(gate.is_closed_and_idle());
        assert!(gate.enter().is_none());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod cleanup_tests {
    use super::*;

    #[test]
    fn cleanup_lease_does_not_reopen_external_admission() {
        let tracker = Arc::new(InFlightTracker::new_process());
        tracker.open(17);
        assert!(tracker.enter_cleanup(17).is_none());
        tracker.close();
        assert!(tracker.enter_cleanup(18).is_none());
        let Some(token) = tracker.enter_cleanup(17) else {
            panic!("closed idle generation denied cleanup")
        };
        assert!(tracker.enter().is_none());
        assert!(tracker.enter_cleanup(17).is_none());
        ApiLeaseScope::run(token, || {
            assert!(tracker.enter().is_some());
            let other = tracker.clone();
            assert!(
                std::thread::spawn(move || other.enter().is_none())
                    .join()
                    .unwrap()
            );
        });
        assert!(tracker.is_closed_and_idle());
        assert!(tracker.enter().is_none());
    }

    #[test]
    fn cleanup_delegation_is_generation_scoped_and_drained_after_closure() {
        let tracker = Arc::new(InFlightTracker::new_process());
        tracker.open(41);
        tracker.close();
        let preparation = tracker.begin_cleanup(41).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_tracker = tracker.clone();
        let worker = std::thread::spawn(move || {
            assert!(worker_tracker.enter().is_none());
            assert!(worker_tracker.enter_delegated_cleanup(40).is_none());
            ApiLeaseScope::run(worker_tracker.enter_service().unwrap(), || {
                assert!(ApiLeaseScope::cleanup());
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                // A retained parent still admits nested native cleanup after
                // delegation closes; the caller must drain it before retirement.
                assert!(worker_tracker.enter().is_some());
            });
        });
        assert!(!ApiLeaseScope::cleanup());
        preparation.run(|| {
            assert!(ApiLeaseScope::cleanup());
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        });
        assert!(!ApiLeaseScope::cleanup());
        assert!(!tracker.is_closed_and_idle());
        assert!(!tracker.wait_idle_for(Duration::from_millis(1)));
        assert!(tracker.enter_service().is_none());
        assert!(tracker.enter_delegated_cleanup(41).is_none());
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        tracker.wait_idle();
        assert!(tracker.wait_idle_for(Duration::ZERO));
        tracker.open(42);
        tracker.close();
        tracker.begin_cleanup(42).unwrap().run(|| {
            assert!(tracker.enter_delegated_cleanup(41).is_none());
            assert!(tracker.enter_delegated_cleanup(42).is_some());
        });
    }

    #[test]
    fn unused_or_unwound_cleanup_permission_closes_delegation() {
        let tracker = Arc::new(InFlightTracker::new_process());
        tracker.open(17);
        tracker.close();
        drop(tracker.begin_cleanup(17).unwrap());
        assert!(tracker.enter_delegated_cleanup(17).is_none());
        assert!(tracker.is_closed_and_idle());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tracker.begin_cleanup(17).unwrap().run(|| {
                assert!(ApiLeaseScope::cleanup());
                panic!("injected cleanup unwind");
            });
        }));
        assert!(result.is_err());
        assert!(!ApiLeaseScope::current());
        assert!(!ApiLeaseScope::cleanup());
        assert!(tracker.enter_delegated_cleanup(17).is_none());
        assert!(tracker.is_closed_and_idle());
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
