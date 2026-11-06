// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process admission and object-local rundown protection.

use super::*;

/// One bit closes admission; lower bits count work in this gate.
const ADMISSION_CLOSED: usize = 1 << (usize::BITS - 1);

/// Process admission shared by public and unlocked HSA work. Queue-local
/// trackers remain separate because queue destruction has its own lifetime.
static PROCESS_ADMISSION: std::sync::LazyLock<Arc<InFlightTracker>> =
    std::sync::LazyLock::new(|| Arc::new(InFlightTracker::new_process()));

thread_local! {
    /// Generation retained by this thread's current call or callback.
    static API_LEASE_GENERATION: Cell<u64> = const { Cell::new(0) };
}

/// Owns a process lease and makes it available to nested work on this thread.
pub(crate) struct ApiLeaseScope {
    /// Lease generation restored when this scope exits.
    previous: u64,
    /// The parent count stays live until the thread-local marker is cleared.
    _token: InFlightToken,
}

impl ApiLeaseScope {
    /// Publishes an owned lease on its calling thread until this scope drops.
    pub(crate) fn enter(token: InFlightToken) -> Self {
        let previous = API_LEASE_GENERATION.with(|slot| slot.replace(token.generation()));
        Self {
            previous,
            _token: token,
        }
    }

    /// Reports whether this thread holds the current generation's lease.
    pub(crate) fn current() -> bool {
        API_LEASE_GENERATION.with(|slot| slot.get() != 0)
    }

    /// Allows nested work on this thread to inherit its live parent lease.
    fn retained_generation(generation: u64) -> bool {
        API_LEASE_GENERATION.with(|slot| slot.get() == generation)
    }
}

impl Drop for ApiLeaseScope {
    fn drop(&mut self) {
        API_LEASE_GENERATION.with(|slot| slot.set(self.previous));
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
            idle: Condvar::new(),
            idle_lock: Mutex::new(()),
        }
    }

    /// Initializes the closed process gate before its first HSA generation.
    fn new_process() -> Self {
        Self {
            process: true,
            state: AtomicUsize::new(ADMISSION_CLOSED),
            generation: AtomicU64::new(0),
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
                    (ADMISSION_CLOSED..usize::MAX)
                        .contains(&state)
                        .then_some(state + 1)
                })
                .ok()?;
        }
        Some(InFlightToken {
            tracker: self.clone(),
            generation: self.generation.load(Ordering::Acquire),
        })
    }

    /// Excludes new native work before shutdown waits for existing tokens.
    pub(crate) fn close(&self) {
        self.state.fetch_or(ADMISSION_CLOSED, Ordering::AcqRel);
    }

    /// Waits for idle to complete.
    pub(crate) fn wait_idle(&self) {
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
}

/// One registered operation that must finish before runtime teardown.
pub(crate) struct InFlightToken {
    /// Counter decremented when this operation leaves its native work.
    tracker: Arc<InFlightTracker>,
    /// Generation retained by this token, or zero for an object-local drain.
    generation: u64,
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
        let previous = self.tracker.state.fetch_sub(1, Ordering::AcqRel);
        if previous == ADMISSION_CLOSED + 1 {
            let _idle = self
                .tracker
                .idle_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.tracker.idle.notify_all();
        }
    }
}

/// Admits one raw-handle operation for its full callback duration.
pub(crate) fn enter_public_raw_handle() -> Option<InFlightToken> {
    #[cfg(test)]
    return TEST_RAW_HANDLES.enter();
    #[cfg(not(test))]
    PROCESS_ADMISSION.enter()
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
        let scope = ApiLeaseScope::enter(parent);
        gate.close();
        let outsider = gate.clone();
        assert!(
            std::thread::spawn(move || outsider.enter().is_none())
                .join()
                .unwrap()
        );
        {
            let nested = gate.enter().unwrap();
            assert_eq!(nested.generation(), 7);
        }
        drop(scope);
        gate.wait_idle();
        gate.open(8);
        assert_eq!(gate.enter().unwrap().generation(), 8);
    }
}
