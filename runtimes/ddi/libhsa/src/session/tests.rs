// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::startup::PendingInit;
use super::*;
use crate::lifecycle::HsaLifecycle;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::Duration;

#[test]
#[allow(clippy::unwrap_used)]
fn parked_shutdown_helper_exits_if_commit_is_aborted() {
    let mut lifecycle = HsaLifecycle::new();
    let generation = lifecycle.begin_init().unwrap().created().unwrap();
    let (commit, instruction) = std::sync::mpsc::channel::<u64>();
    let (finished, completion) = std::sync::mpsc::channel();
    let cleanup_called = Arc::new(AtomicBool::new(false));
    let observed = cleanup_called.clone();
    defer_cleanup(instruction, move |instruction| {
        if instruction.recv().is_ok() {
            observed.store(true, Ordering::Release);
        }
        finished.send(()).unwrap();
    })
    .unwrap();
    // A failed commit drops its instruction sender without advancing the
    // generation or asking the parked helper to retire it.
    assert_eq!(
        lifecycle.begin_shutdown().err(),
        Some(INVALID_RUNTIME_STATE)
    );
    drop(commit);
    completion.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(!cleanup_called.load(Ordering::Acquire));
    assert_eq!(lifecycle.begin_init().err(), Some(INVALID_RUNTIME_STATE));
    lifecycle
        .cancel_init(generation, InitRollback::Complete)
        .unwrap();
}

#[test]
/// Verifies that deferred cleanup joins the callback worker after exit.
fn deferred_cleanup_joins_the_callback_worker_after_exit() {
    /// Owns a callback worker and its completion probe.
    struct Owner {
        /// Worker joined by deferred cleanup.
        worker: JoinHandle<()>,
        /// Records whether the worker exited before cleanup finished.
        exited: Arc<AtomicBool>,
        /// Reports the observed exit state to the test thread.
        completed: std::sync::mpsc::Sender<bool>,
    }
    let exited = Arc::new(AtomicBool::new(false));
    let (owner_tx, owner_rx) = std::sync::mpsc::channel::<Owner>();
    let (completed_tx, completed_rx) = std::sync::mpsc::channel();
    let worker_exited = exited.clone();
    let worker = thread::spawn(move || {
        let owner = owner_rx.recv().unwrap();
        assert_eq!(owner.worker.thread().id(), thread::current().id());
        assert!(
            defer_cleanup(owner, |owner| {
                owner.worker.join().unwrap();
                owner
                    .completed
                    .send(owner.exited.load(AtomicOrdering::Acquire))
                    .unwrap();
            })
            .is_ok()
        );
        worker_exited.store(true, AtomicOrdering::Release);
    });
    owner_tx
        .send(Owner {
            worker,
            exited,
            completed: completed_tx,
        })
        .unwrap();
    assert!(completed_rx.recv_timeout(Duration::from_secs(2)).unwrap());
}

#[test]
fn initialization_rollback_closes_admission_before_releasing_gpu_borrows() {
    struct Activated {
        session: NativeSession,
        released: Arc<AtomicUsize>,
    }
    impl Drop for Activated {
        fn drop(&mut self) {
            assert_eq!(
                self.session.state().unwrap(),
                rocddi::session::SessionState::Retiring
            );
            self.released.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }
    for completed_activations in [0, 1, 2] {
        let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
        let released = Arc::new(AtomicUsize::new(0));
        let gpus = (0..completed_activations)
            .map(|_| Activated {
                session: session.clone(),
                released: released.clone(),
            })
            .collect();
        let mut pending = PendingInit {
            debugger: None,
            native_session: Some(session),
            gpus,
        };
        assert!(matches!(pending.cleanup(), InitRollback::Complete));
        assert_eq!(
            released.load(AtomicOrdering::Relaxed),
            completed_activations
        );
    }
}

#[test]
#[allow(clippy::panic)]
fn initialization_rollback_keeps_native_retirement_retryable() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let borrower = session.clone();
    let mut pending = PendingInit::<()> {
        debugger: None,
        native_session: Some(session),
        gpus: Vec::new(),
    };
    let InitRollback::NativeRetry(mut retirement) = pending.cleanup() else {
        panic!("a live session borrower must retain native retirement");
    };
    assert_eq!(
        borrower.state().unwrap(),
        rocddi::session::SessionState::Retiring
    );
    assert!(matches!(retirement.try_close(), RetirementOutcome::Busy));
    drop(borrower);
    assert!(matches!(retirement.try_close(), RetirementOutcome::Closed));
    assert!(matches!(pending.cleanup(), InitRollback::Complete));
}

#[test]
#[allow(clippy::panic)]
fn initialization_rollback_retains_the_session_if_frontend_release_unwinds() {
    struct PanickingOwner;
    impl Drop for PanickingOwner {
        fn drop(&mut self) {
            panic!("injected frontend rollback unwind");
        }
    }
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let mut observer = session.clone();
    let mut pending = PendingInit {
        debugger: None,
        native_session: Some(session),
        gpus: vec![PanickingOwner],
    };
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pending.cleanup())).is_err());
    drop(pending);
    assert_eq!(
        observer.state().unwrap(),
        rocddi::session::SessionState::Retiring
    );
    assert!(matches!(observer.try_close(), RetirementOutcome::Busy));
}
