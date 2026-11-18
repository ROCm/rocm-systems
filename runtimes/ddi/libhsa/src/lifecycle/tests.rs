// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Lifecycle ownership and resumable retirement regressions.

use super::*;
use crate::session::testing::session_without_devices;
use crate::signal::SignalState;
use rocddi::session::SessionState;
use rocddi::session::{DriverContextLifetime, Session as NativeSession};

fn active(session: NativeSession) -> HsaLifecycle {
    HsaLifecycle {
        last_generation: 1,
        state: LifecycleState::Active {
            session: PublishedSession {
                number: 1,
                state: RetainOnDrop::new(session_without_devices(session)),
            },
            references: std::num::NonZeroU32::MIN,
        },
    }
}

#[test]
fn prepared_shutdown_closes_core_after_cleanup_services_finish() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let mut lifecycle = active(session);
    assert!(matches!(
        lifecycle.begin_shutdown().unwrap(),
        ShutdownWork::Prepare(1)
    ));
    assert_eq!(observer.state().unwrap(), SessionState::Open);
    drop(SignalState::new(&observer).unwrap());
    assert!(lifecycle.finish_prepare_shutdown(1).is_err());
    lifecycle.finish_tool_preparation(1).unwrap();
    let prepared = lifecycle.finish_prepare_shutdown(1).unwrap();
    assert_eq!(observer.state().unwrap(), SessionState::Retiring);
    assert!(SignalState::new(&observer).is_err());
    drop(observer);
    let outcome = prepared.retire();
    assert!(matches!(outcome, ShutdownOutcome::Complete));
    lifecycle.finish_shutdown(1, outcome).unwrap();
    assert!(matches!(lifecycle.state, LifecycleState::Inactive));
}

#[test]
fn timed_out_calls_keep_the_generation_and_resume_the_first_checkpoint() {
    let mut lifecycle = active(NativeSession::new(DriverContextLifetime::Session).unwrap());
    let gate = lifecycle.as_ref().unwrap().control.inflight.clone();
    let caller = gate.enter().unwrap();
    assert!(matches!(
        lifecycle.begin_shutdown().unwrap(),
        ShutdownWork::Prepare(1)
    ));
    assert!(!gate.wait_idle_for(Duration::ZERO));
    lifecycle.pause_shutdown(1).unwrap();
    assert!(!lifecycle.init_transition_in_progress());
    assert_eq!(lifecycle.begin_init().err(), Some(INVALID_RUNTIME_STATE));
    assert_eq!(lifecycle.active_generation(), Some(1));
    assert!(gate.enter().is_none());
    ApiLeaseScope::run(caller, || {
        let nested = gate.enter().unwrap();
        assert_eq!(nested.generation(), 1);
    });
    assert!(gate.wait_idle_for(Duration::ZERO));
    assert!(matches!(
        lifecycle.begin_shutdown().unwrap(),
        ShutdownWork::Prepare(1)
    ));
    assert_eq!(
        lifecycle.preparation_stage(1).unwrap(),
        ShutdownStage::Calls
    );
    lifecycle.finish_tool_preparation(1).unwrap();
    let outcome = lifecycle.finish_prepare_shutdown(1).unwrap().retire();
    lifecycle.finish_shutdown(1, outcome).unwrap();
    assert!(matches!(
        lifecycle.begin_init().unwrap(),
        InitWork::Create(2)
    ));
    lifecycle.cancel_init(2, InitRollback::Complete).unwrap();
}

#[test]
fn timed_out_cleanup_resumes_without_repeating_tool_preparation() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let mut lifecycle = active(session);
    let gate = lifecycle.as_ref().unwrap().control.inflight.clone();
    let ordinary_stop = lifecycle.as_ref().unwrap().control.operation_stop();
    lifecycle.begin_shutdown().unwrap();
    assert!(ordinary_stop.load(std::sync::atomic::Ordering::Acquire));
    let (cleanup, cleanup_stop) = gate.begin_cleanup(1).unwrap().run(|| {
        (
            gate.enter_delegated_cleanup(1).unwrap(),
            lifecycle.as_ref().unwrap().control.operation_stop(),
        )
    });
    lifecycle.finish_tool_preparation(1).unwrap();
    assert!(!cleanup_stop.load(std::sync::atomic::Ordering::Acquire));
    assert!(!gate.wait_idle_for(Duration::ZERO));
    assert_eq!(
        lifecycle.finish_prepare_shutdown(1).err(),
        Some(RESOURCE_BUSY)
    );
    assert_eq!(observer.state().unwrap(), SessionState::Open);
    lifecycle.pause_shutdown(1).unwrap();
    assert!(!cleanup_stop.load(std::sync::atomic::Ordering::Acquire));
    assert!(gate.enter().is_none());
    drop(cleanup);
    assert!(gate.wait_idle_for(Duration::ZERO));
    lifecycle.begin_shutdown().unwrap();
    assert_eq!(
        lifecycle.preparation_stage(1).unwrap(),
        ShutdownStage::Cleanup
    );
    assert_eq!(
        lifecycle.finish_tool_preparation(1),
        Err(INVALID_RUNTIME_STATE)
    );
    assert!(!cleanup_stop.load(std::sync::atomic::Ordering::Acquire));
    let callbacks = lifecycle.as_mut().unwrap().begin_tool_shutdown();
    assert!(cleanup_stop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(callbacks.quiesce(), SUCCESS);
    drop(observer);
    let outcome = lifecycle.finish_prepare_shutdown(1).unwrap().retire();
    lifecycle.finish_shutdown(1, outcome).unwrap();
    assert!(matches!(lifecycle.state, LifecycleState::Inactive));
}

#[test]
fn initialization_retries_only_pending_native_rollback() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    session.begin_retirement().unwrap();
    let mut lifecycle = HsaLifecycle::new();
    assert!(matches!(
        lifecycle.begin_init().unwrap(),
        InitWork::Create(1)
    ));
    lifecycle
        .cancel_init(1, InitRollback::NativeRetry(session.into_retirement()))
        .unwrap();
    assert!(lifecycle.as_ref().is_none());
    let InitWork::NativeRetry(session, number) = lifecycle.begin_init().unwrap() else {
        panic!("startup retry was not retained");
    };
    let outcome = ShutdownOutcome::retry(session);
    assert_eq!(outcome.status(), RESOURCE_BUSY);
    lifecycle.finish_init_retirement(number, outcome).unwrap();
    let InitWork::NativeRetry(session, number) = lifecycle.begin_init().unwrap() else {
        panic!("busy startup retry was not retained");
    };
    drop(observer);
    let outcome = ShutdownOutcome::retry(session);
    assert_eq!(outcome.status(), SUCCESS);
    lifecycle.finish_init_retirement(number, outcome).unwrap();
    assert!(matches!(
        lifecycle.begin_init().unwrap(),
        InitWork::Create(2)
    ));
    lifecycle.cancel_init(2, InitRollback::Complete).unwrap();
}

#[test]
fn abnormal_preparation_retains_ownership_but_timeout_remains_resumable() {
    let mut lifecycle = active(NativeSession::new(DriverContextLifetime::Session).unwrap());
    lifecycle.begin_shutdown().unwrap();
    lifecycle.pause_shutdown(1).unwrap();
    // An old executor's guard cannot consume a paused, safely owned checkpoint.
    lifecycle.retain_failed_transition(TransitionPhase::Preparing, 1);
    assert!(matches!(
        lifecycle.state,
        LifecycleState::PendingShutdown { .. }
    ));
    lifecycle.begin_shutdown().unwrap();
    lifecycle.retain_failed_transition(TransitionPhase::Preparing, 1);
    assert!(
        matches!(&lifecycle.state, LifecycleState::Retained { _session: session } if session.number == 1)
    );
    assert!(lifecycle.as_ref().is_none());
    assert_eq!(lifecycle.begin_init().err(), Some(INVALID_RUNTIME_STATE));
    assert_eq!(
        lifecycle.begin_shutdown().err(),
        Some(INVALID_RUNTIME_STATE)
    );
}

#[test]
fn uncertain_frontend_shutdown_retains_the_whole_generation() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let state = session_without_devices(session);
    state.control.health.quarantine();
    let outcome = state.stop();
    assert!(matches!(outcome, ShutdownOutcome::RetainedSession { .. }));
    assert_eq!(observer.state().unwrap(), SessionState::Retiring);
    let mut lifecycle = HsaLifecycle {
        last_generation: 1,
        state: LifecycleState::Stopping(1),
    };
    lifecycle.finish_shutdown(1, outcome).unwrap();
    assert!(matches!(lifecycle.state, LifecycleState::Retained { .. }));
    assert_eq!(lifecycle.begin_init().err(), Some(INVALID_RUNTIME_STATE));
    assert_eq!(
        lifecycle.begin_shutdown().err(),
        Some(INVALID_RUNTIME_STATE)
    );
}

#[test]
fn failed_native_shutdown_retains_session_until_retry_succeeds() {
    let session = NativeSession::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let outcome = session_without_devices(session).stop();
    assert_eq!(outcome.status(), RESOURCE_BUSY);
    assert_eq!(observer.state().unwrap(), SessionState::Retiring);
    let mut lifecycle = HsaLifecycle {
        last_generation: 1,
        state: LifecycleState::Stopping(1),
    };
    lifecycle.finish_shutdown(1, outcome).unwrap();
    assert_eq!(lifecycle.begin_init().err(), Some(INVALID_RUNTIME_STATE));
    let ShutdownWork::NativeRetry(session, number) = lifecycle.begin_shutdown().unwrap() else {
        panic!("native retry was not retained");
    };
    let outcome = ShutdownOutcome::retry(session);
    assert_eq!(outcome.status(), RESOURCE_BUSY);
    lifecycle.finish_shutdown(number, outcome).unwrap();
    let ShutdownWork::NativeRetry(session, number) = lifecycle.begin_shutdown().unwrap() else {
        panic!("busy native retry was not retained");
    };
    drop(observer);
    let outcome = ShutdownOutcome::retry(session);
    assert_eq!(outcome.status(), SUCCESS);
    lifecycle.finish_shutdown(number, outcome).unwrap();
    assert!(matches!(
        lifecycle.begin_init().unwrap(),
        InitWork::Create(2)
    ));
    lifecycle.cancel_init(2, InitRollback::Complete).unwrap();
}

#[test]
/// Verifies that initialization gate rejects overlapping generations.
fn initialization_gate_rejects_overlapping_generations() {
    let mut registry = HsaLifecycle::new();
    let first = registry.begin_init().unwrap().created().unwrap();
    assert_eq!(registry.begin_init().err().unwrap(), INVALID_RUNTIME_STATE);
    assert_eq!(registry.begin_shutdown().err(), Some(INVALID_RUNTIME_STATE));
    registry.cancel_init(first, InitRollback::Complete).unwrap();
    let second = registry.begin_init().unwrap().created().unwrap();
    assert_ne!(second, first);
    registry
        .cancel_init(second, InitRollback::Complete)
        .unwrap();
}

#[test]
fn concurrent_init_waits_for_the_first_generation_to_finish() {
    let lifecycle = Arc::new(Mutex::new(HsaLifecycle::new()));
    let changed = Arc::new(Condvar::new());
    let generation = lifecycle
        .lock()
        .unwrap()
        .begin_init()
        .unwrap()
        .created()
        .unwrap();
    let ready = Arc::new(std::sync::Barrier::new(2));
    let worker_lifecycle = lifecycle.clone();
    let worker_changed = changed.clone();
    let worker_ready = ready.clone();
    let worker = std::thread::spawn(move || {
        let guard = worker_lifecycle.lock().unwrap();
        worker_ready.wait();
        begin_init_waiting(guard, &worker_changed)
    });
    ready.wait();
    lifecycle
        .lock()
        .unwrap()
        .cancel_init_with_notification(generation, InitRollback::Complete, &changed)
        .unwrap();
    let next = worker.join().unwrap().unwrap().created().unwrap();
    assert_ne!(next, generation);
    lifecycle
        .lock()
        .unwrap()
        .cancel_init(next, InitRollback::Complete)
        .unwrap();
}

#[test]
fn callback_init_does_not_wait_for_preparing_shutdown() {
    let lifecycle = Mutex::new(HsaLifecycle::new());
    let generation = lifecycle
        .lock()
        .unwrap()
        .begin_init()
        .unwrap()
        .created()
        .unwrap();
    lifecycle
        .lock()
        .unwrap()
        .stop_without_runtime_for_test(generation);
    CallbackScope::run(|| {
        assert_eq!(
            begin_init_waiting(lifecycle.lock().unwrap(), &Condvar::new()).err(),
            Some(INVALID_RUNTIME_STATE)
        );
    });
}

#[test]
fn admitted_call_init_does_not_wait_for_preparing_shutdown() {
    let lifecycle = Mutex::new(HsaLifecycle::new());
    let generation = lifecycle
        .lock()
        .unwrap()
        .begin_init()
        .unwrap()
        .created()
        .unwrap();
    lifecycle
        .lock()
        .unwrap()
        .stop_without_runtime_for_test(generation);
    let gate = Arc::new(InFlightTracker::new());
    gate.close();
    gate.open(generation);
    ApiLeaseScope::run(gate.enter().unwrap(), || {
        assert_eq!(
            begin_init_waiting(lifecycle.lock().unwrap(), &Condvar::new()).err(),
            Some(INVALID_RUNTIME_STATE)
        );
    });
}

#[test]
fn same_thread_lifecycle_reentry_is_rejected() {
    let entry = LifecycleCallScope::enter().unwrap();
    assert_eq!(
        LifecycleCallScope::enter().err(),
        Some(INVALID_RUNTIME_STATE)
    );
    let helper = LifecycleCallScope::enter_helper();
    drop(helper);
    assert_eq!(
        LifecycleCallScope::enter().err(),
        Some(INVALID_RUNTIME_STATE)
    );
    drop(entry);
    assert!(LifecycleCallScope::enter().is_ok());
}

#[test]
/// Verifies that failed initialization cleanup quarantines the next generation.
fn failed_initialization_cleanup_quarantines_the_next_generation() {
    let mut registry = HsaLifecycle::new();
    let generation = registry.begin_init().unwrap().created().unwrap();
    registry
        .cancel_init(generation, InitRollback::Uncertain)
        .unwrap();
    assert_eq!(registry.begin_init().err().unwrap(), INVALID_RUNTIME_STATE);
    assert_eq!(registry.begin_shutdown().err(), Some(INVALID_RUNTIME_STATE));
}

#[test]
#[ignore = "requires a qualified GPU and KFD runtime"]
/// Verifies that HSA runtime reinitializes after shutdown.
fn hsa_runtime_reinitializes_after_shutdown() {
    for _ in 0..3 {
        assert_eq!(crate::hsa_init(), SUCCESS);
        assert_eq!(crate::hsa_shut_down(), SUCCESS);
    }
}
