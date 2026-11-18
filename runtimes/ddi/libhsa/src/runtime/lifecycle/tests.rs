// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Lifecycle ownership and resumable retirement regressions.

use super::*;
use rocddi::session::SessionState;
use rocddi::topology::host::{HostCaches, HostTopologyGraphSnapshot};

/// Builds a generation without discovery or native device acquisition.
fn state_without_devices(session: Session) -> HsaState {
    let health = Arc::new(GenerationHealth::new());
    let inflight = Arc::new(InFlightTracker::new_process());
    inflight.open(1);
    HsaState {
        debugger: None,
        sdma_enabled: false,
        peer_sdma_enabled: false,
        logging: Arc::new(LogOutput::new()),
        next_handle: 0x4853_4101_0000_0000,
        queue_state: QueueState::new(),
        loader: LoaderState::new(),
        image_state: ImageState::new(),
        memory_state: MemoryState::new(health.clone()),
        signal_state: SignalState::new(&session).unwrap(),
        pc_sampling_state: PcSamplingState::new(health.clone()),
        host_page_size: 4096,
        system_alloc_max_bytes: 0,
        cpus: Vec::new(),
        full_profile: true,
        disable_fragment_allocator: false,
        vm_fault_details: None,
        system_event_handlers: Vec::new(),
        system_event_worker: None,
        system_event_failed: Arc::new(AtomicBool::new(false)),
        stop_workers: Arc::new(AtomicBool::new(false)),
        cancel_work: Arc::new(AtomicBool::new(false)),
        inflight,
        health,
        caches: Vec::new(),
        topology: TopologyGraphSnapshot {
            host: HostTopologyGraphSnapshot {
                memory_bytes: None,
                cpu: None,
                cpu_apic_ids: Vec::new(),
                online_cpus: None,
                caches: HostCaches {
                    records: Vec::new(),
                    complete: true,
                },
                domains: Vec::new(),
                domains_complete: true,
            },
            endpoints: Vec::new(),
            sources: Vec::new(),
            nodes: Vec::new(),
            attachments: Vec::new(),
            routes: Vec::new(),
            caches: Vec::new(),
            memory_locations: Vec::new(),
            memory_banks: Vec::new(),
        },
        gpus: Vec::new(),
        session: Some(session),
    }
}

fn active(session: Session) -> HsaLifecycle {
    HsaLifecycle {
        last_generation: 1,
        state: LifecycleState::Active {
            generation: PublishedGeneration {
                number: 1,
                state: RetainOnDrop::new(state_without_devices(session)),
            },
            references: std::num::NonZeroU32::MIN,
        },
    }
}

#[test]
fn prepared_shutdown_closes_core_after_cleanup_services_finish() {
    let session = Session::new(DriverContextLifetime::Session).unwrap();
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
    let mut lifecycle = active(Session::new(DriverContextLifetime::Session).unwrap());
    let gate = lifecycle.as_ref().unwrap().inflight.clone();
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
    let session = Session::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let mut lifecycle = active(session);
    let gate = lifecycle.as_ref().unwrap().inflight.clone();
    let ordinary_stop = lifecycle.as_ref().unwrap().operation_stop();
    lifecycle.begin_shutdown().unwrap();
    assert!(ordinary_stop.load(std::sync::atomic::Ordering::Acquire));
    let (cleanup, cleanup_stop) = gate.begin_cleanup(1).unwrap().run(|| {
        (
            gate.enter_delegated_cleanup(1).unwrap(),
            lifecycle.as_ref().unwrap().operation_stop(),
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
    let session = Session::new(DriverContextLifetime::Session).unwrap();
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
    let mut lifecycle = active(Session::new(DriverContextLifetime::Session).unwrap());
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
        matches!(&lifecycle.state, LifecycleState::Retained { _generation: generation } if generation.number == 1)
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
    let session = Session::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let state = state_without_devices(session);
    state.health.quarantine();
    let outcome = state.stop();
    assert!(matches!(
        outcome,
        ShutdownOutcome::RetainedGeneration { .. }
    ));
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
    drop(RetainOnDrop::new(Owner(drops.clone())).into_inner());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn failed_native_shutdown_retains_session_until_retry_succeeds() {
    let session = Session::new(DriverContextLifetime::Session).unwrap();
    let observer = session.clone();
    let outcome = state_without_devices(session).stop();
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
