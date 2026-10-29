// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Lifecycle, event, clock, topology, and logging regression tests.

use super::*;
use rocddi::device::gpu::profiling::ClockCounters;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

#[test]
/// Verifies that precise clock sampling keeps the shortest complete sample.
fn precise_clock_sampling_keeps_the_shortest_complete_sample() {
    let mut samples = [
        Ok::<_, &str>((10, Duration::from_micros(40))),
        Ok((20, Duration::from_micros(3))),
        Ok((30, Duration::from_micros(12))),
        Ok((40, Duration::from_micros(1))),
    ]
    .into_iter();
    assert_eq!(best_clock_sample(4, || samples.next().unwrap()), Ok(40));

    let mut calls = 0;
    assert_eq!(
        best_clock_sample(4, || {
            calls += 1;
            if calls == 2 {
                Err("late ioctl failure")
            } else {
                Ok((17, Duration::from_micros(2)))
            }
        }),
        Ok(17)
    );
    assert_eq!(calls, 2);
}

#[test]
/// Verifies that native busy reports resource contention across frontends.
fn native_busy_reports_resource_contention_across_frontends() {
    let busy = || rocddi::Error::Operation {
        kind: rocddi::ErrorKind::Busy,
        detail: "injected live native dependency",
    };
    assert_eq!(map_error(busy()), RESOURCE_BUSY);
    assert_ne!(map_error(busy()), INVALID_QUEUE);
}

#[test]
#[allow(clippy::unwrap_used)]
/// Verifies that shutdown waits for inflight calls.
fn shutdown_waits_for_inflight_calls() {
    let inflight = Arc::new(InFlightTracker::new());
    let token = inflight.enter().unwrap();
    let (started_send, started_recv) = std::sync::mpsc::channel();
    let (done_send, done_recv) = std::sync::mpsc::channel();
    let worker_inflight = inflight.clone();
    let worker = thread::spawn(move || {
        started_send.send(()).unwrap();
        worker_inflight.wait_idle();
        done_send.send(()).unwrap();
    });
    started_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(done_recv.recv_timeout(Duration::from_millis(10)).is_err());
    drop(token);
    done_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.join().unwrap();
}

#[test]
/// Verifies that callback scope restricts queue and sampling teardown.
fn callback_scope_restricts_queue_and_sampling_teardown() {
    assert!(!CallbackScope::active());
    {
        let _scope = CallbackScope::enter();
        // SAFETY: Queue teardown is rejected before inspecting the
        // non-null pointer while a callback is active.
        assert_eq!(
            unsafe {
                crate::api::queue::core::hsa_queue_destroy(
                    std::ptr::NonNull::<HsaQueue>::dangling().as_ptr(),
                )
            },
            INVALID_RUNTIME_STATE
        );
        assert_eq!(
            crate::api::pc_sampling::hsa_ven_amd_pcs_stop(HsaPcSampling { handle: 1 }),
            INVALID_RUNTIME_STATE
        );
        let _nested = CallbackScope::enter();
        assert!(CallbackScope::active());
    }
    assert!(!CallbackScope::active());
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
/// Verifies that initialization gate rejects overlapping generations.
fn initialization_gate_rejects_overlapping_generations() {
    let mut registry = HsaLifecycle::new();
    let first = registry.begin_init().unwrap().unwrap();
    assert_eq!(registry.begin_init().unwrap_err(), INVALID_RUNTIME_STATE);
    assert_eq!(registry.begin_shutdown().err(), Some(INVALID_RUNTIME_STATE));
    registry.cancel_init(first, false).unwrap();
    let second = registry.begin_init().unwrap().unwrap();
    assert_ne!(second, first);
    registry.cancel_init(second, false).unwrap();
}

#[test]
fn concurrent_init_waits_for_the_first_generation_to_finish() {
    let lifecycle = Arc::new(Mutex::new(HsaLifecycle::new()));
    let changed = Arc::new(Condvar::new());
    let generation = lifecycle.lock().unwrap().begin_init().unwrap().unwrap();
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
        .cancel_init_with_notification(generation, false, &changed)
        .unwrap();
    let next = worker.join().unwrap().unwrap().unwrap();
    assert_ne!(next, generation);
    lifecycle.lock().unwrap().cancel_init(next, false).unwrap();
}

#[test]
fn callback_init_does_not_wait_for_preparing_shutdown() {
    let lifecycle = Mutex::new(HsaLifecycle::new());
    let generation = lifecycle.lock().unwrap().begin_init().unwrap().unwrap();
    lifecycle
        .lock()
        .unwrap()
        .prepare_without_runtime_for_test(generation);
    let _scope = CallbackScope::enter();
    assert_eq!(
        begin_init_waiting(lifecycle.lock().unwrap(), &Condvar::new()).err(),
        Some(INVALID_RUNTIME_STATE)
    );
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
    let generation = registry.begin_init().unwrap().unwrap();
    registry.cancel_init(generation, true).unwrap();
    assert_eq!(registry.begin_init().unwrap_err(), INVALID_RUNTIME_STATE);
}

#[test]
/// Verifies that first and later GPU activation failures retire the session
/// join.
fn first_and_later_gpu_activation_failures_retire_the_session_join() {
    /// Counts activated GPU owners in a rollback test.
    struct Activated {
        /// Number of mock GPU owners still live.
        live: Arc<AtomicUsize>,
    }
    impl Drop for Activated {
        /// Records destruction of the test owner.
        fn drop(&mut self) {
            self.live.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }
    /// Tracks retirement of a mock session.
    struct Joined {
        /// Number of GPU owners that must retire before the session.
        live: Arc<AtomicUsize>,
        /// Number of mock session joins still outstanding.
        joins: Arc<AtomicUsize>,
    }
    impl RetireSession for Joined {
        /// Checks native owners are gone before retiring the mock session.
        fn retire(&mut self) -> bool {
            assert_eq!(self.live.load(AtomicOrdering::Relaxed), 0);
            self.joins.fetch_sub(1, AtomicOrdering::Relaxed);
            true
        }
    }

    for completed_activations in [0, 1] {
        let live = Arc::new(AtomicUsize::new(completed_activations));
        let joins = Arc::new(AtomicUsize::new(1));
        let mut pending = PendingInit {
            session: Some(Joined {
                live: live.clone(),
                joins: joins.clone(),
            }),
            gpus: (0..completed_activations)
                .map(|_| Activated { live: live.clone() })
                .collect(),
        };
        assert!(pending.cleanup());
        assert_eq!(live.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(joins.load(AtomicOrdering::Relaxed), 0);
    }
}

/// Captures system-event callback observations.
struct ObservedEvent {
    /// Number of times the callback ran.
    count: AtomicU32,
    /// Last event type delivered to the callback.
    event_type: AtomicU32,
}

/// Records an event kind and callback count for assertions.
unsafe extern "C" fn observe_system_event(event: *const HsaAmdEvent, data: *mut c_void) -> Status {
    // SAFETY: The test passes live event and observation storage.
    let observed = unsafe { &*data.cast::<ObservedEvent>() };
    // SAFETY: The event pointer is live for this synchronous callback.
    let event = unsafe { &*event };
    observed.count.fetch_add(1, AtomicOrdering::Relaxed);
    observed
        .event_type
        .store(event.event_type, AtomicOrdering::Relaxed);
    ERROR
}

/// Tracks stop requests and subsequent event callbacks.
struct StopEventProbe {
    /// Stop request written by the first callback.
    stop: AtomicBool,
    /// Calls that reached a later callback after stop.
    later_calls: AtomicU32,
}

/// Requests worker stop from a system-event callback.
unsafe extern "C" fn stop_from_fault_handler(
    _event: *const HsaAmdEvent,
    data: *mut c_void,
) -> Status {
    // SAFETY: The test keeps the probe live until both callbacks finish.
    let probe = unsafe { &*data.cast::<StopEventProbe>() };
    probe.stop.store(true, AtomicOrdering::Release);
    ERROR
}

/// Records whether a later fault handler was invoked.
unsafe extern "C" fn count_later_fault_handler(
    _event: *const HsaAmdEvent,
    data: *mut c_void,
) -> Status {
    // SAFETY: The test keeps the probe live until both callbacks finish.
    let probe = unsafe { &*data.cast::<StopEventProbe>() };
    probe.later_calls.fetch_add(1, AtomicOrdering::Relaxed);
    SUCCESS
}

#[test]
/// Verifies that shutdown requested by fault handler skips later handlers.
fn shutdown_requested_by_fault_handler_skips_later_handlers() {
    let probe = StopEventProbe {
        stop: AtomicBool::new(false),
        later_calls: AtomicU32::new(0),
    };
    // SAFETY: The probe outlives synchronous event delivery.
    let data = unsafe { CallbackArg::new(std::ptr::from_ref(&probe).cast_mut().cast()) };
    let handlers = [
        (stop_from_fault_handler as SystemEventHandler, data),
        (count_later_fault_handler as SystemEventHandler, data),
    ];
    let event = HsaAmdEvent {
        event_type: AMD_GPU_MEMORY_FAULT_EVENT,
        payload: [0; 3],
    };

    assert!(!notify_system_event(&handlers, &event, Some(&probe.stop)));
    assert!(probe.stop.load(AtomicOrdering::Acquire));
    assert_eq!(probe.later_calls.load(AtomicOrdering::Relaxed), 0);
}

#[test]
/// Verifies that canonical GPU agent name uses the gfx target.
fn canonical_gpu_agent_name_uses_the_gfx_target() {
    assert_eq!(
        gpu_agent_name(&GpuInfo {
            gfx_major: 12,
            gfx_minor: 0,
            gfx_stepping: 1,
            ..GpuInfo::default()
        }),
        "gfx1201"
    );
}

#[test]
/// Verifies that HDP flush registers are adjacent words in the MMIO page.
fn hdp_flush_registers_are_adjacent_words_in_the_mmio_page() {
    assert_eq!(hdp_flush_pointers(None), [0, 0]);
    assert_eq!(hdp_flush_pointers(Some(0x1000)), [0x1000, 0x1004]);
}

#[test]
/// Verifies that zero local memory identifies a full profile platform.
fn zero_local_memory_identifies_a_full_profile_platform() {
    assert!(!full_profile_platform([]));
    assert!(full_profile_platform([0]));
    assert!(full_profile_platform([0, 0]));
    assert!(!full_profile_platform([0, 1]));
}

#[test]
/// Verifies that shutdown notifies every registered system event handler.
fn shutdown_notifies_every_registered_system_event_handler() {
    let first = ObservedEvent {
        count: AtomicU32::new(0),
        event_type: AtomicU32::new(u32::MAX),
    };
    let second = ObservedEvent {
        count: AtomicU32::new(0),
        event_type: AtomicU32::new(u32::MAX),
    };
    notify_system_shutdown(&[
        (
            observe_system_event,
            // SAFETY: The first probe outlives synchronous delivery.
            unsafe { CallbackArg::new(std::ptr::from_ref(&first).cast_mut().cast()) },
        ),
        (
            observe_system_event,
            // SAFETY: The second probe outlives synchronous delivery.
            unsafe { CallbackArg::new(std::ptr::from_ref(&second).cast_mut().cast()) },
        ),
    ]);
    assert_eq!(first.count.load(AtomicOrdering::Relaxed), 1);
    assert_eq!(second.count.load(AtomicOrdering::Relaxed), 1);
    assert_eq!(
        first.event_type.load(AtomicOrdering::Relaxed),
        AMD_SYSTEM_SHUTDOWN_EVENT
    );
    assert_eq!(
        second.event_type.load(AtomicOrdering::Relaxed),
        AMD_SYSTEM_SHUTDOWN_EVENT
    );
}

#[test]
/// Verifies that memory fault events match the AMD extension layout.
fn memory_fault_events_match_the_amd_extension_layout() {
    let agent = HsaAgent { handle: 0x1234 };
    for (cause, extra_reason) in [
        (GpuMemoryFaultCause::None, 0),
        (GpuMemoryFaultCause::SramEcc, AMD_MEMORY_FAULT_SRAM_ECC),
        (GpuMemoryFaultCause::DramEcc, AMD_MEMORY_FAULT_DRAM_ECC),
        (GpuMemoryFaultCause::Hang, AMD_MEMORY_FAULT_HANG),
    ] {
        let event = memory_fault_event(
            agent,
            GpuMemoryFault {
                endpoint_id: Some([42; 16]),
                virtual_address: 0x5678_9000,
                page_not_present: true,
                read_only: true,
                no_execute: true,
                imprecise: true,
                cause,
            },
        );
        assert_eq!(event.event_type, AMD_GPU_MEMORY_FAULT_EVENT);
        assert_eq!(event.payload[0], agent.handle);
        assert_eq!(event.payload[1], 0x5678_9000);
        assert_eq!(
            event.payload[2],
            u64::from(
                AMD_MEMORY_FAULT_PAGE_NOT_PRESENT
                    | AMD_MEMORY_FAULT_READ_ONLY
                    | AMD_MEMORY_FAULT_NO_EXECUTE
                    | AMD_MEMORY_FAULT_IMPRECISE
                    | extra_reason
            )
        );
    }
}

#[test]
/// Verifies that hardware exception events match the AMD extension layout.
fn hardware_exception_events_match_the_amd_extension_layout() {
    let agent = HsaAgent { handle: 0x1234 };
    for (cause, expected_cause) in [
        (GpuResetCause::Hang, AMD_HW_EXCEPTION_CAUSE_GPU_HANG),
        (GpuResetCause::Ecc, AMD_HW_EXCEPTION_CAUSE_ECC),
    ] {
        let event = hardware_exception_event(
            agent,
            GpuHardwareException {
                endpoint_id: Some([42; 16]),
                scope: rocddi::device::event::GpuResetScope::WholeGpu,
                memory_lost: false,
                cause,
            },
        );
        assert_eq!(event.event_type, AMD_GPU_HW_EXCEPTION_EVENT);
        assert_eq!(event.payload[0], agent.handle);
        assert_eq!(
            event.payload[1],
            u64::from(AMD_HW_EXCEPTION_RESET_TYPE_OTHER) | (u64::from(expected_cause) << 32)
        );
        assert_eq!(event.payload[2], 0);
    }
}

#[test]
/// Verifies that GPU ticks translate around the correlated system sample.
fn gpu_ticks_translate_around_the_correlated_system_sample() {
    let counters = ClockCounters {
        gpu: 100,
        host: 0,
        system: 1_000,
        system_frequency: 1_000_000_000,
        gpu_frequency: 100_000_000,
    };
    assert_eq!(translate_gpu_tick(counters, 110), Ok(1_100));
    assert_eq!(translate_gpu_tick(counters, 90), Ok(900));
    assert_eq!(
        translate_gpu_tick(
            ClockCounters {
                gpu_frequency: 200_000_000,
                ..counters
            },
            110,
        ),
        Ok(1_050)
    );
    assert_eq!(
        translate_gpu_tick(
            ClockCounters {
                system_frequency: 0,
                ..counters
            },
            100,
        ),
        Err(ERROR)
    );
}

#[test]
/// Verifies that logging flags use the public 64 bit mask layout.
fn logging_flags_use_the_public_64_bit_mask_layout() {
    let flags = [0b0000_0101, 0, 0, 0, 0, 0, 0, 0];
    assert!(logging_flag_enabled(flags, 0));
    assert!(!logging_flag_enabled(flags, 1));
    assert!(logging_flag_enabled(flags, AMD_LOG_FLAG_INFO));
    assert!(!logging_flag_enabled([u8::MAX; 8], 64));
}

#[test]
/// Verifies that GPU pool handle encoding preserves every pool class.
fn gpu_pool_handle_encoding_preserves_every_pool_class() {
    // Explicit slots pin the existing public handle encoding.
    for (slot, kind) in [
        (1, GpuPoolKind::Coarse),
        (2, GpuPoolKind::Fine),
        (3, GpuPoolKind::Group),
        (4, GpuPoolKind::ExtendedFine),
    ] {
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + slot,
                },
                1,
            ),
            Some((0, kind))
        );
        assert_eq!(HsaState::gpu_pool(0, kind).handle, GPU_POOL_BASE + slot);
    }
    assert_eq!(
        decode_gpu_pool_handle(
            HsaMemoryPool {
                handle: GPU_POOL_BASE + 0x11,
            },
            1,
        ),
        None
    );
}

#[test]
/// Verifies that cache handles and names are stable runtime objects.
fn cache_handles_and_names_are_stable_runtime_objects() {
    let agent = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let cache = Cache::new(agent, b"gfx1201\0ignored", 3, 8_388_608);
    assert_eq!(cache.agent.handle, agent.handle);
    assert_eq!(&*cache.name, b"gfx1201 L3\0");
    assert_eq!(cache.level, 3);
    assert_eq!(cache.size, 8_388_608);
    assert_eq!(
        decode_cache_handle(HsaCache { handle: CACHE_BASE }, 2),
        Some(0)
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE + 1,
            },
            2,
        ),
        Some(1)
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE + 2,
            },
            2,
        ),
        None
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE - 1,
            },
            2,
        ),
        None
    );
}

#[test]
/// Verifies that OS caches are not assigned to an unidentified CPU node.
fn cpu_cache_source_requires_a_unique_cpu_without_domain_evidence() {
    use rocddi::topology::host::{CpuCache, HostCaches, HostTopologySnapshot};

    let cache = |cpu| CpuCache {
        cpu,
        shared_cpus: None,
        info: rocddi::topology::CacheInfo::new(1, 32 * 1024, None, CpuCacheKind::Data),
    };
    let topology = HostTopologySnapshot {
        memory_bytes: None,
        cpu: None,
        cpu_apic_ids: Vec::new(),
        online_cpus: None,
        caches: HostCaches {
            records: vec![cache(7)],
            complete: true,
        },
        domains: Vec::new(),
        domains_complete: true,
    };
    assert_eq!(
        host_caches_for_cpu(&topology, None, 1).unwrap().records[0].cpu,
        7
    );
    assert!(host_caches_for_cpu(&topology, None, 2).is_none());
}

#[test]
/// Verifies separate KFD CPU nodes retain their bank sizes and APU policy.
fn cpu_agents_follow_kfd_cpu_nodes_and_system_banks() {
    use rocddi::topology::host::{HostCaches, HostTopologySnapshot};
    use rocddi::topology::{
        CpuNodeInfo, MemoryBank, MemoryBankKind, NativeNodeId, NodeId, NodeKind, SourceId,
        TopologyNode,
    };

    let host = CpuInfo {
        name: "test CPU".into(),
        compute_units: 12,
    };
    let topology = TopologySnapshot {
        host: HostTopologySnapshot {
            memory_bytes: Some(3 << 30),
            cpu: Some(host.clone()),
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
        nodes: [(2, NodeKind::Host, 8), (7, NodeKind::Gpu, 4)]
            .into_iter()
            .enumerate()
            .map(|(id, (native, kind, count))| TopologyNode {
                id: NodeId(id),
                source: SourceId(0),
                native_id: NativeNodeId::LinuxKfd(native),
                kind,
                cpu: Some(CpuNodeInfo {
                    core_count: count,
                    core_id_base: None,
                    maximum_clock_mhz: None,
                }),
            })
            .collect(),
        attachments: Vec::new(),
        routes: Vec::new(),
        caches: Vec::new(),
        memory_locations: Vec::new(),
        memory_banks: [1_u64 << 30, 2_u64 << 30]
            .into_iter()
            .enumerate()
            .map(|(node, capacity_bytes)| MemoryBank {
                node: NodeId(node),
                native_ordinal: 0,
                kind: MemoryBankKind::System,
                capacity_bytes,
                native_flags: 0,
                width_bits: None,
                maximum_clock_mhz: None,
            })
            .collect(),
    };
    let cpus = cpu_agents(&topology, &host, 3 << 30);
    assert_eq!(cpus.len(), 2);
    assert_eq!(
        (
            cpus[0].native_node,
            cpus[0].compute_units,
            cpus[0].memory_bytes
        ),
        (2, 8, 1 << 30)
    );
    assert_eq!(
        (
            cpus[1].native_node,
            cpus[1].compute_units,
            cpus[1].memory_bytes
        ),
        (7, 4, 2 << 30)
    );
    assert!(!cpus[0].apu);
    assert!(cpus[1].apu);
}

#[test]
/// Verifies that cache iteration sorts levels without changing handles.
fn cache_iteration_sorts_levels_without_changing_handles() {
    let agent = HsaAgent { handle: CPU_AGENT };
    let other = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let caches = [
        Cache::new(agent, b"CPU", 3, 100),
        Cache::new(agent, b"CPU", 1, 100),
        Cache::new(agent, b"CPU", 2, 100),
        Cache::new(agent, b"CPU", 1, 100),
        Cache::new(other, b"GPU", 1, 100),
    ];
    let handles = cache_handles_in_level_order(&caches, agent)
        .into_iter()
        .map(|cache| cache.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        handles,
        [CACHE_BASE + 1, CACHE_BASE + 3, CACHE_BASE + 2, CACHE_BASE]
    );
}

#[test]
/// Verifies that legacy GPU cache sizes match `ROCr` aggregation.
fn legacy_gpu_cache_sizes_match_rocr_aggregation() {
    let agent = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let other = HsaAgent {
        handle: GPU_AGENT_BASE + 1,
    };
    let caches = [
        Cache::new(agent, b"gfx1201", 1, 32 * 1024),
        Cache::new(agent, b"gfx1201", 1, 32 * 1024),
        Cache::new(agent, b"gfx1201", 2, 256 * 1024),
        Cache::new(agent, b"gfx1201", 2, 256 * 1024),
        Cache::new(agent, b"gfx1201", 3, 8192 * 1024),
        Cache::new(other, b"gfx1201", 3, 8192 * 1024),
    ];
    assert_eq!(
        cache_sizes(&caches, agent),
        [32 * 1024, 512 * 1024, 8192 * 1024, 0]
    );
    assert_eq!(cache_sizes(&caches, other), [0, 0, 8192 * 1024, 0]);
    assert_eq!(
        cpu_cache_sizes(&caches, agent),
        [32 * 1024, 256 * 1024, 8192 * 1024, 0]
    );
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
