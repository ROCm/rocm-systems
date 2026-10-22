// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;

fn batch_fixture(count: usize) -> (SignalState, HsaAmdSignalBatch, Vec<HsaSignal>) {
    let slots: Box<[AmdSignal]> = (0..count)
        .map(|index| AmdSignal::user(SignalValue::try_from(index).unwrap()))
        .collect();
    let signals = slots
        .iter()
        .map(|slot| HsaSignal {
            handle: std::ptr::from_ref(slot) as u64,
        })
        .collect::<Vec<_>>();
    let batch = HsaAmdSignalBatch { handle: 42 };
    let mut state = SignalState::new(DriverContextLifetime::Process);
    for signal in &signals {
        state
            .batch_members
            .insert(signal.handle as usize, batch.handle);
    }
    state.batches.insert(
        batch.handle,
        SignalBatchStorage {
            _backing: SignalBatchBacking::Test(slots),
            remaining: count,
            handle_live: true,
            host_atomics_supported: true,
            flush_range: None,
        },
    );
    (state, batch, signals)
}

#[test]
fn batch_storage_survives_both_destruction_orders() {
    for handle_first in [false, true] {
        let (mut state, batch, signals) = batch_fixture(3);
        if handle_first {
            assert_eq!(state.destroy_batch(batch), SUCCESS);
            assert_eq!(state.destroy_batch(batch), INVALID_ARGUMENT);
        }
        assert_eq!(state.destroy_signal(signals[0]), SUCCESS);
        assert_eq!(state.destroy_signal(signals[1]), SUCCESS);
        assert_eq!(state.destroy_signal(signals[0]), INVALID_SIGNAL);
        assert!(state.batches.contains_key(&batch.handle));
        assert!(state.owns_signal(signals[2]));
        if !handle_first {
            assert_eq!(state.destroy_batch(batch), SUCCESS);
        }
        assert_eq!(state.destroy_signal(signals[2]), SUCCESS);
        assert!(state.batches.is_empty());
        assert!(state.batch_members.is_empty());
    }
}

#[test]
fn batch_storage_waits_for_retired_async_member() {
    let (mut state, batch, signals) = batch_fixture(1);
    state.retain_async_signal(signals[0]).unwrap();
    assert_eq!(state.destroy_signal(signals[0]), SUCCESS);
    assert!(!state.owns_signal(signals[0]));
    assert_eq!(state.destroy_batch(batch), SUCCESS);
    assert!(state.batches.contains_key(&batch.handle));
    state.release_async_signal(signals[0]);
    assert!(state.batches.is_empty());
    assert!(state.batch_members.is_empty());
}

#[test]
fn device_write_readback_does_not_restore_an_intervening_update() {
    let value = AtomicI64::new(1);
    // Model a GPU decrement after the host's store but before its flush.
    value.fetch_sub(1, Ordering::AcqRel);
    flush_device_write(&value);
    assert_eq!(value.load(Ordering::Acquire), 0);
}

#[test]
fn posted_store_batch_range_tracks_member_lifetime() {
    let (mut state, batch, signals) = batch_fixture(1);
    let signal = signals[0];
    assert_eq!(state.async_completion_status(signal), SUCCESS);
    let range = (
        signal.handle as usize,
        signal.handle as usize + SIGNAL_BYTES,
    );
    flush_batches().push(range);
    HAS_FLUSH_BATCHES.store(true, Ordering::Release);
    let storage = state.batches.get_mut(&batch.handle).unwrap();
    storage.flush_range = Some(range);
    storage.host_atomics_supported = false;
    assert_eq!(state.async_completion_status(signal), NOT_SUPPORTED);
    assert!(batch_needs_flush(range.0));
    // SAFETY: The fixture retains the aligned signal through all operations.
    unsafe {
        hsa_signal_store_relaxed(signal, 5);
        assert_eq!(hsa_signal_exchange_relaxed(signal, 8), 5);
        hsa_signal_add_relaxed(signal, 2);
        assert_eq!(hsa_signal_load_relaxed(signal), 10);
    }
    assert_eq!(state.destroy_signal(signal), SUCCESS);
    assert_eq!(state.async_completion_status(signal), INVALID_SIGNAL);
    assert!(batch_needs_flush(range.0));
    assert_eq!(state.destroy_batch(batch), SUCCESS);
    assert!(!batch_needs_flush(range.0));
}

unsafe extern "C" fn count_rearmed_handler(_value: SignalValue, arg: *mut c_void) -> bool {
    // SAFETY: The test passes a live AtomicU64 for the worker lifetime.
    let calls = unsafe { &*(arg.cast::<AtomicU64>()) };
    calls.fetch_add(1, Ordering::Relaxed) < 2
}

struct AsyncSerializationProbe {
    active: AtomicU64,
    maximum_active: AtomicU64,
    completed: AtomicU64,
}

unsafe extern "C" fn serialized_async_function(arg: *mut c_void) {
    // SAFETY: The test passes a live probe for the dispatcher lifetime.
    let probe = unsafe { &*(arg.cast::<AsyncSerializationProbe>()) };
    let active = probe.active.fetch_add(1, Ordering::AcqRel) + 1;
    probe.maximum_active.fetch_max(active, Ordering::AcqRel);
    thread::sleep(Duration::from_millis(2));
    probe.active.fetch_sub(1, Ordering::AcqRel);
    probe.completed.fetch_add(1, Ordering::Release);
}

struct StopDispatcherProbe {
    stop: std::sync::atomic::AtomicBool,
    later_calls: AtomicU64,
}

unsafe extern "C" fn stop_dispatcher_from_callback(arg: *mut c_void) {
    // SAFETY: The probe remains live until the dispatcher returns.
    let probe = unsafe { &*(arg.cast::<StopDispatcherProbe>()) };
    probe.stop.store(true, Ordering::Release);
}

unsafe extern "C" fn count_later_callback(arg: *mut c_void) {
    // SAFETY: The probe remains live until the dispatcher returns.
    let probe = unsafe { &*(arg.cast::<StopDispatcherProbe>()) };
    probe.later_calls.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn dispatcher_does_not_invoke_queued_callbacks_after_callback_requests_stop() {
    let probe = StopDispatcherProbe {
        stop: std::sync::atomic::AtomicBool::new(false),
        later_calls: AtomicU64::new(0),
    };
    // SAFETY: The probe outlives the dispatcher and uses atomic fields.
    let argument = unsafe { CallbackArg::new((&raw const probe).cast_mut().cast()) };
    let (sender, receiver) = std::sync::mpsc::channel();
    sender
        .send(AsyncRequest::Function {
            callback: stop_dispatcher_from_callback,
            arg: argument,
        })
        .unwrap();
    sender
        .send(AsyncRequest::Function {
            callback: count_later_callback,
            arg: argument,
        })
        .unwrap();
    drop(sender);

    run_async_dispatcher(&receiver, &probe.stop);

    assert!(probe.stop.load(Ordering::Acquire));
    assert_eq!(probe.later_calls.load(Ordering::Relaxed), 0);
}

unsafe extern "C" fn shutdown_from_async_callback(arg: *mut c_void) {
    // SAFETY: The test retains this atomic until the dispatcher exits.
    let status = unsafe { &*(arg.cast::<AtomicU64>()) };
    status.store(u64::from(crate::hsa_shut_down()), Ordering::Release);
}

#[test]
fn async_callback_without_runtime_reports_not_initialized() {
    let status = AtomicU64::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    sender
        .send(AsyncRequest::Function {
            callback: shutdown_from_async_callback,
            // SAFETY: The atomic remains live until dispatch returns.
            arg: unsafe { CallbackArg::new((&raw const status).cast_mut().cast()) },
        })
        .unwrap();
    drop(sender);
    let stop = std::sync::atomic::AtomicBool::new(false);
    run_async_dispatcher(&receiver, &stop);
    assert_eq!(status.load(Ordering::Acquire), u64::from(NOT_INITIALIZED));
}

struct ShutdownSignalProbe {
    status: AtomicU64,
    stop: std::sync::atomic::AtomicBool,
}

unsafe extern "C" fn shutdown_from_signal_handler(_value: SignalValue, arg: *mut c_void) -> bool {
    // SAFETY: The test retains the probe until the dispatcher joins.
    let probe = unsafe { &*(arg.cast::<ShutdownSignalProbe>()) };
    probe
        .status
        .store(u64::from(crate::hsa_shut_down()), Ordering::Release);
    probe.stop.store(true, Ordering::Release);
    false
}

#[test]
fn async_signal_handler_without_runtime_reports_not_initialized() {
    let storage = AmdSignal::user(1);
    let signal = HsaSignal {
        handle: (&raw const storage) as u64,
    };
    let probe = std::sync::Arc::new(ShutdownSignalProbe {
        status: AtomicU64::new(0),
        stop: std::sync::atomic::AtomicBool::new(false),
    });
    let (sender, receiver) = std::sync::mpsc::channel();
    sender
        .send(AsyncRequest::Signal {
            signal,
            condition: SIGNAL_CONDITION_EQ,
            compare_value: 1,
            handler: shutdown_from_signal_handler,
            // SAFETY: The Arc remains live until the dispatcher joins.
            arg: unsafe { CallbackArg::new(std::sync::Arc::as_ptr(&probe).cast_mut().cast()) },
        })
        .unwrap();
    let worker_probe = probe.clone();
    let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_probe.stop));
    worker.join().unwrap();
    drop(sender);
    assert_eq!(
        probe.status.load(Ordering::Acquire),
        u64::from(NOT_INITIALIZED)
    );
}

#[test]
fn async_handler_rearms_while_condition_remains_satisfied() {
    let storage = AmdSignal::user(1);
    let signal = HsaSignal {
        handle: (&raw const storage) as u64,
    };
    let calls = AtomicU64::new(0);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (sender, receiver) = std::sync::mpsc::channel();
    sender
        .send(AsyncRequest::Signal {
            signal,
            condition: SIGNAL_CONDITION_EQ,
            compare_value: 1,
            handler: count_rearmed_handler,
            // SAFETY: The atomic remains live until the dispatcher joins.
            arg: unsafe { CallbackArg::new((&raw const calls).cast_mut().cast()) },
        })
        .unwrap();
    let worker_stop = stop.clone();
    let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_stop));

    let timeout = if cfg!(miri) {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(100)
    };
    let deadline = std::time::Instant::now() + timeout;
    while calls.load(Ordering::Relaxed) < 3 && std::time::Instant::now() < deadline {
        thread::yield_now();
    }
    stop.store(true, Ordering::Release);
    worker.join().unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn async_callbacks_execute_serially() {
    let probe = AsyncSerializationProbe {
        active: AtomicU64::new(0),
        maximum_active: AtomicU64::new(0),
        completed: AtomicU64::new(0),
    };
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (sender, receiver) = std::sync::mpsc::channel();
    // SAFETY: The probe outlives the dispatcher and uses atomic fields.
    let argument = unsafe { CallbackArg::new((&raw const probe).cast_mut().cast()) };
    for _ in 0..2 {
        sender
            .send(AsyncRequest::Function {
                callback: serialized_async_function,
                arg: argument,
            })
            .unwrap();
    }
    let worker_stop = stop.clone();
    let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_stop));

    let timeout = if cfg!(miri) {
        Duration::from_secs(5)
    } else {
        Duration::from_millis(100)
    };
    let deadline = std::time::Instant::now() + timeout;
    while probe.completed.load(Ordering::Acquire) < 2 && std::time::Instant::now() < deadline {
        thread::yield_now();
    }
    stop.store(true, Ordering::Release);
    worker.join().unwrap();

    assert_eq!(probe.completed.load(Ordering::Acquire), 2);
    assert_eq!(probe.maximum_active.load(Ordering::Acquire), 1);
}

#[test]
fn lock_free_signal_lookup_observes_atomic_invalidation() {
    let storage = std::sync::Arc::new(AmdSignal::user(7));
    let handle = HsaSignal {
        handle: std::sync::Arc::as_ptr(&storage) as u64,
    };
    let observed_live = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader_storage = storage.clone();
    let reader_observed = observed_live.clone();
    let reader = thread::spawn(move || {
        loop {
            // SAFETY: Both threads retain the signal allocation until join.
            if unsafe { signal_ptr(handle) }.is_none() {
                break;
            }
            reader_observed.store(true, Ordering::Release);
            thread::yield_now();
        }
        drop(reader_storage);
    });
    while !observed_live.load(Ordering::Acquire) {
        thread::yield_now();
    }
    storage
        .kind
        .store(AMD_SIGNAL_KIND_INVALID, Ordering::Release);
    reader.join().unwrap();
    // SAFETY: The allocation remains live through this final lookup.
    assert!(unsafe { signal_ptr(handle) }.is_none());
}

#[test]
fn amd_signal_helpers_match_the_public_abi() {
    let _: unsafe extern "C" fn(HsaSignal, *mut u32) -> Status = hsa_amd_signal_get_event_id;
    let _: unsafe extern "C" fn(AsyncFunction, *mut c_void) -> Status = hsa_amd_async_function;
    let _: unsafe extern "C" fn(HsaSignal, *mut HsaAmdIpcSignal) -> Status =
        hsa_amd_ipc_signal_create;
    let _: unsafe extern "C" fn(*const HsaAmdIpcSignal, *mut HsaSignal) -> Status =
        hsa_amd_ipc_signal_attach;
}

#[test]
fn signal_consumers_reject_duplicates() {
    let consumers = [HsaAgent { handle: 1 }, HsaAgent { handle: 1 }];
    // SAFETY: The array contains the declared number of readable handles.
    assert!(!unsafe { valid_consumers(consumers.len() as u32, consumers.as_ptr()) });
    // SAFETY: A zero count permits a null consumer pointer.
    assert!(unsafe { valid_consumers(0, std::ptr::null()) });
    assert!(amd_signal_uses_consumer_list(0));
    assert!(!amd_signal_uses_consumer_list(AMD_SIGNAL_AMD_GPU_ONLY));
    assert!(!amd_signal_uses_consumer_list(AMD_SIGNAL_IPC));
    assert!(!amd_signal_uses_consumer_list(
        AMD_SIGNAL_AMD_GPU_ONLY | AMD_SIGNAL_IPC
    ));
}

#[test]
fn signal_consumers_select_rocr_signal_types() {
    let cpu = [HsaAgent { handle: CPU_AGENT }];
    let gpu = [HsaAgent {
        handle: GPU_AGENT_BASE,
    }];
    // SAFETY: Each nonzero count names a complete readable array.
    unsafe {
        assert_eq!(signal_uses_interrupt(0, std::ptr::null(), 0), Ok(true));
        assert_eq!(signal_uses_interrupt(1, cpu.as_ptr(), 0), Ok(true));
        assert_eq!(signal_uses_interrupt(1, gpu.as_ptr(), 0), Ok(false));
        assert_eq!(
            signal_uses_interrupt(1, std::ptr::null(), AMD_SIGNAL_AMD_GPU_ONLY),
            Ok(false)
        );
        assert_eq!(
            signal_uses_interrupt(1, std::ptr::null(), AMD_SIGNAL_IPC),
            Ok(false)
        );
    }
}

#[test]
fn interrupt_signal_exposes_the_kfd_mailbox_layout() {
    let signal = AmdSignal::interrupt(7, 0x1234_5000, 19);
    assert_eq!(signal.kind.load(Ordering::Relaxed), AMD_SIGNAL_KIND_USER);
    assert_eq!(signal.value.load(Ordering::Relaxed), 7);
    assert_eq!(signal.event_mailbox_ptr, 0x1234_5000);
    assert_eq!(signal.event_id, 19);
}

#[test]
fn value_pointer_requires_a_busy_wait_signal() {
    let user = AmdSignal::user(7);
    let doorbell = AmdSignal::doorbell(0x1000, 0x2000);
    assert_eq!(
        signal_value_pointer(&user, false),
        Ok((&raw const user.value).cast::<SignalValue>().cast_mut())
    );
    assert_eq!(signal_value_pointer(&user, true), Err(INVALID_ARGUMENT));
    assert_eq!(
        signal_value_pointer(&doorbell, false),
        Err(INVALID_ARGUMENT)
    );
}

#[test]
fn signal_group_entry_points_match_the_public_abi() {
    let _: unsafe extern "C" fn(
        u32,
        *const HsaSignal,
        u32,
        *const HsaAgent,
        *mut HsaSignalGroup,
    ) -> Status = hsa_signal_group_create;
    let _: extern "C" fn(HsaSignalGroup) -> Status = hsa_signal_group_destroy;
    let _: unsafe extern "C" fn(
        HsaSignalGroup,
        *const u32,
        *const SignalValue,
        u32,
        *mut HsaSignal,
        *mut SignalValue,
    ) -> Status = hsa_signal_group_wait_any_relaxed;
    let _: unsafe extern "C" fn(
        HsaSignalGroup,
        *const u32,
        *const SignalValue,
        u32,
        *mut HsaSignal,
        *mut SignalValue,
    ) -> Status = hsa_signal_group_wait_any_scacquire;
}

#[test]
fn base_atomic_entry_points_preserve_signal_values() {
    let storage = AmdSignal::user(7);
    let signal = HsaSignal {
        handle: (&raw const storage) as u64,
    };

    // SAFETY: The handle points to live, aligned user-signal storage for
    // the duration of every call below.
    unsafe {
        assert_eq!(hsa_signal_load_acquire(signal), 7);
        hsa_signal_store_release(signal, 11);
        assert_eq!(hsa_signal_exchange_relaxed(signal, 13), 11);

        assert_eq!(hsa_signal_cas_scacq_screl(signal, 12, 17), 13);
        assert_eq!(hsa_signal_load_relaxed(signal), 13);
        assert_eq!(hsa_signal_cas_acq_rel(signal, 13, 17), 13);

        hsa_signal_add_relaxed(signal, 5);
        hsa_signal_subtract_acquire(signal, 2);
        hsa_signal_and_screlease(signal, 0xf);
        hsa_signal_or_acq_rel(signal, 0x20);
        hsa_signal_xor_release(signal, 0x3);
        assert_eq!(hsa_signal_load_scacquire(signal), 0x27);

        hsa_signal_silent_store_screlease(signal, 41);
        assert_eq!(
            hsa_signal_wait_relaxed(signal, SIGNAL_CONDITION_EQ, 41, 0, 0),
            41
        );
        assert_eq!(
            hsa_signal_wait_acquire(signal, SIGNAL_CONDITION_GTE, 40, 0, 0),
            41
        );
    }
}

#[test]
fn wait_timeout_uses_system_timestamp_ticks() {
    let frequency = 100_000_000;
    assert!(!timeout_elapsed(Duration::from_nanos(9), 1, frequency));
    assert!(timeout_elapsed(Duration::from_nanos(10), 1, frequency));
    assert!(!timeout_elapsed(
        Duration::from_millis(999),
        frequency,
        frequency
    ));
    assert!(timeout_elapsed(
        Duration::from_secs(1),
        frequency,
        frequency
    ));
    assert!(!timeout_elapsed(
        Duration::from_secs(1),
        u64::MAX,
        frequency
    ));
}

#[test]
fn wait_many_reports_original_indices_and_satisfying_values() {
    let first = AmdSignal::user(3);
    let second = AmdSignal::user(8);
    let signals = [
        HsaSignal { handle: 0 },
        HsaSignal { handle: 1 },
        HsaSignal {
            handle: (&raw const first) as u64,
        },
        HsaSignal {
            handle: (&raw const second) as u64,
        },
    ];
    let conditions = [
        SIGNAL_CONDITION_EQ,
        SIGNAL_CONDITION_EQ,
        SIGNAL_CONDITION_GTE,
        SIGNAL_CONDITION_LT,
    ];
    let values = [0, 0, 3, 9];
    let mut satisfying = 0;
    let stop = AtomicBool::new(false);

    // SAFETY: The validated handles point to live, aligned signal storage
    // for the duration of each wait; the invalid handle is never read.
    unsafe {
        let mut valid = collect_wait_signals(&signals, |signal| {
            signal.handle == signals[2].handle || signal.handle == signals[3].handle
        });
        assert_eq!(
            signal_wait_any_open(
                &valid,
                &conditions,
                &values,
                0,
                0,
                &raw mut satisfying,
                &stop
            ),
            2
        );
        assert_eq!(satisfying, 3);

        let mut satisfying_all = [-1; 4];
        assert_eq!(
            signal_wait_all_open(
                &mut valid,
                &conditions,
                &values,
                0,
                0,
                satisfying_all.as_mut_ptr(),
                &stop,
            ),
            0
        );
        assert_eq!(satisfying_all, [0, 0, 3, 8]);
    }
}

#[test]
fn wait_all_zeroes_unsatisfied_values_on_timeout() {
    let storage = AmdSignal::user(3);
    let signals = [HsaSignal {
        handle: (&raw const storage) as u64,
    }];
    let conditions = [SIGNAL_CONDITION_EQ];
    let values = [4];
    let mut satisfying = [-1];
    let stop = AtomicBool::new(false);

    // SAFETY: The validated signal storage remains live for the wait.
    unsafe {
        let mut valid = collect_wait_signals(&signals, |_| true);
        assert_eq!(
            signal_wait_all_open(
                &mut valid,
                &conditions,
                &values,
                0,
                0,
                satisfying.as_mut_ptr(),
                &stop,
            ),
            u32::MAX
        );
    }
    assert_eq!(satisfying, [0]);
}

#[test]
fn wait_all_accepts_comparison_values_as_output_storage() -> Result<(), &'static str> {
    let storage = AmdSignal::user(7);
    let signals = [HsaSignal {
        handle: (&raw const storage) as u64,
    }];
    let conditions = [SIGNAL_CONDITION_EQ];
    let mut values = [7];
    let stop = AtomicBool::new(false);

    // SAFETY: The signal storage and input arrays remain live through the
    // copy. The output aliases the comparison array only after the copy.
    unsafe {
        let inputs = signal_wait_inputs(1, signals.as_ptr(), conditions.as_ptr(), values.as_ptr())
            .ok_or("valid wait inputs")?;
        let mut entries = collect_wait_signals(&inputs.signals, |_| true);
        assert_eq!(
            signal_wait_all_open(
                &mut entries,
                &inputs.conditions,
                &inputs.values,
                0,
                0,
                values.as_mut_ptr(),
                &stop,
            ),
            0
        );
    }
    assert_eq!(values, [7]);
    Ok(())
}

#[test]
fn shutdown_cancels_unbounded_multi_signal_waits() {
    let storage = AmdSignal::user(3);
    let handles = [HsaSignal {
        handle: (&raw const storage) as u64,
    }];
    let conditions = [SIGNAL_CONDITION_EQ];
    let values = [4];
    let stop = AtomicBool::new(true);
    // SAFETY: The accepted handle stays live through both calls.
    let mut signals = unsafe { collect_wait_signals(&handles, |_| true) };
    // SAFETY: Both calls receive valid slices and no output pointer.
    unsafe {
        assert_eq!(
            signal_wait_any_open(
                &signals,
                &conditions,
                &values,
                u64::MAX,
                0,
                std::ptr::null_mut(),
                &stop,
            ),
            u32::MAX
        );
        assert_eq!(
            signal_wait_all_open(
                &mut signals,
                &conditions,
                &values,
                u64::MAX,
                0,
                std::ptr::null_mut(),
                &stop,
            ),
            u32::MAX
        );
    }
}

#[test]
#[allow(clippy::unwrap_used)]
fn retired_signal_ends_multi_signal_waits() {
    let retired = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let worker_retired = retired.clone();
    let worker_stop = stop.clone();
    let (ready_send, ready_recv) = std::sync::mpsc::channel();
    let (result_send, result_recv) = std::sync::mpsc::channel();
    let worker = thread::spawn(move || {
        let storage = AmdSignal::user(3);
        let handle = HsaSignal {
            handle: (&raw const storage) as u64,
        };
        // SAFETY: This worker owns the signal storage until the wait exits.
        let mut entries = unsafe { collect_wait_signals(&[handle], |_| true) };
        entries[0].retired = Some(worker_retired);
        ready_send.send(()).unwrap();
        let conditions = [SIGNAL_CONDITION_EQ];
        let values = [4];
        // SAFETY: The signal and both input slices remain live for the wait.
        let result = unsafe {
            signal_wait_any_open(
                &entries,
                &conditions,
                &values,
                u64::MAX,
                0,
                std::ptr::null_mut(),
                &worker_stop,
            )
        };
        result_send.send(result).unwrap();
    });
    ready_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    retired.store(true, Ordering::Release);
    let result = result_recv.recv_timeout(Duration::from_secs(1));
    if result.is_err() {
        stop.store(true, Ordering::Release);
    }
    worker.join().unwrap();
    assert!(matches!(result, Ok(value) if value == u32::MAX));

    stop.store(false, Ordering::Release);
    let storage = AmdSignal::user(3);
    let handle = HsaSignal {
        handle: (&raw const storage) as u64,
    };
    // SAFETY: The signal storage remains live for this immediate wait.
    let mut entries = unsafe { collect_wait_signals(&[handle], |_| true) };
    entries[0].retired = Some(retired);
    let conditions = [SIGNAL_CONDITION_EQ];
    let values = [4];
    // SAFETY: Both input slices remain live for this immediate wait.
    assert_eq!(
        unsafe {
            signal_wait_all_open(
                &mut entries,
                &conditions,
                &values,
                u64::MAX,
                0,
                std::ptr::null_mut(),
                &stop,
            )
        },
        u32::MAX
    );
}

#[test]
fn ipc_reattach_does_not_clear_a_prior_waiters_retirement() {
    let mut record = AsyncSignalRecord {
        references: 1,
        retired: Arc::new(AtomicBool::new(false)),
        copy_profile: None,
    };
    let prior_waiter = record.retired.clone();
    record.retired.store(true, Ordering::Release);
    record.revive();
    assert!(prior_waiter.load(Ordering::Acquire));
    assert!(!record.retired.load(Ordering::Acquire));
    assert!(!Arc::ptr_eq(&prior_waiter, &record.retired));

    let current_waiter = record.retired.clone();
    record.revive();
    assert!(Arc::ptr_eq(&current_waiter, &record.retired));
}

#[test]
fn amd_multi_waits_require_runtime_initialization() {
    let signal = AmdSignal::user(3);
    let signals = [HsaSignal {
        handle: (&raw const signal) as u64,
    }];
    let conditions = [SIGNAL_CONDITION_EQ];
    let values = [3];
    let mut satisfying = -1;

    // SAFETY: All arrays contain one readable/writable entry and the signal
    // storage stays live. The runtime is intentionally not initialized.
    unsafe {
        assert_eq!(
            hsa_amd_signal_wait_any(
                1,
                signals.as_ptr(),
                conditions.as_ptr(),
                values.as_ptr(),
                0,
                0,
                &raw mut satisfying,
            ),
            u32::MAX
        );
        assert_eq!(satisfying, -1);
        assert_eq!(
            hsa_amd_signal_wait_all(
                1,
                signals.as_ptr(),
                conditions.as_ptr(),
                values.as_ptr(),
                0,
                0,
                &raw mut satisfying,
            ),
            u32::MAX
        );
        assert_eq!(satisfying, -1);
    }
}

#[test]
#[ignore = "requires a qualified GPU and KFD runtime"]
fn destroyed_signal_storage_is_reused() {
    assert_eq!(crate::hsa_init(), SUCCESS);
    let mut signal = HsaSignal { handle: 0 };
    let mut previous = 0;
    for value in 0..10_000 {
        // SAFETY: The test supplies writable output storage and owns each
        // returned signal until destruction below.
        assert_eq!(
            unsafe { hsa_signal_create(value, 0, std::ptr::null(), &raw mut signal) },
            SUCCESS
        );
        assert_eq!(unsafe { hsa_signal_load_relaxed(signal) }, value);
        if value != 0 {
            assert_eq!(signal.handle, previous);
        }
        previous = signal.handle;
        // SAFETY: No other work retains this signal.
        assert_eq!(unsafe { hsa_signal_destroy(signal) }, SUCCESS);
    }
    assert_eq!(crate::hsa_shut_down(), SUCCESS);
}
