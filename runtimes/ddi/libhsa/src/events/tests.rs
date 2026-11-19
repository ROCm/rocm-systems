// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

#[test]
/// Verifies that callback scope restricts queue and sampling teardown.
fn callback_scope_restricts_queue_and_sampling_teardown() {
    assert!(!CallbackScope::active());
    CallbackScope::run(|| {
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
        CallbackScope::run(|| assert!(CallbackScope::active()));
        assert!(CallbackScope::active());
    });
    assert!(!CallbackScope::active());
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
                origin_id: None,
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
