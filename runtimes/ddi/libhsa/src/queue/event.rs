// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native queue event delivery and scratch fault recovery.

use super::*;
use rocddi::device::gpu::queue::{
    AqlQueueException, aql_scratch_lanes, decode_aql_queue_exception, replace_scratch_owner,
};

/// Queue error callback detached from the registry lock before invocation.
struct PendingCallback {
    /// Application callback to invoke synchronously.
    callback: unsafe extern "C" fn(Status, *mut HsaQueue, *mut c_void),
    /// HSA status derived from native error details.
    status: Status,
    /// Public queue pointer passed to the callback.
    queue: usize,
    /// Caller-owned callback argument.
    data: CallbackArg,
}

/// Result of processing one queue event before native rearming.
struct QueueEventOutcome {
    /// Callback to run after leaving the registry lock, if any.
    callback: Option<PendingCallback>,
    /// Whether the native event should be armed for another notification.
    rearm: bool,
}

/// Maps a native queue error code to an HSA status.
pub(super) fn queue_error_status(error: u64) -> Status {
    match decode_aql_queue_exception(error) {
        AqlQueueException::Scratch { .. } | AqlQueueException::Unknown => ERROR,
        AqlQueueException::IncompatibleArguments => INCOMPATIBLE_ARGUMENTS,
        AqlQueueException::InvalidAllocation => INVALID_ALLOCATION,
        AqlQueueException::InvalidCodeObject => INVALID_CODE_OBJECT,
        AqlQueueException::MemoryFault => MEMORY_FAULT,
        AqlQueueException::InvalidPacketFormat => INVALID_PACKET_FORMAT,
        AqlQueueException::InvalidArgument => INVALID_ARGUMENT,
        AqlQueueException::OutOfRegisters => OUT_OF_REGISTERS,
        AqlQueueException::MemoryApertureViolation => MEMORY_APERTURE_VIOLATION,
        AqlQueueException::IllegalInstruction => ILLEGAL_INSTRUCTION,
        AqlQueueException::Exception => EXCEPTION,
    }
}

/// Prepares a callback for a queue event without holding the registry lock.
fn pending_callback(
    runtime: &crate::runtime::HsaState,
    key: usize,
    status: Status,
) -> Option<PendingCallback> {
    let queue = runtime.queue_state.queues.get(&key)?;
    Some(PendingCallback {
        callback: queue.callback?,
        status,
        queue: queue.public_handle,
        data: queue.callback_data,
    })
}

/// Prepares the user error callback after a queue fault.
fn pending_error_callback(
    runtime: &mut crate::runtime::HsaState,
    key: usize,
    status: Status,
) -> Option<PendingCallback> {
    if status == MEMORY_FAULT {
        // Publish the affected queue so a running system event worker can
        // attach the process fault's address and reason.
        let details = runtime.vm_fault_details;
        let defer_to_system = runtime.system_event_worker.is_some()
            && !runtime.system_event_failed.load(Ordering::Acquire);
        let queue = runtime.queue_state.queues.get_mut(&key)?;
        queue.vm_faulted = true;
        if let Some((_, address, reason)) = details.filter(|(agent, _, _)| *agent == queue.agent) {
            queue.vm_fault_address = address;
            queue.vm_fault_reason = reason;
        }
        match queue.vm_fault_callback_state.observe(defer_to_system) {
            VmFaultCallbackAction::Defer => {
                VM_FAULT_CONDVAR.notify_all();
                return None;
            }
            VmFaultCallbackAction::Ignore => return None,
            VmFaultCallbackAction::Callback => {}
        }
        // Without a system event handler, the queue callback is the only
        // fault notification available to the application.
    }
    pending_callback(runtime, key, status)
}

/// Translates a native event observation into queue callback work.
fn handle_queue_event(key: usize, observed: SignalValue) -> Option<QueueEventOutcome> {
    let mut guard = lock().ok()?;
    let runtime = guard.as_mut()?;
    if !runtime.queue_state.queues.contains_key(&key) {
        return None;
    }
    let error = observed as u64;
    let Some(lanes_per_wave) = aql_scratch_lanes(error) else {
        return Some(QueueEventOutcome {
            callback: pending_error_callback(runtime, key, queue_error_status(error)),
            rearm: false,
        });
    };
    let (device, gpu, public, private_segment_size, dispatch_id) = {
        let queue = runtime.queue_state.queues.get(&key)?;
        let index = runtime.gpu_index(queue.agent)?;
        let Some((private_segment_size, dispatch_id)) = dispatch_private_segment(queue) else {
            return Some(QueueEventOutcome {
                callback: pending_callback(runtime, key, ERROR),
                rearm: false,
            });
        };
        (
            runtime.gpus[index].device.clone(),
            runtime.gpus[index].info(),
            queue.public_handle,
            private_segment_size,
            dispatch_id,
        )
    };
    drop(guard);
    crate::tool_events::alloc_start(public as *const HsaQueue, dispatch_id);
    let allocated = allocate_scratch(&device, gpu, private_segment_size, lanes_per_wave);
    let mut guard = if let Ok(guard) = lock() {
        guard
    } else {
        crate::tool_events::alloc_end(public as *const HsaQueue, dispatch_id, 0, 0);
        return None;
    };
    let Some(runtime) = guard.as_mut() else {
        drop(guard);
        crate::tool_events::alloc_end(public as *const HsaQueue, dispatch_id, 0, 0);
        return None;
    };
    let Some(queue) = runtime.queue_state.queues.get_mut(&key) else {
        drop(guard);
        crate::tool_events::alloc_end(public as *const HsaQueue, dispatch_id, 0, 0);
        return None;
    };
    let result = allocated.and_then(|(allocation, scratch)| {
        // SAFETY: The backing stays paired with this queue, and the candidate
        // is staged before its address can be written to native control state.
        let (native, backing) = unsafe { queue.native.parts_mut() };
        let old = replace_scratch_owner(
            &mut backing.scratch,
            &mut backing.uncertain_scratch,
            allocation,
            OUT_OF_RESOURCES,
            || {
                // SAFETY: The inactive signal reports a stopped queue. Both
                // old and candidate allocations remain owned if the update
                // fails or unwinds.
                unsafe { native.set_scratch(scratch) }.map_err(map_error)
            },
        )?;
        let old_visible = queue.scratch_event_visibility == ScratchEventVisibility::Published;
        backing.inactive_signal.release_queue();
        Ok((scratch, old, old_visible))
    });
    let (size, slots) = result.as_ref().map_or((0, 0), |(scratch, _, _)| {
        (
            scratch.byte_length as usize,
            scratch.maximum_wave_count as usize,
        )
    });
    let (outcome, old, old_visible) = match result {
        Ok((_, old, old_visible)) => (
            QueueEventOutcome {
                callback: None,
                rearm: true,
            },
            old,
            old_visible,
        ),
        Err(status) => (
            QueueEventOutcome {
                callback: pending_callback(runtime, key, status),
                rearm: false,
            },
            None,
            false,
        ),
    };
    drop(guard);
    if let Some(old) = old {
        if old_visible {
            crate::tool_events::free_start(public as *const HsaQueue);
            drop(old);
            crate::tool_events::free_end(public as *const HsaQueue);
        } else {
            drop(old);
        }
    }
    crate::tool_events::alloc_end(public as *const HsaQueue, dispatch_id, size, slots);
    Some(outcome)
}

/// Polls queue events and invokes callbacks outside the runtime lock.
pub(super) fn queue_event_worker(
    stop: &AtomicBool,
    alive: &AtomicBool,
    signal: &QueueEventSignal,
    observer_failed: &AtomicBool,
    key: usize,
) {
    let mut inactive_armed = true;
    let mut error_armed = true;
    let mut observer_failure_checked = false;
    while !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
        let error = signal.load_error();
        let callback = if !observer_failure_checked && observer_failed.load(Ordering::Acquire) {
            observer_failure_checked = true;
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let Some(runtime) = guard.as_mut() else {
                return;
            };
            let Some(queue) = runtime.queue_state.queues.get_mut(&key) else {
                return;
            };
            if queue.vm_fault_callback_state.take_after_system_failure() {
                pending_callback(runtime, key, MEMORY_FAULT)
            } else {
                None
            }
        } else if error != 0 && error_armed {
            error_armed = false;
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let Some(runtime) = guard.as_mut() else {
                return;
            };
            if !runtime.queue_state.queues.contains_key(&key) {
                return;
            }
            let callback = pending_error_callback(runtime, key, queue_error_status(error as u64));
            signal.release_error();
            callback
        } else {
            if error == 0 {
                error_armed = true;
            }
            let observed = signal.load_inactive();
            if observed == 0 {
                inactive_armed = true;
                None
            } else if inactive_armed {
                inactive_armed = false;
                handle_queue_event(key, observed).and_then(|outcome| {
                    inactive_armed = outcome.rearm;
                    outcome.callback
                })
            } else {
                None
            }
        };
        if let Some(callback) = callback {
            let _scope = CallbackScope::enter();
            // SAFETY: HSA requires callback and data to remain valid through
            // queue destruction; the worker only invokes it for a live record.
            unsafe {
                (callback.callback)(
                    callback.status,
                    callback.queue as *mut HsaQueue,
                    callback.data.as_ptr(),
                )
            };
        }
        thread::sleep(Duration::from_micros(20));
    }
}
