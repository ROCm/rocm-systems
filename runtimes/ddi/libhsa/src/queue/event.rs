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

impl PendingCallback {
    /// Invokes detached callback work while the worker retains queue lifetime.
    fn invoke(self) {
        let _scope = CallbackScope::enter();
        // SAFETY: Queue destruction joins this worker before callback data expires.
        unsafe { (self.callback)(self.status, self.queue as *mut HsaQueue, self.data.as_ptr()) };
    }
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
    if (error == 512 || aql_scratch_lanes(error).is_some())
        && runtime
            .queue_state
            .queues
            .get(&key)?
            .scratch_reclaim_pending
    {
        // CP can request another allocation while an older packet still uses
        // the bounded backing. Preserve the request until reclamation retires.
        return Some(QueueEventOutcome {
            callback: None,
            rearm: true,
        });
    }
    if error == 512 {
        let queue = runtime.queue_state.queues.get_mut(&key)?;
        let public = queue.public_handle;
        // SAFETY: CP's use-once event proves the last scratch use completed.
        let (native, backing) = unsafe { queue.native.parts_mut() };
        // SAFETY: Backing remains owned until clearing the stopped descriptor succeeds.
        if let Err(error) = unsafe { native.clear_scratch() } {
            return Some(QueueEventOutcome {
                callback: pending_callback(runtime, key, map_error(error)),
                rearm: false,
            });
        }
        let old = backing.scratch.take();
        backing.inactive_signal.release_queue();
        scratch::use_once(&queue.native, false);
        drop(guard);
        if let Some(old) = old {
            crate::tool_events::free_start(public as *const HsaQueue);
            drop(old);
            crate::tool_events::free_end(public as *const HsaQueue);
        }
        return Some(QueueEventOutcome {
            callback: None,
            rearm: true,
        });
    }
    let Some(lanes_per_wave) = aql_scratch_lanes(error) else {
        return Some(QueueEventOutcome {
            callback: pending_error_callback(runtime, key, queue_error_status(error)),
            rearm: false,
        });
    };
    let (
        device,
        gpu,
        public,
        private_segment_size,
        dispatch_id,
        lanes_per_group,
        policy,
        cooperative,
    ) = {
        let queue = runtime.queue_state.queues.get(&key)?;
        let index = runtime.gpu_index(queue.agent)?;
        let Some((private_segment_size, dispatch_id, lanes_per_group)) =
            dispatch_private_segment(queue)
        else {
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
            lanes_per_group,
            queue.scratch_policy.clone(),
            queue.cooperative_refs != 0,
        )
    };
    drop(guard);
    crate::tool_events::alloc_start(public as *const HsaQueue, dispatch_id);
    let allocated = scratch::allocate_with_pressure(
        &device,
        &gpu,
        private_segment_size,
        lanes_per_wave,
        lanes_per_group.div_ceil(lanes_per_wave),
        policy.reduce_occupancy && !cooperative,
    );
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
    let result = allocated.and_then(|(allocation, scratch, reduced)| {
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
        let use_once = reduced
            || policy.reclaim
                && !cooperative
                && scratch.byte_length
                    > if policy.asynchronous {
                        policy.async_limit.load(Ordering::Acquire)
                    } else {
                        policy.single_limit
                    };
        // Publish USE_SCRATCH_ONCE before releasing the CP's stopped queue.
        if let Some(control) = native.info().aql_control {
            // SAFETY: Native queue owns the aligned properties word.
            let properties = unsafe {
                &*(AqlControlRegion::properties_address_at(control.host_base())
                    as *const std::sync::atomic::AtomicU32)
            };
            if use_once {
                properties.fetch_or(1 << 4, Ordering::Release);
            } else {
                properties.fetch_and(!(1 << 4), Ordering::Release);
            }
        }
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

/// Interprets retirement retries only while the native queue remains usable.
/// Success with `true` permits release of the scratch owner.
fn scratch_reclaim_result(
    result: Result<(), rocddi::Error>,
    available: impl FnOnce() -> bool,
    pending: &mut bool,
) -> Result<bool, Status> {
    match result {
        Ok(()) => {
            *pending = false;
            Ok(true)
        }
        Err(error)
            if matches!(
                error.kind(),
                rocddi::ErrorKind::Busy | rocddi::ErrorKind::Unsupported
            ) && available() =>
        {
            *pending |= error.kind() == rocddi::ErrorKind::Busy;
            Ok(false)
        }
        Err(error) => Err(map_error(error)),
    }
}

/// Reclaims over-limit or idle scratch, preserving backing on every failure.
fn reclaim_idle_scratch(key: usize, stop: &AtomicBool) -> Result<(), Status> {
    let mut guard = lock()?;
    let Some(runtime) = guard.as_mut() else {
        return Ok(());
    };
    let Some(queue) = runtime.queue_state.queues.get_mut(&key) else {
        return Ok(());
    };
    if !queue.scratch_policy.asynchronous
        || queue.cooperative_refs != 0
        || queue.inactivated
        || queue.teardown_started
    {
        return Ok(());
    }
    let (read, write) = queue.native.progress().map_err(map_error)?;
    if read == 0 {
        return Ok(());
    }
    if read != write {
        queue.scratch_idle_since = None;
    } else if queue.scratch_idle_since.is_none() {
        queue.scratch_idle_since = Some(std::time::Instant::now());
    }
    let idle = queue
        .scratch_idle_since
        .is_some_and(|since| since.elapsed() >= Duration::from_millis(100));
    if !queue.scratch_reclaim_pending
        && let Some(control) = queue.native.info().aql_control
    {
        // Use-once scratch has its own CP retirement event (512). Do not
        // overlap that protocol with asynchronous index-bound reclamation.
        let properties =
            unsafe { &*(control.properties_host_address() as *const std::sync::atomic::AtomicU32) };
        if properties.load(Ordering::Acquire) & (1 << 4) != 0 {
            return Ok(());
        }
    }
    // SAFETY: Native queue and its scratch stay paired throughout reclamation.
    let (native, backing) = unsafe { queue.native.parts_mut() };
    let Some(allocation) = &backing.scratch else {
        return Ok(());
    };
    let over_limit =
        allocation.info().size > queue.scratch_policy.async_limit.load(Ordering::Acquire);
    if !idle && !over_limit && !queue.scratch_reclaim_pending {
        return Ok(());
    }
    if !scratch_reclaim_result(
        native.reclaim_scratch(stop),
        || native.check().is_ok(),
        &mut queue.scratch_reclaim_pending,
    )? {
        return Ok(());
    }
    let old = backing.scratch.take();
    let public = queue.public_handle;
    queue.scratch_idle_since = None;
    drop(guard);
    if let Some(old) = old {
        crate::tool_events::free_start(public as *const HsaQueue);
        drop(old);
        crate::tool_events::free_end(public as *const HsaQueue);
    }
    Ok(())
}

/// Runs one reclaim poll and ends the worker on a terminal failure, even when
/// no application callback is registered. Notification occurs after polling
/// releases the registry lock and only while callback delivery is still live.
fn reclaim_and_notify(
    reclaim: impl FnOnce() -> Result<(), Status>,
    notify: impl FnOnce(Status),
    stop: &AtomicBool,
    alive: &AtomicBool,
) -> bool {
    if stop.load(Ordering::Acquire) || !alive.load(Ordering::Acquire) {
        return false;
    }
    if let Err(status) = reclaim() {
        if !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
            notify(status);
        }
        return false;
    }
    true
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
    let mut reclaim_check = std::time::Instant::now();
    let mut reclaim_enabled = true;
    while !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
        let Some(_lease) = crate::runtime::enter_runtime_service() else {
            thread::sleep(Duration::from_millis(1));
            continue;
        };
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
            reclaim_enabled = false;
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
                    reclaim_enabled &= outcome.rearm;
                    outcome.callback
                })
            } else {
                None
            }
        };
        if let Some(callback) = callback {
            if stop.load(Ordering::Acquire) || !alive.load(Ordering::Acquire) {
                return;
            }
            callback.invoke();
        }
        if reclaim_enabled && reclaim_check.elapsed() >= Duration::from_millis(25) {
            reclaim_check = std::time::Instant::now();
            if !reclaim_and_notify(
                || reclaim_idle_scratch(key, stop),
                |status| {
                    let callback = lock()
                        .ok()
                        .and_then(|guard| pending_callback(guard.as_ref()?, key, status));
                    if let Some(callback) = callback {
                        callback.invoke();
                    }
                },
                stop,
                alive,
            ) {
                return;
            }
        }
        thread::sleep(Duration::from_micros(20));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rocddi::ErrorKind;

    fn error(kind: ErrorKind) -> rocddi::Error {
        rocddi::Error::Operation {
            kind,
            detail: "injected scratch reclaim failure",
        }
    }

    #[test]
    fn healthy_reclaim_retries_finish_without_error_notification() {
        let stop = AtomicBool::new(false);
        let alive = AtomicBool::new(true);
        let mut pending = false;
        let mut notifications = Vec::new();
        for (result, expected_pending, expected_released) in [
            (Err(error(ErrorKind::Unsupported)), false, false),
            (Err(error(ErrorKind::Busy)), true, false),
            (Ok(()), false, true),
        ] {
            assert!(reclaim_and_notify(
                || {
                    let released = scratch_reclaim_result(result, || true, &mut pending)?;
                    assert_eq!(released, expected_released);
                    Ok(())
                },
                |status| notifications.push(status),
                &stop,
                &alive,
            ));
            assert_eq!(pending, expected_pending);
        }
        assert!(notifications.is_empty());
    }

    #[test]
    fn terminal_reclaim_failure_ends_polling_with_or_without_a_callback() {
        for (kind, available, expected) in [
            (ErrorKind::ResourceExhausted, true, OUT_OF_RESOURCES),
            (ErrorKind::Busy, false, RESOURCE_BUSY),
            (ErrorKind::Unsupported, false, NOT_SUPPORTED),
        ] {
            for has_callback in [false, true] {
                let stop = AtomicBool::new(false);
                let alive = AtomicBool::new(true);
                let mut polls = 0;
                let mut notifications = Vec::new();
                let mut pending = false;
                while reclaim_and_notify(
                    || {
                        polls += 1;
                        scratch_reclaim_result(Err(error(kind)), || available, &mut pending)?;
                        Ok(())
                    },
                    |status| {
                        if has_callback {
                            notifications.push(status);
                        }
                    },
                    &stop,
                    &alive,
                ) {
                    assert!(polls < 2, "terminal failure must end the worker");
                }
                assert_eq!(polls, 1);
                assert_eq!(
                    notifications,
                    if has_callback { vec![expected] } else { vec![] }
                );
            }
        }
    }

    #[test]
    fn stopping_during_reclaim_suppresses_error_delivery_and_further_polls() {
        for stop_generation in [false, true] {
            let stop = AtomicBool::new(false);
            let alive = AtomicBool::new(true);
            let mut polls = 0;
            let mut notifications = Vec::new();
            for _ in 0..2 {
                assert!(!reclaim_and_notify(
                    || {
                        polls += 1;
                        if stop_generation {
                            stop.store(true, Ordering::Release);
                        } else {
                            alive.store(false, Ordering::Release);
                        }
                        Err(OUT_OF_RESOURCES)
                    },
                    |status| notifications.push(status),
                    &stop,
                    &alive,
                ));
            }
            assert_eq!(polls, 1);
            assert!(notifications.is_empty());
        }
    }
}
