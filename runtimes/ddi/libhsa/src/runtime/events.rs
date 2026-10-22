// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA callback scope and native system event delivery.

use super::*;

thread_local! {
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Marks application callbacks whose teardown would wait for the caller.
pub(crate) struct CallbackScope(bool);

impl CallbackScope {
    pub(crate) fn enter() -> Self {
        Self(IN_CALLBACK.with(|active| active.replace(true)))
    }

    pub(crate) fn active() -> bool {
        IN_CALLBACK.with(Cell::get)
    }
}

impl Drop for CallbackScope {
    fn drop(&mut self) {
        IN_CALLBACK.with(|active| active.set(self.0));
    }
}

impl HsaState {
    /// Register a callback and roll it back if native event delivery cannot start.
    pub(crate) fn register_system_event_handler(
        &mut self,
        callback: SystemEventHandler,
        data: CallbackArg,
    ) -> Status {
        if self.system_event_handlers.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        self.system_event_handlers.push((callback, data));
        let status = self.ensure_system_event_worker();
        if status != SUCCESS {
            self.system_event_handlers.pop();
        }
        status
    }

    fn ensure_system_event_worker(&mut self) -> Status {
        if self.system_event_worker.is_some() {
            return SUCCESS;
        }
        let Some(device) = self.gpus.first().map(|gpu| gpu.device.clone()) else {
            return OUT_OF_RESOURCES;
        };
        let subscription = match subscribe(&device) {
            Ok(subscription) => subscription,
            Err(_) => return OUT_OF_RESOURCES,
        };
        let stop = self.stop_workers.clone();
        let worker = match thread::Builder::new()
            .name("rocddi-system-events".into())
            .spawn(move || system_event_worker(subscription, &stop))
        {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        self.system_event_worker = Some(worker);
        SUCCESS
    }
}

pub(super) fn notify_system_shutdown(handlers: &[(SystemEventHandler, CallbackArg)]) {
    let event = HsaAmdEvent {
        event_type: AMD_SYSTEM_SHUTDOWN_EVENT,
        payload: [0; 3],
    };
    let _ = notify_system_event(handlers, &event, None);
}

pub(super) fn notify_system_event(
    handlers: &[(SystemEventHandler, CallbackArg)],
    event: &HsaAmdEvent,
    stop: Option<&AtomicBool>,
) -> bool {
    let mut handled = false;
    for (callback, data) in handlers {
        if stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
            break;
        }
        let _scope = CallbackScope::enter();
        // SAFETY: Registration supplies an ABI-compatible callback. The event
        // remains live for the duration of each synchronous invocation.
        handled |= unsafe { callback(event, data.as_ptr()) } == SUCCESS;
    }
    handled
}

pub(super) fn memory_fault_event(agent: HsaAgent, fault: GpuMemoryFault) -> HsaAmdEvent {
    let mut reason = 0;
    if fault.page_not_present {
        reason |= AMD_MEMORY_FAULT_PAGE_NOT_PRESENT;
    }
    if fault.read_only {
        reason |= AMD_MEMORY_FAULT_READ_ONLY;
    }
    if fault.no_execute {
        reason |= AMD_MEMORY_FAULT_NO_EXECUTE;
    }
    if fault.imprecise {
        reason |= AMD_MEMORY_FAULT_IMPRECISE;
    }
    reason |= match fault.cause {
        GpuMemoryFaultCause::SramEcc => AMD_MEMORY_FAULT_SRAM_ECC,
        GpuMemoryFaultCause::DramEcc => AMD_MEMORY_FAULT_DRAM_ECC,
        GpuMemoryFaultCause::Hang => AMD_MEMORY_FAULT_HANG,
        GpuMemoryFaultCause::None | GpuMemoryFaultCause::Other => 0,
    };
    HsaAmdEvent {
        event_type: AMD_GPU_MEMORY_FAULT_EVENT,
        payload: [agent.handle, fault.virtual_address, u64::from(reason)],
    }
}

pub(super) fn hardware_exception_event(
    agent: HsaAgent,
    exception: GpuHardwareException,
) -> HsaAmdEvent {
    let cause = match exception.cause {
        GpuResetCause::Ecc => AMD_HW_EXCEPTION_CAUSE_ECC,
        GpuResetCause::Hang | GpuResetCause::Other => AMD_HW_EXCEPTION_CAUSE_GPU_HANG,
    };
    HsaAmdEvent {
        event_type: AMD_GPU_HW_EXCEPTION_EVENT,
        // The current HSA ABI exposes only the OTHER reset-type flag.
        payload: [
            agent.handle,
            u64::from(AMD_HW_EXCEPTION_RESET_TYPE_OTHER) | (u64::from(cause) << 32),
            0,
        ],
    }
}

pub(super) fn system_event_worker(mut subscription: DeviceEventSubscription, stop: &AtomicBool) {
    while !stop.load(Ordering::Acquire) {
        match subscription.poll() {
            Ok(Some(DeviceEvent::GpuMemoryFault(fault))) => {
                let notification = {
                    let Ok(mut guard) = lock() else {
                        return;
                    };
                    let Some(runtime) = guard.as_mut() else {
                        return;
                    };
                    let Some(index) = runtime
                        .gpus
                        .iter()
                        .position(|gpu| Some(gpu.endpoint.id) == fault.endpoint_id)
                    else {
                        continue;
                    };
                    let agent = HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    };
                    let event = memory_fault_event(agent, fault);
                    let reason = event.payload[2] as u32;
                    runtime.vm_fault_details = Some((agent, fault.virtual_address, reason));
                    // KFD reports the process fault and the queue error on
                    // different workers. Let the queue worker identify the
                    // faulted queue before delivering the system callback.
                    let Ok((mut guard, _)) = VM_FAULT_CONDVAR.wait_timeout_while(
                        guard,
                        Duration::from_millis(50),
                        |registry| {
                            registry.as_ref().is_some_and(|runtime| {
                                runtime
                                    .queue_state
                                    .queues
                                    .values()
                                    .any(|queue| queue.agent == agent)
                                    && !runtime
                                        .queue_state
                                        .queues
                                        .values()
                                        .any(|queue| queue.agent == agent && queue.vm_faulted)
                            })
                        },
                    ) else {
                        return;
                    };
                    let Some(runtime) = guard.as_mut() else {
                        return;
                    };
                    for queue in runtime.queue_state.queues.values_mut() {
                        if queue.agent == agent && queue.vm_faulted {
                            queue.vm_fault_address = fault.virtual_address;
                            queue.vm_fault_reason = reason;
                        }
                    }
                    (event, runtime.system_event_handlers.clone())
                };
                let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if !handled {
                    std::process::abort();
                }
            }
            Ok(Some(DeviceEvent::GpuHardwareException(exception))) => {
                let notification = {
                    let Ok(guard) = lock() else {
                        return;
                    };
                    let Some(runtime) = guard.as_ref() else {
                        return;
                    };
                    let Some(index) = runtime
                        .gpus
                        .iter()
                        .position(|gpu| Some(gpu.endpoint.id) == exception.endpoint_id)
                    else {
                        continue;
                    };
                    let agent = HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    };
                    (
                        hardware_exception_event(agent, exception),
                        runtime.system_event_handlers.clone(),
                    )
                };
                let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if !handled {
                    std::process::abort();
                }
            }
            Ok(None) => thread::sleep(Duration::from_micros(20)),
            Err(error)
                if matches!(
                    error.kind(),
                    rocddi::ErrorKind::Driver | rocddi::ErrorKind::Busy
                ) =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(_) => return,
        }
    }
}
