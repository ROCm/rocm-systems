// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA callback scope and native system event delivery.

use crate::admission::{ApiLeaseScope, enter_runtime_service, process_admission};
use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::lifecycle::lock;
use rocddi::device::Device;
use rocddi::device::event::{
    DeviceEvent, DeviceEventSubscription, GpuHardwareException, GpuMemoryFault,
    GpuMemoryFaultCause, GpuResetCause, subscribe,
};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Wakes callers waiting for a VM fault notification.
pub(crate) static VM_FAULT_CONDVAR: Condvar = Condvar::new();

thread_local! {
    /// Whether this thread currently invokes an application callback.
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Runs application callbacks whose teardown would wait for the caller.
pub(crate) struct CallbackScope;

/// Lexically restores the callback marker before returning to its caller.
struct CallbackGuard {
    /// Prior callback state restored when this invocation returns.
    previous: bool,
}

impl CallbackScope {
    /// Runs an application callback with reentrancy and lifetime protection.
    pub(crate) fn run<R>(operation: impl FnOnce() -> R) -> R {
        // A synchronous callback already runs under its caller's lease.
        // Independent worker callbacks acquire one before invoking user code.
        // After close, their owning worker is joined before state detachment;
        // new HSA API calls from those callbacks are refused admission.
        let token = (!ApiLeaseScope::current())
            .then(|| process_admission().enter())
            .flatten();
        let invoke = || {
            let _guard = CallbackGuard {
                previous: IN_CALLBACK.with(|active| active.replace(true)),
            };
            operation()
        };
        if let Some(token) = token {
            ApiLeaseScope::run(token, invoke)
        } else {
            invoke()
        }
    }

    /// Checks whether the current thread is executing an application callback.
    pub(crate) fn active() -> bool {
        IN_CALLBACK.with(Cell::get)
    }
}

impl Drop for CallbackGuard {
    /// Restores the previous callback reentrancy state for this thread.
    fn drop(&mut self) {
        IN_CALLBACK.with(|active| active.set(self.previous));
    }
}

/// HSA system-event registration, native observation, and fault metadata.
pub(crate) struct SystemEvents {
    /// Latest reported VM fault details for system event queries.
    fault_details: Option<(HsaAgent, u64, u32)>,
    /// Registered process-level system event callbacks.
    handlers: Vec<(SystemEventHandler, CallbackArg)>,
    /// Native system event worker joined during final teardown.
    worker: Option<JoinHandle<()>>,
    /// Sticky failure of system-event observation, separate from shutdown.
    failed: Arc<AtomicBool>,
}

/// Event work detached under the lifecycle lock for shutdown outside it.
pub(crate) struct SystemEventShutdown {
    /// Final event callbacks retained until synchronous notification.
    handlers: Vec<(SystemEventHandler, CallbackArg)>,
    /// Native system event callback worker.
    worker: Option<JoinHandle<()>>,
}

impl SystemEventShutdown {
    /// Joins the detached event producer without holding the lifecycle lock.
    pub(crate) fn join_worker(&mut self) -> thread::Result<()> {
        self.worker.take().map_or(Ok(()), JoinHandle::join)
    }

    /// Delivers the final event after callback producers have quiesced.
    pub(crate) fn notify(&self) {
        notify_system_shutdown(&self.handlers);
    }
}

impl SystemEvents {
    /// Creates an empty event registry with no native observer.
    pub(crate) fn new() -> Self {
        Self {
            fault_details: None,
            handlers: Vec::new(),
            worker: None,
            failed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register a callback and roll it back if native event delivery cannot start.
    pub(crate) fn register(
        &mut self,
        device: Option<&Device>,
        stop: &Arc<AtomicBool>,
        callback: SystemEventHandler,
        data: CallbackArg,
    ) -> Status {
        if stop.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
        if self.failed.load(Ordering::Acquire) {
            return ERROR;
        }
        if self.handlers.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        self.handlers.push((callback, data));
        let status = self.ensure_worker(device, stop);
        if status != SUCCESS {
            self.handlers.pop();
        }
        status
    }

    /// Detaches callback ownership before joins and final notification.
    pub(crate) fn take_shutdown(&mut self) -> SystemEventShutdown {
        SystemEventShutdown {
            handlers: std::mem::take(&mut self.handlers),
            worker: self.worker.take(),
        }
    }

    /// Checks whether shutdown would join the current event worker.
    pub(crate) fn running_on_worker(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| worker.thread().id() == thread::current().id())
    }

    /// Reports whether process faults can be delivered by the native observer.
    pub(crate) fn observing(&self) -> bool {
        self.worker.is_some() && !self.failed.load(Ordering::Acquire)
    }

    /// Shares observer failure with queue workers coordinating fault callbacks.
    pub(crate) fn failure_flag(&self) -> Arc<AtomicBool> {
        self.failed.clone()
    }

    /// Returns the latest process fault for queue-level event correlation.
    pub(crate) fn fault_details(&self) -> Option<(HsaAgent, u64, u32)> {
        self.fault_details
    }

    /// Ensures system event worker exists before use.
    fn ensure_worker(&mut self, device: Option<&Device>, stop: &Arc<AtomicBool>) -> Status {
        if self.failed.load(Ordering::Acquire) {
            return ERROR;
        }
        if self.worker.is_some() {
            return SUCCESS;
        }
        let Some(device) = device else {
            return OUT_OF_RESOURCES;
        };
        let subscription = match subscribe(device) {
            Ok(subscription) => subscription,
            Err(_) => return OUT_OF_RESOURCES,
        };
        let stop = stop.clone();
        let failed = self.failed.clone();
        let worker = match thread::Builder::new()
            .name("rocddi-system-events".into())
            .spawn(move || system_event_worker(subscription, &stop, &failed))
        {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        self.worker = Some(worker);
        SUCCESS
    }
}

/// Notifies registered handlers of final system shutdown.
pub(crate) fn notify_system_shutdown(handlers: &[(SystemEventHandler, CallbackArg)]) {
    let event = HsaAmdEvent {
        event_type: AMD_SYSTEM_SHUTDOWN_EVENT,
        payload: [0; 3],
    };
    let _ = notify_system_event(handlers, &event, None);
}

/// Invokes system event handlers outside the global runtime lock.
pub(crate) fn notify_system_event(
    handlers: &[(SystemEventHandler, CallbackArg)],
    event: &HsaAmdEvent,
    stop: Option<&AtomicBool>,
) -> bool {
    let mut handled = false;
    for (callback, data) in handlers {
        if stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
            break;
        }
        // SAFETY: Registration supplies an ABI-compatible callback. The event
        // remains live for the duration of each synchronous invocation.
        handled |= CallbackScope::run(|| unsafe { callback(event, data.as_ptr()) }) == SUCCESS;
    }
    handled
}

/// Encodes a GPU memory fault as an AMD system event record.
pub(crate) fn memory_fault_event(agent: HsaAgent, fault: GpuMemoryFault) -> HsaAmdEvent {
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

/// Encodes a GPU hardware exception as an AMD system event record.
pub(crate) fn hardware_exception_event(
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

/// Receives native device events and dispatches HSA callbacks.
pub(crate) fn system_event_worker(
    mut subscription: DeviceEventSubscription,
    stop: &AtomicBool,
    failed: &AtomicBool,
) {
    system_event_worker_with(
        |timeout| subscription.wait(timeout),
        thread::sleep,
        stop,
        failed,
    );
}

/// Runs the worker policy with an injectable event source and retry pause.
fn system_event_worker_with(
    mut wait: impl FnMut(Duration) -> Result<Option<DeviceEvent>, rocddi::Error>,
    mut pause: impl FnMut(Duration),
    stop: &AtomicBool,
    failed: &AtomicBool,
) {
    let mut retry_delay = Duration::from_millis(1);
    let mut driver_failures = 0u8;
    while !stop.load(Ordering::Acquire) {
        let Some(token) = enter_runtime_service() else {
            pause(Duration::from_millis(1));
            continue;
        };
        if crate::admission::ApiLeaseScope::run(token, || {
            match wait(Duration::from_millis(1)) {
                Ok(Some(DeviceEvent::GpuMemoryFault(fault))) => {
                    retry_delay = Duration::from_millis(1);
                    driver_failures = 0;
                    let notification = {
                        let Ok(mut guard) = lock() else {
                            return std::ops::ControlFlow::Break(());
                        };
                        let Some(runtime) = guard.as_mut() else {
                            return std::ops::ControlFlow::Break(());
                        };
                        let Some(index) = runtime
                            .agents
                            .gpus
                            .iter()
                            .position(|gpu| Some(gpu.endpoint.id) == fault.endpoint_id)
                        else {
                            return std::ops::ControlFlow::Continue(());
                        };
                        let agent = HsaAgent {
                            handle: GPU_AGENT_BASE + index as u64,
                        };
                        let event = memory_fault_event(agent, fault);
                        let reason = event.payload[2] as u32;
                        runtime.events.fault_details = Some((agent, fault.virtual_address, reason));
                        // KFD reports the process fault and the queue error on
                        // different workers. Let the queue worker identify the
                        // faulted queue before delivering the system callback.
                        let Ok((mut guard, _)) =
                            VM_FAULT_CONDVAR.wait_timeout_while(
                                guard,
                                Duration::from_millis(50),
                                |registry| {
                                    registry.as_ref().is_some_and(|runtime| {
                                        runtime
                                            .queue_state
                                            .queues
                                            .values()
                                            .any(|queue| queue.agent == agent)
                                            && !runtime.queue_state.queues.values().any(|queue| {
                                                queue.agent == agent && queue.vm_faulted
                                            })
                                    })
                                },
                            )
                        else {
                            return std::ops::ControlFlow::Break(());
                        };
                        let Some(runtime) = guard.as_mut() else {
                            return std::ops::ControlFlow::Break(());
                        };
                        for queue in runtime.queue_state.queues.values_mut() {
                            if queue.agent == agent && queue.vm_faulted {
                                queue.vm_fault_address = fault.virtual_address;
                                queue.vm_fault_reason = reason;
                                // Only queue errors already observed are covered
                                // by this callback. A later queue error has no
                                // native ID to correlate with this process event.
                                queue.vm_fault_callback_state.report_by_system();
                            }
                        }
                        (event, runtime.events.handlers.clone())
                    };
                    let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                    if stop.load(Ordering::Acquire) {
                        return std::ops::ControlFlow::Break(());
                    }
                    if !handled {
                        crate::core_dump::fatal_abort();
                    }
                }
                Ok(Some(DeviceEvent::GpuHardwareException(exception))) => {
                    retry_delay = Duration::from_millis(1);
                    driver_failures = 0;
                    let notification = {
                        let Ok(guard) = lock() else {
                            return std::ops::ControlFlow::Break(());
                        };
                        let Some(runtime) = guard.as_ref() else {
                            return std::ops::ControlFlow::Break(());
                        };
                        let Some(index) = runtime
                            .agents
                            .gpus
                            .iter()
                            .position(|gpu| Some(gpu.endpoint.id) == exception.endpoint_id)
                        else {
                            return std::ops::ControlFlow::Continue(());
                        };
                        let agent = HsaAgent {
                            handle: GPU_AGENT_BASE + index as u64,
                        };
                        (
                            hardware_exception_event(agent, exception),
                            runtime.events.handlers.clone(),
                        )
                    };
                    let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                    if stop.load(Ordering::Acquire) {
                        return std::ops::ControlFlow::Break(());
                    }
                    if !handled {
                        crate::core_dump::fatal_abort();
                    }
                }
                Ok(None) => {
                    retry_delay = Duration::from_millis(1);
                    driver_failures = 0;
                }
                Err(error) if error.kind() == rocddi::ErrorKind::Busy => {
                    driver_failures = 0;
                    pause(retry_delay);
                    retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
                }
                Err(error) if error.kind() == rocddi::ErrorKind::Driver && driver_failures < 16 => {
                    driver_failures += 1;
                    pause(retry_delay);
                    retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
                }
                Err(_) => {
                    // Queue error callbacks remain available even after this
                    // process event observer can no longer deliver notifications.
                    failed.store(true, Ordering::Release);
                    VM_FAULT_CONDVAR.notify_all();
                    return std::ops::ControlFlow::Break(());
                }
            }
            std::ops::ControlFlow::Continue(())
        })
        .is_break()
        {
            return;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;

#[cfg(test)]
mod worker_tests {
    use super::*;

    #[test]
    fn repeated_driver_errors_latch_only_system_event_failure() {
        let stop = AtomicBool::new(false);
        let failed = AtomicBool::new(false);
        let mut calls = 0;
        let mut pauses = Vec::new();
        system_event_worker_with(
            |_| {
                calls += 1;
                Err(rocddi::Error::Operation {
                    kind: rocddi::ErrorKind::Driver,
                    detail: "injected event poll failure",
                })
            },
            |delay| pauses.push(delay),
            &stop,
            &failed,
        );
        assert_eq!(calls, 17);
        assert_eq!(pauses.len(), 16);
        assert_eq!(pauses[0], Duration::from_millis(1));
        assert_eq!(pauses[15], Duration::from_millis(20));
        assert!(failed.load(Ordering::Acquire));
        assert!(!stop.load(Ordering::Acquire));
    }
}
