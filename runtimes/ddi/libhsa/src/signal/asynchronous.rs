// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Retained signal callbacks and their serialized dispatcher.

use super::*;

/// Work accepted by the single asynchronous callback dispatcher.
#[derive(Clone, Copy)]
pub(super) enum AsyncRequest {
    /// Run an application callback without a signal condition.
    Function {
        callback: unsafe extern "C" fn(*mut c_void),
        arg: CallbackArg,
    },
    /// Register a callback for a signal comparison.
    Signal {
        signal: HsaSignal,
        condition: u32,
        compare_value: SignalValue,
        handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
        arg: CallbackArg,
    },
}

/// Persistent signal condition re-evaluated by the dispatcher.
struct AsyncSignalHandler {
    /// Public signal handle whose value is polled.
    signal: HsaSignal,
    /// Comparison operation for handler activation.
    condition: u32,
    /// Value compared against the signal.
    compare_value: SignalValue,
    /// Application callback invoked on a satisfied condition.
    handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
    /// Caller-owned callback argument.
    arg: CallbackArg,
}

/// Process-wide serialized callback worker and its request channel.
pub(crate) struct AsyncDispatcher {
    /// Channel sending requests to the serialized callback worker.
    sender: Sender<AsyncRequest>,
    /// Worker joined before signal storage is released.
    worker: thread::JoinHandle<()>,
}

impl AsyncDispatcher {
    fn send(&self, request: AsyncRequest) -> Status {
        self.sender.send(request).map_or(ERROR, |()| SUCCESS)
    }
}

impl SignalState {
    pub(crate) fn running_on_worker(&self) -> bool {
        self.async_dispatcher
            .as_ref()
            .is_some_and(|dispatcher| dispatcher.worker.thread().id() == thread::current().id())
    }

    /// Joins the callback worker after the shared stop request is published.
    pub(crate) fn join_worker(&mut self) {
        if let Some(dispatcher) = self.async_dispatcher.take() {
            let _ = dispatcher.worker.join();
        }
    }

    fn enqueue_async(&mut self, request: AsyncRequest, stop_workers: &Arc<AtomicBool>) -> Status {
        if self.async_dispatcher.is_none() {
            let (sender, receiver) = std::sync::mpsc::channel();
            let stop = stop_workers.clone();
            let worker = match thread::Builder::new()
                .name("rocddi-async-events".into())
                .spawn(move || run_async_dispatcher(&receiver, &stop))
            {
                Ok(worker) => worker,
                Err(_) => return OUT_OF_RESOURCES,
            };
            self.async_dispatcher = Some(AsyncDispatcher { sender, worker });
        }
        self.async_dispatcher
            .as_ref()
            .map_or(ERROR, |dispatcher| dispatcher.send(request))
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_async_function")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_async_function(
    callback: AsyncFunction,
    arg: *mut c_void,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        runtime.signal_state.enqueue_async(
            AsyncRequest::Function {
                callback,
                arg: unsafe { CallbackArg::new(arg) },
            },
            &runtime.stop_workers,
        )
    })
}

fn accept_async_request(request: AsyncRequest, handlers: &mut Vec<AsyncSignalHandler>) {
    match request {
        AsyncRequest::Function { callback, arg } => {
            let _scope = CallbackScope::enter();
            // SAFETY: The public contract requires callback and arg to remain
            // valid until the callback executes.
            unsafe { callback(arg.as_ptr()) };
        }
        AsyncRequest::Signal {
            signal,
            condition,
            compare_value,
            handler,
            arg,
        } => handlers.push(AsyncSignalHandler {
            signal,
            condition,
            compare_value,
            handler,
            arg,
        }),
    }
}

fn drain_async_requests(
    receiver: &Receiver<AsyncRequest>,
    handlers: &mut Vec<AsyncSignalHandler>,
    stop: &std::sync::atomic::AtomicBool,
) -> bool {
    loop {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        match receiver.try_recv() {
            Ok(request) => {
                if stop.load(Ordering::Acquire) {
                    return false;
                }
                accept_async_request(request, handlers);
            }
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        }
    }
}

pub(super) fn run_async_dispatcher(
    receiver: &Receiver<AsyncRequest>,
    stop: &std::sync::atomic::AtomicBool,
) {
    let mut handlers = Vec::new();
    while !stop.load(Ordering::Acquire) {
        if handlers.is_empty() {
            match receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(request) => {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    accept_async_request(request, &mut handlers);
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        if !drain_async_requests(receiver, &mut handlers, stop) {
            return;
        }

        let mut invoked = false;
        let mut index = 0;
        while index < handlers.len() {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let registration = &handlers[index];
            // SAFETY: The runtime retains each registered signal until this
            // worker releases its last handler reference.
            let observed = unsafe { hsa_signal_load_scacquire(registration.signal) };
            if stop.load(Ordering::Acquire) {
                return;
            }
            if !condition_met(registration.condition, observed, registration.compare_value) {
                index += 1;
                continue;
            }
            invoked = true;
            let _scope = CallbackScope::enter();
            // SAFETY: HSA requires the callback and argument to remain valid
            // while the registration is active.
            let keep = unsafe { (registration.handler)(observed, registration.arg.as_ptr()) };
            if keep {
                index += 1;
            } else {
                let removed = handlers.remove(index);
                if let Ok(mut guard) = lock() {
                    if let Some(runtime) = guard.as_mut() {
                        runtime.signal_state.release_async_signal(removed.signal);
                    }
                }
            }
        }

        if stop.load(Ordering::Acquire) {
            return;
        }
        if invoked {
            thread::yield_now();
        } else {
            match receiver.recv_timeout(Duration::from_micros(20)) {
                Ok(request) => {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    accept_async_request(request, &mut handlers);
                }
                Err(RecvTimeoutError::Timeout) => (),
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_async_handler")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_signal_async_handler(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    handler: SignalHandler,
    arg: *mut c_void,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(handler) = handler else {
            return INVALID_ARGUMENT;
        };
        if !matches!(
            condition,
            SIGNAL_CONDITION_EQ | SIGNAL_CONDITION_NE | SIGNAL_CONDITION_LT | SIGNAL_CONDITION_GTE
        ) {
            return INVALID_ARGUMENT;
        }
        if !runtime.signal_state.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        let address = signal.handle as usize;
        if !runtime
            .signal_state
            .interrupt_signals
            .contains_key(&address)
            && !runtime
                .signal_state
                .owned_ipc_signals
                .contains_key(&address)
            && !runtime
                .signal_state
                .imported_ipc_signals
                .contains_key(&address)
        {
            return INVALID_SIGNAL;
        }
        if let Err(status) = runtime.signal_state.retain_async_signal(signal) {
            return status;
        }
        let status = runtime.signal_state.enqueue_async(
            AsyncRequest::Signal {
                signal,
                condition,
                compare_value,
                handler,
                arg: unsafe { CallbackArg::new(arg) },
            },
            &runtime.stop_workers,
        );
        if status != SUCCESS {
            runtime.signal_state.release_async_signal(signal);
        }
        status
    })
}
