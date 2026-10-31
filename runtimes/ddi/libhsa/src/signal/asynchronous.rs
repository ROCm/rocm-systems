// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Retained signal callbacks and their serialized dispatcher.

use super::*;
use crate::api::signal::atomic::hsa_signal_load_scacquire;

/// Work accepted by the single asynchronous callback dispatcher.
#[derive(Clone, Copy)]
pub(crate) enum AsyncRequest {
    /// Run an application callback without a signal condition.
    Function {
        /// One-shot application function scheduled on the dispatcher.
        callback: unsafe extern "C" fn(*mut c_void),
        /// Opaque application argument retained for callback delivery.
        arg: CallbackArg,
    },
    /// Register a callback for a signal comparison.
    Signal {
        /// Signal whose condition activates the handler.
        signal: HsaSignal,
        /// HSA comparison condition evaluated by the dispatcher.
        condition: u32,
        /// Value compared against the observed signal value.
        compare_value: SignalValue,
        /// Application callback invoked when the signal condition holds.
        handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
        /// Opaque application argument retained for callback delivery.
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
    /// Sends a registration or shutdown request to the callback dispatcher.
    fn send(&self, request: AsyncRequest) -> Status {
        self.sender.send(request).map_or(ERROR, |()| SUCCESS)
    }
}

impl SignalState {
    /// Checks whether the caller is the asynchronous callback worker.
    pub(crate) fn running_on_worker(&self) -> bool {
        self.async_dispatcher
            .as_ref()
            .is_some_and(|dispatcher| dispatcher.worker.thread().id() == thread::current().id())
    }

    /// Joins the callback worker after the shared stop request is published.
    pub(crate) fn join_worker(&mut self) {
        if let Some(worker) = self.take_worker() {
            let _ = worker.join();
        }
    }

    /// Detaches the callback worker for joining outside the lifecycle lock.
    pub(crate) fn take_worker(&mut self) -> Option<thread::JoinHandle<()>> {
        self.async_dispatcher
            .take()
            .map(|dispatcher| dispatcher.worker)
    }

    /// Starts the dispatcher if needed and enqueues a callback request.
    pub(crate) fn enqueue_async(
        &mut self,
        request: AsyncRequest,
        stop_workers: &Arc<AtomicBool>,
    ) -> Status {
        if stop_workers.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
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

/// Adds or removes a signal handler from the dispatcher state.
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

/// Drains async requests.
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

/// Receives requests and invokes ready handlers outside the runtime lock.
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
