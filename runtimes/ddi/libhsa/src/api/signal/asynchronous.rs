// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA asynchronous signal callback C entry points.

use std::ffi::c_void;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{boundary, initialized_mut, lock};
use crate::signal::asynchronous::AsyncRequest;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_async_function")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Queues a function to run on the asynchronous callback worker.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_async_handler")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Registers a callback for a signal condition.
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
        if !runtime.signal_state.supports_async_handler(signal) {
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
