// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA C ABI entry points for this domain.

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
use crate::api_trace;
use crate::ffi::*;
use crate::runtime::{HsaState, LifecycleTransition, boundary, lock};
use crate::{complete_deferred_shutdown, runtime, signal};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_init")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_init() -> Status {
    boundary(|| {
        let generation = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match guard.begin_init() {
                Ok(None) => return SUCCESS,
                Ok(Some(generation)) => generation,
                Err(status) => return status,
            }
        };
        let mut transition = LifecycleTransition::starting(generation);
        let result = HsaState::create();
        let frequency = result
            .as_ref()
            .ok()
            .and_then(|runtime| runtime.gpus.first())
            .and_then(|gpu| runtime::clock_counters(&gpu.device).ok())
            .map_or(0, |counters| counters.system_frequency);
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let (status, transitioned, published) = match result {
            Ok(runtime) => match guard.publish_init(generation, runtime) {
                Ok(()) => {
                    signal::set_system_frequency(frequency);
                    (SUCCESS, true, true)
                }
                Err(status) => (status, false, false),
            },
            Err(failure) => match guard.cancel_init(generation, failure.cleanup_failed) {
                Ok(()) => (failure.status, true, false),
                Err(error) => (error, false, false),
            },
        };
        drop(guard);
        if published {
            #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
            api_trace::load();
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            if let Err(status) = guard.finish_init(generation) {
                return status;
            }
        }
        if transitioned {
            transition.disarm();
        }
        status
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_shut_down")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_shut_down() -> Status {
    boundary(|| {
        if runtime::LogWriteScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        let (runtime, generation) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match guard.begin_shutdown() {
                Ok(None) => return SUCCESS,
                Ok(Some(runtime)) => runtime,
                Err(status) => return status,
            }
        };
        let mut transition = LifecycleTransition::stopping(generation);
        runtime.request_stop();
        if runtime::CallbackScope::active() || runtime.running_on_worker() {
            match runtime::defer_cleanup(runtime, move |runtime| {
                complete_deferred_shutdown(runtime, generation);
            }) {
                Ok(()) => {
                    transition.disarm();
                    return SUCCESS;
                }
                Err(runtime) => {
                    // A callback may still be executing against signal storage.
                    if let Some(runtime) = runtime {
                        std::mem::forget(runtime);
                    }
                    return OUT_OF_RESOURCES;
                }
            }
        }
        let status = runtime.stop();
        #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
        api_trace::unload();
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(error) => return error,
        };
        let result = guard.finish_shutdown(generation, status);
        drop(guard);
        if result.is_ok() {
            transition.disarm();
        }
        signal::set_system_frequency(0);
        result.map_or_else(|error| error, |()| status)
    })
}
