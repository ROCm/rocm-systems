// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process initialization and shutdown entry points for the HSA C ABI.

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
use crate::api_trace;
use crate::ffi::*;
use crate::runtime::{
    HsaState, LifecycleCallScope, LifecycleTransition, ShutdownWork, boundary, lock,
};
use crate::{runtime, signal};

/// Quiesces tool callbacks while HSA storage is still published, then retires
/// the prepared generation. Callback workers may enter this on a helper thread.
fn finish_prepared_shutdown(generation: u64) -> Status {
    let _entry = LifecycleCallScope::enter_helper();
    let mut transition = LifecycleTransition::preparing(generation);
    // Operations admitted before Preparing may still publish a queue or PC
    // sampling session after leaving the lifecycle lock. Stop new workers and
    // wait for those operations before taking the callback-worker snapshot.
    let (inflight, logging) = {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return ERROR;
        };
        runtime.request_stop();
        (runtime.inflight.clone(), runtime.logging.clone())
    };
    // A borrowed FILE can call back into HSA from fwrite. Stop logging only
    // after releasing the lifecycle mutex so that callback can finish.
    logging.stop();
    inflight.wait_idle();
    let callbacks = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return ERROR;
        };
        runtime.begin_tool_shutdown()
    };
    let status = callbacks.quiesce();
    if status != SUCCESS {
        return status;
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
    {
        // OnUnload may call copied HSA functions and inspect existing handles.
        // Callback producers have stopped and native storage is still live;
        // functions requiring new native work report shutdown state.
        api_trace::unload();
    }
    // Raw signal and queue-index calls do not take the lifecycle mutex. Close
    // their admission while the generation is prepared, then wait without
    // that mutex; an admitted doorbell may itself be inside a callback.
    {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.active_generation() != Some(generation) {
            return ERROR;
        }
        runtime::close_public_raw_handles();
    }
    runtime::wait_public_raw_handles_idle();
    let runtime = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        match guard.finish_prepare_shutdown(generation) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        }
    };
    transition.advance_to_stopping(generation);
    let mut outcome = runtime.stop();
    let status = outcome.status;
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(error) => return error,
    };
    let result = guard.finish_shutdown(generation, &mut outcome);
    drop(guard);
    if result.is_ok() {
        transition.disarm();
    }
    signal::set_system_frequency(0);
    result.map_or_else(|error| error, |()| status)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_init")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Initializes the process HSA runtime and increments its reference count.
pub extern "C" fn hsa_init() -> Status {
    boundary(|| {
        let _entry = match LifecycleCallScope::enter() {
            Ok(entry) => entry,
            Err(status) => return status,
        };
        let generation = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match runtime::begin_init_waiting(guard, &runtime::LIFECYCLE_CONDVAR) {
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
/// Releases one HSA initialization reference and tears down the final state.
pub extern "C" fn hsa_shut_down() -> Status {
    boundary(|| {
        let _entry = match LifecycleCallScope::enter() {
            Ok(entry) => entry,
            Err(status) => return status,
        };
        if runtime::LogWriteScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        let work = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            match guard.begin_shutdown() {
                Ok(work) => work,
                Err(status) => return status,
            }
        };
        let generation = match work {
            ShutdownWork::Reference => return SUCCESS,
            ShutdownWork::Prepare(generation) => generation,
            ShutdownWork::NativeRetry(retirement, generation) => {
                let mut transition = LifecycleTransition::stopping(generation);
                let mut outcome = runtime::ShutdownOutcome::retry(retirement);
                let status = outcome.status;
                let mut guard = match lock() {
                    Ok(guard) => guard,
                    Err(error) => return error,
                };
                let result = guard.finish_shutdown(generation, &mut outcome);
                drop(guard);
                if result.is_ok() {
                    transition.disarm();
                }
                return result.map_or_else(|error| error, |()| status);
            }
        };
        let (on_worker, stop_workers, inflight) = match lock() {
            Ok(guard) => {
                let Some(runtime) = guard.as_ref() else {
                    return ERROR;
                };
                (
                    runtime.running_on_worker(),
                    runtime.stop_workers.clone(),
                    runtime.inflight.clone(),
                )
            }
            Err(status) => return status,
        };
        if runtime::CallbackScope::active() || on_worker {
            // The callback or worker must return before its own join. Keep the
            // prepared state published while a helper completes that join and
            // invokes OnUnload afterward.
            if runtime::defer_cleanup(generation, |generation| {
                let _ = runtime::boundary(|| finish_prepared_shutdown(generation));
            })
            .is_ok()
            {
                // The callback can return an unhandled fault before the helper
                // runs. Publish its stop request before reporting success so
                // the event worker cannot abort or invoke later handlers.
                runtime::publish_stop_request(&stop_workers, &inflight);
                return SUCCESS;
            }
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            return guard
                .cancel_shutdown(generation)
                .map_or_else(|status| status, |()| OUT_OF_RESOURCES);
        }
        finish_prepared_shutdown(generation)
    })
}
