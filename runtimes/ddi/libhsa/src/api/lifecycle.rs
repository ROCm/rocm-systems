// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process initialization and shutdown entry points for the HSA C ABI.

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
use crate::api_trace;
use crate::ffi::*;
use crate::runtime::{
    ApiLeaseScope, HsaState, LifecycleCallScope, LifecycleTransition, ShutdownWork, lock,
    plain_boundary,
};
use crate::{runtime, signal};

/// Quiesces tool callbacks while HSA storage is still published, then retires
/// the prepared generation. Callback workers may enter this on a helper thread.
fn finish_prepared_shutdown(generation: u64) -> Status {
    let _entry = LifecycleCallScope::enter_helper();
    let mut transition = LifecycleTransition::preparing(generation);
    // Operations admitted before final closure may still publish a queue or
    // sampling session after leaving the lifecycle lock. Wait for them before
    // taking the callback-worker snapshot.
    let (inflight, logging) = {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return ERROR;
        };
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
        // Tools can still inspect storage they own, but copied HSA functions
        // observe the final reference count of zero and reject new admission.
        api_trace::unload();
    }
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
    let outcome = runtime.stop();
    let status = outcome.status();
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(error) => return error,
    };
    let result = guard.finish_shutdown(generation, outcome);
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
    plain_boundary(|| {
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
    plain_boundary(|| {
        let _entry = match LifecycleCallScope::enter() {
            Ok(entry) => entry,
            Err(status) => return status,
        };
        if runtime::LogWriteScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        let mut deferred = None;
        let work = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let will_prepare = match guard.shutdown_will_prepare() {
                Ok(will_prepare) => will_prepare,
                Err(status) => return status,
            };
            if will_prepare {
                let Some(runtime) = guard.as_ref() else {
                    return ERROR;
                };
                if ApiLeaseScope::current()
                    || runtime::CallbackScope::active()
                    || runtime.running_on_worker()
                {
                    // Spawn a parked helper before changing the reference
                    // count. A failed spawn leaves the runtime fully active.
                    let (send, receive) = std::sync::mpsc::channel::<u64>();
                    if runtime::defer_cleanup(receive, |receive| {
                        if let Ok(generation) = receive.recv() {
                            let _ =
                                runtime::plain_boundary(|| finish_prepared_shutdown(generation));
                        }
                    })
                    .is_err()
                    {
                        return OUT_OF_RESOURCES;
                    }
                    deferred = Some(send);
                }
            }
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
                let outcome = runtime::ShutdownOutcome::retry(retirement);
                let status = outcome.status();
                let mut guard = match lock() {
                    Ok(guard) => guard,
                    Err(error) => return error,
                };
                let result = guard.finish_shutdown(generation, outcome);
                drop(guard);
                if result.is_ok() {
                    transition.disarm();
                }
                return result.map_or_else(|error| error, |()| status);
            }
        };
        if let Some(send) = deferred {
            if send.send(generation).is_err() {
                // The committed generation has no cleanup worker. The
                // transition guard retains it instead of exposing it again.
                drop(LifecycleTransition::preparing(generation));
                return ERROR;
            }
            return SUCCESS;
        }
        finish_prepared_shutdown(generation)
    })
}
