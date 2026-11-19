// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Exclusive initialization and retirement actions outside the registry lock.

use super::*;
#[cfg(all(hsa_tool_dispatch, not(test)))]
use crate::api_trace;

/// Keeps a completed checkpoint owned and makes its executor available to retry.
fn pause_shutdown(number: u64, transition: &mut LifecycleTransition) -> Status {
    let mut registry = match lock() {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    match registry.pause_shutdown(number) {
        Ok(()) => {
            transition.disarm();
            RESOURCE_BUSY
        }
        Err(status) => status,
    }
}

/// Retires one HSA session, resuming only its last incomplete drain checkpoint.
fn finish_prepared_shutdown(number: u64) -> Status {
    let _entry = LifecycleCallScope::enter_helper();
    let mut transition = LifecycleTransition::preparing(number);
    let (stage, inflight, logging) = {
        let registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        let stage = match registry.preparation_stage(number) {
            Ok(stage) => stage,
            Err(status) => return status,
        };
        let Some(session) = registry.as_ref() else {
            return ERROR;
        };
        (
            stage,
            session.control.inflight.clone(),
            session.logging.clone(),
        )
    };
    if stage == ShutdownStage::Calls {
        // A borrowed FILE may reenter HSA; its drain stays outside the registry.
        logging.stop();
        if !inflight.wait_idle_for(Duration::from_secs(5)) {
            return pause_shutdown(number, &mut transition);
        }
        #[cfg(all(hsa_tool_dispatch, not(test)))]
        {
            // The cleanup window keeps core services open while ordinary HSA
            // callers remain excluded. Delegation closes before the next drain.
            let preparation = match begin_tool_preparation() {
                Ok(preparation) => preparation,
                Err(status) => return status,
            };
            let status = preparation.run(|| api_trace::prepare_shutdown(number));
            if status != SUCCESS {
                return status;
            }
        }
        let mut registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        if let Err(status) = registry.finish_tool_preparation(number) {
            return status;
        }
    }
    // Successful preparation is recorded before this timeout. A retry cannot
    // replay SDK preparation or reopen delegation for additional tool work.
    if !inflight.wait_idle_for(Duration::from_secs(5)) {
        return pause_shutdown(number, &mut transition);
    }
    let callbacks = {
        let mut registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        let Some(session) = registry.as_mut() else {
            return ERROR;
        };
        session.begin_tool_shutdown()
    };
    let status = callbacks.quiesce();
    if status != SUCCESS {
        return status;
    }
    {
        let mut registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        let Some(session) = registry.as_mut() else {
            return ERROR;
        };
        if session.control.health.is_quarantined() {
            return INVALID_RUNTIME_STATE;
        }
        // Producers and callbacks have stopped. Native queues must stop before
        // a tool can release operands; public descriptors remain valid for it.
        if let Err(status) = session.queue_state.stop_native_for_tool_cleanup() {
            return status;
        }
    }
    #[cfg(all(hsa_tool_dispatch, not(test)))]
    {
        let token = match enter_tool_cleanup() {
            Ok(token) => token,
            Err(status) => return status,
        };
        let status = ApiLeaseScope::run(token, || {
            let status = api_trace::retire_profiler();
            if status != SUCCESS {
                return status;
            }
            api_trace::unload()
        });
        if status != SUCCESS {
            return status;
        }
    }
    let prepared = {
        let mut registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        match registry.finish_prepare_shutdown(number) {
            Ok(prepared) => prepared,
            Err(status) => return status,
        }
    };
    transition.advance_to_stopping();
    let outcome = prepared.retire();
    let status = outcome.status();
    let mut registry = match lock() {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    match registry.finish_shutdown(number, outcome) {
        Ok(()) => {
            transition.disarm();
            status
        }
        Err(status) => status,
    }
}

/// Acquires one public initialization reference after any startup rollback retires.
pub(crate) fn initialize() -> Status {
    if let Err(status) = crate::platform::bind_process() {
        return status;
    }
    let _entry = match LifecycleCallScope::enter() {
        Ok(entry) => entry,
        Err(status) => return status,
    };
    let number = loop {
        let registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        let work = match begin_init_waiting(registry, &LIFECYCLE_CONDVAR) {
            Ok(work) => work,
            Err(status) => return status,
        };
        match work {
            InitWork::Reference => return SUCCESS,
            InitWork::Create(number) => break number,
            InitWork::NativeRetry(session, number) => {
                let mut transition = LifecycleTransition::stopping(number);
                let outcome = ShutdownOutcome::retry(session);
                let status = outcome.status();
                let mut registry = match lock() {
                    Ok(registry) => registry,
                    Err(status) => return status,
                };
                if let Err(status) = registry.finish_init_retirement(number, outcome) {
                    return status;
                }
                transition.disarm();
                if status != SUCCESS {
                    return status;
                }
            }
        }
    };
    let mut transition = LifecycleTransition::starting(number);
    let result = HsaSession::create().map(RetainOnDrop::new);
    let mut registry = match lock() {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    match result {
        Ok(state) => {
            if let Err(status) = registry.publish_init(number, state.into_inner()) {
                return status;
            }
        }
        Err(failure) => {
            return match registry.cancel_init(number, failure.rollback) {
                Ok(()) => {
                    transition.disarm();
                    failure.status
                }
                Err(status) => status,
            };
        }
    }
    drop(registry);
    #[cfg(all(hsa_tool_dispatch, not(test)))]
    api_trace::load();
    let mut registry = match lock() {
        Ok(registry) => registry,
        Err(status) => return status,
    };
    match registry.finish_init(number) {
        Ok(()) => {
            transition.disarm();
            SUCCESS
        }
        Err(status) => status,
    }
}

/// Releases one reference, or exclusively resumes the HSA session's retirement.
pub(crate) fn shutdown() -> Status {
    if !crate::platform::process_is_current() {
        return NOT_INITIALIZED;
    }
    let _entry = match LifecycleCallScope::enter() {
        Ok(entry) => entry,
        Err(status) => return status,
    };
    if LogWriteScope::active() {
        return INVALID_RUNTIME_STATE;
    }
    let mut deferred = None;
    let work = {
        let mut registry = match lock() {
            Ok(registry) => registry,
            Err(status) => return status,
        };
        if registry.shutdown_will_prepare() {
            let Some(state) = registry.as_ref() else {
                return ERROR;
            };
            if ApiLeaseScope::current() || CallbackScope::active() || state.running_on_worker() {
                // Reserve a helper before committing closure. Failed spawning
                // leaves active references or a pending retry untouched.
                let (send, receive) = std::sync::mpsc::channel::<u64>();
                if defer_cleanup(receive, |receive| {
                    if let Ok(number) = receive.recv() {
                        let _ = plain_boundary(|| finish_prepared_shutdown(number));
                    }
                })
                .is_err()
                {
                    return OUT_OF_RESOURCES;
                }
                deferred = Some(send);
            }
        }
        match registry.begin_shutdown() {
            Ok(work) => work,
            Err(status) => return status,
        }
    };
    let number = match work {
        ShutdownWork::Reference => return SUCCESS,
        ShutdownWork::Prepare(number) => number,
        ShutdownWork::NativeRetry(session, number) => {
            let mut transition = LifecycleTransition::stopping(number);
            let outcome = ShutdownOutcome::retry(session);
            let status = outcome.status();
            let mut registry = match lock() {
                Ok(registry) => registry,
                Err(status) => return status,
            };
            return match registry.finish_shutdown(number, outcome) {
                Ok(()) => {
                    transition.disarm();
                    status
                }
                Err(status) => status,
            };
        }
    };
    if let Some(send) = deferred {
        if send.send(number).is_err() {
            drop(LifecycleTransition::preparing(number));
            return ERROR;
        }
        return SUCCESS;
    }
    finish_prepared_shutdown(number)
}
