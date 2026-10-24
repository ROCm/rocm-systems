// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD queue interception and runtime queue registration C entry points.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::queue_intercept::{InterceptHandler, InterceptQueue, QueueNotifier};
use crate::runtime::{boundary, initialized_mut, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates an intercept queue in front of a hardware queue.
pub unsafe extern "C" fn hsa_amd_queue_intercept_create(
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    callback: QueueErrorCallback,
    data: *mut c_void,
    private_segment_size: u32,
    group_segment_size: u32,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        if queue.is_null()
            || size < 4
            || !size.is_power_of_two()
            || !matches!(queue_type, QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE)
        {
            return INVALID_ARGUMENT;
        }
        let mut lower = ptr::null_mut();
        // SAFETY: The stack output is writable. The lower queue is unpublished
        // until the proxy is ready, so it starts without a user callback.
        let status = unsafe {
            crate::api::queue::core::hsa_queue_create(
                agent,
                size,
                queue_type,
                None,
                ptr::null_mut(),
                private_segment_size,
                group_segment_size,
                &raw mut lower,
            )
        };
        if status != SUCCESS {
            return status;
        }
        let created = (|| -> Result<Arc<InterceptQueue>, Status> {
            let mut guard = lock()?;
            let runtime = initialized_mut(&mut guard)?;
            let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
            runtime
                .queue_state
                .intercept_queues
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
            // SAFETY: The C caller retains callback data until queue
            // destruction and the last callback.
            let owner = InterceptQueue::create(
                &runtime.gpus[index].device,
                lower as usize,
                size,
                callback,
                unsafe { CallbackArg::new(data) },
            )?;
            if !runtime.queue_state.queues.contains_key(&(lower as usize)) {
                return Err(ERROR);
            }
            owner.start()?;
            let lower_record = runtime
                .queue_state
                .queues
                .get_mut(&(lower as usize))
                .ok_or(ERROR)?;
            // SAFETY: The C caller retains callback data through the last
            // queue callback and the proxy remains owned until teardown.
            lower_record
                .attach_intercept(owner.public, callback, unsafe { CallbackArg::new(data) });
            runtime
                .queue_state
                .intercept_queues
                .insert(owner.public, owner.clone());
            Ok(owner)
        })();
        match created {
            Ok(owner) => {
                // SAFETY: The caller supplied writable output storage and the
                // proxy control is retained until queue destruction.
                unsafe { queue.write(owner.public as *mut HsaQueue) };
                SUCCESS
            }
            Err(status) => {
                // SAFETY: This call alone owns the unpublished lower queue.
                // If shutdown already owns it, its runtime stop releases it.
                let _ = unsafe { crate::api::queue::core::hsa_queue_destroy(lower) };
                status
            }
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Updates registration for the queue intercept register HSA operation.
pub unsafe extern "C" fn hsa_amd_queue_intercept_register(
    queue: *mut HsaQueue,
    callback: Option<InterceptHandler>,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        let Some(owner) = runtime
            .queue_state
            .intercept_queues
            .get(&(queue as usize))
            .cloned()
        else {
            return INVALID_QUEUE;
        };
        drop(guard);
        // SAFETY: The C caller retains callback data until interception and
        // the final queue callback have completed.
        owner.register(callback, unsafe { CallbackArg::new(data) })
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_runtime_queue_create_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Registers a callback for future runtime queue creation.
pub unsafe extern "C" fn hsa_amd_runtime_queue_create_register(
    callback: Option<QueueNotifier>,
    data: *mut c_void,
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
        if runtime.queue_state.internal_queue_notifier.is_some() {
            return ERROR;
        }
        // SAFETY: This is only retained and delivered for future hidden
        // runtime queues. The caller owns its callback data lifetime.
        runtime.queue_state.internal_queue_notifier =
            callback.map(|callback| (callback, unsafe { CallbackArg::new(data) }));
        SUCCESS
    })
}
