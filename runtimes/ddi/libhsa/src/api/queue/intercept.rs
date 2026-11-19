// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD queue interception and runtime queue registration C entry points.

use std::ffi::c_void;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::lifecycle::{initialized_mut, lock};
use crate::queue_intercept::{InterceptHandler, QueueNotifier};
use crate::status::boundary;

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_create")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
/// Creates an intercept queue in front of a hardware queue.
pub unsafe extern "C" fn hsa_amd_queue_intercept_create(
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    callback: QueueErrorCallback,
    data: *mut c_void,
    private_segment_size: u32,
    _group_segment_size: u32,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        if queue.is_null()
            || size < 4
            || !size.is_power_of_two()
            || !matches!(
                queue_type,
                QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE | QUEUE_TYPE_COOPERATIVE
            )
        {
            return INVALID_ARGUMENT;
        }
        let mut created_log = None;
        // SAFETY: Proxy publication consumes the same writable output and
        // retained callback arguments as ordinary hardware queue creation.
        unsafe {
            crate::queue::create_hardware_queue(
                None,
                true,
                agent,
                size,
                queue_type,
                rocddi::device::gpu::queue::QueuePriority::Normal,
                callback,
                CallbackArg::new(data),
                private_segment_size,
                rocddi::device::gpu::queue::QueueRingMemory::System,
                rocddi::device::gpu::queue::QueueRingMemory::System,
                None,
                queue,
                &mut created_log,
            )
        }
    })
}

/// Versioned private tooling hook. The matching attach helper resolves it only
/// from the runtime image that owns its supplied native HSA table. It shares
/// descriptor validation and partial-batch publication with the public API.
/// # Safety
/// The caller supplies `descriptor_count` readable descriptors with writable queue
/// outputs and retains callback arguments through the final queue callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rocddi_hsa_queue_intercept_create_v1(
    agent: HsaAgent,
    descriptors: *mut HsaAmdQueueCreateDesc,
    descriptor_count: u32,
) -> Status {
    // SAFETY: The versioned hook has exactly the public descriptor ABI.
    unsafe {
        crate::api::queue::core::create_amd_queues(agent, descriptors, descriptor_count, true)
    }
}

#[cfg_attr(
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_register")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
    all(hsa_tool_dispatch, not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_runtime_queue_create_register")
)]
#[cfg_attr(any(not(hsa_tool_dispatch), test), unsafe(no_mangle))]
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
