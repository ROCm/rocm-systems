// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA copy and SVM C entry points.

use std::ffi::c_void;

use rocddi::device::gpu::CopyRect;

use crate::ffi::*;
use crate::memory::copy_svm::async_copy::*;
use crate::memory::copy_svm::*;
use crate::runtime::{boundary, initialized_mut, lock};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Submits a copy after dependency signals complete.
pub unsafe extern "C" fn hsa_amd_memory_async_copy(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    // SAFETY: The caller supplies readable dependency handles and live copy
    // ranges for the HSA asynchronous operation.
    unsafe {
        submit_async_copy(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            count,
            dependencies,
            completion,
            None,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy_on_engine")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Submits a copy on the requested SDMA engine.
pub unsafe extern "C" fn hsa_amd_memory_async_copy_on_engine(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
    engine: u32,
    _force: bool,
) -> Status {
    // SAFETY: The caller supplies readable dependency handles and live copy
    // ranges for the HSA asynchronous operation.
    unsafe {
        submit_async_copy(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            count,
            dependencies,
            completion,
            Some(engine),
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_batch_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Submits a batch of copy entries with control operations.
pub unsafe extern "C" fn hsa_amd_memory_async_batch_copy(
    copy_operations: *const HsaAmdMemoryCopyOp,
    operation_count: u32,
    dependency_count: u32,
    dependencies: *const HsaSignal,
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
        if operation_count == 0
            || copy_operations.is_null()
            || (dependency_count == 0 && !dependencies.is_null())
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies readable descriptors and dependency
        // handles. Every pointer list is copied into worker-owned storage
        // before this call returns.
        let operations = match unsafe { batch_slice(copy_operations, operation_count as usize) } {
            Ok(operations) => operations,
            Err(status) => return status,
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies readable dependency handles.
            match unsafe { batch_slice(dependencies, dependency_count as usize) } {
                Ok(dependencies) => dependencies,
                Err(status) => return status,
            }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        // SAFETY: batch_slice established the descriptor storage for this call.
        let prepared = match unsafe { prepare_batch_copy(runtime, operations) } {
            Ok(prepared) => prepared,
            Err(status) => return status,
        };
        enqueue_copy_job(runtime, prepared.tasks, dependency_slice)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy_rect")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Submits a rectangular copy between pitched ranges.
pub unsafe extern "C" fn hsa_amd_memory_async_copy_rect(
    dst: *const HsaPitchedPtr,
    dst_offset: *const HsaDim3,
    src: *const HsaPitchedPtr,
    src_offset: *const HsaDim3,
    range: *const HsaDim3,
    copy_agent: HsaAgent,
    direction: u32,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
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
        if dst.is_null()
            || dst_offset.is_null()
            || src.is_null()
            || src_offset.is_null()
            || range.is_null()
            || completion.handle == 0
            || !(1..=3).contains(&direction)
            || (dependency_count == 0 && !dependencies.is_null())
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.signal_state.owns_signal(completion) {
            return INVALID_SIGNAL;
        }
        // SAFETY: The HSA caller supplies readable descriptor storage.
        let (dst, dst_offset, src, src_offset, range) =
            unsafe { (*dst, *dst_offset, *src, *src_offset, *range) };
        let Some(index) = runtime.gpu_index(copy_agent) else {
            return INVALID_AGENT;
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if range.x == 0 || range.y == 0 || range.z == 0 {
            return SUCCESS;
        }
        if !runtime.gpus[index]
            .device
            .gpu()
            .is_ok_and(|gpu| gpu.supports_linear_copy())
        {
            return NOT_SUPPORTED;
        }
        let (destination, dst_extent) = match pitched_copy_range(dst, dst_offset, range) {
            Some(range) => range,
            None => return INVALID_ARGUMENT,
        };
        let (source, src_extent) = match pitched_copy_range(src, src_offset, range) {
            Some(range) => range,
            None => return INVALID_ARGUMENT,
        };
        let dst_description = match copy_pointer_description(runtime, destination, dst_extent) {
            Ok(description) => description,
            Err(status) => return status,
        };
        let src_description = match copy_pointer_description(runtime, source, src_extent) {
            Ok(description) => description,
            Err(status) => return status,
        };
        let (destination, dst_base) = match gpu_copy_address(
            runtime,
            dst_description.as_ref(),
            destination,
            dst_extent,
            copy_agent,
            ACCESS_PERMISSION_WO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let (source, src_base) = match gpu_copy_address(
            runtime,
            src_description.as_ref(),
            source,
            src_extent,
            copy_agent,
            ACCESS_PERMISSION_RO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        enqueue_async_copy(
            runtime,
            ResolvedAsyncCopy {
                route: AsyncCopyRoute::Gpu {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    index,
                    entry: GpuCopyEntry::Rect(CopyRect {
                        destination,
                        source,
                        width: u64::from(range.x),
                        height: range.y,
                        depth: range.z,
                        destination_pitch: dst.pitch as u64,
                        source_pitch: src.pitch as u64,
                        destination_slice: dst.slice as u64,
                        source_slice: src.slice as u64,
                    }),
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            },
            dependency_slice,
            completion,
            None,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_copy_engine_status")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Reports copy engines currently available for an agent.
pub unsafe extern "C" fn hsa_amd_memory_copy_engine_status(
    dst: HsaAgent,
    src: HsaAgent,
    mask: *mut u32,
) -> Status {
    boundary(|| {
        if mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if !runtime.is_agent(dst) || !runtime.is_agent(src) {
            return INVALID_AGENT;
        }
        let available = match copy_engine_mask(runtime, dst, src) {
            Ok(available) => available,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(available) };
        if available == 0 {
            OUT_OF_RESOURCES
        } else {
            SUCCESS
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_get_preferred_copy_engine")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Selects the preferred engine for a copy route.
pub unsafe extern "C" fn hsa_amd_memory_get_preferred_copy_engine(
    dst: HsaAgent,
    src: HsaAgent,
    mask: *mut u32,
) -> Status {
    boundary(|| {
        if mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if !runtime.is_agent(dst) || !runtime.is_agent(src) {
            return INVALID_AGENT;
        }
        let available = match copy_engine_mask(runtime, dst, src) {
            Ok(available) => available,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(available & available.wrapping_neg()) };
        SUCCESS
    })
}
