// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA copy and SVM C entry points.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};
use std::thread;

pub(crate) mod async_copy;

use crate::ffi::*;
use crate::memory::copy_svm::*;
use crate::memory::pointer::*;
use crate::memory::*;
use crate::runtime::{boundary, initialized_mut, lock, map_error};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Copies bytes between validated host and GPU memory ranges.
pub unsafe extern "C" fn hsa_memory_copy(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
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
        match memory_copy_has_work(dst, src, size) {
            Ok(false) => return SUCCESS,
            Ok(true) => {}
            Err(status) => return status,
        }
        let host_destination = host_copy_address(runtime, dst as usize, size, ACCESS_PERMISSION_WO);
        let host_source = host_copy_address(runtime, src as usize, size, ACCESS_PERMISSION_RO);
        if let (Ok(destination), Ok(source)) = (host_destination, host_source) {
            // SAFETY: HSA requires both ranges to be valid for size bytes.
            // ptr::copy tolerates overlap. The registry lock retains owned
            // backing throughout this host copy.
            unsafe { std::ptr::copy(source as *const u8, destination as *mut u8, size) };
            fence(Ordering::SeqCst);
            return SUCCESS;
        }

        let resolved = match resolve_sync_gpu_copy(
            runtime,
            dst as usize,
            src as usize,
            size,
            host_destination,
            host_source,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        let mut retained = [None; 2];
        for (slot, base) in retained.iter_mut().zip(resolved.borrowed_memory) {
            if let Some(base) = base {
                if let Err(status) = runtime.memory_state.retain_copy_borrow(base) {
                    for base in retained.into_iter().flatten() {
                        runtime.memory_state.release_copy_borrow(base);
                    }
                    return status;
                }
                *slot = Some(base);
            }
        }
        let stop = runtime.stop_workers.clone();
        let health = runtime.health.clone();
        drop(guard);

        let (status, may_be_live) = match execute_sync_gpu_copy(resolved.route, &stop) {
            Ok(SyncGpuCopyOutcome::Done) => (SUCCESS, false),
            Ok(SyncGpuCopyOutcome::LoadedDestination { address, bytes }) => {
                (write_loaded_copy_destination(address, &bytes), false)
            }
            Err(failure) => (failure.status, failure.operands_may_be_live),
        };
        if may_be_live {
            health.quarantine();
        }
        release_async_resources(&[], &retained, may_be_live);
        status
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_fill")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Fills an allocation with repeated 32-bit values.
pub unsafe extern "C" fn hsa_amd_memory_fill(
    pointer: *mut c_void,
    value: u32,
    count: usize,
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
        let size = match memory_fill_size(pointer, count) {
            Ok(Some(size)) => size,
            Ok(None) => return SUCCESS,
            Err(status) => return status,
        };
        let Some(description) = pointer_description(runtime, pointer as usize) else {
            let host =
                match crate::loader::loaded_host_range(&runtime.loader, pointer as usize, size) {
                    Ok(Some(host)) => host,
                    Ok(None) => return INVALID_ALLOCATION,
                    Err(status) => return status,
                };
            if host % align_of::<u32>() != 0 {
                return INVALID_ALLOCATION;
            }
            // SAFETY: The registry lock retains the complete writable loaded
            // mapping until the fill returns.
            unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                .fill(MaybeUninit::new(value));
            fence(Ordering::SeqCst);
            return SUCCESS;
        };
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return INVALID_ALLOCATION;
        }
        if let Some(host) = description.host_range(pointer as usize, size) {
            if vmem_copy_access(
                runtime,
                &description,
                HsaAgent { handle: CPU_AGENT },
                ACCESS_PERMISSION_WO,
            )
            .is_ok()
            {
                // SAFETY: The complete aligned output range belongs to a live
                // allocation retained by the runtime lock. Its prior contents
                // need not have been initialized as u32 values.
                unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                    .fill(MaybeUninit::new(value));
                fence(Ordering::SeqCst);
                return SUCCESS;
            }
        }
        let Some((index, agent)) = selected_copy_gpu(runtime, &description) else {
            return INVALID_ALLOCATION;
        };
        let (address, base) = match gpu_copy_address(
            runtime,
            Some(&description),
            pointer as usize,
            size,
            agent,
            ACCESS_PERMISSION_WO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let Ok(native_count) = u64::try_from(count) else {
            return INVALID_ARGUMENT;
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        if let Err(status) = runtime.memory_state.retain_copy_borrow(base) {
            return status;
        }
        let device = runtime.gpus[index].device.clone();
        let stop = runtime.stop_workers.clone();
        let health = runtime.health.clone();
        drop(guard);

        let (status, may_be_live) = match device.gpu() {
            Ok(gpu) => {
                // SAFETY: The complete writable GPU range was checked above.
                // Its allocation borrow and in-flight token remain live until
                // retirement or process teardown after an uncertain result.
                match unsafe { gpu.fill_u32(address, value, native_count, &stop) } {
                    Ok(()) => (SUCCESS, false),
                    Err(failure) => (map_error(failure.error), failure.operands_may_be_live),
                }
            }
            Err(error) => (map_error(error), false),
        };
        if may_be_live {
            health.quarantine();
        }
        release_async_resources(&[], &[Some(base)], may_be_live);
        status
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_set")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Sets SVM policy attributes for a virtual address range.
pub unsafe extern "C" fn hsa_amd_svm_attributes_set(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
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
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let pairs = if attribute_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires attribute_count readable pairs.
            unsafe { std::slice::from_raw_parts(attribute_list, attribute_count) }
        };
        crate::memory::advice::set(runtime, address, size, pairs)
            .map_or_else(|status| status, |()| SUCCESS)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_get")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Reads SVM policy attributes for a virtual address range.
pub unsafe extern "C" fn hsa_amd_svm_attributes_get(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
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
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        // SAFETY: The HSA ABI supplies this writable array; the helper reads
        // only initialized input fields and stages all output before publication.
        unsafe {
            crate::memory::advice::get(runtime, address, size, attribute_list, attribute_count)
        }
        .map_or_else(|status| status, |()| SUCCESS)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_prefetch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Prefetches an SVM range after its dependency signals complete.
pub unsafe extern "C" fn hsa_amd_svm_prefetch_async(
    pointer: *mut c_void,
    size: usize,
    agent: HsaAgent,
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
        if dependency_count != 0 && dependencies.is_null() {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        let service = match svm_service(runtime, Some(agent)) {
            Ok(service) => service,
            Err(status) => return status,
        };
        let location = match svm_location_from_agent(runtime, &service, agent, false) {
            Ok(location) => location,
            Err(status) => return status,
        };
        if let Err(error) = service.validate_prefetch(
            address,
            size,
            location,
            rocddi::memory::advice::PrefetchCompletion::Migration,
        ) {
            return map_error(error);
        }
        if dependency_slice
            .iter()
            .any(|dependency| !runtime.signal_state.owns_signal(*dependency))
            || (completion.handle != 0 && !runtime.signal_state.owns_signal(completion))
        {
            return INVALID_SIGNAL;
        }
        if completion.handle != 0 {
            let status = runtime.signal_state.async_completion_status(completion);
            if status != SUCCESS {
                return status;
            }
        }
        let mut signal_handles = Vec::new();
        runtime.memory_state.reap_finished_workers();
        if signal_handles
            .try_reserve_exact(dependency_slice.len() + usize::from(completion.handle != 0))
            .is_err()
            || runtime.memory_state.workers.try_reserve(1).is_err()
        {
            return OUT_OF_RESOURCES;
        }
        signal_handles.extend_from_slice(dependency_slice);
        if completion.handle != 0 {
            signal_handles.push(completion);
        }
        let mut pending = match PendingAsyncRetention::new(
            &mut runtime.signal_state,
            &mut runtime.memory_state,
            Arc::new(signal_handles),
        ) {
            Ok(pending) => pending,
            Err(status) => return status,
        };
        let stop = runtime.stop_workers.clone();
        let worker_handles = pending.handles.clone();
        let dependency_len = dependency_slice.len();
        let worker = match thread::Builder::new()
            .name("rocddi-svm-prefetch".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    svm_prefetch_worker(
                        service,
                        stop,
                        &worker_handles[..dependency_len],
                        completion,
                        address,
                        size,
                        location,
                    );
                }));
                if result.is_err() {
                    finish_async_signal(completion, false);
                }
                release_async_resources(&worker_handles, &[], false);
            }) {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        pending.transfer_to_worker();
        pending.memory.workers.push(worker);
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Discards the requested SVM ranges asynchronously.
pub unsafe extern "C" fn hsa_amd_svm_discard_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null() || sizes.is_null() || count == 0 {
            return INVALID_ARGUMENT;
        }
        // GFX1201 reports XNACK disabled through hsa_system_get_info, so ROCr
        // rejects this operation before inspecting dependencies or ranges.
        XNACK_DISABLED
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_and_prefetch_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Discards and then prefetches SVM ranges asynchronously.
pub unsafe extern "C" fn hsa_amd_svm_discard_and_prefetch_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    destination_agents: *const HsaAgent,
    destination_count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null()
            || sizes.is_null()
            || count == 0
            || destination_agents.is_null()
            || destination_count == 0
        {
            return INVALID_ARGUMENT;
        }
        // See hsa_amd_svm_discard_batch_async: XNACK is unavailable on the
        // currently supported target, matching ROCr's required status.
        XNACK_DISABLED
    })
}
