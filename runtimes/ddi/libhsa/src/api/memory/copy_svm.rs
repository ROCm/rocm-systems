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
use crate::platform::memory as linux_interop;
use crate::platform::memory::{
    KfdSvmAccess as SvmAccess, KfdSvmAttribute as SvmAttribute, KfdSvmLocation as SvmLocation,
};
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
        let (mut attributes, _) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut scalar_attributes = 0_u16;
        let mut agents = Vec::new();
        if agents.try_reserve_exact(attribute_count).is_err() {
            return OUT_OF_RESOURCES;
        }
        let mut set_flags = 0_u32;
        let mut clear_flags = 0_u32;
        for pair in pairs {
            let scalar_bit = match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_MIGRATION_GRANULARITY
                | AMD_SVM_ATTRIB_PREFERRED_LOCATION
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => Some(1_u16 << pair.attribute),
                _ => None,
            };
            if let Some(bit) = scalar_bit {
                if scalar_attributes & bit != 0 {
                    return INCOMPATIBLE_ARGUMENTS;
                }
                scalar_attributes |= bit;
            }
            match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => match pair.value {
                    AMD_SVM_GLOBAL_FLAG_FINE_GRAINED => set_flags |= SVM_FLAG_COHERENT,
                    AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED => clear_flags |= SVM_FLAG_COHERENT,
                    _ => return INVALID_ARGUMENT,
                },
                AMD_SVM_ATTRIB_READ_ONLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_ONLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_ONLY;
                    }
                }
                AMD_SVM_ATTRIB_HIVE_LOCAL => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_HIVE_LOCAL;
                    } else {
                        clear_flags |= SVM_FLAG_HIVE_LOCAL;
                    }
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    attributes.push(SvmAttribute::MigrationGranularity(
                        pair.value.min(SVM_MAX_MIGRATION_GRANULARITY) as u32,
                    ))
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    let location = match svm_location_from_agent(
                        runtime,
                        HsaAgent { handle: pair.value },
                        true,
                    ) {
                        Ok(location) => location,
                        Err(status) => return status,
                    };
                    attributes.push(SvmAttribute::PreferredLocation(location));
                }
                AMD_SVM_ATTRIB_READ_MOSTLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    }
                }
                AMD_SVM_ATTRIB_GPU_EXEC => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_EXECUTE;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_EXECUTE;
                    }
                }
                AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE
                | AMD_SVM_ATTRIB_AGENT_NO_ACCESS => {
                    let agent = HsaAgent { handle: pair.value };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if agents.contains(&agent.handle) {
                        return INCOMPATIBLE_ARGUMENTS;
                    }
                    agents.push(agent.handle);
                    if runtime.is_cpu_agent(agent) {
                        if pair.attribute == AMD_SVM_ATTRIB_AGENT_NO_ACCESS {
                            clear_flags |= SVM_FLAG_HOST_ACCESS;
                        } else {
                            set_flags |= SVM_FLAG_HOST_ACCESS;
                        }
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        let access = match pair.attribute {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE => SvmAccess::Accessible,
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE => {
                                SvmAccess::AccessibleInPlace
                            }
                            _ => SvmAccess::NoAccess,
                        };
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
            clear_flags &= !SVM_FLAG_HOST_ACCESS;
        }
        if clear_flags != 0 {
            attributes.push(SvmAttribute::ClearFlags(clear_flags));
        }
        if set_flags != 0 {
            attributes.push(SvmAttribute::SetFlags(set_flags));
        }
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        let Some(session) = runtime.session() else {
            return ERROR;
        };
        linux_interop::set_kfd_svm_attributes(session, address, size, &attributes)
            .map_or_else(map_error, |()| SUCCESS)
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
        let (mut attributes, mut indices) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut get_flags = false;
        for index in 0..attribute_count {
            // SAFETY: The caller supplies this complete pair array. Only each
            // attribute tag is initialized for a query; its value is input
            // only for ACCESS_QUERY.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => {
                    get_flags = true;
                    indices.push(None);
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::MigrationGranularity(0));
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PreferredLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PrefetchLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY requires the agent value as input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if runtime.is_cpu_agent(agent) {
                        get_flags = true;
                        indices.push(None);
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        indices.push(Some(attributes.len()));
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access: SvmAccess::Accessible,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        let flag_indices = if get_flags {
            let clear = attributes.len();
            attributes.push(SvmAttribute::ClearFlags(0));
            let set = attributes.len();
            attributes.push(SvmAttribute::SetFlags(0));
            Some((clear, set))
        } else {
            None
        };
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        let Some(session) = runtime.session() else {
            return ERROR;
        };
        if let Err(error) =
            linux_interop::get_kfd_svm_attributes(session, address, size, &mut attributes)
        {
            return map_error(error);
        }
        let (clear_flags, set_flags) = flag_indices.map_or((0, 0), |(clear, set)| {
            let clear = match attributes.get(clear).copied() {
                Some(SvmAttribute::ClearFlags(value)) => value,
                _ => 0,
            };
            let set = match attributes.get(set).copied() {
                Some(SvmAttribute::SetFlags(value)) => value,
                _ => 0,
            };
            (clear, set)
        });
        for (index, native_index) in indices.iter().copied().enumerate() {
            // SAFETY: The caller retains the complete writable pair array.
            // Read only the initialized tag until an output value is written.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            let value = match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => {
                    if set_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_FINE_GRAINED
                    } else if clear_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED
                    } else {
                        AMD_SVM_GLOBAL_FLAG_INDETERMINATE
                    }
                }
                AMD_SVM_ATTRIB_READ_ONLY => u64::from(set_flags & SVM_FLAG_GPU_READ_ONLY),
                AMD_SVM_ATTRIB_HIVE_LOCAL => u64::from(set_flags & SVM_FLAG_HIVE_LOCAL),
                AMD_SVM_ATTRIB_READ_MOSTLY => u64::from(set_flags & SVM_FLAG_GPU_READ_MOSTLY),
                AMD_SVM_ATTRIB_GPU_EXEC => u64::from(set_flags & SVM_FLAG_GPU_EXECUTE),
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let Some(SvmAttribute::MigrationGranularity(value)) =
                        attributes.get(native).copied()
                    else {
                        return ERROR;
                    };
                    u64::from(value)
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION | AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let location = match attributes.get(native).copied() {
                        Some(SvmAttribute::PreferredLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFERRED_LOCATION =>
                        {
                            location
                        }
                        Some(SvmAttribute::PrefetchLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFETCH_LOCATION =>
                        {
                            location
                        }
                        _ => return ERROR,
                    };
                    let agent = match svm_agent_from_location(runtime, location) {
                        Ok(agent) => agent,
                        Err(status) => return status,
                    };
                    agent.handle
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY retains its initialized agent input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    let result = if runtime.is_cpu_agent(agent) {
                        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                        } else {
                            AMD_SVM_ATTRIB_AGENT_NO_ACCESS
                        }
                    } else {
                        let Some(native) = native_index else {
                            return ERROR;
                        };
                        match attributes.get(native).copied() {
                            Some(SvmAttribute::Access {
                                access: SvmAccess::Accessible,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::AccessibleInPlace,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::NoAccess,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
                            _ => return ERROR,
                        }
                    };
                    // SAFETY: The pair's initialized tag is the output field
                    // for ACCESS_QUERY; its input agent value is preserved.
                    unsafe { (&raw mut (*pair).attribute).write(result) };
                    continue;
                }
                _ => return INVALID_ARGUMENT,
            };
            // SAFETY: The caller supplied writable value storage. Its prior
            // bytes need not have been initialized for output-only queries.
            unsafe { (&raw mut (*pair).value).write(value) };
        }
        SUCCESS
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
        let location = match svm_location_from_agent(runtime, agent, false) {
            Ok(location) => location,
            Err(status) => return status,
        };
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
        let Some(session) = runtime.session.clone() else {
            return ERROR;
        };
        let stop = runtime.stop_workers.clone();
        let worker_handles = pending.handles.clone();
        let dependency_len = dependency_slice.len();
        let worker = match thread::Builder::new()
            .name("rocddi-svm-prefetch".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    svm_prefetch_worker(
                        session,
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
