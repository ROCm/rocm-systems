// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA memory pointer C entry points.

use std::ffi::c_void;
use std::mem::size_of;

use crate::ffi::*;
use crate::memory::pointer::*;
use crate::memory::pool::*;
use crate::runtime::{boundary, initialized_mut, lock, map_error};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_pointer_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the allocation type, bounds, and owner for a pointer.
pub unsafe extern "C" fn hsa_amd_pointer_info(
    pointer: *const c_void,
    info: *mut HsaAmdPointerInfo,
    allocator: PointerAllocator,
    num_agents_accessible: *mut u32,
    accessible: *mut *mut HsaAgent,
) -> Status {
    boundary(|| {
        if pointer.is_null() || info.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires info to reference initialized size storage.
        let input_size = unsafe { info.cast::<u32>().read() };
        if input_size == 0 {
            return INVALID_ARGUMENT;
        }
        let description = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            pointer_description(runtime, pointer as usize)
        };
        // SAFETY: The caller supplies input_size writable bytes at info.
        unsafe { write_pointer_info(info, input_size, description.as_ref()) };

        let (Some(description), Some(allocator)) = (description, allocator) else {
            return SUCCESS;
        };
        if num_agents_accessible.is_null() || accessible.is_null() {
            return SUCCESS;
        }
        let count = match u32::try_from(description.accessible.len()) {
            Ok(count) => count,
            Err(_) => return OUT_OF_RESOURCES,
        };
        // SAFETY: Both output pointers were checked above.
        unsafe { num_agents_accessible.write(count) };
        if description.accessible.is_empty() {
            // SAFETY: The caller supplied writable pointer storage.
            unsafe { accessible.write(std::ptr::null_mut()) };
            return SUCCESS;
        }
        let Some(bytes) = description
            .accessible
            .len()
            .checked_mul(size_of::<HsaAgent>())
        else {
            return OUT_OF_RESOURCES;
        };
        // SAFETY: The allocator is supplied by the caller for this exact purpose.
        let storage = unsafe { allocator(bytes) }.cast::<HsaAgent>();
        if storage.is_null() {
            return OUT_OF_RESOURCES;
        }
        // SAFETY: The allocator returned storage for every accessible handle and
        // the caller supplied writable output pointer storage.
        unsafe {
            std::ptr::copy_nonoverlapping(
                description.accessible.as_ptr(),
                storage,
                description.accessible.len(),
            );
            accessible.write(storage);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_pointer_info_set_userdata")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Associates caller-owned data with a tracked allocation.
pub unsafe extern "C" fn hsa_amd_pointer_info_set_userdata(
    pointer: *const c_void,
    user_data: *mut c_void,
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
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if set_pointer_userdata(runtime, pointer as usize, user_data as usize) {
            SUCCESS
        } else {
            INVALID_ARGUMENT
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agents_allow_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Grants the listed agents access to a memory allocation.
pub unsafe extern "C" fn hsa_amd_agents_allow_access(
    num_agents: u32,
    agents: *const HsaAgent,
    flags: *const u32,
    pointer: *const c_void,
) -> Status {
    boundary(|| {
        if num_agents == 0 || agents.is_null() || !flags.is_null() || pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The caller supplies num_agents readable entries.
        let agents = unsafe { std::slice::from_raw_parts(agents, num_agents as usize) };
        if agents.iter().any(|agent| !runtime.is_agent(*agent)) {
            return INVALID_AGENT;
        }
        let Some(key) = runtime
            .memory_state
            .allocations
            .iter()
            .find(|(_, memory)| memory.contains(pointer as usize))
            .map(|(key, _)| *key)
        else {
            // This API is defined for memory-pool allocations. IPC, graphics,
            // and locked ranges need their own native mapping lifecycle.
            return INVALID_ALLOCATION;
        };
        let memory = &runtime.memory_state.allocations[&key];
        if runtime
            .memory_state
            .copy_borrowed(memory.description.agent_base)
        {
            return OUT_OF_RESOURCES;
        }
        let owner = memory.description.owner;
        let storage = if runtime.is_cpu_agent(owner) {
            PoolStorage::System
        } else if memory.pool_global_flags & POOL_FLAG_FINE != 0 {
            PoolStorage::LocalFine
        } else {
            PoolStorage::LocalCoarse
        };
        if agents.iter().any(|agent| {
            agent_pool_relationship(runtime, *agent, owner, storage).0 == POOL_ACCESS_NEVER
                || (runtime.is_cpu_agent(*agent) && memory.description.host_base.is_none())
        }) {
            return INVALID_ARGUMENT;
        }
        let mut desired = Vec::new();
        if desired.try_reserve(agents.len().saturating_add(1)).is_err() {
            return OUT_OF_RESOURCES;
        }
        for agent in agents.iter().copied().chain(std::iter::once(owner)) {
            if !desired.contains(&agent) {
                desired.push(agent);
            }
        }
        let mut devices = Vec::new();
        if devices.try_reserve(desired.len()).is_err() {
            return OUT_OF_RESOURCES;
        }
        for agent in &desired {
            if let Some(index) = runtime.gpu_index(*agent) {
                devices.push(&runtime.gpus[index].device);
            }
        }
        let Some(origin) = runtime
            .gpus
            .iter()
            .map(|gpu| &gpu.device)
            .find(|device| memory.allocation.originates_from(device))
        else {
            return ERROR;
        };
        let cpu_count = runtime.cpus.len() as u64;
        let Some(memory) = runtime.memory_state.allocations.get_mut(&key) else {
            return ERROR;
        };
        match memory.allocation.set_device_access(origin, &devices) {
            Ok(()) => {
                memory.description.accessible = desired;
                SUCCESS
            }
            Err(error) => {
                // Native calls can complete a prefix before reporting failure.
                // Report only mappings still known to be available.
                let is_cpu = |agent: HsaAgent| {
                    agent
                        .handle
                        .checked_sub(CPU_AGENT)
                        .is_some_and(|index| index < cpu_count)
                };
                memory.description.accessible.retain(|agent| is_cpu(*agent));
                if is_cpu(owner) && !memory.description.accessible.contains(&owner) {
                    memory.description.accessible.push(owner);
                }
                if memory
                    .description
                    .accessible
                    .try_reserve(runtime.gpus.len())
                    .is_ok()
                {
                    for (index, gpu) in runtime.gpus.iter().enumerate() {
                        if memory.allocation.device_address(&gpu.device).is_ok() {
                            memory.description.accessible.push(HsaAgent {
                                handle: GPU_AGENT_BASE + index as u64,
                            });
                        }
                    }
                }
                match error.kind() {
                    rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                        INVALID_ARGUMENT
                    }
                    _ => map_error(error),
                }
            }
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_can_migrate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Reports whether memory can migrate between two pools.
pub unsafe extern "C" fn hsa_amd_memory_pool_can_migrate(
    source_pool: HsaMemoryPool,
    destination_pool: HsaMemoryPool,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if result.is_null() {
            return INVALID_ARGUMENT;
        }
        if pool_owner(runtime, source_pool).is_none()
            || pool_owner(runtime, destination_pool).is_none()
        {
            return INVALID_MEMORY_POOL;
        }
        // ROCr does not currently implement migration between otherwise valid
        // memory pools and publishes a negative result before reporting that.
        unsafe { result.write(false) };
        OUT_OF_RESOURCES
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_migrate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Migrates an allocation to the requested memory pool.
pub unsafe extern "C" fn hsa_amd_memory_migrate(
    pointer: *const c_void,
    memory_pool: HsaMemoryPool,
    flags: u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if pointer.is_null() || flags != 0 {
            return INVALID_ARGUMENT;
        }
        if pool_owner(runtime, memory_pool).is_none() {
            return INVALID_MEMORY_POOL;
        }
        OUT_OF_RESOURCES
    })
}
