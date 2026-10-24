// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA allocation, release, and callback C entry points.

use rocddi::topology::MemoryLinkInfo;
use std::ffi::c_void;

use crate::ffi::*;
use crate::memory::allocation::*;
use crate::memory::pool::*;
use crate::memory::*;
use crate::runtime::{boundary, initialized_mut, lock, map_error};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_allocate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Allocates memory memory and returns its address.
pub unsafe extern "C" fn hsa_memory_allocate(
    region: HsaRegion,
    size: usize,
    pointer: *mut *mut c_void,
) -> Status {
    // SAFETY: Legacy regions and AMD pools identify the same memory resources.
    unsafe {
        memory_pool_allocate(
            HsaMemoryPool {
                handle: region.handle,
            },
            size,
            0,
            pointer,
            INVALID_REGION,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases the memory resource.
pub unsafe extern "C" fn hsa_memory_free(pointer: *mut c_void) -> Status {
    if pointer.is_null() {
        return boundary(|| {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            if guard.as_ref().is_some() {
                SUCCESS
            } else {
                NOT_INITIALIZED
            }
        });
    }
    // SAFETY: Both entry points release allocations owned by this runtime.
    unsafe { hsa_amd_memory_pool_free(pointer) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Updates registration for the memory register HSA operation.
pub extern "C" fn hsa_memory_register(pointer: *mut c_void, size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if !pointer.is_null() && size == 0 {
            return INVALID_ARGUMENT;
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_deregister")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Updates registration for the memory deregister HSA operation.
pub extern "C" fn hsa_memory_deregister(_pointer: *mut c_void, _size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_some() {
            SUCCESS
        } else {
            NOT_INITIALIZED
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_assign_agent")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Assigns an agent access to an HSA memory allocation.
pub extern "C" fn hsa_memory_assign_agent(
    pointer: *mut c_void,
    agent: HsaAgent,
    access: u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if pointer.is_null()
            || access == ACCESS_PERMISSION_NONE
            || !valid_access_permissions(access)
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_memory_pool_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested agent-to-pool relationship attribute.
pub unsafe extern "C" fn hsa_amd_agent_memory_pool_get_info(
    agent: HsaAgent,
    pool: HsaMemoryPool,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let Some(owner) = pool_owner(runtime, pool) else {
            return INVALID_MEMORY_POOL;
        };
        let storage = runtime
            .decode_gpu_pool(pool)
            .map_or(PoolStorage::System, |(_, kind)| match kind {
                GpuPoolKind::Group => PoolStorage::Group,
                GpuPoolKind::Fine => PoolStorage::LocalFine,
                _ => PoolStorage::LocalCoarse,
            });
        let (access, link) = agent_pool_relationship(runtime, agent, owner, storage);
        let same = owner == agent;
        let hops = if same || access == POOL_ACCESS_NEVER {
            0
        } else {
            link.map_or(0, MemoryLinkInfo::hop_count)
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                AGENT_POOL_INFO_ACCESS => value.cast::<u32>().write(access),
                AGENT_POOL_INFO_NUM_LINK_HOPS => value.cast::<u32>().write(hops),
                AGENT_POOL_INFO_LINK_INFO => value
                    .cast::<PoolLinkInfo>()
                    .write(pool_link_info(if hops == 0 { None } else { link })),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_allocate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Allocates memory pool memory and returns its address.
pub unsafe extern "C" fn hsa_amd_memory_pool_allocate(
    pool: HsaMemoryPool,
    size: usize,
    flags: u32,
    pointer: *mut *mut c_void,
) -> Status {
    // SAFETY: This forwards the public ABI arguments to the common allocator.
    unsafe { memory_pool_allocate(pool, size, flags, pointer, INVALID_MEMORY_POOL) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_register_deallocation_callback")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Registers a callback for allocation deallocation.
pub unsafe extern "C" fn hsa_amd_register_deallocation_callback(
    pointer: *mut c_void,
    callback: DeallocationCallbackFn,
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
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if let Some(memory) = runtime
            .memory_state
            .allocations
            .values_mut()
            .find(|memory| memory.contains(pointer as usize))
        {
            if memory.deallocation_callbacks.try_reserve(1).is_err() {
                return OUT_OF_RESOURCES;
            }
            memory
                .deallocation_callbacks
                .push(RegisteredDeallocationCallback {
                    pointer: pointer as usize,
                    callback,
                    user_data: user_data as usize,
                });
            return SUCCESS;
        }
        if let Some((_, reservation)) = runtime
            .memory_state
            .vmem_reservations
            .range_mut(..=pointer as usize)
            .next_back()
            .filter(|(_, reservation)| reservation.contains_range(pointer as usize, 1))
        {
            if reservation.deallocation_callbacks.try_reserve(1).is_err() {
                return OUT_OF_RESOURCES;
            }
            reservation
                .deallocation_callbacks
                .push(RegisteredDeallocationCallback {
                    pointer: pointer as usize,
                    callback,
                    user_data: user_data as usize,
                });
            return SUCCESS;
        }
        INVALID_ALLOCATION
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_deregister_deallocation_callback")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Removes a registered deallocation callback.
pub unsafe extern "C" fn hsa_amd_deregister_deallocation_callback(
    pointer: *mut c_void,
    callback: DeallocationCallbackFn,
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
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if let Some(memory) = runtime
            .memory_state
            .allocations
            .values_mut()
            .find(|memory| memory.contains(pointer as usize))
        {
            let previous_len = memory.deallocation_callbacks.len();
            memory.deallocation_callbacks.retain(|registered| {
                registered.pointer != pointer as usize
                    || registered.callback as usize != callback as usize
            });
            return if memory.deallocation_callbacks.len() == previous_len {
                INVALID_ARGUMENT
            } else {
                SUCCESS
            };
        }
        if let Some((_, reservation)) = runtime
            .memory_state
            .vmem_reservations
            .range_mut(..=pointer as usize)
            .next_back()
            .filter(|(_, reservation)| reservation.contains_range(pointer as usize, 1))
        {
            let previous_len = reservation.deallocation_callbacks.len();
            reservation.deallocation_callbacks.retain(|registered| {
                registered.pointer != pointer as usize
                    || registered.callback as usize != callback as usize
            });
            return if reservation.deallocation_callbacks.len() == previous_len {
                INVALID_ARGUMENT
            } else {
                SUCCESS
            };
        }
        INVALID_ARGUMENT
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases the memory pool resource.
pub unsafe extern "C" fn hsa_amd_memory_pool_free(pointer: *mut c_void) -> Status {
    boundary(|| {
        enum Removed {
            Allocation { key: usize, memory: Memory },
            Locked { index: usize, memory: LockedMemory },
        }

        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }

        // Transfer the native owner out of the public registry before calling
        // the driver. The token keeps its session alive if shutdown starts.
        let (mut removed, token, tracker) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let base = runtime
                .memory_state
                .allocations
                .get(&(pointer as usize))
                .map_or(pointer as usize, |memory| memory.description.agent_base);
            if runtime.memory_state.copy_borrowed(base) {
                return OUT_OF_RESOURCES;
            }
            let removed = if let Some(memory) =
                runtime.memory_state.allocations.remove(&(pointer as usize))
            {
                Removed::Allocation {
                    key: pointer as usize,
                    memory,
                }
            } else {
                let Some(index) = runtime
                    .memory_state
                    .locked_allocations
                    .iter()
                    .rposition(|memory| memory.description.agent_base == pointer as usize)
                else {
                    return INVALID_ALLOCATION;
                };
                if runtime.memory_state.copy_borrowed(
                    runtime.memory_state.locked_allocations[index]
                        .description
                        .agent_base,
                ) {
                    return OUT_OF_RESOURCES;
                }
                Removed::Locked {
                    index,
                    memory: runtime.memory_state.locked_allocations.remove(index),
                }
            };
            let tracker = runtime.inflight.clone();
            let Some(token) = tracker.enter() else {
                match removed {
                    Removed::Allocation { key, memory } => {
                        runtime.memory_state.allocations.insert(key, memory);
                    }
                    Removed::Locked { index, memory } => {
                        runtime
                            .memory_state
                            .locked_allocations
                            .insert(index, memory);
                    }
                }
                return OUT_OF_RESOURCES;
            };
            (removed, token, tracker)
        };
        let result = match &mut removed {
            Removed::Allocation { memory, .. } => memory.free(),
            Removed::Locked { memory, .. } => memory.free(),
        };
        if let Err(error) = result {
            let status = map_error(error);
            let mut slot = Some(removed);
            if let Ok(mut guard) = lock() {
                if let Some(runtime) = guard.as_mut() {
                    let can_restore = slot.as_ref().is_some_and(|owner| match owner {
                        Removed::Allocation { key, .. } => {
                            !runtime.memory_state.allocations.contains_key(key)
                        }
                        Removed::Locked { memory, .. } => !runtime
                            .memory_state
                            .locked_allocations
                            .iter()
                            .any(|current| {
                                current.description.agent_base == memory.description.agent_base
                            }),
                    });
                    if can_restore {
                        if let Some(owner) = slot.take() {
                            match owner {
                                Removed::Allocation { key, memory } => {
                                    runtime.memory_state.allocations.insert(key, memory);
                                }
                                Removed::Locked { index, memory } => {
                                    let index =
                                        index.min(runtime.memory_state.locked_allocations.len());
                                    runtime
                                        .memory_state
                                        .locked_allocations
                                        .insert(index, memory);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(owner) = slot {
                // Shutdown already detached the generation, or the registry
                // is poisoned. Its native context cannot now be retired.
                tracker.quarantine();
                std::mem::forget(owner);
            }
            drop(token);
            return status;
        }
        let callbacks = match removed {
            Removed::Allocation { mut memory, .. } => {
                std::mem::take(&mut memory.deallocation_callbacks)
            }
            Removed::Locked { .. } => Vec::new(),
        };
        drop(token);
        for registered in callbacks {
            // SAFETY: The caller registered this callback and its user data for
            // synchronous notification when this allocation was released.
            unsafe {
                (registered.callback)(
                    registered.pointer as *mut c_void,
                    registered.user_data as *mut c_void,
                )
            };
        }
        SUCCESS
    })
}
