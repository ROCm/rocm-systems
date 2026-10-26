// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA pointer metadata, access, and migration operations.

use super::*;

/// Finds the owned allocation containing a device address.
pub(super) fn owned_memory(runtime: &HsaState, address: usize) -> Option<&Memory> {
    runtime
        .memory_state
        .allocations
        .values()
        .find(|memory| memory.contains(address))
        .or_else(|| {
            runtime
                .memory_state
                .ipc_allocations
                .values()
                .find(|memory| memory.contains(address))
        })
        .or_else(|| {
            runtime
                .memory_state
                .interop_allocations
                .values()
                .find(|memory| memory.contains(address))
        })
}

/// Builds HSA pointer metadata for an address known to the runtime.
pub(crate) fn pointer_description(
    runtime: &HsaState,
    address: usize,
) -> Option<PointerDescription> {
    vmem_pointer_description(runtime, address).or_else(|| {
        owned_memory(runtime, address)
            .map(|memory| memory.description.clone())
            .or_else(|| {
                runtime
                    .memory_state
                    .locked_allocations
                    .iter()
                    .rev()
                    .find(|memory| memory.contains(address))
                    .map(|memory| memory.description.clone())
            })
    })
}

/// Builds pointer metadata for a virtual memory mapping.
fn vmem_pointer_description(runtime: &HsaState, address: usize) -> Option<PointerDescription> {
    if let Some((_, mapping)) = runtime
        .memory_state
        .vmem_mappings
        .range(..=address)
        .next_back()
        .filter(|(_, mapping)| mapping.contains(address))
    {
        let handle = runtime.memory_state.vmem_handles.get(&mapping.handle)?;
        let reservation = runtime
            .memory_state
            .vmem_reservations
            .get(&mapping.reservation)?;
        let accessible = mapping
            .access
            .iter()
            .filter(|(_, access)| access.permissions != ACCESS_PERMISSION_NONE)
            .map(|(&agent, _)| HsaAgent { handle: agent })
            .collect();
        let host_access = mapping.access.iter().any(|(handle, access)| {
            runtime.is_cpu_agent(HsaAgent { handle: *handle })
                && access.permissions != ACCESS_PERMISSION_NONE
        });
        let owner = pool_owner(runtime, handle.pool).unwrap_or(HsaAgent { handle: 0 });
        let global_flags = pool_global_flags(runtime, handle.pool).unwrap_or(0);
        return Some(PointerDescription {
            pointer_type: POINTER_TYPE_HSA_VMEM,
            agent_base: mapping.address,
            host_base: host_access.then_some(mapping.address),
            size: mapping.size,
            owner,
            global_flags,
            registered: true,
            alloc_flags: if cpu_pool(handle.pool) {
                POINTER_ALLOC_NONPAGED
            } else {
                0
            } | if host_access {
                POINTER_ALLOC_HOST_ACCESS
            } else {
                0
            },
            user_data: reservation.description.user_data,
            accessible,
        });
    }
    runtime
        .memory_state
        .vmem_reservations
        .range(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
        .map(|(_, reservation)| reservation.description.clone())
}

/// Returns a host-accessible address for a validated copy range.
pub(crate) fn host_copy_address(
    runtime: &HsaState,
    address: usize,
    size: usize,
    required_access: u32,
) -> Result<usize, Status> {
    address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    if let Some(description) = pointer_description(runtime, address) {
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return Err(INVALID_ALLOCATION);
        }
        if description.pointer_type == POINTER_TYPE_HSA_VMEM
            && !description
                .accessible
                .iter()
                .copied()
                .filter(|agent| runtime.is_cpu_agent(*agent))
                .any(|agent| {
                    vmem_copy_access(runtime, &description, agent, required_access).is_ok()
                })
        {
            return Err(INVALID_ALLOCATION);
        }
        return description
            .host_range(address, size)
            .ok_or(INVALID_ALLOCATION);
    }
    crate::loader::loaded_host_range(&runtime.loader, address, size)
        .map(|host| host.unwrap_or(address))
}

/// Checks copy pointers and size, distinguishing an empty copy.
pub(crate) fn memory_copy_has_work(
    destination: *mut c_void,
    source: *const c_void,
    size: usize,
) -> Result<bool, Status> {
    if destination.is_null() || source.is_null() || size > isize::MAX as usize {
        return Err(INVALID_ARGUMENT);
    }
    Ok(size != 0)
}

/// Computes the byte extent of a 32-bit fill operation.
pub(crate) fn memory_fill_size(
    pointer: *mut c_void,
    count: usize,
) -> Result<Option<usize>, Status> {
    if pointer.is_null() || (pointer as usize) % align_of::<u32>() != 0 {
        return Err(INVALID_ARGUMENT);
    }
    if count == 0 {
        return Ok(None);
    }
    count
        .checked_mul(size_of::<u32>())
        .filter(|size| isize::try_from(*size).is_ok())
        .map(Some)
        .ok_or(INVALID_ARGUMENT)
}

/// Writes pointer info to the validated destination.
pub(crate) unsafe fn write_pointer_info(
    destination: *mut HsaAmdPointerInfo,
    input_size: u32,
    description: Option<&PointerDescription>,
) {
    let returned_size = usize::try_from(input_size)
        .unwrap_or(usize::MAX)
        .min(size_of::<HsaAmdPointerInfo>());
    let mut output = HsaAmdPointerInfo {
        size: returned_size as u32,
        pointer_type: POINTER_TYPE_UNKNOWN,
        agent_base_address: std::ptr::null_mut(),
        host_base_address: std::ptr::null_mut(),
        size_in_bytes: 0,
        user_data: std::ptr::null_mut(),
        agent_owner: HsaAgent { handle: 0 },
        global_flags: 0,
        registered: false,
        registered_padding: [0; 3],
        alloc_flags: 0,
        tail_padding: [0; 4],
    };
    if let Some(description) = description {
        output.pointer_type = description.pointer_type;
        output.agent_base_address = description.agent_base as *mut c_void;
        output.host_base_address = description
            .host_base
            .map_or(std::ptr::null_mut(), |address| address as *mut c_void);
        output.size_in_bytes = description.size;
        output.agent_owner = description.owner;
        output.global_flags = description.global_flags;
        output.registered = description.registered;
        output.alloc_flags = description.alloc_flags;
        output.user_data = description.user_data as *mut c_void;
    }
    // SAFETY: The caller supplies input_size writable bytes and returned_size is
    // bounded by both that value and the supported structure size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&raw const output).cast::<u8>(),
            destination.cast::<u8>(),
            returned_size,
        );
    }
}

/// Attaches caller data to a tracked allocation or reserved address range.
pub(crate) fn set_pointer_userdata(
    runtime: &mut HsaState,
    address: usize,
    user_data: usize,
) -> bool {
    if let Some((_, reservation)) = runtime
        .memory_state
        .vmem_reservations
        .range_mut(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
    {
        reservation.description.user_data = user_data;
        return true;
    }
    for memory in runtime.memory_state.allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.ipc_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.interop_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.locked_allocations.iter_mut().rev() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    false
}
