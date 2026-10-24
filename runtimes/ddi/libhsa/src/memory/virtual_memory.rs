// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA virtual address reservations, handles, mappings, and access policy.

use super::*;

/// Finds a reservation containing the complete virtual address range.
pub(crate) fn vmem_reservation_base(
    runtime: &HsaState,
    address: usize,
    size: usize,
) -> Option<usize> {
    let (&base, reservation) = runtime
        .memory_state
        .vmem_reservations
        .range(..=address)
        .next_back()?;
    reservation.contains_range(address, size).then_some(base)
}

/// Collects all mappings intersecting the requested virtual address range.
pub(crate) fn vmem_mapping_bases(
    runtime: &HsaState,
    address: usize,
    size: usize,
) -> Result<Vec<usize>, Status> {
    let end = address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    let mut bases = Vec::new();
    let mut cursor = address;
    while cursor < end {
        let mapping = runtime
            .memory_state
            .vmem_mappings
            .get(&cursor)
            .ok_or(INVALID_ALLOCATION)?;
        bases.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
        bases.push(cursor);
        cursor = cursor.checked_add(mapping.size).ok_or(INVALID_ALLOCATION)?;
    }
    if cursor == end {
        Ok(bases)
    } else {
        Err(INVALID_ALLOCATION)
    }
}

/// Replaces access permissions for the specified mappings and agents.
pub(crate) fn replace_vmem_access(
    runtime: &mut HsaState,
    mapping_base: usize,
    descriptor: HsaAmdMemoryAccessDesc,
) -> Status {
    let agent = descriptor.agent_handle.handle;
    let agent_is_cpu = runtime.is_cpu_agent(descriptor.agent_handle);
    let existing_permission = runtime
        .memory_state
        .vmem_mappings
        .get(&mapping_base)
        .and_then(|mapping| mapping.access.get(&agent))
        .map(|access| access.permissions);
    if existing_permission == Some(descriptor.permissions) {
        return SUCCESS;
    }
    let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
        return INVALID_ALLOCATION;
    };
    if descriptor.permissions != ACCESS_PERMISSION_NONE
        && existing_permission.is_none()
        && mapping.access.try_reserve(1).is_err()
    {
        return OUT_OF_RESOURCES;
    }
    if agent_is_cpu {
        // Host page tables are process-wide. CPU agents retain separate HSA
        // permissions, but their native map enforces the union of those grants.
        let old_permissions = mapping.host_permissions();
        let permissions = mapping.host_permissions_after(agent, descriptor.permissions);
        if permissions == ACCESS_PERMISSION_NONE {
            if let Some(host_mapping) = &mut mapping.host_mapping {
                if let Err(error) = host_mapping.free() {
                    return map_error(error);
                }
                mapping.host_mapping = None;
            }
        } else if let Some(host_mapping) = &mut mapping.host_mapping {
            if permissions != old_permissions {
                let Some(access) = DeviceAccess::from_bits(permissions) else {
                    return INVALID_ARGUMENT;
                };
                if let Err(error) = host_mapping.set_permissions(access) {
                    return map_error(error);
                }
            }
        } else {
            let Some(gpu_index) = runtime
                .gpus
                .iter()
                .position(|gpu| gpu.info.fabric_handles_supported)
                .or_else(|| (!runtime.gpus.is_empty()).then_some(0))
            else {
                return NOT_SUPPORTED;
            };
            let gpu = &runtime.gpus[gpu_index];
            let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&mapping.handle) else {
                return INVALID_ALLOCATION;
            };
            if matches!(&handle.backing, VmemBacking::Fabric { .. })
                && !gpu.info.fabric_handles_supported
            {
                return NOT_SUPPORTED;
            }
            let memory = match handle.memory_for_access(gpu_index, &gpu.device) {
                Ok(memory) => memory,
                Err(status) => return status,
            };
            let Some(reservation) = runtime
                .memory_state
                .vmem_reservations
                .get(&mapping.reservation)
            else {
                return INVALID_ALLOCATION;
            };
            let Some(access) = DeviceAccess::from_bits(permissions) else {
                return INVALID_ARGUMENT;
            };
            mapping.host_mapping = Some(
                match memory.map_host(
                    &reservation.address,
                    mapping.address as u64,
                    mapping.offset as u64,
                    mapping.size as u64,
                    access,
                ) {
                    Ok(host_mapping) => host_mapping,
                    Err(error) => return map_error(error),
                },
            );
        }
        if descriptor.permissions == ACCESS_PERMISSION_NONE {
            mapping.access.remove(&agent);
        } else {
            mapping.access.insert(
                agent,
                VmemAccess {
                    permissions: descriptor.permissions,
                    owner: VmemAccessOwner::Host,
                },
            );
        }
        return SUCCESS;
    }
    let old_access = runtime
        .memory_state
        .vmem_mappings
        .get_mut(&mapping_base)
        .and_then(|mapping| mapping.access.remove(&agent));
    if let Some(mut access) = old_access {
        if let Err(error) = access.free() {
            if let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) {
                mapping.access.insert(agent, access);
            }
            return map_error(error);
        }
    }
    if descriptor.permissions == ACCESS_PERMISSION_NONE {
        return SUCCESS;
    }

    let Some(mapping) = runtime.memory_state.vmem_mappings.get(&mapping_base) else {
        return INVALID_ALLOCATION;
    };
    let gpu_index = {
        let Some(index) = runtime.gpu_index(descriptor.agent_handle) else {
            return INVALID_AGENT;
        };
        index
    };
    let Some(gpu) = runtime.gpus.get(gpu_index) else {
        return NOT_SUPPORTED;
    };
    let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&mapping.handle) else {
        return INVALID_ALLOCATION;
    };
    if matches!(&handle.backing, VmemBacking::Fabric { .. }) && !gpu.info.fabric_handles_supported {
        return NOT_SUPPORTED;
    }
    let memory = match handle.memory_for_access(gpu_index, &gpu.device) {
        Ok(memory) => memory,
        Err(status) => return status,
    };
    let Some(reservation) = runtime
        .memory_state
        .vmem_reservations
        .get(&mapping.reservation)
    else {
        return INVALID_ALLOCATION;
    };
    let Some(permissions) = DeviceAccess::from_bits(descriptor.permissions) else {
        return INVALID_ARGUMENT;
    };
    let owner = match gpu.device.map_virtual_memory(
        memory,
        &reservation.address,
        mapping.address as u64,
        mapping.offset as u64,
        mapping.size as u64,
        permissions,
    ) {
        Ok(mapping) => VmemAccessOwner::Device(mapping),
        Err(error) => return map_error(error),
    };
    let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
        return ERROR;
    };
    mapping.access.insert(
        agent,
        VmemAccess {
            permissions: descriptor.permissions,
            owner,
        },
    );
    SUCCESS
}

/// # Safety
/// `info` must point to at least `caller_size` writable bytes, beginning with
/// an aligned `size_t`.
pub(crate) unsafe fn fill_vmem_handle_info(
    info: *mut HsaAmdVmemHandleInfo,
    caller_size: usize,
    size: usize,
    agent: HsaAgent,
) {
    let returned_size = caller_size.min(size_of::<HsaAmdVmemHandleInfo>());
    // SAFETY: Each field is written only when the caller's layout contains it.
    unsafe {
        info.cast::<usize>().write(returned_size);
        if returned_size >= size_of::<HsaAmdVmemHandleInfoPrefix>() {
            std::ptr::addr_of_mut!((*info).alloc_size).write(size);
        }
        if returned_size >= size_of::<HsaAmdVmemHandleInfo>() {
            std::ptr::addr_of_mut!((*info).agent).write(agent);
        }
    }
}

#[cfg(test)]
/// Virtual-memory ABI compatibility checks for older caller record sizes.
mod handle_info_tests {
    use super::*;

    #[test]
    /// Checks short caller records leave later fields and guard bytes intact.
    fn old_layouts_leave_later_fields_and_guard_bytes_untouched() {
        for caller_size in [8, 16, 20, 24, 32] {
            let mut words = [0xabab_abab_abab_abab_u64; 4];
            let info = words.as_mut_ptr().cast::<HsaAmdVmemHandleInfo>();
            // SAFETY: The aligned buffer has 32 writable bytes.
            unsafe { fill_vmem_handle_info(info, caller_size, 0x1234, HsaAgent { handle: 0x55 }) };
            assert_eq!(words[0], caller_size.min(24) as u64);
            assert_eq!(
                words[1],
                if caller_size >= 16 {
                    0x1234
                } else {
                    0xabab_abab_abab_abab
                }
            );
            assert_eq!(
                words[2],
                if caller_size >= 24 {
                    0x55
                } else {
                    0xabab_abab_abab_abab
                }
            );
            assert_eq!(words[3], 0xabab_abab_abab_abab);
        }
    }
}
