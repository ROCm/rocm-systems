// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA virtual address reservations, handles, mappings, and access policy.

use super::*;

fn vmem_reservation_base(runtime: &HsaState, address: usize, size: usize) -> Option<usize> {
    let (&base, reservation) = runtime
        .memory_state
        .vmem_reservations
        .range(..=address)
        .next_back()?;
    reservation.contains_range(address, size).then_some(base)
}

fn vmem_mapping_bases(
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

fn replace_vmem_access(
    runtime: &mut HsaState,
    mapping_base: usize,
    descriptor: HsaAmdMemoryAccessDesc,
) -> Status {
    let existing_permission = runtime
        .memory_state
        .vmem_mappings
        .get(&mapping_base)
        .and_then(|mapping| mapping.access.get(&descriptor.agent_handle.handle))
        .map(|access| access.permissions);
    if existing_permission == Some(descriptor.permissions) {
        return SUCCESS;
    }
    if descriptor.permissions != ACCESS_PERMISSION_NONE {
        let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
            return INVALID_ALLOCATION;
        };
        if existing_permission.is_none() && mapping.access.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
    }
    let old_access = runtime
        .memory_state
        .vmem_mappings
        .get_mut(&mapping_base)
        .and_then(|mapping| mapping.access.remove(&descriptor.agent_handle.handle));
    if let Some(mut access) = old_access {
        if let Err(error) = access.free() {
            if let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) {
                mapping
                    .access
                    .insert(descriptor.agent_handle.handle, access);
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
    let gpu_index = if descriptor.agent_handle.handle == CPU_AGENT {
        runtime
            .gpus
            .iter()
            .position(|gpu| gpu.info.fabric_handles_supported)
            .unwrap_or(0)
    } else {
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
    let owner = if descriptor.agent_handle.handle == CPU_AGENT {
        match memory.map_host(
            &reservation.address,
            mapping.address as u64,
            mapping.offset as u64,
            mapping.size as u64,
            permissions,
        ) {
            Ok(mapping) => VmemAccessOwner::Host(mapping),
            Err(error) => return map_error(error),
        }
    } else {
        match gpu.device.map_virtual_memory(
            memory,
            &reservation.address,
            mapping.address as u64,
            mapping.offset as u64,
            mapping.size as u64,
            permissions,
        ) {
            Ok(mapping) => VmemAccessOwner::Device(mapping),
            Err(error) => return map_error(error),
        }
    };
    let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
        return ERROR;
    };
    mapping.access.insert(
        descriptor.agent_handle.handle,
        VmemAccess {
            permissions: descriptor.permissions,
            owner,
        },
    );
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_reserve")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_address_reserve(
    address: *mut *mut c_void,
    size: usize,
    requested: u64,
    flags: u64,
) -> Status {
    // SAFETY: This entry point is the default-alignment form of the same ABI.
    let page = match current_host_page() {
        Ok(page) => page,
        Err(status) => return status,
    };
    unsafe { hsa_amd_vmem_address_reserve_align(address, size, requested, page as u64, flags) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_reserve_align")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_address_reserve_align(
    address: *mut *mut c_void,
    size: usize,
    requested: u64,
    alignment: u64,
    flags: u64,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if address.is_null()
            || size == 0
            || size % page != 0
            || alignment < page as u64
            || !alignment.is_power_of_two()
            || flags & !VMEM_ADDRESS_NO_REGISTER != 0
        {
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
        let devices = runtime
            .gpus
            .iter()
            .map(|gpu| &gpu.device)
            .collect::<Vec<_>>();
        let reservation = match VirtualAddress::reserve(&devices, size as u64, alignment, requested)
        {
            Ok(reservation) => reservation,
            Err(error) => return map_error(error),
        };
        let base = reservation.info().address as usize;
        if runtime.memory_state.vmem_reservations.contains_key(&base) {
            return ERROR;
        }
        runtime.memory_state.vmem_reservations.insert(
            base,
            VmemReservation::new(reservation, flags & VMEM_ADDRESS_NO_REGISTER == 0),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { address.write(base as *mut c_void) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_vmem_address_free(address: *mut c_void, size: usize) -> Status {
    boundary(|| {
        if address.is_null() || size == 0 {
            return INVALID_ARGUMENT;
        }
        let callbacks = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let key = address as usize;
            let Some(mut reservation) = runtime.memory_state.vmem_reservations.remove(&key) else {
                return INVALID_ALLOCATION;
            };
            if reservation.description.size != size {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
                return INVALID_ARGUMENT;
            }
            if reservation.mappings != 0 {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
                return RESOURCE_FREE;
            }
            if let Err(error) = reservation.free() {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
                return map_error(error);
            }
            std::mem::take(&mut reservation.deallocation_callbacks)
        };
        for registered in callbacks {
            // SAFETY: The caller registered this callback and data for the live
            // reservation; release notification is synchronous.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_handle_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_handle_create(
    pool: HsaMemoryPool,
    size: usize,
    memory_type: u32,
    flags: u64,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if size == 0
            || size % page != 0
            || !matches!(memory_type, MEMORY_TYPE_NONE | MEMORY_TYPE_PINNED)
            || flags & !u64::from(ALLOC_UNCACHED) != 0
            || memory_handle.is_null()
        {
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
        let mut uncached = flags & u64::from(ALLOC_UNCACHED) != 0;
        let (device_index, kind) = if cpu_pool(pool) {
            (0, MemoryKind::System)
        } else if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
            if pool_kind == GPU_POOL_GROUP {
                return INVALID_ALLOCATION;
            }
            uncached |= pool_kind == GPU_POOL_EXTENDED;
            (
                index,
                MemoryKind::DeviceLocal {
                    host_visible: false,
                    coherent: gpu_pool_coherent(pool_kind, uncached, false),
                    uncached,
                    contiguous: false,
                },
            )
        } else {
            return INVALID_ARGUMENT;
        };
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let memory = match runtime.gpus[device_index].device.create_virtual_memory(
            kind,
            size as u64,
            memory_type == MEMORY_TYPE_PINNED,
            uncached,
        ) {
            Ok(memory) => memory,
            Err(error) => return map_error(error),
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(VmemBacking::Owned(memory), pool, memory_type),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_handle_release")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_vmem_handle_release(memory_handle: HsaAmdVmemAllocHandle) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(handle) = runtime
            .memory_state
            .vmem_handles
            .get_mut(&memory_handle.handle)
        else {
            return INVALID_ALLOCATION;
        };
        if handle.references == 0 {
            return INVALID_ALLOCATION;
        }
        handle.references -= 1;
        if handle.references != 0 || handle.mappings != 0 {
            return SUCCESS;
        }
        let Some(mut handle) = runtime
            .memory_state
            .vmem_handles
            .remove(&memory_handle.handle)
        else {
            return ERROR;
        };
        if let Err(error) = handle.free() {
            handle.references = 1;
            runtime
                .memory_state
                .vmem_handles
                .insert(memory_handle.handle, handle);
            return map_error(error);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_map")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_vmem_map(
    address: *mut c_void,
    size: usize,
    offset: usize,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if address.is_null()
            || (address as usize) % page != 0
            || size == 0
            || size % page != 0
            || offset % page != 0
            || flags != 0
        {
            return INVALID_ARGUMENT;
        }
        let base = address as usize;
        let end = match base.checked_add(size) {
            Some(end) => end,
            None => return INVALID_ARGUMENT,
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(reservation) = vmem_reservation_base(runtime, base, size) else {
            return INVALID_ARGUMENT;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ARGUMENT;
        };
        let Some(extent) = offset.checked_add(size) else {
            return INVALID_ARGUMENT;
        };
        if handle
            .known_size()
            .is_some_and(|known_size| extent as u64 > known_size)
        {
            return INVALID_ARGUMENT;
        }
        if runtime
            .memory_state
            .vmem_mappings
            .range(..=base)
            .next_back()
            .is_some_and(|(&mapped, mapping)| {
                mapped != base
                    && mapped
                        .checked_add(mapping.size)
                        .is_none_or(|mapped_end| mapped_end > base)
            })
            || runtime
                .memory_state
                .vmem_mappings
                .range(base..)
                .next()
                .is_some_and(|(&mapped, _)| mapped < end)
        {
            return INVALID_ARGUMENT;
        }
        let Some(handle) = runtime
            .memory_state
            .vmem_handles
            .get_mut(&memory_handle.handle)
        else {
            return ERROR;
        };
        handle.mappings = match handle.mappings.checked_add(1) {
            Some(count) => count,
            None => return OUT_OF_RESOURCES,
        };
        let Some(reservation_owner) = runtime.memory_state.vmem_reservations.get_mut(&reservation)
        else {
            handle.mappings -= 1;
            return ERROR;
        };
        let Some(reservation_mappings) = reservation_owner.mappings.checked_add(1) else {
            handle.mappings -= 1;
            return OUT_OF_RESOURCES;
        };
        reservation_owner.mappings = reservation_mappings;
        runtime.memory_state.vmem_mappings.insert(
            base,
            VmemMapping {
                handle: memory_handle.handle,
                reservation,
                address: base,
                offset,
                size,
                access: HashMap::new(),
            },
        );
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_unmap")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_vmem_unmap(address: *mut c_void, size: usize) -> Status {
    boundary(|| {
        if address.is_null() || size == 0 {
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
        let bases = match vmem_mapping_bases(runtime, address as usize, size) {
            Ok(bases) => bases,
            Err(status) => return status,
        };
        if bases
            .iter()
            .any(|&base| runtime.memory_state.copy_borrowed(base))
        {
            return OUT_OF_RESOURCES;
        }
        for base in bases {
            let Some(mut mapping) = runtime.memory_state.vmem_mappings.remove(&base) else {
                return ERROR;
            };
            if let Err(error) = mapping.free_access() {
                runtime.memory_state.vmem_mappings.insert(base, mapping);
                return map_error(error);
            }
            if runtime
                .memory_state
                .vmem_reservations
                .get(&mapping.reservation)
                .is_none_or(|reservation| reservation.mappings == 0)
            {
                runtime.memory_state.vmem_mappings.insert(base, mapping);
                return ERROR;
            }
            let Some(handle) = runtime
                .memory_state
                .vmem_handles
                .get(&mapping.handle)
                .filter(|handle| handle.mappings != 0)
            else {
                runtime.memory_state.vmem_mappings.insert(base, mapping);
                return ERROR;
            };
            let release_handle = handle.mappings == 1 && handle.references == 0;
            if release_handle {
                let Some(mut handle) = runtime.memory_state.vmem_handles.remove(&mapping.handle)
                else {
                    runtime.memory_state.vmem_mappings.insert(base, mapping);
                    return ERROR;
                };
                if let Err(error) = handle.free() {
                    runtime
                        .memory_state
                        .vmem_handles
                        .insert(mapping.handle, handle);
                    runtime.memory_state.vmem_mappings.insert(base, mapping);
                    return map_error(error);
                }
            }
            if let Some(reservation) = runtime
                .memory_state
                .vmem_reservations
                .get_mut(&mapping.reservation)
            {
                reservation.mappings -= 1;
            }
            if !release_handle {
                if let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&mapping.handle) {
                    handle.mappings -= 1;
                }
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_set_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_set_access(
    address: *mut c_void,
    size: usize,
    descriptors: *const HsaAmdMemoryAccessDesc,
    descriptor_count: usize,
) -> Status {
    boundary(|| {
        if address.is_null()
            || size == 0
            || descriptors.is_null()
            || descriptor_count == 0
            || descriptor_count > isize::MAX as usize / size_of::<HsaAmdMemoryAccessDesc>()
        {
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
        let mut requested = Vec::new();
        if requested.try_reserve_exact(descriptor_count).is_err() {
            return OUT_OF_RESOURCES;
        }
        // A CPU access change can remove the mapping that holds the caller's
        // descriptors. Own them before the first native access update.
        // SAFETY: The caller supplies descriptor_count readable entries.
        requested.extend_from_slice(unsafe {
            std::slice::from_raw_parts(descriptors, descriptor_count)
        });
        for descriptor in &requested {
            if !runtime.is_agent(descriptor.agent_handle) {
                return INVALID_AGENT;
            }
            if !valid_access_permissions(descriptor.permissions) {
                return INVALID_ARGUMENT;
            }
        }
        let bases = match vmem_mapping_bases(runtime, address as usize, size) {
            Ok(bases) => bases,
            Err(status) => return status,
        };
        if bases
            .iter()
            .any(|&base| runtime.memory_state.copy_borrowed(base))
        {
            return OUT_OF_RESOURCES;
        }
        for base in bases {
            for descriptor in &requested {
                let status = replace_vmem_access(runtime, base, *descriptor);
                if status != SUCCESS {
                    return status;
                }
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_get_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_get_access(
    address: *mut c_void,
    permissions: *mut u32,
    agent: HsaAgent,
) -> Status {
    boundary(|| {
        if address.is_null() || permissions.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let Some((_, mapping)) = runtime
            .memory_state
            .vmem_mappings
            .range(..=address as usize)
            .next_back()
            .filter(|(_, mapping)| mapping.contains(address as usize))
        else {
            return INVALID_ALLOCATION;
        };
        let value = mapping
            .access
            .get(&agent.handle)
            .map_or(ACCESS_PERMISSION_NONE, |access| access.permissions);
        // SAFETY: The caller supplied writable output storage.
        unsafe { permissions.write(value) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_export_shareable_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_export_shareable_handle(
    descriptor: *mut i32,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        if descriptor.is_null() || flags & !DMABUF_MAPPING_PCIE != 0 {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.imported() {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let Some(memory) = handle.owned_memory() else {
            return ERROR;
        };
        let index = if cpu_pool(handle.pool) {
            0
        } else if let Some((index, _)) = runtime.decode_gpu_pool(handle.pool) {
            index
        } else {
            return INVALID_ALLOCATION;
        };
        if flags & DMABUF_MAPPING_PCIE != 0 && !gpu_pcie_dmabuf_supported(runtime, index) {
            return NOT_SUPPORTED;
        }
        let dma_buf = match linux_interop::export_virtual_memory(memory) {
            Ok(dma_buf) => dma_buf,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage and ownership of
        // the descriptor is transferred only after export succeeds.
        unsafe { descriptor.write(dma_buf.into_fd().into_raw_fd()) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_import_shareable_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_import_shareable_handle(
    descriptor: i32,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        if descriptor < 0 || memory_handle.is_null() {
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
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let descriptor = match linux_interop::duplicate_descriptor(descriptor) {
            Ok(descriptor) => descriptor,
            Err(error) => return map_error(error),
        };
        let memory =
            match linux_interop::import_virtual_memory(&runtime.session, descriptor.as_fd()) {
                Ok(memory) => memory,
                Err(error) => return map_error(error),
            };
        let pool = match memory.info().placement {
            Some(VirtualMemoryPlacement::System) => HsaMemoryPool {
                handle: CPU_POOL_FINE,
            },
            Some(VirtualMemoryPlacement::Device(id)) => runtime
                .gpus
                .iter()
                .position(|gpu| gpu.info.native_gpu_id == id)
                .map_or(HsaMemoryPool { handle: 0 }, |index| {
                    HsaState::gpu_pool(index, GPU_POOL_COARSE)
                }),
            None => HsaMemoryPool { handle: 0 },
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(VmemBacking::DmaBuf(memory), pool, MEMORY_TYPE_NONE),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_retain_alloc_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_retain_alloc_handle(
    memory_handle: *mut HsaAmdVmemAllocHandle,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if memory_handle.is_null() || address.is_null() {
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
        let Some(mapping) = runtime.memory_state.vmem_mappings.get(&(address as usize)) else {
            return INVALID_ALLOCATION;
        };
        let handle_id = mapping.handle;
        let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&handle_id) else {
            return INVALID_ALLOCATION;
        };
        handle.references = match handle.references.checked_add(1) {
            Some(references) => references,
            None => return OUT_OF_RESOURCES,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle: handle_id }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_get_alloc_properties_from_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_get_alloc_properties_from_handle(
    memory_handle: HsaAmdVmemAllocHandle,
    pool: *mut HsaMemoryPool,
    memory_type: *mut u32,
) -> Status {
    boundary(|| {
        if pool.is_null() || memory_type.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        // SAFETY: Both outputs were validated and are written only after lookup.
        unsafe {
            pool.write(handle.pool);
            memory_type.write(handle.memory_type);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_get_vmem_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_get_vmem_info(
    memory_handle: HsaAmdVmemAllocHandle,
    info: *mut HsaAmdVmemHandleInfo,
) -> Status {
    boundary(|| {
        if info.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: Every accepted layout begins with a writable size_t.
        let caller_size = unsafe { info.cast::<usize>().read() };
        if caller_size < size_of::<usize>() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        let (size, agent) = match &handle.backing {
            VmemBacking::Owned(memory) => {
                let agent = if cpu_pool(handle.pool) {
                    HsaAgent { handle: CPU_AGENT }
                } else if let Some((index, _)) = runtime.decode_gpu_pool(handle.pool) {
                    HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    }
                } else {
                    return INVALID_ALLOCATION;
                };
                (memory.info().size, agent)
            }
            VmemBacking::DmaBuf(memory) => {
                let agent = match memory.info().placement {
                    Some(VirtualMemoryPlacement::System) => HsaAgent { handle: CPU_AGENT },
                    Some(VirtualMemoryPlacement::Device(id)) => {
                        let Some(index) = runtime
                            .gpus
                            .iter()
                            .position(|gpu| gpu.info.native_gpu_id == id)
                        else {
                            return INVALID_ALLOCATION;
                        };
                        HsaAgent {
                            handle: GPU_AGENT_BASE + index as u64,
                        }
                    }
                    None => return INVALID_ALLOCATION,
                };
                (memory.info().size, agent)
            }
            VmemBacking::Fabric { .. } => return INVALID_ALLOCATION,
        };
        let Ok(size) = usize::try_from(size) else {
            return INVALID_ALLOCATION;
        };
        // SAFETY: The caller promises caller_size writable bytes.
        unsafe { fill_vmem_handle_info(info, caller_size, size, agent) };
        SUCCESS
    })
}

/// # Safety
/// `info` must point to at least `caller_size` writable bytes, beginning with
/// an aligned `size_t`.
unsafe fn fill_vmem_handle_info(
    info: *mut HsaAmdVmemHandleInfo,
    caller_size: usize,
    size: usize,
    agent: HsaAgent,
) {
    let returned_size = caller_size.min(size_of::<HsaAmdVmemHandleInfo>());
    // SAFETY: Each field is written only when the caller's layout contains it.
    unsafe {
        info.cast::<usize>().write(returned_size);
        if returned_size >= std::mem::offset_of!(HsaAmdVmemHandleInfo, agent) {
            std::ptr::addr_of_mut!((*info).alloc_size).write(size);
        }
        if returned_size >= size_of::<HsaAmdVmemHandleInfo>() {
            std::ptr::addr_of_mut!((*info).agent).write(agent);
        }
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_export_fabric_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_export_fabric_handle(
    fabric_handle: *mut HsaFabricHandle,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        if fabric_handle.is_null() || flags != 0 {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.imported() {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let Some((index, _)) = runtime.decode_gpu_pool(handle.pool) else {
            return INVALID_ARGUMENT;
        };
        if !runtime.gpus[index].info.fabric_handles_supported {
            return NOT_SUPPORTED;
        }
        let Some(memory) = handle.owned_memory() else {
            return ERROR;
        };
        match linux_interop::fabric_virtual_memory_ready(memory) {
            Ok(true) => (),
            Ok(false) => return RESOURCE_NOT_READY,
            Err(error) => return map_error(error),
        }
        let fabric = match linux_interop::export_fabric_virtual_memory(memory) {
            Ok(fabric) => fabric,
            Err(error) if error.kind() == rocddi::ErrorKind::Busy => return RESOURCE_NOT_READY,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { fabric_handle.write(HsaFabricHandle { handle: fabric }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_import_fabric_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_import_fabric_handle(
    fabric_handle: HsaFabricHandle,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        if memory_handle.is_null() {
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
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(
                VmemBacking::Fabric {
                    token: fabric_handle.handle,
                    imports: HashMap::new(),
                },
                HsaMemoryPool { handle: 0 },
                MEMORY_TYPE_NONE,
            ),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[cfg(test)]
mod handle_info_tests {
    use super::*;

    #[test]
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
