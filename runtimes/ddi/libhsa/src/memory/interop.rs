// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! IPC, descriptor, host locking, and external memory interop.

use super::*;

/// Writes an AIS transfer result into the caller’s output record.
unsafe fn publish_ais_result(
    result: linux_interop::AisFileResult,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    // SAFETY: Both outputs are optional in the public ABI and, when non-null,
    // point to caller-owned writable storage for the duration of this call.
    unsafe {
        if !size_copied.is_null() {
            size_copied.write(result.size_copied);
        }
        if !operation_status.is_null() {
            operation_status.write(result.status);
        }
    }
    if result.status == 0 { SUCCESS } else { ERROR }
}

/// Transfers data between an HSA allocation and an AIS file handle.
pub(super) unsafe fn ais_transfer(
    descriptor: i32,
    host: usize,
    size: usize,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    if size > 0 && (host == 0 || host.checked_add(size).is_none() || size > isize::MAX as usize) {
        return ERROR;
    }
    // SAFETY: ais_file_io validates the full host mapping and holds the
    // runtime lock until this synchronous positioned I/O call returns.
    let buffer = match operation {
        AisOperation::Read => linux_interop::AisHostBuffer::Read(if size == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(host as *mut u8, size) }
        }),
        AisOperation::Write => linux_interop::AisHostBuffer::Write(if size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(host as *const u8, size) }
        }),
    };
    let result = linux_interop::ais_host_transfer(descriptor, buffer, file_offset);
    // SAFETY: The caller owns the optional output storage for this call.
    unsafe { publish_ais_result(result, size_copied, operation_status) }
}

/// Reads or writes a file descriptor for an AIS transfer.
pub(crate) unsafe fn ais_file_io(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    // SAFETY: Linux defines the public union's active member as fd.
    let descriptor = unsafe { handle.fd };
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if device_pointer.is_null() || descriptor < 0 {
        return INVALID_ARGUMENT;
    }
    let transfer_size = match usize::try_from(size.min(linux_interop::AIS_MAX_TRANSFER_BYTES)) {
        Ok(size) => size,
        Err(_) => return ERROR,
    };
    let Some(description) = pointer_description(runtime, device_pointer as usize) else {
        return ERROR;
    };
    if runtime.is_cpu_agent(description.owner) {
        return ERROR;
    }
    let address = device_pointer as usize;
    let Some(offset) = description.offset_range(address, transfer_size) else {
        return ERROR;
    };
    // Hold the runtime lock through either synchronous transfer, preventing
    // another HSA call from freeing the allocation during the borrow.
    let status = if let Some(host) = description.host_address(address) {
        // SAFETY: The mapped host range stays owned by the locked runtime.
        // The public call keeps its descriptor live until this call returns.
        unsafe {
            ais_transfer(
                descriptor,
                host,
                transfer_size,
                file_offset,
                size_copied,
                operation_status,
                operation,
            )
        }
    } else {
        let Some(memory) = owned_memory(runtime, address).filter(|memory| {
            memory.description.agent_base == description.agent_base
                && memory.description.host_base == description.host_base
                && memory.description.offset_range(address, transfer_size) == Some(offset)
        }) else {
            return ERROR;
        };
        let offset = match u64::try_from(offset) {
            Ok(offset) => offset,
            Err(_) => return ERROR,
        };
        let size = match u64::try_from(transfer_size) {
            Ok(size) => size,
            Err(_) => return ERROR,
        };
        match linux_interop::ais_transfer(
            &memory.allocation,
            descriptor,
            offset,
            size,
            file_offset,
            operation,
        ) {
            Ok(result) => {
                // SAFETY: Optional outputs belong to the caller for this call.
                unsafe { publish_ais_result(result, size_copied, operation_status) }
            }
            Err(error) => {
                // The ioctl input overlaps its output. On error KFD's copied
                // count cannot be trusted, so leave size_copied untouched.
                let errno = error
                    .native_error_code()
                    .filter(|errno| *errno > 0)
                    .unwrap_or(EIO);
                if !operation_status.is_null() {
                    // SAFETY: The non-null optional output is caller-owned.
                    unsafe { operation_status.write(-errno) };
                }
                ERROR
            }
        }
    };
    drop(guard);
    status
}

/// Exports an allocation as a portable DMA-BUF descriptor.
pub(crate) unsafe fn portable_export_dmabuf(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
    flags: u64,
) -> Status {
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if pointer.is_null() || descriptor.is_null() || offset.is_null() || size == 0 {
        return INVALID_ARGUMENT;
    }
    let Some((memory, pointer_offset)) =
        runtime
            .memory_state
            .allocations
            .values()
            .find_map(|memory| {
                memory
                    .description
                    .offset_range(pointer as usize, size)
                    .map(|pointer_offset| (memory, pointer_offset))
            })
    else {
        return INVALID_ALLOCATION;
    };
    if runtime.is_cpu_agent(memory.description.owner) {
        return INVALID_AGENT;
    }
    let Some(index) = runtime.gpu_index(memory.description.owner) else {
        return INVALID_AGENT;
    };
    if flags & DMABUF_MAPPING_PCIE != 0 && !gpu_pcie_dmabuf_supported(runtime, index) {
        return NOT_SUPPORTED;
    }
    let dma_buf = match linux_interop::export_dma_buf(&memory.allocation) {
        Ok(dma_buf) => dma_buf,
        Err(error) => return map_error(error),
    };
    let exported_offset = match dma_buf
        .info()
        .source_offset
        .checked_add(pointer_offset as u64)
    {
        Some(exported_offset) => exported_offset,
        None => return INVALID_ALLOCATION,
    };
    let exported_descriptor = dma_buf.into_fd().into_raw_fd();
    // SAFETY: Both output pointers were validated above and are written only
    // after the export has completed successfully.
    unsafe {
        descriptor.write(exported_descriptor);
        offset.write(exported_offset);
    }
    SUCCESS
}

/// # Safety
/// `host_ptr` must cover `size` live host bytes through successful unlock,
/// `agents` must contain `num_agents` readable handles when non-null, and
/// `agent_ptr` must point to writable output storage. After an ambiguous
/// native registration failure, the caller must retain the host page cover
/// through process teardown.
pub(crate) unsafe fn memory_lock_to_pool(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    pool: Option<HsaMemoryPool>,
    flags: u32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let pool = pool.unwrap_or_else(|| {
        runtime.cpus.iter().position(|cpu| !cpu.apu).map_or_else(
            || HsaState::cpu_pool(0, CpuPoolKind::Fine),
            |index| HsaState::cpu_pool(index, CpuPoolKind::Coarse),
        )
    });
    if host_ptr.is_null() || size == 0 || agent_ptr.is_null() {
        return INVALID_ARGUMENT;
    }
    if num_agents < 0
        || (agents.is_null() && num_agents != 0)
        || (!agents.is_null() && num_agents == 0)
    {
        return INVALID_ARGUMENT;
    }
    if runtime.decode_cpu_pool(pool).is_none() {
        return INVALID_MEMORY_POOL;
    }
    let global_flags = match pool_global_flags(runtime, pool) {
        Some(flags) => flags,
        None => return INVALID_MEMORY_POOL,
    };
    let owner = pool_owner(runtime, pool).ok_or(INVALID_MEMORY_POOL);
    let owner = match owner {
        Ok(owner) => owner,
        Err(status) => return status,
    };
    if runtime.full_profile {
        // SAFETY: The caller supplied writable pointer storage.
        unsafe { agent_ptr.write(host_ptr) };
        return SUCCESS;
    }

    // Keep the output untouched while reading the caller's agent array: the
    // two may share storage under the public C ABI.
    let mut gpu_indexes = Vec::new();
    if agents.is_null() {
        gpu_indexes.extend(0..runtime.gpus.len());
    } else {
        let count = match usize::try_from(num_agents) {
            Ok(count) => count,
            Err(_) => return INVALID_ARGUMENT,
        };
        // SAFETY: The ABI requires num_agents readable handles when agents is non-null.
        let requested = unsafe { std::slice::from_raw_parts(agents, count) };
        for agent in requested {
            if runtime.is_cpu_agent(*agent) {
                continue;
            }
            let Some(index) = runtime.gpu_index(*agent) else {
                return INVALID_AGENT;
            };
            if !gpu_indexes.contains(&index) {
                gpu_indexes.push(index);
            }
        }
    }

    if runtime
        .memory_state
        .locked_allocations
        .try_reserve(1)
        .is_err()
    {
        return OUT_OF_RESOURCES;
    }
    let host_base = host_ptr as usize;
    let (allocation, device_base) =
        if let Some((&first_index, peer_indexes)) = gpu_indexes.split_first() {
            let (allocation, page_offset) = {
                let first = &runtime.gpus[first_index].device;
                let peers: Vec<_> = peer_indexes
                    .iter()
                    .map(|&index| &runtime.gpus[index].device)
                    .collect();
                // SAFETY: The HSA lock caller retains the complete host page
                // cover until unlock succeeds; ambiguous native failures keep
                // the registration backing caller-owned through teardown.
                match unsafe {
                    first.register_host_range_with_peers(
                        &peers,
                        host_base,
                        cpu_pool_mapping_policy(pool, flags & ALLOC_UNCACHED),
                        size as u64,
                        runtime.host_page_size as u64,
                        DeviceAccess::READ | DeviceAccess::WRITE,
                    )
                } {
                    Ok(allocation) => allocation,
                    Err(error) => return map_error(error),
                }
            };
            let Some(device_base) =
                (allocation.info().device_address as usize).checked_add(page_offset as usize)
            else {
                return OUT_OF_RESOURCES;
            };
            (Some(allocation), device_base)
        } else {
            (None, host_base)
        };
    let accessible = gpu_indexes
        .into_iter()
        .map(|index| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        })
        .collect();
    runtime
        .memory_state
        .locked_allocations
        .push(LockedMemory::new(
            allocation,
            host_base,
            device_base,
            size,
            owner,
            effective_global_flags(global_flags, flags & ALLOC_UNCACHED != 0, false),
            accessible,
        ));
    // SAFETY: The caller supplied writable pointer storage.
    unsafe { agent_ptr.write(device_base as *mut c_void) };
    SUCCESS
}

#[allow(clippy::too_many_arguments)]
/// Maps an external graphics buffer into HSA-visible memory.
pub(crate) unsafe fn interop_map_buffer(
    num_agents: u32,
    agents: *mut HsaAgent,
    interop_handle: HsaHandle,
    flags: u32,
    size_hint: usize,
    size: *mut usize,
    pointer: *mut *mut c_void,
    metadata_size: *mut usize,
    metadata: *mut *const c_void,
) -> Status {
    if num_agents == 0
        || agents.is_null()
        || interop_handle < 0
        || flags != 0
        || size.is_null()
        || pointer.is_null()
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
    // SAFETY: The caller supplies num_agents readable handles.
    let agents = unsafe { std::slice::from_raw_parts(agents, num_agents as usize) };
    let mut devices = Vec::new();
    let mut accessible = Vec::new();
    if devices.try_reserve(agents.len()).is_err() || accessible.try_reserve(agents.len()).is_err() {
        return OUT_OF_RESOURCES;
    }
    for agent in agents {
        let Some(index) = runtime.gpu_index(*agent) else {
            return INVALID_AGENT;
        };
        devices.push(&runtime.gpus[index].device);
        accessible.push(*agent);
    }
    let descriptor = match linux_interop::duplicate_descriptor(interop_handle) {
        Ok(descriptor) => descriptor,
        Err(error) => return map_error(error),
    };
    let Some(session) = runtime.session() else {
        return ERROR;
    };
    let allocation = match linux_interop::import_graphics_dma_buf(
        session,
        &devices,
        descriptor.as_fd(),
        size_hint as u64,
    ) {
        Ok(allocation) => allocation,
        Err(error) => {
            return match error.kind() {
                rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                    INVALID_ARGUMENT
                }
                rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
                _ => ERROR,
            };
        }
    };
    let info = allocation.info();
    let imported_size = match usize::try_from(info.size) {
        Ok(size) => size,
        Err(_) => return OUT_OF_RESOURCES,
    };
    let imported_pointer = info.device_address as usize;
    let owner = runtime
        .gpus
        .iter()
        .enumerate()
        .find(|(_, gpu)| allocation.originates_from(&gpu.device))
        .map_or(HsaAgent { handle: 0 }, |(index, _)| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        });
    let metadata_bytes = allocation.metadata();
    let imported_metadata_size = metadata_bytes.len();
    let imported_metadata = if metadata_bytes.is_empty() {
        std::ptr::null()
    } else {
        metadata_bytes.as_ptr().cast::<c_void>()
    };
    if runtime
        .memory_state
        .interop_allocations
        .try_reserve(1)
        .is_err()
    {
        return OUT_OF_RESOURCES;
    }
    if runtime
        .memory_state
        .interop_allocations
        .contains_key(&imported_pointer)
    {
        return ERROR;
    }
    runtime.memory_state.interop_allocations.insert(
        imported_pointer,
        Memory::new_interop(allocation, owner, accessible),
    );
    // SAFETY: Required outputs were validated above. Optional metadata outputs
    // are written independently, matching the public ABI.
    unsafe {
        size.write(imported_size);
        pointer.write(imported_pointer as *mut c_void);
        if !metadata_size.is_null() {
            metadata_size.write(imported_metadata_size);
        }
        if !metadata.is_null() {
            metadata.write(imported_metadata);
        }
    }
    SUCCESS
}
