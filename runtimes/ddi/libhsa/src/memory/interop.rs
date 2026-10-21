// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! IPC, descriptor, host locking, and external memory interop.

use super::*;

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_ipc_memory_create(
    pointer: *mut c_void,
    size: usize,
    handle: *mut HsaAmdIpcMemory,
) -> Status {
    boundary(|| {
        if pointer.is_null() || size == 0 || handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(memory) = runtime.memory_state.allocations.values().find(|memory| {
            memory.description.agent_base == pointer as usize
                && memory.description.size == size
                && memory.description.owner.handle != CPU_AGENT
        }) else {
            return INVALID_ARGUMENT;
        };
        let exported = match linux_interop::export_kfd_ipc_memory(&memory.allocation) {
            Ok(handle) => handle.words,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage and publication
        // occurs only after the native export completed successfully.
        unsafe { handle.write(HsaAmdIpcMemory { handle: exported }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_attach")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_ipc_memory_attach(
    handle: *const HsaAmdIpcMemory,
    size: usize,
    num_agents: u32,
    mapping_agents: *const HsaAgent,
    mapped_pointer: *mut *mut c_void,
) -> Status {
    boundary(|| {
        if handle.is_null()
            || size == 0
            || mapped_pointer.is_null()
            || (num_agents != 0 && mapping_agents.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // Read inputs before publishing output; the C ABI does not forbid
        // mapped_pointer from overlapping the handle or agent array.
        // SAFETY: The caller supplied a readable IPC handle.
        let words = unsafe { handle.read() }.handle;
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let requested = if num_agents == 0 {
            None
        } else {
            // SAFETY: The caller supplied num_agents readable entries.
            Some(unsafe { std::slice::from_raw_parts(mapping_agents, num_agents as usize) })
        };
        if requested.is_some_and(|agents| agents.iter().any(|agent| !runtime.is_agent(*agent))) {
            return INVALID_AGENT;
        }

        let mut accessible = Vec::new();
        let mut mapping_indices = Vec::new();
        if let Some(agents) = requested {
            if accessible.try_reserve(agents.len()).is_err()
                || mapping_indices.try_reserve(agents.len()).is_err()
            {
                return OUT_OF_RESOURCES;
            }
            for agent in agents {
                if !accessible.contains(agent) {
                    accessible.push(*agent);
                }
                if let Some(index) = runtime.gpu_index(*agent) {
                    if !mapping_indices.contains(&index) {
                        mapping_indices.push(index);
                    }
                }
            }
        } else {
            if accessible.try_reserve(runtime.gpus.len()).is_err()
                || mapping_indices.try_reserve(runtime.gpus.len()).is_err()
            {
                return OUT_OF_RESOURCES;
            }
            for index in 0..runtime.gpus.len() {
                accessible.push(HsaAgent {
                    handle: GPU_AGENT_BASE + index as u64,
                });
                mapping_indices.push(index);
            }
        }

        let allocation = {
            let devices = runtime
                .gpus
                .iter()
                .map(|gpu| &gpu.device)
                .collect::<Vec<_>>();
            let mapping_devices = mapping_indices
                .iter()
                .map(|index| &runtime.gpus[*index].device)
                .collect::<Vec<_>>();
            match linux_interop::import_kfd_ipc_memory(
                &runtime.session,
                &devices,
                &mapping_devices,
                IpcMemoryHandle { words },
                size as u64,
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
            }
        };
        let info = allocation.info();
        let imported_pointer = info.device_address as usize;
        let owner = runtime
            .gpus
            .iter()
            .enumerate()
            .find(|(_, gpu)| allocation.originates_from(&gpu.device))
            .map_or(HsaAgent { handle: 0 }, |(index, _)| HsaAgent {
                handle: GPU_AGENT_BASE + index as u64,
            });
        if runtime.memory_state.ipc_allocations.try_reserve(1).is_err()
            || runtime
                .memory_state
                .ipc_allocations
                .contains_key(&imported_pointer)
        {
            return OUT_OF_RESOURCES;
        }
        runtime.memory_state.ipc_allocations.insert(
            imported_pointer,
            Memory::new_ipc(allocation, size, owner, accessible),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { mapped_pointer.write(imported_pointer as *mut c_void) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_detach")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_ipc_memory_detach(mapped_pointer: *mut c_void) -> Status {
    boundary(|| {
        if mapped_pointer.is_null() {
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
        let key = mapped_pointer as usize;
        let base = runtime
            .memory_state
            .ipc_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        let Some(mut memory) = runtime.memory_state.ipc_allocations.remove(&key) else {
            return INVALID_ARGUMENT;
        };
        if memory.free().is_ok() {
            SUCCESS
        } else {
            runtime.memory_state.ipc_allocations.insert(key, memory);
            INVALID_ARGUMENT
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_external_semaphore_handle_open")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_external_semaphore_handle_open(
    agent: HsaAgent,
    descriptor: *const HsaAmdExternalSemaphoreHandleDescriptor,
    semaphore: *mut HsaAmdExternalSemaphore,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if descriptor.is_null() || semaphore.is_null() {
            return INVALID_ARGUMENT;
        }
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        NOT_SUPPORTED
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_external_semaphore_handle_close")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_external_semaphore_handle_close(
    _semaphore: HsaAmdExternalSemaphore,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        ERROR
    })
}

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

unsafe fn ais_file_io(
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
    if description.owner.handle == CPU_AGENT {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ais_file_write")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_ais_file_write(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    boundary(|| {
        // SAFETY: ais_file_io validates the public arguments and allocation.
        unsafe {
            ais_file_io(
                handle,
                device_pointer,
                size,
                file_offset,
                size_copied,
                operation_status,
                AisOperation::Write,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ais_file_read")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_ais_file_read(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    boundary(|| {
        // SAFETY: ais_file_io validates the public arguments and allocation.
        unsafe {
            ais_file_io(
                handle,
                device_pointer,
                size,
                file_offset,
                size_copied,
                operation_status,
                AisOperation::Read,
            )
        }
    })
}

unsafe fn portable_export_dmabuf(
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
    if memory.description.owner.handle == CPU_AGENT {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_export_dmabuf")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_portable_export_dmabuf(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the validated public ABI arguments.
        unsafe { portable_export_dmabuf(pointer, size, descriptor, offset, DMABUF_MAPPING_NONE) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_export_dmabuf_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_portable_export_dmabuf_v2(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
    flags: u64,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the validated public ABI arguments.
        unsafe { portable_export_dmabuf(pointer, size, descriptor, offset, flags) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_close_dmabuf")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_portable_close_dmabuf(descriptor: i32) -> Status {
    boundary(|| {
        // SAFETY: The ABI transfers responsibility for closing this descriptor.
        if unsafe { linux_interop::close_owned_descriptor(descriptor) }.is_ok() {
            SUCCESS
        } else {
            RESOURCE_FREE
        }
    })
}

pub(super) fn registered_range(
    host: usize,
    size: usize,
    page: usize,
) -> Option<(usize, usize, usize)> {
    host.checked_add(size)?;
    let page_offset = host & (page - 1);
    let extent = page_offset
        .checked_add(size)?
        .checked_add(page - 1)
        .map(|extent| extent & !(page - 1))?;
    Some((host - page_offset, extent, page_offset))
}

/// # Safety
/// `host_ptr` must cover `size` live host bytes through successful unlock,
/// `agents` must contain `num_agents` readable handles when non-null, and
/// `agent_ptr` must point to writable output storage. After an ambiguous
/// native registration failure, the caller must retain the host page cover
/// through process teardown.
unsafe fn memory_lock_to_pool(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    pool: HsaMemoryPool,
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
    if host_ptr.is_null() || size == 0 || agent_ptr.is_null() {
        return INVALID_ARGUMENT;
    }
    if num_agents < 0
        || (agents.is_null() && num_agents != 0)
        || (!agents.is_null() && num_agents == 0)
    {
        return INVALID_ARGUMENT;
    }
    if !cpu_pool(pool) {
        return INVALID_MEMORY_POOL;
    }
    let global_flags = match pool_global_flags(runtime, pool) {
        Some(flags) => flags,
        None => return INVALID_MEMORY_POOL,
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
            if agent.handle == CPU_AGENT {
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
            let Some((registered_base, native_size, page_offset)) =
                registered_range(host_base, size, runtime.host_page_size)
            else {
                return INVALID_ARGUMENT;
            };
            let allocation = {
                let first = &runtime.gpus[first_index].device;
                let peers: Vec<_> = peer_indexes
                    .iter()
                    .map(|&index| &runtime.gpus[index].device)
                    .collect();
                // SAFETY: The HSA lock caller retains the complete host page
                // cover until unlock succeeds; ambiguous native failures keep
                // the registration backing caller-owned through teardown.
                match unsafe {
                    first.register_host_with_peers(
                        &peers,
                        registered_base,
                        cpu_pool_mapping_policy(pool, flags & ALLOC_UNCACHED),
                        native_size as u64,
                        runtime.host_page_size as u64,
                        DeviceAccess::READ | DeviceAccess::WRITE,
                    )
                } {
                    Ok(allocation) => allocation,
                    Err(error) => return map_error(error),
                }
            };
            let Some(device_base) =
                (allocation.info().device_address as usize).checked_add(page_offset)
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
            effective_global_flags(global_flags, flags & ALLOC_UNCACHED != 0, false),
            accessible,
        ));
    // SAFETY: The caller supplied writable pointer storage.
    unsafe { agent_ptr.write(device_base as *mut c_void) };
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_lock(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    boundary(|| {
        // SAFETY: The HSA C caller provides live input and output storage and
        // retains the host page cover through unlock or uncertain teardown.
        unsafe {
            memory_lock_to_pool(
                host_ptr,
                size,
                agents,
                num_agents,
                HsaMemoryPool {
                    handle: CPU_POOL_COARSE,
                },
                0,
                agent_ptr,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock_to_pool")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_lock_to_pool(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    pool: HsaMemoryPool,
    flags: u32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    boundary(|| {
        // SAFETY: The HSA C caller provides live input and output storage and
        // retains the host page cover through unlock or uncertain teardown.
        unsafe { memory_lock_to_pool(host_ptr, size, agents, num_agents, pool, flags, agent_ptr) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_unlock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_unlock(host_ptr: *mut c_void) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.full_profile {
            return SUCCESS;
        }
        if host_ptr.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(index) = runtime
            .memory_state
            .locked_allocations
            .iter()
            .rposition(|memory| memory.description.host_base == Some(host_ptr as usize))
        else {
            return INVALID_ARGUMENT;
        };
        if copy_borrowed(
            runtime,
            runtime.memory_state.locked_allocations[index]
                .description
                .agent_base,
        ) {
            return OUT_OF_RESOURCES;
        }
        let mut memory = runtime.memory_state.locked_allocations.remove(index);
        match memory.free() {
            Ok(()) => SUCCESS,
            Err(error) => {
                runtime
                    .memory_state
                    .locked_allocations
                    .insert(index, memory);
                map_error(error)
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn interop_map_buffer(
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
    let allocation = match linux_interop::import_graphics_dma_buf(
        &runtime.session,
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_map_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_interop_map_buffer(
    num_agents: u32,
    agents: *mut HsaAgent,
    interop_handle: HsaHandle,
    flags: u32,
    size: *mut usize,
    pointer: *mut *mut c_void,
    metadata_size: *mut usize,
    metadata: *mut *const c_void,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the public ABI arguments with a zero size hint.
        unsafe {
            interop_map_buffer(
                num_agents,
                agents,
                interop_handle,
                flags,
                0,
                size,
                pointer,
                metadata_size,
                metadata,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_map_buffer_with_size")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_interop_map_buffer_with_size(
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
    boundary(|| {
        // SAFETY: This forwards the public ABI arguments unchanged.
        unsafe {
            interop_map_buffer(
                num_agents,
                agents,
                interop_handle,
                flags,
                size_hint,
                size,
                pointer,
                metadata_size,
                metadata,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_unmap_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_interop_unmap_buffer(pointer: *mut c_void) -> Status {
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
            return SUCCESS;
        }
        let key = pointer as usize;
        let base = runtime
            .memory_state
            .interop_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        if let Some(mut memory) = runtime.memory_state.interop_allocations.remove(&key) {
            if let Err(error) = memory.free() {
                runtime.memory_state.interop_allocations.insert(key, memory);
                return map_error(error);
            }
        }
        SUCCESS
    })
}
