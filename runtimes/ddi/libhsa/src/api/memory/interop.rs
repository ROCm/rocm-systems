// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA IPC, external memory, and Linux interop C entry points.

use std::ffi::c_void;

use crate::ffi::*;
use crate::memory::interop::*;
use crate::memory::*;
use crate::platform::memory as linux_interop;
use crate::platform::memory::{
    AisFileOperation as AisOperation, KfdIpcMemoryHandle as IpcMemoryHandle,
};
use crate::runtime::{boundary, initialized_mut, lock, map_error};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Exports a tracked allocation as an IPC memory handle.
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
                && !runtime.is_cpu_agent(memory.description.owner)
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
/// Attaches shared memory from an exported IPC handle.
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
/// Detaches an imported IPC memory mapping.
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
        if runtime.memory_state.copy_borrowed(base) {
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
/// Imports an external semaphore handle for HSA queue use.
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
/// Releases an imported external semaphore handle.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ais_file_write")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes memory contents to an AIS file handle.
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
/// Reads an AIS file handle into HSA-visible memory.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_export_dmabuf")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Exports an allocation as a portable DMA-BUF descriptor.
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
/// Exports an allocation as a DMA-BUF with version-two options.
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
/// Closes an exported portable DMA-BUF descriptor.
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Pins host memory and makes it accessible to HSA agents.
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
        unsafe { memory_lock_to_pool(host_ptr, size, agents, num_agents, None, 0, agent_ptr) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock_to_pool")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Pins host memory using a selected memory pool.
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
        unsafe {
            memory_lock_to_pool(
                host_ptr,
                size,
                agents,
                num_agents,
                Some(pool),
                flags,
                agent_ptr,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_unlock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Unpins a previously locked host memory range.
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
        if runtime.memory_state.copy_borrowed(
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_map_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Maps an external graphics buffer for HSA access.
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
/// Maps an external graphics buffer with an explicit size.
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
/// Unmaps a graphics buffer imported through interop.
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
        if runtime.memory_state.copy_borrowed(base) {
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
