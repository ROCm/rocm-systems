// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA allocation, release, and deallocation callback operations.

use super::*;

/// Allocates native pool memory and publishes its HSA pointer metadata.
pub(crate) unsafe fn memory_pool_allocate(
    pool: HsaMemoryPool,
    size: usize,
    flags: u32,
    pointer: *mut *mut c_void,
    invalid_pool: Status,
) -> Status {
    boundary(|| {
        let (
            device,
            kind,
            rounded,
            page,
            permissions,
            owner,
            global_flags,
            uncached,
            device_index,
            in_flight,
        ) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(generation) = guard.active_generation() else {
                return NOT_INITIALIZED;
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            if pointer.is_null() || size == 0 || flags & !ALLOC_FLAGS != 0 {
                return INVALID_ARGUMENT;
            }
            let page = runtime.host_page_size;
            let rounded = match size.checked_add(page - 1).map(|value| value & !(page - 1)) {
                Some(size) => size,
                None => return INVALID_ALLOCATION,
            };
            let cpu_pool_index = runtime.decode_cpu_pool(pool).map(|(index, _)| index);
            let (device_index, kind) = if let Some(cpu_index) = cpu_pool_index {
                (
                    runtime.gpu_for_cpu(cpu_index),
                    cpu_pool_memory_kind(pool, runtime.lifetime, flags),
                )
            } else if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
                if pool_kind == GpuPoolKind::Group {
                    return INVALID_ALLOCATION;
                }
                let uncached = flags & ALLOC_UNCACHED != 0;
                // A partial BAR can map an individual public allocation, but
                // the rest of local VRAM must remain available for private
                // allocations from the same HSA pool.
                let host_visible = u64::try_from(rounded).is_ok_and(|size| {
                    size <= runtime.gpus[index].endpoint.host_visible_local_memory_bytes
                });
                (
                    index,
                    MemoryKind::DeviceLocal {
                        host_visible,
                        coherent: gpu_pool_coherent(pool_kind, uncached, flags & ALLOC_PCIE != 0),
                        extended_coherent: pool_kind == GpuPoolKind::ExtendedFine && !uncached,
                        uncached,
                        contiguous: flags & ALLOC_CONTIGUOUS != 0,
                    },
                )
            } else {
                return invalid_pool;
            };
            let maximum = if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
                if pool_kind == GpuPoolKind::Group {
                    0
                } else {
                    runtime.gpu_pool_size(index, pool_kind) & !(page - 1)
                }
            } else if cpu_pool_index.is_some() {
                runtime.system_alloc_max_bytes
            } else {
                return invalid_pool;
            };
            if rounded > maximum {
                return INVALID_ALLOCATION;
            }
            let owner = match pool_owner(runtime, pool) {
                Some(owner) => owner,
                None => return invalid_pool,
            };
            let global_flags = match pool_global_flags(runtime, pool) {
                Some(flags) => flags,
                None => return invalid_pool,
            };
            let permissions = if flags & ALLOC_EXECUTABLE != 0 {
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE
            } else {
                DeviceAccess::READ | DeviceAccess::WRITE
            };
            let uncached = flags & ALLOC_UNCACHED != 0 || matches!(kind, MemoryKind::System);
            let in_flight =
                match InFlightGeneration::enter(generation, &runtime.inflight, &runtime.health) {
                    Ok(in_flight) => in_flight,
                    Err(status) => return status,
                };
            (
                runtime.gpus[device_index].device.clone(),
                kind,
                rounded,
                page,
                permissions,
                owner,
                global_flags,
                uncached,
                device_index,
                in_flight,
            )
        };
        let allocation = match device.allocate(kind, rounded as u64, page as u64, permissions) {
            Ok(allocation) => allocation,
            Err(error) => return map_error(error),
        };
        let info = allocation.info();
        let pointer_value = info.host_address.unwrap_or(info.device_address as usize);
        let mut accessible = vec![HsaAgent {
            handle: GPU_AGENT_BASE + device_index as u64,
        }];
        if owner.handle != GPU_AGENT_BASE + device_index as u64 {
            accessible.push(owner);
        }
        let memory = Memory::new(
            allocation,
            size,
            owner,
            global_flags,
            uncached,
            flags,
            accessible,
        );
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return retire_unpublished_memory(memory, &in_flight, status),
        };
        if !in_flight.is_current(&guard) {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, INVALID_RUNTIME_STATE);
        }
        let Some(runtime) = guard.as_mut() else {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, NOT_INITIALIZED);
        };
        if runtime.memory_state.allocations.try_reserve(1).is_err()
            || runtime
                .memory_state
                .allocations
                .contains_key(&pointer_value)
        {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, OUT_OF_RESOURCES);
        }
        runtime
            .memory_state
            .allocations
            .insert(pointer_value, memory);
        // SAFETY: The caller supplied writable output storage, and the
        // allocation is registered before its address is exposed.
        unsafe { pointer.write(pointer_value as *mut c_void) };
        SUCCESS
    })
}

/// Retires an allocation that failed before frontend publication.
fn retire_unpublished_memory(
    mut memory: Memory,
    in_flight: &InFlightGeneration,
    status: Status,
) -> Status {
    match memory.free() {
        Ok(()) => status,
        Err(error) => {
            in_flight.quarantine();
            std::mem::forget(memory);
            map_error(error)
        }
    }
}
