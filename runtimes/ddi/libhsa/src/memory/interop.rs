// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! IPC, descriptor, host locking, and external memory interop.

use super::*;

impl MemoryState {
    /// Reserves bookkeeping before acquiring a native registration.
    fn reserve_host_lock(&mut self, host: usize) -> Result<usize, Status> {
        let count = match self.host_locks.get(&host) {
            Some(HostLockState::Locked(count)) => count.checked_add(1).ok_or(OUT_OF_RESOURCES)?,
            Some(HostLockState::Retiring) => return Err(OUT_OF_RESOURCES),
            None => 1,
        };
        self.locked_allocations
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        self.host_locks
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        Ok(count)
    }

    /// Publishes a registration after its native acquisition succeeds.
    fn add_host_lock(&mut self, memory: LockedMemory, host: usize, count: usize) {
        self.locked_allocations.push(memory);
        self.host_locks.insert(host, HostLockState::Locked(count));
    }

    /// Releases a host-address lock, preserving every alias until the last unlock.
    pub(crate) fn unlock_host(&mut self, host: usize) -> Status {
        self.unlock_host_with(host, |memory| memory.free().map_err(map_error))
    }

    /// Keeps native cleanup injectable so failure and retirement need no GPU in tests.
    fn unlock_host_with(
        &mut self,
        host: usize,
        mut free: impl FnMut(&mut LockedMemory) -> Result<(), Status>,
    ) -> Status {
        match self.host_locks.get_mut(&host) {
            Some(HostLockState::Locked(count)) if *count > 1 => {
                *count -= 1;
                return SUCCESS;
            }
            Some(HostLockState::Locked(_)) => {
                if self.locked_allocations.iter().any(|memory| {
                    memory.description.host_base == Some(host)
                        && self.copy_borrowed(memory.description.agent_base)
                }) {
                    return OUT_OF_RESOURCES;
                }
                self.host_locks.insert(host, HostLockState::Retiring);
            }
            Some(HostLockState::Retiring) => {}
            None => return INVALID_ARGUMENT,
        }

        for memory in self
            .locked_allocations
            .iter_mut()
            .filter(|memory| memory.description.host_base == Some(host) && !memory.retired)
        {
            if let Err(status) = free(memory) {
                // Keep every alias as a rejecting tombstone, including earlier
                // successful frees. A retry only visits unfinished owners.
                return status;
            }
            memory.retired = true;
        }
        self.locked_allocations
            .retain(|memory| memory.description.host_base != Some(host));
        self.host_locks.remove(&host);
        SUCCESS
    }
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
    if runtime.agents.is_cpu_agent(memory.description.owner) {
        return INVALID_AGENT;
    }
    let Some(index) = runtime.agents.gpu_index(memory.description.owner) else {
        return INVALID_AGENT;
    };
    if flags & DMABUF_MAPPING_PCIE != 0 && !gpu_pcie_dmabuf_supported(runtime, index) {
        return NOT_SUPPORTED;
    }
    let dma_buf = match native_memory::export_dma_buf(&memory.allocation) {
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
    let exported_descriptor = dma_buf.into_descriptor();
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
        runtime
            .agents
            .cpus
            .iter()
            .position(|cpu| !cpu.apu)
            .map_or_else(
                || crate::agent::Agents::cpu_pool(0, CpuPoolKind::Fine),
                |index| crate::agent::Agents::cpu_pool(index, CpuPoolKind::Coarse),
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
    if runtime.agents.decode_cpu_pool(pool).is_none() {
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
    if runtime.agents.full_profile {
        // SAFETY: The caller supplied writable pointer storage.
        unsafe { agent_ptr.write(host_ptr) };
        return SUCCESS;
    }

    // Keep the output untouched while reading the caller's agent array: the
    // two may share storage under the public C ABI.
    let mut gpu_indexes = Vec::new();
    if agents.is_null() {
        gpu_indexes.extend(0..runtime.agents.gpus.len());
    } else {
        let count = match usize::try_from(num_agents) {
            Ok(count) => count,
            Err(_) => return INVALID_ARGUMENT,
        };
        // SAFETY: The ABI requires num_agents readable handles when agents is non-null.
        let requested = unsafe { std::slice::from_raw_parts(agents, count) };
        for agent in requested {
            if runtime.agents.is_cpu_agent(*agent) {
                continue;
            }
            let Some(index) = runtime.agents.gpu_index(*agent) else {
                return INVALID_AGENT;
            };
            if !gpu_indexes.contains(&index) {
                gpu_indexes.push(index);
            }
        }
    }

    let host_base = host_ptr as usize;
    let lock_count = match runtime.memory_state.reserve_host_lock(host_base) {
        Ok(count) => count,
        Err(status) => return status,
    };
    let (allocation, device_base) =
        if let Some((&first_index, peer_indexes)) = gpu_indexes.split_first() {
            let (allocation, page_offset) = {
                let first = &runtime.agents.gpus[first_index].device;
                let peers: Vec<_> = peer_indexes
                    .iter()
                    .map(|&index| &runtime.agents.gpus[index].device)
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
                        runtime.agents.host_page_size as u64,
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
    runtime.memory_state.add_host_lock(
        LockedMemory::new(
            allocation,
            host_base,
            device_base,
            size,
            owner,
            effective_global_flags(global_flags, flags & ALLOC_UNCACHED != 0, false),
            accessible,
        ),
        host_base,
        lock_count,
    );
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
        || !native_memory::valid_graphics_handle(interop_handle)
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
        let Some(index) = runtime.agents.gpu_index(*agent) else {
            return INVALID_AGENT;
        };
        devices.push(&runtime.agents.gpus[index].device);
        accessible.push(*agent);
    }
    let descriptor = match native_memory::duplicate_graphics_handle(interop_handle) {
        Ok(descriptor) => descriptor,
        Err(error) => return map_error(error),
    };
    let Some(session) = runtime.native_session() else {
        return ERROR;
    };
    let allocation = match native_memory::import_graphics_dma_buf(
        session,
        &devices,
        &descriptor,
        size_hint as u64,
    ) {
        Ok(allocation) => allocation,
        Err(error) => {
            return match error.kind() {
                rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
                rocddi::ErrorKind::Unsupported => NOT_SUPPORTED,
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
        .agents
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod host_lock_tests {
    use super::*;

    /// Uses metadata-only registrations to exercise lifetime without native devices.
    fn state() -> MemoryState {
        MemoryState::new(Arc::new(crate::session::SessionHealth::new()))
    }

    /// Publishes the same metadata that a successful host lock records.
    fn register(state: &mut MemoryState, host: usize, alias: usize, size: usize, gpu: u64) {
        let count = state.reserve_host_lock(host).unwrap();
        state.add_host_lock(
            LockedMemory::new(
                None,
                host,
                alias,
                size,
                HsaAgent { handle: CPU_AGENT },
                POOL_FLAG_COARSE,
                vec![HsaAgent { handle: gpu }],
            ),
            host,
            count,
        );
    }

    #[test]
    fn earlier_unlock_preserves_new_alias_and_its_copy_borrow() {
        let mut state = state();
        let host = 0x10_0000;
        let older = 0x20_0000;
        let newer = 0x30_0000;
        register(&mut state, host, older, 0x41000, GPU_AGENT_BASE);
        register(&mut state, host, newer, 0x41000, GPU_AGENT_BASE);
        state.retain_copy_borrow(newer).unwrap();
        let mut freed = Vec::new();
        assert_eq!(
            state.unlock_host_with(host, |memory| {
                freed.push(memory.description.agent_base);
                Ok(())
            }),
            SUCCESS
        );
        assert!(freed.is_empty());
        for alias in [older, newer] {
            let description = state
                .locked_pointer_description(alias + 0x290, 0x40000)
                .unwrap();
            assert_eq!(description.pointer_type, POINTER_TYPE_LOCKED);
            assert_eq!(description.agent_base, alias);
            assert_eq!(
                description.host_range(alias + 0x290, 0x40000),
                Some(host + 0x290)
            );
            assert_eq!(description.accessible.len(), 1);
            assert_eq!(description.accessible[0].handle, GPU_AGENT_BASE);
        }
        assert_eq!(state.unlock_host(host), OUT_OF_RESOURCES);
        state.release_copy_borrow(newer);
        assert_eq!(
            state.unlock_host_with(host, |memory| {
                freed.push(memory.description.agent_base);
                Ok(())
            }),
            SUCCESS
        );
        assert_eq!(freed, [older, newer]);
        assert!(state.locked_pointer_description(newer, 1).is_none());
        assert_eq!(state.unlock_host(host), INVALID_ARGUMENT);
    }

    #[test]
    fn same_alias_keeps_larger_registration_bounds_and_access() {
        let mut state = state();
        let host = 0x10_0000;
        let alias = 0x20_0000;
        register(&mut state, host, alias, 0x41000, GPU_AGENT_BASE);
        register(&mut state, host, alias, 0x1000, GPU_AGENT_BASE + 1);
        for pointer in [host, alias] {
            let description = state
                .locked_pointer_description(pointer + 0x290, 0x40000)
                .unwrap();
            assert_eq!(description.size, 0x41000);
            assert_eq!(description.accessible.len(), 1);
            assert_eq!(description.accessible[0].handle, GPU_AGENT_BASE);
            assert_eq!(
                description.host_range(pointer + 0x290, 0x40000),
                Some(host + 0x290)
            );
        }
        assert_eq!(state.unlock_host(host), SUCCESS);
        assert_eq!(
            state
                .locked_pointer_description(alias, 0x41000)
                .unwrap()
                .size,
            0x41000
        );
        assert_eq!(state.unlock_host(host), SUCCESS);
        assert!(state.locked_allocations.is_empty());
    }

    #[test]
    fn same_alias_keeps_each_live_gpu_grant() {
        let mut state = state();
        let host = 0x10_0000;
        let alias = 0x20_0000;
        register(&mut state, host, alias, 0x41000, GPU_AGENT_BASE);
        register(&mut state, host, alias, 0x1000, GPU_AGENT_BASE + 1);
        for pointer in [host, alias] {
            for (gpu, size) in [(GPU_AGENT_BASE, 0x41000), (GPU_AGENT_BASE + 1, 0x1000)] {
                let description = state
                    .locked_gpu_description(pointer + 0x80, 0x100, HsaAgent { handle: gpu })
                    .unwrap();
                assert_eq!(description.size, size);
                assert_eq!(description.accessible.len(), 1);
                assert_eq!(description.accessible[0].handle, gpu);
            }
            assert!(
                state
                    .locked_gpu_description(
                        pointer,
                        0x41000,
                        HsaAgent {
                            handle: GPU_AGENT_BASE + 1
                        },
                    )
                    .is_none()
            );
        }
        assert_eq!(state.unlock_host(host), SUCCESS);
        assert!(
            state
                .locked_gpu_description(
                    alias,
                    0x100,
                    HsaAgent {
                        handle: GPU_AGENT_BASE
                    },
                )
                .is_some()
        );
        assert_eq!(state.unlock_host(host), SUCCESS);
    }

    #[test]
    fn failed_final_unlock_retries_owners_and_keeps_rejecting_tombstones() {
        let mut state = state();
        let host = 0x10_0000;
        let aliases = [0x20_0000, 0x30_0000, 0x40_0000];
        for alias in aliases {
            register(&mut state, host, alias, 0x1000, GPU_AGENT_BASE);
        }
        assert_eq!(state.unlock_host(host), SUCCESS);
        assert_eq!(state.unlock_host(host), SUCCESS);
        let mut attempted = Vec::new();
        assert_eq!(
            state.unlock_host_with(host, |memory| {
                let alias = memory.description.agent_base;
                attempted.push(alias);
                if alias == aliases[1] {
                    Err(ERROR)
                } else {
                    Ok(())
                }
            }),
            ERROR
        );
        assert_eq!(attempted, aliases[..2]);
        assert_eq!(state.reserve_host_lock(host).err(), Some(OUT_OF_RESOURCES));
        for pointer in [host, aliases[0], aliases[1], aliases[2]] {
            let description = state.locked_pointer_description(pointer, 1).unwrap();
            assert_eq!(description.pointer_type, POINTER_TYPE_RESERVED_ADDR);
            assert!(description.accessible.is_empty());
        }

        // A successfully freed VA can be reused by an unrelated registration.
        let replacement_host = 0x50_0000;
        register(
            &mut state,
            replacement_host,
            aliases[0],
            0x1000,
            GPU_AGENT_BASE,
        );
        let replacement = state
            .locked_pointer_description(aliases[0], 0x1000)
            .unwrap();
        assert_eq!(replacement.pointer_type, POINTER_TYPE_LOCKED);
        assert_eq!(replacement.host_base, Some(replacement_host));

        attempted.clear();
        assert_eq!(
            state.unlock_host_with(host, |memory| {
                attempted.push(memory.description.agent_base);
                Ok(())
            }),
            SUCCESS
        );
        assert_eq!(attempted, aliases[1..]);
        assert!(state.locked_pointer_description(aliases[1], 1).is_none());
        assert_eq!(state.unlock_host(replacement_host), SUCCESS);
    }

    #[test]
    fn overlapping_host_ranges_with_distinct_starts_unlock_independently() {
        let mut state = state();
        register(&mut state, 0x10_0000, 0x30_0000, 0x41000, GPU_AGENT_BASE);
        register(&mut state, 0x14_0000, 0x40_0000, 0x41000, GPU_AGENT_BASE);
        assert_eq!(state.unlock_host(0x10_0000), SUCCESS);
        assert!(state.locked_pointer_description(0x30_0000, 1).is_none());
        assert_eq!(
            state
                .locked_pointer_description(0x40_0000, 0x41000)
                .unwrap()
                .host_base,
            Some(0x14_0000)
        );
        assert_eq!(state.unlock_host(0x14_0000), SUCCESS);
    }

    #[test]
    fn failed_unlock_does_not_hide_an_older_live_host_cover() {
        let mut state = state();
        register(&mut state, 0x10_0000, 0x30_0000, 0x41000, GPU_AGENT_BASE);
        register(&mut state, 0x10_1000, 0x40_0000, 0x1000, GPU_AGENT_BASE);
        assert_eq!(state.unlock_host_with(0x10_1000, |_| Err(ERROR)), ERROR);
        let live = state.locked_pointer_description(0x10_1080, 0x100).unwrap();
        assert_eq!(live.pointer_type, POINTER_TYPE_LOCKED);
        assert_eq!(live.agent_base, 0x30_0000);
        assert_eq!(live.host_range(0x10_1080, 0x100), Some(0x10_1080));
        let retired = state.locked_pointer_description(0x40_0080, 0x100).unwrap();
        assert_eq!(retired.pointer_type, POINTER_TYPE_RESERVED_ADDR);
        assert_eq!(state.unlock_host(0x10_1000), SUCCESS);
        assert_eq!(state.unlock_host(0x10_0000), SUCCESS);
    }

    #[test]
    fn lock_count_overflow_is_rejected_before_registration() {
        let mut state = state();
        state
            .host_locks
            .insert(0x10_0000, HostLockState::Locked(usize::MAX));
        assert_eq!(
            state.reserve_host_lock(0x10_0000).err(),
            Some(OUT_OF_RESOURCES)
        );
        assert!(state.locked_allocations.is_empty());
    }
}
