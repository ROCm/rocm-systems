// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA pool, region, and cache projections over rocddi topology.

use super::*;

/// Checks whether a memory pool handle names a host CPU pool.
pub(super) fn cpu_pool(pool: HsaMemoryPool) -> bool {
    cpu_pool_kind(pool).is_some()
}

/// Decodes the pool class while leaving generation ownership to `HsaState`.
pub(super) fn cpu_pool_kind(pool: HsaMemoryPool) -> Option<CpuPoolKind> {
    let offset = pool.handle.checked_sub(CPU_POOL_BASE)?;
    if pool.handle >= GPU_POOL_BASE {
        return None;
    }
    CpuPoolKind::from_slot(offset % POOL_HANDLE_STRIDE)
}

/// Maps a CPU pool handle to its native memory kind.
pub(super) fn cpu_pool_memory_kind(
    pool: HsaMemoryPool,
    lifetime: DriverContextLifetime,
    flags: u32,
) -> MemoryKind {
    if cpu_pool_kind(pool) == Some(CpuPoolKind::Kernarg)
        || lifetime == DriverContextLifetime::Session
    {
        MemoryKind::System
    } else {
        MemoryKind::OwnedHost {
            policy: cpu_pool_mapping_policy(pool, flags),
        }
    }
}

/// Chooses the host mapping policy for a CPU pool and flags.
pub(super) fn cpu_pool_mapping_policy(pool: HsaMemoryPool, flags: u32) -> HostMappingPolicy {
    if cpu_pool_kind(pool) == Some(CpuPoolKind::Kernarg) || flags & ALLOC_UNCACHED != 0 {
        HostMappingPolicy::Uncached
    } else if cpu_pool_kind(pool) == Some(CpuPoolKind::ExtendedFine) {
        HostMappingPolicy::Extended
    } else if cpu_pool_kind(pool) == Some(CpuPoolKind::Fine) || flags & ALLOC_PCIE != 0 {
        HostMappingPolicy::Fine
    } else {
        HostMappingPolicy::Coarse
    }
}

/// Combines pool flags with uncached and `PCIe` allocation properties.
pub(super) fn effective_global_flags(pool_flags: u32, uncached: bool, pcie: bool) -> u32 {
    if uncached {
        (pool_flags & !(POOL_FLAG_COARSE | POOL_FLAG_EXTENDED_FINE))
            | POOL_FLAG_FINE
            | POOL_FLAG_KERNARG
    } else if pcie {
        (pool_flags & !POOL_FLAG_COARSE) | POOL_FLAG_FINE
    } else {
        pool_flags
    }
}

/// Returns the agent that owns a memory pool handle.
pub(super) fn pool_owner(runtime: &HsaState, pool: HsaMemoryPool) -> Option<HsaAgent> {
    if let Some((index, _)) = runtime.decode_cpu_pool(pool) {
        Some(HsaAgent {
            handle: CPU_AGENT + index as u64,
        })
    } else {
        runtime.decode_gpu_pool(pool).map(|(index, _)| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        })
    }
}

/// Returns the global flags advertised by a memory pool.
pub(super) fn pool_global_flags(runtime: &HsaState, pool: HsaMemoryPool) -> Option<u32> {
    let flags = match runtime.decode_cpu_pool(pool).map(|(_, kind)| kind) {
        Some(CpuPoolKind::Fine) => POOL_FLAG_FINE,
        Some(CpuPoolKind::ExtendedFine) => POOL_FLAG_EXTENDED_FINE,
        Some(CpuPoolKind::Kernarg) => POOL_FLAG_KERNARG | POOL_FLAG_FINE,
        Some(CpuPoolKind::Coarse) => POOL_FLAG_COARSE,
        _ => match runtime.decode_gpu_pool(pool) {
            Some((_, GpuPoolKind::Coarse)) => POOL_FLAG_COARSE,
            Some((_, GpuPoolKind::Fine)) => POOL_FLAG_FINE,
            Some((_, GpuPoolKind::ExtendedFine)) => POOL_FLAG_EXTENDED_FINE,
            Some((_, GpuPoolKind::Group)) => 0,
            _ => return None,
        },
    };
    Some(flags)
}

/// Enumerates the native memory pools exposed by a GPU agent.
pub(super) fn gpu_agent_pools(
    index: usize,
    fine_grain: bool,
    extended: bool,
) -> Vec<HsaMemoryPool> {
    let mut pools = Vec::with_capacity(usize::from(fine_grain) + usize::from(extended) + 2);
    pools.push(HsaState::gpu_pool(index, GpuPoolKind::Coarse));
    if extended {
        pools.push(HsaState::gpu_pool(index, GpuPoolKind::ExtendedFine));
    }
    if fine_grain {
        pools.push(HsaState::gpu_pool(index, GpuPoolKind::Fine));
    }
    pools.push(HsaState::gpu_pool(index, GpuPoolKind::Group));
    pools
}

/// Backing and coherence class represented by one HSA memory pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PoolStorage {
    /// Host system memory.
    System,
    /// GPU local memory with coarse-grain coherence.
    LocalCoarse,
    /// GPU local memory with fine-grain coherence.
    LocalFine,
    /// Per-workgroup memory.
    Group,
}

/// Determines an agent’s access mode to a memory pool.
pub(super) fn pool_access(
    same_owner: bool,
    storage: PoolStorage,
    linked: bool,
    same_hive: bool,
) -> u32 {
    if same_owner {
        POOL_ACCESS_DEFAULT
    } else if matches!(storage, PoolStorage::Group) || !linked {
        POOL_ACCESS_NEVER
    } else {
        match storage {
            PoolStorage::System | PoolStorage::LocalCoarse => POOL_ACCESS_DISALLOWED,
            PoolStorage::LocalFine if same_hive => POOL_ACCESS_DISALLOWED,
            PoolStorage::LocalFine | PoolStorage::Group => POOL_ACCESS_NEVER,
        }
    }
}

/// A partial visible VRAM aperture can back individual KFD PUBLIC allocations.
/// KFD decides whether each requested allocation fits that host-mappable pool.
fn gpu_pool_has_host_aperture(runtime: &HsaState, index: usize) -> bool {
    runtime.gpus[index].endpoint.host_visible_local_memory_bytes != 0
}

/// Queue and signal mechanisms that can address arbitrary VRAM need full BAR
/// visibility rather than a host-mappable subset.
fn gpu_pool_fully_host_visible(runtime: &HsaState, index: usize) -> bool {
    let endpoint = &runtime.gpus[index].endpoint;
    endpoint.local_memory_bytes != 0
        && endpoint.host_visible_local_memory_bytes >= endpoint.local_memory_bytes
}

#[derive(Clone, Copy)]
/// Host or device memory placement selected for signal batches.
pub(crate) enum SignalBatchPool {
    /// Host-visible signal storage allocated from a CPU pool.
    System,
    /// GPU-visible signal storage with the selected mapping policy.
    Device {
        /// Index of the GPU that owns the selected device pool.
        index: usize,
        /// Whether the selected pool requests extended coherent memory.
        extended: bool,
        /// Whether signal writes require an HDP flush.
        flush: bool,
    },
}

/// Selects a host or device pool for allocating signal batches.
pub(crate) fn signal_batch_pool(
    runtime: &HsaState,
    pool: HsaMemoryPool,
) -> Option<SignalBatchPool> {
    if matches!(
        runtime.decode_cpu_pool(pool),
        Some((_, CpuPoolKind::Fine | CpuPoolKind::Kernarg))
    ) {
        return Some(SignalBatchPool::System);
    }
    let (index, kind) = runtime.decode_gpu_pool(pool)?;
    let gpu = &runtime.gpus[index];
    let extended = kind == GpuPoolKind::ExtendedFine
        && gpu.info.gfx_major == 9
        && matches!(gpu.info.gfx_minor, 4 | 5);
    let coarse = kind == GpuPoolKind::Coarse && gpu.info.gfx_major == 12 && gpu.info.gfx_minor >= 5;
    let host_atomic_link = runtime.route_from_nearest_cpu(index).is_some_and(|link| {
        link.kind == MemoryLinkType::Xgmi && link.coherent_support && link.atomic_support_64bit
    });
    (gpu_pool_fully_host_visible(runtime, index) && (extended || coarse)).then_some(
        SignalBatchPool::Device {
            index,
            extended,
            flush: !host_atomic_link,
        },
    )
}

/// Checks DMA-BUF support for `PCIe` memory on a GPU.
pub(super) fn gpu_pcie_dmabuf_supported(runtime: &HsaState, index: usize) -> bool {
    // Full VRAM visibility supplies a large BAR. A CPU-to-GPU XGMI link is
    // the other route that can satisfy a requested PCIe DMA-BUF mapping.
    gpu_pool_fully_host_visible(runtime, index)
        || runtime
            .route_from_nearest_cpu(index)
            .is_some_and(|link| link.kind == MemoryLinkType::Xgmi)
}

/// Checks whether a GPU pool provides coherent access.
pub(super) fn gpu_pool_coherent(pool_kind: GpuPoolKind, uncached: bool, pcie: bool) -> bool {
    matches!(pool_kind, GpuPoolKind::Fine | GpuPoolKind::ExtendedFine) || uncached || pcie
}

/// Builds an agent-to-pool access and link description.
pub(super) fn agent_pool_relationship(
    runtime: &HsaState,
    agent: HsaAgent,
    owner: HsaAgent,
    storage: PoolStorage,
) -> (u32, Option<MemoryLinkInfo>) {
    let same = owner == agent;
    let requester_gpu = runtime.gpu_index(agent);
    let owner_gpu = runtime.gpu_index(owner);
    let native_node = |agent: HsaAgent| {
        runtime
            .cpu_index(agent)
            .and_then(|index| runtime.cpus[index].node)
            .or_else(|| {
                runtime.gpu_index(agent).and_then(|index| {
                    runtime
                        .topology
                        .nodes_for_endpoint(&runtime.gpus[index].endpoint)
                        .find(|node| {
                            runtime.topology.sources[node.source.0].kind
                                == rocddi::topology::SourceKind::LinuxKfd
                        })
                        .map(|node| node.id)
                })
            })
    };
    let requester_node = native_node(agent);
    let owner_node = native_node(owner);
    let same_hive = match (requester_gpu, owner_gpu) {
        (Some(requester), Some(owner)) => {
            runtime.gpus[requester].info.hive_id == runtime.gpus[owner].info.hive_id
        }
        (None, Some(owner)) => runtime.gpus[owner].info.hive_id == 0,
        _ => false,
    };
    let link = requester_node
        .zip(owner_node)
        .and_then(|(from, to)| runtime.topology.route_between_nodes(from, to));
    let same_node = requester_node.is_some() && requester_node == owner_node;
    let linked = same
        || same_node
        || match (requester_gpu, owner_gpu) {
            (Some(requester), Some(owner))
                if matches!(storage, PoolStorage::LocalCoarse | PoolStorage::LocalFine) =>
            {
                runtime.session.can_access_local_memory(
                    &runtime.gpus[requester].endpoint,
                    &runtime.gpus[owner].endpoint,
                )
            }
            _ => link.is_some_and(|link| link.hop_count() != 0),
        };
    let cpu_cannot_access_local = runtime.is_cpu_agent(agent)
        && owner_gpu.is_some_and(|index| !gpu_pool_has_host_aperture(runtime, index))
        && matches!(storage, PoolStorage::LocalCoarse | PoolStorage::LocalFine);
    let access = if cpu_cannot_access_local {
        POOL_ACCESS_NEVER
    } else if runtime.is_cpu_agent(agent) && storage == PoolStorage::System && linked {
        POOL_ACCESS_DEFAULT
    } else {
        pool_access(same, storage, linked, same_hive)
    };
    (access, link)
}

/// Converts native memory-link metadata to the HSA link record.
pub(super) fn pool_link_info(link: Option<MemoryLinkInfo>) -> PoolLinkInfo {
    let Some(link) = link else {
        return PoolLinkInfo {
            min_latency: 0,
            max_latency: 0,
            min_bandwidth: 0,
            max_bandwidth: 0,
            atomic_support_32bit: false,
            atomic_support_64bit: false,
            coherent_support: false,
            link_type: 0,
            numa_distance: 0,
        };
    };
    PoolLinkInfo {
        min_latency: link.minimum_latency,
        max_latency: link.maximum_latency,
        min_bandwidth: link.minimum_bandwidth,
        max_bandwidth: link.maximum_bandwidth,
        atomic_support_32bit: link.atomic_support_32bit,
        atomic_support_64bit: link.atomic_support_64bit,
        coherent_support: link.coherent_support,
        link_type: hsa_link_type(link.kind),
        numa_distance: link.numa_distance,
    }
}

/// Maps a native memory link kind to its HSA ABI value.
pub(super) const fn hsa_link_type(kind: MemoryLinkType) -> u32 {
    match kind {
        MemoryLinkType::HyperTransport | MemoryLinkType::Unknown => 0,
        MemoryLinkType::Qpi => 1,
        MemoryLinkType::Pcie => 2,
        MemoryLinkType::Infiniband => 3,
        MemoryLinkType::Xgmi => 4,
    }
}

/// Enumerates the memory pools visible to an HSA agent.
fn agent_pools(runtime: &HsaState, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
    if let Some(index) = runtime.cpu_index(agent) {
        Ok(CpuPoolKind::ALL
            .into_iter()
            .filter(|kind| *kind != CpuPoolKind::Coarse || !runtime.cpus[index].apu)
            .map(|kind| HsaState::cpu_pool(index, kind))
            .collect())
    } else if let Some(index) = runtime.gpu_index(agent) {
        let gpu = &runtime.gpus[index];
        let mut pools = gpu_agent_pools(
            index,
            gpu.fine_grain_pool,
            !(gpu.info.gfx_major == 12 && gpu.info.gfx_minor == 0),
        );
        pools.retain(|pool| runtime.decode_gpu_pool(*pool).is_some());
        Ok(pools)
    } else {
        Err(INVALID_AGENT)
    }
}

/// Enumerates the memory regions visible to an HSA agent.
fn agent_regions(runtime: &HsaState, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
    if !runtime.is_cpu_agent(agent) && runtime.gpu_index(agent).is_none() {
        return Err(INVALID_AGENT);
    }
    let mut regions = if runtime.is_cpu_agent(agent) {
        Vec::new()
    } else {
        agent_pools(runtime, agent)?
    };
    for cpu_index in 0..runtime.cpus.len() {
        regions.extend(agent_pools(
            runtime,
            HsaAgent {
                handle: CPU_AGENT + cpu_index as u64,
            },
        )?);
    }
    Ok(regions)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_iterate_memory_pools")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits pools visible to an HSA agent.
pub unsafe extern "C" fn hsa_amd_agent_iterate_memory_pools(
    agent: HsaAgent,
    callback: PoolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let pools = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            if callback.is_none() {
                return INVALID_ARGUMENT;
            }
            match agent_pools(runtime, agent) {
                Ok(pools) => pools,
                Err(status) => return status,
            }
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for pool in pools {
            // SAFETY: HSA defines traversal callbacks as synchronous and the
            // caller keeps its data pointer live through this call.
            let status = unsafe { callback(pool, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_iterate_regions")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits memory regions visible to an HSA agent.
pub unsafe extern "C" fn hsa_agent_iterate_regions(
    agent: HsaAgent,
    callback: RegionCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let regions = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if callback.is_none() {
                return INVALID_ARGUMENT;
            }
            match agent_regions(runtime, agent) {
                Ok(pools) => pools
                    .into_iter()
                    .map(|pool| HsaRegion {
                        handle: pool.handle,
                    })
                    .collect::<Vec<_>>(),
                Err(status) => return status,
            }
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for region in regions {
            // SAFETY: HSA region traversal is synchronous and data stays live.
            let status = unsafe { callback(region, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_iterate_caches")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Visits cache handles associated with an HSA agent.
pub unsafe extern "C" fn hsa_agent_iterate_caches(
    agent: HsaAgent,
    callback: CacheCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let caches = {
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
            let Some(_) = callback else {
                return INVALID_ARGUMENT;
            };
            cache_handles_in_level_order(&runtime.caches, agent)
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for cache in caches {
            // SAFETY: Cache traversal is synchronous and data stays live.
            let status = unsafe { callback(cache, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_cache_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested cache attribute.
pub unsafe extern "C" fn hsa_cache_get_info(
    cache: HsaCache,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(index) = runtime.cache_index(cache) else {
            return INVALID_CACHE;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let cache = &runtime.caches[index];
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                CACHE_INFO_NAME_LENGTH => value.cast::<u32>().write((cache.name.len() - 1) as u32),
                CACHE_INFO_NAME => value
                    .cast::<*const c_char>()
                    .write(cache.name.as_ptr().cast()),
                CACHE_INFO_LEVEL => value.cast::<u8>().write(cache.level),
                CACHE_INFO_SIZE => value.cast::<u32>().write(cache.size),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

/// Writes a requested pool attribute to the caller’s output.
unsafe fn memory_pool_get_info(
    runtime: &HsaState,
    pool: HsaMemoryPool,
    attribute: u32,
    value: *mut c_void,
    invalid_pool: Status,
) -> Status {
    let cpu_pool = runtime.decode_cpu_pool(pool);
    let gpu_pool = runtime.decode_gpu_pool(pool);
    if cpu_pool.is_none() && gpu_pool.is_none() {
        return invalid_pool;
    }
    let group = matches!(gpu_pool, Some((_, GpuPoolKind::Group)));
    let flags = pool_global_flags(runtime, pool).unwrap_or(0);
    let page = runtime.host_page_size;
    let size = if let Some((index, kind)) = gpu_pool {
        runtime.gpu_pool_size(index, kind)
    } else if let Some((index, _)) = cpu_pool {
        runtime.cpus[index].memory_bytes
    } else {
        return invalid_pool;
    };
    // SAFETY: Each arm writes the public value type for the attribute.
    unsafe {
        match attribute {
            POOL_INFO_SEGMENT => {
                value
                    .cast::<u32>()
                    .write(if group { SEGMENT_GROUP } else { SEGMENT_GLOBAL })
            }
            POOL_INFO_GLOBAL_FLAGS => value.cast::<u32>().write(flags),
            POOL_INFO_SIZE => value.cast::<usize>().write(size),
            POOL_INFO_ALLOC_MAX_SIZE => value.cast::<usize>().write(if group {
                0
            } else if cpu_pool.is_some() {
                runtime.system_alloc_max_bytes
            } else {
                size & !(page - 1)
            }),
            POOL_INFO_RUNTIME_ALLOC_ALLOWED => value.cast::<bool>().write(!group),
            POOL_INFO_RUNTIME_ALLOC_GRANULE | POOL_INFO_RUNTIME_ALLOC_ALIGNMENT => {
                value.cast::<usize>().write(if group { 0 } else { page })
            }
            POOL_INFO_RUNTIME_ALLOC_REC_GRANULE => value.cast::<usize>().write(if group {
                0
            } else if gpu_pool.is_some() && !runtime.disable_fragment_allocator {
                2 * 1024 * 1024
            } else {
                page
            }),
            POOL_INFO_ACCESSIBLE_BY_ALL => value.cast::<bool>().write(cpu_pool.is_some()),
            POOL_INFO_SIGNAL_CAPABLE => value
                .cast::<bool>()
                .write(signal_batch_pool(runtime, pool).is_some()),
            POOL_INFO_LOCATION => {
                if group {
                    return INVALID_ARGUMENT;
                }
                value.cast::<u32>().write(u32::from(cpu_pool.is_none()));
            }
            _ => return INVALID_ARGUMENT,
        }
    }
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested memory-pool attribute.
pub unsafe extern "C" fn hsa_amd_memory_pool_get_info(
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
        // SAFETY: The caller provided writable storage for the requested attribute.
        unsafe { memory_pool_get_info(runtime, pool, attribute, value, INVALID_MEMORY_POOL) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_region_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested memory-region attribute.
pub unsafe extern "C" fn hsa_region_get_info(
    region: HsaRegion,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let pool = HsaMemoryPool {
            handle: region.handle,
        };
        let pool_attribute = match attribute {
            POOL_INFO_SEGMENT
            | POOL_INFO_GLOBAL_FLAGS
            | POOL_INFO_SIZE
            | POOL_INFO_RUNTIME_ALLOC_ALLOWED
            | POOL_INFO_RUNTIME_ALLOC_GRANULE
            | POOL_INFO_RUNTIME_ALLOC_ALIGNMENT => Some(attribute),
            REGION_INFO_ALLOC_MAX_SIZE => Some(POOL_INFO_ALLOC_MAX_SIZE),
            _ => None,
        };
        if let Some(pool_attribute) = pool_attribute {
            // SAFETY: The caller provided writable storage for the requested attribute.
            return unsafe {
                memory_pool_get_info(runtime, pool, pool_attribute, value, INVALID_REGION)
            };
        }
        if runtime.decode_cpu_pool(pool).is_none() && runtime.decode_gpu_pool(pool).is_none() {
            return INVALID_REGION;
        }
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE => return INVALID_ARGUMENT,
                AMD_REGION_INFO_HOST_ACCESSIBLE => value
                    .cast::<bool>()
                    .write(runtime.decode_cpu_pool(pool).is_some()),
                AMD_REGION_INFO_BASE => value.cast::<*mut c_void>().write(std::ptr::null_mut()),
                AMD_REGION_INFO_BUS_WIDTH => value.cast::<u32>().write(
                    runtime
                        .pool_bank(pool)
                        .and_then(|bank| bank.width_bits)
                        .unwrap_or(0),
                ),
                AMD_REGION_INFO_MAX_CLOCK_FREQUENCY => value.cast::<u32>().write(
                    runtime
                        .pool_bank(pool)
                        .and_then(|bank| bank.maximum_clock_mhz)
                        .unwrap_or(0),
                ),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}
