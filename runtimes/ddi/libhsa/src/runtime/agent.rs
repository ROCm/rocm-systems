// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA agent facts and cache projections over rocddi topology.

use super::*;
use crate::isa::gfx_target_name;

/// A transient native query failure leaves support undecided so public calls
/// can retry after the underlying condition clears.
pub(super) fn pc_sampling_supported_by_query(
    result: Result<Vec<PcSamplingConfiguration>, rocddi::Error>,
) -> bool {
    match result {
        Ok(configurations) => !configurations.is_empty(),
        Err(error) => error.kind() != rocddi::ErrorKind::Unsupported,
    }
}

/// Chooses the public HSA name for a discovered GPU.
pub(super) fn gpu_agent_name(gpu: &GpuInfo) -> String {
    gfx_target_name(gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping)
}

/// Builds the two HDP flush addresses required by the HSA ABI.
pub(super) fn hdp_flush_pointers(address: Option<usize>) -> [usize; 2] {
    address.map_or([0; 2], |address| [address, address + 4])
}

/// Checks whether all local-memory sizes permit the full HSA profile.
pub(super) fn full_profile_platform(local_memory_bytes: impl IntoIterator<Item = u64>) -> bool {
    let mut local_memory_bytes = local_memory_bytes.into_iter();
    local_memory_bytes
        .next()
        .is_some_and(|bytes| bytes == 0 && local_memory_bytes.all(|bytes| bytes == 0))
}

/// Decodes GPU pool handle into its runtime components.
pub(super) fn decode_gpu_pool_handle(
    pool: HsaMemoryPool,
    gpu_count: usize,
) -> Option<(usize, u64)> {
    let offset = pool.handle.checked_sub(GPU_POOL_BASE)?;
    let index = usize::try_from(offset / 0x10).ok()?;
    let kind = offset % 0x10;
    (index < gpu_count && matches!(kind, 1..=4)).then_some((index, kind))
}

/// Decodes cache handle into its runtime components.
pub(super) fn decode_cache_handle(cache: HsaCache, cache_count: usize) -> Option<usize> {
    let index = cache.handle.checked_sub(CACHE_BASE)?;
    let index = usize::try_from(index).ok()?;
    (index < cache_count).then_some(index)
}

/// Selects CPU caches from the available runtime resources.
pub(super) fn selected_cpu_caches(topology: &HostTopologySnapshot) -> &HostCaches {
    topology
        .domain(0)
        .map_or(&topology.caches, |domain| &domain.caches)
}

/// Returns cache handles ordered by level for an agent.
pub(crate) fn cache_handles_in_level_order(caches: &[Cache], agent: HsaAgent) -> Vec<HsaCache> {
    let mut indices = caches
        .iter()
        .enumerate()
        .filter(|(_, cache)| cache.agent == agent)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    indices.sort_by_key(|&index| caches[index].level);
    indices
        .into_iter()
        .map(|index| HsaCache {
            handle: CACHE_BASE + index as u64,
        })
        .collect()
}

/// Collects per-level cache sizes for an agent.
pub(super) fn cache_sizes(caches: &[Cache], agent: HsaAgent) -> [u32; 4] {
    let mut sizes = [0_u32; 4];
    for cache in caches.iter().filter(|cache| cache.agent == agent) {
        let Some(index) = usize::from(cache.level)
            .checked_sub(1)
            .filter(|index| *index < sizes.len())
        else {
            continue;
        };
        let bytes = cache.size;
        if index == 0 {
            if sizes[index] == 0 {
                sizes[index] = bytes;
            }
        } else {
            sizes[index] = sizes[index].saturating_add(bytes);
        }
    }
    sizes
}

/// Activated GPU and the immutable compatibility facts derived at startup.
pub(crate) struct Gpu {
    /// Passive endpoint used for identity and topology queries.
    pub(crate) endpoint: Endpoint,
    /// Cached hardware facts from passive discovery.
    pub(crate) info: GpuInfo,
    /// Process-wide recoverable page-fault mode observed at activation.
    pub(crate) xnack_enabled: bool,
    /// Activated native device and its driver resources.
    pub(crate) device: Device,
    /// HSA-visible short target name.
    pub(crate) name: Box<str>,
    /// HSA-visible product name.
    pub(crate) product_name: Box<str>,
    /// ASIC family used by image layout and target checks.
    pub(crate) asic_family_id: u32,
    /// GPU timestamp ticks per second.
    pub(crate) timestamp_frequency_hz: u64,
    /// Host addresses of native HDP flush controls, when available.
    pub(crate) hdp_flush: [usize; 2],
    /// Retains the MMIO allocation supplying those controls.
    pub(super) _mmio_remap: Option<Allocation>,
    /// HSA coherency classification derived from native capabilities.
    pub(crate) coherency_type: u32,
    /// Whether a fine-grain local pool is exposed for this GPU.
    pub(crate) fine_grain_pool: bool,
    /// Whether this target has a PC sampling path; transient query failures
    /// leave the capability visible so later operations can retry.
    pub(crate) pc_sampling_supported: bool,
    /// Current native persisting L2 reservation shared across HSA views.
    pub(crate) persisting_l2_cache_size: Arc<Mutex<usize>>,
}

/// Stable HSA cache object associated with one CPU or GPU agent.
pub(crate) struct Cache {
    /// HSA agent that reports this cache.
    pub(crate) agent: HsaAgent,
    /// Stable NUL-terminated cache name.
    pub(crate) name: Box<[u8]>,
    /// Cache hierarchy level.
    pub(crate) level: u8,
    /// Cache capacity in bytes.
    pub(crate) size: u32,
}

impl Cache {
    /// Builds a cache record with stable name bytes and level metadata.
    pub(super) fn new(agent: HsaAgent, agent_name: &[u8], level: u32, size: u32) -> Self {
        let name_length = agent_name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(agent_name.len())
            .min(63);
        let mut name = Vec::with_capacity(name_length + 13);
        name.extend_from_slice(&agent_name[..name_length]);
        name.extend_from_slice(b" L");
        name.extend_from_slice(level.to_string().as_bytes());
        name.push(0);
        Self {
            agent,
            name: name.into_boxed_slice(),
            level: level as u8,
            size,
        }
    }
}
