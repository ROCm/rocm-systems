// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA agent facts and cache projections over rocddi topology.

use crate::ffi::*;

use crate::isa::gfx_target_name;
use crate::isa::isa_name;
use crate::memory::{CpuPoolKind, GpuPoolKind, POOL_HANDLE_STRIDE};
use rocddi::device::Device;
use rocddi::device::gpu::profiling::PcSamplingConfiguration;
use rocddi::memory::device::Allocation;
use rocddi::topology::host::{HostCaches, HostTopologyGraphSnapshot};
use rocddi::topology::{Endpoint, GpuInfo, MemoryBank, MemoryLinkInfo};
use rocddi::topology::{MemoryBankKind, NodeKind, TopologyGraphSnapshot};
use std::sync::{Arc, Mutex};

/// A transient native query failure leaves support undecided so public calls
/// can retry after the underlying condition clears.
pub(crate) fn pc_sampling_supported_by_query(
    result: Result<Vec<PcSamplingConfiguration>, rocddi::Error>,
) -> bool {
    match result {
        Ok(configurations) => !configurations.is_empty(),
        Err(error) => error.kind() != rocddi::ErrorKind::Unsupported,
    }
}

/// Chooses the public HSA name for a discovered GPU.
pub(crate) fn gpu_agent_name(gpu: &GpuInfo) -> String {
    gfx_target_name(gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping)
}

/// Builds the two HDP flush addresses required by the HSA ABI.
pub(crate) fn hdp_flush_pointers(address: Option<usize>) -> [usize; 2] {
    address.map_or([0; 2], |address| [address, address + 4])
}

/// Checks whether all local-memory sizes permit the full HSA profile.
pub(crate) fn full_profile_platform(local_memory_bytes: impl IntoIterator<Item = u64>) -> bool {
    let mut local_memory_bytes = local_memory_bytes.into_iter();
    local_memory_bytes
        .next()
        .is_some_and(|bytes| bytes == 0 && local_memory_bytes.all(|bytes| bytes == 0))
}

/// Decodes GPU pool handle into its runtime components.
pub(crate) fn decode_gpu_pool_handle(
    pool: HsaMemoryPool,
    gpu_count: usize,
) -> Option<(usize, GpuPoolKind)> {
    let offset = pool.handle.checked_sub(GPU_POOL_BASE)?;
    let index = usize::try_from(offset / POOL_HANDLE_STRIDE).ok()?;
    let kind = GpuPoolKind::from_slot(offset % POOL_HANDLE_STRIDE)?;
    (index < gpu_count).then_some((index, kind))
}

/// Decodes cache handle into its runtime components.
pub(crate) fn decode_cache_handle(cache: HsaCache, cache_count: usize) -> Option<usize> {
    let index = cache.handle.checked_sub(CACHE_BASE)?;
    let index = usize::try_from(index).ok()?;
    (index < cache_count).then_some(index)
}

/// Selects OS cache records only when a CPU node can identify their domain.
pub(crate) fn host_caches_for_cpu(
    topology: &HostTopologyGraphSnapshot,
    core_id_base: Option<u32>,
    cpu_count: usize,
) -> Option<&HostCaches> {
    if let Some(domain) = core_id_base.and_then(|base| topology.domain_for_apic_id(base)) {
        return Some(&domain.caches);
    }
    (cpu_count == 1).then_some(&topology.caches)
}

/// CPU agent built from one KFD CPU-bearing node or an OS fallback.
pub(crate) struct Cpu {
    /// Source-qualified native node, absent only when KFD reports no CPU node.
    pub(crate) node: Option<rocddi::topology::NodeId>,
    /// KFD node ordinal used by HSA node queries.
    pub(crate) native_node: u32,
    /// Stable public CPU name.
    pub(crate) name: Box<str>,
    /// CPU cores reported for this native node.
    pub(crate) compute_units: u32,
    /// Maximum native CPU clock in MHz, when reported.
    pub(crate) maximum_clock_mhz: u32,
    /// Capacity of its reported system memory bank.
    pub(crate) memory_bytes: usize,
    /// Whether the same native node also reports a GPU.
    pub(crate) apu: bool,
}

/// Builds one CPU agent per KFD node that actually reports CPU cores.
pub(crate) fn cpu_agents(
    topology: &TopologyGraphSnapshot,
    host: &rocddi::topology::host::CpuInfo,
    host_memory_bytes: usize,
) -> Vec<Cpu> {
    let mut cpus = topology
        .nodes
        .iter()
        .filter_map(|node| {
            let cpu = node.cpu?;
            let native_node = crate::platform::native_cpu_node_ordinal(node)?;
            let memory_bytes = topology
                .banks_for_node(node.id)
                .find(|bank| bank.kind == MemoryBankKind::System && bank.capacity_bytes != 0)
                .map_or(0, |bank| {
                    usize::try_from(bank.capacity_bytes).unwrap_or(usize::MAX)
                });
            Some(Cpu {
                node: Some(node.id),
                native_node,
                name: host.name.as_deref().unwrap_or("CPU").into(),
                compute_units: cpu.core_count,
                maximum_clock_mhz: cpu.maximum_clock_mhz.unwrap_or(0),
                memory_bytes,
                apu: node.kind == NodeKind::Gpu,
            })
        })
        .collect::<Vec<_>>();
    if cpus.is_empty() {
        cpus.push(Cpu {
            node: None,
            native_node: 0,
            name: host.name.as_deref().unwrap_or("CPU").into(),
            compute_units: host.compute_units,
            maximum_clock_mhz: 0,
            memory_bytes: host_memory_bytes,
            apu: false,
        });
    }
    cpus
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
pub(crate) fn cache_sizes(caches: &[Cache], agent: HsaAgent) -> [u32; 4] {
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

/// Projects CPU cache-size attributes using `ROCr`'s last-record-per-level rule.
pub(crate) fn cpu_cache_sizes(caches: &[Cache], agent: HsaAgent) -> [u32; 4] {
    let mut sizes = [0_u32; 4];
    for cache in caches.iter().filter(|cache| cache.agent == agent) {
        if let Some(index) = usize::from(cache.level)
            .checked_sub(1)
            .filter(|index| *index < sizes.len())
        {
            sizes[index] = cache.size;
        }
    }
    sizes
}

/// Activated GPU and its HSA-specific compatibility state.
pub(crate) struct Gpu {
    /// Target-resolved scratch allocation and reclamation policy.
    pub(crate) scratch: Arc<crate::queue::ScratchPolicy>,
    /// Process CU restriction after visible-agent reordering.
    pub(crate) global_cu_mask: Option<Vec<u32>>,
    /// Confirmed CP capabilities; failed probes remain retryable.
    pub(crate) cp_caps: Arc<Mutex<Option<u32>>>,
    /// Shared measured profiling map for this activated GPU's counter epoch.
    pub(crate) clock: Arc<crate::clock::GpuClock>,
    /// Passive endpoint used for identity and topology queries.
    pub(crate) endpoint: Endpoint,
    /// Process-wide recoverable page-fault mode observed at activation.
    pub(crate) xnack_enabled: bool,
    /// Activated native device and its driver resources.
    pub(crate) device: Device,
    /// HSA-visible short target name.
    pub(crate) name: Box<str>,
    /// Host addresses of native HDP flush controls, when available.
    pub(crate) hdp_flush: [usize; 2],
    /// Retains the MMIO allocation supplying those controls.
    pub(crate) _mmio_remap: Option<Allocation>,
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

impl Gpu {
    /// Reads immutable GPU facts from the populated native endpoint.
    #[allow(
        clippy::expect_used,
        reason = "Gpu is constructed only after confirming an activated GPU endpoint"
    )]
    pub(crate) fn info(&self) -> GpuInfo {
        *self.endpoint.gpu().expect("activated HSA GPU endpoint")
    }

    /// Returns the HSA display name, including the public fallback.
    pub(crate) fn product_name(&self) -> &str {
        self.endpoint
            .product_name
            .as_deref()
            .unwrap_or("AMD Radeon Graphics")
    }

    /// Returns the provider-resolved family from the activated device.
    #[allow(
        clippy::expect_used,
        reason = "Gpu is constructed only after confirming an activated GPU endpoint"
    )]
    pub(crate) fn asic_family_id(&self) -> u32 {
        self.device
            .gpu()
            .expect("activated HSA GPU endpoint")
            .asic_family_id()
    }

    /// Returns the optional native timestamp frequency as an HSA value.
    pub(crate) fn timestamp_frequency_hz(&self) -> u64 {
        self.info().counter_frequency_hz.unwrap_or(0)
    }
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
    pub(crate) fn new(agent: HsaAgent, agent_name: &[u8], level: u32, size: u32) -> Self {
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

/// Agent projections and activated devices belonging to one HSA generation.
pub(crate) struct Agents {
    /// System page size used by host memory pools.
    pub(crate) host_page_size: usize,
    /// Maximum system allocation across all reported CPU memory banks.
    pub(crate) system_alloc_max_bytes: usize,
    /// Passive CPU agents derived from KFD nodes or a host fallback.
    pub(crate) cpus: Vec<Cpu>,
    /// Whether all exposed agents support the full HSA profile.
    pub(crate) full_profile: bool,
    /// CPU and GPU cache records exposed through HSA queries.
    pub(crate) caches: Vec<Cache>,
    /// Passive node, cache, and reported-route facts for this HSA generation.
    pub(crate) topology: TopologyGraphSnapshot,
    /// Activated GPU agents and their native owners.
    pub(crate) gpus: Vec<Gpu>,
}

impl Agents {
    /// Returns the GPU index encoded in a valid HSA agent handle.
    pub(crate) fn gpu_index(&self, agent: HsaAgent) -> Option<usize> {
        let offset = agent.handle.checked_sub(GPU_AGENT_BASE)?;
        let index = usize::try_from(offset).ok()?;
        (index < self.gpus.len()).then_some(index)
    }

    /// Returns the CPU index encoded in a valid HSA agent handle.
    pub(crate) fn cpu_index(&self, agent: HsaAgent) -> Option<usize> {
        let offset = agent.handle.checked_sub(CPU_AGENT)?;
        let index = usize::try_from(offset).ok()?;
        (index < self.cpus.len()).then_some(index)
    }

    /// Checks whether the handle names a CPU agent in this generation.
    pub(crate) fn is_cpu_agent(&self, agent: HsaAgent) -> bool {
        self.cpu_index(agent).is_some()
    }

    /// Selects the CPU node nearest a GPU by exact KFD node routes.
    pub(crate) fn nearest_cpu_agent(&self, endpoint: &Endpoint) -> HsaAgent {
        let gpu_node = self.topology.native_nodes_for_endpoint(endpoint).next();
        let best = self.cpus.iter().enumerate().min_by_key(|(index, cpu)| {
            let distance = match (gpu_node, cpu.node) {
                (Some(gpu), Some(node)) if gpu.id == node => Some((0, 0)),
                (Some(gpu), Some(node)) => self
                    .topology
                    .route_between_nodes(gpu.id, node)
                    .map(|route| (route.hop_count(), route.numa_distance)),
                _ => None,
            };
            (
                distance.is_none(),
                distance.unwrap_or((u32::MAX, u32::MAX)),
                *index,
            )
        });
        HsaAgent {
            handle: CPU_AGENT + best.map_or(0, |(index, _)| index as u64),
        }
    }

    /// Selects a native GPU allocation path nearest the CPU pool owner.
    pub(crate) fn gpu_for_cpu(&self, cpu_index: usize) -> usize {
        let cpu_node = self.cpus[cpu_index].node;
        self.gpus
            .iter()
            .enumerate()
            .min_by_key(|(index, gpu)| {
                let gpu_node = self
                    .topology
                    .native_nodes_for_endpoint(&gpu.endpoint)
                    .next();
                let distance = match (cpu_node, gpu_node) {
                    (Some(cpu), Some(gpu)) if cpu == gpu.id => Some((0, 0)),
                    (Some(cpu), Some(gpu)) => self
                        .topology
                        .route_between_nodes(cpu, gpu.id)
                        .or_else(|| self.topology.route_between_nodes(gpu.id, cpu))
                        .map(|route| (route.hop_count(), route.numa_distance)),
                    _ => None,
                };
                (
                    distance.is_none(),
                    distance.unwrap_or((u32::MAX, u32::MAX)),
                    *index,
                )
            })
            .map_or(0, |(index, _)| index)
    }

    /// Directed KFD route from the nearest CPU node to a GPU agent.
    pub(crate) fn route_from_nearest_cpu(&self, gpu_index: usize) -> Option<MemoryLinkInfo> {
        let endpoint = &self.gpus[gpu_index].endpoint;
        let cpu_index = self.cpu_index(self.nearest_cpu_agent(endpoint))?;
        let cpu_node = self.cpus[cpu_index].node?;
        let gpu_node = self.topology.native_nodes_for_endpoint(endpoint).next()?;
        self.topology.route_between_nodes(cpu_node, gpu_node.id)
    }

    /// Checks whether a handle names the host CPU or a discovered GPU.
    pub(crate) fn is_agent(&self, agent: HsaAgent) -> bool {
        self.is_cpu_agent(agent) || self.gpu_index(agent).is_some()
    }

    /// Returns the cache index encoded in a valid HSA cache handle.
    pub(crate) fn cache_index(&self, cache: HsaCache) -> Option<usize> {
        decode_cache_handle(cache, self.caches.len())
    }

    /// Returns cache sizes by level for the requested agent.
    pub(crate) fn agent_cache_sizes(&self, agent: HsaAgent) -> [u32; 4] {
        if self.is_cpu_agent(agent) {
            cpu_cache_sizes(&self.caches, agent)
        } else {
            cache_sizes(&self.caches, agent)
        }
    }

    /// Decodes an ISA handle into GPU index and variant.
    pub(crate) fn isa_parts(&self, isa: HsaIsa) -> Option<(usize, u64)> {
        let offset = isa.handle.checked_sub(ISA_BASE)?;
        let index = usize::try_from(offset / ISA_COUNT_PER_GPU).ok()?;
        let variant = offset % ISA_COUNT_PER_GPU;
        let gpu = self.gpus.get(index)?;
        isa_name(&gpu.info(), gpu.xnack_enabled, variant).map(|_| (index, variant))
    }

    /// Returns the GPU index associated with a valid ISA handle.
    pub(crate) fn isa_index(&self, isa: HsaIsa) -> Option<usize> {
        self.isa_parts(isa).map(|(index, _)| index)
    }

    /// Finds the ISA handle associated with a wavefront handle.
    pub(crate) fn wavefront_isa(&self, wavefront: HsaWavefront) -> Option<HsaIsa> {
        let offset = wavefront.handle.checked_sub(WAVEFRONT_BASE)?;
        let isa = HsaIsa {
            handle: ISA_BASE.checked_add(offset)?,
        };
        self.isa_parts(isa).map(|_| isa)
    }

    /// Builds a synthetic memory-pool handle for a CPU and pool kind.
    pub(crate) fn cpu_pool(index: usize, kind: CpuPoolKind) -> HsaMemoryPool {
        HsaMemoryPool {
            handle: CPU_POOL_BASE + (index as u64) * POOL_HANDLE_STRIDE + kind as u64,
        }
    }

    /// Decodes a CPU pool and checks that its owner and kind are exposed.
    pub(crate) fn decode_cpu_pool(&self, pool: HsaMemoryPool) -> Option<(usize, CpuPoolKind)> {
        let offset = pool.handle.checked_sub(CPU_POOL_BASE)?;
        let index = usize::try_from(offset / POOL_HANDLE_STRIDE).ok()?;
        let kind = CpuPoolKind::from_slot(offset % POOL_HANDLE_STRIDE)?;
        let cpu = self.cpus.get(index)?;
        (kind != CpuPoolKind::Coarse || !cpu.apu).then_some((index, kind))
    }

    /// Builds a synthetic memory-pool handle for a GPU and pool kind.
    pub(crate) fn gpu_pool(index: usize, kind: GpuPoolKind) -> HsaMemoryPool {
        HsaMemoryPool {
            handle: GPU_POOL_BASE + (index as u64) * POOL_HANDLE_STRIDE + kind as u64,
        }
    }

    /// Capacity of a reported GPU pool, using a native bank when unambiguous.
    pub(crate) fn gpu_pool_size(&self, index: usize, kind: GpuPoolKind) -> usize {
        let gpu = &self.gpus[index];
        let bank_kind = if kind == GpuPoolKind::Group {
            MemoryBankKind::Lds
        } else {
            MemoryBankKind::FrameBufferPublic
        };
        let mut banks = self
            .topology
            .native_nodes_for_endpoint(&gpu.endpoint)
            .flat_map(|node| self.topology.banks_for_node(node.id))
            .filter(|bank| {
                bank.capacity_bytes != 0
                    && (bank.kind == bank_kind
                        || (kind != GpuPoolKind::Group
                            && bank.kind == MemoryBankKind::FrameBufferPrivate))
            });
        if let Some(bank) = banks.next()
            && banks.next().is_none()
        {
            return usize::try_from(bank.capacity_bytes).unwrap_or(usize::MAX);
        }
        // The allocation API cannot select a bank ordinal. For multiple
        // framebuffer reports use the driver's aggregate capacity instead.
        let bytes = if kind == GpuPoolKind::Group {
            gpu.info().local_data_share_byte_length
        } else {
            gpu.endpoint.local_memory_bytes
        };
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }

    /// Finds the native bank that supplies region width and clock facts.
    pub(crate) fn pool_bank(&self, pool: HsaMemoryPool) -> Option<&MemoryBank> {
        if let Some((index, _)) = self.decode_cpu_pool(pool) {
            let node = self.cpus[index].node?;
            return self
                .topology
                .banks_for_node(node)
                .find(|bank| bank.kind == MemoryBankKind::System && bank.capacity_bytes != 0);
        }
        let (index, kind) = self.decode_gpu_pool(pool)?;
        let node = self
            .topology
            .native_nodes_for_endpoint(&self.gpus[index].endpoint)
            .next()?;
        self.topology.banks_for_node(node.id).find(|bank| {
            bank.capacity_bytes != 0
                && if kind == GpuPoolKind::Group {
                    bank.kind == MemoryBankKind::Lds
                } else {
                    matches!(
                        bank.kind,
                        MemoryBankKind::FrameBufferPublic | MemoryBankKind::FrameBufferPrivate
                    )
                }
        })
    }

    /// Decodes GPU pool into its runtime components.
    pub(crate) fn decode_gpu_pool(&self, pool: HsaMemoryPool) -> Option<(usize, GpuPoolKind)> {
        let (index, kind) = decode_gpu_pool_handle(pool, self.gpus.len())?;
        let gpu = &self.gpus[index];
        let local = self.gpu_pool_size(index, kind) != 0;
        let exposed = match kind {
            GpuPoolKind::Fine => local && gpu.fine_grain_pool,
            GpuPoolKind::ExtendedFine => {
                local && !(gpu.info().gfx_major == 12 && gpu.info().gfx_minor == 0)
            }
            GpuPoolKind::Coarse | GpuPoolKind::Group => local,
        };
        exposed.then_some((index, kind))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
