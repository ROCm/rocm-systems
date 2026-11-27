// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use rocddi::topology::host::{CpuCacheKind, CpuInfo};

#[test]
/// Verifies that canonical GPU agent name uses the gfx target.
fn canonical_gpu_agent_name_uses_the_gfx_target() {
    assert_eq!(
        gpu_agent_name(&GpuInfo {
            gfx_major: 12,
            gfx_minor: 0,
            gfx_stepping: 1,
            ..GpuInfo::default()
        }),
        "gfx1201"
    );
}

#[test]
/// Verifies that HDP flush registers are adjacent words in the MMIO page.
fn hdp_flush_registers_are_adjacent_words_in_the_mmio_page() {
    assert_eq!(hdp_flush_pointers(None), [0, 0]);
    assert_eq!(hdp_flush_pointers(Some(0x1000)), [0x1000, 0x1004]);
}

#[test]
/// Verifies that zero local memory identifies a full profile platform.
fn zero_local_memory_identifies_a_full_profile_platform() {
    assert!(!full_profile_platform([]));
    assert!(full_profile_platform([0]));
    assert!(full_profile_platform([0, 0]));
    assert!(!full_profile_platform([0, 1]));
}

#[test]
/// Verifies that GPU pool handle encoding preserves every pool class.
fn gpu_pool_handle_encoding_preserves_every_pool_class() {
    // Explicit slots pin the existing public handle encoding.
    for (slot, kind) in [
        (1, GpuPoolKind::Coarse),
        (2, GpuPoolKind::Fine),
        (3, GpuPoolKind::Group),
        (4, GpuPoolKind::ExtendedFine),
    ] {
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + slot,
                },
                1,
            ),
            Some((0, kind))
        );
        assert_eq!(
            crate::agent::Agents::gpu_pool(0, kind).handle,
            GPU_POOL_BASE + slot
        );
    }
    assert_eq!(
        decode_gpu_pool_handle(
            HsaMemoryPool {
                handle: GPU_POOL_BASE + 0x11,
            },
            1,
        ),
        None
    );
}

#[test]
/// Verifies that cache handles and names are stable runtime objects.
fn cache_handles_and_names_are_stable_runtime_objects() {
    let agent = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let cache = Cache::new(agent, b"gfx1201\0ignored", 3, 8_388_608);
    assert_eq!(cache.agent.handle, agent.handle);
    assert_eq!(&*cache.name, b"gfx1201 L3\0");
    assert_eq!(cache.level, 3);
    assert_eq!(cache.size, 8_388_608);
    assert_eq!(
        decode_cache_handle(HsaCache { handle: CACHE_BASE }, 2),
        Some(0)
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE + 1,
            },
            2,
        ),
        Some(1)
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE + 2,
            },
            2,
        ),
        None
    );
    assert_eq!(
        decode_cache_handle(
            HsaCache {
                handle: CACHE_BASE - 1,
            },
            2,
        ),
        None
    );
}

#[test]
/// Verifies that OS caches are not assigned to an unidentified CPU node.
fn cpu_cache_source_requires_a_unique_cpu_without_domain_evidence() {
    use rocddi::topology::host::{CpuCache, HostCaches, HostTopologyGraphSnapshot};

    let cache = |cpu| CpuCache {
        cpu,
        shared_cpus: None,
        info: rocddi::topology::CacheInfo::new(1, 32 * 1024, None, CpuCacheKind::Data),
    };
    let topology = HostTopologyGraphSnapshot {
        memory_bytes: None,
        cpu: None,
        cpu_apic_ids: Vec::new(),
        online_cpus: None,
        caches: HostCaches {
            records: vec![cache(rocddi::topology::host::CpuId::linux(7))],
            complete: true,
        },
        domains: Vec::new(),
        domains_complete: true,
    };
    assert_eq!(
        host_caches_for_cpu(&topology, None, 1).unwrap().records[0].cpu,
        rocddi::topology::host::CpuId::linux(7)
    );
    assert!(host_caches_for_cpu(&topology, None, 2).is_none());
}

#[cfg(target_os = "linux")]
#[test]
/// Verifies separate KFD CPU nodes retain their bank sizes and APU policy.
fn cpu_agents_follow_kfd_cpu_nodes_and_system_banks() {
    use rocddi::topology::host::{HostCaches, HostTopologyGraphSnapshot};
    use rocddi::topology::{
        CpuNodeInfo, MemoryBank, MemoryBankKind, NativeNodeId, NodeId, NodeKind, SourceId,
        TopologyNode,
    };

    let host = CpuInfo {
        name: Some("test CPU".into()),
        compute_units: 12,
    };
    let topology = TopologyGraphSnapshot {
        host: HostTopologyGraphSnapshot {
            memory_bytes: Some(3 << 30),
            cpu: Some(host.clone()),
            cpu_apic_ids: Vec::new(),
            online_cpus: None,
            caches: HostCaches {
                records: Vec::new(),
                complete: true,
            },
            domains: Vec::new(),
            domains_complete: true,
        },
        endpoints: Vec::new(),
        sources: Vec::new(),
        nodes: [(2, NodeKind::Host, 8), (7, NodeKind::Gpu, 4)]
            .into_iter()
            .enumerate()
            .map(|(id, (native, kind, count))| TopologyNode {
                id: NodeId(id),
                source: SourceId(0),
                native_id: NativeNodeId::LinuxKfd(native),
                kind,
                cpu: Some(CpuNodeInfo {
                    core_count: count,
                    core_id_base: None,
                    maximum_clock_mhz: None,
                }),
            })
            .collect(),
        attachments: Vec::new(),
        routes: Vec::new(),
        caches: Vec::new(),
        memory_locations: Vec::new(),
        memory_banks: [1_u64 << 30, 2_u64 << 30]
            .into_iter()
            .enumerate()
            .map(|(node, capacity_bytes)| MemoryBank {
                node: NodeId(node),
                native_ordinal: 0,
                kind: MemoryBankKind::System,
                capacity_bytes,
                native_flags: 0,
                width_bits: None,
                maximum_clock_mhz: None,
            })
            .collect(),
    };
    let cpus = cpu_agents(&topology, &host, 3 << 30);
    assert_eq!(cpus.len(), 2);
    assert_eq!(
        (
            cpus[0].native_node,
            cpus[0].compute_units,
            cpus[0].memory_bytes
        ),
        (2, 8, 1 << 30)
    );
    assert_eq!(
        (
            cpus[1].native_node,
            cpus[1].compute_units,
            cpus[1].memory_bytes
        ),
        (7, 4, 2 << 30)
    );
    assert!(!cpus[0].apu);
    assert!(cpus[1].apu);
}

#[test]
/// Verifies that cache iteration sorts levels without changing handles.
fn cache_iteration_sorts_levels_without_changing_handles() {
    let agent = HsaAgent { handle: CPU_AGENT };
    let other = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let caches = [
        Cache::new(agent, b"CPU", 3, 100),
        Cache::new(agent, b"CPU", 1, 100),
        Cache::new(agent, b"CPU", 2, 100),
        Cache::new(agent, b"CPU", 1, 100),
        Cache::new(other, b"GPU", 1, 100),
    ];
    let handles = cache_handles_in_level_order(&caches, agent)
        .into_iter()
        .map(|cache| cache.handle)
        .collect::<Vec<_>>();
    assert_eq!(
        handles,
        [CACHE_BASE + 1, CACHE_BASE + 3, CACHE_BASE + 2, CACHE_BASE]
    );
}

#[test]
/// Verifies that legacy GPU cache sizes match `ROCr` aggregation.
fn legacy_gpu_cache_sizes_match_rocr_aggregation() {
    let agent = HsaAgent {
        handle: GPU_AGENT_BASE,
    };
    let other = HsaAgent {
        handle: GPU_AGENT_BASE + 1,
    };
    let caches = [
        Cache::new(agent, b"gfx1201", 1, 32 * 1024),
        Cache::new(agent, b"gfx1201", 1, 32 * 1024),
        Cache::new(agent, b"gfx1201", 2, 256 * 1024),
        Cache::new(agent, b"gfx1201", 2, 256 * 1024),
        Cache::new(agent, b"gfx1201", 3, 8192 * 1024),
        Cache::new(other, b"gfx1201", 3, 8192 * 1024),
    ];
    assert_eq!(
        cache_sizes(&caches, agent),
        [32 * 1024, 512 * 1024, 8192 * 1024, 0]
    );
    assert_eq!(cache_sizes(&caches, other), [0, 0, 8192 * 1024, 0]);
    assert_eq!(
        cpu_cache_sizes(&caches, agent),
        [32 * 1024, 256 * 1024, 8192 * 1024, 0]
    );
}
