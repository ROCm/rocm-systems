// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive host CPU and NUMA facts shared by runtime frontends.
//!
//! A host domain describes OS topology, not an activated device or a promise
//! that the calling process may execute on its CPUs. Host observations come
//! from independent OS sources and are not one atomic hardware generation.
//! The source-qualified topology graph projects these observations into host
//! nodes and caches while retaining this richer CPU and cache-sharing view.

use crate::topology::CacheInfo;

/// Host cache kind, shared with endpoint cache information.
pub use crate::topology::CacheKind as CpuCacheKind;

#[cfg(target_os = "linux")]
mod linux;

/// CPU identity reported by the host OS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuInfo {
    /// First reported processor model name, when available.
    pub name: Option<String>,
    /// Number of processor entries reported by the OS source.
    pub compute_units: u32,
}

/// Source-qualified logical CPU identity. Linux uses its logical index;
/// Windows uses a processor group and an index within that group. Neither is
/// an APIC ID or a Windows CPU-set scheduling ID.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CpuId(u64);

impl CpuId {
    /// Constructs a Linux logical CPU identity.
    #[must_use]
    pub const fn linux(index: u32) -> Self {
        Self(index as u64)
    }

    /// Constructs a Windows group/index identity; a group has at most 64 CPUs.
    #[must_use]
    pub const fn windows(group: u16, index: u8) -> Option<Self> {
        if index < 64 {
            Some(Self((1_u64 << 63) | ((group as u64) << 6) | index as u64))
        } else {
            None
        }
    }
}

/// One explicit association between a logical CPU and its hardware APIC ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuApicId {
    /// Logical CPU ID used by sysfs CPU lists and cache directories.
    pub logical_cpu: CpuId,
    /// APIC ID used by KFD's `cpu_core_id_base` topology property.
    pub apic_id: u32,
}

/// Sorted, nonoverlapping CPU ID ranges without expanding large sysfs ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuSet {
    pub(crate) ranges: Vec<(u64, u64)>,
}

impl CpuSet {
    /// Returns the lowest CPU ID, if the set is nonempty.
    #[must_use]
    pub fn first(&self) -> Option<CpuId> {
        self.ranges.first().map(|(first, _)| CpuId(*first))
    }

    /// Returns whether this set contains the CPU ID.
    #[must_use]
    pub fn contains(&self, cpu: CpuId) -> bool {
        self.ranges
            .iter()
            .any(|(first, last)| *first <= cpu.0 && cpu.0 <= *last)
    }

    /// Returns whether two sets contain at least one common CPU ID.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        let mut left = 0;
        let mut right = 0;
        while let (Some(&(a, b)), Some(&(c, d))) = (self.ranges.get(left), other.ranges.get(right))
        {
            if a <= d && c <= b {
                return true;
            }
            if b < d {
                left += 1;
            } else {
                right += 1;
            }
        }
        false
    }
}

/// One CPU cache record reported by the host OS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuCache {
    /// Reporting CPU on Linux; lowest sharing CPU for Windows cache records.
    pub cpu: CpuId,
    /// CPUs sharing this cache, when the OS supplied a valid list.
    pub shared_cpus: Option<CpuSet>,
    /// Passive cache facts. The line size does not qualify maintenance.
    pub info: CacheInfo,
}

impl CpuCache {
    /// Returns the first CPU sharing this cache, when the list was valid.
    #[must_use]
    pub fn first_shared_cpu(&self) -> Option<CpuId> {
        self.shared_cpus.as_ref().and_then(CpuSet::first)
    }
}

/// Best-effort cache scan, including records read before an incomplete entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCaches {
    /// Complete records in native CPU and index order.
    pub records: Vec<CpuCache>,
    /// False when a cache observation could not be read or parsed.
    pub complete: bool,
}

/// Host CPU membership and caches reported for one OS NUMA node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostDomain {
    /// OS NUMA node ID. It is not a KFD topology node ordinal.
    pub numa_node: u32,
    /// CPUs listed for this node, if its list was available and valid.
    pub cpus: Option<CpuSet>,
    /// Cache records for this node's CPUs, preserving CPU and index order.
    pub caches: HostCaches,
}

/// Host topology graph observations captured without creating a session or
/// activating a device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostTopologyGraphSnapshot {
    /// System memory capacity, if the OS reported a nonzero value.
    pub memory_bytes: Option<usize>,
    /// Processor count and optional model name, when the count is complete.
    pub cpu: Option<CpuInfo>,
    /// Explicit APIC associations for cross-source matching, when available.
    pub cpu_apic_ids: Vec<CpuApicId>,
    /// CPUs reported online by a complete membership observation.
    pub online_cpus: Option<CpuSet>,
    /// Cache records reported by the host OS.
    pub caches: HostCaches,
    /// OS NUMA domains discovered without assuming node zero exists.
    pub domains: Vec<HostDomain>,
    /// False when the NUMA enumeration or a node entry was incomplete.
    pub domains_complete: bool,
}

impl HostTopologyGraphSnapshot {
    /// Resolves a KFD APIC ID only when it identifies exactly one logical CPU.
    #[must_use]
    pub fn logical_cpu_for_apic_id(&self, apic_id: u32) -> Option<CpuId> {
        let mut matches = self
            .cpu_apic_ids
            .iter()
            .filter(|cpu| cpu.apic_id == apic_id);
        let logical_cpu = matches.next()?.logical_cpu;
        matches.next().is_none().then_some(logical_cpu)
    }

    /// Finds an OS NUMA domain through an explicit APIC-to-logical association.
    #[must_use]
    pub fn domain_for_apic_id(&self, apic_id: u32) -> Option<&HostDomain> {
        let logical_cpu = self.logical_cpu_for_apic_id(apic_id)?;
        self.domains.iter().find(|domain| {
            domain
                .cpus
                .as_ref()
                .is_some_and(|cpus| cpus.contains(logical_cpu))
        })
    }

    /// Returns the OS NUMA domain with the selected ID, if discovered.
    #[must_use]
    pub fn domain(&self, numa_node: u32) -> Option<&HostDomain> {
        self.domains
            .iter()
            .find(|domain| domain.numa_node == numa_node)
    }
}

/// Captures host topology graph facts as independent, best-effort OS observations.
///
/// Missing identity and memory fields remain `None`; incomplete cache and
/// domain scans retain their complete records and report `complete = false`.
/// A later call may observe CPU hotplug or another topology change.
#[cfg(target_os = "linux")]
#[must_use]
pub fn graph_snapshot() -> HostTopologyGraphSnapshot {
    linux::graph_snapshot()
}

#[cfg(target_os = "linux")]
pub(crate) use linux::attached_numa_node_for_render;

#[cfg(windows)]
mod windows;
#[cfg(any(windows, test))]
mod windows_records;

/// Captures independent best-effort Windows host observations without a GPU.
#[cfg(windows)]
#[must_use]
pub fn graph_snapshot() -> HostTopologyGraphSnapshot {
    windows::graph_snapshot()
}
