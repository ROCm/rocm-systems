// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive host CPU and NUMA facts shared by runtime frontends.
//!
//! A host domain describes OS topology, not an activated device or a promise
//! that the calling process may execute on its CPUs. Host observations come
//! from independent OS files and are not one atomic hardware generation.
//! The source-qualified topology model projects these observations into host
//! nodes and caches while retaining this richer CPU and cache-sharing view.

use crate::topology::CacheInfo;

/// Host cache kind, shared with endpoint cache information.
pub use crate::topology::CacheKind as CpuCacheKind;

#[cfg(target_os = "linux")]
mod linux;

/// CPU identity reported by the host OS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuInfo {
    /// First reported processor model name.
    pub name: String,
    /// Number of processor entries reported by the OS source.
    pub compute_units: u32,
}

/// Sorted, nonoverlapping CPU ID ranges without expanding large sysfs ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuSet {
    pub(crate) ranges: Vec<(u32, u32)>,
}

impl CpuSet {
    /// Returns the lowest CPU ID, if the set is nonempty.
    #[must_use]
    pub fn first(&self) -> Option<u32> {
        self.ranges.first().map(|(first, _)| *first)
    }

    /// Returns whether this set contains the CPU ID.
    #[must_use]
    pub fn contains(&self, cpu: u32) -> bool {
        self.ranges
            .iter()
            .any(|(first, last)| *first <= cpu && cpu <= *last)
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
    /// CPU directory that reported this cache.
    pub cpu: u32,
    /// CPUs sharing this cache, when the OS supplied a valid list.
    pub shared_cpus: Option<CpuSet>,
    /// Passive cache facts. The line size does not qualify maintenance.
    pub info: CacheInfo,
}

impl CpuCache {
    /// Returns the first CPU sharing this cache, when the list was valid.
    #[must_use]
    pub fn first_shared_cpu(&self) -> Option<u32> {
        self.shared_cpus.as_ref().and_then(CpuSet::first)
    }
}

/// Best-effort cache scan, including records read before an incomplete entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCaches {
    /// Complete records in native CPU and index order.
    pub records: Vec<CpuCache>,
    /// False when a directory or cache property could not be read or parsed.
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

/// Passive host facts read without creating a session or activating a device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostTopologySnapshot {
    /// System memory capacity, if the OS reported a nonzero value.
    pub memory_bytes: Option<usize>,
    /// CPU model and processor-entry count, if both were reported.
    pub cpu: Option<CpuInfo>,
    /// CPUs reported online when this file was read, if its list was valid.
    pub online_cpus: Option<CpuSet>,
    /// Cache records from the system CPU directory.
    pub caches: HostCaches,
    /// OS NUMA domains discovered without assuming node zero exists.
    pub domains: Vec<HostDomain>,
    /// False when the NUMA directory or a node entry could not be read.
    pub domains_complete: bool,
}

impl HostTopologySnapshot {
    /// Returns the OS NUMA domain with the selected ID, if discovered.
    #[must_use]
    pub fn domain(&self, numa_node: u32) -> Option<&HostDomain> {
        self.domains
            .iter()
            .find(|domain| domain.numa_node == numa_node)
    }
}

/// Reads current host topology as independent, best-effort OS observations.
///
/// Missing identity and memory fields remain `None`; incomplete cache and
/// domain scans retain their complete records and report `complete = false`.
/// A later call may observe CPU hotplug or another topology change.
#[cfg(target_os = "linux")]
#[must_use]
pub fn snapshot() -> HostTopologySnapshot {
    linux::snapshot()
}

#[cfg(target_os = "linux")]
pub(crate) use linux::attached_numa_node_for_render;
