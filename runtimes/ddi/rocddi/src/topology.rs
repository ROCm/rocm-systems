// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive hardware facts and source-qualified topology relationships.
//!
//! The records defined here describe hardware independently of a topology
//! graph: an endpoint can be opened directly, and cache, link, PCI, and GPU
//! information can be used without a node ID. `graph` binds those facts to
//! source-qualified nodes and reported relationships in one snapshot. Its
//! public types are re-exported here so callers have one topology interface.
//!
//! An endpoint is a passive discovery record. Optional facts from other
//! sources may be populated later. Inspecting or cloning one does not activate
//! hardware, acquire an address space, or grant memory access; an activated
//! endpoint is a separate session-owned `crate::device::Device`. Endpoint-kind
//! variants contain only capabilities meaningful to that kind.

use crate::driver;
use crate::memory::{DeviceAccess, HostCacheability};
use std::ops::{BitOr, BitOrAssign};

mod gpu;
mod graph;
pub mod host;
pub mod platform;
pub use gpu::{
    GpuCopyTransport, GpuInfo, GpuQueueCapabilities, GpuScratchEncoding, GpuSdmaDialect,
    GpuTargetProfile,
};
pub use graph::{
    AttachmentKind, CpuNodeInfo, EndpointAttachment, MemoryBank, MemoryBankKind, MemoryLocation,
    MemoryLocationKind, NativeNodeId, NodeId, NodeKind, ReportedRoute, SourceId, SourceKind,
    TopologyCache, TopologyGraphSnapshot, TopologyNode, TopologySource,
};

/// Exact PCI function and configuration identity reported by the OS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PciInfo {
    /// PCI segment/domain.
    pub domain: u32,
    /// PCI bus.
    pub bus: u32,
    /// PCI device number.
    pub device: u32,
    /// PCI function number.
    pub function: u32,
    /// PCI vendor ID.
    pub vendor_id: u32,
    /// PCI device ID.
    pub device_id: u32,
    /// PCI subsystem vendor ID reported by the platform.
    pub subsystem_vendor_id: u32,
    /// PCI subsystem device ID from the same platform configuration record.
    pub subsystem_device_id: u32,
    /// PCI configuration revision, distinct from the ASIC target revision.
    pub revision_id: u32,
}

/// Data and instruction classification reported by a topology source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheKind {
    /// Data cache.
    Data,
    /// Instruction cache.
    Instruction,
    /// Cache holding both data and instructions.
    Unified,
    /// Another or unknown cache type.
    Other,
}

/// Processor classes positively associated with a cache by its source.
///
/// These flags can be combined. An absent flag means the source did not report
/// that association; it does not prove the processor cannot use the cache.
/// Source-qualified nodes identify the particular cache and endpoint, while
/// these flags only classify the reported processor kinds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheAssociations(u32);

impl CacheAssociations {
    /// No processor association was reported.
    pub const NONE: Self = Self(0);
    /// The source associates this cache with CPUs.
    pub const CPU: Self = Self(1 << 0);
    /// The source associates this cache with GPU compute units.
    pub const GPU_COMPUTE_UNIT: Self = Self(1 << 1);

    /// Returns whether every association in `required` was reported.
    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
}

impl BitOr for CacheAssociations {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for CacheAssociations {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Passive cache information shared by host and endpoint topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheInfo {
    /// Hierarchy level reported by the topology source.
    pub level: u32,
    /// Cache capacity in bytes.
    pub size_bytes: u64,
    /// Cache-line size in bytes, when the topology source reports one.
    pub line_size_bytes: Option<u32>,
    /// Data and instruction classification of this cache.
    pub kind: CacheKind,
    /// Processor classes positively associated with this cache by its source.
    pub associations: CacheAssociations,
}

impl CacheInfo {
    /// Creates cache information without a reported processor association.
    #[must_use]
    pub const fn new(
        level: u32,
        size_bytes: u64,
        line_size_bytes: Option<u32>,
        kind: CacheKind,
    ) -> Self {
        Self {
            level,
            size_bytes,
            line_size_bytes,
            kind,
            associations: CacheAssociations::NONE,
        }
    }
}

/// Physical or logical interconnect kind reported by topology discovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryLinkType {
    /// `HyperTransport` link.
    HyperTransport,
    /// `QuickPath` Interconnect link.
    Qpi,
    /// PCI Express link.
    Pcie,
    /// InfiniBand link.
    Infiniband,
    /// AMD XGMI link.
    Xgmi,
    /// Link type not recognized by this runtime.
    Unknown,
}

/// Passive properties of one directed native topology link.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLinkInfo {
    /// Native minimum latency value for this direction.
    pub minimum_latency: u32,
    /// Native maximum latency value for this direction.
    pub maximum_latency: u32,
    /// Native minimum bandwidth value for this direction.
    pub minimum_bandwidth: u32,
    /// Native maximum bandwidth value for this direction.
    pub maximum_bandwidth: u32,
    /// Whether 32-bit atomics may cross this link.
    pub atomic_support_32bit: bool,
    /// Whether 64-bit atomics may cross this link.
    pub atomic_support_64bit: bool,
    /// Whether the link preserves coherent memory access.
    pub coherent_support: bool,
    /// Native interconnect classification.
    pub kind: MemoryLinkType,
    /// Native NUMA weight used for relative distance queries.
    pub numa_distance: u32,
}

impl MemoryLinkInfo {
    /// Number of native hops implied by the reported distance weight.
    #[must_use]
    pub const fn hop_count(self) -> u32 {
        if self.numa_distance == 0 { 0 } else { 1 }
    }
}

/// Kind-specific capabilities attached to one passive endpoint.
///
/// The enum is intentionally open-ended. A backend publishes only a variant
/// it can populate faithfully; it must not fabricate zero-valued GPU fields for
/// a CPU, NPU, or another kind of accelerator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum EndpointKind {
    /// A host processor endpoint.
    Cpu,
    /// An AMD GPU endpoint and its GPU-specific capabilities.
    Gpu {
        /// GPU target, compute geometry, and qualified transports.
        info: GpuInfo,
    },
    /// A neural-processing or other machine-learning accelerator endpoint.
    Npu,
    /// An endpoint whose engine kind is not represented by an earlier variant.
    Other,
}

/// Passive description tied to one native endpoint identity. Its kind and
/// memory fields describe expected implemented services. Optional information
/// may be populated later from other native sources. Activation separately
/// qualifies the installed native interface and obtains usable address-space
/// resources. Cloning this record neither activates a device nor grants access
/// to backing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Endpoint {
    /// Opaque platform identity, unique among endpoints in one session.
    pub id: [u8; 16],
    /// UTF-8 diagnostic name with NUL termination.
    pub name: [u8; 128],
    /// Optional product name for display, populated from an additional source.
    pub product_name: Option<String>,
    /// Optional PCI attachment and configuration identity. Endpoints connected
    /// through another bus or fabric report `None`.
    pub pci: Option<PciInfo>,
    /// Engine kind and kind-specific capabilities.
    pub kind: EndpointKind,
    /// Total local bytes, without counting public/private views twice.
    pub local_memory_bytes: u64,
    /// Host-visible subset of local storage.
    pub host_visible_local_memory_bytes: u64,
    /// Qualified host cache policy for host-visible local allocations. `None`
    /// leaves the physical aperture size intact but does not qualify a mapping.
    pub host_local_cacheability: Option<HostCacheability>,
    /// Native allocation page granularity expected by this implementation.
    pub allocation_granularity: u64,
    /// Width of the qualified low device virtual-address interval.
    pub address_bit_count: u32,
    /// Inclusive architectural lower device address bound.
    pub minimum_address: u64,
    /// Inclusive architectural upper device address bound.
    pub maximum_address: u64,
    /// Supported device permission bits. Every allocation requires READ and may
    /// independently request the supported WRITE and EXECUTE bits.
    pub supported_permissions: DeviceAccess,
    pub(crate) driver_instance: driver::DriverInstanceId,
    pub(crate) selector: driver::EndpointSelector,
}

impl Endpoint {
    /// Compares all discovered facts while excluding optional information
    /// populated later from other native sources.
    pub(crate) fn matches_discovery(&self, other: &Self) -> bool {
        let mut expected = self.clone();
        expected.product_name.clone_from(&other.product_name);
        if let (EndpointKind::Gpu { info: expected }, EndpointKind::Gpu { info: actual }) =
            (&mut expected.kind, &other.kind)
        {
            expected.counter_frequency_hz = actual.counter_frequency_hz;
        }
        expected == *other
    }

    /// Returns GPU-specific information when this is a GPU endpoint.
    #[must_use]
    pub const fn gpu(&self) -> Option<&GpuInfo> {
        match &self.kind {
            EndpointKind::Gpu { info } => Some(info),
            EndpointKind::Cpu | EndpointKind::Npu | EndpointKind::Other => None,
        }
    }
}
