// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive hardware facts and source-qualified topology relationships.
//!
//! The records defined here describe hardware independently of a topology
//! graph: an endpoint can be opened directly, and cache, link, PCI, and GPU
//! information can be used without a node ID. `model` defines the sources,
//! nodes, and relationships that bind those facts into one snapshot. Its public
//! types are re-exported here so callers have one topology interface.
//!
//! An endpoint is a passive discovery record. Optional facts from other
//! sources may be populated later. Inspecting or cloning one does not activate
//! hardware, acquire an address space, or grant memory access; an activated
//! endpoint is a separate session-owned `crate::device::Device`. Endpoint-kind
//! variants contain only capabilities meaningful to that kind.

use crate::driver;
use crate::memory::{DeviceAccess, HostCacheability};

pub mod host;
mod model;
pub mod platform;
pub use model::{
    AttachmentKind, CpuNodeInfo, EndpointAttachment, MemoryBank, MemoryBankKind, MemoryLocation,
    MemoryLocationKind, NativeNodeId, NodeId, NodeKind, ReportedRoute, SourceId, SourceKind,
    TopologyCache, TopologyNode, TopologySnapshot, TopologySource,
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
    /// Whether the source classifies this as a CPU cache.
    pub cpu_cache: bool,
    /// Whether the source marks this cache as associated with compute units.
    pub(crate) compute_unit_cache: bool,
}

impl CacheInfo {
    /// Creates cache information without a compute-unit association.
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
            cpu_cache: false,
            compute_unit_cache: false,
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

/// Queue transports qualified for one AMD GPU endpoint.
///
/// Queue formats are GPU capabilities rather than properties shared by every
/// endpoint kind. A `false` value means the current backend did not qualify the
/// corresponding transport; it does not make claims about the hardware in a
/// different operating-system or driver environment.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each independently qualified GPU queue transport is an observable capability"
)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuQueueCapabilities {
    /// Direct AQL transport is implemented for this endpoint.
    pub aql: bool,
    /// System-memory release/acquire through AQL packet fence scopes is
    /// qualified for this endpoint and host architecture.
    pub aql_system_cache_control: bool,
    /// Direct native PM4 transport is implemented for this endpoint.
    pub pm4: bool,
    /// Bounded driver-mediated PM4 submission is qualified for this endpoint.
    pub kernel_pm4: bool,
    /// System-memory release/acquire through PM4 cache controls is qualified
    /// for this endpoint and host architecture.
    pub pm4_system_cache_control: bool,
    /// Direct monotonic-64-bit SDMA transport is implemented.
    pub sdma: bool,
    /// Bounded driver-mediated SDMA submission is qualified for this endpoint.
    pub kernel_sdma: bool,
    /// System-memory release/acquire through the selected SDMA protocol is
    /// qualified for this endpoint and host architecture.
    pub sdma_system_cache_control: bool,
    /// General-purpose SDMA engines; a count does not imply independence.
    pub sdma_engine_count: u32,
}

/// GPU target and enabled compute geometry captured without activation.
/// Required geometry fields must be complete and consistent before this record
/// is published; opening an endpoint never substitutes guessed dimensions.
/// Optional DRM facts may be populated later.
#[allow(
    clippy::struct_excessive_bools,
    reason = "GPU topology reports independent native capabilities as flags"
)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuInfo {
    /// Native KFD GPU identity used to resolve imported allocation placement.
    pub native_gpu_id: u32,
    /// Exact GFX major version.
    pub gfx_major: u32,
    /// Exact GFX minor version.
    pub gfx_minor: u32,
    /// Exact GFX stepping.
    pub gfx_stepping: u32,
    /// Globally unique immutable GPU identifier when reported by the backend.
    pub unique_id: Option<u64>,
    /// AMD GPU hive identifier, or zero when the endpoint has no hive.
    pub hive_id: u64,
    /// ASIC family identifier reported by KFD topology.
    pub asic_family_id: u32,
    /// ASIC family reported by the matching DRM device, when available.
    pub drm_asic_family_id: Option<u32>,
    /// GPU timestamp frequency in hertz reported by DRM, when available.
    pub counter_frequency_hz: Option<u64>,
    /// Target ASIC revision, not PCI configuration revision.
    pub asic_revision: u32,
    /// Maximum compute-engine clock in megahertz.
    pub maximum_engine_clock_mhz: u32,
    /// VRAM interface width in bits reported by the native memory bank.
    pub memory_bus_width_bits: u32,
    /// Maximum VRAM clock in megahertz reported by the native memory bank.
    pub maximum_memory_clock_mhz: u32,
    /// Hardware wavefront lanes.
    pub wavefront_size: u32,
    /// Active compute units after harvesting.
    pub compute_unit_count: u32,
    /// SIMD units composing one compute unit.
    pub simd_count_per_compute_unit: u32,
    /// Maximum resident waves per compute unit.
    pub maximum_wave_count_per_compute_unit: u32,
    /// Maximum number of address watchpoints reported by the backend.
    pub maximum_address_watch_point_count: u32,
    /// Hardware context-save scratch-wave bound per compute unit.
    pub maximum_scratch_wave_count_per_compute_unit: u32,
    /// LDS bytes per compute unit.
    pub local_data_share_byte_length: u64,
    /// Active XCC count.
    pub xcc_count: u32,
    /// Uniform shader engines per XCC.
    pub shader_engine_count_per_xcc: u32,
    /// SIMD arrays composing one shader engine.
    pub shader_array_count_per_engine: u32,
    /// SDMA engines dedicated to XGMI transfers.
    pub sdma_xgmi_engine_count: u32,
    /// Global wave synchronization resources.
    pub gws_count: u32,
    /// The node exposes an IOMMU v2 translation agent.
    pub iommu_v2_supported: bool,
    /// Device-local memory can be coherently accessed by the host.
    pub coherent_host_access: bool,
    /// SRAM ECC mode reported by the native GPU topology.
    pub sram_ecc_enabled: bool,
    /// Packet processor firmware revision reported by the backend.
    pub packet_processor_firmware_version: u32,
    /// SDMA firmware revision reported by the backend.
    pub sdma_firmware_version: u32,
    /// Maximum persisting L2 cache reservation in bytes reported by topology.
    /// Zero means the native driver did not report support.
    pub persisting_l2_cache_size_max: u32,
    /// The native GPU exposes `UALink` fabric handles for virtual memory.
    pub fabric_handles_supported: bool,
    /// GPU queue transports qualified by the current backend.
    pub queues: GpuQueueCapabilities,
}

impl GpuInfo {
    pub(crate) const fn scratch_bytes_per_xcc(gfx_major: u32) -> u64 {
        if gfx_major >= 12 {
            8_u64 << 30
        } else {
            4_u64 << 30
        }
    }

    /// Maximum per-device scratch aperture implied by the GPU generation and
    /// active XCC count. This is an address-space bound, not a reservation.
    #[must_use]
    pub const fn maximum_scratch_aperture_bytes(&self) -> u64 {
        (self.xcc_count as u64).saturating_mul(Self::scratch_bytes_per_xcc(self.gfx_major))
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
            expected.drm_asic_family_id = actual.drm_asic_family_id;
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
