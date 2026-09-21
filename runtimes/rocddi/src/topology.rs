//! Passive hardware discovery records and topology relationships.
//!
//! An endpoint describes hardware discovered in one generation-consistent
//! topology snapshot. It is inert: inspecting or cloning an endpoint does not
//! activate hardware, acquire an address space, or grant access to memory. An
//! activated endpoint is represented separately by `crate::device::Device`.
//!
//! Endpoint-kind variants carry only the capabilities meaningful to that kind.
//! A backend publishes a variant only when it can populate and activate that
//! endpoint faithfully; an activated device remains a separate session-owned
//! state object.

use crate::driver;
use crate::host_storage::{Buffer, Shared};
use crate::memory::{DeviceAccess, HostCacheability};

pub mod platform;

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

/// Immutable native cache metadata associated with one endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheInfo {
    pub(crate) level: u32,
    pub(crate) size: u32,
    pub(crate) kind: u32,
}

impl CacheInfo {
    /// Returns the native cache level.
    #[must_use]
    pub const fn level(&self) -> u32 {
        self.level
    }

    /// Returns the cache capacity value reported by native topology.
    #[must_use]
    pub const fn size(&self) -> u32 {
        self.size
    }
}

/// Physical or logical interconnect kind reported by topology discovery.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryLinkType {
    HyperTransport,
    Qpi,
    Pcie,
    Infiniband,
    Xgmi,
    Unknown,
}

/// Immutable properties of one directed native topology link.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLinkInfo {
    pub(crate) minimum_latency: u32,
    pub(crate) maximum_latency: u32,
    pub(crate) minimum_bandwidth: u32,
    pub(crate) maximum_bandwidth: u32,
    pub(crate) atomic_support_32bit: bool,
    pub(crate) atomic_support_64bit: bool,
    pub(crate) coherent_support: bool,
    pub(crate) kind: MemoryLinkType,
    pub(crate) numa_distance: u32,
}

impl MemoryLinkInfo {
    #[doc(hidden)]
    #[must_use]
    pub const fn minimum_latency(self) -> u32 {
        self.minimum_latency
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn maximum_latency(self) -> u32 {
        self.maximum_latency
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn minimum_bandwidth(self) -> u32 {
        self.minimum_bandwidth
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn maximum_bandwidth(self) -> u32 {
        self.maximum_bandwidth
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn supports_32bit_atomics(self) -> bool {
        self.atomic_support_32bit
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn supports_64bit_atomics(self) -> bool {
        self.atomic_support_64bit
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn is_coherent(self) -> bool {
        self.coherent_support
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn kind(self) -> MemoryLinkType {
        self.kind
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn numa_distance(self) -> u32 {
        self.numa_distance
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn hop_count(self) -> u32 {
        if self.numa_distance == 0 { 0 } else { 1 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TopologyKey {
    pub(crate) group: u32,
    pub(crate) member: u32,
}

/// Provider-local directed link. Host classification is explicit rather than
/// inferred from an operating-system node or GPU identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MemoryLink {
    pub(crate) source: TopologyKey,
    pub(crate) target: TopologyKey,
    pub(crate) source_is_host: bool,
    pub(crate) target_is_host: bool,
    pub(crate) info: MemoryLinkInfo,
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
    /// Bounded DRM-mediated PM4 submission is qualified for this endpoint.
    pub kernel_pm4: bool,
    /// System-memory release/acquire through PM4 cache controls is qualified
    /// for this endpoint and host architecture.
    pub pm4_system_cache_control: bool,
    /// Direct monotonic-64-bit SDMA transport is implemented.
    pub sdma: bool,
    /// Bounded DRM-mediated SDMA submission is qualified for this endpoint.
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuInfo {
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
    /// ASIC family identifier reported by the backend.
    pub asic_family_id: u32,
    /// Target ASIC revision, not PCI configuration revision.
    pub asic_revision: u32,
    /// Maximum compute-engine clock in megahertz.
    pub maximum_engine_clock_mhz: u32,
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
    /// Native context-save scratch-wave bound per compute unit.
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
    /// Packet processor firmware revision reported by the backend.
    pub packet_processor_firmware_version: u32,
    /// SDMA firmware revision reported by the backend.
    pub sdma_firmware_version: u32,
    /// GPU queue transports qualified by the current backend.
    pub queues: GpuQueueCapabilities,
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
/// memory fields describe expected implemented services. Activation separately
/// qualifies the installed native interface and obtains usable address-space
/// resources. Cloning this record neither activates a device nor grants access
/// to backing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Endpoint {
    /// Provider-scoped opaque native identity.
    pub id: [u8; 16],
    /// UTF-8 diagnostic name with NUL termination.
    pub name: [u8; 128],
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
    pub(crate) caches: Option<Shared<Buffer<CacheInfo>>>,
    pub(crate) memory_links: Option<Shared<Buffer<MemoryLink>>>,
    pub(crate) topology_key: TopologyKey,
    pub(crate) provider_instance: u64,
    pub(crate) native: driver::EndpointSelector,
}

impl Endpoint {
    /// Returns this endpoint's kind-specific capability record.
    #[must_use]
    pub const fn kind(&self) -> &EndpointKind {
        &self.kind
    }

    /// Returns GPU-specific information when this is a GPU endpoint.
    #[must_use]
    pub const fn gpu(&self) -> Option<&GpuInfo> {
        match &self.kind {
            EndpointKind::Gpu { info } => Some(info),
            EndpointKind::Cpu | EndpointKind::Npu | EndpointKind::Other => None,
        }
    }

    /// Returns the native cache records associated with this endpoint.
    #[must_use]
    pub fn caches(&self) -> &[CacheInfo] {
        self.caches.as_ref().map_or(&[], |caches| caches.as_slice())
    }

    /// Returns the directed native link from this endpoint to another endpoint.
    #[doc(hidden)]
    #[must_use]
    pub fn memory_link_to(&self, owner: &Self) -> Option<MemoryLinkInfo> {
        if self.provider_instance != owner.provider_instance {
            return None;
        }
        self.memory_links.as_ref().and_then(|links| {
            links.iter().rev().find_map(|link| {
                (link.source == self.topology_key
                    && link.target == owner.topology_key
                    && !link.source_is_host
                    && !link.target_is_host)
                    .then_some(link.info)
            })
        })
    }

    /// Returns the directed native link from this endpoint to host memory.
    #[doc(hidden)]
    #[must_use]
    pub fn memory_link_to_host(&self) -> Option<MemoryLinkInfo> {
        self.memory_links.as_ref().and_then(|links| {
            links.iter().rev().find_map(|link| {
                (link.source == self.topology_key && !link.source_is_host && link.target_is_host)
                    .then_some(link.info)
            })
        })
    }

    /// Returns the directed native link from the host to this endpoint's memory.
    #[doc(hidden)]
    #[must_use]
    pub fn memory_link_from_host(&self) -> Option<MemoryLinkInfo> {
        self.memory_links.as_ref().and_then(|links| {
            links.iter().rev().find_map(|link| {
                (link.source_is_host && !link.target_is_host && link.target == self.topology_key)
                    .then_some(link.info)
            })
        })
    }

    /// Returns whether this consumer has a cached direct route to the owner's
    /// LOCAL memory. The owner itself is always reachable. Peer routes are
    /// accepted only from the generation-consistent topology snapshot used to
    /// construct this endpoint record.
    #[must_use]
    pub fn can_access_local_memory(&self, owner: &Self) -> bool {
        self.provider_instance == owner.provider_instance
            && ((self.topology_key == owner.topology_key)
                || self
                    .memory_link_to(owner)
                    .is_some_and(|link| link.hop_count() != 0))
    }
}
