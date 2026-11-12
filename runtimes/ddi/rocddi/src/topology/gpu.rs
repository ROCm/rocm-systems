// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU-specific passive facts and qualified transports.
//!
//! These records are captured at discovery without activating a device.
//! Backend discovery decides which transports are qualified.

/// Provider-selected execution path for the shared SDMA copy service.
/// This qualifies copy operands and completion, beyond queue creation alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCopyTransport {
    /// Direct queue with device-visible fence and checked ring retirement.
    User,
    /// Driver-mediated command submission with checked retirement.
    Kernel,
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
    /// Copy path qualified by this provider for the advertised packet dialect.
    /// Absence means that generic queue support alone does not qualify copies.
    pub copy: Option<GpuCopyTransport>,
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

/// Qualified packet and scratch dialect selected from the discovered target.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuTargetProfile {
    /// No packet dialect has been qualified for this target.
    #[default]
    Other,
    /// Qualified wave64 GFX9 targets.
    Gfx9,
    /// GFX10 encoding family.
    Gfx10,
    /// GFX11 encoding family.
    Gfx11,
    /// GFX12.0.0 packet dialect.
    Gfx1200,
    /// GFX12.0.1 packet dialect.
    Gfx1201,
    /// GFX12.5.0 packet dialect.
    Gfx1250,
}

/// Native scratch descriptor encoding for a qualified target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuScratchEncoding {
    /// Wave64 scratch descriptor with GFX9 alignment.
    Gfx9,
    /// Wave32 scratch descriptor with GFX12 alignment.
    Gfx12,
}

/// SDMA packet layout qualified for a target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSdmaDialect {
    /// GFX9 SDMA packets.
    Gfx9,
    /// GFX12.0 SDMA packets.
    Gfx1200,
    /// GFX12.5 SDMA packets.
    Gfx1250,
}

impl GpuTargetProfile {
    /// Returns the fixed scratch descriptor supported by this wavefront.
    #[must_use]
    pub const fn scratch_encoding(self, wavefront_size: u32) -> Option<GpuScratchEncoding> {
        match self {
            Self::Gfx9 if wavefront_size == 64 => Some(GpuScratchEncoding::Gfx9),
            Self::Gfx1200 | Self::Gfx1201 | Self::Gfx1250 if wavefront_size == 32 => {
                Some(GpuScratchEncoding::Gfx12)
            }
            Self::Other
            | Self::Gfx9
            | Self::Gfx10
            | Self::Gfx11
            | Self::Gfx1200
            | Self::Gfx1201
            | Self::Gfx1250 => None,
        }
    }

    /// Returns an SDMA dialect only when its packet format is implemented.
    #[must_use]
    pub const fn sdma_dialect(self) -> Option<GpuSdmaDialect> {
        match self {
            Self::Gfx9 => Some(GpuSdmaDialect::Gfx9),
            Self::Gfx1200 | Self::Gfx1201 => Some(GpuSdmaDialect::Gfx1200),
            Self::Gfx1250 => Some(GpuSdmaDialect::Gfx1250),
            Self::Other | Self::Gfx10 | Self::Gfx11 => None,
        }
    }

    /// Whether SDMA FENCE packets expose the explicit system bit.
    #[must_use]
    pub const fn sdma_fence_system(self) -> bool {
        match self {
            Self::Gfx1200 | Self::Gfx1201 | Self::Gfx1250 => true,
            Self::Other | Self::Gfx9 | Self::Gfx10 | Self::Gfx11 => false,
        }
    }

    /// Whether SDMA COPY and FENCE packets expose memory scope fields.
    #[must_use]
    pub const fn sdma_memory_scope(self) -> bool {
        match self {
            Self::Gfx1250 => true,
            Self::Other
            | Self::Gfx9
            | Self::Gfx10
            | Self::Gfx11
            | Self::Gfx1200
            | Self::Gfx1201 => false,
        }
    }

    /// Whether this target has the implemented image sampler descriptor.
    #[must_use]
    pub const fn supports_image_sampler(self) -> bool {
        match self {
            Self::Gfx1200 | Self::Gfx1201 => true,
            Self::Other | Self::Gfx9 | Self::Gfx10 | Self::Gfx11 | Self::Gfx1250 => false,
        }
    }

    /// Whether AMDF can advertise PM4 dispatch scratch on this target.
    #[must_use]
    pub const fn supports_pm4_dispatch_scratch(self) -> bool {
        match self {
            Self::Gfx11 | Self::Gfx1200 | Self::Gfx1201 | Self::Gfx1250 => true,
            Self::Other | Self::Gfx9 | Self::Gfx10 => false,
        }
    }
}

/// GPU target and enabled compute geometry captured without activation.
/// Required geometry fields must be complete and consistent before this record
/// is published; opening an endpoint never substitutes guessed dimensions.
/// Optional provider facts may be populated later.
#[allow(
    clippy::struct_excessive_bools,
    reason = "GPU topology reports independent native capabilities as flags"
)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuInfo {
    /// Exact GFX major version.
    pub gfx_major: u32,
    /// Exact GFX minor version.
    pub gfx_minor: u32,
    /// Exact GFX stepping.
    pub gfx_stepping: u32,
    /// Packet and scratch dialect resolved from the native target at discovery.
    pub target_profile: GpuTargetProfile,
    /// Globally unique immutable GPU identifier when reported by the backend.
    pub unique_id: Option<u64>,
    /// AMD GPU hive identifier, or zero when the endpoint has no hive.
    pub hive_id: u64,
    /// ASIC family identifier reported at discovery. The activated GPU view
    /// exposes the family resolved by its provider for runtime use.
    pub asic_family_id: u32,
    /// GPU timestamp frequency in hertz reported by the provider, when available.
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
