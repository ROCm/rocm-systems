// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD GPU-specific capabilities layered over an activated device.
//!
//! The root [`Device`] contract is endpoint-kind neutral. Queue transports, GPU
//! faults, trap handlers, stream performance
//! monitoring, scratch apertures, and MMIO remap pages are exposed
//! through this explicit view so CPU and NPU backends do not inherit GPU-only
//! requirements.

use crate::device::Device;
use crate::driver::GpuDriverRef;
use crate::memory::Allocation;
use crate::topology::{CacheInfo, CacheKind, GpuInfo};
use crate::{Error, ErrorKind};

mod copy;
pub mod event;
pub mod image;
pub(crate) mod kernel_queue;
pub mod profiling;
pub mod queue;
pub use copy::{
    CopyFailure, CopyRect, GpuCopyOptions, GpuCopySequence, GpuCopySignal, GpuCopySignalOperation,
    GpuCopyTimestamps, GpuCopyWait,
};

/// Services shared by cloned handles for one activated GPU.
#[derive(Default)]
pub(crate) struct GpuResources {
    /// Bounded SDMA contexts shared across clones of the activated device.
    copy_pool: copy::CopyResourcePool,
}

/// Returns whether a topology cache is a non-instruction GPU compute-unit
/// cache.
///
/// Cache records themselves remain endpoint-kind neutral. This classifier is
/// exposed here because the native type bits it interprets describe AMD GPU
/// compute-cache placement.
#[doc(hidden)]
#[must_use]
pub const fn is_compute_data_cache(cache: &CacheInfo) -> bool {
    cache.compute_unit_cache && matches!(cache.kind, CacheKind::Data | CacheKind::Other)
}

/// A borrowed GPU capability view of an activated [`Device`].
///
/// The view owns no native state and cannot outlive the device. Construct it
/// with [`Device::gpu`], which verifies the endpoint kind instead of relying on
/// a caller convention. Linux KFD currently supplies the GPU operations.
#[derive(Clone, Copy)]
pub struct GpuDevice<'a> {
    /// Activated owner from which this capability view borrows.
    pub(crate) device: &'a Device,
    /// GPU-only driver projection validated when the view was created.
    pub(crate) driver: GpuDriverRef<'a>,
    /// GPU topology and transport facts captured at activation.
    pub(crate) info: &'a GpuInfo,
    /// Shared GPU service owners retained by the activated device.
    pub(super) resources: &'a GpuResources,
}

/// Address configuration reported by the active GPU driver for surface layout.
///
/// The register value describes hardware pipe and interleave geometry. A
/// surface layout implementation must use the value from the same device that
/// will access the image; topology names alone do not determine its tiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSurfaceConfig {
    /// Hardware `GB_ADDR_CONFIG` register value.
    pub gb_addr_config: u32,
}

impl GpuDevice<'_> {
    /// Queries the active KFD process's recoverable page-fault mode.
    ///
    /// # Errors
    /// Returns a native or device-lifetime failure when the mode is unavailable.
    pub fn xnack_enabled(&self) -> Result<bool, Error> {
        self.driver.xnack_enabled()
    }

    /// Returns the GPU target, geometry, and transport capabilities captured at
    /// endpoint activation.
    #[must_use]
    pub const fn info(&self) -> &GpuInfo {
        self.info
    }

    /// Queries the active device's hardware surface address configuration.
    ///
    /// # Errors
    /// Reports an unavailable native query or device-lifetime failure.
    pub fn surface_config(&self) -> Result<GpuSurfaceConfig, Error> {
        self.driver.surface_config()
    }

    /// Returns the underlying activated device for operations that are valid
    /// for every endpoint kind.
    #[must_use]
    pub const fn device(&self) -> &Device {
        self.device
    }

    /// Returns whether this GPU and the active native kernel interface can
    /// use expert queue scheduling.
    ///
    /// # Errors
    /// Returns a native interface or device-lifetime failure.
    pub fn supports_expert_scheduling(&self) -> Result<bool, Error> {
        if self.info.gfx_major < 12 {
            return Ok(false);
        }
        self.driver.supports_expert_scheduling()
    }

    /// Requests a process-VM persisting L2 reservation for this GPU.
    ///
    /// The Linux KFD driver validates the request against its topology limit and
    /// uses the render file bound to the activated VM. A successful call changes
    /// native state; callers own any API-specific cached request value.
    ///
    /// # Errors
    /// Rejects values beyond the reported maximum and returns native failures.
    pub fn set_persisting_l2_cache_size(&self, size_bytes: u32) -> Result<(), Error> {
        if size_bytes > self.info.persisting_l2_cache_size_max {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "persisting L2 request exceeds the native limit",
            });
        }
        self.driver.set_persisting_l2_cache_size(size_bytes)
    }

    /// Returns the bytes currently available for allocation on this GPU.
    ///
    /// The value comes from the active driver context. It can change as other
    /// work allocates or releases memory; it is not a reservation.
    ///
    /// # Errors
    /// Returns native query or device-lifetime failures.
    pub fn available_memory(&self) -> Result<u64, Error> {
        self.driver.available_memory()
    }

    /// Creates device-local backing inside this GPU's scratch aperture.
    /// The returned owner retains its backing and aperture range until free.
    ///
    /// # Errors
    /// Reports invalid extents, unavailable local storage, or driver failures.
    pub fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        self.driver.allocate_queue_scratch(size)
    }

    /// Maps this GPU's process-level MMIO remap page.
    ///
    /// The allocation must remain live while any derived address is in use.
    ///
    /// # Errors
    /// Reports unsupported mappings or driver allocation and mapping failures.
    pub fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        self.driver.map_mmio_remap()
    }
}
