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
use crate::memory::Allocation;
use crate::topology::{CacheInfo, GpuInfo};
use crate::{Error, ErrorKind};

mod copy;
pub mod event;
pub(crate) mod kernel_queue;
pub mod profiling;
pub mod queue;
pub use copy::{CopyFailure, CopyRect, GpuCopySequence, GpuCopyTimestamps};

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
    const INSTRUCTION: u32 = 1 << 1;
    const GPU_COMPUTE_UNIT: u32 = 1 << 3;
    cache.kind & GPU_COMPUTE_UNIT != 0 && cache.kind & INSTRUCTION == 0
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
    /// GPU topology and transport facts captured at activation.
    pub(crate) info: &'a GpuInfo,
    /// Shared GPU service owners retained by the activated device.
    pub(super) resources: &'a GpuResources,
}

impl GpuDevice<'_> {
    /// Returns the GPU target, geometry, and transport capabilities captured at
    /// endpoint activation.
    #[must_use]
    pub const fn info(&self) -> &GpuInfo {
        self.info
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
        self.device.driver_state.supports_expert_scheduling()
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
        self.device
            .driver_state
            .set_persisting_l2_cache_size(size_bytes)
    }

    /// Returns the bytes currently available for allocation on this GPU.
    ///
    /// The value comes from the active driver context. It can change as other
    /// work allocates or releases memory; it is not a reservation.
    ///
    /// # Errors
    /// Returns native query or device-lifetime failures.
    pub fn available_memory(&self) -> Result<u64, Error> {
        self.device.driver_state.available_memory()
    }

    /// Creates device-local backing inside this GPU's scratch aperture.
    /// The returned owner retains its backing and aperture range until free.
    ///
    /// # Errors
    /// Reports invalid extents, unavailable local storage, or driver failures.
    pub fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        self.device
            .driver_state
            .allocate_queue_scratch(self.device.endpoint.driver_instance, size)
    }

    /// Maps this GPU's process-level MMIO remap page.
    ///
    /// The allocation must remain live while any derived address is in use.
    ///
    /// # Errors
    /// Reports unsupported mappings or driver allocation and mapping failures.
    pub fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        self.device
            .driver_state
            .map_mmio_remap(self.device.endpoint.driver_instance)
    }
}
