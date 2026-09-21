//! AMD GPU-specific capabilities layered over an activated device.
//!
//! The root [`Device`] contract is endpoint-kind neutral. Queue transports, GPU
//! faults, trap handlers, stream performance
//! monitoring, PC sampling, scratch apertures, and MMIO remap pages are exposed
//! through this explicit view so CPU and NPU backends do not inherit GPU-only
//! requirements.

use crate::device::Device;
use crate::topology::{CacheInfo, GpuInfo};

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
/// a caller convention.
#[derive(Clone, Copy)]
pub struct GpuDevice<'a> {
    pub(crate) device: &'a Device,
    pub(crate) info: &'a GpuInfo,
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
}

/// GPU event and notification contracts.
pub mod event {
    /// Linux KFD event interoperability.
    #[cfg(target_os = "linux")]
    pub mod linux {
        pub use crate::event::{
            GpuMemoryFault, SignalEvent, SignalEventInfo, create_signal_event, poll_memory_fault,
        };
    }
}

/// GPU timing and performance-monitoring contracts.
pub mod profiling {
    pub use crate::profiling::*;
}

/// AMD GPU queue formats, transports, and resource owners.
pub mod queue {
    pub use crate::kernel_queue::*;
    pub use crate::queue::*;
}
