//! Activated endpoint state and core device lifecycle.
//!
//! A `Device` is deliberately distinct from a passive topology endpoint. It
//! represents successful activation in one session. Resource-specific methods
//! are implemented beside memory, queue, event, and profiling ownership.

use crate::Error;
use crate::driver::{self, DeviceDriver};
use crate::host_storage::Shared;
use crate::topology::Endpoint;

/// One explicitly activated endpoint and a borrow of its session controller.
/// Dropping this wrapper releases that borrow; its VM binding remains owned by
/// the session for recreation. Allocations and queues retain concrete native
/// dependencies without extending the lifetime of this public-facing wrapper.
#[derive(Clone)]
pub struct Device {
    pub(crate) driver: Shared<driver::PlatformDriver>,
    pub(crate) state: driver::DeviceState,
    pub(crate) endpoint: Endpoint,
}

impl Device {
    /// Returns the endpoint snapshot accepted at activation. This is a borrowed
    /// metadata view with no native observation or freshness guarantee.
    #[must_use]
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Observes this device's native loss source without waiting for work.
    /// A previously observed loss stays latched. This check does not refresh
    /// endpoint metadata, acquire mappings, or establish execution retirement.
    ///
    /// # Errors
    /// Returns `DeviceLost` once loss is observed, `Unsupported` for inherited
    /// process state after fork, or the native error from the observation.
    pub fn check(&self) -> Result<(), Error> {
        self.driver.check(&self.state)
    }

    /// Returns whether a prior native observation latched terminal device loss.
    /// This reads cached process state and performs no system call or wait.
    #[must_use]
    pub fn has_observed_loss(&self) -> bool {
        self.state.has_observed_loss()
    }

    /// Returns whether two activated handles address the same native device VM.
    /// This is a cached identity check; it grants no access or lifetime by itself.
    #[must_use]
    pub fn shares_address_domain(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.driver, &other.driver) && self.state.shares_vm(&other.state)
    }

    /// Returns the inclusive device-address bounds captured from the installed
    /// process address space during activation. This performs no native query
    /// and does not promise that every address in the interval is allocatable.
    #[must_use]
    pub fn address_range(&self) -> (u64, u64) {
        self.state.address_range()
    }

    /// Borrows the GPU-specific capability view when this device was activated
    /// from a GPU endpoint.
    ///
    /// CPU, NPU, and future non-GPU endpoints return `Unsupported`. Callers must
    /// not infer a GPU from PCI attachment, address-space shape, or any
    /// zero-valued capability field.
    ///
    /// # Errors
    /// Returns `Unsupported` when the activated endpoint is not a GPU.
    pub fn gpu(&self) -> Result<crate::gpu::GpuDevice<'_>, Error> {
        self.endpoint
            .gpu()
            .map(|info| crate::gpu::GpuDevice { device: self, info })
            .ok_or(Error::Operation {
                kind: crate::ErrorKind::Unsupported,
                detail: "activated endpoint is not a GPU",
            })
    }
}
