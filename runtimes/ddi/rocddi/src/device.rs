// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Activated endpoint state and core device lifecycle.
//!
//! A `Device` is deliberately distinct from a passive topology endpoint. It
//! represents successful activation in one session. Resource-specific methods
//! are implemented beside memory, queue, event, and profiling ownership.

use crate::Error;
use crate::driver::{self, DriverActivation};
use crate::host_storage::Shared;
use crate::topology::Endpoint;

/// One explicitly activated endpoint and its concrete driver state.
/// Dropping this wrapper releases its driver borrow. A driver may retain
/// context bindings for recreation under the session's lifetime policy.
/// Allocations and queues retain their own concrete dependencies without
/// extending this wrapper's lifetime.
#[derive(Clone)]
pub struct Device {
    pub(crate) activation: DriverActivation,
    pub(crate) endpoint: Endpoint,
    pub(crate) copy_pool: Option<Shared<crate::gpu::CopyResourcePool>>,
}

impl Device {
    /// Borrows the KFD driver instance and state for a service implemented by KFD.
    /// Other driver families cannot enter KFD-specific operations.
    pub(crate) fn linux_kfd(
        &self,
    ) -> Result<(&Shared<driver::KfdDriver>, &driver::KfdDeviceState), Error> {
        self.activation.linux_kfd()
    }

    /// Returns the endpoint snapshot accepted at activation. This is a borrowed
    /// metadata view with no native observation or freshness guarantee.
    #[must_use]
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Returns whether two activated handles address the same native memory
    /// domain.
    /// This is a cached identity check; it grants no access or lifetime by itself.
    #[must_use]
    pub fn shares_address_domain(&self, other: &Self) -> bool {
        self.activation.shares_address_domain(&other.activation)
    }

    /// Returns the inclusive device-address bounds captured from the driver's
    /// activated address domain. This performs no native query
    /// and does not promise that every address in the interval is allocatable.
    #[must_use]
    pub fn address_range(&self) -> (u64, u64) {
        self.activation.address_range()
    }

    /// Borrows the GPU-specific capability view when this device was activated
    /// from a GPU endpoint.
    ///
    /// CPU, NPU, and future non-GPU endpoints return `Unsupported`. Callers must
    /// not infer a GPU from PCI attachment, address-space shape, or any
    /// zero-valued capability field.
    ///
    /// # Errors
    /// Returns `Unsupported` when the endpoint is not a GPU or its driver has
    /// no GPU capability view. The current view is implemented by Linux KFD.
    pub fn gpu(&self) -> Result<crate::gpu::GpuDevice<'_>, Error> {
        let info = self.endpoint.gpu().ok_or(Error::Operation {
            kind: crate::ErrorKind::Unsupported,
            detail: "activated endpoint is not a GPU",
        })?;
        let (driver, state) = self.linux_kfd()?;
        Ok(crate::gpu::GpuDevice {
            device: self,
            driver,
            state,
            info,
        })
    }
}

/// Driver-independent device notification contracts.
pub mod event {
    pub use crate::event::{
        DeviceEvent, DeviceEventSubscription, GpuHardwareException, GpuMemoryFault, subscribe,
    };
}
