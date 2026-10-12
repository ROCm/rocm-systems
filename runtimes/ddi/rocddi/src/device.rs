// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Activated endpoint state and core device lifecycle.
//!
//! A `Device` is deliberately distinct from a passive topology endpoint. It
//! represents successful activation in one session. Shared memory services
//! remain in `memory`; family-specific services live below this module.

use crate::Error;
use crate::driver::{self, DeviceDriverState};
use crate::host_storage::Shared;
use crate::topology::Endpoint;

pub mod event;
pub mod gpu;

/// Family-specific services retained by one activated device.
///
/// The state is shared when a device handle is cloned. Each family owns its
/// service resources within its module; the common `Device` owns only this
/// selection and the activated driver state.
#[derive(Clone)]
pub(crate) enum DeviceResources {
    /// GPU services shared by every clone of one activated handle.
    Gpu(Shared<gpu::GpuResources>),
    /// Activated endpoint with no family-specific service state yet.
    None,
}

impl DeviceResources {
    /// Prepares services qualified by both the endpoint kind and its driver.
    pub(crate) fn for_activated(
        endpoint: &Endpoint,
        driver_state: &DeviceDriverState,
    ) -> Result<Self, Error> {
        if endpoint.gpu().is_some() && driver_state.is_gpu() {
            return Ok(Self::Gpu(Shared::new(
                gpu::GpuResources::default(),
                driver_state.allocator(),
            )?));
        }
        Ok(Self::None)
    }
}

/// One explicitly activated endpoint and its concrete driver state.
/// Dropping this wrapper releases its driver borrow. A driver may retain
/// context bindings for recreation under the session's lifetime policy.
/// Allocations and queues retain their own concrete dependencies without
/// extending this wrapper's lifetime.
#[derive(Clone)]
pub struct Device {
    /// Driver-owned state for this activated endpoint.
    pub(crate) driver_state: DeviceDriverState,
    /// Passive facts retained for metadata queries and kind checks.
    pub(crate) endpoint: Endpoint,
    /// Family-specific service owners shared across clones.
    pub(crate) resources: DeviceResources,
}

impl Device {
    /// Borrows the KFD driver instance and state for a service implemented by KFD.
    /// Other driver families cannot enter KFD-specific operations.
    pub(crate) fn linux_kfd(
        &self,
    ) -> Result<(&Shared<driver::KfdDriver>, &driver::KfdDeviceState), Error> {
        self.driver_state.linux_kfd()
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
        self.driver_state.shares_address_domain(&other.driver_state)
    }

    /// Returns the inclusive device-address bounds captured from the driver's
    /// activated address domain. This performs no native query
    /// and does not promise that every address in the interval is allocatable.
    #[must_use]
    pub fn address_range(&self) -> (u64, u64) {
        self.driver_state.address_range()
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
    pub fn gpu(&self) -> Result<crate::device::gpu::GpuDevice<'_>, Error> {
        let info = self.endpoint.gpu().ok_or(Error::Operation {
            kind: crate::ErrorKind::Unsupported,
            detail: "activated endpoint is not a GPU",
        })?;
        let resources = match &self.resources {
            DeviceResources::Gpu(resources) => &**resources,
            DeviceResources::None => {
                return Err(Error::Operation {
                    kind: crate::ErrorKind::Unsupported,
                    detail: "activated driver has no GPU capability",
                });
            }
        };
        Ok(crate::device::gpu::GpuDevice {
            device: self,
            info,
            resources,
        })
    }
}
