// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned host backing with independent device-driver registrations.

use super::{Allocation, DeviceAccess, HostMappingPolicy, HostRegistration};
use crate::device::Device;
use crate::memory::host::{HostAllocation, HostAllocationInfo, host_page_size};
use crate::memory::storage::TryPush;
use crate::{Error, ErrorKind};
use std::ops::Deref;

/// One device-driver registration of rocddi-owned host backing.
///
/// The host allocation can be registered independently with several drivers.
/// Each registration has its own address and access set, while all retain the
/// same mapped backing. Host cleanup remains busy until every registration is
/// freed. A failed native cleanup retains the backing through process teardown.
pub struct RegisteredHostAllocation {
    allocation: Allocation,
}

impl RegisteredHostAllocation {
    /// Replaces the access set within this registration's driver. Devices
    /// registered through another driver have independent mapping owners.
    ///
    /// # Errors
    /// Reports invalid origin or peer devices and native mapping failures.
    pub fn set_device_access(&mut self, origin: &Device, devices: &[&Device]) -> Result<(), Error> {
        self.allocation.set_device_access(origin, devices)
    }

    /// Releases the device registration and its retained host backing.
    /// Failure preserves both owners for retry; a repeated successful release
    /// is harmless.
    ///
    /// # Errors
    /// Propagates the native allocation cleanup error.
    pub fn free(&mut self) -> Result<(), Error> {
        self.allocation.free()
    }
}

impl Deref for RegisteredHostAllocation {
    type Target = Allocation;

    fn deref(&self) -> &Self::Target {
        &self.allocation
    }
}

/// Owns one host backing and independent registrations with device drivers.
///
/// Each driver receives its own registration and may report a different device
/// address. Registrations are released before the host backing. The backing
/// remains mapped if native cleanup cannot prove that every registration has
/// stopped using it.
pub struct RegisteredHostMemory {
    registrations: Vec<RegisteredHostAllocation>,
    backing: HostAllocation,
}

impl RegisteredHostMemory {
    /// Takes host backing before any device registration is established.
    #[must_use]
    pub fn new(backing: HostAllocation) -> Self {
        Self {
            registrations: Vec::new(),
            backing,
        }
    }

    /// Returns the host backing's original address and extent.
    #[must_use]
    pub fn host_info(&self) -> HostAllocationInfo {
        self.backing.info()
    }

    /// Registers the backing once in `device`'s driver and its listed peers.
    /// Another driver can be added with a subsequent call. The registration
    /// metadata is reserved before the native driver can acquire the pages.
    ///
    /// # Errors
    /// Rejects a second registration in the same driver and propagates host or
    /// native registration failures. An uncertain native failure can retain the
    /// host backing through process teardown.
    pub fn register(
        &mut self,
        device: &Device,
        peers: &[&Device],
        policy: HostMappingPolicy,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<(), Error> {
        if self.registrations.iter().any(|registration| {
            registration.allocation.driver_instance == device.endpoint.driver_instance
        }) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "host backing is already registered with this driver",
            });
        }
        self.registrations.try_reserve(1)?;
        let registration = device.register_host_allocation(
            &self.backing,
            peers,
            policy,
            alignment,
            permissions,
        )?;
        // SAFETY: The capacity was reserved before native registration.
        self.registrations
            .try_push(registration)
            .map_err(Error::from)
    }

    /// Returns this backing's qualified address on one registered device.
    ///
    /// # Errors
    /// Reports an absent mapping or the driver's current availability error.
    pub fn device_address(&self, device: &Device) -> Result<u64, Error> {
        let registration = self
            .registrations
            .iter()
            .find(|registration| {
                registration.allocation.driver_instance == device.endpoint.driver_instance
            })
            .ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "host backing is not registered with this device driver",
            })?;
        registration.device_address(device)
    }

    /// Releases registrations in reverse order, followed by the host backing.
    /// A failed cleanup keeps its owner available for retry.
    ///
    /// # Errors
    /// Returns the first registration or host unmapping error.
    pub fn free(&mut self) -> Result<(), Error> {
        for registration in self.registrations.iter_mut().rev() {
            registration.free()?;
        }
        self.backing.free()
    }
}

impl Device {
    /// Registers a rocddi-owned host allocation in this device driver and its
    /// listed peers. A second driver can register the same backing separately;
    /// each returned owner reports addresses qualified for its own devices.
    /// This method retains the host backing without a raw-pointer lifetime
    /// obligation on the caller.
    ///
    /// # Errors
    /// Rejects a freed host allocation, an invalid device alignment, or peers
    /// from another driver. Native registration errors can leave host backing
    /// retained until process teardown when the driver cannot prove whether it
    /// still references those pages.
    #[allow(unsafe_code)]
    pub fn register_host_allocation(
        &self,
        host: &HostAllocation,
        peers: &[&Self],
        policy: HostMappingPolicy,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<RegisteredHostAllocation, Error> {
        let page = host_page_size()?;
        if !alignment.is_power_of_two() || alignment < page {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid host registration alignment",
            });
        }
        if peers
            .iter()
            .any(|peer| peer.endpoint.driver_instance != self.endpoint.driver_instance)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "registered peers must belong to one driver",
            });
        }
        if !self.driver_state.supports_host_registration(&self.endpoint)
            || peers
                .iter()
                .any(|peer| !peer.driver_state.supports_host_registration(&peer.endpoint))
        {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device does not support host registration",
            });
        }
        let backing = host.registration_lease()?;
        let (address, size) = backing.extent();
        let request = HostRegistration {
            address,
            policy,
            size,
            alignment,
            permissions,
            backing: Some(backing),
        };
        // SAFETY: The request transfers its page-cover lease to the driver,
        // which retains it until native cleanup succeeds or process teardown.
        let allocation = unsafe {
            self.driver_state
                .register_host_with_peers(self, peers, request)?
        };
        Ok(RegisteredHostAllocation { allocation })
    }
}
