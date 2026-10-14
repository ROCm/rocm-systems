// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Minimal non-GPU driver used to exercise heterogeneous session routing.

use super::{AddressSpaceInfo, AllocationOperations, CachedInfo, Driver, EndpointSelector};
use crate::memory::host::HostBackingLease;
use crate::memory::storage::Owned;
use crate::memory::{AllocationInfo, DeviceAccess, HostRegistration, MemoryKind};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, EndpointKind, TopologyKey};
use crate::{Error, ErrorKind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Test driver instance with only the common driver contract.
pub(crate) struct TestDriver {
    instance: u64,
    id: [u8; 16],
    kind: EndpointKind,
    lifetime: DriverContextLifetime,
    host_registration: bool,
    pub(crate) shutdowns: Arc<AtomicUsize>,
    pub(crate) fail_shutdown: Arc<AtomicBool>,
}

/// Activated test endpoint. It intentionally has no GPU resource capabilities.
#[derive(Clone)]
pub(crate) struct TestDevice {
    id: [u8; 16],
}

/// Numeric-only allocation used to test backing lifetimes across drivers.
pub(crate) struct TestAllocation {
    id: [u8; 16],
    host_address: usize,
    device_address: u64,
    size: u64,
    released: bool,
    backing: Option<HostBackingLease>,
}

impl CachedInfo for TestAllocation {
    type Info = AllocationInfo;

    fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: self.device_address,
            host_address: Some(self.host_address),
            size: self.size,
            native_size: self.size,
            physical_backing_id: [0; 2],
        }
    }
}

impl TestDriver {
    pub(crate) fn new(id: [u8; 16], kind: EndpointKind, lifetime: DriverContextLifetime) -> Self {
        Self {
            instance: super::new_driver_instance(),
            id,
            kind,
            lifetime,
            host_registration: false,
            shutdowns: Arc::new(AtomicUsize::new(0)),
            fail_shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn with_host_registration(mut self) -> Self {
        self.host_registration = true;
        self
    }

    fn endpoint(&self) -> Endpoint {
        let mut name = [0; 128];
        name[..8].copy_from_slice(b"test-npu");
        Endpoint {
            id: self.id,
            name,
            pci: None,
            kind: self.kind,
            local_memory_bytes: 0,
            host_visible_local_memory_bytes: 0,
            host_local_cacheability: None,
            allocation_granularity: 4096,
            address_bit_count: 48,
            minimum_address: 0x1000,
            maximum_address: 0xffff,
            supported_permissions: if self.host_registration {
                DeviceAccess::READ | DeviceAccess::WRITE
            } else {
                DeviceAccess::NONE
            },
            caches: None,
            memory_links: None,
            topology_key: TopologyKey {
                group: 0,
                member: 0,
            },
            driver_instance: self.instance,
            selector: EndpointSelector::Opaque(u64::from_le_bytes([
                self.id[0], self.id[1], self.id[2], self.id[3], self.id[4], self.id[5], self.id[6],
                self.id[7],
            ])),
        }
    }
}

impl AddressSpaceInfo for TestDevice {
    fn address_range(&self) -> (u64, u64) {
        (0x1000, 0xffff)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Driver for TestDriver {
    type DeviceState = TestDevice;

    fn driver_instance(&self) -> u64 {
        self.instance
    }

    fn context_lifetime(&self) -> DriverContextLifetime {
        self.lifetime
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        self.shutdowns.fetch_add(1, Ordering::Relaxed);
        if self.fail_shutdown.swap(false, Ordering::Relaxed) {
            return Err(Error::Operation {
                kind: ErrorKind::Driver,
                detail: "injected shutdown failure",
            });
        }
        Ok(())
    }

    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        visitor(self.endpoint())
    }

    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        if id != self.id {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "unknown test endpoint",
            });
        }
        Ok(self.endpoint())
    }

    fn activate(&self, endpoint: &Endpoint) -> Result<TestDevice, Error> {
        if endpoint != &self.endpoint() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "foreign test endpoint",
            });
        }
        Ok(TestDevice { id: self.id })
    }
}

impl AllocationOperations for TestDriver {
    type Allocation = TestAllocation;

    fn supports_host_registration(&self, _: &Endpoint) -> bool {
        self.host_registration
    }

    fn check_allocation(allocation: &Self::Allocation) -> Result<(), Error> {
        if allocation.released {
            Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "test registration has been freed",
            })
        } else {
            Ok(())
        }
    }

    fn allocation_is_device_local(_: &Self::Allocation) -> bool {
        false
    }

    fn allocation_device_address(
        allocation: &Self::Allocation,
        device: &Self::DeviceState,
    ) -> Result<u64, Error> {
        Self::check_allocation(allocation)?;
        if allocation.id != device.id {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "test device has no registration",
            });
        }
        Ok(allocation.device_address)
    }

    fn allocation_is_owned_by(allocation: &Self::Allocation, device: &Self::DeviceState) -> bool {
        allocation.id == device.id && !allocation.released
    }

    fn free_allocation(allocation: &mut Self::Allocation) -> Result<(), Error> {
        allocation.released = true;
        allocation.backing = None;
        Ok(())
    }

    fn allocate_owned(
        &self,
        _: &Self::DeviceState,
        _: &[&Self::DeviceState],
        _: MemoryKind,
        _: u64,
        _: u64,
        _: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "test driver has no owned allocation",
        })
    }

    /// # Safety
    /// Borrowed callers keep the supplied host pages mapped until registration
    /// ends; owned requests transfer a backing lease into the allocation.
    #[allow(unsafe_code)]
    unsafe fn register_host(
        &self,
        device: &Self::DeviceState,
        peers: &[&Self::DeviceState],
        request: HostRegistration,
    ) -> Result<Owned<Self::Allocation>, Error> {
        if !self.host_registration
            || request.address == 0
            || request.size == 0
            || !request.alignment.is_power_of_two()
            || !request.permissions.contains(DeviceAccess::READ)
            || peers.iter().any(|peer| peer.id != device.id)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid test host registration",
            });
        }
        let device_address = u64::from(self.id[0]) * 0x1_0000 + 0x1_0000;
        Ok(Owned::new(TestAllocation {
            id: device.id,
            host_address: request.address,
            device_address,
            size: request.size,
            released: false,
            backing: request.backing,
        })?)
    }
}
