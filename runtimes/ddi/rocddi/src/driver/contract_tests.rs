// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Driver-independent tests for capability and resource-owner contracts.

use super::*;
use crate::Error;
use crate::host_storage::{Allocator, Owned, Shared};
use crate::memory::{
    AllocationInfo, DeviceAccess, DriverAllocation, DriverHostAllocation, HostAllocationInfo,
    HostCacheability, MemoryKind, OwnedMemoryKind,
};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, EndpointKind, TopologyKey};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

const ID: [u8; 16] = [0x42; 16];

struct FakeDriver {
    instance: u64,
    host_drops: Arc<AtomicUsize>,
    allocation_drops: Arc<AtomicUsize>,
    fail_activation: bool,
    fail_allocation: bool,
    fail_shutdown: bool,
}

#[derive(Clone)]
struct FakeDevice {
    id: [u8; 16],
}

struct FakeHost {
    drops: Arc<AtomicUsize>,
    fail_free: bool,
    released: bool,
}

impl Drop for FakeHost {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

struct FakeAllocation {
    device_id: [u8; 16],
    drops: Arc<AtomicUsize>,
    fail_free: bool,
    released: bool,
}

impl Drop for FakeAllocation {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

fn failure(kind: crate::ErrorKind, detail: &'static str) -> Error {
    Error::Operation { kind, detail }
}

impl FakeDriver {
    fn new() -> Self {
        Self {
            instance: new_driver_instance(),
            host_drops: Arc::new(AtomicUsize::new(0)),
            allocation_drops: Arc::new(AtomicUsize::new(0)),
            fail_activation: false,
            fail_allocation: false,
            fail_shutdown: false,
        }
    }

    fn endpoint(&self) -> Endpoint {
        let mut name = [0; 128];
        name[..8].copy_from_slice(b"fake-cpu");
        Endpoint {
            id: ID,
            name,
            pci: None,
            kind: EndpointKind::Cpu,
            local_memory_bytes: 0,
            host_visible_local_memory_bytes: 0,
            host_local_cacheability: None::<HostCacheability>,
            allocation_granularity: 4096,
            address_bit_count: 48,
            minimum_address: 0,
            maximum_address: u64::MAX,
            supported_permissions: DeviceAccess::READ | DeviceAccess::WRITE,
            caches: None,
            memory_links: None,
            topology_key: TopologyKey {
                group: 0,
                member: 0,
            },
            driver_instance: self.instance,
            selector: EndpointSelector::Opaque(7),
        }
    }
}

impl CachedInfo for FakeHost {
    type Info = HostAllocationInfo;
    fn cached_info(&self) -> HostAllocationInfo {
        HostAllocationInfo {
            host_address: 0,
            size: if self.released { 0 } else { 4096 },
        }
    }
}

impl CachedInfo for FakeAllocation {
    type Info = AllocationInfo;
    fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: if self.released { 0 } else { 0x10000 },
            host_address: None,
            size: if self.released { 0 } else { 4096 },
            native_size: if self.released { 0 } else { 4096 },
            physical_backing_id: [0; 2],
        }
    }
}

impl DeviceStateType for FakeDriver {
    type DeviceState = FakeDevice;
}

impl Driver for FakeDriver {
    fn driver_instance(&self) -> u64 {
        self.instance
    }

    fn context_lifetime(&self) -> DriverContextLifetime {
        DriverContextLifetime::Session
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        if self.fail_shutdown {
            self.fail_shutdown = false;
            return Err(failure(
                crate::ErrorKind::Driver,
                "injected shutdown failure",
            ));
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
        if id != ID {
            return Err(failure(
                crate::ErrorKind::InvalidArgument,
                "unknown fake endpoint",
            ));
        }
        Ok(self.endpoint())
    }

    fn activate(&self, endpoint: &Endpoint) -> Result<FakeDevice, Error> {
        if endpoint.driver_instance != self.instance
            || endpoint.id != ID
            || endpoint.selector != EndpointSelector::Opaque(7)
            || endpoint.kind != EndpointKind::Cpu
        {
            return Err(failure(
                crate::ErrorKind::InvalidArgument,
                "foreign fake endpoint",
            ));
        }
        if self.fail_activation {
            return Err(failure(
                crate::ErrorKind::Driver,
                "injected activation failure",
            ));
        }
        Ok(FakeDevice { id: endpoint.id })
    }
}

impl HostMemoryDriver for FakeDriver {
    type HostAllocation = FakeHost;

    fn allocate_host(&self, size: u64, alignment: u64) -> Result<Owned<FakeHost>, Error> {
        if size != 4096 || alignment != 4096 {
            return Err(failure(
                crate::ErrorKind::InvalidArgument,
                "fake host extent",
            ));
        }
        Ok(Owned::new(
            FakeHost {
                drops: self.host_drops.clone(),
                fail_free: true,
                released: false,
            },
            Allocator::system(),
        )?)
    }

    fn free_host(allocation: &mut FakeHost) -> Result<(), Error> {
        if allocation.fail_free {
            allocation.fail_free = false;
            return Err(failure(
                crate::ErrorKind::Driver,
                "injected host free failure",
            ));
        }
        allocation.released = true;
        Ok(())
    }

    fn host_page_size() -> Result<u64, Error> {
        Ok(4096)
    }

    fn host_cache_line_size() -> Result<u32, Error> {
        Ok(64)
    }

    #[allow(unsafe_code)]
    unsafe fn host_cache_control(_: usize, _: u64, _: u32) -> Result<(), Error> {
        Err(failure(
            crate::ErrorKind::Unsupported,
            "no fake cache control",
        ))
    }
}

impl AllocationDriver for FakeDriver {
    type Allocation = FakeAllocation;

    fn supports_host_registration(&self, _: &Endpoint) -> bool {
        false
    }

    fn allocation_is_device_local(_: &FakeAllocation) -> bool {
        false
    }
    fn check_allocation(allocation: &FakeAllocation) -> Result<(), Error> {
        if allocation.released {
            Err(failure(
                crate::ErrorKind::InvalidArgument,
                "released fake allocation",
            ))
        } else {
            Ok(())
        }
    }

    fn allocation_device_address(
        allocation: &FakeAllocation,
        device: &FakeDevice,
    ) -> Result<u64, Error> {
        Self::check_allocation(allocation)?;
        if allocation.device_id != device.id {
            return Err(failure(
                crate::ErrorKind::InvalidArgument,
                "foreign fake device",
            ));
        }
        Ok(0x10000)
    }

    fn allocation_is_owned_by(allocation: &FakeAllocation, device: &FakeDevice) -> bool {
        allocation.device_id == device.id && !allocation.released
    }

    fn free_allocation(allocation: &mut FakeAllocation) -> Result<(), Error> {
        if allocation.fail_free {
            allocation.fail_free = false;
            return Err(failure(
                crate::ErrorKind::Driver,
                "injected allocation free failure",
            ));
        }
        allocation.released = true;
        Ok(())
    }

    fn allocate_owned(
        &self,
        device: &FakeDevice,
        peers: &[&FakeDevice],
        kind: OwnedMemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<FakeAllocation>, Error> {
        if !peers.is_empty()
            || kind.get() != MemoryKind::System
            || size != 4096
            || alignment != 4096
            || !permissions.contains(DeviceAccess::READ)
        {
            return Err(failure(
                crate::ErrorKind::InvalidArgument,
                "fake allocation request",
            ));
        }
        if self.fail_allocation {
            return Err(failure(
                crate::ErrorKind::Driver,
                "injected allocation failure",
            ));
        }
        Ok(Owned::new(
            FakeAllocation {
                device_id: device.id,
                drops: self.allocation_drops.clone(),
                fail_free: true,
                released: false,
            },
            Allocator::system(),
        )?)
    }
}

#[test]
fn non_gpu_resource_owners_preserve_cleanup_retry() -> Result<(), Error> {
    let fake = FakeDriver::new();
    let host_drops = fake.host_drops.clone();
    let allocation_drops = fake.allocation_drops.clone();
    let driver = Shared::new(fake, Allocator::system())?;
    let device_state = driver.activate(&driver.endpoint())?;
    let mut host = DriverHostAllocation::<FakeDriver>::new(driver.allocate_host(4096, 4096)?);
    assert_eq!(host.info().size, 4096);
    assert!(host.free().is_err());
    assert_eq!(host_drops.load(Ordering::Relaxed), 0);
    host.free()?;
    drop(host);
    assert_eq!(host_drops.load(Ordering::Relaxed), 1);

    let mut allocation = DriverAllocation::<FakeDriver>::new(driver.allocate_owned(
        &device_state,
        &[],
        OwnedMemoryKind::try_from(MemoryKind::System)?,
        4096,
        4096,
        DeviceAccess::READ,
    )?);
    assert_eq!(allocation.info().device_address, 0x10000);
    allocation.check()?;
    assert_eq!(allocation.device_address(&device_state)?, 0x10000);
    assert!(allocation.originates_from(&device_state));
    assert!(allocation.free().is_err());
    assert_eq!(allocation_drops.load(Ordering::Relaxed), 0);
    allocation.free()?;
    drop(allocation);
    assert_eq!(allocation_drops.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn non_gpu_driver_contract_covers_discovery_activation_and_failed_cleanup() -> Result<(), Error> {
    let mut driver = FakeDriver::new();
    let mut discovered = Vec::new();
    driver.enumerate(&mut |endpoint| {
        discovered.push(endpoint.id);
        Ok(())
    })?;
    assert_eq!(discovered, [ID]);
    let endpoint = driver.open_endpoint(ID)?;
    assert!(endpoint.pci.is_none());
    assert!(endpoint.gpu().is_none());
    assert!(endpoint.linux_kfd_drm_info().is_none());
    assert!(!driver.supports_host_registration(&endpoint));
    assert!(driver.activate(&FakeDriver::new().endpoint()).is_err());

    driver.fail_activation = true;
    assert!(driver.activate(&endpoint).is_err());
    driver.fail_activation = false;
    let device = driver.activate(&endpoint)?;
    assert_eq!(device.id, ID);

    let mut host = driver.allocate_host(4096, 4096)?;
    assert_eq!(host.cached_info().size, 4096);
    assert!(FakeDriver::free_host(&mut host).is_err());
    assert_eq!(driver.host_drops.load(Ordering::Relaxed), 0);
    FakeDriver::free_host(&mut host)?;
    drop(host);
    assert_eq!(driver.host_drops.load(Ordering::Relaxed), 1);

    driver.fail_allocation = true;
    assert!(
        driver
            .allocate_owned(
                &device,
                &[],
                OwnedMemoryKind::try_from(MemoryKind::System)?,
                4096,
                4096,
                DeviceAccess::READ
            )
            .is_err()
    );
    assert_eq!(driver.allocation_drops.load(Ordering::Relaxed), 0);
    driver.fail_allocation = false;
    let mut allocation = driver.allocate_owned(
        &device,
        &[],
        OwnedMemoryKind::try_from(MemoryKind::System)?,
        4096,
        4096,
        DeviceAccess::READ,
    )?;
    assert_eq!(
        FakeDriver::allocation_device_address(&allocation, &device)?,
        0x10000
    );
    assert_eq!(allocation.cached_info().device_address, 0x10000);
    assert!(FakeDriver::free_allocation(&mut allocation).is_err());
    assert_eq!(driver.allocation_drops.load(Ordering::Relaxed), 0);
    FakeDriver::free_allocation(&mut allocation)?;
    drop(allocation);
    assert_eq!(driver.allocation_drops.load(Ordering::Relaxed), 1);

    driver.fail_shutdown = true;
    assert!(driver.shutdown().is_err());
    driver.shutdown()?;
    Ok(())
}
