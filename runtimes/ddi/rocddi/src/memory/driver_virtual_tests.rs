// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Driver-independent virtual-memory ownership and interval tests.

use super::*;
use crate::driver::{
    AddressSpaceInfo, CachedInfo, Driver, DriverInstanceId, VirtualMemoryOperations,
};
use crate::memory::storage::{Owned, Shared};
use crate::session::DriverContextLifetime;
use crate::topology::Endpoint;
use crate::{Error, ErrorKind};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct State(u8);

impl AddressSpaceInfo for State {
    fn address_range(&self) -> (u64, u64) {
        (0x1000, u64::MAX)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

struct FakeDriver {
    drops: Arc<Mutex<Vec<&'static str>>>,
}

impl Drop for FakeDriver {
    fn drop(&mut self) {
        self.drops.lock().unwrap().push("driver");
    }
}

struct Address {
    info: VirtualAddressInfo,
    drops: Arc<Mutex<Vec<&'static str>>>,
}

impl Drop for Address {
    fn drop(&mut self) {
        self.drops.lock().unwrap().push("reservation");
    }
}

impl CachedInfo for Address {
    type Info = VirtualAddressInfo;
    fn cached_info(&self) -> VirtualAddressInfo {
        self.info
    }
}

struct Backing {
    info: VirtualMemoryInfo,
    drops: Arc<Mutex<Vec<&'static str>>>,
}

impl Drop for Backing {
    fn drop(&mut self) {
        self.drops.lock().unwrap().push("backing");
    }
}

impl CachedInfo for Backing {
    type Info = VirtualMemoryInfo;
    fn cached_info(&self) -> VirtualMemoryInfo {
        self.info
    }
}

struct Mapping {
    fail_once: bool,
}

static MAPPING_DROP_LOCK: Mutex<()> = Mutex::new(());

impl Drop for Mapping {
    fn drop(&mut self) {
        let _held = MAPPING_DROP_LOCK.lock().unwrap();
    }
}

impl Driver for FakeDriver {
    type DeviceState = State;

    fn driver_instance(&self) -> DriverInstanceId {
        DriverInstanceId::TEST
    }

    fn context_lifetime(&self) -> DriverContextLifetime {
        DriverContextLifetime::Session
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn enumerate(
        &self,
        _visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn open_endpoint(&self, _id: [u8; 16]) -> Result<Endpoint, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "virtual-memory test has no endpoints",
        })
    }

    fn activate(&self, _endpoint: &Endpoint) -> Result<State, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "virtual-memory test has no activated endpoints",
        })
    }
}

impl VirtualMemoryOperations for FakeDriver {
    type VirtualAddress = Address;
    type VirtualDeviceMapping = Mapping;
    type VirtualHostMapping = Mapping;
    type VirtualMemory = Backing;
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<Address>, Error> {
        let address = if address == 0 { bounds.0 } else { address };
        Ok(Owned::new(Address {
            info: VirtualAddressInfo {
                address,
                size,
                mapping_granularity: alignment,
            },
            drops: self.drops.clone(),
        })?)
    }

    fn free_virtual_address(_address: &mut Address) -> Result<(), Error> {
        Ok(())
    }

    fn create_virtual_memory(
        &self,
        _device: &State,
        _kind: MemoryKind,
        size: u64,
        _options: VirtualMemoryOptions,
    ) -> Result<Owned<Backing>, Error> {
        Ok(Owned::new(Backing {
            info: VirtualMemoryInfo {
                size,
                mapping_granularity: 0x1000,
                physical_backing_id: [1, 2],
                placement: None,
            },
            drops: self.drops.clone(),
        })?)
    }

    fn free_virtual_memory(_memory: &mut Backing) -> Result<(), Error> {
        Ok(())
    }

    fn map_virtual_device(
        _memory: &Backing,
        _reservation: &Address,
        _device: &State,
        _address: u64,
        _offset: u64,
        _size: u64,
        _permissions: DeviceAccess,
    ) -> Result<Owned<Mapping>, Error> {
        Ok(Owned::new(Mapping { fail_once: false })?)
    }

    fn free_virtual_device_mapping(_mapping: &mut Mapping) -> Result<(), Error> {
        Ok(())
    }

    fn map_virtual_host(
        _memory: &Backing,
        _reservation: &Address,
        _address: u64,
        _offset: u64,
        _size: u64,
        _permissions: DeviceAccess,
    ) -> Result<Owned<Mapping>, Error> {
        Ok(Owned::new(Mapping { fail_once: true })?)
    }

    fn set_virtual_host_mapping_permissions(
        _mapping: &mut Mapping,
        _permissions: DeviceAccess,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn free_virtual_host_mapping(mapping: &mut Mapping) -> Result<(), Error> {
        if mapping.fail_once {
            mapping.fail_once = false;
            return Err(Error::Operation {
                kind: ErrorKind::Driver,
                detail: "injected host unmap failure",
            });
        }
        Ok(())
    }
}

fn owners(
    driver: &Shared<FakeDriver>,
) -> Result<
    (
        DriverVirtualAddress<FakeDriver>,
        DriverVirtualMemory<FakeDriver>,
    ),
    Error,
> {
    let reservation = driver.reserve_virtual_address((0x1000, u64::MAX), 0x4000, 0x1000, 0)?;
    let address = DriverVirtualAddress::new(
        driver.clone(),
        Shared::new(reservation)?,
        Shared::new(HostIntervals::new())?,
        Shared::new(DeviceIntervals::new())?,
        crate::os::ProcessIdentity::capture().map_err(|source| {
            crate::os::native_error("test virtual reservation process identity", source)
        })?,
    );
    let backing = driver.create_virtual_memory(
        &State(1),
        MemoryKind::System,
        0x4000,
        VirtualMemoryOptions::default(),
    )?;
    let process = crate::os::ProcessIdentity::capture().map_err(|source| {
        crate::os::native_error("test virtual backing process identity", source)
    })?;
    let memory = DriverVirtualMemory::new(driver.clone(), Shared::new(backing)?, process);
    Ok((address, memory))
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
#[allow(unsafe_code, clippy::unwrap_used)]
fn fork_child_drops_unmapped_virtual_owners_without_callbacks() -> Result<(), Error> {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let driver = Shared::new(FakeDriver {
        drops: drops.clone(),
    })?;
    let (address, memory) = owners(&driver)?;
    let held_owner = drops.lock().unwrap();
    // SAFETY: The child drops only copied owners and exits without running
    // test-harness destructors. The held lock exposes inherited callbacks.
    let child = unsafe { libc::syscall(libc::SYS_fork) };
    assert!(child >= 0);
    if child == 0 {
        // SAFETY: Bound a regression if a copied callback lock is entered.
        unsafe { libc::alarm(3) };
        drop(memory);
        drop(address);
        drop(driver);
        // SAFETY: _exit avoids inherited test-harness state.
        unsafe { libc::_exit(0) }
    }
    let pid = i32::try_from(child).unwrap();
    let mut status = 0;
    // SAFETY: This waits only for the child created above.
    assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    assert_eq!(status, 0);
    drop(held_owner);
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
#[allow(unsafe_code, clippy::unwrap_used)]
fn fork_child_drops_virtual_owners_without_inherited_callbacks() -> Result<(), Error> {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let driver = Shared::new(FakeDriver {
        drops: drops.clone(),
    })?;
    let (mut address, mut memory) = owners(&driver)?;
    let mut host = memory.map_host(&address, 0x1000, 0, 0x1000, DeviceAccess::READ)?;
    let mut device = memory.map_device(
        &driver,
        &State(1),
        &address,
        VirtualMapRequest {
            address: 0x1000,
            offset: 0,
            size: 0x1000,
            permissions: DeviceAccess::READ,
        },
    )?;
    let held_mapping = MAPPING_DROP_LOCK.lock().unwrap();
    let held_owner = drops.lock().unwrap();
    // SAFETY: The child checks copied owners and exits without running test
    // harness destructors. Held locks expose any inherited cleanup callback.
    let child = unsafe { libc::syscall(libc::SYS_fork) };
    assert!(child >= 0);
    if child == 0 {
        // SAFETY: Bound a regression if a copied callback lock is entered.
        unsafe { libc::alarm(3) };
        let rejected = host
            .set_permissions(DeviceAccess::READ)
            .err()
            .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument)
            && host
                .free()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument)
            && device
                .free()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument)
            && memory
                .free()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument)
            && address
                .free()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument);
        drop(host);
        drop(device);
        drop(memory);
        drop(address);
        drop(driver);
        // SAFETY: _exit avoids inherited test-harness state.
        unsafe { libc::_exit(i32::from(!rejected)) }
    }
    let pid = i32::try_from(child).unwrap();
    let mut status = 0;
    // SAFETY: This waits only for the child created above.
    assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    assert_eq!(status, 0);
    drop(held_owner);
    drop(held_mapping);
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
#[allow(unsafe_code, clippy::unwrap_used)]
fn fork_child_rejects_virtual_mapping_before_inherited_interval_lock() -> Result<(), Error> {
    let driver = Shared::new(FakeDriver {
        drops: Arc::new(Mutex::new(Vec::new())),
    })?;
    let (mut address, memory) = owners(&driver)?;
    address.with_locked_host_intervals_for_test(|| {
        // SAFETY: The child only checks the inherited owner and exits without
        // dropping the fake driver or entering a test-harness lock.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: Bound a regression if the copied interval mutex is entered.
            unsafe { libc::alarm(3) };
            let base = address.info().address;
            let rejected = memory
                .map_host(&address, base, 0, 0x1000, DeviceAccess::READ)
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidArgument);
            // SAFETY: _exit leaves copied owners and mutexes untouched.
            unsafe { libc::_exit(i32::from(!rejected)) }
        }
        let pid = i32::try_from(child).unwrap();
        let mut status = 0;
        // SAFETY: This waits only for the child created above.
        assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
        assert_eq!(status, 0);
    });
    address.free()?;
    Ok(())
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fake-driver lifecycle covers occupancy, retry, and owner drop order"
)]
fn fake_driver_preserves_mapping_occupancy_and_cleanup_order() -> Result<(), Error> {
    let drops = Arc::new(Mutex::new(Vec::new()));
    let driver = Shared::new(FakeDriver {
        drops: drops.clone(),
    })?;
    let (mut address, mut memory) = owners(&driver)?;
    let other_driver = Shared::new(FakeDriver {
        drops: Arc::new(Mutex::new(Vec::new())),
    })?;
    let (_, foreign_memory) = owners(&other_driver)?;
    let base = address.info().address;
    assert_eq!(
        foreign_memory
            .map_host(&address, base, 0, 0x1000, DeviceAccess::READ)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidArgument
    );

    let mut host = memory.map_host(&address, base, 0, 0x1000, DeviceAccess::READ)?;
    assert_eq!(
        memory
            .map_host(&address, base, 0, 0x1000, DeviceAccess::READ)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Busy
    );
    assert_eq!(address.free().unwrap_err().kind(), ErrorKind::Busy);
    assert_eq!(memory.free().unwrap_err().kind(), ErrorKind::Busy);
    assert_eq!(host.free().unwrap_err().kind(), ErrorKind::Driver);
    assert_eq!(
        memory
            .map_host(&address, base, 0, 0x1000, DeviceAccess::READ)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Busy
    );
    host.free()?;

    let mut first = memory.map_device(
        &driver,
        &State(1),
        &address,
        VirtualMapRequest {
            address: base,
            offset: 0,
            size: 0x1000,
            permissions: DeviceAccess::READ,
        },
    )?;
    assert_eq!(
        memory
            .map_device(
                &driver,
                &State(1),
                &address,
                VirtualMapRequest {
                    address: base,
                    offset: 0,
                    size: 0x1000,
                    permissions: DeviceAccess::READ,
                },
            )
            .err()
            .unwrap()
            .kind(),
        ErrorKind::Busy
    );
    let mut second = memory.map_device(
        &driver,
        &State(2),
        &address,
        VirtualMapRequest {
            address: base,
            offset: 0,
            size: 0x1000,
            permissions: DeviceAccess::READ,
        },
    )?;
    first.free()?;
    second.free()?;
    memory.free()?;
    address.free()?;
    drop(host);
    drop(first);
    drop(second);
    drop(memory);
    drop(address);
    drop(driver);
    let sequence = drops.lock().unwrap();
    let driver_drop = sequence
        .iter()
        .position(|entry| *entry == "driver")
        .unwrap();
    let backing = sequence
        .iter()
        .position(|entry| *entry == "backing")
        .unwrap();
    let reservation = sequence
        .iter()
        .position(|entry| *entry == "reservation")
        .unwrap();
    assert!(backing < driver_drop && reservation < driver_drop);
    Ok(())
}
