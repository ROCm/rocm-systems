// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Virtual-address reservations, detached memory, mappings, and interval ownership.

use super::{DeviceAccess, MemoryKind, VirtualAddressInfo, VirtualMapRequest, VirtualMemoryInfo};
use crate::device::Device;
use crate::driver::{self, AddressSpaceInfo, CachedInfo, VirtualMemoryOperations};
use crate::memory::storage::TryPush;
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HostInterval {
    address: u64,
    size: u64,
}

impl HostInterval {
    fn overlaps(self, address: u64, size: u64) -> bool {
        // Both ends were checked by validate_virtual_mapping before insertion.
        address < self.address + self.size && self.address < address + size
    }
}

/// Serializes host page-table replacement in one VA reservation. A failed
/// unmap keeps its interval occupied until retry succeeds.
pub(crate) struct HostIntervals {
    ranges: Mutex<Vec<HostInterval>>,
}

impl HostIntervals {
    pub(crate) fn new() -> Self {
        Self {
            ranges: Mutex::new(Vec::new()),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Vec<HostInterval>>, Error> {
        self.ranges.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "virtual-address host-mapping lock is poisoned",
        })
    }

    fn overlaps(ranges: &[HostInterval], address: u64, size: u64) -> bool {
        ranges.iter().any(|range| range.overlaps(address, size))
    }

    fn release(ranges: &mut Vec<HostInterval>, address: u64, size: u64) {
        if let Some(index) = ranges
            .iter()
            .position(|range| range.address == address && range.size == size)
        {
            let last = ranges.len() - 1;
            ranges.as_mut_slice().swap(index, last);
            let _ = ranges.pop();
        }
    }
}

struct DeviceInterval<S: AddressSpaceInfo> {
    device: S,
    range: HostInterval,
}

/// Serializes mappings within each native device address space. Different
/// device VMs may map the same process reservation at the same address.
pub(crate) struct DeviceIntervals<S: AddressSpaceInfo> {
    ranges: Mutex<Vec<DeviceInterval<S>>>,
}

impl<S: AddressSpaceInfo> DeviceIntervals<S> {
    pub(crate) fn new() -> Self {
        Self {
            ranges: Mutex::new(Vec::new()),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Vec<DeviceInterval<S>>>, Error> {
        self.ranges.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "virtual-address device-mapping lock is poisoned",
        })
    }

    fn overlaps(ranges: &[DeviceInterval<S>], device: &S, address: u64, size: u64) -> bool {
        ranges.iter().any(|range| {
            S::shares_address_domain(&range.device, device) && range.range.overlaps(address, size)
        })
    }

    fn release(ranges: &mut Vec<DeviceInterval<S>>, device: &S, address: u64, size: u64) {
        if let Some(index) = ranges.iter().position(|range| {
            S::shares_address_domain(&range.device, device)
                && range.range.address == address
                && range.range.size == size
        }) {
            let last = ranges.len() - 1;
            ranges.as_mut_slice().swap(index, last);
            let _ = ranges.pop();
        }
    }
}

/// Shared reservation ownership for any driver with virtual-memory support.
pub(crate) struct DriverVirtualAddress<D: VirtualMemoryOperations>
where
    D::DeviceState: AddressSpaceInfo,
{
    // The reservation owner is dropped before the shared driver instance.
    pub(crate) inner: Shared<Owned<D::VirtualAddress>>,
    host_intervals: Shared<HostIntervals>,
    device_intervals: Shared<DeviceIntervals<D::DeviceState>>,
    driver: Shared<D>,
    info: VirtualAddressInfo,
}

impl<D: VirtualMemoryOperations> DriverVirtualAddress<D>
where
    D::DeviceState: AddressSpaceInfo,
{
    /// Constructs one reservation and its mapping bookkeeping after the
    /// caller has checked the devices' common address aperture.
    pub(crate) fn reserve(
        driver: &Shared<D>,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Self, Error> {
        // Allocate metadata before reserving native address space so a
        // metadata failure cannot strand a successful driver reservation.
        let owner = Shared::try_new_uninit()?;
        let host_intervals = Shared::new(HostIntervals::new())?;
        let device_intervals = Shared::new(DeviceIntervals::new())?;
        let inner = driver.reserve_virtual_address(bounds, size, alignment, address)?;
        Ok(Self::new(
            driver.clone(),
            owner.write(inner),
            host_intervals,
            device_intervals,
        ))
    }

    pub(crate) fn new(
        driver: Shared<D>,
        inner: Shared<Owned<D::VirtualAddress>>,
        host_intervals: Shared<HostIntervals>,
        device_intervals: Shared<DeviceIntervals<D::DeviceState>>,
    ) -> Self {
        let info = inner.cached_info();
        Self {
            inner,
            host_intervals,
            device_intervals,
            driver,
            info,
        }
    }

    pub(crate) fn info(&self) -> VirtualAddressInfo {
        self.info
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-address reservation has live mappings",
        })?;
        D::free_virtual_address(inner)
    }
}

/// Owns a process virtual-address reservation used by detached memory mappings.
pub struct VirtualAddress {
    pub(crate) inner: driver::VirtualAddressState,
}

impl VirtualAddress {
    /// Reserves one process virtual-address range common to the supplied
    /// activated devices. All devices must use the same driver instance. A
    /// nonzero requested address is a hint and may fall back to another address
    /// in the common aperture. The selected driver's virtual-memory capability
    /// owns the reservation.
    ///
    /// # Errors
    /// Rejects an empty or mixed-driver device set, incompatible apertures,
    /// invalid page-aligned extents, unsupported virtual memory, or address-space
    /// exhaustion. Driver process and lifecycle errors are propagated.
    pub fn reserve(
        devices: &[&Device],
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Self, Error> {
        let first = devices.first().ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "virtual-address reservation requires an activated device",
        })?;
        let selected_driver = &first.driver_state;
        let mut bounds = selected_driver.address_range();
        for device in &devices[1..] {
            if !device.driver_state.shares_driver_instance(selected_driver) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "virtual-address devices must share one driver instance",
                });
            }
            let range = device.driver_state.address_range();
            bounds = (bounds.0.max(range.0), bounds.1.min(range.1));
        }
        if bounds.0 > bounds.1 {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "activated devices have no common virtual-address aperture",
            });
        }
        Ok(Self {
            inner: selected_driver.reserve_virtual_address(bounds, size, alignment, address)?,
        })
    }

    /// Returns the immutable base and extent of this reservation.
    #[must_use]
    pub fn info(&self) -> VirtualAddressInfo {
        self.inner.info()
    }

    /// Releases an unused address range. All device and host mapping owners
    /// that refer to this range must have been freed first.
    ///
    /// # Errors
    /// Reports a native release failure while retaining the owner for retry.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}

/// Shared detached-backing ownership for any virtual-memory driver.
pub(crate) struct DriverVirtualMemory<D: VirtualMemoryOperations> {
    // Keep the driver instance alive until backing cleanup completes.
    pub(crate) inner: Shared<Owned<D::VirtualMemory>>,
    driver: Shared<D>,
    info: VirtualMemoryInfo,
}

impl<D: VirtualMemoryOperations> DriverVirtualMemory<D>
where
    D::DeviceState: AddressSpaceInfo,
{
    pub(crate) fn new(driver: Shared<D>, inner: Shared<Owned<D::VirtualMemory>>) -> Self {
        let info = inner.cached_info();
        Self {
            inner,
            driver,
            info,
        }
    }

    pub(crate) fn info(&self) -> VirtualMemoryInfo {
        self.info
    }

    pub(crate) fn driver_state(&self) -> &D::VirtualMemory {
        &self.inner
    }

    pub(crate) fn map_host(
        &self,
        reservation: &DriverVirtualAddress<D>,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<DriverVirtualHostMapping<D>, Error> {
        if !Shared::ptr_eq(&self.driver, &reservation.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "virtual-memory owners must belong to one session",
            });
        }
        validate_virtual_mapping(self.info, reservation.info, address, offset, size)?;
        let mut ranges = reservation.host_intervals.lock()?;
        if HostIntervals::overlaps(ranges.as_slice(), address, size) {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "virtual-address range already has a host mapping",
            });
        }
        // Reserve metadata before the driver can replace host page tables.
        ranges.try_push(HostInterval { address, size })?;
        let inner = match D::map_virtual_host(
            &self.inner,
            &reservation.inner,
            address,
            offset,
            size,
            permissions,
        ) {
            Ok(inner) => inner,
            Err(error) => {
                HostIntervals::release(&mut ranges, address, size);
                return Err(error);
            }
        };
        drop(ranges);
        Ok(DriverVirtualHostMapping {
            inner,
            driver: Some(self.driver.clone()),
            reservation: Some(reservation.inner.clone()),
            backing: Some(self.inner.clone()),
            host_intervals: Some(reservation.host_intervals.clone()),
            address,
            size,
        })
    }

    pub(crate) fn map_device(
        &self,
        device_driver: &Shared<D>,
        state: &D::DeviceState,
        reservation: &DriverVirtualAddress<D>,
        request: VirtualMapRequest,
    ) -> Result<DriverVirtualDeviceMapping<D>, Error> {
        let VirtualMapRequest {
            address,
            offset,
            size,
            permissions,
        } = request;
        if !Shared::ptr_eq(device_driver, &self.driver)
            || !Shared::ptr_eq(device_driver, &reservation.driver)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "virtual-memory owners must belong to one session",
            });
        }
        validate_virtual_mapping(self.info, reservation.info, address, offset, size)?;
        let mut ranges = reservation.device_intervals.lock()?;
        if DeviceIntervals::overlaps(ranges.as_slice(), state, address, size) {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "virtual-address range already has a device mapping",
            });
        }
        ranges.try_push(DeviceInterval {
            device: state.clone(),
            range: HostInterval { address, size },
        })?;
        let inner = match D::map_virtual_device(
            &self.inner,
            &reservation.inner,
            state,
            address,
            offset,
            size,
            permissions,
        ) {
            Ok(inner) => inner,
            Err(error) => {
                // The driver must make an ambiguous result unavailable before
                // this tentative occupancy record can be released.
                DeviceIntervals::release(&mut ranges, state, address, size);
                return Err(error);
            }
        };
        drop(ranges);
        Ok(DriverVirtualDeviceMapping {
            inner,
            driver: Some(self.driver.clone()),
            reservation: Some(reservation.inner.clone()),
            backing: Some(self.inner.clone()),
            device_intervals: Some(reservation.device_intervals.clone()),
            device_state: Some(state.clone()),
            address,
            size,
        })
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-memory backing has live mappings",
        })?;
        D::free_virtual_memory(inner)
    }
}

/// Owns detached physical backing used by virtual-memory mappings.
pub struct VirtualMemory {
    pub(crate) inner: driver::VirtualMemoryState,
}

impl VirtualMemory {
    pub(crate) fn new(inner: driver::VirtualMemoryState) -> Self {
        Self { inner }
    }

    /// Returns immutable backing facts.
    #[must_use]
    pub fn info(&self) -> VirtualMemoryInfo {
        self.inner.info()
    }

    /// Maps a subrange of this backing into the process host page tables at a
    /// range owned by `reservation` with exactly the requested permissions.
    ///
    /// # Errors
    /// Rejects cross-session or out-of-range mappings and reports a native
    /// host-mapping failure.
    pub fn map_host(
        &self,
        reservation: &VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<VirtualHostMapping, Error> {
        Ok(VirtualHostMapping {
            inner: self
                .inner
                .map_host(&reservation.inner, address, offset, size, permissions)?,
        })
    }

    /// Releases detached physical backing after every mapping has been freed.
    ///
    /// # Errors
    /// Reports native cleanup failure while retaining unfinished ownership.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}

/// Shared device-mapping ownership, including retryable native cleanup.
pub(crate) struct DriverVirtualDeviceMapping<D: VirtualMemoryOperations>
where
    D::DeviceState: AddressSpaceInfo,
{
    inner: Owned<D::VirtualDeviceMapping>,
    driver: Option<Shared<D>>,
    reservation: Option<Shared<Owned<D::VirtualAddress>>>,
    backing: Option<Shared<Owned<D::VirtualMemory>>>,
    device_intervals: Option<Shared<DeviceIntervals<D::DeviceState>>>,
    device_state: Option<D::DeviceState>,
    address: u64,
    size: u64,
}

impl<D: VirtualMemoryOperations> DriverVirtualDeviceMapping<D>
where
    D::DeviceState: AddressSpaceInfo,
{
    pub(crate) fn free(&mut self) -> Result<(), Error> {
        let mut ranges = self
            .device_intervals
            .as_ref()
            .map(|intervals| intervals.lock())
            .transpose()?;
        D::free_virtual_device_mapping(&mut self.inner)?;
        if let (Some(ranges), Some(device)) = (&mut ranges, &self.device_state) {
            DeviceIntervals::release(ranges, device, self.address, self.size);
        }
        drop(ranges);
        self.device_intervals = None;
        self.device_state = None;
        self.reservation = None;
        self.backing = None;
        self.driver = None;
        Ok(())
    }
}

impl<D: VirtualMemoryOperations> Drop for DriverVirtualDeviceMapping<D>
where
    D::DeviceState: AddressSpaceInfo,
{
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(intervals) = self.device_intervals.take() {
                std::mem::forget(intervals);
            }
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
            if let Some(driver) = self.driver.take() {
                std::mem::forget(driver);
            }
        }
    }
}

/// Owns one virtual-memory mapping in an activated device VM.
pub struct VirtualDeviceMapping {
    pub(crate) inner: driver::VirtualDeviceMappingState,
}

impl VirtualDeviceMapping {
    /// Removes the device mapping and releases its imported native-memory
    /// reference.
    ///
    /// # Errors
    /// Reports an unmap or handle-release failure while retaining cleanup state.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}

/// Shared host-mapping ownership, including retryable native cleanup.
pub(crate) struct DriverVirtualHostMapping<D: VirtualMemoryOperations> {
    inner: Owned<D::VirtualHostMapping>,
    driver: Option<Shared<D>>,
    reservation: Option<Shared<Owned<D::VirtualAddress>>>,
    backing: Option<Shared<Owned<D::VirtualMemory>>>,
    host_intervals: Option<Shared<HostIntervals>>,
    address: u64,
    size: u64,
}

impl<D: VirtualMemoryOperations> DriverVirtualHostMapping<D> {
    pub(crate) fn free(&mut self) -> Result<(), Error> {
        let mut ranges = self
            .host_intervals
            .as_ref()
            .map(|intervals| intervals.lock())
            .transpose()?;
        D::free_virtual_host_mapping(&mut self.inner)?;
        if let Some(ranges) = &mut ranges {
            HostIntervals::release(ranges, self.address, self.size);
        }
        drop(ranges);
        self.host_intervals = None;
        self.reservation = None;
        self.backing = None;
        self.driver = None;
        Ok(())
    }
}

impl<D: VirtualMemoryOperations> Drop for DriverVirtualHostMapping<D> {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(intervals) = self.host_intervals.take() {
                std::mem::forget(intervals);
            }
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
            if let Some(driver) = self.driver.take() {
                std::mem::forget(driver);
            }
        }
    }
}

/// Owns one virtual-memory mapping in the process host page tables.
pub struct VirtualHostMapping {
    pub(crate) inner: driver::VirtualHostMappingState,
}

impl VirtualHostMapping {
    /// Removes the host mapping while preserving the encompassing address
    /// reservation for reuse.
    ///
    /// # Errors
    /// Reports a native mapping failure while retaining cleanup state.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}

pub(crate) fn validate_virtual_mapping(
    memory: VirtualMemoryInfo,
    reservation: VirtualAddressInfo,
    address: u64,
    offset: u64,
    size: u64,
) -> Result<(), Error> {
    if !memory.mapping_granularity.is_power_of_two()
        || !reservation.mapping_granularity.is_power_of_two()
        || memory.size == 0
        || memory.size % memory.mapping_granularity != 0
        || reservation.address % reservation.mapping_granularity != 0
        || reservation.size == 0
        || reservation.size % reservation.mapping_granularity != 0
    {
        return Err(Error::Operation {
            kind: ErrorKind::DriverContract,
            detail: "virtual-memory owner has invalid mapping granularity",
        });
    }
    let granularity = memory
        .mapping_granularity
        .max(reservation.mapping_granularity);
    let mapping_end = address.checked_add(size).ok_or(Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail: "virtual-memory mapping address overflows",
    })?;
    let reservation_end =
        reservation
            .address
            .checked_add(reservation.size)
            .ok_or(Error::Operation {
                kind: ErrorKind::DriverContract,
                detail: "virtual-address reservation extent overflows",
            })?;
    let backing_end = offset.checked_add(size).ok_or(Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail: "virtual-memory backing offset overflows",
    })?;
    if size == 0
        || address % granularity != 0
        || offset % granularity != 0
        || size % granularity != 0
        || address < reservation.address
        || mapping_end > reservation_end
        || backing_end > memory.size
    {
        return Err(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "virtual-memory mapping is outside its aligned owners",
        });
    }
    Ok(())
}

impl Device {
    /// Creates detached physical backing for later virtual-address mappings.
    /// The allocation has no usable device or host address until a mapping
    /// owner is created with explicit access permissions.
    ///
    /// # Errors
    /// Rejects unsupported placement or invalid extents and reports native
    /// allocation or native share-handle creation failures.
    pub fn create_virtual_memory(
        &self,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<VirtualMemory, Error> {
        self.driver_state
            .create_virtual_memory(kind, size, pinned, uncached)
    }

    /// Maps a subrange of detached physical backing into this device's VM at a
    /// range owned by `reservation` with exactly the requested permissions.
    ///
    /// # Errors
    /// Rejects cross-session owners, ranges outside either owner, invalid
    /// permissions, device loss, or native import and mapping failures.
    pub fn map_virtual_memory(
        &self,
        memory: &VirtualMemory,
        reservation: &VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<VirtualDeviceMapping, Error> {
        Ok(VirtualDeviceMapping {
            inner: self.driver_state.map_virtual_memory(
                &memory.inner,
                &reservation.inner,
                VirtualMapRequest {
                    address,
                    offset,
                    size,
                    permissions,
                },
            )?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct FakeDeviceState(u8);

    impl AddressSpaceInfo for FakeDeviceState {
        fn address_range(&self) -> (u64, u64) {
            (0, u64::MAX)
        }

        fn shares_address_domain(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    #[test]
    fn device_intervals_scope_occupancy_to_driver_vm() -> Result<(), Error> {
        let intervals = DeviceIntervals::<FakeDeviceState>::new();
        let mut ranges = intervals.lock()?;
        ranges.try_push(DeviceInterval {
            device: FakeDeviceState(1),
            range: HostInterval {
                address: 0x1000,
                size: 0x2000,
            },
        })?;
        assert!(DeviceIntervals::overlaps(
            ranges.as_slice(),
            &FakeDeviceState(1),
            0x2000,
            0x1000
        ));
        assert!(!DeviceIntervals::overlaps(
            ranges.as_slice(),
            &FakeDeviceState(2),
            0x2000,
            0x1000
        ));
        assert!(!DeviceIntervals::overlaps(
            ranges.as_slice(),
            &FakeDeviceState(1),
            0x3000,
            0x1000
        ));
        DeviceIntervals::release(&mut ranges, &FakeDeviceState(2), 0x1000, 0x2000);
        assert_eq!(ranges.len(), 1);
        DeviceIntervals::release(&mut ranges, &FakeDeviceState(1), 0x1000, 0x2000);
        assert!(ranges.is_empty());
        Ok(())
    }

    #[test]
    #[ignore = "requires a qualified GPU and live DRM virtual-memory mapping"]
    fn live_device_mapping_rejects_overlap_and_reuses_freed_interval() {
        use crate::session::{DriverContextLifetime, Session};

        // Each ignored GPU test runs in the same Rust test process. Use a
        // separate KFD context so one test cannot retain the primary VM that
        // another test would try to acquire.
        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let mut endpoint = None;
        session
            .enumerate(&mut |candidate| {
                if endpoint.is_none() && candidate.gpu().is_some() {
                    endpoint = Some(candidate);
                }
                Ok(())
            })
            .unwrap();
        let device = session.activate(&endpoint.unwrap()).unwrap();
        let size = 2 * 1024 * 1024;
        let mut first_memory = device
            .create_virtual_memory(MemoryKind::System, size, false, false)
            .unwrap();
        let mut second_memory = device
            .create_virtual_memory(MemoryKind::System, size, false, false)
            .unwrap();
        let granularity = first_memory
            .info()
            .mapping_granularity
            .max(second_memory.info().mapping_granularity);
        let mut reservation = VirtualAddress::reserve(&[&device], size, granularity, 0).unwrap();
        let address = reservation.info().address;
        let mut first = device
            .map_virtual_memory(
                &first_memory,
                &reservation,
                address,
                0,
                granularity,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .unwrap();
        assert_eq!(
            device
                .map_virtual_memory(
                    &second_memory,
                    &reservation,
                    address,
                    0,
                    granularity,
                    DeviceAccess::READ,
                )
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Busy
        );
        first.free().unwrap();
        let mut second = device
            .map_virtual_memory(
                &second_memory,
                &reservation,
                address,
                0,
                granularity,
                DeviceAccess::READ,
            )
            .unwrap();
        second.free().unwrap();
        first_memory.free().unwrap();
        second_memory.free().unwrap();
        reservation.free().unwrap();
        drop((
            first,
            second,
            first_memory,
            second_memory,
            reservation,
            device,
        ));
        session.destroy().unwrap();
    }

    #[test]
    #[ignore = "requires a qualified GPU and live KFD virtual-memory mapping"]
    fn live_host_mapping_rejects_overlap_and_reuses_freed_interval() {
        use crate::session::{DriverContextLifetime, Session};

        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let mut endpoint = None;
        session
            .enumerate(&mut |candidate| {
                if endpoint.is_none() && candidate.gpu().is_some() {
                    endpoint = Some(candidate);
                }
                Ok(())
            })
            .unwrap();
        let endpoint = endpoint.unwrap();
        let device = session.activate(&endpoint).unwrap();
        let mut reservation = VirtualAddress::reserve(&[&device], 4096, 4096, 0).unwrap();
        let mut first_memory = device
            .create_virtual_memory(MemoryKind::System, 4096, false, false)
            .unwrap();
        let mut second_memory = device
            .create_virtual_memory(MemoryKind::System, 4096, false, false)
            .unwrap();
        let address = reservation.info().address;
        let mut first = first_memory
            .map_host(
                &reservation,
                address,
                0,
                4096,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .unwrap();
        assert_eq!(
            second_memory
                .map_host(&reservation, address, 0, 4096, DeviceAccess::READ)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Busy
        );
        first.free().unwrap();
        let mut second = second_memory
            .map_host(&reservation, address, 0, 4096, DeviceAccess::READ)
            .unwrap();
        second.free().unwrap();
        first_memory.free().unwrap();
        second_memory.free().unwrap();
        reservation.free().unwrap();
        drop(first);
        drop(second);
        drop(first_memory);
        drop(second_memory);
        drop(reservation);
        drop(device);
        session.destroy().unwrap();
    }

    #[test]
    fn virtual_mapping_ranges_must_fit_both_owners() {
        let memory = VirtualMemoryInfo {
            size: 0x4000,
            mapping_granularity: 0x1000,
            physical_backing_id: [1, 2],
        };
        let reservation = VirtualAddressInfo {
            address: 0x1_0000,
            size: 0x8000,
            mapping_granularity: 0x1000,
        };

        assert!(validate_virtual_mapping(memory, reservation, 0x1_1000, 0x2000, 0x2000).is_ok());
        for (address, offset, size) in [
            (0x1_1001, 0x2000, 0x1000),
            (0x1_1000, 0x2001, 0x1000),
            (0x1_1000, 0x2000, 0),
            (0x0_f000, 0, 0x1000),
            (0x1_7000, 0, 0x2000),
            (0x1_1000, 0x3000, 0x2000),
        ] {
            assert_eq!(
                validate_virtual_mapping(memory, reservation, address, offset, size)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidArgument
            );
        }
    }

    #[test]
    fn virtual_mapping_uses_the_stricter_owner_granularity() {
        let memory = VirtualMemoryInfo {
            size: 0x2_0000,
            mapping_granularity: 0x1_0000,
            physical_backing_id: [1, 2],
        };
        let reservation = VirtualAddressInfo {
            address: 0x10_0000,
            size: 0x4_0000,
            mapping_granularity: 0x1000,
        };

        assert!(validate_virtual_mapping(memory, reservation, 0x10_0000, 0, 0x1_0000).is_ok());
        assert_eq!(
            validate_virtual_mapping(memory, reservation, 0x10_1000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );

        let invalid = VirtualMemoryInfo {
            mapping_granularity: 0,
            ..memory
        };
        assert_eq!(
            validate_virtual_mapping(invalid, reservation, 0x10_0000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );

        let invalid = VirtualAddressInfo {
            address: reservation.address + 1,
            ..reservation
        };
        assert_eq!(
            validate_virtual_mapping(memory, invalid, 0x10_0000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );
    }
}
