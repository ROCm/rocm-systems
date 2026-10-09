// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Memory placement, ownership, mapping, sharing, and host-cache services.
//!
//! Public owners retain the exact native dependencies required for teardown.
//! The platform driver supplies backing and mappings; this module enforces
//! cross-session ownership and range validation. Platform-specific sharing
//! transports live under [`interop`] so native handles do not leak into the
//! implementation-neutral allocation model.

mod access;
pub use access::DeviceAccess;
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod driver_virtual_tests;
pub mod interop;
mod types;
pub(crate) use types::{AllocationDesc, AllocationLimits};

use crate::device::Device;
use crate::driver::{
    self, AddressSpaceInfo, AllocationDriver, CachedInfo, GpuQueueResourceDriver, HostMemoryDriver,
    VirtualMemoryDriver,
};
use crate::gpu::GpuDevice;
use crate::host_storage::{Buffer, Owned, Shared};
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
    ranges: Mutex<Buffer<HostInterval>>,
}

impl HostIntervals {
    pub(crate) fn new(allocator: crate::host_storage::Allocator) -> Self {
        Self {
            ranges: Mutex::new(Buffer::new(allocator)),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Buffer<HostInterval>>, Error> {
        self.ranges.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "virtual-address host-mapping lock is poisoned",
        })
    }

    fn overlaps(ranges: &[HostInterval], address: u64, size: u64) -> bool {
        ranges.iter().any(|range| range.overlaps(address, size))
    }

    fn release(ranges: &mut Buffer<HostInterval>, address: u64, size: u64) {
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
    ranges: Mutex<Buffer<DeviceInterval<S>>>,
}

impl<S: AddressSpaceInfo> DeviceIntervals<S> {
    pub(crate) fn new(allocator: crate::host_storage::Allocator) -> Self {
        Self {
            ranges: Mutex::new(Buffer::new(allocator)),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Buffer<DeviceInterval<S>>>, Error> {
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

    fn release(ranges: &mut Buffer<DeviceInterval<S>>, device: &S, address: u64, size: u64) {
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

/// Host mapping cache policy qualified by the native backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostCacheability {
    /// Ordinary write-back host mapping.
    WriteBack,
    /// Write-combined host mapping.
    WriteCombined,
}

/// GPU cache and coherence policy for host pages in a device address space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostCachePolicy {
    /// Require explicit synchronization between CPU and GPU views.
    Coarse,
    /// Provide coherent CPU and GPU access.
    Fine,
    /// Provide extended-scope coherent access where supported.
    Extended,
    /// Bypass device caches while retaining coherent access.
    Uncached,
}

/// Platform-neutral backing and placement requested from the native backend.
///
/// These variants describe ownership, visibility, and address behavior that a
/// caller can observe. They deliberately do not encode operating-system handle
/// types or driver allocation mechanisms; each platform backend selects the
/// native mechanism that satisfies the requested contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKind {
    /// Platform-managed, host-accessible system backing with device access
    /// established in the activated address space. The native platform memory
    /// manager owns the physical backing; rocddi owns its returned allocation
    /// and persistent host mapping. Host-only allocations instead use
    /// [`Session::allocate_host`](crate::session::Session::allocate_host).
    System,
    /// rocddi-owned host pages made accessible to the device at the same host
    /// and device virtual address. The backend may pin, register, or otherwise
    /// bind those pages without exposing that native mechanism.
    OwnedHost {
        /// Requested GPU cache and coherence policy for the owned pages.
        cache: HostCachePolicy,
    },
    /// Caller-owned host pages made accessible to the device. `address` is the
    /// logical host base and may be subpage aligned. The caller keeps the
    /// complete page cover live until [`Allocation::free`] succeeds. Safe
    /// allocation methods reject this variant; use [`Device::register_host`]
    /// or [`Device::register_host_with_peers`] instead.
    RegisteredHost {
        /// Borrowed logical host address; ownership remains with the caller.
        address: usize,
        /// Requested GPU cache and coherence policy for the host pages.
        cache: HostCachePolicy,
    },
    /// Device-local storage; host visibility is an explicit requirement.
    DeviceLocal {
        /// Require a persistent host mapping of the local storage.
        host_visible: bool,
        /// Request coherent device mappings for fine-grained access.
        coherent: bool,
        /// Bypass device caches for this allocation.
        uncached: bool,
        /// Request physically contiguous backing.
        contiguous: bool,
    },
}

/// An owned request cannot carry a borrowed host address into a safe driver call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OwnedMemoryKind(MemoryKind);

impl TryFrom<MemoryKind> for OwnedMemoryKind {
    type Error = Error;

    fn try_from(kind: MemoryKind) -> Result<Self, Self::Error> {
        if matches!(kind, MemoryKind::RegisteredHost { .. }) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "borrowed host pages require unsafe registration",
            });
        }
        Ok(Self(kind))
    }
}

impl OwnedMemoryKind {
    pub(crate) fn get(self) -> MemoryKind {
        self.0
    }
}

/// Parameters for the unsafe host-registration driver boundary.
#[derive(Clone, Copy)]
pub(crate) struct HostRegistration {
    pub(crate) address: usize,
    pub(crate) cache: HostCachePolicy,
    pub(crate) size: u64,
    pub(crate) alignment: u64,
    pub(crate) permissions: DeviceAccess,
}

/// Cached addresses and extents of one successfully mapped native allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationInfo {
    /// Address in the activated device's native address space.
    pub device_address: u64,
    /// Existing host mapping when requested and supported.
    pub host_address: Option<usize>,
    /// Exact requested bytes.
    pub size: u64,
    /// Complete native physical allocation or imported backing extent.
    pub native_size: u64,
    /// Stable native identity for imported or shareable backing, when available.
    pub physical_backing_id: [u64; 2],
}

/// Immutable facts describing one detached virtual-memory allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualMemoryInfo {
    /// Complete physical backing extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings of this backing.
    pub mapping_granularity: u64,
    /// Stable native identity shared by equivalent handles for this backing.
    pub physical_backing_id: [u64; 2],
}

/// Cached facts for one reserved process virtual-address range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualAddressInfo {
    /// First byte in the reserved range.
    pub address: u64,
    /// Exact reserved extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings in this reservation.
    pub mapping_granularity: u64,
}

/// One checked placement request for a driver virtual-memory mapping.
#[derive(Clone, Copy)]
struct VirtualMapRequest {
    address: u64,
    offset: u64,
    size: u64,
    permissions: DeviceAccess,
}

/// Shared reservation ownership for any driver with virtual-memory support.
pub(crate) struct DriverVirtualAddress<D: VirtualMemoryDriver>
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

impl<D: VirtualMemoryDriver> DriverVirtualAddress<D>
where
    D::DeviceState: AddressSpaceInfo,
{
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

    fn info(&self) -> VirtualAddressInfo {
        self.info
    }

    fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-address reservation has live mappings",
        })?;
        D::free_virtual_address(inner)
    }
}

/// Owns a process virtual-address reservation used by detached memory mappings.
pub struct VirtualAddress {
    pub(crate) inner: DriverVirtualAddress<driver::KfdDriver>,
}

impl VirtualAddress {
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
pub(crate) struct DriverVirtualMemory<D: VirtualMemoryDriver> {
    // Keep the driver instance alive until backing cleanup completes.
    pub(crate) inner: Shared<Owned<D::VirtualMemory>>,
    driver: Shared<D>,
    info: VirtualMemoryInfo,
}

impl<D: VirtualMemoryDriver> DriverVirtualMemory<D>
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

    fn info(&self) -> VirtualMemoryInfo {
        self.info
    }

    pub(crate) fn driver_state(&self) -> &D::VirtualMemory {
        &self.inner
    }

    fn map_host(
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
            self.inner.allocator(),
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

    fn map_device(
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

    fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-memory backing has live mappings",
        })?;
        D::free_virtual_memory(inner)
    }
}

/// Owns detached physical backing used by virtual-memory mappings.
pub struct VirtualMemory {
    pub(crate) inner: DriverVirtualMemory<driver::KfdDriver>,
}

impl VirtualMemory {
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
pub(crate) struct DriverVirtualDeviceMapping<D: VirtualMemoryDriver>
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

impl<D: VirtualMemoryDriver> DriverVirtualDeviceMapping<D>
where
    D::DeviceState: AddressSpaceInfo,
{
    fn free(&mut self) -> Result<(), Error> {
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

impl<D: VirtualMemoryDriver> Drop for DriverVirtualDeviceMapping<D>
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
    inner: DriverVirtualDeviceMapping<driver::KfdDriver>,
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
pub(crate) struct DriverVirtualHostMapping<D: VirtualMemoryDriver> {
    inner: Owned<D::VirtualHostMapping>,
    driver: Option<Shared<D>>,
    reservation: Option<Shared<Owned<D::VirtualAddress>>>,
    backing: Option<Shared<Owned<D::VirtualMemory>>>,
    host_intervals: Option<Shared<HostIntervals>>,
    address: u64,
    size: u64,
}

impl<D: VirtualMemoryDriver> DriverVirtualHostMapping<D> {
    fn free(&mut self) -> Result<(), Error> {
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

impl<D: VirtualMemoryDriver> Drop for DriverVirtualHostMapping<D> {
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
    inner: DriverVirtualHostMapping<driver::KfdDriver>,
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

/// Allocation owner and cached facts shared by drivers that can
/// allocate device-accessible backing.
pub(crate) struct DriverAllocation<D: AllocationDriver> {
    inner: Owned<D::Allocation>,
    info: AllocationInfo,
}

impl<D: AllocationDriver> DriverAllocation<D> {
    pub(crate) fn new(inner: Owned<D::Allocation>) -> Self {
        let info = inner.cached_info();
        Self { inner, info }
    }

    pub(crate) fn driver_state(&self) -> &D::Allocation {
        &self.inner
    }

    pub(crate) fn driver_state_mut(&mut self) -> &mut D::Allocation {
        &mut self.inner
    }

    pub(crate) fn info(&self) -> AllocationInfo {
        self.info
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        D::check_allocation(&self.inner)
    }

    pub(crate) fn is_device_local(&self) -> bool {
        D::allocation_is_device_local(&self.inner)
    }

    pub(crate) fn device_address(&self, device: &D::DeviceState) -> Result<u64, Error> {
        D::allocation_device_address(&self.inner, device)
    }

    pub(crate) fn originates_from(&self, device: &D::DeviceState) -> bool {
        D::allocation_is_owned_by(&self.inner, device)
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        D::free_allocation(&mut self.inner)
    }
}

/// Owns one native allocation, its exact VM dependencies, and cleanup progress.
/// Explicit free reports failures. If final Drop cannot complete cleanup, it
/// retains potentially reachable native backing instead of recycling its VA.
pub struct Allocation {
    pub(crate) inner: DriverAllocation<driver::KfdDriver>,
    pub(crate) driver_instance: u64,
}

impl Allocation {
    pub(crate) fn from_native(inner: Owned<driver::KfdAllocation>, driver_instance: u64) -> Self {
        Self {
            inner: DriverAllocation::new(inner),
            driver_instance,
        }
    }

    /// Returns creation-time addresses and extent without touching the driver.
    /// The adapter controls their public lifetime. This snapshot remains cached
    /// after cleanup begins, when its addresses must no longer be used.
    #[must_use]
    pub fn info(&self) -> AllocationInfo {
        self.inner.info()
    }
    /// Checks native loss and whether this allocation still permits access.
    /// No mapping is added and no cached address is refreshed.
    ///
    /// # Errors
    /// Returns `DeviceLost` for latched loss, `Unsupported` once freeing has begun
    /// or the mapping is unavailable, and the native cause if observation fails.
    pub fn check(&self) -> Result<(), Error> {
        self.inner.check()
    }

    /// Returns the established address for one device in this allocation's
    /// immutable access set.
    ///
    /// # Errors
    /// Returns `InvalidArgument` for another session, `Unsupported` if the
    /// device has no mapping, `DeviceLost` for a latched native loss, or the
    /// native error from the availability check.
    pub fn device_address(&self, device: &Device) -> Result<u64, Error> {
        if self.driver_instance != device.endpoint.driver_instance {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "device belongs to another allocation session",
            });
        }
        let (_, state) = device.linux_kfd()?;
        self.inner.device_address(state)
    }

    /// Replaces the GPU access set of a live allocation. `origin` identifies the
    /// device whose backing this allocation owns. The backend maps new
    /// devices before revoking old ones, preserving access shared by both sets.
    /// Native failures retain unfinished mapping work for a later retry or
    /// cleanup; callers should query [`Self::device_address`] before reporting
    /// access after an error.
    ///
    /// # Errors
    /// Rejects devices from another session or without a common address range,
    /// unsupported backing, and native mapping or unmapping failures.
    pub fn set_device_access(&mut self, origin: &Device, devices: &[&Device]) -> Result<(), Error> {
        if !self.originates_from(origin) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "origin device does not own this allocation",
            });
        }
        let mut states = Vec::new();
        states
            .try_reserve(devices.len())
            .map_err(|_| Error::Operation {
                kind: ErrorKind::ResourceExhausted,
                detail: "device access list is exhausted",
            })?;
        for device in devices {
            if self.driver_instance != device.endpoint.driver_instance {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "access device belongs to another allocation session",
                });
            }
            if self.inner.is_device_local()
                && !device.endpoint.can_access_local_memory(&origin.endpoint)
            {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "access device has no qualified route to local memory",
                });
            }
            let (_, state) = device.linux_kfd()?;
            states.push(state);
        }
        <driver::KfdDriver as AllocationDriver>::set_allocation_access(
            self.inner.driver_state_mut(),
            &states,
        )
    }

    /// Returns whether this allocation's physical backing originated on the
    /// supplied device. Devices from another session never match.
    #[must_use]
    pub fn originates_from(&self, device: &Device) -> bool {
        self.driver_instance == device.endpoint.driver_instance
            && device
                .linux_kfd()
                .is_ok_and(|(_, state)| self.inner.originates_from(state))
    }

    /// Returns opaque driver metadata retained with an imported native
    /// resource. Its format is defined by the interop operation that created
    /// the allocation; ordinary allocations return an empty slice.
    #[must_use]
    pub fn metadata(&self) -> &[u8] {
        self.inner.driver_state().metadata()
    }
    /// Releases device mappings, the native allocation, and then its virtual
    /// address reservation.
    /// The adapter must first ensure all host mappings and future device uses
    /// have ended. Cleanup performs no execution wait or cache transition, and
    /// resumes only the unfinished native steps after a retryable failure.
    ///
    /// # Errors
    /// Returns the failing native cleanup operation without discarding remaining
    /// ownership. Ambiguous native outcomes return `DriverContract` and retain
    /// backing for process teardown; retrying cannot make an unsafe replay safe.
    /// Cached addresses must not be used once freeing has begun.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}

/// Cached host-only storage facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAllocationInfo {
    /// Host address of the mapped storage.
    pub host_address: usize,
    /// Requested mapped bytes.
    pub size: u64,
}
/// Host allocation owner and cached facts supplied by a host-memory driver.
pub(crate) struct DriverHostAllocation<D: HostMemoryDriver> {
    inner: Owned<D::HostAllocation>,
    info: HostAllocationInfo,
}

impl<D: HostMemoryDriver> DriverHostAllocation<D> {
    pub(crate) fn new(inner: Owned<D::HostAllocation>) -> Self {
        let info = inner.cached_info();
        Self { inner, info }
    }

    pub(crate) fn info(&self) -> HostAllocationInfo {
        self.info
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        D::free_host(&mut self.inner)
    }
}

/// Owns host-only storage and the allocator used for its ownership record.
/// It has no activated-device or session-connection dependency. The adapter
/// still enforces its public scope and mapping lifetimes, and callback state
/// must outlive this owner. Host cleanup is the same under both driver context
/// lifetime policies.
pub struct HostAllocation {
    pub(crate) inner: DriverHostAllocation<driver::KfdDriver>,
}

impl HostAllocation {
    /// Copies the original host address and native extent without a system call.
    /// Freeing the storage does not rewrite this snapshot; its address is valid
    /// only until cleanup begins.
    #[must_use]
    pub fn info(&self) -> HostAllocationInfo {
        self.inner.info()
    }
    /// Releases the host mapping after all accesses through its address have ended.
    /// The adapter owns logical mapping borrows and must discharge them first.
    /// Repeated successful cleanup is harmless; no device operation is involved.
    ///
    /// # Errors
    /// A native unmapping error leaves the reservation owned for retry.
    /// Inherited process state is rejected before unmapping. This operation
    /// allocates no metadata.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner.free()
    }
}
/// Returns the native host virtual-memory page size for allocation rounding.
/// This query does not acquire an endpoint and is independent of any device
/// queue or address-space page layout.
///
/// # Errors
/// Returns a native error if the platform does not report a valid page size.
pub fn host_page_size() -> Result<u64, Error> {
    driver::KfdDriver::host_page_size()
}

/// Qualifies explicit host-cache maintenance and returns its line granularity.
/// The adapter uses success to advertise a mapping's maintenance recipe. A
/// backend qualifies both an available maintenance operation and its exact
/// granularity; this call opens no native endpoint and allocates no metadata.
///
/// # Errors
/// Returns `Unsupported` when the target has no implemented recipe or lacks the
/// required instruction and valid line size. Callers must not advertise a cache
/// operation using an unqualified nominal hardware line size.
pub fn host_cache_line_size() -> Result<u32, Error> {
    driver::KfdDriver::host_cache_line_size()
}
/// Executes the qualified host-cache maintenance recipe over a nonempty host
/// range. The backend provides the writeback, invalidation, and ordering
/// sequence required by the current host architecture. The adapter checks
/// mapping permissions and handles any public empty-range no-op before calling
/// this bridge. This operation neither waits for device work nor supplies its
/// retirement. A zero `line_size` selects an ordering-only recipe for a
/// write-combined view; nonzero values select the qualified write-back recipe.
///
/// # Errors
/// Rejects an unsupported recipe, a line size different from the qualified
/// value, a null address, an empty range, or numeric address overflow before
/// executing cache instructions. Address validity itself is a caller obligation.
///
/// # Safety
/// Every intersecting cache line, including bytes outside the logical range,
/// must remain mapped through the call. The caller must externally synchronize
/// access to those complete lines and prevent concurrent unmapping.
#[allow(unsafe_code)]
pub unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error> {
    // SAFETY: The public caller provides the live mapped range required by the
    // private native boundary; the backend validates the numeric extent.
    unsafe { driver::KfdDriver::host_cache_control(pointer, length, line_size) }
}

impl Device {
    /// Returns the current bytes available for allocation on this GPU device.
    ///
    /// This compatibility entry point requires a GPU endpoint. New GPU callers
    /// can use [`GpuDevice::available_memory`] directly.
    #[doc(hidden)]
    pub fn available_memory(&self) -> Result<u64, Error> {
        self.gpu()?.available_memory()
    }

    /// Creates owned backing and establishes its device mapping before
    /// returning. `kind` selects owned system or local placement; borrowed
    /// caller pages must use [`Self::register_host`]. Permissions are never
    /// widened. The native extent is page-aligned and separate from the logical
    /// byte range maintained by the adapter. The owner retains its address-space
    /// and platform dependencies, and takes its metadata allocator from that
    /// address space.
    ///
    /// # Errors
    /// Rejects unsupported permissions or placement and invalid extents before
    /// mutation. Native allocation, host mapping, device mapping, or loss checks
    /// can then fail. Partial mapping progress is preserved during rollback; an
    /// ambiguous kernel result retains any possibly reachable backing.
    pub fn allocate(
        &self,
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        let kind = OwnedMemoryKind::try_from(kind)?;
        let (driver, state) = self.linux_kfd()?;
        let inner = driver.allocate_owned(state, &[], kind, size, alignment, permissions)?;
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

    /// Registers caller-owned host pages for device access.
    ///
    /// # Errors
    /// Returns invalid extents, permission, native registration, and mapping
    /// errors with the same cleanup guarantees as [`Self::allocate`].
    ///
    /// # Safety
    /// `address` must identify `size` bytes of mapped host memory, and the
    /// complete aligned page cover must remain mapped. The caller must retain
    /// that backing and synchronize every CPU and device access until
    /// [`Allocation::free`] succeeds. After an ambiguous native failure, the
    /// pages must remain live until process teardown because KFD may still
    /// reference them.
    #[allow(unsafe_code)]
    pub unsafe fn register_host(
        &self,
        address: usize,
        cache: HostCachePolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        // SAFETY: The caller owns the complete page cover through successful
        // free or process teardown, as required by this public raw contract.
        let (driver, state) = self.linux_kfd()?;
        let inner = unsafe {
            driver.register_host(
                state,
                &[],
                HostRegistration {
                    address,
                    cache,
                    size,
                    alignment,
                    permissions,
                },
            )?
        };
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

    /// Creates device-local backing inside this process's native scratch
    /// aperture. The returned owner keeps both the physical allocation and its
    /// aperture range live until [`Allocation::free`] succeeds.
    ///
    /// # Errors
    /// Rejects invalid extents or unavailable local storage. Native scratch-base
    /// programming, allocation, mapping, and loss checks may also fail.
    pub(crate) fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        let (driver, state) = self.linux_kfd()?;
        let inner = driver.allocate_queue_scratch(state, size)?;
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

    /// Maps the device's process-level MMIO remap page.
    #[doc(hidden)]
    pub(crate) fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        let (driver, state) = self.linux_kfd()?;
        let inner = driver.map_mmio_remap(state)?;
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

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
        if matches!(kind, MemoryKind::RegisteredHost { .. }) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "borrowed host pages cannot back detached virtual memory",
            });
        }
        let kind = OwnedMemoryKind::try_from(kind)?;
        let (driver, state) = self.linux_kfd()?;
        let owner = Shared::try_new_uninit(driver.allocator())?;
        let inner = driver.create_virtual_memory(state, kind, size, pinned, uncached)?;
        Ok(VirtualMemory {
            inner: DriverVirtualMemory::new(driver.clone(), owner.write(inner)),
        })
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
        let (driver, state) = self.linux_kfd()?;
        Ok(VirtualDeviceMapping {
            inner: memory.inner.map_device(
                driver,
                state,
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

    /// Creates one owned allocation mapped into this device and every
    /// peer address space before returning. All devices must belong to this
    /// core session, and the native backend establishes one common device
    /// virtual address. Repeated live-device wrappers sharing one native
    /// address space reuse that mapping. Device-local placement additionally
    /// requires a cached directional route from every peer to this physical
    /// owner. Per-device permission differences are not representable by the
    /// current native path.
    ///
    /// # Errors
    /// Rejects a session mismatch, differing permissions, unsupported
    /// placement, or an empty common address envelope before native allocation.
    /// Native acquisition and rollback failures have the same ownership behavior
    /// as [`Self::allocate`]. Borrowed host pages must use
    /// [`Self::register_host_with_peers`].
    pub fn allocate_with_peers(
        &self,
        peers: &[&Self],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        let kind = OwnedMemoryKind::try_from(kind)?;
        let states = self.peer_states(peers, kind.get())?;
        let (driver, state) = self.linux_kfd()?;
        let inner =
            driver.allocate_owned(state, states.as_slice(), kind, size, alignment, permissions)?;
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

    /// Registers caller-owned host pages in this device and all peer VMs.
    ///
    /// # Errors
    /// Reports the same validation, native acquisition, and rollback failures
    /// as [`Self::allocate_with_peers`].
    ///
    /// # Safety
    /// `address` must identify `size` bytes of mapped host memory, and the
    /// complete aligned page cover must remain mapped. The caller must retain
    /// that backing and synchronize every CPU and device access until
    /// [`Allocation::free`] succeeds. After an ambiguous native failure, the
    /// pages must remain live until process teardown because a peer mapping
    /// may still reference them.
    #[allow(unsafe_code)]
    pub unsafe fn register_host_with_peers(
        &self,
        peers: &[&Self],
        address: usize,
        cache: HostCachePolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        let states = self.peer_states(peers, MemoryKind::RegisteredHost { address, cache })?;
        // SAFETY: The caller retains the complete page cover through successful
        // free or process teardown across every requested peer VM.
        let (driver, state) = self.linux_kfd()?;
        let inner = unsafe {
            driver.register_host(
                state,
                states.as_slice(),
                HostRegistration {
                    address,
                    cache,
                    size,
                    alignment,
                    permissions,
                },
            )?
        };
        Ok(Allocation::from_native(
            inner,
            self.endpoint.driver_instance,
        ))
    }

    fn peer_states<'a>(
        &self,
        peers: &'a [&Self],
        kind: MemoryKind,
    ) -> Result<Buffer<&'a driver::KfdDeviceState>, Error> {
        let (driver, state) = self.linux_kfd()?;
        for peer in peers {
            let (peer_driver, _) = peer.linux_kfd()?;
            if !Shared::ptr_eq(driver, peer_driver) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "peer devices must belong to one session",
                });
            }
            if matches!(kind, MemoryKind::DeviceLocal { .. })
                && !peer.endpoint.can_access_local_memory(&self.endpoint)
            {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "peer device has no qualified route to local memory",
                });
            }
        }
        let mut states = Buffer::try_with_capacity(peers.len(), driver.allocator())?;
        for peer in peers {
            let (_, peer_state) = peer.linux_kfd()?;
            if driver::AddressSpaceInfo::shares_address_domain(state, peer_state)
                || states.iter().any(|state: &&driver::KfdDeviceState| {
                    driver::AddressSpaceInfo::shares_address_domain(*state, peer_state)
                })
            {
                continue;
            }
            states.try_push(peer_state)?;
        }
        Ok(states)
    }
}

impl GpuDevice<'_> {
    /// Creates device-local backing inside this GPU's native scratch aperture.
    /// The returned owner keeps both the physical allocation and its aperture
    /// range live until [`Allocation::free`] succeeds.
    ///
    /// # Errors
    /// Rejects invalid extents or unavailable local storage. Native
    /// scratch-base programming, allocation, mapping, and loss checks may also
    /// fail.
    pub fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        self.device.allocate_queue_scratch(size)
    }

    /// Maps this GPU's process-level MMIO remap page.
    ///
    /// This is a GPU transport capability rather than a universal device
    /// memory operation. Callers must keep the returned allocation alive for
    /// every use of addresses derived from the mapping.
    ///
    /// # Errors
    /// Returns `Unsupported` when the backend or GPU exposes no MMIO remap page,
    /// and otherwise reports native allocation or mapping failures.
    pub fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        self.device.map_mmio_remap()
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
        let intervals =
            DeviceIntervals::<FakeDeviceState>::new(crate::host_storage::Allocator::system());
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
        let mut reservation = session
            .reserve_virtual_address(&[&device], size, granularity, 0)
            .unwrap();
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
        let mut reservation = session
            .reserve_virtual_address(&[&device], 4096, 4096, 0)
            .unwrap();
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
