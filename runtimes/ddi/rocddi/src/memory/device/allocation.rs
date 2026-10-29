// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device-accessible allocation ownership and registration.

use super::{
    AllocationInfo, DeviceAccess, HostMappingPolicy, HostRegistration, MemoryKind,
    registered_host_page_cover,
};
use crate::device::Device;
use crate::driver::{self, AddressSpaceInfo, AllocationOperations, CachedInfo};
use crate::memory::host::host_page_size;
use crate::memory::storage::TryPush;
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

/// Native cleanup observed while a construction call rolls back.
///
/// The original construction error remains the operation result. This record
/// carries only the first independent cleanup error and a conservative
/// allocation-owned payload charge.
#[derive(Default)]
pub struct RollbackReport {
    /// First native release failure, if any.
    pub status: Option<Error>,
    /// Payload bytes whose native release was not established.
    pub abandoned_byte_length: u64,
    charge_backing: bool,
}

thread_local! {
    static CURRENT_ROLLBACK: Cell<*mut RollbackReport> = const { Cell::new(std::ptr::null_mut()) };
}

struct RollbackScope {
    previous: *mut RollbackReport,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for RollbackScope {
    fn drop(&mut self) {
        CURRENT_ROLLBACK.with(|current| current.set(self.previous));
    }
}

/// Captures cleanup failures from native owners discarded during construction.
/// The scope is thread-local to the synchronous native call and never becomes
/// instance or session state.
pub fn with_rollback_report<R>(
    charge_backing: bool,
    body: impl FnOnce() -> R,
) -> (R, RollbackReport) {
    let mut report = RollbackReport {
        charge_backing,
        ..RollbackReport::default()
    };
    let scope = RollbackScope {
        previous: CURRENT_ROLLBACK.with(|current| current.replace(&raw mut report)),
        _not_send: PhantomData,
    };
    let result = body();
    drop(scope);
    (result, report)
}

#[allow(unsafe_code)]
pub(crate) fn record_rollback(error: Error, abandoned_byte_length: u64) {
    CURRENT_ROLLBACK.with(|current| {
        let report = current.get();
        if !report.is_null() {
            // SAFETY: The pointer belongs to the active synchronous capture
            // scope on this thread and is restored before that scope returns.
            let report = unsafe { &mut *report };
            if report.status.is_none() {
                report.status = Some(error);
            }
            if report.charge_backing {
                report.abandoned_byte_length = report
                    .abandoned_byte_length
                    .saturating_add(abandoned_byte_length);
            }
        }
    });
}

/// Allocation owner and cached facts shared by drivers that can allocate
/// device-accessible backing. The driver remains live through native cleanup.
pub(crate) struct DriverAllocation<D: AllocationOperations> {
    /// Native allocation whose cleanup runs before the driver is released.
    inner: Owned<D::Allocation>,
    /// Keeps the creating driver alive after the device handle is dropped.
    driver: Shared<D>,
    /// Copyable address and extent facts retained across cleanup.
    info: AllocationInfo,
    /// Prevents replay of a completed native release.
    released: bool,
}

impl<D: AllocationOperations> DriverAllocation<D> {
    pub(crate) fn new(driver: Shared<D>, inner: Owned<D::Allocation>) -> Self {
        let info = inner.cached_info();
        Self {
            inner,
            driver,
            info,
            released: false,
        }
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

    pub(crate) fn shares_driver_instance(&self, driver: &Shared<D>) -> bool {
        Shared::ptr_eq(&self.driver, driver)
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
        if !self.released {
            D::free_allocation(&mut self.inner)?;
            self.released = true;
        }
        Ok(())
    }
}

impl<D: AllocationOperations> Drop for DriverAllocation<D> {
    fn drop(&mut self) {
        if self.free().is_err() {
            // Native cleanup may still reference this driver's context. The
            // native owner quarantines its own dependencies on final drop.
            std::mem::forget(self.driver.clone());
        }
    }
}

/// Validates and applies a replacement access set for any allocation driver.
pub(crate) fn set_device_access_for<D, F>(
    allocation: &mut DriverAllocation<D>,
    origin: &Device,
    devices: &[&Device],
    state_of: F,
) -> Result<(), Error>
where
    D: AllocationOperations,
    F: for<'a> Fn(&'a Device) -> Option<(&'a Shared<D>, &'a D::DeviceState)>,
{
    let mut states = crate::memory::storage::try_vec_with_capacity(devices.len())?;
    for device in devices {
        let (driver, state) = state_of(device).ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "access device belongs to another allocation session",
        })?;
        if !allocation.shares_driver_instance(driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "access device belongs to another allocation session",
            });
        }
        if allocation.is_device_local() && !device.can_access_local_memory(origin) {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "access device has no qualified route to local memory",
            });
        }
        states.try_push(state)?;
    }

    D::set_allocation_access(allocation.driver_state_mut(), states.as_slice())
}

/// Owns one native allocation, its exact VM dependencies, and cleanup progress.
/// Explicit free reports failures. If final Drop cannot complete cleanup, it
/// retains potentially reachable native backing instead of recycling its VA.
pub struct Allocation {
    /// Concrete owner and driver selected when the allocation was created.
    pub(crate) inner: driver::AllocationState,
}

impl Allocation {
    pub(crate) fn new(inner: driver::AllocationState) -> Self {
        Self { inner }
    }

    pub(crate) fn shares_driver_instance(&self, device: &Device) -> bool {
        self.inner.shares_driver_instance(&device.driver_state)
    }

    /// Returns creation-time addresses and extent without touching the driver.
    /// The adapter controls their public lifetime. This snapshot remains cached
    /// after cleanup begins, when its addresses must no longer be used.
    #[must_use]
    pub fn info(&self) -> AllocationInfo {
        self.inner.info()
    }

    /// Returns the registered page base's offset within an existing physical
    /// backing when this allocation reuses another mapped BO. Add the byte
    /// offset returned by `register_host_range` for the exact caller pointer.
    #[must_use]
    pub fn host_alias_source_offset(&self) -> Option<u64> {
        self.inner.host_alias_source_offset()
    }

    /// Publishes an opaque frontend identity for later registrations of this
    /// owned host-visible BO. This identity is used only when no native
    /// DMA-BUF identity is available; callers must keep it unique among their
    /// live physical backings in this process.
    ///
    /// # Errors
    /// Rejects an allocation without a mapped BO or a conflicting identity.
    pub fn set_host_alias_identity(&self, identity: [u64; 2]) -> Result<(), Error> {
        self.inner.set_host_alias_identity(identity)
    }

    /// Reports allocation-owned payload still tied to a native backing handle
    /// or a live host mapping. Imported and borrowed backing reports zero.
    #[must_use]
    pub fn remaining_backing_byte_length(&self) -> u64 {
        self.inner.remaining_backing_byte_length()
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
        if !self.shares_driver_instance(device) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "device belongs to another allocation session",
            });
        }
        self.inner.device_address(&device.driver_state)
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
        self.inner.set_device_access(origin, devices)
    }

    /// Returns whether this allocation's physical backing originated on the
    /// supplied device. Devices from another session never match.
    #[must_use]
    pub fn originates_from(&self, device: &Device) -> bool {
        self.shares_driver_instance(device) && self.inner.originates_from(&device.driver_state)
    }

    /// Returns opaque driver metadata retained with an imported native
    /// resource. Its format is defined by the interop operation that created
    /// the allocation; ordinary allocations return an empty slice.
    #[must_use]
    pub fn metadata(&self) -> &[u8] {
        self.inner.metadata()
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

impl Device {
    /// Registers an arbitrary host byte range, including an unaligned first
    /// byte. Returns the allocation for the complete page cover and the byte
    /// offset of `address` from its base. Add that offset to a device address
    /// obtained from the returned allocation.
    ///
    /// # Errors
    /// Rejects an invalid page cover and propagates native registration errors.
    ///
    /// # Safety
    /// The complete page cover must remain mapped and synchronized until
    /// successful release. An ambiguous native failure extends that obligation
    /// through process teardown.
    #[allow(unsafe_code)]
    pub unsafe fn register_host_range(
        &self,
        address: usize,
        policy: HostMappingPolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<(Allocation, u64), Error> {
        // SAFETY: The caller retains the complete page cover.
        unsafe {
            self.register_host_range_with_peers(&[], address, policy, size, alignment, permissions)
        }
    }

    /// Registers a host byte range in this device and every listed peer.
    /// The returned offset applies to each mapped device address.
    ///
    /// # Errors
    /// Rejects an invalid page cover or peer set and propagates native errors.
    ///
    /// # Safety
    /// The complete page cover must remain mapped and synchronized until
    /// successful release. An ambiguous native failure extends that obligation
    /// through process teardown.
    #[allow(unsafe_code)]
    pub unsafe fn register_host_range_with_peers(
        &self,
        peers: &[&Self],
        address: usize,
        policy: HostMappingPolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<(Allocation, u64), Error> {
        let page = usize::try_from(host_page_size()?).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host page size exceeds address width",
        })?;
        let size = usize::try_from(size).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host registration size exceeds address width",
        })?;
        let (base, cover) =
            registered_host_page_cover(address, size, page).ok_or(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid registered host page cover",
            })?;
        // SAFETY: The caller retains the cover, which is exactly the extent
        // passed to the native registration.
        let allocation = unsafe {
            self.register_host_with_peers(
                peers,
                base,
                policy,
                cover as u64,
                alignment,
                permissions,
            )?
        };
        Ok((allocation, (address - base) as u64))
    }

    /// Creates owned backing and establishes this device's mapping before
    /// returning. `kind` selects owned system or local placement; borrowed
    /// caller pages must use [`Self::register_host`]. Permissions are never
    /// widened. The native extent is page-aligned and separate from the logical
    /// byte range maintained by the adapter. The owner retains its address-space
    /// and platform dependencies. A host CPU represented by an activated endpoint uses
    /// this same device interface when its driver supports the placement;
    /// host-only storage remains available through
    /// [`HostAllocation::allocate`](crate::memory::host::HostAllocation::allocate).
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
        self.driver_state
            .allocate_owned(kind, size, alignment, permissions)
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
    /// pages must remain live until process teardown because the driver may still
    /// reference them.
    #[allow(unsafe_code)]
    pub unsafe fn register_host(
        &self,
        address: usize,
        policy: HostMappingPolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        // SAFETY: The caller owns the complete page cover through successful
        // free or process teardown, as required by this public raw contract.
        let request = HostRegistration {
            address,
            policy,
            size,
            alignment,
            permissions,
            backing: None,
        };
        // SAFETY: The caller preserves the complete registered page cover.
        unsafe { self.driver_state.register_host(request) }
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
        self.driver_state
            .allocate_with_peers(self, peers, kind, size, alignment, permissions)
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
        policy: HostMappingPolicy,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        let request = HostRegistration {
            address,
            policy,
            size,
            alignment,
            permissions,
            backing: None,
        };
        // SAFETY: The caller retains the complete page cover through
        // successful free or process teardown in every requested VM.
        unsafe {
            self.driver_state
                .register_host_with_peers(self, peers, request)
        }
    }

    pub(crate) fn peer_states_for<'a, D, F>(
        &self,
        peers: &'a [&Self],
        requires_local_route: bool,
        driver: &Shared<D>,
        origin_state: &D::DeviceState,
        state_of: F,
    ) -> Result<Vec<&'a D::DeviceState>, Error>
    where
        D: AllocationOperations + 'a,
        D::DeviceState: AddressSpaceInfo,
        F: for<'b> Fn(&'b Device) -> Option<(&'b Shared<D>, &'b D::DeviceState)>,
    {
        for peer in peers {
            let (peer_driver, _) = state_of(peer).ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "peer device has no matching allocation driver",
            })?;
            if !Shared::ptr_eq(driver, peer_driver) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "peer devices must belong to one session",
                });
            }
            if requires_local_route && !peer.can_access_local_memory(self) {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "peer device has no qualified route to local memory",
                });
            }
        }

        let mut states = crate::memory::storage::try_vec_with_capacity(peers.len())?;
        for peer in peers {
            let (_, peer_state) = state_of(peer).ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "peer device has no matching allocation driver",
            })?;
            if origin_state.shares_address_domain(peer_state)
                || states
                    .iter()
                    .any(|state: &&D::DeviceState| state.shares_address_domain(peer_state))
            {
                continue;
            }
            states.try_push(peer_state)?;
        }
        Ok(states)
    }
}

#[cfg(test)]
mod rollback_tests {
    use super::*;

    #[test]
    fn rollback_capture_keeps_first_error_and_charges_only_owned_payload() {
        let failure = || Error::Operation {
            kind: ErrorKind::Busy,
            detail: "injected release failure",
        };
        let (result, report) = with_rollback_report(true, || {
            record_rollback(failure(), 4096);
            record_rollback(failure(), 1024);
            7
        });
        assert_eq!(result, 7);
        assert_eq!(
            report.status.as_ref().map(Error::kind),
            Some(ErrorKind::Busy)
        );
        assert_eq!(report.abandoned_byte_length, 5120);

        let ((), borrowed) = with_rollback_report(false, || record_rollback(failure(), 4096));
        assert_eq!(
            borrowed.status.as_ref().map(Error::kind),
            Some(ErrorKind::Busy)
        );
        assert_eq!(borrowed.abandoned_byte_length, 0);
    }
}
