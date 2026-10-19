// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device-accessible allocation ownership and registration.

use super::{AllocationInfo, DeviceAccess, HostMappingPolicy, HostRegistration, MemoryKind};
use crate::device::Device;
use crate::driver::{self, AddressSpaceInfo, AllocationOperations, CachedInfo};
use crate::memory::storage::TryPush;
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};

/// Allocation owner and cached facts shared by drivers that can allocate
/// device-accessible backing. The driver remains live through native cleanup.
pub(crate) struct DriverAllocation<D: AllocationOperations> {
    inner: Owned<D::Allocation>,
    // A live allocation keeps its driver open even after the creating device
    // is dropped. Drop releases the native owner before this reference.
    driver: Shared<D>,
    info: AllocationInfo,
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
    driver_instance: u64,
    origin: &Device,
    devices: &[&Device],
    state_of: F,
) -> Result<(), Error>
where
    D: AllocationOperations,
    F: for<'a> Fn(&'a Device) -> Option<&'a D::DeviceState>,
{
    let mut states = crate::memory::storage::try_vec_with_capacity(devices.len())?;
    for device in devices {
        if driver_instance != device.endpoint.driver_instance {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "access device belongs to another allocation session",
            });
        }
        if allocation.is_device_local()
            && !device.endpoint.can_access_local_memory(&origin.endpoint)
        {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "access device has no qualified route to local memory",
            });
        }
        states.try_push(state_of(device).ok_or(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "access device has no matching allocation driver",
        })?)?;
    }

    D::set_allocation_access(allocation.driver_state_mut(), states.as_slice())
}

/// Owns one native allocation, its exact VM dependencies, and cleanup progress.
/// Explicit free reports failures. If final Drop cannot complete cleanup, it
/// retains potentially reachable native backing instead of recycling its VA.
pub struct Allocation {
    pub(crate) inner: driver::AllocationState,
    pub(crate) driver_instance: u64,
}

impl Allocation {
    pub(crate) fn new(inner: driver::AllocationState, driver_instance: u64) -> Self {
        Self {
            inner,
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
        self.inner
            .set_device_access(self.driver_instance, origin, devices)
    }

    /// Returns whether this allocation's physical backing originated on the
    /// supplied device. Devices from another session never match.
    #[must_use]
    pub fn originates_from(&self, device: &Device) -> bool {
        self.driver_instance == device.endpoint.driver_instance
            && self.inner.originates_from(&device.driver_state)
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
        self.driver_state.allocate_owned(
            self.endpoint.driver_instance,
            kind,
            size,
            alignment,
            permissions,
        )
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
        unsafe {
            self.driver_state
                .register_host(self.endpoint.driver_instance, request)
        }
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
            if requires_local_route && !peer.endpoint.can_access_local_memory(&self.endpoint) {
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
