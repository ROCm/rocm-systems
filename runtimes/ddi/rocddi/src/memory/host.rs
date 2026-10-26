// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only backing and address reservations.
//!
//! These operations do not activate an endpoint or establish device access.
//! A CPU executing host code can use this backing independently of whether it
//! is also represented as an HSA agent. Caller-owned `malloc` or Rust allocator
//! memory stays under the caller's ownership; device registration is separate.
//! Device registration of rocddi-owned backing retains this owner through
//! [`Device::register_host_allocation`](crate::device::Device::register_host_allocation).
//! CPU access to device-backed memory uses that resource's qualified host
//! mapping, not a second host-only allocation.
//! Native page mapping and release stay behind this module's private OS owner.

use crate::Error;
use crate::ErrorKind;
use crate::memory::storage::Shared;
use crate::os;

/// Cached host-only storage facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAllocationInfo {
    /// Host address of the mapped storage.
    pub host_address: usize,
    /// Requested mapped bytes.
    pub size: u64,
}
/// Owns host-only storage and its ownership record.
/// It has no activated-device or session-connection dependency. The adapter
/// still enforces its public scope and mapping lifetimes. Host cleanup is the
/// same under both driver context lifetime policies.
pub struct HostAllocation {
    /// OS-owned mapping retained while device registrations borrow its pages.
    inner: Shared<os::HostAllocation>,
    /// Host address and byte extent cached before cleanup.
    info: HostAllocationInfo,
}

/// Stable host address and extent of one reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostReservationInfo {
    /// Base of the reserved, initially inaccessible range.
    pub address: usize,
    /// Reserved bytes.
    pub size: u64,
}

/// Owns a host address range whose page tables can be replaced independently
/// of the source allocation's original host mapping.
pub struct HostReservation {
    inner: os::HostReservation,
    info: HostReservationInfo,
}

impl HostReservation {
    /// Reserves inaccessible host addresses without allocating backing.
    ///
    /// # Errors
    /// Rejects invalid page geometry or reports native address exhaustion.
    pub fn reserve(size: u64, alignment: u64) -> Result<Self, Error> {
        let size = usize::try_from(size).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host reservation length exceeds address width",
        })?;
        let alignment = usize::try_from(alignment).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host reservation alignment exceeds address width",
        })?;
        let inner = os::HostReservation::create(size, alignment, std::process::id())
            .map_err(|source| os::native_error("host reservation mmap", source))?;
        let (address, size) = inner.extent();
        Ok(Self {
            inner,
            info: HostReservationInfo {
                address,
                size: size as u64,
            },
        })
    }

    /// Returns the stable address and length without a native call.
    #[must_use]
    pub const fn info(&self) -> HostReservationInfo {
        self.info
    }

    /// Installs a remappable host backing range at the requested offset.
    /// The caller keeps the source alive until this range is replaced or
    /// detached, and synchronizes all accesses before replacement.
    ///
    /// # Errors
    /// Reports invalid ranges, unsupported sources, or native mapping failure.
    pub fn replace(
        &mut self,
        target_offset: u64,
        length: u64,
        source: &HostAllocation,
        source_offset: u64,
        writable: bool,
    ) -> Result<(), Error> {
        self.inner
            .replace(
                target_offset,
                length,
                &source.inner,
                source_offset,
                writable,
            )
            .map_err(|error| os::native_error("host reservation replace", error))
    }

    /// Makes a subrange inaccessible while retaining its stable address.
    ///
    /// # Errors
    /// Reports invalid geometry or native mapping failure.
    pub fn clear(&mut self, offset: u64, length: u64) -> Result<(), Error> {
        self.inner
            .clear(offset, length)
            .map_err(|error| os::native_error("host reservation detach", error))
    }

    /// Removes the complete reservation after all users have finished.
    ///
    /// # Errors
    /// Retains the owner for retry after a native release failure.
    pub fn free(&mut self) -> Result<(), Error> {
        self.inner
            .free()
            .map_err(|error| os::native_error("host reservation munmap", error))
    }
}

/// Retains mapped host pages while a device driver may still reference them.
/// The native mapping representation stays private to host memory.
pub(crate) struct HostBackingLease {
    /// Shared owner that prevents host unmapping during device registration.
    inner: Shared<os::HostAllocation>,
}

impl HostBackingLease {
    pub(crate) fn extent(&self) -> (usize, u64) {
        self.inner.extent()
    }
}

impl HostAllocation {
    /// Allocates host-only storage without activating a device or opening a
    /// driver context. `size` is a nonzero multiple of [`host_page_size`];
    /// `alignment` is a power of two at least that large. The host service may
    /// reserve a larger backing extent while preserving the requested range.
    ///
    /// # Errors
    /// Rejects invalid extents and reports metadata exhaustion or native
    /// mapping failure. The allocation retains its creating process identity
    /// so inherited state cannot be released in a fork child.
    pub fn allocate(size: u64, alignment: u64) -> Result<Self, Error> {
        let inner = os::HostAllocation::create(size, alignment, std::process::id())?;
        let (host_address, size) = inner.extent();
        let info = HostAllocationInfo { host_address, size };
        Ok(Self { inner, info })
    }

    /// Allocates host pages that can back independently replaceable stable
    /// host reservations. The owner keeps the native source alive until every
    /// installed reservation alias has been detached.
    ///
    /// # Errors
    /// Rejects invalid geometry or reports native allocation failure.
    pub fn allocate_remappable(size: u64, alignment: u64) -> Result<Self, Error> {
        let inner = os::HostAllocation::create_remappable(size, alignment, std::process::id())?;
        let (host_address, size) = inner.extent();
        Ok(Self {
            inner,
            info: HostAllocationInfo { host_address, size },
        })
    }

    /// Returns whether this owner supplies stable reservation backing.
    #[must_use]
    pub fn is_remappable(&self) -> bool {
        self.inner.source().is_some()
    }

    /// Copies the original host address and native extent without a system call.
    /// Freeing the storage does not rewrite this snapshot; its address is valid
    /// only until cleanup begins.
    #[must_use]
    pub fn info(&self) -> HostAllocationInfo {
        self.info
    }

    /// Retains mapped backing for a device registration. A successfully freed
    /// host allocation has no pages to register.
    pub(crate) fn registration_lease(&self) -> Result<HostBackingLease, Error> {
        if self.inner.extent().1 == 0 {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "host allocation has already been freed",
            });
        }
        Ok(HostBackingLease {
            inner: self.inner.clone(),
        })
    }

    /// Releases the host mapping after all accesses through its address have ended.
    /// The adapter owns logical mapping borrows and must discharge them first.
    /// Repeated successful cleanup is harmless; no device operation is involved.
    ///
    /// # Errors
    /// Returns `Busy` while device registrations retain the backing. A native
    /// unmapping error leaves the reservation owned for retry.
    /// Inherited process state is rejected before unmapping. This operation
    /// allocates no metadata.
    pub fn free(&mut self) -> Result<(), Error> {
        let backing = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "host allocation has live device registrations",
        })?;
        backing.free()
    }
}
/// Returns the native host virtual-memory page size for allocation rounding.
/// This query does not acquire an endpoint and is independent of any device
/// queue or address-space page layout.
///
/// # Errors
/// Returns a native error if the platform does not report a valid page size.
pub fn host_page_size() -> Result<u64, Error> {
    os::host_page_size()
}
