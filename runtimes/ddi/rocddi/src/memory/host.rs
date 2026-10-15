// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only backing and host cache operations.
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
use crate::cpu_cache;
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
    inner: Shared<os::HostAllocation>,
    info: HostAllocationInfo,
}

/// Retains mapped host pages while a device driver may still reference them.
/// The native mapping representation stays private to host memory.
pub(crate) struct HostBackingLease {
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
    cpu_cache::cache_line_size()
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
    unsafe { cpu_cache::cache_control(pointer, length, line_size) }
}
