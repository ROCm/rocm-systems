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
use std::mem::ManuallyDrop;

/// Host address reservation and replacement geometry supported by the OS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostReservationCapabilities {
    /// Required reservation size multiple.
    pub reservation_granularity: u64,
    /// Independently replaceable native page unit.
    pub mapping_granularity: u64,
    /// Largest supported reservation, independent of live VA availability.
    pub maximum_size: u64,
    /// Largest power-of-two base alignment accepted by this implementation.
    pub maximum_alignment: u64,
    /// Inclusive usable host address lower bound.
    pub minimum_address: u64,
    /// Inclusive usable host address upper bound.
    pub maximum_address: u64,
}

/// OS-qualified host backing and replacement support, independent of GPUs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostMemoryCapabilities {
    /// Geometry of ordinary committed CPU backing.
    pub allocation: super::constraints::AllocationLimits,
    /// Backing can be projected into independent reservation views.
    pub remappable_backing: bool,
    /// Reservation support, absent until replacement and release are implemented.
    pub reservation: Option<HostReservationCapabilities>,
}

/// Reports supported host mechanisms without acquiring backing or a GPU.
///
/// # Errors
/// Fails if native host geometry cannot be determined.
pub fn capabilities() -> Result<HostMemoryCapabilities, Error> {
    os::host_memory_capabilities()
}

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
    inner: ManuallyDrop<Shared<os::HostAllocation>>,
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
    /// Selected replacement unit; the final unit may end at the reservation end.
    pub mapping_granularity: u64,
}

/// Owns a host address range whose page tables can be replaced independently
/// of the source allocation's original host mapping.
pub struct HostReservation {
    inner: os::HostReservation,
    info: HostReservationInfo,
}

impl HostReservation {
    /// Reserves inaccessible host addresses without allocating backing.
    /// Zero mapping granularity selects OS policy; nonzero requests select an
    /// exact replacement unit. The returned information records that choice.
    ///
    /// # Errors
    /// Rejects invalid page geometry or reports native address exhaustion.
    pub fn reserve(size: u64, alignment: u64, mapping_granularity: u64) -> Result<Self, Error> {
        let size = usize::try_from(size).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host reservation length exceeds address width",
        })?;
        let alignment = usize::try_from(alignment).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "host reservation alignment exceeds address width",
        })?;
        let mapping_granularity =
            usize::try_from(mapping_granularity).map_err(|_| Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "mapping granularity exceeds address width",
            })?;
        let inner =
            os::HostReservation::create(size, alignment, mapping_granularity, std::process::id())
                .map_err(|source| os::native_error("host address reservation", source))?;
        let (address, size) = inner.extent();
        let mapping_granularity = inner.mapping_granularity() as u64;
        Ok(Self {
            inner,
            info: HostReservationInfo {
                address,
                size: size as u64,
                mapping_granularity,
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
            .map_err(|error| os::native_error("host reservation release", error))
    }
}

/// Retains mapped host pages while a device driver may still reference them.
/// The native mapping representation stays private to host memory.
pub(crate) struct HostBackingLease {
    /// Shared owner that prevents host unmapping during device registration.
    inner: ManuallyDrop<Shared<os::HostAllocation>>,
}

impl HostBackingLease {
    pub(crate) fn extent(&self) -> Result<(usize, u64), Error> {
        self.inner
            .check_process()
            .map_err(|source| os::native_error("host backing process identity", source))?;
        Ok(self.inner.extent())
    }
}

impl Drop for HostBackingLease {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.inner.check_process().is_ok() {
            // SAFETY: This owner is manually dropped exactly once on its
            // creating process's path. A fork child leaves it untouched.
            unsafe { ManuallyDrop::drop(&mut self.inner) };
        }
    }
}

impl HostAllocation {
    fn ensure_process(&self) -> Result<(), Error> {
        self.inner
            .check_process()
            .map_err(|source| os::native_error("host allocation process identity", source))
    }

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
        Ok(Self {
            inner: ManuallyDrop::new(inner),
            info,
        })
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
            inner: ManuallyDrop::new(inner),
            info: HostAllocationInfo { host_address, size },
        })
    }

    /// Returns whether this owner supplies stable reservation backing.
    #[must_use]
    pub fn is_remappable(&self) -> bool {
        self.inner.is_remappable()
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
        self.ensure_process()?;
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
        self.ensure_process()?;
        let backing = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "host allocation has live device registrations",
        })?;
        backing.free()
    }
}

impl Drop for HostAllocation {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.inner.check_process().is_ok() {
            // SAFETY: This owner is manually dropped exactly once on its
            // creating process's path. A fork child leaves it untouched.
            unsafe { ManuallyDrop::drop(&mut self.inner) };
        }
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

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::memory::storage::Allocator;
    use std::ffi::c_void;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CallbackContext {
        lock: Mutex<()>,
        frees: AtomicUsize,
    }

    impl CallbackContext {
        fn allocator(&self) -> Allocator {
            // SAFETY: This context outlives every owner created by the test.
            unsafe {
                Allocator::from_callbacks(
                    std::ptr::from_ref(self).cast_mut().cast(),
                    Some(allocate),
                    Some(free),
                )
                .unwrap()
            }
        }
    }

    unsafe extern "C" fn allocate(_data: *mut c_void, length: u64, alignment: u64) -> *mut c_void {
        let (Ok(length), Ok(alignment)) = (usize::try_from(length), usize::try_from(alignment))
        else {
            return std::ptr::null_mut();
        };
        let mut pointer = std::ptr::null_mut();
        // SAFETY: The allocator contract supplies a valid power-of-two
        // alignment and a nonzero allocation length.
        if unsafe { libc::posix_memalign(&raw mut pointer, alignment, length) } == 0 {
            pointer
        } else {
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn free(data: *mut c_void, pointer: *mut c_void) {
        // SAFETY: Each test retains its callback context through parent drop
        // and child _exit.
        let context = unsafe { &*data.cast::<CallbackContext>() };
        let _held = context.lock.lock().unwrap();
        context.frees.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The pointer came from posix_memalign in this test allocator.
        unsafe { libc::free(pointer) };
    }

    fn wait_for_child(child: libc::c_long) {
        let pid = i32::try_from(child).unwrap();
        let mut status = 0;
        // SAFETY: This waits only for the child created by the calling test.
        assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
        assert_eq!(status, 0);
    }

    #[test]
    fn inherited_host_backing_cannot_register_without_parent_session() {
        let page = host_page_size().unwrap();
        let mut host = HostAllocation::allocate(page, page).unwrap();
        // No session was created by this test before fork. The backing must
        // reject registration independently of the runtime-image fence.
        // SAFETY: The child calls only process-guarded methods and exits
        // without running inherited Rust destructors.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: Bound a regression if inherited state is entered.
            unsafe { libc::alarm(3) };
            let rejected = host
                .registration_lease()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::Unsupported)
                && host
                    .free()
                    .err()
                    .is_some_and(|error| error.kind() == ErrorKind::Unsupported);
            // SAFETY: _exit avoids inherited cleanup.
            unsafe { libc::_exit(i32::from(!rejected)) }
        }
        wait_for_child(child);
        host.free().unwrap();
    }

    #[test]
    fn inherited_final_host_owner_skips_allocator_callback() {
        let context = CallbackContext {
            lock: Mutex::new(()),
            frees: AtomicUsize::new(0),
        };
        let page = host_page_size().unwrap();
        let mut host = context
            .allocator()
            .with_scope(|| HostAllocation::allocate(page, page))
            .unwrap();
        let held = context.lock.lock().unwrap();
        // SAFETY: The child drops only the copied owner and exits without
        // running any other inherited destructor.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: Detect a callback entering the copied locked mutex.
            unsafe { libc::alarm(3) };
            drop(host);
            // SAFETY: _exit avoids inherited test-harness state.
            unsafe { libc::_exit(0) }
        }
        wait_for_child(child);
        drop(held);
        host.free().unwrap();
        drop(host);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn inherited_final_lease_skips_allocator_callback() {
        let context = CallbackContext {
            lock: Mutex::new(()),
            frees: AtomicUsize::new(0),
        };
        let page = host_page_size().unwrap();
        let host = context
            .allocator()
            .with_scope(|| HostAllocation::allocate(page, page))
            .unwrap();
        let lease = host.registration_lease().unwrap();
        drop(host);
        let held = context.lock.lock().unwrap();
        // SAFETY: The child drops only the copied last lease and exits.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: Detect a callback entering the copied locked mutex.
            unsafe { libc::alarm(3) };
            let rejected = lease
                .extent()
                .err()
                .is_some_and(|error| error.kind() == ErrorKind::Unsupported);
            drop(lease);
            // SAFETY: _exit avoids inherited test-harness state.
            unsafe { libc::_exit(i32::from(!rejected)) }
        }
        wait_for_child(child);
        drop(held);
        drop(lease);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn freed_host_owner_retains_process_identity_until_metadata_drop() {
        let context = CallbackContext {
            lock: Mutex::new(()),
            frees: AtomicUsize::new(0),
        };
        let page = host_page_size().unwrap();
        let mut host = context
            .allocator()
            .with_scope(|| HostAllocation::allocate(page, page))
            .unwrap();
        host.free().unwrap();
        let held = context.lock.lock().unwrap();
        // SAFETY: The child drops only copied, already-freed metadata.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: Detect a callback entering the copied locked mutex.
            unsafe { libc::alarm(3) };
            drop(host);
            // SAFETY: _exit avoids inherited test-harness state.
            unsafe { libc::_exit(0) }
        }
        wait_for_child(child);
        drop(held);
        drop(host);
        assert_eq!(context.frees.load(Ordering::Relaxed), 1);
    }
}
