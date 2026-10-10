// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux host memory and address-reservation ownership.
//!
//! Host allocation is independent of GPU activation and works for both driver
//! context lifetime policies. `Reservation` owns the underlying virtual range;
//! explicit release keeps that owner when unmapping fails so cleanup can retry.

#![allow(unsafe_code)]

mod reservation;

pub(crate) use reservation::Reservation;

use super::{invalid_data, process_identity};
use crate::error::error;
use crate::host_storage::{Allocator, Owned};
use crate::os::native_error;
use crate::{Error, ErrorKind};
use std::ffi::{c_int, c_void};
use std::io;
use std::ptr;
use std::sync::atomic::AtomicU32;

unsafe extern "C" {
    fn getpagesize() -> c_int;
    fn madvise(address: *mut c_void, length: usize, advice: c_int) -> c_int;
}

/// Returns the Linux page size after checking it is a nonzero power of two.
pub(crate) fn page_size() -> io::Result<usize> {
    // SAFETY: getpagesize has no pointer arguments or mutable process state.
    let size = unsafe { getpagesize() };
    let size = usize::try_from(size).map_err(invalid_data)?;
    if !size.is_power_of_two() {
        return Err(invalid_data(
            "Linux page size is not a nonzero power of two",
        ));
    }
    Ok(size)
}

/// An unpublished page that can hold a process identity marker across `fork`.
/// The page is unmapped on drop unless publication transfers it to process
/// lifetime ownership.
pub(super) struct ForkMarkerPage {
    address: usize,
    length: usize,
}

impl ForkMarkerPage {
    /// Allocates a private page that Linux will zero in a fork child.
    pub(super) fn new() -> io::Result<Self> {
        let length = page_size()?;
        if length < size_of::<AtomicU32>() {
            return Err(invalid_data("page is too small for a process marker"));
        }
        // SAFETY: This acquires a new private anonymous mapping. No Rust
        // reference is created until the page is published by its owner.
        let mapped = unsafe {
            libc::mmap(
                ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let page = Self {
            address: mapped as usize,
            length,
        };
        if page.address <= 1 {
            return Err(invalid_data(
                "process marker mapping overlaps a state sentinel",
            ));
        }
        // SAFETY: This owner retains the complete live mapping. The advice
        // changes fork inheritance without releasing or aliasing its storage.
        if unsafe { madvise(mapped, length, libc::MADV_WIPEONFORK) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(page)
    }

    /// Returns the unpublished page address for atomic publication.
    pub(super) fn address(&self) -> usize {
        self.address
    }

    /// Transfers the page to process lifetime ownership after publication.
    pub(super) fn retain_until_exit(self) -> usize {
        let address = self.address;
        std::mem::forget(self);
        address
    }
}

impl Drop for ForkMarkerPage {
    fn drop(&mut self) {
        // SAFETY: Only an unpublished page reaches this destructor. It still
        // owns the complete mapping acquired by mmap.
        let _ = unsafe { libc::munmap(self.address as *mut c_void, self.length) };
    }
}

/// Page-aligned CPU storage with retryable mapping destruction.
pub(crate) struct HostAllocation {
    /// `Some` until the mapping is released successfully. A failed release
    /// leaves the same owner here for a later explicit retry.
    mapping: Option<Reservation>,
}

impl HostAllocation {
    /// Allocates ownership metadata before acquiring page-aligned host memory.
    /// `size` must be page-multiple and `alignment` a page-sized or larger
    /// power of two. The retained `process` guards later unmapping.
    pub(crate) fn create(
        size: u64,
        alignment: u64,
        allocator: Allocator,
        process: u32,
    ) -> Result<Owned<Self>, Error> {
        process_identity::prepare_for_hot_checks();
        let size = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host allocation size exceeds address width",
            )
        })?;
        let alignment = usize::try_from(alignment).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host allocation alignment exceeds address width",
            )
        })?;
        let page = page_size().map_err(|e| native_error("host page size", e))?;
        if size == 0
            || size % page != 0
            || size > isize::MAX as usize
            || !alignment.is_power_of_two()
            || alignment < page
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid host allocation extent or alignment",
            ));
        }
        let owner = Owned::<Self>::try_new_uninit(allocator)?;
        let mapping =
            Reservation::new_host_in_process(size, alignment, (0, isize::MAX as u64), process)
                .map_err(|e| native_error("host allocation mmap", e))?;
        Ok(owner.write(Self {
            mapping: Some(mapping),
        }))
    }

    /// Returns the numeric address and size of the live mapped extent.
    /// After successful release, both values are zero.
    pub(crate) fn extent(&self) -> (usize, u64) {
        (
            self.mapping.as_ref().map_or(0, Reservation::address),
            self.mapping.as_ref().map_or(0, |m| m.usable_size() as u64),
        )
    }

    /// Releases the mapping once. Native failure leaves ownership available
    /// for retry; repeated successful calls do nothing.
    pub(crate) fn free(&mut self) -> Result<(), Error> {
        if let Some(mapping) = self.mapping.as_mut() {
            mapping
                .release()
                .map_err(|e| native_error("host allocation munmap", e))?;
            self.mapping = None;
        }
        Ok(())
    }
}

/// Returns the validated host page size without opening a device driver.
pub(crate) fn host_page_size() -> Result<u64, Error> {
    page_size()
        .map(|size| size as u64)
        .map_err(|source| native_error("host page size", source))
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        if self.free().is_err() {
            // The mapping may still be live. Keep its owner from running a
            // second, unchecked unmap during this destructor.
            if let Some(mapping) = self.mapping.take() {
                std::mem::forget(mapping);
            }
        }
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::memory::{host_cache_control, host_cache_line_size, host_page_size};
    use crate::os::linux::errno;
    use crate::session::{DriverContextLifetime, Session};
    use std::sync::atomic::Ordering;
    #[test]
    fn cpu_storage_supports_both_lifetimes_and_retains_custom_allocator() {
        for policy in [
            DriverContextLifetime::Session,
            DriverContextLifetime::Process,
        ] {
            let callbacks = crate::test_support::allocator::State::default();
            // SAFETY: State remains stationary until all native owners drop.
            let allocator = unsafe { callbacks.allocator() };
            let count = crate::test_support::allocation_counter::allocations(|| {
                let mut session = Session::with_allocator(policy, allocator).unwrap();
                let page = host_page_size().unwrap();
                let size = page * 2;
                let alignment = page * 16;
                let mut allocation = session.allocate_host(size, alignment).unwrap();
                assert_eq!(allocation.info().host_address as u64 % alignment, 0);
                assert_eq!(allocation.info().size, size);
                if let Ok(line) = host_cache_line_size() {
                    // SAFETY: The newly allocated extent remains mapped.
                    unsafe {
                        host_cache_control(allocation.info().host_address, size, line).unwrap();
                    }
                }
                allocation.free().unwrap();
                allocation.free().unwrap();
                drop(allocation);
                session.destroy().unwrap();
                assert!(session.allocate_host(page, page).is_err());
            });
            assert_eq!(count, 0);
            let allocations = callbacks.allocations.load(Ordering::Relaxed);
            assert!(allocations >= 2);
            assert_eq!(callbacks.frees.load(Ordering::Relaxed), allocations);
        }
    }
    #[test]
    fn failed_host_free_preserves_the_mapping_and_retries_only_remaining_work() {
        let page = host_page_size().unwrap();
        let mut allocation =
            HostAllocation::create(page, page, Allocator::default(), std::process::id()).unwrap();
        let extent = allocation.extent();
        allocation
            .mapping
            .as_mut()
            .unwrap()
            .fail_release_once(errno::EIO);
        assert_eq!(
            allocation.free().unwrap_err().native_error_code(),
            Some(errno::EIO)
        );
        assert_eq!(allocation.extent(), extent);
        allocation.free().unwrap();
        assert!(allocation.mapping.is_none());
    }
}
