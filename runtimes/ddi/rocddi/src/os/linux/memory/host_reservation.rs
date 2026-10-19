// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned Linux host address ranges and explicit page-table replacement.

#![allow(unsafe_code)]

use super::{HostAllocation, Reservation, page_size};
use crate::os::linux::process_identity::check_process;
use libc::{MAP_ANONYMOUS, MAP_FIXED, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE};
use std::ffi::c_void;
use std::io;
use std::os::fd::AsRawFd;

pub(crate) struct HostReservation {
    range: Reservation,
    process: u32,
}

impl HostReservation {
    pub(crate) fn create(size: usize, alignment: usize, process: u32) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0
            || size % page != 0
            || size > isize::MAX as usize
            || !alignment.is_power_of_two()
            || alignment < page
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let range = Reservation::new(size, alignment, (page as u64, usize::MAX as u64), false)?;
        Ok(Self { range, process })
    }

    pub(crate) fn extent(&self) -> (usize, usize) {
        (self.range.address(), self.range.usable_size())
    }

    fn target(&self, offset: u64, length: u64) -> io::Result<(usize, usize)> {
        check_process(self.process)?;
        if !self.range.is_live() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let page = page_size()? as u64;
        let size = self.range.usable_size() as u64;
        if length == 0
            || offset % page != 0
            || length % page != 0
            || offset > size
            || length > size - offset
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let offset =
            usize::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let length =
            usize::try_from(length).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        Ok((self.range.address() + offset, length))
    }

    pub(crate) fn replace(
        &mut self,
        target_offset: u64,
        length: u64,
        source: &HostAllocation,
        source_offset: u64,
        writable: bool,
    ) -> io::Result<()> {
        let (address, length) = self.target(target_offset, length)?;
        let (source_address, source_length) = source.extent();
        let page = page_size()? as u64;
        if source_address == 0
            || source_offset % page != 0
            || source_offset > source_length
            || length as u64 > source_length - source_offset
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let file = source
            .source()
            .ok_or_else(|| io::Error::from(io::ErrorKind::Unsupported))?;
        let offset = i64::try_from(source_offset)
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let protection = PROT_READ | if writable { PROT_WRITE } else { 0 };
        // SAFETY: target is a checked subrange of this exclusively borrowed
        // reservation. The file remains live for the synchronous native call.
        let mapped = unsafe {
            libc::mmap(
                address as *mut c_void,
                length,
                protection,
                MAP_SHARED | MAP_FIXED,
                file.as_raw_fd(),
                offset,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self, offset: u64, length: u64) -> io::Result<()> {
        let (address, length) = self.target(offset, length)?;
        // SAFETY: This restores an inaccessible mapping inside our owned range.
        let mapped = unsafe {
            libc::mmap(
                address as *mut c_void,
                length,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    pub(crate) fn free(&mut self) -> io::Result<()> {
        self.range.release()
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn released_reservation_cannot_replace_a_new_mapping() {
        let page = page_size().unwrap();
        let mut reservation = HostReservation::create(page, page, std::process::id()).unwrap();
        let source =
            HostAllocation::create_remappable(page as u64, page as u64, std::process::id())
                .unwrap();
        let (address, _) = reservation.extent();
        reservation.free().unwrap();

        // Reoccupy the former address to prove a stale MAP_FIXED would replace
        // another owner's mapping. MAP_FIXED_NOREPLACE does not evict it.
        let occupied = unsafe {
            libc::mmap(
                address as *mut c_void,
                page,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        if occupied == libc::MAP_FAILED {
            // Another mapping won the address. The liveness checks remain
            // valid without observing the replacement contents.
            assert_eq!(
                reservation.clear(0, page as u64).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
            assert_eq!(
                reservation
                    .replace(0, page as u64, &source, 0, true)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
            return;
        }
        unsafe { occupied.cast::<u8>().write_volatile(0xa5) };
        assert_eq!(
            reservation.clear(0, page as u64).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            reservation
                .replace(0, page as u64, &source, 0, true)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(unsafe { occupied.cast::<u8>().read_volatile() }, 0xa5);
        assert_eq!(unsafe { libc::munmap(occupied, page) }, 0);
    }

    #[test]
    fn failed_release_keeps_the_range_eligible_for_retry_and_clear() {
        let page = page_size().unwrap();
        let mut reservation = HostReservation::create(page, page, std::process::id()).unwrap();
        reservation.range.fail_release_once(libc::EIO);
        assert!(reservation.free().is_err());
        reservation.clear(0, page as u64).unwrap();
        reservation.free().unwrap();
        assert_eq!(
            reservation.clear(0, page as u64).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
