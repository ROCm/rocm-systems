// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned Linux virtual-address reservations and file mappings.
//!
//! Owners retain the exact mapping and process identity through explicit
//! release. Native GPU users must stop before their backing is unmapped.

#![allow(unsafe_code)]

use super::{madvise, page_size};
use crate::os::linux::{invalid_data, process_identity::check_process};
use libc::{
    MADV_DONTFORK, MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE,
    PROT_READ, PROT_WRITE, mmap, munmap,
};
use std::ffi::{c_int, c_void};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::ptr;
use std::sync::atomic::AtomicU64;

const PROT_READ_WRITE: c_int = PROT_READ | PROT_WRITE;
// libc does not expose this Linux flag for the supported GNU targets.
const MAP_FIXED_NOREPLACE: c_int = 0x10_0000;

/// Owns the entire reservation, including alignment padding. Keeping padding
/// avoids fallible partial unmaps during construction and leaves one range to
/// release after all native users stop accessing its usable extent.
pub(crate) struct Reservation {
    /// Base address to unmap or restore when this record owns a mapping.
    base: usize,
    /// Bytes owned from `base`; zero after release or before a view is mapped.
    length: usize,
    /// Aligned start of the usable range exposed to callers.
    address: usize,
    /// Usable bytes from `address`, excluding any alignment padding.
    size: usize,
    /// Acquiring process; an inherited owner must not unmap its parent's range.
    process: u32,
    /// Whether CPU writes to the current usable mapping are permitted.
    writable: bool,
    /// Restores inaccessible memory inside a retained parent on release.
    restore_on_release: bool,
    /// One injected unmapping failure used to verify retryable ownership.
    #[cfg(test)]
    release_error: Option<i32>,
}

impl Reservation {
    /// Reserves an anonymous aligned range within inclusive `bounds`.
    /// A host range is writable; a device address reservation is inaccessible.
    /// Ownership includes any alignment padding until release.
    pub(crate) fn new(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        host: bool,
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, host, false, std::process::id())
    }

    /// Captures a session process ID that was checked before host allocation.
    /// The retained owner checks that identity again before release.
    pub(crate) fn new_host_in_process(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        process: u32,
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, true, false, process)
    }

    /// Attempts the requested address without replacing an existing mapping.
    /// An unavailable or invalid hint falls back to any aligned range within
    /// the same inclusive bounds.
    pub(crate) fn new_at(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        address: usize,
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        if address != 0
            && address % alignment == 0
            && address as u64 >= bounds.0
            && address
                .checked_add(size - 1)
                .is_some_and(|last| last as u64 <= bounds.1)
        {
            // SAFETY: MAP_FIXED_NOREPLACE either acquires this exact free range
            // or fails without replacing an existing mapping.
            let mapped = unsafe {
                mmap(
                    address as *mut c_void,
                    size,
                    PROT_NONE,
                    MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE | MAP_FIXED_NOREPLACE,
                    -1,
                    0,
                )
            };
            if mapped as usize == address {
                return Ok(Self {
                    base: address,
                    length: size,
                    address,
                    size,
                    process: std::process::id(),
                    writable: false,
                    restore_on_release: false,
                    #[cfg(test)]
                    release_error: None,
                });
            }
            if mapped != libc::MAP_FAILED {
                // SAFETY: A nonfailed, unexpected mapping is owned by this
                // call and must not escape while the requested hint falls back.
                let _ = unsafe { munmap(mapped, size) };
            }
        }
        Self::new(size, alignment, bounds, false)
    }

    /// Reserves shared, writable anonymous host pages within `bounds`.
    pub(crate) fn new_shared_host(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, true, true, std::process::id())
    }

    /// Allocates writable host pages and trims alignment padding before
    /// publication, leaving only the usable extent owned by this record.
    pub(crate) fn new_aligned_host(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let length = size
            .checked_add(alignment - page)
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: This creates a new writable anonymous mapping without
        // replacing existing memory. Its padding is trimmed below.
        let mapped = unsafe {
            mmap(
                ptr::null_mut(),
                length,
                PROT_READ_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let base = mapped as usize;
        let Some(address) = base
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
        else {
            // SAFETY: The complete mapping is still owned by this call.
            let _ = unsafe { munmap(mapped, length) };
            return Err(invalid_data(
                "mmap alignment overflows the host address width",
            ));
        };
        let last = address.checked_add(size - 1);
        if (address as u64) < bounds.0 || last.is_none_or(|last| last as u64 > bounds.1) {
            // SAFETY: The complete mapping is still owned by this call.
            let _ = unsafe { munmap(mapped, length) };
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        let prefix = address - base;
        if prefix != 0 {
            // SAFETY: This removes only the leading alignment padding.
            if unsafe { munmap(mapped, prefix) } != 0 {
                let source = io::Error::last_os_error();
                // SAFETY: The complete mapping remains live after failed munmap.
                let _ = unsafe { munmap(mapped, length) };
                return Err(source);
            }
        }
        let end = address + size;
        let mapping_end = base + length;
        if end < mapping_end {
            // SAFETY: This removes only the trailing alignment padding.
            if unsafe { munmap(end as *mut c_void, mapping_end - end) } != 0 {
                let source = io::Error::last_os_error();
                // SAFETY: The aligned extent and trailing padding remain live;
                // the prefix was already released.
                let _ = unsafe { munmap(address as *mut c_void, mapping_end - address) };
                return Err(source);
            }
        }
        Ok(Self {
            base: address,
            length: size,
            address,
            size,
            process: std::process::id(),
            writable: true,
            restore_on_release: false,
            #[cfg(test)]
            release_error: None,
        })
    }

    /// Common anonymous mapping constructor. It retains alignment padding so
    /// a later release can unmap one complete range.
    fn new_with_sharing(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        host: bool,
        shared: bool,
        process: u32,
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let length = size
            .checked_add(alignment - page)
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let protection = if host { PROT_READ_WRITE } else { PROT_NONE };
        let flags = MAP_ANONYMOUS
            | if shared { MAP_SHARED } else { MAP_PRIVATE }
            | if host { 0 } else { MAP_NORESERVE };
        // SAFETY: This creates a new anonymous mapping without replacing any
        // existing mapping. Length is nonzero, page aligned, and representable;
        // no Rust references are created to the returned memory.
        let mapped = unsafe { mmap(ptr::null_mut(), length, protection, flags, -1, 0) };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let base = mapped as usize;
        let address = base
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1));
        let mut reservation = Self {
            base,
            length,
            address: base,
            size,
            process,
            writable: host,
            restore_on_release: false,
            #[cfg(test)]
            release_error: None,
        };
        let Some(address) = address else {
            return Err(invalid_data(
                "mmap alignment overflows the host address width",
            ));
        };
        reservation.address = address;
        let last = address.checked_add(size - 1);
        if (address as u64) < bounds.0 || last.is_none_or(|last| last as u64 > bounds.1) {
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        Ok(reservation)
    }

    /// Creates a view into a reservation retained by another owner. If a file
    /// mapping replaces this view, releasing it restores anonymous reserved VA
    /// instead of exposing a hole inside the parent range.
    pub(crate) fn view(address: usize, size: usize) -> Self {
        debug_assert_ne!(address, 0);
        debug_assert_ne!(size, 0);
        Self {
            base: address,
            length: 0,
            address,
            size,
            process: std::process::id(),
            writable: false,
            restore_on_release: true,
            #[cfg(test)]
            release_error: None,
        }
    }

    /// Returns the configured usable size; callers must separately retain the
    /// reservation owner to rely on that extent remaining mapped.
    pub(crate) fn usable_size(&self) -> usize {
        self.size
    }

    /// Returns the configured usable address, which is not a liveness check.
    pub(crate) fn address(&self) -> usize {
        self.address
    }

    /// Prevents a live mapping from being inherited across `fork`.
    pub(crate) fn dont_fork(&self) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 {
            return Err(invalid_data("address reservation is not live"));
        }
        // SAFETY: The live owned mapping covers this exact usable extent.
        // MADV_DONTFORK changes inheritance policy without creating aliases.
        let result = unsafe { madvise(self.address as *mut c_void, self.size, MADV_DONTFORK) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Map the BO over this reservation's usable extent without changing its
    /// address. Keeping the surrounding reservation prevents a failed partial
    /// cleanup from making a still-referenced GPU address available for reuse.
    pub(crate) fn map_render(&mut self, render: &File, offset: u64) -> io::Result<()> {
        check_process(self.process)?;
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(render, offset, PROT_READ_WRITE)
    }

    /// Installs a render-node BO mapping without granting CPU access. KFD
    /// scratch allocations require this VMA even though userspace never reads
    /// or writes the backing through the CPU.
    pub(crate) fn map_render_inaccessible(&mut self, render: &File, offset: u64) -> io::Result<()> {
        check_process(self.process)?;
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(render, offset, PROT_NONE)
    }

    /// Replaces the usable range with a writable DMA-BUF view at offset zero.
    pub(crate) fn map_dma_buf(&mut self, dma_buf: &File) -> io::Result<()> {
        self.map_file(dma_buf, 0, PROT_READ_WRITE)
    }

    /// Maps a DMA-BUF using permission bits 0 through 3 for none, read, write,
    /// or read-write CPU access. Other bits are rejected before mapping.
    pub(crate) fn map_dma_buf_with_permissions(
        &mut self,
        dma_buf: &File,
        offset: u64,
        permission_bits: u32,
    ) -> io::Result<()> {
        let protection = match permission_bits {
            0 => PROT_NONE,
            1 => PROT_READ,
            2 => PROT_WRITE,
            3 => PROT_READ_WRITE,
            _ => return Err(io::Error::from(io::ErrorKind::InvalidInput)),
        };
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(dma_buf, offset, protection)
    }

    /// Replaces the owned range with a shared writable file mapping.
    /// The signed offset preserves native tokens whose high bits are set.
    pub(crate) fn map_file_read_write(&mut self, file: &File, offset: i64) -> io::Result<()> {
        self.map_file(file, offset, PROT_READ_WRITE)
    }

    /// Replaces only the usable range with a file mapping. A view into a parent
    /// becomes the owner of that replacement and later restores inaccessible
    /// address space instead of removing the parent's range.
    fn map_file(&mut self, file: &File, offset: i64, protection: c_int) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 && !self.restore_on_release {
            return Err(invalid_data("file mapping has no live address reservation"));
        }
        if offset % i64::try_from(page_size()?).map_err(invalid_data)? != 0 {
            return Err(invalid_data("file mapping offset is not page aligned"));
        }
        // SAFETY: MAP_FIXED replaces only the aligned usable extent of our own
        // reservation. The original range is still owned and no Rust references
        // alias it. The caller keeps the allocation and render file alive.
        let mapped = unsafe {
            mmap(
                self.address as *mut c_void,
                self.size,
                protection,
                MAP_SHARED | MAP_FIXED,
                file.as_raw_fd(),
                offset,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        if self.restore_on_release {
            self.base = self.address;
            self.length = self.size;
        }
        // SAFETY: The successful mapping covers the exact live usable extent.
        // Avoid inheriting device mappings into a fork child that cannot own
        // the corresponding KFD process resources.
        let _ = unsafe { madvise(mapped, self.size, MADV_DONTFORK) };
        self.writable = protection == PROT_READ_WRITE;
        Ok(())
    }

    /// Returns one aligned atomic word within live writable mapping storage.
    /// The owner must keep the mapping and any external writer live while the
    /// returned reference is used. The caller chooses the publication order.
    pub(crate) fn atomic_u64_at(&self, offset: usize) -> io::Result<&AtomicU64> {
        self.check_write(offset, size_of::<AtomicU64>())?;
        let address = self.address + offset;
        if address % align_of::<AtomicU64>() != 0 {
            return Err(invalid_data("unaligned atomic word in mapped storage"));
        }
        // SAFETY: The checked live mapping contains an initialized, aligned
        // atomic-width word, and no ordinary Rust reference aliases it.
        Ok(unsafe { &*(address as *const AtomicU64) })
    }

    /// Zeros the complete writable usable range before it is published.
    pub(crate) fn zero(&mut self) -> io::Result<()> {
        self.check_write(0, self.size)?;
        // SAFETY: The checked live mapping covers exactly size writable bytes.
        // Queue construction has not published it and creates no Rust aliases.
        unsafe { ptr::write_bytes(self.address as *mut u8, 0, self.size) };
        Ok(())
    }

    /// Copies bytes into a checked subrange of the writable mapping.
    pub(crate) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> io::Result<()> {
        self.check_write(offset, bytes.len())?;
        // SAFETY: Both extents are valid and disjoint: input is an ordinary
        // Rust slice, while this private mapping exposes no Rust references.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.address + offset) as *mut u8,
                bytes.len(),
            );
        };
        Ok(())
    }

    /// Initializes `count` consecutive records after validating their total
    /// extent against the writable mapping.
    pub(crate) fn fill_records(&mut self, record: &[u8], count: usize) -> io::Result<()> {
        let size = record
            .len()
            .checked_mul(count)
            .ok_or_else(|| invalid_data("record initialization overflows"))?;
        self.check_write(0, size)?;
        for index in 0..count {
            // SAFETY: The entire destination was checked once above, and each
            // disjoint record occupies its own part of that private mapping.
            unsafe {
                ptr::copy_nonoverlapping(
                    record.as_ptr(),
                    (self.address + index * record.len()) as *mut u8,
                    record.len(),
                );
            };
        }
        Ok(())
    }

    /// Checks process identity, mapping liveness, CPU write access, and range.
    fn check_write(&self, offset: usize, length: usize) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0
            || !self.writable
            || offset.checked_add(length).is_none_or(|end| end > self.size)
        {
            return Err(invalid_data("write exceeds live CPU-visible reservation"));
        }
        Ok(())
    }

    /// Unmaps an owned range or restores an inaccessible parent view. A native
    /// failure retains the live ownership record for explicit retry; release
    /// after success is harmless.
    pub(crate) fn release(&mut self) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(errno) = self.release_error.take() {
            return Err(io::Error::from_raw_os_error(errno));
        }
        if self.restore_on_release {
            // SAFETY: This exact subrange replaced part of a retained parent
            // reservation. Native GPU cleanup is complete, so replace the BO
            // mapping with inaccessible anonymous VA for future scratch reuse.
            let mapped = unsafe {
                mmap(
                    self.base as *mut c_void,
                    self.length,
                    PROT_NONE,
                    MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE | MAP_FIXED,
                    -1,
                    0,
                )
            };
            if mapped == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
        } else {
            // SAFETY: This owner holds the entire live, page-aligned range.
            // Native GPU cleanup must have completed before release.
            let result = unsafe { munmap(self.base as *mut c_void, self.length) };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        self.length = 0;
        Ok(())
    }

    /// Injects one release error without dropping the reservation.
    #[cfg(test)]
    pub(crate) fn fail_release_once(&mut self, errno: i32) {
        self.release_error = Some(errno);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // The resource owner forgets this reservation if GPU cleanup fails.
        // Reaching this destructor means only CPU address ownership remains.
        if self.length != 0 || self.restore_on_release {
            let _ = self.release();
        }
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn foreign_process_rejects_release_and_keeps_the_reservation_owned() {
        let page = page_size().unwrap();
        let mut reservation = Reservation::new(page, page * 4, (0, u64::MAX), false).unwrap();
        assert_eq!(reservation.address() % (page * 4), 0);
        let length = reservation.length;
        reservation.process = std::process::id().wrapping_add(1);
        assert_eq!(
            reservation.release().unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
        assert_eq!(reservation.length, length);
        reservation.process = std::process::id();
        reservation.fail_release_once(12);
        assert!(reservation.release().is_err());
        assert_eq!(reservation.length, length);
        reservation.release().unwrap();
        assert_eq!(reservation.length, 0);
        reservation.release().unwrap();
    }

    #[test]
    fn render_mapping_replaces_only_the_owned_usable_extent() {
        use std::os::unix::fs::FileExt;
        let page = page_size().unwrap();
        let path = std::env::temp_dir().join(format!("rocddi-render-map-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        file.set_len(page as u64).unwrap();
        file.write_all_at(&[0x5a], 0).unwrap();
        let mut reservation = Reservation::new(page, page * 4, (0, u64::MAX), false).unwrap();
        let base = reservation.base;
        let length = reservation.length;
        reservation.map_render(&file, 0).unwrap();
        assert_eq!(reservation.base, base);
        assert_eq!(reservation.length, length);
        assert!(reservation.address >= base && reservation.address + page <= base + length);
        // SAFETY: The live writable file mapping covers this byte and no other
        // thread uses the private file. No Rust reference aliases the mapped bytes.
        unsafe {
            let byte = reservation.address as *mut u8;
            assert_eq!(byte.read_volatile(), 0x5a);
            byte.write_volatile(0xa5);
        }
        reservation.release().unwrap();
        let mut bytes = [0];
        file.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(bytes, [0xa5]);
    }

    #[test]
    fn inaccessible_view_restores_its_parent_reservation() {
        let page = page_size().unwrap();
        let path = std::env::temp_dir().join(format!(
            "rocddi-inaccessible-render-map-{}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        file.set_len(page as u64).unwrap();
        let parent = Reservation::new(page * 2, page, (0, u64::MAX), false).unwrap();
        let mut view = Reservation::view(parent.address(), page);

        view.map_render_inaccessible(&file, 0).unwrap();
        assert_eq!(view.length, page);
        assert!(!view.writable);
        view.release().unwrap();
        assert_eq!(view.length, 0);

        drop(view);
        drop(parent);
    }

    #[test]
    fn requested_virtual_address_falls_back_without_replacing_live_memory() {
        let page = page_size().unwrap();
        let occupied = Reservation::new(page, page, (0, u64::MAX), false).unwrap();
        let requested = occupied.address();
        let fallback = Reservation::new_at(page, page, (0, u64::MAX), requested).unwrap();

        assert_ne!(fallback.address(), requested);
        assert_eq!(occupied.address(), requested);
    }

    #[test]
    fn virtual_host_mapping_rejects_unknown_permission_bits() {
        let page = page_size().unwrap();
        let parent = Reservation::new(page, page, (0, u64::MAX), false).unwrap();
        let file = File::open("/dev/zero").unwrap();
        let mut view = Reservation::view(parent.address(), page);

        assert_eq!(
            view.map_dma_buf_with_permissions(&file, 0, 4)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
