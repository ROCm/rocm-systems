// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Shared Linux host utilities used by the KFD backend.
//!
//! These wrappers centralize process-identity checks, descriptor duplication and
//! closure, page-size discovery, DMA-BUF identity, and the small cache-control
//! operations required by qualified host mappings. They intentionally expose
//! `io::Result` so resource owners decide whether an error is retryable,
//! terminal, or leaves a native outcome uncertain.

#![allow(unsafe_code)]

use std::ffi::c_int;
use std::fs::File;
use std::io;
use std::os::fd::{FromRawFd, IntoRawFd};
use std::os::unix::fs::{FileExt, MetadataExt};

unsafe extern "C" {
    fn close(fd: c_int) -> c_int;
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    fn getpagesize() -> c_int;
}

const F_DUPFD_CLOEXEC: c_int = 1030;

fn invalid_data(_detail: impl std::fmt::Display) -> io::Error {
    io::Error::from(io::ErrorKind::InvalidData)
}

pub(super) fn check_process(process: u32) -> io::Result<()> {
    crate::driver::process_identity::check_process(process)
}

pub(in crate::driver) fn page_size() -> io::Result<usize> {
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

/// Stable size and filesystem identity obtained from one DMA-BUF descriptor.
pub(super) struct DmaBufFileInfo {
    pub(super) size: u64,
    pub(super) physical_id: [u64; 2],
}

pub(super) fn duplicate_file(descriptor: i32) -> io::Result<File> {
    if descriptor < 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    // SAFETY: fcntl borrows the caller descriptor and returns a new descriptor
    // owned by the caller on success. The zero third argument is the lower bound.
    let duplicate = unsafe { fcntl(descriptor, F_DUPFD_CLOEXEC, 0) };
    if duplicate < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fcntl returned a new descriptor that this File now owns.
    Ok(unsafe { File::from_raw_fd(duplicate) })
}

/// Gets the file length from an owned duplicate of a borrowed descriptor.
pub(super) fn descriptor_length(descriptor: i32) -> io::Result<u64> {
    duplicate_file(descriptor)?
        .metadata()
        .map(|metadata| metadata.len())
}

/// Reads an exact range without closing or moving the caller's descriptor.
pub(super) fn read_descriptor_exact_at(
    descriptor: i32,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    duplicate_file(descriptor)?.read_exact_at(buffer, offset)
}

/// Consumes one owned descriptor even when the underlying close reports an error.
pub(super) fn close_descriptor(descriptor: i32) -> io::Result<()> {
    // SAFETY: The caller transfers ownership of this descriptor for one close.
    if unsafe { close(descriptor) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Reads at a fixed file offset without changing the shared file position.
pub(super) fn read_descriptor_at(
    descriptor: i32,
    buffer: &mut [u8],
    offset: i64,
) -> io::Result<usize> {
    if descriptor < 0 {
        return Err(io::Error::from_raw_os_error(9));
    }
    let offset = u64::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    duplicate_file(descriptor)?.read_at(buffer, offset)
}

/// Writes at a fixed file offset without changing the shared file position.
pub(super) fn write_descriptor_at(
    descriptor: i32,
    buffer: &[u8],
    offset: i64,
) -> io::Result<usize> {
    if descriptor < 0 {
        return Err(io::Error::from_raw_os_error(9));
    }
    let offset = u64::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    duplicate_file(descriptor)?.write_at(buffer, offset)
}

pub(super) fn dma_buf_file_info(file: &File) -> io::Result<DmaBufFileInfo> {
    let metadata = file.metadata()?;
    let size = metadata.len();
    let physical_id = [metadata.dev(), metadata.ino()];
    if size == 0 || physical_id == [0, 0] {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(DmaBufFileInfo { size, physical_id })
}

pub(in crate::driver) fn host_cache_line_size() -> io::Result<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: CPUID leaf one exists on every x86-64 processor and does not
        // read caller memory. CLFLUSH is used only when its feature bit is set.
        #[allow(
            unused_unsafe,
            reason = "CPUID was unsafe on the supported Rust 1.85 toolchain"
        )]
        let leaf = unsafe { std::arch::x86_64::__cpuid(1) };
        let bytes = ((leaf.ebx >> 8) & 255) * 8;
        if leaf.edx & (1 << 19) == 0 || !bytes.is_power_of_two() {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        Ok(bytes)
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

pub(in crate::driver) unsafe fn host_cache_control(
    pointer: usize,
    length: u64,
    line_size: u32,
) -> io::Result<()> {
    let qualified_line_size = host_cache_line_size()?;
    if (line_size != 0 && line_size != qualified_line_size) || pointer == 0 || length == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let length =
        usize::try_from(length).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let end = pointer
        .checked_add(length - 1)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    // A zero line size selects the write-combined mapping recipe. No cache
    // line is touched, but prior buffered writes must drain before return.
    if line_size == 0 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_mfence();
        };
        return Ok(());
    }
    let mask = line_size as usize - 1;
    let line = pointer & !mask;
    let last = end & !mask;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: The core caller guarantees every intersecting cache line
        // stays mapped. CLFLUSH support and exact line size were checked above.
        // MFENCE brackets completion so the cache operation orders both prior
        // writes and subsequent memory accesses in this calling CPU thread.
        unsafe { std::arch::x86_64::_mm_mfence() };
        let mut line = line;
        loop {
            unsafe { std::arch::x86_64::_mm_clflush(line as *const u8) };
            if line == last {
                break;
            }
            line += line_size as usize;
        }
        unsafe { std::arch::x86_64::_mm_mfence() };
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (line, last);
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

/// Linux consumes the descriptor even when close reports an error; remove it
/// from the owner before calling so retries never target a recycled descriptor.
pub(super) fn close_file(file: &mut Option<File>) -> io::Result<()> {
    let Some(file) = file.take() else {
        return Ok(());
    };
    close_descriptor(file.into_raw_fd())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn descriptor_snapshot_preserves_caller_ownership_and_file_position() {
        let path = std::env::temp_dir().join(format!(
            "rocddi-descriptor-test-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(b"abcdef").unwrap();
        file.seek(SeekFrom::Start(2)).unwrap();
        let descriptor = file.as_raw_fd();

        assert_eq!(descriptor_length(descriptor).unwrap(), 6);
        let mut bytes = [0; 3];
        read_descriptor_exact_at(descriptor, &mut bytes, 1).unwrap();
        assert_eq!(&bytes, b"bcd");
        assert_eq!(file.stream_position().unwrap(), 2);
        file.read_exact(&mut bytes[..1]).unwrap();
        assert_eq!(bytes[0], b'c');

        assert_eq!(
            read_descriptor_exact_at(descriptor, &mut bytes, 5)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert!(descriptor_length(-1).is_err());
        assert!(read_descriptor_exact_at(-1, &mut bytes, 0).is_err());
    }
}
