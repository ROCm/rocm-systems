// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux raw descriptor ownership, positioned I/O, and DMA-BUF file identity.
//!
//! These operations adapt borrowed C descriptors to owned Rust [`File`] values.
//! Resource owners decide whether a native failure is retryable or leaves an
//! uncertain result.

#![allow(unsafe_code)]

use super::errno;
use libc::close;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::fs::{FileExt, MetadataExt};

/// Stable size and filesystem identity obtained from one DMA-BUF descriptor.
pub(crate) struct DmaBufFileInfo {
    /// Length reported by the descriptor's filesystem metadata.
    pub(crate) size: u64,
    /// Device and inode numbers used to recognize the same underlying file.
    pub(crate) physical_id: [u64; 2],
}

/// Duplicates a borrowed descriptor with close-on-exec set. The returned
/// [`File`] owns only the new descriptor; the caller retains its original one.
pub(crate) fn duplicate_file(descriptor: i32) -> io::Result<File> {
    if descriptor < 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    // SAFETY: fcntl validates this raw integer in the kernel. A stale or
    // closed descriptor reports EBADF without creating a Rust BorrowedFd.
    let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 0) };
    if duplicate < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: A successful F_DUPFD_CLOEXEC returns a new owned descriptor.
    Ok(unsafe { File::from_raw_fd(duplicate) })
}

/// Consumes one owned descriptor even when the underlying close reports an error.
pub(crate) fn close_descriptor(descriptor: i32) -> io::Result<()> {
    // SAFETY: The caller transfers ownership of this descriptor for one close.
    if unsafe { close(descriptor) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Reads at a fixed file offset without changing the shared file position.
pub(crate) fn read_descriptor_at(
    descriptor: i32,
    buffer: &mut [u8],
    offset: i64,
) -> io::Result<usize> {
    if descriptor < 0 {
        return Err(io::Error::from_raw_os_error(errno::EBADF));
    }
    let offset = u64::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    read_at(&duplicate_file(descriptor)?, buffer, offset)
}

/// Writes at a fixed file offset without changing the shared file position.
pub(crate) fn write_descriptor_at(
    descriptor: i32,
    buffer: &[u8],
    offset: i64,
) -> io::Result<usize> {
    if descriptor < 0 {
        return Err(io::Error::from_raw_os_error(errno::EBADF));
    }
    let offset = u64::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    write_at(&duplicate_file(descriptor)?, buffer, offset)
}

pub(crate) fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<usize> {
    file.read_at(bytes, offset)
}

pub(crate) fn write_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<usize> {
    // Linux pwrite ignores the offset on append-mode open-file descriptions.
    // SAFETY: File keeps the descriptor live; F_GETFL only reads status flags.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::O_APPEND != 0 {
        return Err(io::Error::from_raw_os_error(errno::EINVAL));
    }
    file.write_at(bytes, offset)
}

pub(crate) fn path(file: &File) -> io::Result<std::path::PathBuf> {
    let path = std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
    if !path.is_absolute() {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    let source = file.metadata()?;
    let target = std::fs::metadata(&path)?;
    if (source.dev(), source.ino()) != (target.dev(), target.ino()) {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(path)
}

/// Reads a DMA-BUF's nonzero size and filesystem identity from one owner.
pub(crate) fn dma_buf_file_info(file: &File) -> io::Result<DmaBufFileInfo> {
    let metadata = file.metadata()?;
    let size = metadata.len();
    let physical_id = [metadata.dev(), metadata.ino()];
    if size == 0 || physical_id == [0, 0] {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(DmaBufFileInfo { size, physical_id })
}

/// Linux consumes the descriptor even when close reports an error; remove it
/// from the owner before calling so retries never target a recycled descriptor.
pub(crate) fn close_file(file: &mut Option<File>) -> io::Result<()> {
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
    fn positioned_writes_reject_append_mode_without_changing_shared_flags() {
        let path = std::env::temp_dir().join(format!(
            "rocddi-append-test-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut caller = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        caller.write_all(b"abc").unwrap();
        caller.seek(SeekFrom::Start(1)).unwrap();
        let retained = duplicate_file(caller.as_raw_fd()).unwrap();
        // SAFETY: The caller retains the descriptor throughout this flag query.
        let original_flags = unsafe { libc::fcntl(caller.as_raw_fd(), libc::F_GETFL) };
        assert!(original_flags >= 0);

        let error = crate::host::file::write_at(&retained, b"X", 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(error.raw_os_error(), Some(errno::EINVAL));
        let result = crate::memory::interop::linux::ais_host_transfer(
            caller.as_raw_fd(),
            crate::memory::interop::linux::AisHostBuffer::Write(b"X"),
            0,
        );
        assert_eq!(result.status, -errno::EINVAL);
        assert_eq!(result.size_copied, 0);

        assert_eq!(caller.stream_position().unwrap(), 1);
        assert_eq!(caller.metadata().unwrap().len(), 3);
        let mut bytes = [0; 4];
        crate::host::file::read_exact_at(&retained, &mut bytes[..3], 0).unwrap();
        assert_eq!(&bytes[..3], b"abc");
        for file in [&caller, &retained] {
            // SAFETY: Each File retains its descriptor throughout the query.
            assert_eq!(
                unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) },
                original_flags
            );
        }

        caller.write_all(b"Y").unwrap();
        crate::host::file::read_exact_at(&retained, &mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"abcY");
    }

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

        let snapshot = duplicate_file(descriptor).unwrap();
        assert_eq!(snapshot.metadata().unwrap().len(), 6);
        let mut bytes = [0; 3];
        crate::host::file::read_exact_at(&snapshot, &mut bytes, 1).unwrap();
        assert_eq!(&bytes, b"bcd");
        assert_eq!(file.stream_position().unwrap(), 2);
        assert_eq!(crate::host::file::write_at(&snapshot, b"Z", 5).unwrap(), 1);
        assert_eq!(file.stream_position().unwrap(), 2);
        file.read_exact(&mut bytes[..1]).unwrap();
        assert_eq!(bytes[0], b'c');

        assert_eq!(
            crate::host::file::read_exact_at(&snapshot, &mut bytes, 5)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert!(duplicate_file(-1).is_err());
    }
}
