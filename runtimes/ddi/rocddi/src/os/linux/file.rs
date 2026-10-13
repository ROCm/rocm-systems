// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux raw descriptor ownership, positioned I/O, and DMA-BUF file identity.
//!
//! These operations adapt borrowed C descriptors to owned Rust [`File`] values.
//! Resource owners decide whether a native failure is retryable or leaves an
//! uncertain result.

#![allow(unsafe_code)]

use super::errno;
use libc::{F_DUPFD_CLOEXEC, close, fcntl};
use std::fs::File;
use std::io;
use std::os::fd::{FromRawFd, IntoRawFd};
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
pub(crate) fn descriptor_length(descriptor: i32) -> io::Result<u64> {
    duplicate_file(descriptor)?
        .metadata()
        .map(|metadata| metadata.len())
}

/// Reads an exact range without closing or moving the caller's descriptor.
pub(crate) fn read_descriptor_exact_at(
    descriptor: i32,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    duplicate_file(descriptor)?.read_exact_at(buffer, offset)
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
    duplicate_file(descriptor)?.read_at(buffer, offset)
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
    duplicate_file(descriptor)?.write_at(buffer, offset)
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
