// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Ordinary positioned file operations, independent of GPU memory transports.
//!
//! Callers retain a `File` for the operation. Positioned reads and writes do
//! not change its shared cursor. Native-handle acquisition stays in the OS
//! adapter; a descriptor, HANDLE, and C stream are different contracts.
//! Native file requests use signed 64-bit positions so offsets cannot be
//! interpreted as an append sentinel.

use std::{fs::File, io, path::PathBuf};

/// Reads bytes at an explicit offset without changing the shared file cursor.
///
/// # Errors
/// Reports invalid input for offsets above `i64::MAX`, or the native file error.
/// A short read is returned as progress.
pub fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<usize> {
    if offset > i64::MAX as u64 {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    crate::os::file::read_at(file, bytes, offset)
}

/// Writes bytes at an explicit offset without changing the shared file cursor.
///
/// # Errors
/// Reports invalid input for offsets above `i64::MAX`, or the native file error.
/// A short write is returned as progress.
pub fn write_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<usize> {
    if offset > i64::MAX as u64 {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    crate::os::file::write_at(file, bytes, offset)
}

/// Reads an exact range, retrying interrupted ordinary file reads.
///
/// # Errors
/// Reports EOF, range overflow, or the native failure. The output can contain
/// partial data after an error and must not be published as a full snapshot.
pub fn read_exact_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
    offset
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    while !bytes.is_empty() {
        match read_at(file, bytes, offset) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            Ok(count) => {
                offset += count as u64;
                bytes = &mut bytes[count..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => (),
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Resolves a reopenable path to the retained file, when one exists.
///
/// # Errors
/// Anonymous, unlinked, or renamed objects may have no verifiable path.
/// Callers can still snapshot their bytes when origin metadata is unavailable.
pub fn path(file: &File) -> io::Result<PathBuf> {
    crate::os::file::path(file)
}
