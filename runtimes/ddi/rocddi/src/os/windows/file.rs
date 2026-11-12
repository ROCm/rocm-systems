// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows file operations requiring cursor-preserving qualification.
//!
//! Rust's Windows `FileExt::seek_read`/`seek_write` change the shared file cursor,
//! so they cannot implement the common positioned-I/O contract.

use std::{fs::File, io, path::PathBuf};

pub(crate) fn read_at(_file: &File, _bytes: &mut [u8], _offset: u64) -> io::Result<usize> {
    Err(io::ErrorKind::Unsupported.into())
}

pub(crate) fn write_at(_file: &File, _bytes: &[u8], _offset: u64) -> io::Result<usize> {
    Err(io::ErrorKind::Unsupported.into())
}

pub(crate) fn path(_file: &File) -> io::Result<PathBuf> {
    Err(io::ErrorKind::Unsupported.into())
}
