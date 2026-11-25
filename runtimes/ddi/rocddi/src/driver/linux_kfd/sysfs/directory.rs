// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Numeric Linux sysfs directory enumeration with stack-owned storage.

#![allow(unsafe_code)]

use std::ffi::c_long;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;

unsafe extern "C" {
    fn syscall(number: c_long, ...) -> c_long;
}

/// Enumerates numeric Linux sysfs directories without an allocator-owned path
/// or libc directory buffer. The caller decides where durable records live.
pub(super) fn numeric_directories(
    path: &std::path::Path,
    visitor: &mut dyn FnMut(u32) -> Result<(), crate::Error>,
) -> Result<(), crate::Error> {
    #[cfg(target_arch = "x86_64")]
    const GETDENTS64: c_long = 217;
    #[cfg(target_arch = "aarch64")]
    const GETDENTS64: c_long = 61;
    let directory =
        File::open(path).map_err(|source| super::native_error("sysfs directory open", source))?;
    let mut storage = [0u8; 4096];
    loop {
        // SAFETY: getdents64 receives a live directory descriptor and a writable
        // byte array of exactly the advertised length. Records are decoded as
        // bytes below, without assuming alignment of their kernel layout.
        let result = unsafe {
            syscall(
                GETDENTS64,
                directory.as_raw_fd(),
                storage.as_mut_ptr(),
                storage.len(),
            )
        };
        if result < 0 {
            return Err(super::native_error(
                "sysfs directory read",
                io::Error::last_os_error(),
            ));
        }
        if result == 0 {
            return Ok(());
        }
        let length = usize::try_from(result)
            .map_err(|_| super::error(crate::ErrorKind::InvalidData, "invalid directory length"))?;
        if length > storage.len() {
            return Err(super::error(
                crate::ErrorKind::InvalidData,
                "oversized directory response",
            ));
        }
        let mut offset = 0;
        while offset < length {
            let record = &storage[offset..length];
            if record.len() < 20 {
                return Err(super::error(
                    crate::ErrorKind::InvalidData,
                    "truncated directory record",
                ));
            }
            let size = usize::from(u16::from_ne_bytes([record[16], record[17]]));
            if size < 20 || size > record.len() {
                return Err(super::error(
                    crate::ErrorKind::InvalidData,
                    "invalid directory record",
                ));
            }
            let name = &record[19..size];
            let end = name.iter().position(|byte| *byte == 0).ok_or_else(|| {
                super::error(crate::ErrorKind::InvalidData, "unterminated directory name")
            })?;
            let name = &name[..end];
            if !name.is_empty() && name.iter().all(u8::is_ascii_digit) {
                let mut number = 0u32;
                for byte in name {
                    number = number
                        .checked_mul(10)
                        .and_then(|n| n.checked_add(u32::from(byte - b'0')))
                        .ok_or_else(|| {
                            super::error(
                                crate::ErrorKind::InvalidData,
                                "directory ordinal overflow",
                            )
                        })?;
                }
                visitor(number)?;
            }
            offset += size;
        }
    }
}
