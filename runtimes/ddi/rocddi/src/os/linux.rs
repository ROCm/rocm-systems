// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux host services used by the DDI core and device drivers.
//!
//! `file` owns raw descriptor transfer and positioned I/O. `memory` owns page
//! discovery, host allocations, address reservations, and the fork marker page.
//! `process_identity` publishes and checks that marker. Error classification
//! stays here so callers can preserve the native operation and errno.

pub(crate) mod file;
pub(crate) mod memory;
pub(crate) mod process_identity;

pub(crate) use libc as errno;
pub(crate) use memory::{HostAllocation, host_page_size};

use crate::{Error, ErrorKind};
use std::io;

/// Classifies malformed native data without allocating an error message.
/// The caller supplies the operation label when translating the I/O error.
pub(crate) fn invalid_data(_detail: impl std::fmt::Display) -> io::Error {
    io::Error::from(io::ErrorKind::InvalidData)
}

/// Classifies a Linux I/O failure while preserving its operation and errno.
pub(crate) fn native_error(operation: &'static str, source: io::Error) -> Error {
    let kind = match source.kind() {
        io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        io::ErrorKind::OutOfMemory => ErrorKind::ResourceExhausted,
        io::ErrorKind::Unsupported => ErrorKind::Unsupported,
        io::ErrorKind::InvalidData => ErrorKind::DriverContract,
        io::ErrorKind::WouldBlock => ErrorKind::Busy,
        _ => match source.raw_os_error() {
            Some(errno::EAGAIN | errno::EBUSY) => ErrorKind::Busy,
            Some(errno::ENOMEM | errno::ENOSPC) => ErrorKind::ResourceExhausted,
            Some(errno::ENOTTY | errno::EOPNOTSUPP) => ErrorKind::Unsupported,
            _ => ErrorKind::Driver,
        },
    };
    Error::NativeOperation {
        kind,
        operation,
        source,
    }
}
