// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Error classification shared by Linux host and device services.

use crate::{Error, ErrorKind};
use std::io;

pub(super) fn error(kind: ErrorKind, detail: &'static str) -> Error {
    Error::Operation { kind, detail }
}

pub(super) fn native_error(operation: &'static str, source: io::Error) -> Error {
    let kind = match source.kind() {
        io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        io::ErrorKind::OutOfMemory => ErrorKind::ResourceExhausted,
        io::ErrorKind::Unsupported => ErrorKind::Unsupported,
        io::ErrorKind::InvalidData => ErrorKind::DriverContract,
        io::ErrorKind::WouldBlock => ErrorKind::Busy,
        _ => match source.raw_os_error() {
            Some(11 | 16) => ErrorKind::Busy,
            Some(12 | 28) => ErrorKind::ResourceExhausted,
            Some(25 | 95) => ErrorKind::Unsupported,
            _ => ErrorKind::Driver,
        },
    };
    Error::NativeOperation {
        kind,
        operation,
        source,
    }
}
