// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows host services, independent of any GPU driver provider.

pub(crate) mod file;
mod memory;
pub(crate) mod module;
pub(crate) mod process_identity;

use crate::{Error, ErrorKind};
pub(crate) use memory::{
    HostAllocation, HostReservation, host_memory_capabilities, host_page_size,
};
use std::io;

pub(crate) fn native_error(operation: &'static str, source: io::Error) -> Error {
    let kind = match source.kind() {
        io::ErrorKind::PermissionDenied => ErrorKind::PermissionDenied,
        io::ErrorKind::NotFound => ErrorKind::NotFound,
        io::ErrorKind::OutOfMemory => ErrorKind::ResourceExhausted,
        io::ErrorKind::Unsupported => ErrorKind::Unsupported,
        io::ErrorKind::InvalidInput => ErrorKind::InvalidArgument,
        io::ErrorKind::InvalidData => ErrorKind::DriverContract,
        io::ErrorKind::WouldBlock => ErrorKind::Busy,
        _ => ErrorKind::Driver,
    };
    Error::NativeOperation {
        kind,
        operation,
        source,
    }
}

/// QPC is monotonic and system-wide; convert ticks using wide arithmetic.
/// This does not imply correlation with any device provider's timebase.
#[allow(unsafe_code)]
pub(crate) fn host_clock() -> Result<u64, Error> {
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };
    let mut ticks = 0;
    let mut frequency = 0;
    // SAFETY: Both outputs are complete, live i64 stack slots.
    if unsafe { QueryPerformanceCounter(&raw mut ticks) } == 0
        || unsafe { QueryPerformanceFrequency(&raw mut frequency) } == 0
    {
        return Err(native_error(
            "Windows performance counter",
            io::Error::last_os_error(),
        ));
    }
    let value = u128::try_from(ticks)
        .ok()
        .zip(u128::try_from(frequency).ok())
        .filter(|(_, frequency)| *frequency != 0)
        .and_then(|(ticks, frequency)| u64::try_from(ticks * 1_000_000_000 / frequency).ok())
        .ok_or_else(|| {
            native_error(
                "Windows performance counter conversion",
                io::Error::from(io::ErrorKind::InvalidData),
            )
        })?;
    Ok(value)
}
