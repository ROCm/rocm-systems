// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux implementation of the shared profiling clock.

#![allow(unsafe_code)]

use super::native_error;
use crate::Error;

pub(crate) fn host_clock() -> Result<u64, Error> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: The kernel writes the complete stack-local timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &raw mut time) } != 0 {
        return Err(native_error(
            "Linux monotonic raw clock",
            std::io::Error::last_os_error(),
        ));
    }
    let seconds = u64::try_from(time.tv_sec).map_err(|_| {
        native_error(
            "Linux monotonic raw clock",
            std::io::Error::from(std::io::ErrorKind::InvalidData),
        )
    })?;
    let nanos = u64::try_from(time.tv_nsec).map_err(|_| {
        native_error(
            "Linux monotonic raw clock",
            std::io::Error::from(std::io::ErrorKind::InvalidData),
        )
    })?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or_else(|| {
            native_error(
                "Linux monotonic raw clock",
                std::io::Error::from(std::io::ErrorKind::InvalidData),
            )
        })
}
