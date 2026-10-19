// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Portable host profiling clock shared by runtime frontends.

use crate::{Error, os};

/// Frequency and meaningful width of the host profiling counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostClockInfo {
    /// Counter ticks per second.
    pub frequency_hz: u64,
    /// Meaningful counter bits.
    pub counter_bit_count: u32,
}

/// Returns the stable rate and width of the host profiling clock.
#[must_use]
pub const fn host_clock_info() -> HostClockInfo {
    HostClockInfo {
        frequency_hz: 1_000_000_000,
        counter_bit_count: 64,
    }
}

/// Reads the monotonic, wall-clock-independent host profiling clock.
///
/// # Errors
/// Reports a native clock-read failure without changing any caller output.
pub fn host_clock() -> Result<u64, Error> {
    os::host_clock()
}
