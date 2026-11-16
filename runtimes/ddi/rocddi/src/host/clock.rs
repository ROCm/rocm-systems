// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host profiling timebase shared by runtime frontends.
//!
//! Its nanosecond counter rate is not a CPU core frequency. A read is a live
//! observation rather than a field of the host topology graph snapshot.

use crate::{Error, os};

/// Frequency and meaningful width of the host profiling counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClockInfo {
    /// Counter ticks per second.
    pub frequency_hz: u64,
    /// Meaningful counter bits, or zero when the native width is unspecified.
    pub counter_bit_count: u32,
}

/// Returns the stable rate and width of the host profiling clock.
#[must_use]
pub const fn info() -> ClockInfo {
    ClockInfo {
        frequency_hz: 1_000_000_000,
        counter_bit_count: 64,
    }
}

/// Reads the monotonic, wall-clock-independent host profiling clock.
///
/// # Errors
/// Reports a native clock-read failure without changing any caller output.
pub fn read() -> Result<u64, Error> {
    os::host_clock()
}

impl ClockInfo {
    /// Converts an interval using wide arithmetic, rejecting zero rates and
    /// unrepresentable results. This never equates epochs of separate clocks.
    ///
    /// # Errors
    /// Reports invalid rates or a result outside the destination counter width.
    pub fn scale_interval(self, ticks: u64, destination_hz: u64) -> Result<u64, Error> {
        if self.frequency_hz == 0 || destination_hz == 0 {
            return Err(clock_error("clock rate is unavailable"));
        }
        u64::try_from(
            u128::from(ticks) * u128::from(destination_hz) / u128::from(self.frequency_hz),
        )
        .map_err(|_| clock_error("scaled clock interval exceeds counter width"))
    }

    /// Signed distance to the nearest tick in the same uninterrupted epoch.
    /// Unknown width uses ordinary subtraction and never guesses a wrap.
    /// Known width admits only distances strictly below half a counter period.
    /// Reset, suspend, removal, or source restart invalidates old correlations;
    /// callers must obtain a new sample rather than interpret a reset as wrap.
    ///
    /// # Errors
    /// Rejects invalid widths, high bits outside the counter, and ambiguous
    /// half-period distances. These checks cannot detect every device reset.
    pub fn tick_delta(self, origin: u64, tick: u64) -> Result<i128, Error> {
        let bits = self.counter_bit_count;
        if bits == 0 {
            return Ok(i128::from(tick) - i128::from(origin));
        }
        if bits > 64 {
            return Err(clock_error("invalid counter width"));
        }
        let modulus = 1_u128 << bits;
        if u128::from(origin) >= modulus || u128::from(tick) >= modulus {
            return Err(clock_error("timestamp exceeds counter width"));
        }
        let forward = (u128::from(tick) + modulus - u128::from(origin)) % modulus;
        if forward == modulus / 2 {
            return Err(clock_error("ambiguous counter interval"));
        }
        let signed =
            i128::try_from(forward).map_err(|_| clock_error("counter interval overflow"))?;
        if forward < modulus / 2 {
            Ok(signed)
        } else {
            Ok(signed
                - i128::try_from(modulus).map_err(|_| clock_error("counter modulus overflow"))?)
        }
    }
}

fn clock_error(detail: &'static str) -> Error {
    Error::Operation {
        kind: crate::ErrorKind::InvalidData,
        detail,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn widths_disambiguate_wrap_without_guessing_unknown_counter_geometry() {
        let info = ClockInfo {
            frequency_hz: 100,
            counter_bit_count: 8,
        };
        assert_eq!(info.tick_delta(250, 3).unwrap(), 9);
        assert_eq!(info.tick_delta(3, 250).unwrap(), -9);
        assert!(info.tick_delta(0, 128).is_err());
        assert!(info.tick_delta(256, 1).is_err());
        assert_eq!(
            ClockInfo {
                counter_bit_count: 0,
                ..info
            }
            .tick_delta(250, 3)
            .unwrap(),
            -247
        );
        assert_eq!(
            ClockInfo {
                counter_bit_count: 64,
                ..info
            }
            .tick_delta(u64::MAX, 0)
            .unwrap(),
            1
        );
        assert!(
            ClockInfo {
                counter_bit_count: 65,
                ..info
            }
            .tick_delta(0, 1)
            .is_err()
        );
    }

    #[test]
    fn scaling_reports_overflow_instead_of_saturating_or_wrapping() {
        let info = ClockInfo {
            frequency_hz: 3,
            counter_bit_count: 0,
        };
        assert_eq!(info.scale_interval(6, 10).unwrap(), 20);
        assert!(info.scale_interval(u64::MAX, u64::MAX).is_err());
        assert!(info.scale_interval(6, 0).is_err());
        assert!(
            ClockInfo {
                frequency_hz: 0,
                ..info
            }
            .scale_interval(1, 10)
            .is_err()
        );
    }
}
