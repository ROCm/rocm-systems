// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Allocation and mapping constraints shared by host and device memory.
//!
//! These values prove arithmetic constraints only. Native owners separately
//! establish process validity, address domains, access, and release semantics.

/// One native memory request with explicit size and alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationDesc {
    /// Requested bytes. Host registration measures this extent from its exact
    /// address; owned allocations satisfy the allocator's size granularity.
    pub size: u64,
    /// Required base alignment in bytes, expressed as a nonzero power of two.
    pub alignment: u64,
}

/// Native allocation limits for one backing kind. A successful query describes support,
/// not a promise that sufficient memory will still be available when allocating.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationLimits {
    /// Minimum base alignment in bytes, expressed as a power of two.
    pub alignment: u64,
    /// Required size multiple in bytes. Native requests are not silently rounded.
    pub granularity: u64,
    /// Largest request supported by this allocator, independent of free capacity.
    pub maximum_size: u64,
}

impl AllocationLimits {
    /// Checks a request against these facts without querying or allocating on a
    /// device. The allocator still validates the request against current native
    /// state; retaining this value does not keep a removed device operational.
    #[must_use]
    pub fn supports(self, desc: AllocationDesc) -> bool {
        self.is_valid()
            && desc.size != 0
            && desc.size <= self.maximum_size
            && desc.size % self.granularity == 0
            && desc.alignment.is_power_of_two()
            && desc.alignment >= self.alignment
    }

    pub(crate) fn is_valid(self) -> bool {
        self.alignment.is_power_of_two()
            && self.granularity != 0
            && self.maximum_size >= self.granularity
    }
}

/// Whether a mapping can end with a partial replacement unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RangeTail {
    /// Every mapping consists of complete units.
    WholeUnits,
    /// A short final unit is permitted only at the end of its container.
    EndOfReservation,
}

/// Contained, nonempty range with an aligned offset and a checked tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedRange {
    offset: u64,
    size: u64,
}

impl CheckedRange {
    /// Checks containment without overflowing an exclusive end address.
    /// Native page geometry must be checked separately when `unit` is a
    /// larger frontend replacement unit.
    #[must_use]
    pub fn new(
        container_size: u64,
        offset: u64,
        size: u64,
        unit: u64,
        tail: RangeTail,
    ) -> Option<Self> {
        (unit != 0
            && size != 0
            && offset % unit == 0
            && offset <= container_size
            && size <= container_size - offset
            && (size % unit == 0
                || (tail == RangeTail::EndOfReservation && size == container_size - offset)))
            .then_some(Self { offset, size })
    }

    /// Checked offset from the container base.
    #[must_use]
    pub fn offset(self) -> u64 {
        self.offset
    }

    /// Checked range length in bytes.
    #[must_use]
    pub fn size(self) -> u64 {
        self.size
    }
}

/// Checks a nonempty aligned range against inclusive address bounds.
#[must_use]
pub fn fits_address_bounds(address: u64, size: u64, alignment: u64, bounds: (u64, u64)) -> bool {
    alignment.is_power_of_two()
        && size != 0
        && address % alignment == 0
        && address >= bounds.0
        && address
            .checked_add(size - 1)
            .is_some_and(|last| last <= bounds.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_final_replacement_does_not_weaken_native_page_geometry() {
        assert!(
            CheckedRange::new(
                68 * 1024,
                64 * 1024,
                4096,
                64 * 1024,
                RangeTail::EndOfReservation
            )
            .is_some()
        );
        assert!(
            CheckedRange::new(68 * 1024, 64 * 1024, 4096, 64 * 1024, RangeTail::WholeUnits)
                .is_none()
        );
        assert!(
            CheckedRange::new(68 * 1024, 0, 4096, 64 * 1024, RangeTail::EndOfReservation).is_none()
        );
        assert!(
            CheckedRange::new(68 * 1024, 64 * 1024, 4096, 4096, RangeTail::WholeUnits).is_some()
        );
        assert!(
            CheckedRange::new(68 * 1024, 64 * 1024, 4095, 4096, RangeTail::WholeUnits).is_none()
        );
    }

    #[test]
    fn range_checks_cover_zero_overflow_and_inclusive_upper_bounds() {
        assert!(CheckedRange::new(u64::MAX, u64::MAX - 1, 3, 1, RangeTail::WholeUnits).is_none());
        assert!(CheckedRange::new(4096, 0, 4096, 0, RangeTail::WholeUnits).is_none());
        assert!(CheckedRange::new(4096, 0, 0, 4096, RangeTail::WholeUnits).is_none());
        assert!(fits_address_bounds(
            u64::MAX - 4095,
            4096,
            4096,
            (0, u64::MAX)
        ));
        assert!(!fits_address_bounds(
            u64::MAX - 4095,
            8192,
            4096,
            (0, u64::MAX)
        ));
        assert!(!fits_address_bounds(0x11000, 4096, 65536, (0, u64::MAX)));
        assert!(!fits_address_bounds(4096, 0, 4096, (0, u64::MAX)));
    }
}
