//! Validated device-memory access flags and their set operations.

use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
};

/// A set of independently composable device-memory permissions.
///
/// The integer representation is deliberately private. Values originating at
/// an ABI boundary must pass through [`Self::from_bits`], which rejects bits
/// rocddi does not understand. Callers compose known permissions with the
/// ordinary bitwise operators, for example
/// `DeviceAccess::READ | DeviceAccess::WRITE`.
///
/// Individual operations may impose narrower rules. Native device allocations,
/// for example, may require [`Self::READ`], while a mapping implementation may
/// reject executable access even though it is a valid member of this set.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceAccess(u32);

impl DeviceAccess {
    const VALID_BITS: u32 = Self::READ.0 | Self::WRITE.0 | Self::EXECUTE.0;

    /// No device access.
    pub const NONE: Self = Self(0);
    /// Device read permission.
    pub const READ: Self = Self(1 << 0);
    /// Device write permission.
    pub const WRITE: Self = Self(1 << 1);
    /// Device instruction-fetch permission.
    pub const EXECUTE: Self = Self(1 << 2);

    /// Constructs a permission set if every supplied bit is known to rocddi.
    ///
    /// Returns `None` rather than silently discarding unknown bits, so a caller
    /// cannot accidentally widen or reinterpret permissions across an ABI or
    /// version boundary.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::VALID_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Returns the underlying integer bits used by native and public interface
    /// adapters. This does not create a separate Rust ABI stability promise.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns whether this set contains every permission in `required`.
    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// Returns whether this set and `other` share at least one permission.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Returns whether this set contains no permissions.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for DeviceAccess {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for DeviceAccess {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for DeviceAccess {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for DeviceAccess {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitXor for DeviceAccess {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for DeviceAccess {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for DeviceAccess {
    type Output = Self;

    fn not(self) -> Self::Output {
        // Mask the result to the known domain. Unlike integer `!`, this cannot
        // manufacture unknown permission bits in an otherwise valid value.
        Self(Self::VALID_BITS & !self.0)
    }
}

impl Sub for DeviceAccess {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 & !rhs.0)
    }
}

impl SubAssign for DeviceAccess {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 &= !rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_access_is_a_checked_composable_bit_set() {
        let read_write = DeviceAccess::READ | DeviceAccess::WRITE;
        let all = read_write | DeviceAccess::EXECUTE;

        assert_eq!(DeviceAccess::default(), DeviceAccess::NONE);
        assert_eq!(DeviceAccess::from_bits(0), Some(DeviceAccess::NONE));
        assert_eq!(DeviceAccess::from_bits(3), Some(read_write));
        assert_eq!(DeviceAccess::from_bits(7), Some(all));
        assert_eq!(DeviceAccess::from_bits(8), None);
        assert_eq!(DeviceAccess::from_bits(u32::MAX), None);

        assert!(all.contains(read_write));
        assert!(all.intersects(DeviceAccess::EXECUTE));
        assert!(!read_write.intersects(DeviceAccess::EXECUTE));
        assert!(!read_write.is_empty());
        assert!(DeviceAccess::NONE.is_empty());
        assert_eq!(read_write.bits(), 3);
        assert_eq!(
            all - DeviceAccess::WRITE,
            DeviceAccess::READ | DeviceAccess::EXECUTE
        );
        assert_eq!(
            !DeviceAccess::WRITE,
            DeviceAccess::READ | DeviceAccess::EXECUTE
        );
    }
}
