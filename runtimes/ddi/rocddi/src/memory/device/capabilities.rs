// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive, request-specific provider memory guarantees.

use super::{DeviceAccess, HostCacheability, HostMappingPolicy, MemoryKind};
use crate::memory::constraints::{AllocationLimits, fits_address_bounds};

/// The construction contract whose support is being queried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryCapabilityRequest<'a> {
    /// Provider-owned backing with the requested placement and visibility.
    Allocation(MemoryKind),
    /// Caller-owned host bytes mapped under the requested coherence policy.
    Registration(HostMappingPolicy),
    /// Access to backing allocated by another endpoint. Geometry describes
    /// that owner's backing; access guarantees describe the consuming endpoint.
    Access {
        /// Endpoint that owns the physical backing.
        owner: &'a crate::topology::Endpoint,
        /// Construction kind whose backing will be accessed.
        kind: MemoryKind,
    },
}

/// Native and logical geometry plus access guarantees for one request.
/// These are support facts, not a reservation of memory or mapping resources.
#[allow(
    clippy::struct_excessive_bools,
    reason = "independently qualified native memory properties"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceMemoryCapabilities {
    /// Native allocation/page-cover limits; logical lengths may be rounded by
    /// a frontend only if its own contract permits that rounding.
    pub native: AllocationLimits,
    /// Largest supported power-of-two base alignment.
    pub maximum_alignment: u64,
    /// Minimum caller address/alignment for the logical construction.
    pub logical_alignment: u64,
    /// Required logical byte-length multiple, separate from the native cover.
    pub logical_size_granularity: u64,
    /// Access every device mapping must include.
    pub required_access: DeviceAccess,
    /// Access the provider can implement for this request.
    pub supported_access: DeviceAccess,
    /// A persistent host projection is available independently of cache policy.
    pub host_visible: bool,
    /// Host cache policy when known; absence does not deny a host projection.
    pub host_cacheability: Option<HostCacheability>,
    /// Whether the host projection permits writes as well as reads.
    pub host_writable: bool,
    /// Host/device coherence is guaranteed for the requested policy.
    pub host_coherent: bool,
    /// Backing is qualified for this provider's queue storage requirements.
    pub queue_storage: bool,
}

impl DeviceMemoryCapabilities {
    /// Checks the logical caller range and device access before native work.
    #[must_use]
    pub fn supports_range(self, address: u64, size: u64, access: DeviceAccess) -> bool {
        fits_address_bounds(address, size, self.logical_alignment, (0, u64::MAX))
            && self.logical_size_granularity != 0
            && size <= self.native.maximum_size
            && size % self.logical_size_granularity == 0
            && access.contains(self.required_access)
            && self.supported_access.contains(access)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_ranges_reject_overflow_and_preserve_inclusive_endpoints() {
        let capabilities = DeviceMemoryCapabilities {
            native: AllocationLimits {
                alignment: 4096,
                granularity: 4096,
                maximum_size: 4096,
            },
            maximum_alignment: 4096,
            logical_alignment: 1,
            logical_size_granularity: 1,
            required_access: DeviceAccess::READ,
            supported_access: DeviceAccess::READ,
            host_visible: true,
            host_cacheability: None,
            host_writable: false,
            host_coherent: false,
            queue_storage: false,
        };
        for (address, size, supported) in [
            (0, 1, true),
            (u64::MAX, 1, true),
            (u64::MAX - 1, 2, true),
            (u64::MAX, 2, false),
            (u64::MAX - 1, 3, false),
            (u64::MAX, 0, false),
            (0, 0, false),
        ] {
            assert_eq!(
                capabilities.supports_range(address, size, DeviceAccess::READ),
                supported,
                "address={address:#x}, size={size}"
            );
        }
    }
}
