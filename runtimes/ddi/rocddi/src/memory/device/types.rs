// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native allocation requests and limits, independent of kernel handle formats.

/// One native memory request with explicit size and alignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationDesc {
    /// Requested bytes. Host registration measures this extent from its exact
    /// address; owned allocations satisfy the allocator's size granularity.
    pub size: u64,
    /// Required base alignment in bytes, expressed as a nonzero power of two.
    pub alignment: u64,
}

/// Returns the aligned base and page cover for `size` bytes starting at the
/// exact registered host address. Both KFD and DRM registration use this rule.
pub(crate) fn registered_host_page_cover(
    address: usize,
    size: usize,
    page: usize,
) -> Option<(usize, usize)> {
    if address == 0 || size == 0 || !page.is_power_of_two() {
        return None;
    }
    let offset = address & (page - 1);
    let base = address - offset;
    let cover = offset.checked_add(size)?.checked_add(page - 1)? & !(page - 1);
    base.checked_add(cover)?;
    Some((base, cover))
}

#[cfg(test)]
mod registration_tests {
    use super::registered_host_page_cover;

    #[test]
    fn exact_unaligned_extent_covers_the_page_after_its_end() {
        assert_eq!(
            registered_host_page_cover(0x1001, 4096, 4096),
            Some((0x1000, 8192))
        );
        assert_eq!(
            registered_host_page_cover(0x1001, 4095, 4096),
            Some((0x1000, 4096))
        );
        assert_eq!(registered_host_page_cover(0x1001, 0, 4096), None);
    }
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

use super::access::DeviceAccess;
use crate::memory::host::HostBackingLease;

/// Host mapping cache policy qualified by the native backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostCacheability {
    /// Ordinary write-back host mapping.
    WriteBack,
    /// Write-combined host mapping.
    WriteCombined,
}

/// Host/device visibility and cache behavior requested for host pages.
///
/// These are observable memory contracts, independent of how a driver binds
/// the pages. A device driver accepts only policies it can implement for the
/// selected endpoint. The API frontend chooses the policy for its own memory
/// model; a driver cannot silently weaken it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostMappingPolicy {
    /// Require explicit synchronization between host and device views.
    Coarse,
    /// Provide coherent host and device access.
    Fine,
    /// Provide coherent access across the device's extended scope.
    Extended,
    /// Bypass device caches while retaining host/device coherence.
    Uncached,
}

/// Platform-neutral backing and placement requested from the native backend.
///
/// These variants describe ownership, visibility, and address behavior that a
/// caller can observe. They deliberately do not encode operating-system handle
/// types or driver allocation mechanisms; each platform backend selects the
/// native mechanism that satisfies the requested contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKind {
    /// Platform-managed, host-accessible system backing with device access
    /// established in the activated address space. The native platform memory
    /// manager owns the physical backing; rocddi owns its returned allocation
    /// and persistent host mapping. Host-only allocations instead use
    /// [`HostAllocation::allocate`](crate::memory::host::HostAllocation::allocate).
    System,
    /// rocddi-owned host pages made accessible to the device at the same host
    /// and device virtual address. The backend may pin, register, or otherwise
    /// bind those pages without exposing that native mechanism.
    OwnedHost {
        /// Requested host/device mapping policy for the owned pages.
        policy: HostMappingPolicy,
    },
    /// Device-local storage; host visibility is an explicit requirement.
    DeviceLocal {
        /// Require a persistent host mapping of the local storage.
        host_visible: bool,
        /// Request coherent device mappings for fine-grained access.
        coherent: bool,
        /// Promote device-scope atomic coherence to extended system scope.
        extended_coherent: bool,
        /// Bypass device caches for this allocation.
        uncached: bool,
        /// Request physically contiguous backing.
        contiguous: bool,
    },
}

/// Parameters and optional owned backing for host registration.
/// A driver drops `backing` if no native reference remains after failure,
/// retains it with a live registration, and quarantines it when cleanup cannot
/// prove that native access has ended.
pub(crate) struct HostRegistration {
    /// First caller-owned or runtime-owned host byte.
    pub(crate) address: usize,
    /// Native cache behavior requested for the registered pages.
    pub(crate) policy: HostMappingPolicy,
    /// Exact byte extent to expose to the device.
    pub(crate) size: u64,
    /// Required device address alignment.
    pub(crate) alignment: u64,
    /// Device access established at registration.
    pub(crate) permissions: DeviceAccess,
    /// Retained owner for runtime-allocated backing, when present.
    pub(crate) backing: Option<HostBackingLease>,
}

/// Cached addresses and extents of one successfully mapped native allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationInfo {
    /// Address in the activated device's native address space.
    pub device_address: u64,
    /// Existing host mapping when requested and supported.
    pub host_address: Option<usize>,
    /// Exact requested bytes.
    pub size: u64,
    /// Complete native physical allocation or imported backing extent.
    pub native_size: u64,
    /// Stable native identity for imported or shareable backing, when available.
    pub physical_backing_id: [u64; 2],
}

/// Immutable facts describing one detached virtual-memory allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualMemoryInfo {
    /// Complete physical backing extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings of this backing.
    pub mapping_granularity: u64,
    /// Stable native identity shared by equivalent handles for this backing.
    pub physical_backing_id: [u64; 2],
    /// Native placement, if the driver could describe the backing.
    pub placement: Option<VirtualMemoryPlacement>,
}

/// Placement class of a detached virtual-memory backing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VirtualMemoryPlacement {
    /// System memory visible through a CPU agent.
    System,
    /// Device memory owned by the KFD GPU with this native identity.
    Device(u32),
}

/// Cached facts for one reserved process virtual-address range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualAddressInfo {
    /// First byte in the reserved range.
    pub address: u64,
    /// Exact reserved extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings in this reservation.
    pub mapping_granularity: u64,
}

/// One checked placement request for a driver virtual-memory mapping.
#[derive(Clone, Copy)]
pub(crate) struct VirtualMapRequest {
    /// First virtual byte to map inside the reservation.
    pub(crate) address: u64,
    /// Byte offset into physical backing.
    pub(crate) offset: u64,
    /// Byte extent of the mapping.
    pub(crate) size: u64,
    /// Access granted to the selected address space.
    pub(crate) permissions: DeviceAccess,
}
