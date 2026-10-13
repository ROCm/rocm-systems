// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Runtime storage, host backing, and device-accessible memory.
//!
//! [`storage`] allocates runtime records through the session's callback
//! allocator. [`host`] owns host-only backing and cache operations without
//! activating an endpoint. [`device`] owns memory and mappings whose access
//! has been qualified by an activated [`Device`](crate::device::Device). A CPU
//! can use this interface when a driver exposes it as an activated endpoint.
//! Rocddi-owned host backing can be registered independently with several
//! drivers while each registration retains its pages. Ordinary caller-owned
//! host memory remains outside rocddi ownership and uses the unsafe raw
//! registration API.
//! Platform-specific sharing transports live under [`interop`].

pub mod device;
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod driver_virtual_tests;
pub mod host;
pub mod interop;
pub mod storage;

pub(crate) use device::{
    Allocation, AllocationDesc, AllocationInfo, AllocationLimits, DeviceAccess, DriverAllocation,
    DriverVirtualAddress, DriverVirtualDeviceMapping, DriverVirtualHostMapping,
    DriverVirtualMemory, HostCacheability, HostMappingPolicy, HostRegistration, MemoryKind,
    VirtualAddressInfo, VirtualMapRequest, VirtualMemory, VirtualMemoryInfo, set_device_access_for,
};
#[cfg(test)]
pub(crate) use device::{DeviceIntervals, HostIntervals};
pub(crate) use host::host_page_size;
