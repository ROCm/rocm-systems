// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Memory whose access is qualified for activated devices.
//!
//! Placement and access belong to an activated
//! [`Device`](crate::device::Device), whose endpoint can be a GPU, CPU, NPU, or
//! another supported kind. An allocation owns its native backing and mappings;
//! a raw registered host range borrows caller-owned pages until registration is
//! released. [`RegisteredHostAllocation`] safely retains rocddi-owned host
//! backing across an independent registration in each driver. A host address
//! exists only when the backend has established a host mapping. HSA policy for
//! CPU and accelerator pools remains in the HSA frontend.

mod access;
mod allocation;
mod registered_host;
mod types;
mod virtual_memory;

pub use access::DeviceAccess;
pub use allocation::Allocation;
pub use registered_host::{RegisteredHostAllocation, RegisteredHostMemory};
pub use types::{
    AllocationInfo, HostCacheability, HostMappingPolicy, MemoryKind, VirtualAddressInfo,
    VirtualMemoryInfo,
};
pub use virtual_memory::{VirtualAddress, VirtualDeviceMapping, VirtualHostMapping, VirtualMemory};

pub(crate) use allocation::{DriverAllocation, set_device_access_for};
pub(crate) use types::{AllocationDesc, AllocationLimits, HostRegistration, VirtualMapRequest};
#[cfg(test)]
pub(crate) use virtual_memory::{DeviceIntervals, HostIntervals};
pub(crate) use virtual_memory::{
    DriverVirtualAddress, DriverVirtualDeviceMapping, DriverVirtualHostMapping, DriverVirtualMemory,
};
