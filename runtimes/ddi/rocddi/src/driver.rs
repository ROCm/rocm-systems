// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private interface implemented by each concrete device driver.
//!
//! [`Driver`] owns discovery, activation, and shutdown. A GPU driver also
//! implements [`GpuDriver`], including creation of both GPU queue models.
//! Allocation and virtual-memory operations remain separate because their
//! owners can be shared with other device families. Queue operations belong to
//! the queue resources returned by the GPU driver, not to additional drivers.

mod instance;
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod linux_kfd;
mod resources;
#[cfg(test)]
pub(crate) mod test_driver;
#[cfg(test)]
mod tests;

pub(crate) use instance::{
    DeviceDriverState, DriverInstance, EndpointSelector, EventSubscription, KfdAllocation,
    KfdDeviceState, new_driver_instance,
};
pub(crate) use linux_kfd::KfdEventSubscription;
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) use linux_kfd::{KfdKernelQueue, KfdQueue, KfdSignalEvent, LinuxKfdDriver as KfdDriver};
pub(crate) use resources::{
    AllocationState, KernelQueueState, QueueState, VirtualAddressState, VirtualDeviceMappingState,
    VirtualHostMappingState, VirtualMemoryState,
};

use crate::device::gpu::kernel_queue::{
    KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait,
};
use crate::device::gpu::profiling::ClockCounters;
use crate::device::gpu::queue::{QueueRequest, QueueScratch, QueueTransport};
use crate::host_storage::{Allocator, Owned};
use crate::memory::{
    AllocationInfo, DeviceAccess, HostRegistration, OwnedMemoryKind, VirtualAddressInfo,
    VirtualMemoryInfo,
};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};

/// Address-space identity and bounds for devices that can map memory.
///
/// The core uses `shares_address_domain` to serialize overlapping mappings
/// within one device address space. Equal domains must report compatible
/// bounds. Mapping owners retain cloned state and share it across threads.
/// These cached facts do not establish access to a range or report device loss.
pub(crate) trait AddressSpaceInfo: Clone + Send + Sync {
    fn address_range(&self) -> (u64, u64);
    fn shares_address_domain(&self, other: &Self) -> bool;
}

/// Copies immutable facts cached by a driver-owned resource.
///
/// A caller takes the snapshot while the resource owner is live. The value is
/// information, not a fresh driver query or permission to use an address after
/// that owner has been released. Cleanup may invalidate the cached addresses.
pub(crate) trait CachedInfo {
    type Info: Copy;
    fn cached_info(&self) -> Self::Info;
}

/// Operations common to every activated endpoint and its session.
///
/// The session owns one or more driver instances. Each driver's identity ties
/// its passive endpoint snapshots to that instance; activation acquires that
/// driver's state for one device.
/// Discovery and opening are passive. The context lifetime is fixed when the
/// driver is constructed and is reported to the owning session.
/// Endpoint IDs must be unique among drivers installed in one session because
/// callers can open an endpoint by ID alone. The session checks collisions
/// when several drivers are installed.
/// Shutdown may fail after partial cleanup and must permit a retry with the
/// same instance. Capability traits add resource operations only when a caller
/// needs them.
pub(crate) trait Driver: Send + Sync {
    /// State retained for one endpoint activated through this driver instance.
    type DeviceState: Clone + Send + Sync;
    fn driver_instance(&self) -> u64;
    fn context_lifetime(&self) -> DriverContextLifetime;
    /// Allocator used for metadata retained by this driver and its resources.
    fn allocator(&self) -> Allocator;
    fn shutdown(&mut self) -> Result<(), Error>;
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error>;
    fn activate(&self, endpoint: &Endpoint) -> Result<Self::DeviceState, Error>;
}

/// Allocation, registration, and access operations on activated devices.
///
/// An allocation owner retains every native mapping dependency needed for
/// cleanup. `free_allocation` must leave failed cleanup retryable. Registered
/// host pages have an additional caller lifetime obligation stated below.
pub(crate) trait AllocationOperations: Driver {
    type Allocation: CachedInfo<Info = AllocationInfo>;
    fn supports_host_registration(&self, endpoint: &Endpoint) -> bool;
    fn check_allocation(allocation: &Self::Allocation) -> Result<(), Error>;
    fn allocation_is_device_local(allocation: &Self::Allocation) -> bool;
    fn set_allocation_access(
        _allocation: &mut Self::Allocation,
        _devices: &[&Self::DeviceState],
    ) -> Result<(), Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "driver does not support changing allocation access",
        })
    }
    fn allocation_device_address(
        allocation: &Self::Allocation,
        device: &Self::DeviceState,
    ) -> Result<u64, Error>;
    fn allocation_is_owned_by(allocation: &Self::Allocation, device: &Self::DeviceState) -> bool;
    fn free_allocation(allocation: &mut Self::Allocation) -> Result<(), Error>;
    fn allocate_owned(
        &self,
        device: &Self::DeviceState,
        peers: &[&Self::DeviceState],
        kind: OwnedMemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error>;
    /// # Safety
    /// The caller keeps the complete host page cover mapped and synchronized
    /// through successful free or, after ambiguous driver failure, process exit.
    #[allow(unsafe_code)]
    unsafe fn register_host(
        &self,
        _device: &Self::DeviceState,
        _peers: &[&Self::DeviceState],
        _request: HostRegistration,
    ) -> Result<Owned<Self::Allocation>, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "driver does not support host registration",
        })
    }
}

/// Address reservation, physical backing, and virtual mapping operations.
///
/// Reservations, backing, and each mapping have separate owners. The core
/// retains their dependencies through map and unmap; failed release must keep
/// the affected owner available for retry. A failed map must satisfy the
/// rollback or quarantine guarantees on the mapping methods below.
pub(crate) trait VirtualMemoryOperations: Driver {
    type VirtualAddress: CachedInfo<Info = VirtualAddressInfo>;
    type VirtualDeviceMapping;
    type VirtualHostMapping;
    type VirtualMemory: CachedInfo<Info = VirtualMemoryInfo>;
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<Self::VirtualAddress>, Error>;
    fn free_virtual_address(address: &mut Self::VirtualAddress) -> Result<(), Error>;
    fn create_virtual_memory(
        &self,
        device: &Self::DeviceState,
        kind: OwnedMemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<Self::VirtualMemory>, Error>;
    fn free_virtual_memory(memory: &mut Self::VirtualMemory) -> Result<(), Error>;
    /// An error must leave no usable mapping owner. If driver submission is
    /// ambiguous, the driver marks the reservation unusable before returning.
    fn map_virtual_device(
        memory: &Self::VirtualMemory,
        reservation: &Self::VirtualAddress,
        device: &Self::DeviceState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::VirtualDeviceMapping>, Error>;
    fn free_virtual_device_mapping(mapping: &mut Self::VirtualDeviceMapping) -> Result<(), Error>;
    /// An error must leave the process range unmapped or restored to its
    /// reservation state so the core may release tentative interval occupancy.
    fn map_virtual_host(
        memory: &Self::VirtualMemory,
        reservation: &Self::VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: crate::host_storage::Allocator,
    ) -> Result<Owned<Self::VirtualHostMapping>, Error>;
    fn free_virtual_host_mapping(mapping: &mut Self::VirtualHostMapping) -> Result<(), Error>;
}

/// Operations on one driver-owned, directly published GPU queue.
///
/// The resource retains its native transport state. The core separately
/// retains ring, signal, and scratch backing until native retirement is proved.
/// Its device-state type ties queue mapping to the creating GPU driver.
pub(crate) trait UserQueueResource: CachedInfo<Info = QueueTransport> {
    type DeviceState;
    fn check(&self) -> Result<(), Error>;
    fn progress(&self) -> Result<(u64, u64), Error>;
    fn inactivate(&mut self) -> Result<(), Error>;
    fn set_priority(
        &mut self,
        priority: crate::device::gpu::queue::QueuePriority,
    ) -> Result<(), Error>;
    fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error>;
    /// # Safety
    /// The command processor has stopped the queue. The caller retains the new
    /// scratch backing until a later successful replacement or driver
    /// destruction.
    #[allow(unsafe_code)]
    unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error>;
    /// # Safety
    /// Producers and published transport mappings have been retired.
    #[allow(unsafe_code)]
    unsafe fn destroy(&mut self) -> Result<(), Error>;
    fn map_device(&self, device: &Self::DeviceState) -> Result<QueueTransport, Error>;
}

/// Operations on one kernel-mediated GPU submission queue.
///
/// The resource tracks accepted submissions and their completion. Failed or
/// ambiguous teardown must keep command backing reachable for retry.
pub(crate) trait KernelQueueResource {
    /// # Safety
    /// The command range remains device-accessible, executable, and unchanged
    /// until retirement or conclusive driver teardown. An ambiguous driver
    /// outcome must be returned as an accepted submission; `Err` proves that
    /// the command was rejected.
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error>;
    fn status(&self) -> KernelQueueStatus;
    fn refresh_status(&self) -> Result<KernelQueueStatus, Error>;
    fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error>;
    fn destroy(&mut self) -> Result<(), Error>;
}

/// Complete GPU-family contract shared across operating-system drivers.
///
/// Every supported GPU driver supplies allocation, virtual memory, and both
/// queue models. Hardware-dependent operations can report `Unsupported` after
/// checking the activated device's capabilities. NPU drivers do not implement
/// this trait merely because they support a shared memory contract.
pub(crate) trait GpuDriver: Driver + AllocationOperations + VirtualMemoryOperations {
    /// Direct GPU queue retained by this driver's activated device state.
    type UserQueue: UserQueueResource<DeviceState = Self::DeviceState>;
    /// Kernel-mediated submission queue created by this same driver.
    type KernelQueue: KernelQueueResource;
    fn supports_expert_scheduling(&self, device: &Self::DeviceState) -> Result<bool, Error>;
    fn available_sdma_engines(&self, device: &Self::DeviceState) -> Result<u32, Error>;
    /// # Safety
    /// The caller retains all raw signal and scratch addresses that the GPU
    /// may reach, including after an ambiguous driver creation result.
    #[allow(unsafe_code)]
    unsafe fn create_user_queue(
        &self,
        device: &Self::DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<Self::UserQueue>, Error>;
    fn create_kernel_queue(
        &self,
        device: &Self::DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<Self::KernelQueue>, Error>;
    fn gpu_presentation(&self, endpoint: &Endpoint) -> GpuPresentation;
    fn set_persisting_l2_cache_size(
        &self,
        device: &Self::DeviceState,
        size_bytes: u32,
    ) -> Result<(), Error>;
    fn available_memory(&self, device: &Self::DeviceState) -> Result<u64, Error>;
    /// Allocates GPU queue scratch with the normal allocation cleanup contract.
    fn allocate_queue_scratch(
        &self,
        device: &Self::DeviceState,
        size: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;
    /// Maps the GPU queue MMIO transport page as an owned allocation.
    fn map_mmio_remap(&self, device: &Self::DeviceState) -> Result<Owned<Self::Allocation>, Error>;
    fn clock_counters(&self, device: &Self::DeviceState) -> Result<ClockCounters, Error>;
    /// # Safety
    /// Handler code and argument storage remain backed while traps can reach
    /// them, including after an ambiguous driver update.
    #[allow(unsafe_code)]
    unsafe fn set_trap_handler(
        &self,
        device: &Self::DeviceState,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error>;
    fn spm_acquire(&self, device: &Self::DeviceState) -> Result<(), Error>;
    fn spm_release(&self, device: &Self::DeviceState) -> Result<(), Error>;
    /// # Safety
    /// Previous and new writable destinations remain live and exclude
    /// conflicting access until replacement, unset, or conclusive teardown.
    #[allow(unsafe_code)]
    unsafe fn spm_set_destination(
        &self,
        device: &Self::DeviceState,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error>;
}
