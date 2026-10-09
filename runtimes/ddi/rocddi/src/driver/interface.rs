// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private driver capabilities and their resource contracts.
//!
//! A session can install multiple concrete drivers. Each driver implements
//! the common discovery and lifecycle contract; generic resource owners depend
//! only on the capability whose operations they call. Each resource type is
//! declared beside its creation and cleanup operations. A driver can implement
//! an independent capability without supplying unrelated resources.

use crate::host_storage::Owned;
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::memory::{
    AllocationInfo, DeviceAccess, HostAllocationInfo, HostRegistration, OwnedMemoryKind,
    VirtualAddressInfo, VirtualMemoryInfo,
};
use crate::profiling::ClockCounters;
use crate::queue::{QueueRequest, QueueScratch, QueueTransport};
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

/// Shares one activated-device type across independent capabilities.
///
/// The type carries the driver's live device dependencies. Separating this
/// association from [`Driver`] lets a memory or queue owner depend on its
/// required operations without also requiring discovery and shutdown. Resource
/// owners clone the state and can move it between threads.
pub(crate) trait DeviceStateType {
    type DeviceState: Clone + Send + Sync;
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
pub(crate) trait Driver: DeviceStateType + Send + Sync {
    fn driver_instance(&self) -> u64;
    fn context_lifetime(&self) -> DriverContextLifetime;
    fn shutdown(&mut self) -> Result<(), Error>;
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error>;
    fn activate(&self, endpoint: &Endpoint) -> Result<Self::DeviceState, Error>;
}

/// GPU presentation and activated GPU cache control.
///
/// CPU and other non-GPU drivers do not implement this capability. Presentation
/// reads a passive endpoint; cache control requires activated device state.
pub(crate) trait GpuDriver: Driver {
    fn gpu_presentation(&self, endpoint: &Endpoint) -> GpuPresentation;
    fn set_persisting_l2_cache_size(
        &self,
        device: &Self::DeviceState,
        size_bytes: u32,
    ) -> Result<(), Error>;
    fn available_memory(&self, device: &Self::DeviceState) -> Result<u64, Error>;
}

/// Host-only allocation and host-memory maintenance operations.
///
/// Host storage does not require device activation. The returned owner keeps
/// its allocation and cleanup state; a failed `free_host` leaves that state
/// available for retry. Cache control acts on a caller-validated live range.
pub(crate) trait HostMemoryDriver: Send + Sync {
    type HostAllocation: CachedInfo<Info = HostAllocationInfo>;
    fn allocate_host(
        &self,
        size: u64,
        alignment: u64,
    ) -> Result<Owned<Self::HostAllocation>, Error>;
    fn free_host(allocation: &mut Self::HostAllocation) -> Result<(), Error>;
    fn host_page_size() -> Result<u64, Error>;
    fn host_cache_line_size() -> Result<u32, Error>;
    /// # Safety
    /// Every intersecting cache line, including bytes outside the requested
    /// range, remains mapped through the call. The caller synchronizes access
    /// to those complete lines and prevents concurrent unmapping.
    #[allow(unsafe_code)]
    unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error>;
}

/// Allocation, registration, and access operations on activated devices.
///
/// An allocation owner retains every native mapping dependency needed for
/// cleanup. `free_allocation` must leave failed cleanup retryable. Registered
/// host pages have an additional caller lifetime obligation stated below.
pub(crate) trait AllocationDriver: DeviceStateType + Send + Sync {
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

/// GPU queue scratch aperture and MMIO transport-page allocations.
///
/// These resources use the same allocation owner and cleanup contract as
/// ordinary device allocations. They remain separate because a driver with
/// general memory support need not provide GPU queue transport resources.
pub(crate) trait GpuQueueResourceDriver: AllocationDriver {
    fn allocate_queue_scratch(
        &self,
        device: &Self::DeviceState,
        size: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;
    fn map_mmio_remap(&self, device: &Self::DeviceState) -> Result<Owned<Self::Allocation>, Error>;
}

/// Address reservation, physical backing, and virtual mapping operations.
///
/// Reservations, backing, and each mapping have separate owners. The core
/// retains their dependencies through map and unmap; failed release must keep
/// the affected owner available for retry. A failed map must satisfy the
/// rollback or quarantine guarantees on the mapping methods below.
pub(crate) trait VirtualMemoryDriver: DeviceStateType + Send + Sync {
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

/// User-queue creation, transport mapping, progress, and teardown operations.
///
/// The queue owner retains its transport mappings and native state while the
/// core retains externally reachable ring, signal, and scratch backing.
/// Teardown must not release that backing until native retirement is proved.
pub(crate) trait UserQueueDriver: DeviceStateType + Send + Sync {
    type Queue: CachedInfo<Info = QueueTransport>;
    fn supports_expert_scheduling(&self, device: &Self::DeviceState) -> Result<bool, Error>;
    fn check_queue(queue: &Self::Queue) -> Result<(), Error>;
    fn queue_progress(queue: &Self::Queue) -> Result<(u64, u64), Error>;
    fn inactivate_queue(queue: &mut Self::Queue) -> Result<(), Error>;
    fn set_queue_priority(
        queue: &mut Self::Queue,
        priority: crate::queue::QueuePriority,
    ) -> Result<(), Error>;
    fn set_queue_cu_mask(queue: &mut Self::Queue, mask: &[u32]) -> Result<(), Error>;
    /// # Safety
    /// Firmware has stopped the queue, and the caller retains the new scratch
    /// backing until a later successful replacement or driver destruction.
    #[allow(unsafe_code)]
    unsafe fn set_queue_scratch(
        queue: &mut Self::Queue,
        scratch: QueueScratch,
    ) -> Result<(), Error>;
    /// # Safety
    /// Producers and published transport mappings have been retired.
    #[allow(unsafe_code)]
    unsafe fn destroy_queue(queue: &mut Self::Queue) -> Result<(), Error>;
    /// # Safety
    /// The caller retains all raw signal and scratch addresses that firmware
    /// may reach, including after an ambiguous driver creation result.
    #[allow(unsafe_code)]
    unsafe fn create_queue(
        &self,
        device: &Self::DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<Self::Queue>, Error>;
    fn map_queue(queue: &Self::Queue, device: &Self::DeviceState) -> Result<QueueTransport, Error>;
}

/// Kernel-mediated queue publication, submission, and retirement operations.
///
/// The kernel queue owner tracks accepted submissions and their completion.
/// Destroying it must keep command backing reachable if native teardown is
/// ambiguous or fails; callers may retry with the same owner.
pub(crate) trait KernelQueueDriver: DeviceStateType + Send + Sync {
    type KernelQueue;
    fn available_sdma_rings(&self, device: &Self::DeviceState) -> Result<u32, Error>;
    fn create_kernel_queue(
        &self,
        device: &Self::DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<Self::KernelQueue>, Error>;
    /// # Safety
    /// The command range remains device-accessible, executable, and unchanged
    /// until retirement or conclusive driver teardown. An ambiguous driver
    /// outcome must be returned as an accepted submission; `Err` proves that
    /// the command was rejected.
    #[allow(unsafe_code)]
    unsafe fn submit_kernel_queue(
        queue: &Self::KernelQueue,
        command: KernelCommand,
    ) -> Result<u64, Error>;
    fn kernel_queue_status(queue: &Self::KernelQueue) -> KernelQueueStatus;
    fn refresh_kernel_queue(queue: &Self::KernelQueue) -> Result<KernelQueueStatus, Error>;
    fn wait_kernel_queue(
        queue: &Self::KernelQueue,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error>;
    fn destroy_kernel_queue(queue: &mut Self::KernelQueue) -> Result<(), Error>;
}

/// GPU-only timing, trap, and stream-monitor services.
///
/// Trap and monitor destinations may be reached asynchronously after a driver
/// call. The caller must retain their backing until replacement or teardown
/// has conclusively removed that reachability.
pub(crate) trait GpuProfilingDriver: GpuDriver {
    fn clock_counters(&self, device: &Self::DeviceState) -> Result<ClockCounters, Error>;
    /// # Safety
    /// The handler code and argument storage remain backed and valid while
    /// traps can reach them, including after an ambiguous driver update.
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
    /// The previous and new writable destinations remain live and exclude
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
