// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private interface implemented by each concrete device driver.
//!
//! [`Driver`] owns discovery, activation, and shutdown. GPU capabilities are
//! separate traits so each backend implements only the services it provides.
//! Allocation and virtual-memory operations remain separate because their
//! owners can be shared with other device families. Queue creation belongs to
//! the matching GPU capability; queue operations belong to the returned owner.

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
    DeviceDriverState, DriverInstance, DriverInstanceId, EndpointSelector, EventSubscription,
    GpuDriverRef, KfdAccessTarget, KfdAllocation, KfdDeviceState, new_driver_instance,
};
pub(crate) use linux_kfd::KfdEventSubscription;
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) use linux_kfd::{
    KfdKernelQueue, KfdPcSampling, KfdQueue, KfdSignalEvent, LinuxKfdDriver as KfdDriver,
};
pub(crate) use resources::{
    AllocationState, KernelQueueState, PcSamplingState, QueueState, VirtualAddressState,
    VirtualDeviceMappingState, VirtualHostMappingState, VirtualMemoryState,
};

use crate::device::gpu::kernel_queue::{
    KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait,
};
use crate::device::gpu::profiling::{
    ClockCounters, PcSamplingConfiguration, PcSamplingConfigurationSet, PcSamplingRequest,
};
use crate::device::gpu::queue::{QueueRequest, QueueScratch, QueueTransport};
use crate::memory::storage::{Owned, Shared};
use crate::memory::{
    AllocationInfo, DeviceAccess, HostRegistration, MemoryKind, VirtualAddressInfo,
    VirtualMemoryInfo,
};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, NativeNodeId, NodeKind, SourceKind, TopologySnapshot};
use crate::{Error, ErrorKind};

/// Inert constructors for the driver backends compiled into this target.
/// Each session constructs fresh instances from this immutable catalog.
pub(crate) type DriverFactory = fn(DriverContextLifetime) -> Result<DriverInstance, Error>;

pub(crate) const BUILTIN_DRIVER_FACTORIES: &[DriverFactory] = &[
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    new_linux_kfd,
];

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn new_linux_kfd(lifetime: DriverContextLifetime) -> Result<DriverInstance, Error> {
    Ok(DriverInstance::LinuxKfd(Shared::new(
        KfdDriver::with_context_lifetime(lifetime),
    )?))
}

/// Address-space identity and bounds for devices that can map memory.
///
/// The core uses `shares_address_domain` to serialize overlapping mappings
/// within one device address space. Equal domains must report compatible
/// bounds. Mapping owners retain cloned state and share it across threads.
/// These cached facts do not establish access to a range or report device loss.
pub(crate) trait AddressSpaceInfo: Clone + Send + Sync {
    /// Returns the inclusive address bounds captured at activation.
    fn address_range(&self) -> (u64, u64);
    /// Reports whether two devices use the same native address space.
    fn shares_address_domain(&self, other: &Self) -> bool;
}

/// Copies immutable facts cached by a driver-owned resource.
///
/// A caller takes the snapshot while the resource owner is live. The value is
/// information, not a fresh driver query or permission to use an address after
/// that owner has been released. Cleanup may invalidate the cached addresses.
pub(crate) trait CachedInfo {
    /// Copyable facts retained by the resource owner.
    type Info: Copy;
    /// Copies those facts without querying the driver again.
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
/// callers can open an endpoint by ID alone. The session checks duplicates
/// within and across drivers before publishing an enumeration.
/// Shutdown may fail after partial cleanup and must permit a retry with the
/// same instance. Capability traits add resource operations only when a caller
/// needs them.
pub(crate) trait Driver: Send + Sync {
    /// State retained for one endpoint activated through this driver instance.
    type DeviceState: Clone + Send + Sync;
    /// Returns the identity stamped into this driver's passive endpoints.
    fn driver_instance(&self) -> DriverInstanceId;
    /// Returns the context policy fixed when this driver was constructed.
    fn context_lifetime(&self) -> DriverContextLifetime;
    /// Releases acquired context state; a failure must permit a later retry.
    fn shutdown(&mut self) -> Result<(), Error>;
    /// Visits discovered endpoints without acquiring an execution context.
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error>;
    /// Adds one generation-consistent contributor to a candidate snapshot.
    /// Generic drivers report one opaque node per endpoint until they expose
    /// a richer passive topology source.
    fn contribute_topology(&self, snapshot: &mut TopologySnapshot) -> Result<(), Error> {
        let source =
            snapshot.add_source(SourceKind::Driver, None, true, Some(self.driver_instance()))?;
        let mut ordinal = 0_u64;
        self.enumerate(&mut |endpoint| {
            let node = snapshot.add_node(source, NativeNodeId::Driver(ordinal), NodeKind::Other)?;
            ordinal = ordinal.checked_add(1).ok_or(Error::Operation {
                kind: ErrorKind::ResourceExhausted,
                detail: "too many driver topology nodes",
            })?;
            snapshot.add_endpoint(endpoint, node)
        })
    }
    /// Driver-qualified access policy for a passive endpoint pair. A mapping
    /// operation remains authoritative for a particular allocation.
    fn can_access_local_memory(&self, consumer: &Endpoint, owner: &Endpoint) -> bool {
        consumer.id == owner.id
            && consumer.driver_instance == self.driver_instance()
            && owner.driver_instance == self.driver_instance()
    }
    /// Re-reads an endpoint by its driver-assigned ID without activating it.
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error>;
    /// Resolves an ID without enumerating this driver's other endpoints.
    /// `None` means this driver does not recognize the ID; a recognized but
    /// stale ID remains an error from `open_endpoint`.
    fn probe_endpoint(&self, id: [u8; 16]) -> Result<Option<Endpoint>, Error> {
        self.open_endpoint(id).map(Some)
    }
    /// Populates optional endpoint and kind-specific information from
    /// additional sources without activating the endpoint. Drivers with no
    /// additional source leave the endpoint unchanged.
    fn populate_endpoint(&self, _endpoint: &mut Endpoint) -> Result<(), Error> {
        Ok(())
    }
    /// Revalidates an endpoint and acquires its device state.
    fn activate(&self, endpoint: &Endpoint) -> Result<Self::DeviceState, Error>;
}

/// Allocation, registration, and access operations on activated devices.
///
/// An allocation owner retains every native mapping dependency needed for
/// cleanup. `free_allocation` must leave failed cleanup retryable. Registered
/// host pages have an additional caller lifetime obligation stated below.
pub(crate) trait AllocationOperations: Driver {
    /// Native allocation and mapping state retained through cleanup.
    type Allocation: CachedInfo<Info = AllocationInfo>;
    /// Reports whether this endpoint can register caller-owned host pages.
    fn supports_host_registration(&self, endpoint: &Endpoint) -> bool;
    /// Checks that an allocation's native backing remains usable.
    fn check_allocation(allocation: &Self::Allocation) -> Result<(), Error>;
    /// Reports whether the allocation uses device-local backing.
    fn allocation_is_device_local(allocation: &Self::Allocation) -> bool;
    /// Replaces device access without retaining the input slice.
    fn set_allocation_access(
        _allocation: &mut Self::Allocation,
        _devices: &[&Self::DeviceState],
    ) -> Result<(), Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "driver does not support changing allocation access",
        })
    }
    /// Returns the allocation address visible to a selected device.
    fn allocation_device_address(
        allocation: &Self::Allocation,
        device: &Self::DeviceState,
    ) -> Result<u64, Error>;
    /// Checks whether this device owns the allocation's native backing.
    fn allocation_is_owned_by(allocation: &Self::Allocation, device: &Self::DeviceState) -> bool;
    /// A successful release is idempotent. After failure, a later call must
    /// safely resume cleanup or report terminal uncertainty without replaying
    /// an ambiguous native operation. The generic owner also calls this on Drop.
    fn free_allocation(allocation: &mut Self::Allocation) -> Result<(), Error>;
    /// Allocates backing and establishes the requested device access.
    fn allocate_owned(
        &self,
        device: &Self::DeviceState,
        peers: &[&Self::DeviceState],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error>;
    /// Registers caller or rocddi-owned host pages with the selected device.
    ///
    /// # Safety
    /// The caller keeps borrowed pages mapped through successful free or,
    /// after ambiguous driver failure, process exit. When `request.backing` is
    /// present, the driver instead takes that lease into its native owner and
    /// retains it through uncertain cleanup.
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
    /// Native process address reservation retained until every mapping ends.
    type VirtualAddress: CachedInfo<Info = VirtualAddressInfo>;
    /// Mapping of physical backing into a device address space.
    type VirtualDeviceMapping;
    /// Mapping of physical backing into the host address space.
    type VirtualHostMapping;
    /// Driver-owned physical backing used by one or more mappings.
    type VirtualMemory: CachedInfo<Info = VirtualMemoryInfo>;
    /// Reserves an address range within the caller's permitted bounds.
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<Self::VirtualAddress>, Error>;
    /// Releases a reservation after its mappings are gone.
    fn free_virtual_address(address: &mut Self::VirtualAddress) -> Result<(), Error>;
    /// Creates physical backing without mapping it into a device or host.
    fn create_virtual_memory(
        &self,
        device: &Self::DeviceState,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<Self::VirtualMemory>, Error>;
    /// Releases physical backing after its mappings are gone.
    fn free_virtual_memory(memory: &mut Self::VirtualMemory) -> Result<(), Error>;
    /// Maps backing into a device address space.
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
    /// Unmaps one device range, retaining its owner if cleanup fails.
    fn free_virtual_device_mapping(mapping: &mut Self::VirtualDeviceMapping) -> Result<(), Error>;
    /// Maps backing into the host process address space.
    /// An error must leave the process range unmapped or restored to its
    /// reservation state so the core may release tentative interval occupancy.
    fn map_virtual_host(
        memory: &Self::VirtualMemory,
        reservation: &Self::VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::VirtualHostMapping>, Error>;
    /// Unmaps one host range, retaining its owner if cleanup fails.
    fn free_virtual_host_mapping(mapping: &mut Self::VirtualHostMapping) -> Result<(), Error>;
}

/// Operations on one driver-owned, directly published GPU queue.
///
/// The resource retains its native transport state. The core separately
/// retains ring, signal, and scratch backing until native retirement is proved.
/// Its device-state type ties queue mapping to the creating GPU driver.
pub(crate) trait UserQueueResource: CachedInfo<Info = QueueTransport> {
    /// Activated state required to map this queue for a producer.
    type DeviceState;
    /// Checks whether the native queue is still usable.
    fn check(&self) -> Result<(), Error>;
    /// Samples the native read and write positions without retiring work.
    fn progress(&self) -> Result<(u64, u64), Error>;
    /// Stops native execution while retaining the queue's backing.
    fn inactivate(&mut self) -> Result<(), Error>;
    /// Changes the native scheduling priority when supported.
    fn set_priority(
        &mut self,
        priority: crate::device::gpu::queue::QueuePriority,
    ) -> Result<(), Error>;
    /// Replaces the set of compute units available to this queue.
    fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error>;
    /// Replaces queue scratch while retaining uncertain old backing.
    ///
    /// # Safety
    /// The command processor has stopped the queue. The caller retains the new
    /// scratch backing until a later successful replacement or driver
    /// destruction.
    #[allow(unsafe_code)]
    unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error>;
    /// Destroys the native queue and releases its transport.
    ///
    /// # Safety
    /// Producers and published transport mappings have been retired.
    #[allow(unsafe_code)]
    unsafe fn destroy(&mut self) -> Result<(), Error>;
    /// Runs once after native queue retirement while the public control
    /// mapping is still live. A failed retirement does not run the callback.
    ///
    /// # Safety
    /// Producers and published transport mappings have been retired. The
    /// callback may release external backing only after native retirement.
    #[allow(unsafe_code)]
    unsafe fn destroy_with_retained_transport(
        &mut self,
        callback: &mut dyn FnMut(),
    ) -> Result<(), Error>;
    /// Maps this queue's transport into another compatible device view.
    fn map_device(&self, device: &Self::DeviceState) -> Result<QueueTransport, Error>;
}

/// Operations on one kernel-mediated GPU submission queue.
///
/// The resource tracks accepted submissions and their completion. Failed or
/// ambiguous teardown must keep command backing reachable for retry.
pub(crate) trait KernelQueueResource {
    /// Submits a command and returns the accepted submission identity.
    ///
    /// # Safety
    /// The command range remains device-accessible, executable, and unchanged
    /// until retirement or conclusive driver teardown. An ambiguous driver
    /// outcome must be returned as an accepted submission; `Err` proves that
    /// the command was rejected.
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error>;
    /// Returns the last status already observed by this owner.
    fn status(&self) -> KernelQueueStatus;
    /// Queries the driver for current completion and failure status.
    fn refresh_status(&self) -> Result<KernelQueueStatus, Error>;
    /// Waits for a particular accepted submission to retire or time out.
    fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error>;
    /// Releases the native queue after proving accepted submissions retired.
    /// Uncertain cleanup keeps the native backing reachable for retry.
    fn destroy(&mut self) -> Result<(), Error>;
}

/// Native registration state for one PC sampler.
pub(crate) trait PcSamplingResource {
    /// Begins collecting samples with this native registration.
    fn start(&mut self) -> Result<(), Error>;
    /// Stops collection while retaining the registration.
    fn stop(&mut self) -> Result<(), Error>;
    /// Releases the registration after sampling has stopped.
    fn destroy(&mut self) -> Result<(), Error>;
    /// Returns an unstarted adopted registration to its original owner after
    /// a frontend fails to complete activation.
    fn relinquish_adopted(&mut self) -> Result<(), Error>;
}

/// GPU queries and controls shared across operating-system drivers.
///
/// Allocation, virtual memory, queue, and instrumentation capabilities are
/// independent. A GPU backend need not implement them to expose these queries.
pub(crate) trait GpuDriver: Driver {
    /// Queries the process-wide XNACK mode without changing it.
    fn xnack_enabled(&self, device: &Self::DeviceState) -> Result<bool, Error>;
    /// Sets the native VM's persisting L2 reservation in bytes.
    fn set_persisting_l2_cache_size(
        &self,
        device: &Self::DeviceState,
        size_bytes: u32,
    ) -> Result<(), Error>;
    /// Queries currently available device memory in bytes.
    fn available_memory(&self, device: &Self::DeviceState) -> Result<u64, Error>;
    /// Returns the layout capabilities needed by GPU image consumers.
    fn surface_config(
        &self,
        device: &Self::DeviceState,
    ) -> Result<crate::device::gpu::GpuSurfaceConfig, Error>;
}

/// Direct GPU queues and their queue-specific memory resources.
pub(crate) trait GpuUserQueueOperations: GpuDriver + AllocationOperations {
    /// Direct GPU queue retained by this driver's activated device state.
    type UserQueue: UserQueueResource<DeviceState = Self::DeviceState>;
    /// Reports whether this driver can use expert scheduling on the device.
    fn supports_expert_scheduling(&self, device: &Self::DeviceState) -> Result<bool, Error>;
    /// Creates a directly published GPU queue and retains its native backing.
    ///
    /// # Safety
    /// The caller retains all raw signal and scratch addresses that the GPU
    /// may reach, including after an ambiguous driver creation result.
    #[allow(unsafe_code)]
    unsafe fn create_user_queue(
        &self,
        device: &Self::DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<Self::UserQueue>, Error>;
    /// Allocates GPU queue scratch with the normal allocation cleanup contract.
    fn allocate_queue_scratch(
        &self,
        device: &Self::DeviceState,
        size: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;
    /// Maps the GPU queue MMIO transport page as an owned allocation.
    fn map_mmio_remap(&self, device: &Self::DeviceState) -> Result<Owned<Self::Allocation>, Error>;
}

/// Kernel-mediated GPU submission queues, independent of direct queues.
pub(crate) trait GpuKernelQueueOperations: GpuDriver {
    /// Native queue that accepts driver-mediated command submissions.
    type KernelQueue: KernelQueueResource;
    /// Returns the number of native SDMA engines available for submission.
    fn available_sdma_engines(&self, device: &Self::DeviceState) -> Result<u32, Error>;
    /// Creates a driver-mediated queue for the requested command format.
    fn create_kernel_queue(
        &self,
        device: &Self::DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<Self::KernelQueue>, Error>;
}

/// GPU counters, sampling, stream monitoring, and trap instrumentation.
pub(crate) trait GpuInstrumentationOperations: GpuDriver {
    /// Native PC sampling registration retained by this driver instance.
    type PcSampling: PcSamplingResource;
    /// Reads correlated GPU and system clock counters.
    fn clock_counters(&self, device: &Self::DeviceState) -> Result<ClockCounters, Error>;
    /// Lists supported or currently available native PC sampling modes.
    fn pc_sampling_configurations(
        &self,
        device: &Self::DeviceState,
        set: PcSamplingConfigurationSet,
    ) -> Result<Vec<PcSamplingConfiguration>, Error>;
    /// Creates a PC sampling registration, optionally reusing a trace ID.
    fn create_pc_sampling(
        &self,
        device: &Self::DeviceState,
        request: PcSamplingRequest,
        existing_trace_id: Option<std::num::NonZeroU32>,
    ) -> Result<Owned<Self::PcSampling>, Error>;
    /// Installs handler code and memory addresses for GPU traps.
    ///
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
    /// Acquires exclusive access to stream performance monitoring.
    fn spm_acquire(&self, device: &Self::DeviceState) -> Result<(), Error>;
    /// Releases the stream performance monitor acquired by this owner.
    fn spm_release(&self, device: &Self::DeviceState) -> Result<(), Error>;
    /// Replaces the GPU-writable destination for stream monitor records.
    ///
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
