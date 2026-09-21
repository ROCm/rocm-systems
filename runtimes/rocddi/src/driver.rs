//! Native control used by the implementation-neutral rocddi core.
//!
//! Discovery, activation, allocation, queue control, and host maintenance enter
//! through domain-specific capability traits. Cached metadata queries stay in
//! the core and need no
//! native call. Resource owners carry their concrete cleanup state so a failed
//! destruction can resume without replaying released IDs or requiring a global
//! registry. These are internal implementation boundaries, not a second ABI.
mod builtin;
#[cfg(target_os = "linux")]
pub(crate) mod linux_interop;
use crate::Error;
use crate::host_storage::Owned;
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::memory::{DeviceAccess, MemoryKind};
use crate::profiling::{ClockCounters, PcSamplingConfiguration};
use crate::queue::{QueueRequest, QueueScratch, QueueTransport};
use crate::session::SessionLifetime;
use crate::topology::Endpoint;
#[cfg(target_os = "linux")]
pub(crate) use builtin::NativeSignalEvent;
pub(crate) use builtin::{
    DeviceState, EndpointSelector, NativeAllocation, NativeHostAllocation, NativeKernelQueue,
    NativePcSampling, NativeQueue, NativeVirtualAddress, NativeVirtualDeviceMapping,
    NativeVirtualHostMapping, NativeVirtualMemory, PlatformDriver,
};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) trait ProviderDriver: Send + Sync {
    fn provider_instance(&self) -> u64;
    fn supports_host_registration(&self, endpoint: &Endpoint, lifetime: SessionLifetime) -> bool;
    fn shutdown(&mut self) -> Result<(), Error>;
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error>;
    fn activate(
        &self,
        endpoint: &Endpoint,
        lifetime: SessionLifetime,
    ) -> Result<DeviceState, Error>;
}

pub(crate) trait HostDriver: Send + Sync {
    fn allocate_host(
        &self,
        size: u64,
        alignment: u64,
    ) -> Result<Owned<NativeHostAllocation>, Error>;
    fn free_host(allocation: &mut NativeHostAllocation) -> Result<(), Error>;
    fn host_page_size() -> Result<u64, Error>;
    fn host_cache_line_size() -> Result<u32, Error>;
    #[allow(unsafe_code)]
    unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error>;
}

pub(crate) trait AllocationDriver: Send + Sync {
    fn check_allocation(allocation: &NativeAllocation) -> Result<(), Error>;
    fn allocation_device_address(
        allocation: &NativeAllocation,
        device: &DeviceState,
    ) -> Result<u64, Error>;
    fn allocation_is_owned_by(allocation: &NativeAllocation, device: &DeviceState) -> bool;
    fn allocate_queue_scratch(
        &self,
        device: &DeviceState,
        size: u64,
    ) -> Result<Owned<NativeAllocation>, Error>;
    fn map_mmio_remap(&self, device: &DeviceState) -> Result<Owned<NativeAllocation>, Error>;
    fn free_allocation(allocation: &mut NativeAllocation) -> Result<(), Error>;
    fn allocate(
        &self,
        device: &DeviceState,
        peers: &[&DeviceState],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error>;
}

pub(crate) trait VirtualMemoryDriver: Send + Sync {
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<NativeVirtualAddress>, Error>;
    fn free_virtual_address(address: &mut NativeVirtualAddress) -> Result<(), Error>;
    fn create_virtual_memory(
        &self,
        device: &DeviceState,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<NativeVirtualMemory>, Error>;
    fn free_virtual_memory(memory: &mut NativeVirtualMemory) -> Result<(), Error>;
    fn map_virtual_device(
        memory: &NativeVirtualMemory,
        reservation: &NativeVirtualAddress,
        device: &DeviceState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeVirtualDeviceMapping>, Error>;
    fn free_virtual_device_mapping(mapping: &mut NativeVirtualDeviceMapping) -> Result<(), Error>;
    fn map_virtual_host(
        memory: &NativeVirtualMemory,
        reservation: &NativeVirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: crate::host_storage::Allocator,
    ) -> Result<Owned<NativeVirtualHostMapping>, Error>;
    fn free_virtual_host_mapping(mapping: &mut NativeVirtualHostMapping) -> Result<(), Error>;
}

pub(crate) trait QueueDriver: Send + Sync {
    fn check_queue(queue: &NativeQueue) -> Result<(), Error>;
    fn queue_progress(queue: &NativeQueue) -> Result<(u64, u64), Error>;
    fn inactivate_queue(queue: &mut NativeQueue) -> Result<(), Error>;
    fn set_queue_priority(
        queue: &mut NativeQueue,
        priority: crate::queue::QueuePriority,
    ) -> Result<(), Error>;
    fn set_queue_cu_mask(queue: &mut NativeQueue, mask: &[u32]) -> Result<(), Error>;
    fn set_queue_scratch(queue: &mut NativeQueue, scratch: QueueScratch) -> Result<(), Error>;
    fn destroy_queue(queue: &mut NativeQueue) -> Result<(), Error>;
    fn create_queue(
        &self,
        device: &DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<NativeQueue>, Error>;
    fn map_queue(queue: &NativeQueue, device: &DeviceState) -> Result<QueueTransport, Error>;
}

pub(crate) trait KernelQueueDriver: Send + Sync {
    fn create_kernel_queue(
        &self,
        device: &DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<NativeKernelQueue>, Error>;
    fn submit_kernel_queue(queue: &NativeKernelQueue, command: KernelCommand)
    -> Result<u64, Error>;
    fn kernel_queue_status(queue: &NativeKernelQueue) -> KernelQueueStatus;
    fn wait_kernel_queue(
        queue: &NativeKernelQueue,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error>;
    fn destroy_kernel_queue(queue: &mut NativeKernelQueue) -> Result<(), Error>;
}

pub(crate) trait DeviceDriver: Send + Sync {
    fn check(&self, device: &DeviceState) -> Result<(), Error>;
    fn clock_counters(&self, device: &DeviceState) -> Result<ClockCounters, Error>;
    fn available_memory(&self, device: &DeviceState) -> Result<u64, Error>;
    fn set_trap_handler(
        &self,
        device: &DeviceState,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error>;
    fn spm_acquire(&self, device: &DeviceState) -> Result<(), Error>;
    fn spm_release(&self, device: &DeviceState) -> Result<(), Error>;
    fn spm_set_destination(
        &self,
        device: &DeviceState,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error>;
}

pub(crate) trait PcSamplingDriver: Send + Sync {
    fn pc_sampling_configurations(
        &self,
        device: &DeviceState,
    ) -> Result<crate::host_storage::Buffer<PcSamplingConfiguration>, Error>;
    fn create_pc_sampling(
        &self,
        device: &DeviceState,
        configuration: PcSamplingConfiguration,
        interval: u64,
    ) -> Result<Owned<NativePcSampling>, Error>;
    fn adopt_pc_sampling(
        &self,
        device: &DeviceState,
        trace_id: u32,
    ) -> Result<Owned<NativePcSampling>, Error>;
    fn pc_sampling_trace_id(sampling: &NativePcSampling) -> Result<u32, Error>;
    fn start_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error>;
    fn stop_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error>;
    fn destroy_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error>;
}

static NEXT_PROVIDER_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn new_provider_instance() -> u64 {
    NEXT_PROVIDER_INSTANCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .unwrap_or_else(|_| std::process::abort())
}
