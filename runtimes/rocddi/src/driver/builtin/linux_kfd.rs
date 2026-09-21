//! Session-owned Linux native control. Construction is inert; activation opens
//! KFD once for this session and retains each exact VM binding for recreation.
mod allocation;
mod drm;
mod event;
mod host;
mod imported_system;
mod kernel_queue;
mod memory;
mod pc_sampling;
mod process_identity;
mod queue;
mod registered_host;
mod sys;
mod sysfs;
mod uapi;
mod util;
mod vmem;

use crate::driver::{
    AllocationDriver, DeviceDriver, HostDriver, KernelQueueDriver, PcSamplingDriver,
    ProviderDriver, QueueDriver, VirtualMemoryDriver,
    linux_interop::{LinuxGpuEventDriver, LinuxMemoryInteropDriver},
};
use crate::event::GpuMemoryFault;
use crate::host_storage::{Allocator, Owned, Shared};
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::memory::interop::linux::{
    DmaBuf, KfdIpcMemoryHandle, KfdSvmAccess, KfdSvmAttribute, KfdSvmLocation,
};
use crate::memory::{AllocationDesc, AllocationLimits, DeviceAccess, MemoryKind};
use crate::profiling::{ClockCounters, PcSamplingConfiguration};
use crate::queue::{QueuePriority, QueueRequest, QueueScratch, QueueTransport};
use crate::session::SessionLifetime;
use crate::topology::Endpoint;
use crate::{Error, ErrorKind};
pub(crate) use allocation::NativeAllocation;
pub(crate) use event::KfdSignalEvent as NativeSignalEvent;
pub(crate) use host::HostAllocation as NativeHostAllocation;
pub(crate) use kernel_queue::KfdKernelQueue as NativeKernelQueue;
use memory::{error, native_error};
pub(crate) use pc_sampling::KfdPcSampling as NativePcSampling;
pub(crate) use queue::KfdQueue as NativeQueue;
use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, BorrowedFd, RawFd};
use std::sync::{Mutex, OnceLock};
pub(crate) use sysfs::NativeNode as EndpointSelector;
pub(crate) use vmem::{
    KfdVirtualAddress as NativeVirtualAddress,
    KfdVirtualDeviceMapping as NativeVirtualDeviceMapping,
    KfdVirtualHostMapping as NativeVirtualHostMapping, KfdVirtualMemory as NativeVirtualMemory,
};

/// Session-scoped Linux backend with lazy KFD activation.
///
/// Construction records policy and allocator state only. The first explicit GPU
/// activation opens KFD, establishes the retained VM bindings, and enables the
/// runtime under `initialization` so concurrent callers cannot publish partial
/// native state.
pub(crate) struct LinuxKfdDriver {
    allocator: Allocator,
    provider_instance: u64,
    process: u32,
    closing: bool,
    kfd: OnceLock<Shared<sys::Kfd>>,
    initialization: Mutex<()>,
    bindings: memory::VmBindings,
}

/// Lightweight session device reference retaining its activated VM.
#[derive(Clone)]
pub(crate) struct DeviceState {
    vm: Shared<memory::DeviceVm>,
    native: sysfs::NativeNode,
    lifetime: SessionLifetime,
}

impl DeviceState {
    pub(crate) fn address_range(&self) -> (u64, u64) {
        self.vm.address_range()
    }

    pub(crate) fn has_observed_loss(&self) -> bool {
        self.vm.has_observed_loss()
    }

    pub(crate) fn supports_system_dma_buf_import(&self) -> bool {
        self.vm.supports_system_dma_buf_import()
    }

    pub(crate) fn shares_vm(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.vm, &other.vm)
    }
}

impl LinuxKfdDriver {
    const TOPOLOGY_ROOT: &'static str = "/sys/class/kfd/kfd/topology";
    const DRM_ROOT: &'static str = "/sys/class/drm";
    pub(crate) fn new(allocator: Allocator) -> Self {
        Self {
            allocator,
            provider_instance: crate::driver::new_provider_instance(),
            process: std::process::id(),
            closing: false,
            kfd: OnceLock::new(),
            initialization: Mutex::new(()),
            bindings: memory::VmBindings::new(allocator),
        }
    }
    fn ensure_open(&self) -> Result<(), Error> {
        // Reject inherited instances before touching allocator callbacks,
        // filesystem state, or a mutex that may have been held across fork.
        util::check_process(self.process).map_err(|e| native_error("session process check", e))?;
        if self.closing {
            Err(error(
                ErrorKind::InvalidArgument,
                "session teardown already began",
            ))
        } else {
            Ok(())
        }
    }
    fn kfd(&self) -> Result<Shared<sys::Kfd>, Error> {
        self.ensure_open()?;
        if self.kfd.get().is_none() {
            let _guard = self
                .initialization
                .lock()
                .map_err(|_| error(ErrorKind::Internal, "KFD initialization lock poisoned"))?;
            if self.kfd.get().is_none() {
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open("/dev/kfd")
                    .map_err(|e| native_error("KFD endpoint open", e))?;
                let kfd = Shared::new(sys::Kfd::new(file, self.allocator), self.allocator)?;
                let _ = self.kfd.set(kfd);
            }
        }
        self.kfd
            .get()
            .cloned()
            .ok_or_else(|| error(ErrorKind::Internal, "KFD endpoint was not published"))
    }

    fn svm_location_to_native(&self, location: KfdSvmLocation) -> Result<u32, Error> {
        match location {
            KfdSvmLocation::System => Ok(uapi::SVM_LOCATION_SYSTEM),
            KfdSvmLocation::Undefined => Ok(uapi::SVM_LOCATION_UNDEFINED),
            KfdSvmLocation::Device(identity) => self.bindings.svm_gpu_id(identity),
        }
    }

    fn svm_location_from_native(&self, location: u32) -> Result<KfdSvmLocation, Error> {
        match location {
            uapi::SVM_LOCATION_SYSTEM => Ok(KfdSvmLocation::System),
            uapi::SVM_LOCATION_UNDEFINED => Ok(KfdSvmLocation::Undefined),
            gpu_id => self
                .bindings
                .svm_identity(gpu_id)
                .map(KfdSvmLocation::Device),
        }
    }

    fn encode_svm_attributes(
        &self,
        attributes: &[KfdSvmAttribute],
    ) -> Result<crate::host_storage::Buffer<uapi::SvmAttribute>, Error> {
        let mut native =
            crate::host_storage::Buffer::try_with_capacity(attributes.len(), self.allocator)?;
        for attribute in attributes {
            let encoded = match *attribute {
                KfdSvmAttribute::PreferredLocation(location) => uapi::SvmAttribute {
                    attribute_type: uapi::SVM_ATTR_PREFERRED_LOCATION,
                    value: self.svm_location_to_native(location)?,
                },
                KfdSvmAttribute::PrefetchLocation(location) => uapi::SvmAttribute {
                    attribute_type: uapi::SVM_ATTR_PREFETCH_LOCATION,
                    value: self.svm_location_to_native(location)?,
                },
                KfdSvmAttribute::Access { device, access } => uapi::SvmAttribute {
                    attribute_type: match access {
                        KfdSvmAccess::Accessible => uapi::SVM_ATTR_ACCESS,
                        KfdSvmAccess::AccessibleInPlace => uapi::SVM_ATTR_ACCESS_IN_PLACE,
                        KfdSvmAccess::NoAccess => uapi::SVM_ATTR_NO_ACCESS,
                    },
                    value: self.bindings.svm_gpu_id(device)?,
                },
                KfdSvmAttribute::SetFlags(value) => {
                    if value & !uapi::SVM_FLAGS != 0 {
                        return Err(error(
                            ErrorKind::InvalidArgument,
                            "SVM set-flags attribute contains unknown bits",
                        ));
                    }
                    uapi::SvmAttribute {
                        attribute_type: uapi::SVM_ATTR_SET_FLAGS,
                        value,
                    }
                }
                KfdSvmAttribute::ClearFlags(value) => {
                    if value & !uapi::SVM_FLAGS != 0 {
                        return Err(error(
                            ErrorKind::InvalidArgument,
                            "SVM clear-flags attribute contains unknown bits",
                        ));
                    }
                    uapi::SvmAttribute {
                        attribute_type: uapi::SVM_ATTR_CLEAR_FLAGS,
                        value,
                    }
                }
                KfdSvmAttribute::MigrationGranularity(value) => uapi::SvmAttribute {
                    attribute_type: uapi::SVM_ATTR_GRANULARITY,
                    value,
                },
            };
            native.try_push(encoded)?;
        }
        Ok(native)
    }

    fn decode_svm_attributes(
        &self,
        attributes: &mut [KfdSvmAttribute],
        native: &[uapi::SvmAttribute],
    ) -> Result<(), Error> {
        for (attribute, returned) in attributes.iter_mut().zip(native) {
            *attribute = match *attribute {
                KfdSvmAttribute::PreferredLocation(_) => {
                    if returned.attribute_type != uapi::SVM_ATTR_PREFERRED_LOCATION {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM preferred-location query type",
                        ));
                    }
                    KfdSvmAttribute::PreferredLocation(
                        self.svm_location_from_native(returned.value)?,
                    )
                }
                KfdSvmAttribute::PrefetchLocation(_) => {
                    if returned.attribute_type != uapi::SVM_ATTR_PREFETCH_LOCATION {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM prefetch-location query type",
                        ));
                    }
                    KfdSvmAttribute::PrefetchLocation(
                        self.svm_location_from_native(returned.value)?,
                    )
                }
                KfdSvmAttribute::Access { device, .. } => {
                    let access = match returned.attribute_type {
                        uapi::SVM_ATTR_ACCESS => KfdSvmAccess::Accessible,
                        uapi::SVM_ATTR_ACCESS_IN_PLACE => KfdSvmAccess::AccessibleInPlace,
                        uapi::SVM_ATTR_NO_ACCESS => KfdSvmAccess::NoAccess,
                        _ => {
                            return Err(error(
                                ErrorKind::DriverContract,
                                "KFD returned an invalid SVM access mode",
                            ));
                        }
                    };
                    KfdSvmAttribute::Access { device, access }
                }
                KfdSvmAttribute::SetFlags(_) => {
                    if returned.attribute_type != uapi::SVM_ATTR_SET_FLAGS {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM set-flags query type",
                        ));
                    }
                    KfdSvmAttribute::SetFlags(returned.value)
                }
                KfdSvmAttribute::ClearFlags(_) => {
                    if returned.attribute_type != uapi::SVM_ATTR_CLEAR_FLAGS {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM clear-flags query type",
                        ));
                    }
                    KfdSvmAttribute::ClearFlags(returned.value)
                }
                KfdSvmAttribute::MigrationGranularity(_) => {
                    if returned.attribute_type != uapi::SVM_ATTR_GRANULARITY {
                        return Err(error(
                            ErrorKind::DriverContract,
                            "KFD changed an SVM granularity query type",
                        ));
                    }
                    KfdSvmAttribute::MigrationGranularity(returned.value)
                }
            };
        }
        Ok(())
    }
}

impl ProviderDriver for LinuxKfdDriver {
    fn provider_instance(&self) -> u64 {
        self.provider_instance
    }
    fn supports_host_registration(&self, endpoint: &Endpoint, _lifetime: SessionLifetime) -> bool {
        endpoint.provider_instance == self.provider_instance && endpoint.gpu().is_some()
    }
    fn shutdown(&mut self) -> Result<(), Error> {
        util::check_process(self.process)
            .map_err(|e| native_error("session shutdown process check", e))?;
        self.closing = true;
        self.bindings.shutdown()?;
        if let Some(kfd) = self.kfd.get_mut() {
            // Retained native owners also retain this callback-allocated
            // endpoint. Preserve the session until that ownership is gone;
            // successful destruction ends the caller's allocator obligation.
            let endpoint = Shared::get_mut(kfd).ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "retained KFD endpoint dependencies prevent session destruction",
                )
            })?;
            endpoint
                .close()
                .map_err(|e| native_error("KFD endpoint close", e))?;
        }
        let _ = self.kfd.take();
        Ok(())
    }
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        sysfs::enumerate(
            Self::TOPOLOGY_ROOT,
            Self::DRM_ROOT,
            self.allocator,
            &mut |mut endpoint| {
                endpoint.provider_instance = self.provider_instance;
                visitor(endpoint)
            },
        )
    }
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.ensure_open()?;
        let mut endpoint =
            sysfs::open_endpoint(Self::TOPOLOGY_ROOT, Self::DRM_ROOT, id, self.allocator)?;
        endpoint.provider_instance = self.provider_instance;
        Ok(endpoint)
    }
    fn activate(
        &self,
        endpoint: &Endpoint,
        lifetime: SessionLifetime,
    ) -> Result<DeviceState, Error> {
        if endpoint.gpu().is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "the Linux KFD backend activates only GPU endpoints",
            ));
        }
        let current = self.open_endpoint(endpoint.id)?;
        if &current != endpoint {
            return Err(error(
                ErrorKind::DeviceLost,
                "endpoint metadata changed before activation",
            ));
        }
        if util::page_size().map_err(|e| native_error("native page size", e))? != 4096 {
            return Err(error(
                ErrorKind::Unsupported,
                "native implementation requires 4 KiB host pages",
            ));
        }
        let kfd = self.kfd()?;
        kfd.prepare_context(lifetime)
            .map_err(|source| native_error("KFD context selection", source))?;
        let vm = self.bindings.device(&kfd, &current.native)?;
        Ok(DeviceState {
            vm,
            native: current.native,
            lifetime,
        })
    }
}

impl HostDriver for LinuxKfdDriver {
    fn allocate_host(
        &self,
        size: u64,
        alignment: u64,
    ) -> Result<Owned<NativeHostAllocation>, Error> {
        self.ensure_open()?;
        process_identity::prepare_for_hot_checks();
        host::HostAllocation::create(size, alignment, self.allocator, self.process)
    }
    fn free_host(allocation: &mut NativeHostAllocation) -> Result<(), Error> {
        allocation.free()
    }
    fn host_page_size() -> Result<u64, Error> {
        util::page_size()
            .map(|size| size as u64)
            .map_err(|e| native_error("host page size", e))
    }
    fn host_cache_line_size() -> Result<u32, Error> {
        util::host_cache_line_size().map_err(|e| native_error("CPU cache recipe", e))
    }
    #[allow(unsafe_code)]
    unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error> {
        // SAFETY: The core caller guarantees the specified live mapping.
        unsafe { util::host_cache_control(pointer, length, line_size) }
            .map_err(|e| native_error("CPU cache control", e))
    }
}

impl AllocationDriver for LinuxKfdDriver {
    fn check_allocation(allocation: &NativeAllocation) -> Result<(), Error> {
        allocation.check()
    }
    fn allocation_device_address(
        allocation: &NativeAllocation,
        device: &DeviceState,
    ) -> Result<u64, Error> {
        allocation.device_address(&device.vm)
    }
    fn allocation_is_owned_by(allocation: &NativeAllocation, device: &DeviceState) -> bool {
        allocation.is_owned_by(&device.vm)
    }
    fn allocate_queue_scratch(
        &self,
        device: &DeviceState,
        size: u64,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        let desc = AllocationDesc {
            size,
            alignment: 4096,
        };
        let limits = AllocationLimits {
            alignment: 4096,
            granularity: 4096,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(desc) || size > device.native.local_memory_bytes {
            return Err(error(
                ErrorKind::ResourceExhausted,
                "invalid or unavailable native scratch extent",
            ));
        }
        NativeAllocation::create_scratch(&device.vm, desc)
    }
    fn map_mmio_remap(&self, device: &DeviceState) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        NativeAllocation::create_mmio(&device.vm)
    }
    fn free_allocation(allocation: &mut NativeAllocation) -> Result<(), Error> {
        allocation.free()
    }
    fn allocate(
        &self,
        device: &DeviceState,
        peers: &[&DeviceState],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error> {
        let desc = AllocationDesc { size, alignment };
        let limits = AllocationLimits {
            alignment: 4096,
            granularity: 4096,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(desc) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid native allocation extent or alignment",
            ));
        }
        if device.lifetime == SessionLifetime::Session && matches!(kind, MemoryKind::OwnedHost) {
            return Err(error(
                ErrorKind::Unsupported,
                "secondary KFD contexts cannot bind host-owned pages",
            ));
        }
        if let (SessionLifetime::Session, MemoryKind::RegisteredHost { address, uncached }) =
            (device.lifetime, kind)
        {
            return NativeAllocation::create_registered_host(
                device.vm.clone(),
                peers.iter().map(|peer| peer.vm.clone()),
                desc,
                address,
                permissions,
                uncached,
            );
        }
        let native_kind = match kind {
            MemoryKind::System => memory::BufferKind::Gtt,
            MemoryKind::OwnedHost => memory::BufferKind::OwnedUserptr { uncached: false },
            MemoryKind::RegisteredHost { address, uncached } => {
                memory::BufferKind::Userptr { address, uncached }
            }
            MemoryKind::DeviceLocal {
                host_visible,
                coherent,
                uncached,
                contiguous,
            } => {
                let available = if host_visible {
                    device.native.public_memory_bytes
                } else {
                    device.native.local_memory_bytes
                };
                if available == 0 {
                    return Err(error(
                        ErrorKind::Unsupported,
                        "requested local storage is unavailable",
                    ));
                }
                if size > available {
                    return Err(error(
                        ErrorKind::ResourceExhausted,
                        "allocation exceeds local memory capacity",
                    ));
                }
                memory::BufferKind::Vram {
                    public: host_visible,
                    coherent,
                    uncached,
                    contiguous,
                }
            }
        };
        NativeAllocation::create_with_peers(
            device.vm.clone(),
            peers.iter().map(|peer| peer.vm.clone()),
            desc,
            native_kind,
            permissions,
        )
    }
}

impl VirtualMemoryDriver for LinuxKfdDriver {
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<NativeVirtualAddress>, Error> {
        self.ensure_open()?;
        NativeVirtualAddress::reserve(bounds, size, alignment, address, self.allocator)
    }
    fn free_virtual_address(address: &mut NativeVirtualAddress) -> Result<(), Error> {
        address.free()
    }
    fn create_virtual_memory(
        &self,
        device: &DeviceState,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<NativeVirtualMemory>, Error> {
        self.ensure_open()?;
        NativeVirtualMemory::create(device.vm.clone(), kind, size, pinned, uncached)
    }
    fn free_virtual_memory(memory: &mut NativeVirtualMemory) -> Result<(), Error> {
        memory.free()
    }
    fn map_virtual_device(
        memory: &NativeVirtualMemory,
        reservation: &NativeVirtualAddress,
        device: &DeviceState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeVirtualDeviceMapping>, Error> {
        NativeVirtualDeviceMapping::create(
            memory,
            reservation,
            device.vm.clone(),
            address,
            offset,
            size,
            permissions,
        )
    }
    fn free_virtual_device_mapping(mapping: &mut NativeVirtualDeviceMapping) -> Result<(), Error> {
        mapping.free()
    }
    fn map_virtual_host(
        memory: &NativeVirtualMemory,
        reservation: &NativeVirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: Allocator,
    ) -> Result<Owned<NativeVirtualHostMapping>, Error> {
        NativeVirtualHostMapping::create(
            memory,
            reservation,
            address,
            offset,
            size,
            permissions,
            allocator,
        )
    }
    fn free_virtual_host_mapping(mapping: &mut NativeVirtualHostMapping) -> Result<(), Error> {
        mapping.free()
    }
}

impl QueueDriver for LinuxKfdDriver {
    fn check_queue(queue: &NativeQueue) -> Result<(), Error> {
        queue.check()
    }
    fn queue_progress(queue: &NativeQueue) -> Result<(u64, u64), Error> {
        queue.progress()
    }
    fn inactivate_queue(queue: &mut NativeQueue) -> Result<(), Error> {
        queue.inactivate()
    }
    fn set_queue_priority(queue: &mut NativeQueue, priority: QueuePriority) -> Result<(), Error> {
        queue.set_priority(priority)
    }
    fn set_queue_cu_mask(queue: &mut NativeQueue, mask: &[u32]) -> Result<(), Error> {
        queue.set_cu_mask(mask)
    }
    fn set_queue_scratch(queue: &mut NativeQueue, scratch: QueueScratch) -> Result<(), Error> {
        queue.set_scratch(scratch)
    }
    fn destroy_queue(queue: &mut NativeQueue) -> Result<(), Error> {
        queue.destroy()
    }
    fn create_queue(
        &self,
        device: &DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<NativeQueue>, Error> {
        queue::create(device.vm.clone(), &device.native, desc, device.lifetime)
    }
    fn map_queue(queue: &NativeQueue, device: &DeviceState) -> Result<QueueTransport, Error> {
        queue.map_device(device.vm.clone())
    }
}

impl KernelQueueDriver for LinuxKfdDriver {
    fn create_kernel_queue(
        &self,
        device: &DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<NativeKernelQueue>, Error> {
        self.ensure_open()?;
        if !cfg!(target_arch = "x86_64") || device.native.queues.gfx_target != 120_001 {
            return Err(error(
                ErrorKind::Unsupported,
                "kernel command submission is unqualified for this GPU target",
            ));
        }
        match format {
            KernelQueueFormat::Pm4 if !queue::supports_pm4(&device.native) => {
                return Err(error(
                    ErrorKind::Unsupported,
                    "PM4 kernel queue is unavailable",
                ));
            }
            KernelQueueFormat::Sdma if !device.native.queues.sdma_qualified => {
                return Err(error(
                    ErrorKind::Unsupported,
                    "SDMA kernel queue is unavailable",
                ));
            }
            _ => (),
        }
        NativeKernelQueue::create(device.vm.clone(), format)
    }

    fn submit_kernel_queue(
        queue: &NativeKernelQueue,
        command: KernelCommand,
    ) -> Result<u64, Error> {
        queue.submit(command)
    }

    fn kernel_queue_status(queue: &NativeKernelQueue) -> KernelQueueStatus {
        queue.status()
    }

    fn wait_kernel_queue(
        queue: &NativeKernelQueue,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        queue.wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
    }

    fn destroy_kernel_queue(queue: &mut NativeKernelQueue) -> Result<(), Error> {
        queue.destroy()
    }
}

impl DeviceDriver for LinuxKfdDriver {
    fn check(&self, device: &DeviceState) -> Result<(), Error> {
        device.vm.check()
    }
    fn clock_counters(&self, device: &DeviceState) -> Result<ClockCounters, Error> {
        self.ensure_open()?;
        let counters = device
            .vm
            .kfd()
            .clock_counters(device.native.gpu_id)
            .map_err(|source| native_error("KFD clock counter query", source))?;
        Ok(ClockCounters {
            gpu: counters.gpu_clock_counter,
            host: counters.cpu_clock_counter,
            system: counters.system_clock_counter,
            system_frequency: counters.system_clock_frequency,
        })
    }
    fn available_memory(&self, device: &DeviceState) -> Result<u64, Error> {
        self.ensure_open()?;
        device
            .vm
            .kfd()
            .available_memory(device.native.gpu_id)
            .map_err(|source| native_error("KFD available memory query", source))
    }
    fn set_trap_handler(
        &self,
        device: &DeviceState,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        device
            .vm
            .kfd()
            .set_trap_handler(device.native.gpu_id, handler_address, memory_address)
            .map_err(|source| native_error("KFD trap handler update", source))
    }
    fn spm_acquire(&self, device: &DeviceState) -> Result<(), Error> {
        let mut args = uapi::Spm {
            operation: uapi::SPM_OP_ACQUIRE,
            gpu_id: device.native.gpu_id,
            ..uapi::Spm::default()
        };
        device
            .vm
            .kfd()
            .spm(&mut args)
            .map_err(|source| native_error("KFD SPM acquire", source))
    }
    fn spm_release(&self, device: &DeviceState) -> Result<(), Error> {
        let mut args = uapi::Spm {
            operation: uapi::SPM_OP_RELEASE,
            gpu_id: device.native.gpu_id,
            ..uapi::Spm::default()
        };
        device
            .vm
            .kfd()
            .spm(&mut args)
            .map_err(|source| native_error("KFD SPM release", source))
    }
    fn spm_set_destination(
        &self,
        device: &DeviceState,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error> {
        let mut args = uapi::Spm {
            destination: destination.map_or(0, |address| address as u64),
            size,
            operation: uapi::SPM_OP_SET_DESTINATION,
            timeout: *timeout,
            gpu_id: device.native.gpu_id,
            bytes_copied: 0,
            has_data_loss: 0,
        };
        let result = device.vm.kfd().spm(&mut args);
        *timeout = args.timeout;
        *bytes_copied = args.bytes_copied;
        *data_loss = args.has_data_loss != 0;
        result.map_err(|source| native_error("KFD SPM destination update", source))
    }
}

impl PcSamplingDriver for LinuxKfdDriver {
    fn pc_sampling_configurations(
        &self,
        device: &DeviceState,
    ) -> Result<crate::host_storage::Buffer<PcSamplingConfiguration>, Error> {
        self.ensure_open()?;
        NativePcSampling::configurations(&self.kfd()?, device.native.gpu_id, self.allocator)
    }
    fn create_pc_sampling(
        &self,
        device: &DeviceState,
        configuration: PcSamplingConfiguration,
        interval: u64,
    ) -> Result<Owned<NativePcSampling>, Error> {
        self.ensure_open()?;
        NativePcSampling::create(
            self.kfd()?,
            device.native.gpu_id,
            configuration,
            interval,
            self.allocator,
        )
    }
    fn adopt_pc_sampling(
        &self,
        device: &DeviceState,
        trace_id: u32,
    ) -> Result<Owned<NativePcSampling>, Error> {
        self.ensure_open()?;
        NativePcSampling::adopt(self.kfd()?, device.native.gpu_id, trace_id, self.allocator)
    }
    fn pc_sampling_trace_id(sampling: &NativePcSampling) -> Result<u32, Error> {
        sampling.trace_id()
    }
    fn start_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error> {
        sampling.start()
    }
    fn stop_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error> {
        sampling.stop()
    }
    fn destroy_pc_sampling(sampling: &mut NativePcSampling) -> Result<(), Error> {
        sampling.destroy()
    }
}

impl LinuxMemoryInteropDriver for LinuxKfdDriver {
    fn supports_system_dma_buf_import(device: &DeviceState) -> bool {
        device.supports_system_dma_buf_import()
    }

    fn import_virtual_memory(
        &self,
        descriptor: BorrowedFd<'_>,
    ) -> Result<Owned<NativeVirtualMemory>, Error> {
        self.ensure_open()?;
        NativeVirtualMemory::import(descriptor.as_raw_fd(), self.allocator)
    }

    fn export_virtual_memory(memory: &NativeVirtualMemory) -> Result<DmaBuf, Error> {
        memory.export_dma_buf()
    }

    fn import_dma_buf(
        &self,
        device: &DeviceState,
        descriptor: BorrowedFd<'_>,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        NativeAllocation::import_dma_buf(
            device.vm.clone(),
            descriptor.as_raw_fd(),
            source_offset,
            byte_length,
            alignment,
            permissions,
        )
    }

    fn import_system_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: RawFd,
        source_offset: u64,
        byte_length: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        let (owner, peers) = devices.split_first().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "system import requires a device",
            )
        })?;
        if !peers.iter().all(|peer| owner.shares_vm(peer)) {
            return Err(error(
                ErrorKind::Unsupported,
                "system import requires one native GPU address domain",
            ));
        }
        NativeAllocation::import_system_dma_buf(
            owner.vm.clone(),
            descriptor,
            source_offset,
            byte_length,
            alignment,
            permissions,
        )
    }

    fn import_graphics_dma_buf(
        &self,
        devices: &[&DeviceState],
        descriptor: BorrowedFd<'_>,
        size_hint: u64,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        let (device, peers) = devices.split_first().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "graphics import requires at least one device",
            )
        })?;
        NativeAllocation::import_graphics_dma_buf(
            device.vm.clone(),
            peers.iter().map(|peer| peer.vm.clone()),
            descriptor.as_raw_fd(),
            size_hint,
        )
    }

    fn export_dma_buf(allocation: &NativeAllocation) -> Result<DmaBuf, Error> {
        allocation.export_dma_buf()
    }

    fn import_kfd_ipc_memory(
        &self,
        devices: &[&DeviceState],
        mapping_devices: &[&DeviceState],
        handle: KfdIpcMemoryHandle,
        size: u64,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        let words = handle.words();
        let owner = devices
            .iter()
            .find(|device| device.vm.gpu_id() == words[7])
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "IPC exporting GPU is unavailable in this session",
                )
            })?;
        NativeAllocation::import_ipc(
            owner.vm.clone(),
            mapping_devices.iter().map(|device| device.vm.clone()),
            words,
            size,
        )
    }

    fn export_kfd_ipc_memory(allocation: &NativeAllocation) -> Result<KfdIpcMemoryHandle, Error> {
        allocation.export_ipc_memory()
    }

    fn set_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &[KfdSvmAttribute],
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if attributes.is_empty() {
            return Ok(());
        }
        let mut native = self.encode_svm_attributes(attributes)?;
        self.kfd()?
            .svm_attributes(address, size, uapi::SVM_OP_SET_ATTR, native.as_mut_slice())
            .map_err(|source| native_error("AMDKFD_IOC_SVM set attributes", source))
    }

    fn get_kfd_svm_attributes(
        &self,
        address: u64,
        size: u64,
        attributes: &mut [KfdSvmAttribute],
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if attributes.is_empty() {
            return Ok(());
        }
        let mut native = self.encode_svm_attributes(attributes)?;
        self.kfd()?
            .svm_attributes(address, size, uapi::SVM_OP_GET_ATTR, native.as_mut_slice())
            .map_err(|source| native_error("AMDKFD_IOC_SVM get attributes", source))?;
        self.decode_svm_attributes(attributes, native.as_slice())
    }

    fn retain_kfd_signal_event_page(allocation: &mut NativeAllocation) -> Result<(), Error> {
        allocation.retain_signal_event_page()
    }
}

impl LinuxGpuEventDriver for LinuxKfdDriver {
    fn create_kfd_signal_event(
        &self,
        device: &DeviceState,
        event_page: Option<&NativeAllocation>,
    ) -> Result<Owned<NativeSignalEvent>, Error> {
        self.ensure_open()?;
        let event_page_handle = event_page
            .map(|page| page.signal_event_page_handle(&device.vm))
            .transpose()?;
        NativeSignalEvent::create(
            device.vm.kfd_owner(),
            event_page_handle,
            device.vm.allocator(),
        )
    }

    fn destroy_kfd_signal_event(event: &mut NativeSignalEvent) -> Result<(), Error> {
        event.destroy()
    }

    fn poll_kfd_memory_fault(&self, device: &DeviceState) -> Result<Option<GpuMemoryFault>, Error> {
        device.vm.poll_memory_fault()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::session::{Session, SessionLifetime};
    #[test]
    fn construction_is_inert_for_both_lifetime_policies() {
        let native = LinuxKfdDriver::new(Allocator::default());
        assert!(native.kfd.get().is_none());
        assert!(native.bindings.is_empty_for_test());
        for policy in [SessionLifetime::Process, SessionLifetime::Session] {
            let session = Session::new(policy).unwrap();
            assert_eq!(session.state_lifetime(), policy);
        }
    }

    #[test]
    fn inherited_instance_rejects_work_before_mutating_state() {
        let mut native = LinuxKfdDriver::new(Allocator::default());
        native.process = std::process::id().wrapping_add(1);
        assert_eq!(
            native.allocate_host(1, 1).err().unwrap().kind(),
            ErrorKind::Unsupported
        );
        let mut visited = false;
        assert_eq!(
            native
                .enumerate(&mut |_| {
                    visited = true;
                    Ok(())
                })
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        assert!(!visited);
        assert_eq!(
            native.open_endpoint([0; 16]).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(native.kfd().err().unwrap().kind(), ErrorKind::Unsupported);
        assert_eq!(
            native.shutdown().unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert!(!native.closing);
        assert!(native.kfd.get().is_none());

        native.process = std::process::id();
        native.shutdown().unwrap();
    }
}
