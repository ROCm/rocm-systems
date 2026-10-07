// SPDX-License-Identifier: MIT

//! Linux native control. Construction is inert; activation opens KFD and
//! retains each exact VM binding for recreation within its lifetime policy.
mod allocation;
mod drm;
mod event;
mod host;
mod imported_system;
mod kernel_queue;
mod memory;
mod process_identity;
mod queue;
mod registered_host;
mod sys;
mod sysfs;
mod uapi;
mod util;
mod vmem;

use crate::driver::{
    AddressSpaceInfo, AllocationDriver, AllocationOwnerInfo, AllocationTypes, DeviceDriver,
    DeviceStateInfo, EndpointSelector, GpuAuxAllocationDriver, GpuPresentationDriver,
    GpuProfilingDriver, HostDriver, HostOwnerInfo, HostTypes, KernelQueueDriver, KernelQueueTypes,
    ProviderDriver, ProviderTypes, QueueDriver, QueueOwnerInfo, QueueTypes,
    VirtualAddressOwnerInfo, VirtualMemoryDriver, VirtualMemoryOwnerInfo, VirtualMemoryTypes,
    linux_interop::{LinuxGpuEventDriver, LinuxMemoryInteropDriver},
};
use crate::event::GpuMemoryFault;
use crate::host_storage::{Allocator, Owned, Shared};
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::memory::interop::linux::{
    AisFileOperation, AisFileResult, DmaBuf, KfdIpcMemoryHandle, KfdSvmAccess, KfdSvmAttribute,
    KfdSvmLocation,
};
use crate::memory::{
    AllocationDesc, AllocationLimits, DeviceAccess, HostRegistration, MemoryKind, OwnedMemoryKind,
    VirtualAddressInfo, VirtualMemoryInfo,
};
use crate::profiling::ClockCounters;
use crate::queue::{QueuePriority, QueueRequest, QueueScratch, QueueTransport};
use crate::session::SessionLifetime;
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};
pub(crate) use allocation::NativeAllocation;
pub(crate) use event::KfdSignalEvent as NativeSignalEvent;
pub(crate) use host::HostAllocation as NativeHostAllocation;
pub(crate) use kernel_queue::KfdKernelQueue as NativeKernelQueue;
use memory::{error, native_error};
pub(crate) use queue::KfdQueue as NativeQueue;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};
pub(crate) use sysfs::NativeNode as LinuxSelector;
pub(crate) use vmem::{
    KfdVirtualAddress as NativeVirtualAddress,
    KfdVirtualDeviceMapping as NativeVirtualDeviceMapping,
    KfdVirtualHostMapping as NativeVirtualHostMapping, KfdVirtualMemory as NativeVirtualMemory,
};

/// Linux backend with lazy KFD activation under a selected lifetime policy.
///
/// Construction records policy and allocator state only. The first explicit GPU
/// activation opens KFD, establishes the retained VM bindings, and enables the
/// runtime under the native connection's locks so concurrent callers cannot
/// publish partial state. PROCESS sessions share one process-local connection.
pub(crate) struct LinuxKfdDriver {
    allocator: Allocator,
    provider_instance: u64,
    process: u32,
    closing: bool,
    lifetime: SessionLifetime,
    joined_primary: AtomicBool,
    local: NativeConnection,
}

/// Native state for one KFD process context. A process-lifetime context uses
/// the system allocator so no session callback can outlive its instance.
struct NativeConnection {
    allocator: Allocator,
    kfd: OnceLock<Shared<sys::Kfd>>,
    initialization: Mutex<()>,
    bindings: memory::VmBindings,
}

struct ProcessState {
    // KFD keeps the exact primary DRM file object after a session closes.
    // Retain the system-allocated native graph until process exit so a later
    // session can reacquire that same VM instead of receiving EBUSY.
    native: Option<NativeConnection>,
    joined_sessions: u64,
}

struct ProcessSlot {
    process: u32,
    state: Mutex<ProcessState>,
}

// A fork child replaces the inherited slot before touching its mutex. The
// old slot remains allocated because an inherited native owner may still hold
// a reference to it. The process-local primary context lives until exit.
static PRIMARY_SLOT: AtomicPtr<ProcessSlot> = AtomicPtr::new(std::ptr::null_mut());

#[allow(unsafe_code)]
fn primary_slot() -> &'static ProcessSlot {
    let process = std::process::id();
    loop {
        let current = PRIMARY_SLOT.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: Published slots remain allocated for the process lifetime.
            if unsafe { (*current).process == process } {
                // SAFETY: The matching published slot is never freed or moved.
                return unsafe { &*current };
            }
        }
        let candidate = Box::into_raw(Box::new(ProcessSlot {
            process,
            state: Mutex::new(ProcessState {
                native: None,
                joined_sessions: 0,
            }),
        }));
        match PRIMARY_SLOT.compare_exchange(current, candidate, Ordering::AcqRel, Ordering::Acquire)
        {
            // SAFETY: Publication gives this allocation static process scope.
            Ok(_) => return unsafe { &*candidate },
            Err(_) => {
                // SAFETY: This candidate was never published or borrowed.
                unsafe { drop(Box::from_raw(candidate)) };
            }
        }
    }
}

impl DeviceStateInfo for DeviceState {
    fn has_observed_loss(&self) -> bool {
        DeviceState::has_observed_loss(self)
    }
}

impl AddressSpaceInfo for DeviceState {
    fn address_range(&self) -> (u64, u64) {
        DeviceState::address_range(self)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        DeviceState::shares_vm(self, other)
    }
}

impl HostOwnerInfo for NativeHostAllocation {
    fn cached_info(&self) -> crate::memory::HostAllocationInfo {
        NativeHostAllocation::cached_info(self)
    }
}

impl AllocationOwnerInfo for NativeAllocation {
    fn cached_info(&self) -> crate::memory::AllocationInfo {
        NativeAllocation::cached_info(self)
    }
}

impl VirtualAddressOwnerInfo for NativeVirtualAddress {
    fn cached_info(&self) -> VirtualAddressInfo {
        VirtualAddressInfo {
            address: self.address(),
            size: self.size(),
            mapping_granularity: self.mapping_granularity(),
        }
    }
}

impl VirtualMemoryOwnerInfo for NativeVirtualMemory {
    fn cached_info(&self) -> VirtualMemoryInfo {
        self.info()
    }
}

impl QueueOwnerInfo for NativeQueue {
    fn cached_info(&self) -> QueueTransport {
        NativeQueue::cached_info(self)
    }
}

impl ProviderTypes for LinuxKfdDriver {
    type DeviceState = DeviceState;
}

impl HostTypes for LinuxKfdDriver {
    type HostAllocation = NativeHostAllocation;
}

impl AllocationTypes for LinuxKfdDriver {
    type Allocation = NativeAllocation;
}

impl VirtualMemoryTypes for LinuxKfdDriver {
    type VirtualAddress = NativeVirtualAddress;
    type VirtualDeviceMapping = NativeVirtualDeviceMapping;
    type VirtualHostMapping = NativeVirtualHostMapping;
    type VirtualMemory = NativeVirtualMemory;
}

impl QueueTypes for LinuxKfdDriver {
    type Queue = NativeQueue;
}

impl KernelQueueTypes for LinuxKfdDriver {
    type KernelQueue = NativeKernelQueue;
}

/// Lightweight session device reference retaining its activated VM.
#[derive(Clone)]
pub(crate) struct DeviceState {
    vm: Shared<memory::DeviceVm>,
    native: sysfs::NativeNode,
    lifetime: SessionLifetime,
    gpu_counter_frequency_hz: u64,
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

impl NativeConnection {
    fn new(allocator: Allocator) -> Self {
        Self {
            allocator,
            kfd: OnceLock::new(),
            initialization: Mutex::new(()),
            bindings: memory::VmBindings::new(allocator),
        }
    }
    fn kfd(&self) -> Result<Shared<sys::Kfd>, Error> {
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

    fn shutdown(&mut self) -> Result<(), Error> {
        self.bindings.shutdown()?;
        if let Some(kfd) = self.kfd.get_mut() {
            // Retained native owners also retain this endpoint. Preserve the
            // connection until all dependencies have been discharged.
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
}

impl LinuxKfdDriver {
    const TOPOLOGY_ROOT: &'static str = "/sys/class/kfd/kfd/topology";
    const DRM_ROOT: &'static str = "/sys/class/drm";
    #[cfg(test)]
    pub(crate) fn new(allocator: Allocator) -> Self {
        Self::with_lifetime(allocator, SessionLifetime::Session)
    }

    pub(crate) fn with_lifetime(allocator: Allocator, lifetime: SessionLifetime) -> Self {
        Self {
            allocator,
            provider_instance: crate::driver::new_provider_instance(),
            process: std::process::id(),
            closing: false,
            lifetime,
            joined_primary: AtomicBool::new(false),
            local: NativeConnection::new(allocator),
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

    fn with_native<T>(
        &self,
        f: impl FnOnce(&NativeConnection) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.ensure_open()?;
        if self.lifetime == SessionLifetime::Session {
            return f(&self.local);
        }
        if !self.joined_primary.load(Ordering::Acquire) {
            return Err(error(
                ErrorKind::InvalidArgument,
                "this session has not activated the primary KFD context",
            ));
        }
        let slot = primary_slot();
        let state = slot
            .state
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD primary context lock poisoned"))?;
        let native = state.native.as_ref().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "KFD primary context is not active",
            )
        })?;
        f(native)
    }

    fn kfd(&self) -> Result<Shared<sys::Kfd>, Error> {
        self.with_native(NativeConnection::kfd)
    }

    fn svm_location_to_native(&self, location: KfdSvmLocation) -> Result<u32, Error> {
        match location {
            KfdSvmLocation::System => Ok(uapi::SVM_LOCATION_SYSTEM),
            KfdSvmLocation::Undefined => Ok(uapi::SVM_LOCATION_UNDEFINED),
            KfdSvmLocation::Device(identity) => {
                self.with_native(|native| native.bindings.svm_gpu_id(identity))
            }
        }
    }

    fn svm_location_from_native(&self, location: u32) -> Result<KfdSvmLocation, Error> {
        match location {
            uapi::SVM_LOCATION_SYSTEM => Ok(KfdSvmLocation::System),
            uapi::SVM_LOCATION_UNDEFINED => Ok(KfdSvmLocation::Undefined),
            gpu_id => self
                .with_native(|native| native.bindings.svm_identity(gpu_id))
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
                    value: self.with_native(|native| native.bindings.svm_gpu_id(device))?,
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
        if self.lifetime == SessionLifetime::Session {
            return self.local.shutdown();
        }
        if !self.joined_primary.load(Ordering::Acquire) {
            return Ok(());
        }
        let slot = primary_slot();
        let mut state = slot
            .state
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD primary context lock poisoned"))?;
        if state.joined_sessions == 0 {
            return Err(error(
                ErrorKind::Internal,
                "KFD primary session accounting underflow",
            ));
        }
        if state.joined_sessions == 1 {
            if let Some(kfd) = state.native.as_ref().and_then(|native| native.kfd.get()) {
                kfd.disable_runtime()
                    .map_err(|source| native_error("AMDKFD_IOC_RUNTIME_ENABLE disable", source))?;
            }
        }
        state.joined_sessions -= 1;
        self.joined_primary.store(false, Ordering::Release);
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
        let page = util::page_size().map_err(|e| native_error("native page size", e))?;
        if page < 4096 {
            return Err(error(
                ErrorKind::Unsupported,
                "native implementation requires host pages of at least 4 KiB",
            ));
        }
        if lifetime != self.lifetime {
            return Err(error(
                ErrorKind::InvalidArgument,
                "session lifetime differs from its native controller",
            ));
        }
        let EndpointSelector::LinuxKfd(native) = current.native else {
            return Err(error(
                ErrorKind::InvalidData,
                "KFD endpoint has a foreign selector",
            ));
        };
        let vm = if lifetime == SessionLifetime::Session {
            let kfd = self.local.kfd()?;
            kfd.prepare_context(lifetime)
                .map_err(|source| native_error("KFD context selection", source))?;
            self.local.bindings.device(&kfd, &native)?
        } else {
            let slot = primary_slot();
            let mut state = slot
                .state
                .lock()
                .map_err(|_| error(ErrorKind::Internal, "KFD primary context lock poisoned"))?;
            if !self.joined_primary.load(Ordering::Acquire) {
                state.joined_sessions = state.joined_sessions.checked_add(1).ok_or_else(|| {
                    error(
                        ErrorKind::ResourceExhausted,
                        "too many primary KFD sessions",
                    )
                })?;
                // Even a failed activation can enable KFD or retain a VM.
                // Its session must discharge runtime enablement on shutdown.
                self.joined_primary.store(true, Ordering::Release);
            }
            let connection = state
                .native
                .get_or_insert_with(|| NativeConnection::new(Allocator::system()));
            let kfd = connection.kfd()?;
            kfd.prepare_context(lifetime)
                .map_err(|source| native_error("KFD context selection", source))?;
            connection.bindings.device(&kfd, &native)?
        };
        let gpu_counter_frequency_hz = vm
            .render()
            .ok()
            .and_then(|render| drm::device_info_prefix(render).ok())
            .filter(|info| {
                endpoint
                    .pci
                    .is_some_and(|pci| info.device_id == pci.device_id)
            })
            .map_or(0, |info| u64::from(info.gpu_counter_frequency_khz) * 1000);
        Ok(DeviceState {
            vm,
            native,
            lifetime,
            gpu_counter_frequency_hz,
        })
    }
}

impl GpuPresentationDriver for LinuxKfdDriver {
    fn gpu_presentation(&self, endpoint: &Endpoint) -> GpuPresentation {
        let mut presentation = GpuPresentation {
            product_name: None,
            asic_family_id: endpoint.gpu().map_or(0, |gpu| gpu.asic_family_id),
            gpu_counter_frequency_hz: None,
        };
        let Some(pci) = endpoint.pci else {
            return presentation;
        };
        let Some(native) = endpoint.linux_kfd_drm_info() else {
            return presentation;
        };
        let device_info = native
            .render_minor
            .and_then(|minor| sysfs::open_render(minor).ok())
            .and_then(|render| drm::device_info_prefix(&render).ok())
            .filter(|info| info.device_id == pci.device_id);
        if let Some(info) = device_info.filter(|info| info.family_id != 0) {
            presentation.asic_family_id = info.family_id;
        }
        presentation.gpu_counter_frequency_hz = device_info
            .filter(|info| info.gpu_counter_frequency_khz != 0)
            .map(|info| u64::from(info.gpu_counter_frequency_khz) * 1000);
        presentation.product_name = std::fs::metadata("/usr/share/libdrm/amdgpu.ids")
            .ok()
            .filter(|metadata| metadata.len() <= 1024 * 1024)
            .and_then(|_| std::fs::read_to_string("/usr/share/libdrm/amdgpu.ids").ok())
            .and_then(|contents| marketing_name(&contents, pci.device_id, pci.revision_id));
        presentation
    }
}

fn marketing_name(contents: &str, device_id: u32, revision: u32) -> Option<String> {
    contents.lines().find_map(|line| {
        let mut columns = line.splitn(3, ',');
        let device = u32::from_str_radix(columns.next()?.trim(), 16).ok()?;
        let candidate_revision = u32::from_str_radix(columns.next()?.trim(), 16).ok()?;
        let name = columns.next()?.trim();
        (device == device_id && candidate_revision == revision && !name.is_empty())
            .then(|| name.to_owned())
    })
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
    fn allocation_is_device_local(allocation: &NativeAllocation) -> bool {
        allocation.is_device_local()
    }
    fn set_allocation_access(
        allocation: &mut NativeAllocation,
        devices: &[&DeviceState],
    ) -> Result<(), Error> {
        let mut vms = Vec::new();
        vms.try_reserve(devices.len())
            .map_err(|_| error(ErrorKind::ResourceExhausted, "access VM list is exhausted"))?;
        vms.extend(devices.iter().map(|device| &device.vm));
        allocation.set_access(&vms)
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
    fn free_allocation(allocation: &mut NativeAllocation) -> Result<(), Error> {
        allocation.free()
    }
    fn allocate_owned(
        &self,
        device: &DeviceState,
        peers: &[&DeviceState],
        kind: OwnedMemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<NativeAllocation>, Error> {
        let desc = checked_allocation_desc(size, alignment)?;
        let kind = kind.get();
        if device.lifetime == SessionLifetime::Session && matches!(kind, MemoryKind::OwnedHost) {
            return Err(error(
                ErrorKind::Unsupported,
                "secondary KFD contexts cannot bind host-owned pages",
            ));
        }
        let native_kind = match kind {
            MemoryKind::System => memory::BufferKind::Gtt,
            MemoryKind::OwnedHost => memory::BufferKind::OwnedUserptr { uncached: false },
            MemoryKind::RegisteredHost { .. } => {
                return Err(error(
                    ErrorKind::DriverContract,
                    "owned request contains a borrowed host address",
                ));
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

    #[allow(unsafe_code)]
    unsafe fn register_host(
        &self,
        device: &DeviceState,
        peers: &[&DeviceState],
        request: HostRegistration,
    ) -> Result<Owned<NativeAllocation>, Error> {
        let HostRegistration {
            address,
            uncached,
            size,
            alignment,
            permissions,
        } = request;
        let desc = checked_allocation_desc(size, alignment)?;
        if device.lifetime == SessionLifetime::Session {
            // SAFETY: The driver caller retains the page cover and access
            // synchronization required by this registration contract.
            unsafe {
                NativeAllocation::create_registered_host(
                    device.vm.clone(),
                    peers.iter().map(|peer| peer.vm.clone()),
                    desc,
                    address,
                    permissions,
                    uncached,
                )
            }
        } else {
            // SAFETY: The driver caller retains these pages until cleanup or
            // process exit, including an ambiguous KFD result.
            let pages = unsafe { memory::BorrowedHostPages::new(address, uncached) };
            NativeAllocation::create_with_peers(
                device.vm.clone(),
                peers.iter().map(|peer| peer.vm.clone()),
                desc,
                memory::BufferKind::Userptr(pages),
                permissions,
            )
        }
    }
}

fn checked_allocation_desc(size: u64, alignment: u64) -> Result<AllocationDesc, Error> {
    let desc = AllocationDesc { size, alignment };
    let page = util::page_size().map_err(|source| native_error("native page size", source))? as u64;
    let limits = AllocationLimits {
        alignment: page,
        granularity: page,
        maximum_size: isize::MAX as u64,
    };
    if !limits.supports(desc) {
        return Err(error(
            ErrorKind::InvalidArgument,
            "invalid native allocation extent or alignment",
        ));
    }
    Ok(desc)
}

impl GpuAuxAllocationDriver for LinuxKfdDriver {
    fn allocate_queue_scratch(
        &self,
        device: &DeviceState,
        size: u64,
    ) -> Result<Owned<NativeAllocation>, Error> {
        self.ensure_open()?;
        let page =
            util::page_size().map_err(|source| native_error("native page size", source))? as u64;
        let native_size = size
            .checked_add(page - 1)
            .map(|size| size & !(page - 1))
            .ok_or_else(|| error(ErrorKind::ResourceExhausted, "scratch size overflows"))?;
        let desc = AllocationDesc {
            size: native_size,
            alignment: page,
        };
        let limits = AllocationLimits {
            alignment: page,
            granularity: page,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(desc) || native_size > device.native.local_memory_bytes {
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
        kind: OwnedMemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<NativeVirtualMemory>, Error> {
        self.ensure_open()?;
        NativeVirtualMemory::create(device.vm.clone(), kind.get(), size, pinned, uncached)
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
    fn supports_expert_scheduling(&self, device: &DeviceState) -> Result<bool, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        let version = device.vm.version;
        Ok((version.major, version.minor) >= (1, 20))
    }

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
    #[allow(unsafe_code)]
    unsafe fn set_queue_scratch(
        queue: &mut NativeQueue,
        scratch: QueueScratch,
    ) -> Result<(), Error> {
        queue.set_scratch(scratch)
    }
    #[allow(unsafe_code)]
    unsafe fn destroy_queue(queue: &mut NativeQueue) -> Result<(), Error> {
        queue.destroy()
    }
    #[allow(unsafe_code)]
    unsafe fn create_queue(
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
    fn available_sdma_rings(&self, device: &DeviceState) -> Result<u32, Error> {
        self.ensure_open()?;
        if !cfg!(target_arch = "x86_64")
            || device.native.queues.gfx_target != 120_001
            || !device.native.queues.sdma_qualified
        {
            return Err(error(
                ErrorKind::Unsupported,
                "SDMA kernel queue is unqualified for this GPU target",
            ));
        }
        drm::sdma_available_rings(device.vm.render()?)
            .map_err(|source| native_error("DRM SDMA ring query", source))
    }

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
            KernelQueueFormat::Sdma | KernelQueueFormat::SdmaOnRing(_)
                if !device.native.queues.sdma_qualified =>
            {
                return Err(error(
                    ErrorKind::Unsupported,
                    "SDMA kernel queue is unavailable",
                ));
            }
            _ => (),
        }
        NativeKernelQueue::create(device.vm.clone(), format)
    }

    #[allow(unsafe_code)]
    unsafe fn submit_kernel_queue(
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
    fn available_memory(&self, device: &DeviceState) -> Result<u64, Error> {
        self.ensure_open()?;
        device
            .vm
            .kfd()
            .available_memory(device.native.gpu_id)
            .map_err(|source| native_error("KFD available memory query", source))
    }

    fn set_persisting_l2_cache_size(
        &self,
        device: &DeviceState,
        size_bytes: u32,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        device.vm.check()?;
        drm::set_persisting_l2_cache_size(device.vm.render()?, size_bytes).map_err(|source| {
            if source.raw_os_error() == Some(22) {
                Error::NativeOperation {
                    kind: ErrorKind::InvalidArgument,
                    operation: "DRM persisting L2 cache request",
                    source,
                }
            } else {
                native_error("DRM persisting L2 cache request", source)
            }
        })
    }
}

impl GpuProfilingDriver for LinuxKfdDriver {
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
            gpu_frequency: device.gpu_counter_frequency_hz,
        })
    }
    #[allow(unsafe_code)]
    unsafe fn set_trap_handler(
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
    #[allow(unsafe_code)]
    unsafe fn spm_set_destination(
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

impl LinuxMemoryInteropDriver for LinuxKfdDriver {
    #[allow(unsafe_code)]
    unsafe fn close_owned_descriptor(descriptor: RawFd) -> io::Result<()> {
        util::close_descriptor(descriptor)
    }

    fn duplicate_descriptor(descriptor: RawFd) -> Result<OwnedFd, Error> {
        util::duplicate_file(descriptor)
            .map(Into::into)
            .map_err(|source| Error::NativeOperation {
                kind: match source.raw_os_error() {
                    // Linux EBADF and the explicit negative-descriptor rejection.
                    Some(9) => ErrorKind::InvalidArgument,
                    // Linux ENFILE, EMFILE, and ENOMEM.
                    Some(23 | 24 | 12) => ErrorKind::ResourceExhausted,
                    _ if source.kind() == io::ErrorKind::InvalidInput => ErrorKind::InvalidArgument,
                    _ => ErrorKind::Driver,
                },
                operation: "descriptor duplication",
                source,
            })
    }

    fn descriptor_length(descriptor: RawFd) -> io::Result<u64> {
        util::descriptor_length(descriptor)
    }

    fn read_descriptor_exact_at(
        descriptor: RawFd,
        buffer: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        util::read_descriptor_exact_at(descriptor, buffer, offset)
    }

    fn read_descriptor_at(descriptor: RawFd, buffer: &mut [u8], offset: i64) -> io::Result<usize> {
        util::read_descriptor_at(descriptor, buffer, offset)
    }

    fn write_descriptor_at(descriptor: RawFd, buffer: &[u8], offset: i64) -> io::Result<usize> {
        util::write_descriptor_at(descriptor, buffer, offset)
    }

    fn ais_transfer(
        allocation: &Self::Allocation,
        descriptor: RawFd,
        allocation_offset: u64,
        size: u64,
        file_offset: i64,
        operation: AisFileOperation,
    ) -> Result<AisFileResult, Error> {
        allocation.ais_transfer(descriptor, allocation_offset, size, file_offset, operation)
    }

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
}

impl LinuxGpuEventDriver for LinuxKfdDriver {
    fn retain_kfd_signal_event_page(allocation: &mut NativeAllocation) -> Result<(), Error> {
        allocation.retain_signal_event_page()
    }

    fn create_kfd_signal_event(
        &self,
        device: &DeviceState,
        event_page: Option<&NativeAllocation>,
        page_offered: &mut bool,
    ) -> Result<Owned<NativeSignalEvent>, Error> {
        *page_offered = false;
        self.ensure_open()?;
        let event_page_handle = event_page
            .map(|page| page.signal_event_page_handle(&device.vm))
            .transpose()?;
        NativeSignalEvent::create(
            device.vm.kfd_owner(),
            event_page_handle,
            device.vm.allocator(),
            page_offered,
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

    #[test]
    fn marketing_name_matches_device_and_revision() {
        let ids = "# device, revision, name\n7550, C0, AMD Radeon RX 9070 XT\n7550, C3, AMD Radeon RX 9070\n";
        assert_eq!(
            marketing_name(ids, 0x7550, 0xc0).as_deref(),
            Some("AMD Radeon RX 9070 XT")
        );
        assert_eq!(marketing_name(ids, 0x7550, 0xc1), None);
        assert_eq!(marketing_name(ids, 0x7551, 0xc0), None);
        assert_eq!(std::mem::size_of::<drm::DeviceInfoPrefix>(), 32);
    }
    use crate::session::{Session, SessionLifetime};
    #[test]
    fn construction_is_inert_for_both_lifetime_policies() {
        let native = LinuxKfdDriver::new(Allocator::default());
        assert!(native.local.kfd.get().is_none());
        assert!(native.local.bindings.is_empty_for_test());
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
        assert!(native.local.kfd.get().is_none());

        native.process = std::process::id();
        native.shutdown().unwrap();
    }
}
