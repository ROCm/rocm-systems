// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD and DRM driver. Construction is inert; activation opens KFD and
//! retains each exact VM binding for recreation within its lifetime policy.
mod allocation;
mod drm;
mod event;
mod imported_system;
pub(crate) mod interop;
mod kernel_queue;
pub(super) mod memory;
mod operations;
mod queue;
mod registered_host;
pub(super) mod sys;
mod sysfs;
mod uapi;
use crate::os::linux::{
    errno, file as os_file, memory as os_memory, process_identity as os_process,
};
mod vmem;

use crate::driver::{
    AddressSpaceInfo, AllocationOperations, CachedInfo, Driver, EndpointSelector, GpuDriver,
    KernelQueueResource, UserQueueResource, VirtualMemoryOperations,
};
use crate::error::error;
use crate::host_storage::{Allocator, Owned, Shared};
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
use crate::memory::interop::linux::{
    AisFileOperation, AisFileResult, DmaBuf, KfdIpcMemoryHandle, KfdSvmAccess, KfdSvmAttribute,
    KfdSvmLocation,
};
use crate::memory::{
    AllocationDesc, AllocationLimits, DeviceAccess, HostCachePolicy, HostRegistration, MemoryKind,
    OwnedMemoryKind, VirtualAddressInfo, VirtualMemoryInfo,
};
use crate::os::native_error;
use crate::profiling::ClockCounters;
use crate::queue::{QueuePriority, QueueRequest, QueueScratch, QueueTransport};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};
pub(crate) use allocation::LinuxAllocation;
pub(crate) use event::KfdSignalEvent;
pub(crate) use kernel_queue::KfdKernelQueue;
pub(crate) use memory::KfdEventSubscription;
pub(crate) use queue::KfdQueue;
use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, BorrowedFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};
pub(crate) use sysfs::KfdNode as LinuxSelector;
pub(crate) use vmem::{
    KfdVirtualAddress, KfdVirtualDeviceMapping, KfdVirtualHostMapping, KfdVirtualMemory,
};

/// Linux KFD driver with lazy activation under a selected lifetime policy.
///
/// Construction records policy and allocator state only. The first explicit GPU
/// activation opens KFD, establishes the retained VM bindings, and enables the
/// runtime under the KFD connection's locks so concurrent callers cannot
/// publish partial state. PROCESS sessions share one process-local connection.
pub(crate) struct LinuxKfdDriver {
    allocator: Allocator,
    driver_instance: u64,
    process: u32,
    closing: bool,
    lifetime: DriverContextLifetime,
    joined_primary: AtomicBool,
    local: KfdContextResources,
}

/// Connection and VM bindings for one KFD context. A process-lifetime context
/// uses the system allocator so no session callback can outlive its instance.
struct KfdContextResources {
    allocator: Allocator,
    kfd: OnceLock<Shared<sys::Kfd>>,
    initialization: Mutex<()>,
    bindings: memory::VmBindings,
}

/// Primary KFD context resources shared by process-lifetime sessions.
struct ProcessState {
    // KFD keeps the exact primary DRM file object after a session closes.
    // Retain the system-allocated connection and bindings until process exit so
    // a later session can reacquire that same VM instead of receiving EBUSY.
    resources: Option<KfdContextResources>,
    joined_sessions: u64,
}

/// Process-local lock and identity for the shared primary KFD context.
struct ProcessSlot {
    process: u32,
    state: Mutex<ProcessState>,
}

// A fork child replaces the inherited slot before touching its mutex. The
// old slot remains allocated because an inherited KFD owner may still hold
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
                resources: None,
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

impl AddressSpaceInfo for DeviceState {
    fn address_range(&self) -> (u64, u64) {
        DeviceState::address_range(self)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        DeviceState::shares_vm(self, other)
    }
}

impl CachedInfo for LinuxAllocation {
    type Info = crate::memory::AllocationInfo;
    fn cached_info(&self) -> crate::memory::AllocationInfo {
        LinuxAllocation::cached_info(self)
    }
}

impl CachedInfo for KfdVirtualAddress {
    type Info = VirtualAddressInfo;
    fn cached_info(&self) -> VirtualAddressInfo {
        VirtualAddressInfo {
            address: self.address(),
            size: self.size(),
            mapping_granularity: self.mapping_granularity(),
        }
    }
}

impl CachedInfo for KfdVirtualMemory {
    type Info = VirtualMemoryInfo;
    fn cached_info(&self) -> VirtualMemoryInfo {
        self.info()
    }
}

impl CachedInfo for KfdQueue {
    type Info = QueueTransport;
    fn cached_info(&self) -> QueueTransport {
        KfdQueue::cached_info(self)
    }
}

/// Lightweight session device reference retaining its activated VM.
#[derive(Clone)]
pub(crate) struct DeviceState {
    vm: Shared<memory::DeviceVm>,
    native: sysfs::KfdNode,
    lifetime: DriverContextLifetime,
    gpu_counter_frequency_hz: u64,
}

impl DeviceState {
    pub(crate) fn subscribe_events(&self) -> Result<KfdEventSubscription, Error> {
        self.vm.subscribe_events()
    }
    pub(crate) fn address_range(&self) -> (u64, u64) {
        self.vm.address_range()
    }

    pub(crate) fn supports_system_dma_buf_import(&self) -> bool {
        self.vm.supports_system_dma_buf_import()
    }

    pub(crate) fn shares_vm(&self, other: &Self) -> bool {
        Shared::ptr_eq(&self.vm, &other.vm)
    }
}

impl KfdContextResources {
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
            // Retained driver resources also retain this endpoint. Preserve the
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
        Self::with_context_lifetime(allocator, DriverContextLifetime::Session)
    }

    pub(crate) fn with_context_lifetime(
        allocator: Allocator,
        lifetime: DriverContextLifetime,
    ) -> Self {
        Self {
            allocator,
            driver_instance: crate::driver::new_driver_instance(),
            process: std::process::id(),
            closing: false,
            lifetime,
            joined_primary: AtomicBool::new(false),
            local: KfdContextResources::new(allocator),
        }
    }

    fn ensure_open(&self) -> Result<(), Error> {
        // Reject inherited instances before touching allocator callbacks,
        // filesystem state, or a mutex that may have been held across fork.
        os_process::check_process(self.process)
            .map_err(|e| native_error("session process check", e))?;
        if self.closing {
            Err(error(
                ErrorKind::InvalidArgument,
                "session teardown already began",
            ))
        } else {
            Ok(())
        }
    }

    fn with_context_resources<T>(
        &self,
        f: impl FnOnce(&KfdContextResources) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.ensure_open()?;
        if self.lifetime == DriverContextLifetime::Session {
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
        let resources = state.resources.as_ref().ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "KFD primary context is not active",
            )
        })?;
        f(resources)
    }

    fn kfd(&self) -> Result<Shared<sys::Kfd>, Error> {
        self.with_context_resources(KfdContextResources::kfd)
    }

    fn svm_location_to_native(&self, location: KfdSvmLocation) -> Result<u32, Error> {
        match location {
            KfdSvmLocation::System => Ok(uapi::SVM_LOCATION_SYSTEM),
            KfdSvmLocation::Undefined => Ok(uapi::SVM_LOCATION_UNDEFINED),
            KfdSvmLocation::Device(identity) => {
                self.with_context_resources(|resources| resources.bindings.svm_gpu_id(identity))
            }
        }
    }

    fn svm_location_from_native(&self, location: u32) -> Result<KfdSvmLocation, Error> {
        match location {
            uapi::SVM_LOCATION_SYSTEM => Ok(KfdSvmLocation::System),
            uapi::SVM_LOCATION_UNDEFINED => Ok(KfdSvmLocation::Undefined),
            gpu_id => self
                .with_context_resources(|resources| resources.bindings.svm_identity(gpu_id))
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
                    value: self.with_context_resources(|resources| {
                        resources.bindings.svm_gpu_id(device)
                    })?,
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

impl Driver for LinuxKfdDriver {
    type DeviceState = DeviceState;

    fn allocator(&self) -> Allocator {
        self.allocator
    }

    fn driver_instance(&self) -> u64 {
        self.driver_instance
    }
    fn context_lifetime(&self) -> DriverContextLifetime {
        self.lifetime
    }
    fn shutdown(&mut self) -> Result<(), Error> {
        os_process::check_process(self.process)
            .map_err(|e| native_error("session shutdown process check", e))?;
        self.closing = true;
        if self.lifetime == DriverContextLifetime::Session {
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
            if let Some(kfd) = state
                .resources
                .as_ref()
                .and_then(|resources| resources.kfd.get())
            {
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
                endpoint.driver_instance = self.driver_instance;
                visitor(endpoint)
            },
        )
    }
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.ensure_open()?;
        let mut endpoint =
            sysfs::open_endpoint(Self::TOPOLOGY_ROOT, Self::DRM_ROOT, id, self.allocator)?;
        endpoint.driver_instance = self.driver_instance;
        Ok(endpoint)
    }
    fn activate(&self, endpoint: &Endpoint) -> Result<DeviceState, Error> {
        if endpoint.gpu().is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "the Linux KFD driver activates only GPU endpoints",
            ));
        }
        let current = self.open_endpoint(endpoint.id)?;
        if &current != endpoint {
            return Err(error(
                ErrorKind::DeviceLost,
                "endpoint metadata changed before activation",
            ));
        }
        let page = os_memory::page_size().map_err(|e| native_error("native page size", e))?;
        if page < 4096 {
            return Err(error(
                ErrorKind::Unsupported,
                "native implementation requires host pages of at least 4 KiB",
            ));
        }
        let EndpointSelector::LinuxKfd(native) = current.selector else {
            return Err(error(
                ErrorKind::InvalidData,
                "KFD endpoint has a foreign selector",
            ));
        };
        let vm = if self.lifetime == DriverContextLifetime::Session {
            let kfd = self.local.kfd()?;
            kfd.prepare_context(self.lifetime)
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
            let resources = state
                .resources
                .get_or_insert_with(|| KfdContextResources::new(Allocator::system()));
            let kfd = resources.kfd()?;
            kfd.prepare_context(self.lifetime)
                .map_err(|source| native_error("KFD context selection", source))?;
            resources.bindings.device(&kfd, &native)?
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
            lifetime: self.lifetime,
            gpu_counter_frequency_hz,
        })
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
            operations::marketing_name(ids, 0x7550, 0xc0).as_deref(),
            Some("AMD Radeon RX 9070 XT")
        );
        assert_eq!(operations::marketing_name(ids, 0x7550, 0xc1), None);
        assert_eq!(operations::marketing_name(ids, 0x7551, 0xc0), None);
        assert_eq!(std::mem::size_of::<drm::DeviceInfoPrefix>(), 32);
    }
    use crate::session::{DriverContextLifetime, Session};
    #[test]
    fn construction_is_inert_for_both_lifetime_policies() {
        let native = LinuxKfdDriver::new(Allocator::default());
        assert!(native.local.kfd.get().is_none());
        assert!(native.local.bindings.is_empty_for_test());
        for policy in [
            DriverContextLifetime::Process,
            DriverContextLifetime::Session,
        ] {
            let session = Session::new(policy).unwrap();
            assert_eq!(session.driver_context_lifetime(), policy);
        }
    }

    #[test]
    fn inherited_instance_rejects_work_before_mutating_state() {
        let mut native = LinuxKfdDriver::new(Allocator::default());
        native.process = std::process::id().wrapping_add(1);
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
