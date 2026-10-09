// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD and DRM implementations of the portable driver capabilities.

// Capability implementations share the concrete driver's types and helpers.
#[allow(clippy::wildcard_imports)]
use super::*;

impl GpuDriver for LinuxKfdDriver {
    type UserQueue = KfdQueue;
    type KernelQueue = KfdKernelQueue;

    fn supports_expert_scheduling(&self, device: &DeviceState) -> Result<bool, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        let version = device.vm.version;
        Ok((version.major, version.minor) >= (1, 20))
    }

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

    #[allow(unsafe_code)]
    unsafe fn create_user_queue(
        &self,
        device: &DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<KfdQueue>, Error> {
        queue::create(device.vm.clone(), &device.native, desc, device.lifetime)
    }

    fn create_kernel_queue(
        &self,
        device: &DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<KfdKernelQueue>, Error> {
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
        KfdKernelQueue::create(device.vm.clone(), format)
    }

    fn available_memory(&self, device: &DeviceState) -> Result<u64, Error> {
        self.ensure_open()?;
        device
            .vm
            .kfd()
            .available_memory(device.native.gpu_id)
            .map_err(|source| native_error("KFD available memory query", source))
    }

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

    fn allocate_queue_scratch(
        &self,
        device: &DeviceState,
        size: u64,
    ) -> Result<Owned<LinuxAllocation>, Error> {
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
        LinuxAllocation::create_scratch(&device.vm, desc)
    }
    fn map_mmio_remap(&self, device: &DeviceState) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        LinuxAllocation::create_mmio(&device.vm)
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

pub(super) fn marketing_name(contents: &str, device_id: u32, revision: u32) -> Option<String> {
    contents.lines().find_map(|line| {
        let mut columns = line.splitn(3, ',');
        let device = u32::from_str_radix(columns.next()?.trim(), 16).ok()?;
        let candidate_revision = u32::from_str_radix(columns.next()?.trim(), 16).ok()?;
        let name = columns.next()?.trim();
        (device == device_id && candidate_revision == revision && !name.is_empty())
            .then(|| name.to_owned())
    })
}

impl AllocationOperations for LinuxKfdDriver {
    type Allocation = LinuxAllocation;

    fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        endpoint.driver_instance == self.driver_instance && endpoint.gpu().is_some()
    }

    fn check_allocation(allocation: &LinuxAllocation) -> Result<(), Error> {
        allocation.check()
    }
    fn allocation_is_device_local(allocation: &LinuxAllocation) -> bool {
        allocation.is_device_local()
    }
    fn set_allocation_access(
        allocation: &mut LinuxAllocation,
        devices: &[&DeviceState],
    ) -> Result<(), Error> {
        let mut vms = Vec::new();
        vms.try_reserve(devices.len())
            .map_err(|_| error(ErrorKind::ResourceExhausted, "access VM list is exhausted"))?;
        vms.extend(devices.iter().map(|device| &device.vm));
        allocation.set_access(&vms)
    }
    fn allocation_device_address(
        allocation: &LinuxAllocation,
        device: &DeviceState,
    ) -> Result<u64, Error> {
        allocation.device_address(&device.vm)
    }
    fn allocation_is_owned_by(allocation: &LinuxAllocation, device: &DeviceState) -> bool {
        allocation.is_owned_by(&device.vm)
    }
    fn free_allocation(allocation: &mut LinuxAllocation) -> Result<(), Error> {
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
    ) -> Result<Owned<LinuxAllocation>, Error> {
        let desc = checked_allocation_desc(size, alignment)?;
        let kind = kind.get();
        if device.lifetime == DriverContextLifetime::Session
            && matches!(kind, MemoryKind::OwnedHost { .. })
        {
            return Err(error(
                ErrorKind::Unsupported,
                "secondary KFD contexts cannot bind host-owned pages",
            ));
        }
        let native_kind = match kind {
            MemoryKind::System => memory::BufferKind::Gtt,
            MemoryKind::OwnedHost { cache } => memory::BufferKind::OwnedUserptr { cache },
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
        LinuxAllocation::create_with_peers(
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
    ) -> Result<Owned<LinuxAllocation>, Error> {
        let HostRegistration {
            address,
            cache,
            size,
            alignment,
            permissions,
        } = request;
        let desc = checked_allocation_desc(size, alignment)?;
        if device.lifetime == DriverContextLifetime::Session {
            if cache == HostCachePolicy::Extended
                && std::iter::once(device)
                    .chain(peers.iter().copied())
                    .any(|peer| peer.native.queues.gfx_target != 120_001)
            {
                return Err(error(
                    ErrorKind::Unsupported,
                    "extended DRM host registration requires GFX1201 mappings",
                ));
            }
            // On GFX1201 the KFD extended USERPTR allocation and a DRM
            // USERPTR object with the default VM page type both map as NC.
            // SAFETY: The driver caller retains the page cover and access
            // synchronization required by this registration contract.
            unsafe {
                LinuxAllocation::create_registered_host(
                    device.vm.clone(),
                    peers.iter().map(|peer| peer.vm.clone()),
                    desc,
                    address,
                    permissions,
                    cache == HostCachePolicy::Uncached,
                )
            }
        } else {
            // SAFETY: The driver caller retains these pages until cleanup or
            // process exit, including an ambiguous KFD result.
            let pages = unsafe { memory::BorrowedHostPages::new(address, cache) };
            LinuxAllocation::create_with_peers(
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

impl VirtualMemoryOperations for LinuxKfdDriver {
    type VirtualAddress = KfdVirtualAddress;
    type VirtualDeviceMapping = KfdVirtualDeviceMapping;
    type VirtualHostMapping = KfdVirtualHostMapping;
    type VirtualMemory = KfdVirtualMemory;

    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<KfdVirtualAddress>, Error> {
        self.ensure_open()?;
        KfdVirtualAddress::reserve(bounds, size, alignment, address, self.allocator)
    }
    fn free_virtual_address(address: &mut KfdVirtualAddress) -> Result<(), Error> {
        address.free()
    }
    fn create_virtual_memory(
        &self,
        device: &DeviceState,
        kind: OwnedMemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<KfdVirtualMemory>, Error> {
        self.ensure_open()?;
        KfdVirtualMemory::create(device.vm.clone(), kind.get(), size, pinned, uncached)
    }
    fn free_virtual_memory(memory: &mut KfdVirtualMemory) -> Result<(), Error> {
        memory.free()
    }
    fn map_virtual_device(
        memory: &KfdVirtualMemory,
        reservation: &KfdVirtualAddress,
        device: &DeviceState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<KfdVirtualDeviceMapping>, Error> {
        KfdVirtualDeviceMapping::create(
            memory,
            reservation,
            device.vm.clone(),
            address,
            offset,
            size,
            permissions,
        )
    }
    fn free_virtual_device_mapping(mapping: &mut KfdVirtualDeviceMapping) -> Result<(), Error> {
        mapping.free()
    }
    fn map_virtual_host(
        memory: &KfdVirtualMemory,
        reservation: &KfdVirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: Allocator,
    ) -> Result<Owned<KfdVirtualHostMapping>, Error> {
        KfdVirtualHostMapping::create(
            memory,
            reservation,
            address,
            offset,
            size,
            permissions,
            allocator,
        )
    }
    fn free_virtual_host_mapping(mapping: &mut KfdVirtualHostMapping) -> Result<(), Error> {
        mapping.free()
    }
}

impl UserQueueResource for KfdQueue {
    type DeviceState = DeviceState;

    fn check(&self) -> Result<(), Error> {
        KfdQueue::check(self)
    }

    fn progress(&self) -> Result<(u64, u64), Error> {
        KfdQueue::progress(self)
    }

    fn inactivate(&mut self) -> Result<(), Error> {
        KfdQueue::inactivate(self)
    }

    fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        KfdQueue::set_priority(self, priority)
    }

    fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        KfdQueue::set_cu_mask(self, mask)
    }

    #[allow(unsafe_code)]
    unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        KfdQueue::set_scratch(self, scratch)
    }

    #[allow(unsafe_code)]
    unsafe fn destroy(&mut self) -> Result<(), Error> {
        KfdQueue::destroy(self)
    }

    fn map_device(&self, device: &DeviceState) -> Result<QueueTransport, Error> {
        KfdQueue::map_device(self, device.vm.clone())
    }
}

impl KernelQueueResource for KfdKernelQueue {
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        KfdKernelQueue::submit(self, command)
    }

    fn status(&self) -> KernelQueueStatus {
        KfdKernelQueue::status(self)
    }

    fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        KfdKernelQueue::refresh_status(self)
    }

    fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        KfdKernelQueue::wait(
            self,
            submission,
            timeout_nanoseconds,
            poll_duration_nanoseconds,
        )
    }

    fn destroy(&mut self) -> Result<(), Error> {
        KfdKernelQueue::destroy(self)
    }
}
