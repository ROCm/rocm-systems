// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD and DRM implementations of the portable driver capabilities.

// Capability implementations share the concrete driver's types and helpers.
#[allow(clippy::wildcard_imports)]
use super::*;
use crate::memory::storage::TryPush;

/// Reads DRM facts only when they describe the endpoint's PCI device.
pub(super) fn matching_device_info(
    endpoint: &Endpoint,
    render: &std::fs::File,
) -> Option<drm::DeviceInfoPrefix> {
    let pci = endpoint.pci?;
    drm::device_info_prefix(render)
        .ok()
        .filter(|info| info.device_id == pci.device_id)
}

pub(super) fn device_counter_frequency_hz(info: drm::DeviceInfoPrefix) -> Option<u64> {
    (info.gpu_counter_frequency_khz != 0)
        .then_some(u64::from(info.gpu_counter_frequency_khz) * 1000)
}

pub(super) fn populate_endpoint(endpoint: &mut Endpoint) {
    let Some(pci) = endpoint.pci else {
        return;
    };
    let Some(native) = crate::topology::platform::linux::kfd_drm_info(endpoint) else {
        return;
    };
    let device_info = native
        .render_minor
        .and_then(|minor| sysfs::open_render(minor).ok())
        .and_then(|render| matching_device_info(endpoint, &render));
    if let (Some(device_info), crate::topology::EndpointKind::Gpu { info }) =
        (device_info, &mut endpoint.kind)
    {
        info.drm_asic_family_id = (device_info.family_id != 0).then_some(device_info.family_id);
        info.counter_frequency_hz = device_counter_frequency_hz(device_info);
    }
    endpoint.product_name = std::fs::metadata("/usr/share/libdrm/amdgpu.ids")
        .ok()
        .filter(|metadata| metadata.len() <= 1024 * 1024)
        .and_then(|_| std::fs::read_to_string("/usr/share/libdrm/amdgpu.ids").ok())
        .and_then(|contents| marketing_name(&contents, pci.device_id, pci.revision_id));
}

impl GpuDriver for LinuxKfdDriver {
    fn xnack_enabled(&self, device: &DeviceState) -> Result<bool, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        device
            .vm
            .kfd()
            .xnack_enabled()
            .map_err(|source| native_error("KFD XNACK mode query", source))
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
            if source.raw_os_error() == Some(errno::EINVAL) {
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

    fn surface_config(
        &self,
        device: &DeviceState,
    ) -> Result<crate::device::gpu::GpuSurfaceConfig, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        let config = device
            .vm
            .kfd()
            .tile_config(device.native.gpu_id)
            .map_err(|source| native_error("KFD tile configuration query", source))?;
        Ok(crate::device::gpu::GpuSurfaceConfig {
            gb_addr_config: config.gb_addr_config,
        })
    }
}

impl GpuUserQueueOperations for LinuxKfdDriver {
    type UserQueue = KfdQueue;

    fn supports_expert_scheduling(&self, device: &DeviceState) -> Result<bool, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        let version = device.vm.version;
        Ok((version.major, version.minor) >= (1, 20))
    }

    #[allow(unsafe_code)]
    unsafe fn create_user_queue(
        &self,
        device: &DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<KfdQueue>, Error> {
        queue::create(device.vm.clone(), &device.native, desc, self.lifetime)
    }

    fn allocate_queue_scratch(
        &self,
        device: &DeviceState,
        size: u64,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        self.ensure_open()?;
        let page = os_memory::page_size()
            .map_err(|source| native_error("native page size", source))? as u64;
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
}

impl GpuKernelQueueOperations for LinuxKfdDriver {
    type KernelQueue = KfdKernelQueue;

    fn available_sdma_engines(&self, device: &DeviceState) -> Result<u32, Error> {
        self.ensure_open()?;
        if cfg!(target_arch = "x86_64")
            && device.native.queues.target_profile == crate::topology::GpuTargetProfile::Gfx9
            && device.native.queues.sdma_qualified
        {
            let count = device.native.queues.sdma_engines.min(u32::BITS);
            return Ok(u32::MAX >> (u32::BITS - count));
        }
        if !cfg!(target_arch = "x86_64")
            || !device
                .native
                .queues
                .target_profile
                .supports_kernel_submission()
            || !device.native.queues.sdma_qualified
        {
            return Err(error(
                ErrorKind::Unsupported,
                "SDMA kernel queue is unqualified for this GPU target",
            ));
        }
        // Linux currently maps logical engine indices one-to-one to DRM's
        // qualified DMA ring indices. No DRM numbering escapes this driver.
        drm::sdma_available_rings(device.vm.render()?)
            .map_err(|source| native_error("DRM SDMA ring query", source))
    }

    fn create_kernel_queue(
        &self,
        device: &DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<KfdKernelQueue>, Error> {
        self.ensure_open()?;
        if !cfg!(target_arch = "x86_64")
            || !device
                .native
                .queues
                .target_profile
                .supports_kernel_submission()
        {
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
            KernelQueueFormat::Sdma | KernelQueueFormat::SdmaOnEngine(_)
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
}

impl GpuInstrumentationOperations for LinuxKfdDriver {
    type PcSampling = KfdPcSampling;

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

    fn pc_sampling_configurations(
        &self,
        device: &DeviceState,
        set: PcSamplingConfigurationSet,
    ) -> Result<Vec<PcSamplingConfiguration>, Error> {
        const MAX_CONFIGURATIONS: usize = 256;
        self.ensure_open()?;
        device.vm.check()?;
        if (device.vm.version.major, device.vm.version.minor) < (1, 16) {
            return Err(error(
                ErrorKind::Unsupported,
                "KFD PC sampling requires interface version 1.16",
            ));
        }
        let mut native = Vec::<uapi::PcSampleInfo>::new();
        for _ in 0..3 {
            let mut args = uapi::PcSampleArgs {
                operation: uapi::PC_SAMPLE_QUERY_CAPABILITIES,
                gpu_id: device.native.gpu_id,
                flags: match set {
                    PcSamplingConfigurationSet::Supported => uapi::PC_SAMPLE_QUERY_TYPE_FULL,
                    PcSamplingConfigurationSet::Available => 0,
                },
                ..uapi::PcSampleArgs::default()
            };
            let complete = match device.vm.kfd().pc_sample(&mut args, &mut native) {
                Ok(()) => true,
                Err(source) if source.raw_os_error() == Some(errno::ENOSPC) => false,
                Err(source) => {
                    return Err(native_error("KFD PC sampling capability query", source));
                }
            };
            let count = args.num_sample_info as usize;
            if count > MAX_CONFIGURATIONS {
                return Err(error(
                    ErrorKind::DriverContract,
                    "KFD PC sampling capability count exceeds the query bound",
                ));
            }
            if count > native.len() {
                native
                    .try_reserve_exact(count - native.len())
                    .map_err(|_| {
                        error(
                            ErrorKind::ResourceExhausted,
                            "PC sampling query storage unavailable",
                        )
                    })?;
                native.resize(count, uapi::PcSampleInfo::default());
                continue;
            }
            if !complete {
                return Err(error(
                    ErrorKind::DriverContract,
                    "KFD PC sampling query reported a full buffer as too small",
                ));
            }
            native.truncate(count);
            let mut configurations = Vec::new();
            configurations
                .try_reserve_exact(native.len())
                .map_err(|_| {
                    error(
                        ErrorKind::ResourceExhausted,
                        "PC sampling result storage unavailable",
                    )
                })?;
            for item in native {
                let method = match item.method {
                    uapi::PC_SAMPLE_METHOD_HOST_TRAP => PcSamplingMethod::HostTrap,
                    uapi::PC_SAMPLE_METHOD_STOCHASTIC => PcSamplingMethod::Stochastic,
                    _ => continue,
                };
                let unit = match item.unit {
                    uapi::PC_SAMPLE_UNIT_MICROSECONDS => PcSamplingIntervalUnit::Microseconds,
                    uapi::PC_SAMPLE_UNIT_CYCLES => PcSamplingIntervalUnit::ClockCycles,
                    uapi::PC_SAMPLE_UNIT_INSTRUCTIONS => PcSamplingIntervalUnit::Instructions,
                    _ => continue,
                };
                // An active registration is reported as one selected interval;
                // inactive methods carry the driver's supported bounds.
                let (minimum_interval, maximum_interval) =
                    if set == PcSamplingConfigurationSet::Available && item.interval != 0 {
                        (item.interval, item.interval)
                    } else {
                        (item.interval_min, item.interval_max)
                    };
                configurations.push(PcSamplingConfiguration {
                    method,
                    unit,
                    minimum_interval,
                    maximum_interval,
                    flags: item.flags,
                });
            }
            return Ok(configurations);
        }
        Err(error(
            ErrorKind::DriverContract,
            "KFD PC sampling capability count changed repeatedly",
        ))
    }

    fn create_pc_sampling(
        &self,
        device: &DeviceState,
        request: PcSamplingRequest,
        existing_trace_id: Option<std::num::NonZeroU32>,
    ) -> Result<Owned<KfdPcSampling>, Error> {
        self.ensure_open()?;
        device.vm.check()?;
        let supported = self
            .pc_sampling_configurations(device, PcSamplingConfigurationSet::Supported)?
            .into_iter()
            .any(|config| {
                config.method == request.method
                    && config.unit == request.unit
                    && request.interval >= config.minimum_interval
                    && request.interval <= config.maximum_interval
                    && (config.flags & uapi::PC_SAMPLE_FLAG_POWER_OF_TWO == 0
                        || request.interval.is_power_of_two())
            });
        if !supported {
            return Err(error(
                ErrorKind::Unsupported,
                "PC sampling method or interval is unavailable",
            ));
        }
        match existing_trace_id {
            Some(trace_id) => {
                KfdPcSampling::adopt(device.vm.clone(), device.native.gpu_id, trace_id)
            }
            None => KfdPcSampling::create(device.vm.clone(), device.native.gpu_id, request),
        }
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
        timeout: u32,
        destination: Option<usize>,
    ) -> Result<SpmDestinationReport, SpmDestinationFailure> {
        let mut args = uapi::Spm {
            destination: destination.map_or(0, |address| address as u64),
            size,
            operation: uapi::SPM_OP_SET_DESTINATION,
            timeout,
            gpu_id: device.native.gpu_id,
            bytes_copied: 0,
            has_data_loss: 0,
        };
        let result = device.vm.kfd().spm(&mut args);
        let report = SpmDestinationReport {
            timeout: args.timeout,
            bytes_copied: args.bytes_copied,
            data_loss: args.has_data_loss != 0,
        };
        result.map_or_else(
            |source| {
                Err(SpmDestinationFailure {
                    error: native_error("KFD SPM destination update", source),
                    report,
                })
            },
            |()| Ok(report),
        )
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
        let mut vms = crate::memory::storage::try_vec_with_capacity(devices.len())?;
        for device in devices {
            vms.try_push(&device.vm)?;
        }
        allocation.set_access(vms.as_slice())
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
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<LinuxAllocation>, Error> {
        let desc = checked_allocation_desc(size, alignment)?;
        if self.lifetime == DriverContextLifetime::Session
            && matches!(kind, MemoryKind::OwnedHost { .. })
        {
            return Err(error(
                ErrorKind::Unsupported,
                "secondary KFD contexts cannot bind host-owned pages",
            ));
        }
        if let MemoryKind::DeviceLocal {
            coherent,
            extended_coherent: true,
            uncached,
            ..
        } = kind
            && (!coherent || uncached)
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "extended local coherence requires a cached coherent mapping",
            ));
        }
        let native_kind = match kind {
            MemoryKind::System => memory::BufferKind::Gtt,
            MemoryKind::OwnedHost { policy } => memory::BufferKind::OwnedUserptr { cache: policy },
            MemoryKind::DeviceLocal {
                host_visible,
                coherent,
                extended_coherent,
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
                    extended_coherent,
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
            None,
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
            policy,
            size,
            alignment,
            permissions,
            backing,
        } = request;
        let desc = checked_host_registration_desc(size, alignment)?;
        let mut backing = backing;
        if self.lifetime == DriverContextLifetime::Session {
            if policy == HostMappingPolicy::Extended
                && std::iter::once(device)
                    .chain(peers.iter().copied())
                    .any(|peer| {
                        peer.native.queues.target_profile
                            != crate::topology::GpuTargetProfile::Gfx1201
                    })
            {
                return Err(error(
                    ErrorKind::Unsupported,
                    "extended DRM host registration requires GFX1201 mappings",
                ));
            }
            // On GFX1201 the KFD extended USERPTR allocation and a DRM
            // USERPTR object with the default VM page type both map as NC.
            // SAFETY: Raw callers retain the page cover; owned registrations
            // transfer a lease into the resulting native allocation.
            unsafe {
                LinuxAllocation::create_registered_host(
                    device.vm.clone(),
                    peers.iter().map(|peer| peer.vm.clone()),
                    desc,
                    address,
                    permissions,
                    policy == HostMappingPolicy::Uncached,
                    backing,
                )
            }
        } else {
            if let Some(alias) = memory::KfdHostAlias::try_create(
                &device.vm,
                peers,
                address,
                size,
                policy,
                permissions,
                &mut backing,
            )? {
                return LinuxAllocation::from_host_alias(alias);
            }
            // SAFETY: Raw callers retain these pages through cleanup; owned
            // registrations transfer a lease into the native allocation.
            let pages = unsafe { memory::BorrowedHostPages::new(address, policy) };
            LinuxAllocation::create_with_peers(
                device.vm.clone(),
                peers.iter().map(|peer| peer.vm.clone()),
                desc,
                memory::BufferKind::Userptr(pages),
                permissions,
                backing,
            )
        }
    }
}

fn checked_allocation_desc(size: u64, alignment: u64) -> Result<AllocationDesc, Error> {
    let desc = AllocationDesc { size, alignment };
    let page =
        os_memory::page_size().map_err(|source| native_error("native page size", source))? as u64;
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

fn checked_host_registration_desc(size: u64, alignment: u64) -> Result<AllocationDesc, Error> {
    let desc = AllocationDesc { size, alignment };
    let page =
        os_memory::page_size().map_err(|source| native_error("native page size", source))? as u64;
    if size == 0 || size > isize::MAX as u64 || !alignment.is_power_of_two() || alignment < page {
        return Err(error(
            ErrorKind::InvalidArgument,
            "invalid host registration extent or alignment",
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
        KfdVirtualAddress::reserve(bounds, size, alignment, address)
    }
    fn free_virtual_address(address: &mut KfdVirtualAddress) -> Result<(), Error> {
        address.free()
    }
    fn create_virtual_memory(
        &self,
        device: &DeviceState,
        kind: MemoryKind,
        size: u64,
        options: VirtualMemoryOptions,
    ) -> Result<Owned<KfdVirtualMemory>, Error> {
        self.ensure_open()?;
        KfdVirtualMemory::create(device.vm.clone(), kind, size, options)
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
    ) -> Result<Owned<KfdVirtualHostMapping>, Error> {
        KfdVirtualHostMapping::create(memory, reservation, address, offset, size, permissions)
    }
    fn set_virtual_host_mapping_permissions(
        mapping: &mut KfdVirtualHostMapping,
        permissions: DeviceAccess,
    ) -> Result<(), Error> {
        mapping.set_permissions(permissions)
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

    #[allow(unsafe_code)]
    unsafe fn destroy_with_retained_transport(
        &mut self,
        callback: &mut dyn FnMut(),
    ) -> Result<(), Error> {
        KfdQueue::destroy_with_retained_transport(self, callback)
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod registration_extent_tests {
    use super::{checked_allocation_desc, checked_host_registration_desc, os_memory};

    #[test]
    fn host_registration_accepts_an_exact_subpage_length() {
        let page = os_memory::page_size().unwrap() as u64;
        assert!(checked_host_registration_desc(1, page).is_ok());
        assert!(checked_allocation_desc(1, page).is_err());
        assert!(checked_host_registration_desc(0, page).is_err());
    }
}
