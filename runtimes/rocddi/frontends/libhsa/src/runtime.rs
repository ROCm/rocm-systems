//! Process-global HSA initialization, discovery, and object ownership.
//!
//! The single runtime registry owns the rocddi session and every HSA-visible
//! object map. Entry points hold its mutex only while validating or updating
//! registry state; blocking native operations, worker joins, and application
//! callbacks must occur after the relevant ownership has been moved out of the
//! lock. Final shutdown removes public reachability before releasing workers
//! and native resources.
//!
//! Host and GPU discovery currently consume Linux procfs, sysfs, DRM, and KFD
//! information. These probes are implementation details of the present Linux
//! frontend and must be replaced by a platform provider for Windows support.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt::Arguments;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rocddi::device::Device;
use rocddi::gpu::event::linux::{GpuMemoryFault, SignalEvent, poll_memory_fault};
use rocddi::memory::Allocation;
use rocddi::session::{Session, SessionLifetime};
use rocddi::topology::{Endpoint, GpuInfo};

use crate::ffi::*;
use crate::image::{Image, Sampler};
use crate::loader::{CodeObject, CodeSymbol, Executable, Reader, Symbol};
use crate::memory::{LockedMemory, Memory, VmemHandle, VmemMapping, VmemReservation};
use crate::pc_sampling::PcSamplingSession;
use crate::queue::{CountedHardwareQueue, CountedQueue, Queue, QueueSharedEvent, SoftQueue};
use crate::signal::{AsyncDispatcher, ImportedIpcSignal, OwnedIpcSignal, SignalSlab};

fn parse_host_memory_bytes(contents: &str) -> Option<usize> {
    let kilobytes = contents.lines().find_map(|line| {
        let mut fields = line.split_ascii_whitespace();
        (fields.next()? == "MemTotal:")
            .then(|| fields.next()?.parse::<u64>().ok())
            .flatten()
    })?;
    usize::try_from(kilobytes.checked_mul(1024)?).ok()
}

fn host_memory_bytes() -> Result<usize, Status> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| parse_host_memory_bytes(&contents))
        .filter(|bytes| *bytes != 0)
        .ok_or(ERROR)
}

/// Host identity used to materialize the HSA CPU agent.
#[derive(Debug, Eq, PartialEq)]
struct HostInfo {
    name: String,
    compute_units: u32,
}

fn parse_host_info(contents: &str) -> Option<HostInfo> {
    let mut name = None;
    let mut compute_units = 0_usize;
    for line in contents.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" if value.trim().parse::<u32>().is_ok() => {
                compute_units = compute_units.checked_add(1)?;
            }
            "model name" if name.is_none() && !value.trim().is_empty() => {
                name = Some(value.trim().to_owned());
            }
            _ => (),
        }
    }
    Some(HostInfo {
        name: name?,
        compute_units: u32::try_from(compute_units)
            .ok()
            .filter(|count| *count != 0)?,
    })
}

fn host_info() -> HostInfo {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| parse_host_info(&contents))
        .unwrap_or_else(|| HostInfo {
            name: "CPU".to_owned(),
            compute_units: 0,
        })
}

fn numeric_directories(root: &Path, prefix: &str) -> Vec<(u32, std::path::PathBuf)> {
    let mut entries = std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let ordinal = name.strip_prefix(prefix)?.parse().ok()?;
            Some((ordinal, entry.path()))
        })
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    entries
}

fn first_cpu(contents: &str) -> Option<u32> {
    let contents = contents.trim_start();
    let length = contents.bytes().take_while(u8::is_ascii_digit).count();
    contents.get(..length)?.parse().ok()
}

fn parse_cache_size(contents: &str) -> Option<u32> {
    let contents = contents.trim();
    let length = contents.bytes().take_while(u8::is_ascii_digit).count();
    let value = contents.get(..length)?.parse::<u32>().ok()?;
    let multiplier = match contents.get(length..)?.trim() {
        "" => 1,
        "K" => 1024,
        "M" => 1024 * 1024,
        "G" => 1024 * 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(multiplier)
}

fn discover_host_caches(root: &Path, agent: HsaAgent, name: &[u8]) -> Vec<Cache> {
    let mut caches = Vec::new();
    for (cpu, cpu_path) in numeric_directories(root, "cpu") {
        for (_, cache_path) in numeric_directories(&cpu_path.join("cache"), "index") {
            let shared = std::fs::read_to_string(cache_path.join("shared_cpu_list"))
                .ok()
                .and_then(|contents| first_cpu(&contents));
            if shared != Some(cpu) {
                continue;
            }
            if std::fs::read_to_string(cache_path.join("type"))
                .ok()
                .is_none_or(|kind| kind.trim() != "Data")
            {
                continue;
            }
            let Some(level) = std::fs::read_to_string(cache_path.join("level"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(size) = std::fs::read_to_string(cache_path.join("size"))
                .ok()
                .and_then(|value| parse_cache_size(&value))
            else {
                continue;
            };
            caches.push(Cache::new(agent, name, level, size));
        }
    }
    caches
}

fn host_caches(agent: HsaAgent, name: &[u8]) -> Vec<Cache> {
    let numa = Path::new("/sys/devices/system/node/node0");
    let root = if numa.is_dir() {
        numa
    } else {
        Path::new("/sys/devices/system/cpu")
    };
    discover_host_caches(root, agent, name)
}

fn gpu_agent_name(gpu: &GpuInfo) -> String {
    format!("gfx{}{}{}", gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping)
}

#[link(name = "libdrm_amdgpu.so.1", kind = "dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    fn amdgpu_device_initialize(
        descriptor: c_int,
        major: *mut u32,
        minor: *mut u32,
        device: *mut *mut c_void,
    ) -> c_int;
    fn amdgpu_device_deinitialize(device: *mut c_void) -> c_int;
    fn amdgpu_get_marketing_name(device: *mut c_void) -> *const c_char;
    fn amdgpu_query_info(device: *mut c_void, info_id: u32, size: u32, value: *mut c_void)
    -> c_int;
}

const AMDGPU_INFO_DEV_INFO: u32 = 0x16;

#[derive(Default)]
struct DrmGpuInfo {
    product_name: Option<String>,
    family_id: Option<u32>,
}

#[derive(Default)]
#[repr(C)]
struct DrmAmdgpuDeviceInfoPrefix {
    device_id: u32,
    chip_revision: u32,
    external_revision: u32,
    pci_revision: u32,
    family_id: u32,
}

fn select_asic_family_id(topology: u32, drm: Option<u32>) -> u32 {
    drm.filter(|family| *family != 0).unwrap_or(topology)
}

fn hdp_flush_pointers(address: Option<usize>) -> [usize; 2] {
    address.map_or([0; 2], |address| [address, address + 4])
}

fn full_profile_platform(local_memory_bytes: impl IntoIterator<Item = u64>) -> bool {
    let mut local_memory_bytes = local_memory_bytes.into_iter();
    local_memory_bytes
        .next()
        .is_some_and(|bytes| bytes == 0 && local_memory_bytes.all(|bytes| bytes == 0))
}

fn gpu_drm_info(render_minor: u32) -> Option<DrmGpuInfo> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(format!("/dev/dri/renderD{render_minor}"))
        .ok()?;
    let mut major = 0;
    let mut minor = 0;
    let mut device = std::ptr::null_mut();
    // SAFETY: libdrm receives a live render descriptor and writable outputs.
    let status = unsafe {
        amdgpu_device_initialize(
            file.as_raw_fd(),
            &raw mut major,
            &raw mut minor,
            &raw mut device,
        )
    };
    if status != 0 || device.is_null() {
        return None;
    }
    // SAFETY: The initialized device owns this NUL-terminated name until it is
    // deinitialized. Copy it before releasing the device.
    let product_name = unsafe {
        let name = amdgpu_get_marketing_name(device);
        (!name.is_null()).then(|| CStr::from_ptr(name).to_string_lossy().into_owned())
    };
    let mut info = DrmAmdgpuDeviceInfoPrefix::default();
    // SAFETY: The DRM query accepts a prefix of the stable device-info UAPI
    // record and receives writable storage of the advertised size.
    let query_status = unsafe {
        amdgpu_query_info(
            device,
            AMDGPU_INFO_DEV_INFO,
            std::mem::size_of::<DrmAmdgpuDeviceInfoPrefix>() as u32,
            (&raw mut info).cast(),
        )
    };
    // SAFETY: This device was returned by the successful initialization above.
    let _ = unsafe { amdgpu_device_deinitialize(device) };
    Some(DrmGpuInfo {
        product_name,
        family_id: (query_status == 0).then_some(info.family_id),
    })
}

fn notify_system_shutdown(handlers: &[(SystemEventHandler, usize)]) {
    let event = HsaAmdEvent {
        event_type: AMD_SYSTEM_SHUTDOWN_EVENT,
        payload: [0; 3],
    };
    let _ = notify_system_event(handlers, &event);
}

fn notify_system_event(handlers: &[(SystemEventHandler, usize)], event: &HsaAmdEvent) -> bool {
    let mut handled = false;
    for (callback, data) in handlers {
        // SAFETY: Registration supplies an ABI-compatible callback. The event
        // remains live for the duration of each synchronous invocation.
        handled |= unsafe { callback(event, *data as *mut c_void) } == SUCCESS;
    }
    handled
}

fn memory_fault_event(agent: HsaAgent, fault: GpuMemoryFault) -> HsaAmdEvent {
    let mut reason = 0;
    if fault.page_not_present {
        reason |= AMD_MEMORY_FAULT_PAGE_NOT_PRESENT;
    }
    if fault.read_only {
        reason |= AMD_MEMORY_FAULT_READ_ONLY;
    }
    if fault.no_execute {
        reason |= AMD_MEMORY_FAULT_NO_EXECUTE;
    }
    if fault.imprecise {
        reason |= AMD_MEMORY_FAULT_IMPRECISE;
    }
    reason |= match fault.error_type {
        1 => AMD_MEMORY_FAULT_SRAM_ECC,
        2 => AMD_MEMORY_FAULT_DRAM_ECC,
        3 => AMD_MEMORY_FAULT_HANG,
        _ => 0,
    };
    HsaAmdEvent {
        event_type: AMD_GPU_MEMORY_FAULT_EVENT,
        payload: [agent.handle, fault.virtual_address, u64::from(reason)],
    }
}

fn system_event_worker(device: &Device, stop: &AtomicBool) {
    let Ok(gpu_device) = device.gpu() else {
        return;
    };
    while !stop.load(Ordering::Acquire) {
        match poll_memory_fault(gpu_device) {
            Ok(Some(fault)) => {
                let notification = {
                    let Ok(guard) = lock() else {
                        return;
                    };
                    let Some(runtime) = guard.as_ref() else {
                        return;
                    };
                    let Some(index) = runtime.gpus.iter().position(|gpu| {
                        gpu.endpoint.linux_kfd_drm_info().gpu_id == fault.kfd_gpu_id
                    }) else {
                        return;
                    };
                    (
                        memory_fault_event(
                            HsaAgent {
                                handle: GPU_AGENT_BASE + index as u64,
                            },
                            fault,
                        ),
                        runtime.system_event_handlers.clone(),
                    )
                };
                if !notify_system_event(&notification.1, &notification.0) {
                    std::process::abort();
                }
                return;
            }
            Ok(None) => thread::sleep(Duration::from_micros(20)),
            Err(_) => return,
        }
    }
}

fn decode_gpu_pool_handle(pool: HsaMemoryPool, gpu_count: usize) -> Option<(usize, u64)> {
    let offset = pool.handle.checked_sub(GPU_POOL_BASE)?;
    let index = usize::try_from(offset / 0x10).ok()?;
    let kind = offset % 0x10;
    (index < gpu_count && matches!(kind, 1..=3)).then_some((index, kind))
}

fn decode_cache_handle(cache: HsaCache, cache_count: usize) -> Option<usize> {
    let index = cache.handle.checked_sub(CACHE_BASE)?;
    let index = usize::try_from(index).ok()?;
    (index < cache_count).then_some(index)
}

fn cache_sizes(caches: &[Cache], agent: HsaAgent) -> [u32; 4] {
    let mut sizes = [0_u32; 4];
    for cache in caches.iter().filter(|cache| cache.agent == agent) {
        let Some(index) = usize::from(cache.level)
            .checked_sub(1)
            .filter(|index| *index < sizes.len())
        else {
            continue;
        };
        let bytes = cache.size.wrapping_mul(1024);
        if index == 0 {
            if sizes[index] == 0 {
                sizes[index] = bytes;
            }
        } else {
            sizes[index] = sizes[index].wrapping_add(bytes);
        }
    }
    sizes
}

unsafe extern "C" {
    fn fwrite(buffer: *const c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
    fn fflush(stream: *mut c_void) -> c_int;
}

/// Activated GPU and the immutable compatibility facts derived at startup.
pub(crate) struct Gpu {
    pub(crate) endpoint: Endpoint,
    pub(crate) info: GpuInfo,
    pub(crate) device: Device,
    pub(crate) name: Box<str>,
    pub(crate) product_name: Box<str>,
    pub(crate) asic_family_id: u32,
    pub(crate) hdp_flush: [usize; 2],
    _mmio_remap: Option<Allocation>,
    pub(crate) coherency_type: u32,
    pub(crate) fine_grain_pool: bool,
}

/// Stable HSA cache object associated with one CPU or GPU agent.
pub(crate) struct Cache {
    pub(crate) agent: HsaAgent,
    pub(crate) name: Box<[u8]>,
    pub(crate) level: u8,
    pub(crate) size: u32,
}

impl Cache {
    fn new(agent: HsaAgent, agent_name: &[u8], level: u32, size: u32) -> Self {
        let name_length = agent_name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(agent_name.len())
            .min(63);
        let mut name = Vec::with_capacity(name_length + 13);
        name.extend_from_slice(&agent_name[..name_length]);
        name.extend_from_slice(b" L");
        name.extend_from_slice(level.to_string().as_bytes());
        name.push(0);
        Self {
            agent,
            name: name.into_boxed_slice(),
            level: level as u8,
            size,
        }
    }
}

/// Complete process-global HSA state.
///
/// Every integer handle and public pointer accepted by this frontend must map
/// to an owner in this structure. Removing an entry transfers that owner to the
/// entry point performing destruction, allowing callbacks, blocking cleanup,
/// and worker joins to happen after the global mutex is released.
pub(crate) struct Runtime {
    pub(crate) references: u32,
    pub(crate) log_flags: [u8; 8],
    pub(crate) log_file: usize,
    pub(crate) next_handle: u64,
    pub(crate) next_queue_id: u64,
    pub(crate) code_objects: HashMap<u64, CodeObject>,
    pub(crate) code_symbols: HashMap<u64, CodeSymbol>,
    pub(crate) readers: HashMap<u64, Reader>,
    pub(crate) executables: HashMap<u64, Executable>,
    pub(crate) symbols: HashMap<u64, Symbol>,
    pub(crate) allocations: HashMap<usize, Memory>,
    pub(crate) ipc_allocations: HashMap<usize, Memory>,
    pub(crate) interop_allocations: HashMap<usize, Memory>,
    pub(crate) locked_allocations: Vec<LockedMemory>,
    pub(crate) vmem_reservations: BTreeMap<usize, VmemReservation>,
    pub(crate) vmem_handles: HashMap<u64, VmemHandle>,
    pub(crate) vmem_mappings: BTreeMap<usize, VmemMapping>,
    pub(crate) signal_slabs: Vec<SignalSlab>,
    pub(crate) signal_event_page: Option<Allocation>,
    pub(crate) queue_event: Option<Arc<QueueSharedEvent>>,
    pub(crate) interrupt_signals: HashMap<usize, Option<SignalEvent>>,
    pub(crate) signal_event_pool: Vec<SignalEvent>,
    pub(crate) owned_ipc_signals: HashMap<usize, OwnedIpcSignal>,
    pub(crate) imported_ipc_signals: HashMap<usize, ImportedIpcSignal>,
    pub(crate) signal_groups: HashMap<u64, Vec<HsaSignal>>,
    pub(crate) images: HashMap<u64, Image>,
    pub(crate) samplers: HashMap<u64, Sampler>,
    pub(crate) pc_sampling: HashMap<u64, Arc<Mutex<PcSamplingSession>>>,
    pub(crate) pc_sampling_agents: HashMap<usize, u64>,
    pub(crate) queues: HashMap<usize, Queue>,
    pub(crate) soft_queues: HashMap<usize, SoftQueue>,
    pub(crate) counted_queues: HashMap<usize, CountedQueue>,
    pub(crate) counted_queue_pools: HashMap<(u64, u32), Vec<CountedHardwareQueue>>,
    pub(crate) released_counted_queues: HashSet<usize>,
    pub(crate) counted_queue_limit: usize,
    pub(crate) counted_queue_size: u32,
    pub(crate) host_memory_bytes: usize,
    pub(crate) host_name: Box<str>,
    pub(crate) host_compute_units: u32,
    pub(crate) full_profile: bool,
    pub(crate) system_event_handlers: Vec<(SystemEventHandler, usize)>,
    pub(crate) system_event_worker_started: bool,
    pub(crate) async_dispatcher: Option<AsyncDispatcher>,
    pub(crate) workers: Vec<JoinHandle<()>>,
    pub(crate) stop_workers: Arc<AtomicBool>,
    pub(crate) caches: Vec<Cache>,
    pub(crate) gpus: Vec<Gpu>,
    pub(crate) session: Session,
}

impl Runtime {
    pub(crate) fn create() -> Result<Self, Status> {
        let host_memory_bytes = host_memory_bytes()?;
        let host = host_info();
        let force_fine_grain_pcie =
            std::env::var("HSA_FORCE_FINE_GRAIN_PCIE").is_ok_and(|value| value == "1");
        let session = Session::new(SessionLifetime::Process).map_err(map_error)?;
        let mut endpoints = Vec::new();
        session
            .enumerate(&mut |endpoint| {
                if endpoint.gpu().is_some_and(|gpu| gpu.queues.aql) {
                    endpoints.push(endpoint);
                }
                Ok(())
            })
            .map_err(map_error)?;
        if endpoints.is_empty() {
            return Err(OUT_OF_RESOURCES);
        }
        let full_profile =
            full_profile_platform(endpoints.iter().map(|endpoint| endpoint.local_memory_bytes));
        let mut caches = host_caches(HsaAgent { handle: CPU_AGENT }, host.name.as_bytes());
        let mut gpus = Vec::with_capacity(endpoints.len());
        for endpoint in endpoints {
            let Some(info) = endpoint.gpu().copied() else {
                continue;
            };
            let agent = HsaAgent {
                handle: GPU_AGENT_BASE + gpus.len() as u64,
            };
            let name = gpu_agent_name(&info);
            for cache in endpoint
                .caches()
                .iter()
                .filter(|cache| rocddi::gpu::is_compute_data_cache(cache))
            {
                caches.push(Cache::new(
                    agent,
                    name.as_bytes(),
                    cache.level(),
                    cache.size(),
                ));
            }
            let device = session.activate(&endpoint).map_err(map_error)?;
            let drm_info = endpoint
                .linux_kfd_drm_info()
                .render_minor
                .and_then(gpu_drm_info)
                .unwrap_or_default();
            let product_name = drm_info
                .product_name
                .unwrap_or_else(|| "AMD Radeon Graphics".to_owned());
            let asic_family_id = select_asic_family_id(info.asic_family_id, drm_info.family_id);
            let mmio_remap = device.gpu().and_then(|gpu| gpu.map_mmio_remap()).ok();
            let hdp_flush = hdp_flush_pointers(
                mmio_remap
                    .as_ref()
                    .and_then(|mapping| mapping.info().host_address),
            );
            let fine_grain_pool = info.hive_id != 0 || force_fine_grain_pcie;
            gpus.push(Gpu {
                endpoint,
                info,
                device,
                name: name.into_boxed_str(),
                product_name: product_name.into_boxed_str(),
                asic_family_id,
                hdp_flush,
                _mmio_remap: mmio_remap,
                coherency_type: AMD_COHERENCY_TYPE_NONCOHERENT,
                fine_grain_pool,
            });
        }
        Ok(Self {
            references: 1,
            log_flags: [0; 8],
            log_file: 0,
            next_handle: 0x4853_4101_0000_0000,
            next_queue_id: 0,
            code_objects: HashMap::new(),
            code_symbols: HashMap::new(),
            readers: HashMap::new(),
            executables: HashMap::new(),
            symbols: HashMap::new(),
            allocations: HashMap::new(),
            ipc_allocations: HashMap::new(),
            interop_allocations: HashMap::new(),
            locked_allocations: Vec::new(),
            vmem_reservations: BTreeMap::new(),
            vmem_handles: HashMap::new(),
            vmem_mappings: BTreeMap::new(),
            signal_slabs: Vec::new(),
            signal_event_page: None,
            queue_event: None,
            interrupt_signals: HashMap::new(),
            signal_event_pool: Vec::new(),
            owned_ipc_signals: HashMap::new(),
            imported_ipc_signals: HashMap::new(),
            signal_groups: HashMap::new(),
            images: HashMap::new(),
            samplers: HashMap::new(),
            pc_sampling: HashMap::new(),
            pc_sampling_agents: HashMap::new(),
            queues: HashMap::new(),
            soft_queues: HashMap::new(),
            counted_queues: HashMap::new(),
            counted_queue_pools: HashMap::new(),
            released_counted_queues: HashSet::new(),
            counted_queue_limit: environment_u32("GPU_MAX_HW_QUEUES", 4) as usize,
            counted_queue_size: environment_u32("HSA_COUNTED_QUEUE_SIZE", 16_384),
            host_memory_bytes,
            host_name: host.name.into_boxed_str(),
            host_compute_units: host.compute_units,
            full_profile,
            system_event_handlers: Vec::new(),
            system_event_worker_started: false,
            async_dispatcher: None,
            workers: Vec::new(),
            stop_workers: Arc::new(AtomicBool::new(false)),
            caches,
            gpus,
            session,
        })
    }

    pub(crate) fn allocate_handle(&mut self) -> Result<u64, Status> {
        let handle = self.next_handle;
        self.next_handle = self.next_handle.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(handle)
    }

    pub(crate) fn ensure_system_event_worker(&mut self) -> Status {
        if self.system_event_worker_started {
            return SUCCESS;
        }
        let Some(device) = self.gpus.first().map(|gpu| gpu.device.clone()) else {
            return OUT_OF_RESOURCES;
        };
        let stop = self.stop_workers.clone();
        let worker = match thread::Builder::new()
            .name("rocddi-system-events".into())
            .spawn(move || system_event_worker(&device, &stop))
        {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        self.workers.push(worker);
        self.system_event_worker_started = true;
        SUCCESS
    }

    pub(crate) fn allocate_queue_id(&mut self) -> Result<u64, Status> {
        let id = self.next_queue_id;
        self.next_queue_id = self.next_queue_id.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(id)
    }

    pub(crate) fn gpu_index(&self, agent: HsaAgent) -> Option<usize> {
        let offset = agent.handle.checked_sub(GPU_AGENT_BASE)?;
        let index = usize::try_from(offset).ok()?;
        (index < self.gpus.len()).then_some(index)
    }

    pub(crate) fn is_agent(&self, agent: HsaAgent) -> bool {
        agent.handle == CPU_AGENT || self.gpu_index(agent).is_some()
    }

    pub(crate) fn cache_index(&self, cache: HsaCache) -> Option<usize> {
        decode_cache_handle(cache, self.caches.len())
    }

    pub(crate) fn agent_cache_sizes(&self, agent: HsaAgent) -> [u32; 4] {
        cache_sizes(&self.caches, agent)
    }

    pub(crate) fn system_timestamp(&self) -> Result<u64, Status> {
        self.gpus[0]
            .device
            .gpu()
            .and_then(|gpu| gpu.clock_counters())
            .map(|counters| counters.system)
            .map_err(map_error)
    }

    pub(crate) fn translate_gpu_tick(&self, index: usize, tick: u64) -> Result<u64, Status> {
        let counters = self.gpus[index]
            .device
            .gpu()
            .and_then(|gpu| gpu.clock_counters())
            .map_err(map_error)?;
        translate_gpu_tick(counters, tick)
    }

    pub(crate) fn translate_gpu_interval(
        &self,
        index: usize,
        start: u64,
        end: u64,
    ) -> Result<(u64, u64), Status> {
        if start == 0 || end == 0 {
            return Ok((0, 0));
        }
        let counters = self.gpus[index]
            .device
            .gpu()
            .and_then(|gpu| gpu.clock_counters())
            .map_err(map_error)?;
        Ok((
            translate_gpu_tick(counters, start)?,
            translate_gpu_tick(counters, end)?,
        ))
    }

    pub(crate) fn isa_parts(&self, isa: HsaIsa) -> Option<(usize, u64)> {
        let offset = isa.handle.checked_sub(ISA_BASE)?;
        let index = usize::try_from(offset / ISA_COUNT_PER_GPU).ok()?;
        let variant = offset % ISA_COUNT_PER_GPU;
        (index < self.gpus.len()).then_some((index, variant))
    }

    pub(crate) fn isa_index(&self, isa: HsaIsa) -> Option<usize> {
        self.isa_parts(isa).map(|(index, _)| index)
    }

    pub(crate) fn wavefront_isa(&self, wavefront: HsaWavefront) -> Option<HsaIsa> {
        let offset = wavefront.handle.checked_sub(WAVEFRONT_BASE)?;
        let isa = HsaIsa {
            handle: ISA_BASE.checked_add(offset)?,
        };
        self.isa_parts(isa).map(|_| isa)
    }

    pub(crate) fn gpu_pool(index: usize, kind: u64) -> HsaMemoryPool {
        HsaMemoryPool {
            handle: GPU_POOL_BASE + (index as u64) * 0x10 + kind,
        }
    }

    pub(crate) fn decode_gpu_pool(&self, pool: HsaMemoryPool) -> Option<(usize, u64)> {
        let (index, kind) = decode_gpu_pool_handle(pool, self.gpus.len())?;
        (kind != 2 || self.gpus[index].fine_grain_pool).then_some((index, kind))
    }

    pub(crate) fn log(&self, flag: u32, arguments: Arguments<'_>) {
        if !logging_flag_enabled(self.log_flags, flag) {
            return;
        }
        let mut line = String::from("[***rocddi***] ");
        if std::fmt::write(&mut line, arguments).is_err() {
            return;
        }
        line.push('\n');
        if self.log_file == 0 {
            let mut stream = std::io::stderr().lock();
            let _ = stream.write_all(line.as_bytes());
            let _ = stream.flush();
        } else {
            // SAFETY: hsa_amd_enable_logging requires a live C FILE pointer.
            // libc serializes writes to a shared stream.
            unsafe {
                fwrite(
                    line.as_ptr().cast(),
                    1,
                    line.len(),
                    self.log_file as *mut c_void,
                );
                fflush(self.log_file as *mut c_void);
            }
        }
    }

    pub(crate) fn stop(mut self) -> Status {
        notify_system_shutdown(&self.system_event_handlers);
        self.pc_sampling_agents.clear();
        let pc_sampling_status =
            crate::pc_sampling::destroy_sessions(std::mem::take(&mut self.pc_sampling));
        self.stop_workers.store(true, Ordering::Release);
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        self.counted_queues.clear();
        self.counted_queue_pools.clear();
        self.released_counted_queues.clear();
        for (_, mut queue) in self.queues.drain() {
            if crate::queue::destroy_runtime_queue(&mut queue).is_err() {
                // An unresolved native queue can still reference its scratch
                // and inactive signal. Preserve the complete dependency set
                // for KFD process teardown instead of freeing reachable pages.
                std::mem::forget(queue);
            }
        }
        self.queue_event = None;
        self.soft_queues.clear();
        let signal_status = self.destroy_signal_events();
        self.symbols.clear();
        self.executables.clear();
        self.readers.clear();
        self.code_symbols.clear();
        self.code_objects.clear();
        self.locked_allocations.clear();
        self.vmem_mappings.clear();
        self.vmem_handles.clear();
        self.vmem_reservations.clear();
        self.ipc_allocations.clear();
        self.interop_allocations.clear();
        self.allocations.clear();
        self.signal_groups.clear();
        self.images.clear();
        self.samplers.clear();
        self.imported_ipc_signals.clear();
        self.owned_ipc_signals.clear();
        self.signal_slabs.clear();
        self.gpus.clear();
        let session_status = self.session.destroy().map_or_else(map_error, |()| SUCCESS);
        if pc_sampling_status != SUCCESS {
            pc_sampling_status
        } else if signal_status != SUCCESS {
            signal_status
        } else {
            session_status
        }
    }
}

fn environment_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).map_or(default, |value| {
        value.trim().parse::<i32>().map_or(0, |value| value as u32)
    })
}

fn logging_flag_enabled(flags: [u8; 8], flag: u32) -> bool {
    usize::try_from(flag / 8)
        .ok()
        .and_then(|index| flags.get(index))
        .is_some_and(|byte| byte & (1 << (flag % 8)) != 0)
}

fn translate_gpu_tick(
    counters: rocddi::gpu::profiling::ClockCounters,
    tick: u64,
) -> Result<u64, Status> {
    if counters.system_frequency == 0 {
        return Err(ERROR);
    }
    let scaled = |delta: u64| {
        u64::try_from(u128::from(delta) * u128::from(counters.system_frequency) / 100_000_000_u128)
            .unwrap_or(u64::MAX)
    };
    Ok(if tick >= counters.gpu {
        counters.system.wrapping_add(scaled(tick - counters.gpu))
    } else {
        counters.system.wrapping_sub(scaled(counters.gpu - tick))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LifecyclePhase {
    Inactive,
    Starting(u64),
    Active(u64),
    Stopping(u64),
    Quarantined,
}

pub(crate) struct RuntimeRegistry {
    runtime: Option<Runtime>,
    generation: u64,
    phase: LifecyclePhase,
}

impl RuntimeRegistry {
    const fn new() -> Self {
        Self {
            runtime: None,
            generation: 0,
            phase: LifecyclePhase::Inactive,
        }
    }

    /// Returns None after adding a reference to an already active runtime.
    pub(crate) fn begin_init(&mut self) -> Result<Option<u64>, Status> {
        match self.phase {
            LifecyclePhase::Active(_) => {
                let runtime = self.runtime.as_mut().ok_or(ERROR)?;
                if runtime.references == i32::MAX as u32 {
                    return Err(0x100c);
                }
                runtime.references += 1;
                Ok(None)
            }
            LifecyclePhase::Inactive => {
                self.generation = self.generation.checked_add(1).ok_or(ERROR)?;
                self.phase = LifecyclePhase::Starting(self.generation);
                Ok(Some(self.generation))
            }
            LifecyclePhase::Starting(_)
            | LifecyclePhase::Stopping(_)
            | LifecyclePhase::Quarantined => Err(INVALID_RUNTIME_STATE),
        }
    }

    pub(crate) fn publish_init(&mut self, generation: u64, runtime: Runtime) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.runtime = Some(runtime);
        self.phase = LifecyclePhase::Active(generation);
        Ok(())
    }

    pub(crate) fn cancel_init(&mut self, generation: u64) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.phase = LifecyclePhase::Inactive;
        Ok(())
    }

    /// Returns None after releasing one of several active references.
    pub(crate) fn begin_shutdown(&mut self) -> Result<Option<(Runtime, u64)>, Status> {
        match self.phase {
            LifecyclePhase::Active(generation) => {
                let runtime = self.runtime.as_mut().ok_or(ERROR)?;
                if runtime.references > 1 {
                    runtime.references -= 1;
                    return Ok(None);
                }
                let runtime = self.runtime.take().ok_or(ERROR)?;
                self.phase = LifecyclePhase::Stopping(generation);
                Ok(Some((runtime, generation)))
            }
            LifecyclePhase::Inactive => Err(NOT_INITIALIZED),
            LifecyclePhase::Starting(_)
            | LifecyclePhase::Stopping(_)
            | LifecyclePhase::Quarantined => Err(INVALID_RUNTIME_STATE),
        }
    }

    pub(crate) fn finish_shutdown(
        &mut self,
        generation: u64,
        status: Status,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Stopping(generation) {
            return Err(ERROR);
        }
        // A failed stop can leave native resources alive. Do not create a new
        // runtime generation over their process-wide KFD state.
        self.phase = if status == SUCCESS {
            LifecyclePhase::Inactive
        } else {
            LifecyclePhase::Quarantined
        };
        Ok(())
    }
}

impl Deref for RuntimeRegistry {
    type Target = Option<Runtime>;

    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

impl DerefMut for RuntimeRegistry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runtime
    }
}

/// Quarantines a generation if initialization or shutdown unwinds.
pub(crate) struct LifecycleTransition {
    phase: LifecyclePhase,
    armed: bool,
}

impl LifecycleTransition {
    pub(crate) fn starting(generation: u64) -> Self {
        Self {
            phase: LifecyclePhase::Starting(generation),
            armed: true,
        }
    }

    pub(crate) fn stopping(generation: u64) -> Self {
        Self {
            phase: LifecyclePhase::Stopping(generation),
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LifecycleTransition {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut registry = RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if registry.phase == self.phase {
            registry.phase = LifecyclePhase::Quarantined;
        }
    }
}

pub(crate) static RUNTIME: Mutex<RuntimeRegistry> = Mutex::new(RuntimeRegistry::new());

pub(crate) fn lock() -> Result<MutexGuard<'static, RuntimeRegistry>, Status> {
    RUNTIME.lock().map_err(|_| ERROR)
}

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn map_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
        rocddi::ErrorKind::Busy => INVALID_QUEUE,
        _ => ERROR,
    }
}

pub(crate) fn boundary(operation: impl FnOnce() -> Status) -> Status {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(ERROR)
}

pub(crate) fn initialized_mut(runtime: &mut Option<Runtime>) -> Result<&mut Runtime, Status> {
    runtime.as_mut().ok_or(NOT_INITIALIZED)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rocddi::gpu::profiling::ClockCounters;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    static NEXT_FIXTURE: AtomicU32 = AtomicU32::new(0);

    #[test]
    fn initialization_gate_rejects_overlapping_generations() {
        let mut registry = RuntimeRegistry::new();
        let first = registry.begin_init().unwrap().unwrap();
        assert_eq!(registry.begin_init().unwrap_err(), INVALID_RUNTIME_STATE);
        assert_eq!(registry.begin_shutdown().err(), Some(INVALID_RUNTIME_STATE));
        registry.cancel_init(first).unwrap();
        let second = registry.begin_init().unwrap().unwrap();
        assert_ne!(second, first);
        registry.cancel_init(second).unwrap();
    }

    struct ObservedEvent {
        count: AtomicU32,
        event_type: AtomicU32,
    }

    unsafe extern "C" fn observe_system_event(
        event: *const HsaAmdEvent,
        data: *mut c_void,
    ) -> Status {
        // SAFETY: The test passes live event and observation storage.
        let observed = unsafe { &*data.cast::<ObservedEvent>() };
        // SAFETY: The event pointer is live for this synchronous callback.
        let event = unsafe { &*event };
        observed.count.fetch_add(1, AtomicOrdering::Relaxed);
        observed
            .event_type
            .store(event.event_type, AtomicOrdering::Relaxed);
        ERROR
    }

    struct Fixture(std::path::PathBuf);

    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "libhsa-host-cache-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, AtomicOrdering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn cache(&self, cpu: u32, index: u32, shared: &str, kind: &str, level: u32, size: &str) {
            let path = self.0.join(format!("cpu{cpu}/cache/index{index}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("shared_cpu_list"), shared).unwrap();
            std::fs::write(path.join("type"), kind).unwrap();
            std::fs::write(path.join("level"), level.to_string()).unwrap();
            std::fs::write(path.join("size"), size).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn host_cpuinfo_matches_rocr_agent_identity() {
        assert_eq!(
            parse_host_info(
                "processor : 0\nmodel name : AMD Example CPU\n\
                 processor : 1\nmodel name : AMD Example CPU\n"
            ),
            Some(HostInfo {
                name: "AMD Example CPU".to_owned(),
                compute_units: 2,
            })
        );
        assert_eq!(parse_host_info("processor : 0\n"), None);
        assert_eq!(parse_host_info("model name : AMD Example CPU\n"), None);
    }

    #[test]
    fn host_cache_discovery_keeps_one_copy_of_each_data_cache() {
        let fixture = Fixture::new();
        fixture.cache(0, 0, "0-1\n", "Data\n", 1, "48K\n");
        fixture.cache(0, 1, "0-1\n", "Instruction\n", 1, "32K\n");
        fixture.cache(0, 2, "0-3\n", "Unified\n", 2, "1M\n");
        fixture.cache(1, 0, "0-1\n", "Data\n", 1, "48K\n");
        let cpu_agent = HsaAgent { handle: CPU_AGENT };
        let caches = discover_host_caches(&fixture.0, cpu_agent, b"AMD Example CPU");
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].agent.handle, CPU_AGENT);
        assert_eq!(&*caches[0].name, b"AMD Example CPU L1\0");
        assert_eq!(caches[0].level, 1);
        assert_eq!(caches[0].size, 48 * 1024);
        assert_eq!(cache_sizes(&caches, cpu_agent), [48 * 1024 * 1024, 0, 0, 0]);
    }

    #[test]
    fn canonical_gpu_agent_name_uses_the_gfx_target() {
        assert_eq!(
            gpu_agent_name(&GpuInfo {
                gfx_major: 12,
                gfx_minor: 0,
                gfx_stepping: 1,
                ..GpuInfo::default()
            }),
            "gfx1201"
        );
    }

    #[test]
    fn drm_device_info_supplies_missing_asic_family_id() {
        assert_eq!(std::mem::size_of::<DrmAmdgpuDeviceInfoPrefix>(), 20);
        assert_eq!(std::mem::align_of::<DrmAmdgpuDeviceInfoPrefix>(), 4);
        assert_eq!(
            std::mem::offset_of!(DrmAmdgpuDeviceInfoPrefix, family_id),
            16
        );
        assert_eq!(select_asic_family_id(0, Some(0x98)), 0x98);
        assert_eq!(select_asic_family_id(0x91, Some(0)), 0x91);
        assert_eq!(select_asic_family_id(0x91, None), 0x91);
    }

    #[test]
    fn hdp_flush_registers_are_adjacent_words_in_the_mmio_page() {
        assert_eq!(hdp_flush_pointers(None), [0, 0]);
        assert_eq!(hdp_flush_pointers(Some(0x1000)), [0x1000, 0x1004]);
    }

    #[test]
    fn zero_local_memory_identifies_a_full_profile_platform() {
        assert!(!full_profile_platform([]));
        assert!(full_profile_platform([0]));
        assert!(full_profile_platform([0, 0]));
        assert!(!full_profile_platform([0, 1]));
    }

    #[test]
    fn shutdown_notifies_every_registered_system_event_handler() {
        let first = ObservedEvent {
            count: AtomicU32::new(0),
            event_type: AtomicU32::new(u32::MAX),
        };
        let second = ObservedEvent {
            count: AtomicU32::new(0),
            event_type: AtomicU32::new(u32::MAX),
        };
        notify_system_shutdown(&[
            (
                observe_system_event,
                std::ptr::from_ref(&first).cast::<c_void>() as usize,
            ),
            (
                observe_system_event,
                std::ptr::from_ref(&second).cast::<c_void>() as usize,
            ),
        ]);
        assert_eq!(first.count.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(second.count.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(
            first.event_type.load(AtomicOrdering::Relaxed),
            AMD_SYSTEM_SHUTDOWN_EVENT
        );
        assert_eq!(
            second.event_type.load(AtomicOrdering::Relaxed),
            AMD_SYSTEM_SHUTDOWN_EVENT
        );
    }

    #[test]
    fn memory_fault_events_match_the_amd_extension_layout() {
        let agent = HsaAgent { handle: 0x1234 };
        for (error_type, extra_reason) in [
            (0, 0),
            (1, AMD_MEMORY_FAULT_SRAM_ECC),
            (2, AMD_MEMORY_FAULT_DRAM_ECC),
            (3, AMD_MEMORY_FAULT_HANG),
        ] {
            let event = memory_fault_event(
                agent,
                GpuMemoryFault {
                    kfd_gpu_id: 42,
                    virtual_address: 0x5678_9000,
                    page_not_present: true,
                    read_only: true,
                    no_execute: true,
                    imprecise: true,
                    error_type,
                },
            );
            assert_eq!(event.event_type, AMD_GPU_MEMORY_FAULT_EVENT);
            assert_eq!(event.payload[0], agent.handle);
            assert_eq!(event.payload[1], 0x5678_9000);
            assert_eq!(
                event.payload[2],
                u64::from(
                    AMD_MEMORY_FAULT_PAGE_NOT_PRESENT
                        | AMD_MEMORY_FAULT_READ_ONLY
                        | AMD_MEMORY_FAULT_NO_EXECUTE
                        | AMD_MEMORY_FAULT_IMPRECISE
                        | extra_reason
                )
            );
        }
    }

    #[test]
    fn gpu_ticks_translate_around_the_correlated_system_sample() {
        let counters = ClockCounters {
            gpu: 100,
            host: 0,
            system: 1_000,
            system_frequency: 1_000_000_000,
        };
        assert_eq!(translate_gpu_tick(counters, 110), Ok(1_100));
        assert_eq!(translate_gpu_tick(counters, 90), Ok(900));
        assert_eq!(
            translate_gpu_tick(
                ClockCounters {
                    system_frequency: 0,
                    ..counters
                },
                100,
            ),
            Err(ERROR)
        );
    }

    #[test]
    fn logging_flags_use_the_public_64_bit_mask_layout() {
        let flags = [0b0000_0101, 0, 0, 0, 0, 0, 0, 0];
        assert!(logging_flag_enabled(flags, 0));
        assert!(!logging_flag_enabled(flags, 1));
        assert!(logging_flag_enabled(flags, AMD_LOG_FLAG_INFO));
        assert!(!logging_flag_enabled([u8::MAX; 8], 64));
    }

    #[test]
    fn gpu_pool_handle_encoding_reserves_coarse_fine_and_group_memory() {
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + 1,
                },
                1,
            ),
            Some((0, 1))
        );
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + 2,
                },
                1,
            ),
            Some((0, 2))
        );
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + 3,
                },
                1,
            ),
            Some((0, 3))
        );
        assert_eq!(
            decode_gpu_pool_handle(
                HsaMemoryPool {
                    handle: GPU_POOL_BASE + 0x11,
                },
                1,
            ),
            None
        );
    }

    #[test]
    fn linux_meminfo_yields_the_rocr_system_pool_capacity() {
        assert_eq!(
            parse_host_memory_bytes(
                "MemFree:          1024 kB\nMemTotal:       263219812 kB\nMemAvailable:   2048 kB\n"
            ),
            Some(263_219_812 * 1024)
        );
        assert_eq!(parse_host_memory_bytes("MemFree: 1024 kB\n"), None);
        assert_eq!(parse_host_memory_bytes("MemTotal: invalid kB\n"), None);
    }

    #[test]
    fn cache_handles_and_names_are_stable_runtime_objects() {
        let agent = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let cache = Cache::new(agent, b"gfx1201\0ignored", 3, 8_388_608);
        assert_eq!(cache.agent.handle, agent.handle);
        assert_eq!(&*cache.name, b"gfx1201 L3\0");
        assert_eq!(cache.level, 3);
        assert_eq!(cache.size, 8_388_608);
        assert_eq!(
            decode_cache_handle(HsaCache { handle: CACHE_BASE }, 2),
            Some(0)
        );
        assert_eq!(
            decode_cache_handle(
                HsaCache {
                    handle: CACHE_BASE + 1,
                },
                2,
            ),
            Some(1)
        );
        assert_eq!(
            decode_cache_handle(
                HsaCache {
                    handle: CACHE_BASE + 2,
                },
                2,
            ),
            None
        );
        assert_eq!(
            decode_cache_handle(
                HsaCache {
                    handle: CACHE_BASE - 1,
                },
                2,
            ),
            None
        );
    }

    #[test]
    fn legacy_gpu_cache_sizes_match_rocr_aggregation() {
        let agent = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let other = HsaAgent {
            handle: GPU_AGENT_BASE + 1,
        };
        let caches = [
            Cache::new(agent, b"gfx1201", 1, 32),
            Cache::new(agent, b"gfx1201", 1, 32),
            Cache::new(agent, b"gfx1201", 2, 256),
            Cache::new(agent, b"gfx1201", 2, 256),
            Cache::new(agent, b"gfx1201", 3, 8192),
            Cache::new(other, b"gfx1201", 3, 8192),
        ];
        assert_eq!(
            cache_sizes(&caches, agent),
            [32 * 1024, 512 * 1024, 8192 * 1024, 0]
        );
        assert_eq!(cache_sizes(&caches, other), [0, 0, 8192 * 1024, 0]);
    }
}
