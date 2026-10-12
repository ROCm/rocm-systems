// SPDX-License-Identifier: MIT

//! Process-global HSA initialization, discovery, and object ownership.
//!
//! The single runtime registry owns the rocddi session and every HSA-visible
//! object map. Native queries that release its mutex retain an in-flight token
//! until they finish, so shutdown waits before freeing their device VM.
//! Application callbacks and worker joins occur outside the lock. Final
//! shutdown removes public reachability before releasing native resources.
//!
//! Host and GPU discovery go through the rocddi provider.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Arguments;
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::platform::event::{GpuMemoryFault, SignalEvent, SignalEventPage, poll_memory_fault};
use rocddi::device::Device;
use rocddi::memory::Allocation;
use rocddi::session::{Session, SessionLifetime};
use rocddi::topology::{Endpoint, GpuInfo};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::loader::{CodeObject, CodeSymbol, Executable, Reader, Symbol};
use crate::memory::{LockedMemory, Memory, VmemHandle, VmemMapping, VmemReservation};
use crate::platform::host::{self as platform_host, CpuCacheKind, CpuInfo};
use crate::queue::{CountedHardwareQueue, CountedQueue, Queue, QueueSharedEvent, SoftQueue};
use crate::signal::{
    AsyncDispatcher, AsyncSignalRecord, ImportedIpcSignal, OwnedIpcSignal, SignalSlab,
};

thread_local! {
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

// KFD retains a primary process VM after its last descriptor closes. Later
// HSA generations therefore need a private context to acquire a fresh VM.
static PRIMARY_CONTEXT_USED: AtomicBool = AtomicBool::new(false);

pub(crate) struct InitFailure {
    pub(crate) status: Status,
    pub(crate) cleanup_failed: bool,
}

impl From<Status> for InitFailure {
    fn from(status: Status) -> Self {
        Self {
            status,
            cleanup_failed: false,
        }
    }
}

/// Marks application callbacks whose teardown would wait for the caller.
pub(crate) struct CallbackScope(bool);

impl CallbackScope {
    pub(crate) fn enter() -> Self {
        Self(IN_CALLBACK.with(|active| active.replace(true)))
    }

    pub(crate) fn active() -> bool {
        IN_CALLBACK.with(Cell::get)
    }
}

impl Drop for CallbackScope {
    fn drop(&mut self) {
        IN_CALLBACK.with(|active| active.set(self.0));
    }
}

struct LogConfig {
    flags: [u8; 8],
    stopping: bool,
}

pub(crate) struct LogOutput {
    enabled: AtomicU64,
    config: Mutex<LogConfig>,
}

impl LogOutput {
    fn new() -> Self {
        Self {
            enabled: AtomicU64::new(0),
            config: Mutex::new(LogConfig {
                flags: [0; 8],
                stopping: false,
            }),
        }
    }

    pub(crate) fn set(&self, flags: [u8; 8]) -> Status {
        let Ok(mut config) = self.config.lock() else {
            return ERROR;
        };
        if config.stopping {
            return INVALID_RUNTIME_STATE;
        }
        config.flags = flags;
        self.enabled
            .store(u64::from_le_bytes(flags), Ordering::Release);
        SUCCESS
    }

    fn stop(&self) {
        let mut config = self
            .config
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        config.stopping = true;
        config.flags = [0; 8];
        self.enabled.store(0, Ordering::Release);
    }

    fn permits(&self, flag: u32) -> bool {
        self.config
            .lock()
            .is_ok_and(|config| logging_flag_enabled(config.flags, flag))
    }
}

pub(crate) struct LogWork {
    output: Arc<LogOutput>,
    _inflight: InFlightToken,
    flag: u32,
    line: String,
}

impl LogWork {
    pub(crate) fn write(self) {
        if !self.output.permits(self.flag) {
            return;
        }
        let mut stream = std::io::stderr().lock();
        let _ = stream.write_all(self.line.as_bytes());
        let _ = stream.flush();
    }
}

fn gpu_agent_name(gpu: &GpuInfo) -> String {
    format!("gfx{}{}{}", gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping)
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

fn notify_system_shutdown(handlers: &[(SystemEventHandler, CallbackArg)]) {
    let event = HsaAmdEvent {
        event_type: AMD_SYSTEM_SHUTDOWN_EVENT,
        payload: [0; 3],
    };
    let _ = notify_system_event(handlers, &event, None);
}

fn notify_system_event(
    handlers: &[(SystemEventHandler, CallbackArg)],
    event: &HsaAmdEvent,
    stop: Option<&AtomicBool>,
) -> bool {
    let mut handled = false;
    for (callback, data) in handlers {
        if stop.is_some_and(|stop| stop.load(Ordering::Acquire)) {
            break;
        }
        let _scope = CallbackScope::enter();
        // SAFETY: Registration supplies an ABI-compatible callback. The event
        // remains live for the duration of each synchronous invocation.
        handled |= unsafe { callback(event, data.as_ptr()) } == SUCCESS;
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
                        crate::platform::fault_matches_endpoint(&gpu.endpoint, &fault)
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
                let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if !handled {
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
        let bytes = cache.size;
        if index == 0 {
            if sizes[index] == 0 {
                sizes[index] = bytes;
            }
        } else {
            sizes[index] = sizes[index].saturating_add(bytes);
        }
    }
    sizes
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

trait RetireSession {
    fn retire(&mut self) -> bool;
}

impl RetireSession for Session {
    fn retire(&mut self) -> bool {
        self.destroy().is_ok()
    }
}

struct PendingInit<S: RetireSession = Session, G = Gpu> {
    session: Option<S>,
    gpus: Vec<G>,
}

impl<S: RetireSession, G> PendingInit<S, G> {
    fn cleanup(&mut self) -> bool {
        // Device and MMIO owners must release their session borrows first.
        self.gpus.clear();
        let Some(mut session) = self.session.take() else {
            return true;
        };
        if !session.retire() {
            // A failed native close may retain a VM or KFD runtime state.
            // Keep its owning controller alive until process teardown.
            std::mem::forget(session);
            return false;
        }
        true
    }
}

impl<S: RetireSession, G> Drop for PendingInit<S, G> {
    fn drop(&mut self) {
        self.cleanup();
    }
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

/// Counts calls that must finish before shutdown releases runtime storage.
pub(crate) struct InFlightTracker {
    active: Mutex<usize>,
    idle: Condvar,
}

impl InFlightTracker {
    fn new() -> Self {
        Self {
            active: Mutex::new(0),
            idle: Condvar::new(),
        }
    }

    pub(crate) fn enter(self: &Arc<Self>) -> Option<InFlightToken> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *active = active.checked_add(1)?;
        Some(InFlightToken {
            tracker: self.clone(),
        })
    }

    pub(crate) fn wait_idle(&self) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *active != 0 {
            active = self
                .idle
                .wait(active)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

pub(crate) struct InFlightToken {
    tracker: Arc<InFlightTracker>,
}

impl Drop for InFlightToken {
    fn drop(&mut self) {
        let mut active = self
            .tracker
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *active -= 1;
        if *active == 0 {
            self.tracker.idle.notify_all();
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
    pub(crate) logging: Arc<LogOutput>,
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
    pub(crate) recycled_signals: Vec<usize>,
    pub(crate) signal_event_page: Option<SignalEventPage>,
    pub(crate) signal_event_page_confirmed: bool,
    pub(crate) queue_event: Option<Arc<QueueSharedEvent>>,
    pub(crate) interrupt_signals: HashMap<usize, Option<SignalEvent>>,
    pub(crate) signal_event_pool: Vec<SignalEvent>,
    pub(crate) owned_ipc_signals: HashMap<usize, OwnedIpcSignal>,
    pub(crate) imported_ipc_signals: HashMap<usize, ImportedIpcSignal>,
    pub(crate) async_signal_refs: HashMap<usize, AsyncSignalRecord>,
    pub(crate) signal_groups: HashMap<u64, Vec<HsaSignal>>,
    pub(crate) queues: HashMap<usize, Queue>,
    pub(crate) soft_queues: HashMap<usize, SoftQueue>,
    pub(crate) counted_queues: HashMap<usize, CountedQueue>,
    pub(crate) counted_queue_pools: HashMap<(u64, u32), Vec<CountedHardwareQueue>>,
    pub(crate) released_counted_queues: HashSet<usize>,
    pub(crate) counted_queue_limit: usize,
    pub(crate) counted_queue_size: u32,
    pub(crate) host_memory_bytes: usize,
    pub(crate) host_page_size: usize,
    pub(crate) host_name: Box<str>,
    pub(crate) host_compute_units: u32,
    pub(crate) full_profile: bool,
    pub(crate) system_event_handlers: Vec<(SystemEventHandler, CallbackArg)>,
    pub(crate) system_event_worker_started: bool,
    pub(crate) async_dispatcher: Option<AsyncDispatcher>,
    pub(crate) workers: Vec<JoinHandle<()>>,
    pub(crate) stop_workers: Arc<AtomicBool>,
    pub(crate) inflight: Arc<InFlightTracker>,
    pub(crate) caches: Vec<Cache>,
    pub(crate) gpus: Vec<Gpu>,
    pub(crate) session: Session,
    pub(crate) lifetime: SessionLifetime,
}

pub(crate) fn defer_cleanup<T: Send + 'static>(
    owner: T,
    cleanup: impl FnOnce(T) + Send + 'static,
) -> Result<(), Option<T>> {
    let pending = Arc::new(Mutex::new(Some(owner)));
    let worker_pending = pending.clone();
    if thread::Builder::new()
        .name("rocddi-shutdown".into())
        .spawn(move || {
            let owner = worker_pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(owner) = owner {
                cleanup(owner);
            }
        })
        .is_err()
    {
        return Err(pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take());
    }
    Ok(())
}

impl Runtime {
    pub(crate) fn running_on_worker(&self) -> bool {
        self.workers
            .iter()
            .any(|worker| worker.thread().id() == thread::current().id())
    }

    pub(crate) fn request_stop(&self) {
        self.logging.stop();
        self.stop_workers.store(true, Ordering::Release);
    }

    pub(crate) fn create() -> Result<Self, InitFailure> {
        let host_page_size = usize::try_from(rocddi::memory::host_page_size().map_err(map_error)?)
            .map_err(|_| ERROR)?;
        if host_page_size < 4096 || !host_page_size.is_power_of_two() {
            return Err(ERROR.into());
        }
        let host_memory_bytes = platform_host::memory_bytes().ok_or(ERROR)?;
        let host = platform_host::cpu_info().unwrap_or_else(|| CpuInfo {
            name: "CPU".to_owned(),
            compute_units: 0,
        });
        let force_fine_grain_pcie =
            std::env::var("HSA_FORCE_FINE_GRAIN_PCIE").is_ok_and(|value| value == "1");
        // The HSA lifecycle gate admits one initialization at a time. Passive
        // enumeration may fail before a primary KFD VM is acquired.
        let lifetime = if PRIMARY_CONTEXT_USED.load(Ordering::Acquire) {
            SessionLifetime::Session
        } else {
            SessionLifetime::Process
        };
        let mut pending = PendingInit {
            session: Some(Session::new(lifetime).map_err(map_error)?),
            gpus: Vec::new(),
        };
        match Self::create_with_session(
            &mut pending,
            host_page_size,
            host_memory_bytes,
            host,
            force_fine_grain_pcie,
            lifetime,
        ) {
            Ok(runtime) => Ok(runtime),
            Err(status) => Err(InitFailure {
                status,
                cleanup_failed: !pending.cleanup(),
            }),
        }
    }

    fn create_with_session(
        pending: &mut PendingInit,
        host_page_size: usize,
        host_memory_bytes: usize,
        host: CpuInfo,
        force_fine_grain_pcie: bool,
        lifetime: SessionLifetime,
    ) -> Result<Self, Status> {
        let session = pending.session.as_ref().ok_or(ERROR)?;
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
        let cpu_agent = HsaAgent { handle: CPU_AGENT };
        let mut caches = platform_host::caches_for_numa_node(0)
            .unwrap_or_else(platform_host::caches)
            .into_iter()
            .filter(|cache| {
                cache.kind == CpuCacheKind::Data && cache.first_shared_cpu == Some(cache.cpu)
            })
            .map(|cache| {
                Cache::new(
                    cpu_agent,
                    host.name.as_bytes(),
                    cache.level,
                    cache.size_bytes,
                )
            })
            .collect::<Vec<_>>();
        pending
            .gpus
            .try_reserve(endpoints.len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        for endpoint in endpoints {
            let Some(info) = endpoint.gpu().copied() else {
                continue;
            };
            let agent = HsaAgent {
                handle: GPU_AGENT_BASE + pending.gpus.len() as u64,
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
                    u32::try_from(cache.size_bytes()).unwrap_or(u32::MAX),
                ));
            }
            let presentation = session.gpu_presentation(&endpoint).map_err(map_error)?;
            if lifetime == SessionLifetime::Process {
                // Activation may retain a primary VM even when it returns an
                // error. A later generation must use a private context.
                PRIMARY_CONTEXT_USED.store(true, Ordering::Release);
            }
            let device = session.activate(&endpoint).map_err(map_error)?;
            let product_name = presentation
                .product_name
                .unwrap_or_else(|| "AMD Radeon Graphics".to_owned());
            let asic_family_id = presentation.asic_family_id;
            let mmio_remap = device.gpu().and_then(|gpu| gpu.map_mmio_remap()).ok();
            let hdp_flush = hdp_flush_pointers(
                mmio_remap
                    .as_ref()
                    .and_then(|mapping| mapping.info().host_address),
            );
            let fine_grain_pool = info.hive_id != 0 || force_fine_grain_pcie;
            pending.gpus.push(Gpu {
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
        let gpus = std::mem::take(&mut pending.gpus);
        let session = pending.session.take().ok_or(ERROR)?;
        Ok(Self {
            references: 1,
            logging: Arc::new(LogOutput::new()),
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
            recycled_signals: Vec::new(),
            signal_event_page: None,
            signal_event_page_confirmed: false,
            queue_event: None,
            interrupt_signals: HashMap::new(),
            signal_event_pool: Vec::new(),
            owned_ipc_signals: HashMap::new(),
            imported_ipc_signals: HashMap::new(),
            async_signal_refs: HashMap::new(),
            signal_groups: HashMap::new(),
            queues: HashMap::new(),
            soft_queues: HashMap::new(),
            counted_queues: HashMap::new(),
            counted_queue_pools: HashMap::new(),
            released_counted_queues: HashSet::new(),
            counted_queue_limit: environment_u32("GPU_MAX_HW_QUEUES", 4) as usize,
            counted_queue_size: environment_u32("HSA_COUNTED_QUEUE_SIZE", 16_384),
            host_memory_bytes,
            host_page_size,
            host_name: host.name.into_boxed_str(),
            host_compute_units: host.compute_units,
            full_profile,
            system_event_handlers: Vec::new(),
            system_event_worker_started: false,
            async_dispatcher: None,
            workers: Vec::new(),
            stop_workers: Arc::new(AtomicBool::new(false)),
            inflight: Arc::new(InFlightTracker::new()),
            caches,
            gpus,
            session,
            lifetime,
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

    pub(crate) fn prepare_log(&self, flag: u32, arguments: Arguments<'_>) -> Option<LogWork> {
        if flag >= 64 || self.logging.enabled.load(Ordering::Acquire) & (1_u64 << flag) == 0 {
            return None;
        }
        let inflight = self.inflight.enter()?;
        let mut line = String::from("[***rocddi***] ");
        if std::fmt::write(&mut line, arguments).is_err() {
            return None;
        }
        line.push('\n');
        Some(LogWork {
            output: self.logging.clone(),
            _inflight: inflight,
            flag,
            line,
        })
    }

    pub(crate) fn stop(mut self) -> Status {
        if self.running_on_worker() || CallbackScope::active() {
            // Releasing signal storage from its own callback worker is unsafe.
            self.request_stop();
            std::mem::forget(self);
            return INVALID_RUNTIME_STATE;
        }
        self.request_stop();
        // Multi-signal waits observe stop_workers and release their signal
        // references. Native calls finish before we release the device VM.
        self.inflight.wait_idle();
        notify_system_shutdown(&self.system_event_handlers);
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
        self.async_signal_refs.clear();
        self.imported_ipc_signals.clear();
        self.owned_ipc_signals.clear();
        self.recycled_signals.clear();
        self.signal_slabs.clear();
        self.gpus.clear();
        let session_status = self.session.destroy().map_or_else(map_error, |()| SUCCESS);
        if signal_status == SUCCESS {
            session_status
        } else {
            signal_status
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

pub(crate) fn translate_gpu_interval(
    device: &Device,
    start: u64,
    end: u64,
) -> Result<(u64, u64), Status> {
    if start == 0 || end == 0 {
        return Ok((0, 0));
    }
    let counters = device
        .gpu()
        .and_then(|gpu| gpu.clock_counters())
        .map_err(map_error)?;
    Ok((
        translate_gpu_tick(counters, start)?,
        translate_gpu_tick(counters, end)?,
    ))
}

pub(crate) fn translate_gpu_tick(
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

    pub(crate) fn cancel_init(
        &mut self,
        generation: u64,
        cleanup_failed: bool,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.phase = if cleanup_failed {
            LifecyclePhase::Quarantined
        } else {
            LifecyclePhase::Inactive
        };
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
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    #[allow(clippy::unwrap_used)]
    fn shutdown_waits_for_inflight_calls() {
        let inflight = Arc::new(InFlightTracker::new());
        let token = inflight.enter().unwrap();
        let (started_send, started_recv) = std::sync::mpsc::channel();
        let (done_send, done_recv) = std::sync::mpsc::channel();
        let worker_inflight = inflight.clone();
        let worker = thread::spawn(move || {
            started_send.send(()).unwrap();
            worker_inflight.wait_idle();
            done_send.send(()).unwrap();
        });
        started_recv.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(done_recv.recv_timeout(Duration::from_millis(10)).is_err());
        drop(token);
        done_recv.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn callback_scope_restricts_queue_and_sampling_teardown() {
        assert!(!CallbackScope::active());
        {
            let _scope = CallbackScope::enter();
            // SAFETY: Queue teardown is rejected before inspecting the
            // non-null pointer while a callback is active.
            assert_eq!(
                unsafe {
                    crate::queue::hsa_queue_destroy(
                        std::ptr::NonNull::<HsaQueue>::dangling().as_ptr(),
                    )
                },
                INVALID_RUNTIME_STATE
            );
            assert_eq!(
                crate::pc_sampling::hsa_ven_amd_pcs_stop(HsaPcSampling { handle: 1 }),
                INVALID_RUNTIME_STATE
            );
            let _nested = CallbackScope::enter();
            assert!(CallbackScope::active());
        }
        assert!(!CallbackScope::active());
    }

    #[test]
    fn deferred_cleanup_joins_the_callback_worker_after_exit() {
        struct Owner {
            worker: JoinHandle<()>,
            exited: Arc<AtomicBool>,
            completed: std::sync::mpsc::Sender<bool>,
        }
        let exited = Arc::new(AtomicBool::new(false));
        let (owner_tx, owner_rx) = std::sync::mpsc::channel::<Owner>();
        let (completed_tx, completed_rx) = std::sync::mpsc::channel();
        let worker_exited = exited.clone();
        let worker = thread::spawn(move || {
            let owner = owner_rx.recv().unwrap();
            assert_eq!(owner.worker.thread().id(), thread::current().id());
            assert!(
                defer_cleanup(owner, |owner| {
                    owner.worker.join().unwrap();
                    owner
                        .completed
                        .send(owner.exited.load(AtomicOrdering::Acquire))
                        .unwrap();
                })
                .is_ok()
            );
            worker_exited.store(true, AtomicOrdering::Release);
        });
        owner_tx
            .send(Owner {
                worker,
                exited,
                completed: completed_tx,
            })
            .unwrap();
        assert!(completed_rx.recv_timeout(Duration::from_secs(2)).unwrap());
    }

    #[test]
    fn initialization_gate_rejects_overlapping_generations() {
        let mut registry = RuntimeRegistry::new();
        let first = registry.begin_init().unwrap().unwrap();
        assert_eq!(registry.begin_init().unwrap_err(), INVALID_RUNTIME_STATE);
        assert_eq!(registry.begin_shutdown().err(), Some(INVALID_RUNTIME_STATE));
        registry.cancel_init(first, false).unwrap();
        let second = registry.begin_init().unwrap().unwrap();
        assert_ne!(second, first);
        registry.cancel_init(second, false).unwrap();
    }

    #[test]
    fn failed_initialization_cleanup_quarantines_the_next_generation() {
        let mut registry = RuntimeRegistry::new();
        let generation = registry.begin_init().unwrap().unwrap();
        registry.cancel_init(generation, true).unwrap();
        assert_eq!(registry.begin_init().unwrap_err(), INVALID_RUNTIME_STATE);
    }

    #[test]
    fn first_and_later_gpu_activation_failures_retire_the_session_join() {
        struct Activated {
            live: Arc<AtomicUsize>,
        }
        impl Drop for Activated {
            fn drop(&mut self) {
                self.live.fetch_sub(1, AtomicOrdering::Relaxed);
            }
        }
        struct Joined {
            live: Arc<AtomicUsize>,
            joins: Arc<AtomicUsize>,
        }
        impl RetireSession for Joined {
            fn retire(&mut self) -> bool {
                assert_eq!(self.live.load(AtomicOrdering::Relaxed), 0);
                self.joins.fetch_sub(1, AtomicOrdering::Relaxed);
                true
            }
        }

        for completed_activations in [0, 1] {
            let live = Arc::new(AtomicUsize::new(completed_activations));
            let joins = Arc::new(AtomicUsize::new(1));
            let mut pending = PendingInit {
                session: Some(Joined {
                    live: live.clone(),
                    joins: joins.clone(),
                }),
                gpus: (0..completed_activations)
                    .map(|_| Activated { live: live.clone() })
                    .collect(),
            };
            assert!(pending.cleanup());
            assert_eq!(live.load(AtomicOrdering::Relaxed), 0);
            assert_eq!(joins.load(AtomicOrdering::Relaxed), 0);
        }
    }

    #[test]
    #[ignore = "requires a qualified GPU and KFD runtime"]
    fn hsa_runtime_reinitializes_after_shutdown() {
        for _ in 0..3 {
            assert_eq!(crate::hsa_init(), SUCCESS);
            assert_eq!(crate::hsa_shut_down(), SUCCESS);
        }
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

    struct StopEventProbe {
        stop: AtomicBool,
        later_calls: AtomicU32,
    }

    unsafe extern "C" fn stop_from_fault_handler(
        _event: *const HsaAmdEvent,
        data: *mut c_void,
    ) -> Status {
        // SAFETY: The test keeps the probe live until both callbacks finish.
        let probe = unsafe { &*data.cast::<StopEventProbe>() };
        probe.stop.store(true, AtomicOrdering::Release);
        ERROR
    }

    unsafe extern "C" fn count_later_fault_handler(
        _event: *const HsaAmdEvent,
        data: *mut c_void,
    ) -> Status {
        // SAFETY: The test keeps the probe live until both callbacks finish.
        let probe = unsafe { &*data.cast::<StopEventProbe>() };
        probe.later_calls.fetch_add(1, AtomicOrdering::Relaxed);
        SUCCESS
    }

    #[test]
    fn shutdown_requested_by_fault_handler_skips_later_handlers() {
        let probe = StopEventProbe {
            stop: AtomicBool::new(false),
            later_calls: AtomicU32::new(0),
        };
        // SAFETY: The probe outlives synchronous event delivery.
        let data = unsafe { CallbackArg::new(std::ptr::from_ref(&probe).cast_mut().cast()) };
        let handlers = [
            (stop_from_fault_handler as SystemEventHandler, data),
            (count_later_fault_handler as SystemEventHandler, data),
        ];
        let event = HsaAmdEvent {
            event_type: AMD_GPU_MEMORY_FAULT_EVENT,
            payload: [0; 3],
        };

        assert!(!notify_system_event(&handlers, &event, Some(&probe.stop)));
        assert!(probe.stop.load(AtomicOrdering::Acquire));
        assert_eq!(probe.later_calls.load(AtomicOrdering::Relaxed), 0);
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
                // SAFETY: The first probe outlives synchronous delivery.
                unsafe { CallbackArg::new(std::ptr::from_ref(&first).cast_mut().cast()) },
            ),
            (
                observe_system_event,
                // SAFETY: The second probe outlives synchronous delivery.
                unsafe { CallbackArg::new(std::ptr::from_ref(&second).cast_mut().cast()) },
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
            Cache::new(agent, b"gfx1201", 1, 32 * 1024),
            Cache::new(agent, b"gfx1201", 1, 32 * 1024),
            Cache::new(agent, b"gfx1201", 2, 256 * 1024),
            Cache::new(agent, b"gfx1201", 2, 256 * 1024),
            Cache::new(agent, b"gfx1201", 3, 8192 * 1024),
            Cache::new(other, b"gfx1201", 3, 8192 * 1024),
        ];
        assert_eq!(
            cache_sizes(&caches, agent),
            [32 * 1024, 512 * 1024, 8192 * 1024, 0]
        );
        assert_eq!(cache_sizes(&caches, other), [0, 0, 8192 * 1024, 0]);
    }
}
