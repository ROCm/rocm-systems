// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process-global HSA initialization, discovery, and shutdown coordination.
//!
//! A lifecycle gate publishes one HSA adapter state. Queue, signal, memory,
//! loader, image, and sampling modules own their HSA handle maps within that
//! state; rocddi owns native device resources. Native queries that release the
//! gate's mutex retain an in-flight token until they finish, so shutdown waits
//! before freeing their device VM.
//! Application callbacks and worker joins occur outside the lock. Final
//! shutdown removes public reachability before releasing native resources.
//!
//! GPU discovery and passive host CPU information both come from rocddi.

use std::cell::Cell;
use std::ffi::c_void;
use std::fmt::Arguments;
use std::io::Write;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::time::Instant;

use rocddi::device::Device;
use rocddi::device::event::{
    DeviceEvent, DeviceEventSubscription, GpuHardwareException, GpuMemoryFault,
    GpuMemoryFaultCause, GpuResetCause, subscribe,
};
use rocddi::device::gpu::profiling::{PcSamplingConfiguration, PcSamplingConfigurationSet};
use rocddi::memory::device::Allocation;
use rocddi::session::{DriverContextLifetime, Session};
use rocddi::topology::host::{
    self as host_topology, CpuCacheKind, CpuInfo, HostCaches, HostTopologySnapshot,
};
use rocddi::topology::{Endpoint, GpuInfo};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::image_abi::ImageState;
use crate::loader::LoaderState;
use crate::memory::MemoryState;
use crate::pc_sampling::PcSamplingState;
use crate::queue::QueueState;
use crate::signal::SignalState;

thread_local! {
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static IN_LOG_WRITE: Cell<bool> = const { Cell::new(false) };
}

// KFD retains a primary process VM after its last descriptor closes. Later
// HSA generations therefore need a private context to acquire a fresh VM.
static PRIMARY_CONTEXT_USED: AtomicBool = AtomicBool::new(false);
static PRECISE_TIMESTAMPS: AtomicBool = AtomicBool::new(false);

/// Choose the KFD correlation least exposed to preemption between its reads.
pub(crate) fn clock_counters(
    device: &Device,
) -> Result<rocddi::device::gpu::profiling::ClockCounters, rocddi::Error> {
    let gpu = device.gpu()?;
    let samples = if PRECISE_TIMESTAMPS.load(Ordering::Acquire) {
        4
    } else {
        1
    };
    best_clock_sample(samples, || {
        let before = Instant::now();
        let counters = gpu.clock_counters()?;
        Ok((counters, before.elapsed()))
    })
}

fn best_clock_sample<T, E>(
    count: usize,
    mut sample: impl FnMut() -> Result<(T, Duration), E>,
) -> Result<T, E> {
    let (first, elapsed) = sample()?;
    let mut best = (first, elapsed);
    for _ in 1..count {
        match sample() {
            Ok((value, elapsed)) if elapsed < best.1 => best = (value, elapsed),
            Ok(_) => {}
            Err(_) => return Ok(best.0),
        }
    }
    Ok(best.0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_enable_high_precision_timestamps")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_enable_high_precision_timestamps() -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        PRECISE_TIMESTAMPS.store(true, Ordering::Release);
        SUCCESS
    })
}

/// A transient native query failure leaves support undecided so public calls
/// can retry after the underlying condition clears.
fn pc_sampling_supported_by_query(
    result: Result<Vec<PcSamplingConfiguration>, rocddi::Error>,
) -> bool {
    match result {
        Ok(configurations) => !configurations.is_empty(),
        Err(error) => error.kind() != rocddi::ErrorKind::Unsupported,
    }
}

/// Initialization error with evidence that native rollback also failed.
pub(crate) struct InitFailure {
    /// HSA status returned to the caller.
    pub(crate) status: Status,
    /// Native state was retained because cleanup could not be proved.
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

/// Prevents C stream callbacks from recursively entering the log writer or
/// replacing its stream while the borrowed pointer is in use.
pub(crate) struct LogWriteScope;

impl LogWriteScope {
    fn enter() -> Option<Self> {
        if IN_LOG_WRITE.with(|active| active.replace(true)) {
            return None;
        }
        Some(Self)
    }

    pub(crate) fn active() -> bool {
        IN_LOG_WRITE.with(Cell::get)
    }
}

impl Drop for LogWriteScope {
    fn drop(&mut self) {
        IN_LOG_WRITE.with(|active| active.set(false));
    }
}

/// Logging destination and flags protected during C stream use.
struct LogConfig {
    /// Enabled log categories supplied by the caller.
    flags: [u8; 8],
    /// Borrowed C stream, or stderr when absent.
    stream: Option<BorrowedCStream>,
    /// Prevents stream replacement during final shutdown.
    stopping: bool,
}

/// Borrowed C `FILE` pointer governed by the HSA logging contract.
#[derive(Clone, Copy)]
struct BorrowedCStream(NonNull<c_void>);

// SAFETY: The C stdio stream is used only while LogOutput holds its config
// mutex. The HSA caller keeps the stream open while it is configured, and C
// stdio serializes access from other threads using the same FILE pointer.
unsafe impl Send for BorrowedCStream {}

unsafe extern "C" {
    fn fwrite(buffer: *const c_void, size: usize, count: usize, stream: *mut c_void) -> usize;
    fn fflush(stream: *mut c_void) -> i32;
}

/// Shared logging sink with an atomic category fast path.
pub(crate) struct LogOutput {
    /// Enabled categories checked before locking the stream.
    enabled: AtomicU64,
    /// Serializes output and changes to the borrowed stream.
    config: Mutex<LogConfig>,
}

impl LogOutput {
    fn new() -> Self {
        Self {
            enabled: AtomicU64::new(0),
            config: Mutex::new(LogConfig {
                flags: [0; 8],
                stream: None,
                stopping: false,
            }),
        }
    }

    pub(crate) fn set(&self, flags: [u8; 8], stream: *mut c_void) -> Status {
        let Ok(mut config) = self.config.lock() else {
            return ERROR;
        };
        if config.stopping {
            return INVALID_RUNTIME_STATE;
        }
        config.flags = flags;
        config.stream = NonNull::new(stream).map(BorrowedCStream);
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
        config.stream = None;
        self.enabled.store(0, Ordering::Release);
    }

    fn write(&self, flag: u32, line: &[u8]) {
        let Some(_scope) = LogWriteScope::enter() else {
            return;
        };
        let Ok(config) = self.config.lock() else {
            return;
        };
        if config.stopping || !logging_flag_enabled(config.flags, flag) {
            return;
        }
        if let Some(stream) = config.stream {
            // SAFETY: The public logging contract keeps the borrowed FILE open
            // while enabled. This mutex prevents replacement or shutdown from
            // finishing until the C stdio calls finish.
            unsafe {
                let _ = fwrite(line.as_ptr().cast(), 1, line.len(), stream.0.as_ptr());
                let _ = fflush(stream.0.as_ptr());
            }
        } else {
            let mut stream = std::io::stderr().lock();
            let _ = stream.write_all(line);
            let _ = stream.flush();
        }
    }
}

/// Log line that retains its runtime generation until emission completes.
pub(crate) struct LogWork {
    /// Sink retained while writing outside the registry lock.
    output: Arc<LogOutput>,
    /// Blocks shutdown from freeing state while this work is outstanding.
    _inflight: InFlightToken,
    /// Category to check before emitting the line.
    flag: u32,
    /// Complete formatted line owned by this work item.
    line: String,
}

impl LogWork {
    pub(crate) fn write(self) {
        self.output.write(self.flag, self.line.as_bytes());
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
    reason |= match fault.cause {
        GpuMemoryFaultCause::SramEcc => AMD_MEMORY_FAULT_SRAM_ECC,
        GpuMemoryFaultCause::DramEcc => AMD_MEMORY_FAULT_DRAM_ECC,
        GpuMemoryFaultCause::Hang => AMD_MEMORY_FAULT_HANG,
        GpuMemoryFaultCause::None | GpuMemoryFaultCause::Other => 0,
    };
    HsaAmdEvent {
        event_type: AMD_GPU_MEMORY_FAULT_EVENT,
        payload: [agent.handle, fault.virtual_address, u64::from(reason)],
    }
}

fn hardware_exception_event(agent: HsaAgent, exception: GpuHardwareException) -> HsaAmdEvent {
    let cause = match exception.cause {
        GpuResetCause::Ecc => AMD_HW_EXCEPTION_CAUSE_ECC,
        GpuResetCause::Hang | GpuResetCause::Other => AMD_HW_EXCEPTION_CAUSE_GPU_HANG,
    };
    HsaAmdEvent {
        event_type: AMD_GPU_HW_EXCEPTION_EVENT,
        // The current HSA ABI exposes only the OTHER reset-type flag.
        payload: [
            agent.handle,
            u64::from(AMD_HW_EXCEPTION_RESET_TYPE_OTHER) | (u64::from(cause) << 32),
            0,
        ],
    }
}

fn system_event_worker(mut subscription: DeviceEventSubscription, stop: &AtomicBool) {
    while !stop.load(Ordering::Acquire) {
        match subscription.poll() {
            Ok(Some(DeviceEvent::GpuMemoryFault(fault))) => {
                let notification = {
                    let Ok(mut guard) = lock() else {
                        return;
                    };
                    let Some(runtime) = guard.as_mut() else {
                        return;
                    };
                    let Some(index) = runtime
                        .gpus
                        .iter()
                        .position(|gpu| Some(gpu.endpoint.id) == fault.endpoint_id)
                    else {
                        continue;
                    };
                    let agent = HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    };
                    let event = memory_fault_event(agent, fault);
                    let reason = event.payload[2] as u32;
                    runtime.vm_fault_details = Some((agent, fault.virtual_address, reason));
                    // KFD reports the process fault and the queue error on
                    // different workers. Let the queue worker identify the
                    // faulted queue before delivering the system callback.
                    let Ok((mut guard, _)) = VM_FAULT_CONDVAR.wait_timeout_while(
                        guard,
                        Duration::from_millis(50),
                        |registry| {
                            registry.as_ref().is_some_and(|runtime| {
                                runtime
                                    .queue_state
                                    .queues
                                    .values()
                                    .any(|queue| queue.agent == agent)
                                    && !runtime
                                        .queue_state
                                        .queues
                                        .values()
                                        .any(|queue| queue.agent == agent && queue.vm_faulted)
                            })
                        },
                    ) else {
                        return;
                    };
                    let Some(runtime) = guard.as_mut() else {
                        return;
                    };
                    for queue in runtime.queue_state.queues.values_mut() {
                        if queue.agent == agent && queue.vm_faulted {
                            queue.vm_fault_address = fault.virtual_address;
                            queue.vm_fault_reason = reason;
                        }
                    }
                    (event, runtime.system_event_handlers.clone())
                };
                let handled = notify_system_event(&notification.1, &notification.0, Some(stop));
                if stop.load(Ordering::Acquire) {
                    return;
                }
                if !handled {
                    std::process::abort();
                }
            }
            Ok(Some(DeviceEvent::GpuHardwareException(exception))) => {
                let notification = {
                    let Ok(guard) = lock() else {
                        return;
                    };
                    let Some(runtime) = guard.as_ref() else {
                        return;
                    };
                    let Some(index) = runtime
                        .gpus
                        .iter()
                        .position(|gpu| Some(gpu.endpoint.id) == exception.endpoint_id)
                    else {
                        continue;
                    };
                    let agent = HsaAgent {
                        handle: GPU_AGENT_BASE + index as u64,
                    };
                    (
                        hardware_exception_event(agent, exception),
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
            }
            Ok(None) => thread::sleep(Duration::from_micros(20)),
            Err(error)
                if matches!(
                    error.kind(),
                    rocddi::ErrorKind::Driver | rocddi::ErrorKind::Busy
                ) =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(_) => return,
        }
    }
}

fn decode_gpu_pool_handle(pool: HsaMemoryPool, gpu_count: usize) -> Option<(usize, u64)> {
    let offset = pool.handle.checked_sub(GPU_POOL_BASE)?;
    let index = usize::try_from(offset / 0x10).ok()?;
    let kind = offset % 0x10;
    (index < gpu_count && matches!(kind, 1..=4)).then_some((index, kind))
}

fn decode_cache_handle(cache: HsaCache, cache_count: usize) -> Option<usize> {
    let index = cache.handle.checked_sub(CACHE_BASE)?;
    let index = usize::try_from(index).ok()?;
    (index < cache_count).then_some(index)
}

fn selected_cpu_caches(topology: &HostTopologySnapshot) -> &HostCaches {
    topology
        .domain(0)
        .map_or(&topology.caches, |domain| &domain.caches)
}

pub(crate) fn cache_handles_in_level_order(caches: &[Cache], agent: HsaAgent) -> Vec<HsaCache> {
    let mut indices = caches
        .iter()
        .enumerate()
        .filter(|(_, cache)| cache.agent == agent)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    indices.sort_by_key(|&index| caches[index].level);
    indices
        .into_iter()
        .map(|index| HsaCache {
            handle: CACHE_BASE + index as u64,
        })
        .collect()
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
    /// Passive endpoint used for identity and topology queries.
    pub(crate) endpoint: Endpoint,
    /// Cached hardware facts from passive discovery.
    pub(crate) info: GpuInfo,
    /// Activated native device and its driver resources.
    pub(crate) device: Device,
    /// HSA-visible short target name.
    pub(crate) name: Box<str>,
    /// HSA-visible product name.
    pub(crate) product_name: Box<str>,
    /// ASIC family used by image layout and target checks.
    pub(crate) asic_family_id: u32,
    /// GPU timestamp ticks per second.
    pub(crate) timestamp_frequency_hz: u64,
    /// Host addresses of native HDP flush controls, when available.
    pub(crate) hdp_flush: [usize; 2],
    /// Retains the MMIO allocation supplying those controls.
    _mmio_remap: Option<Allocation>,
    /// HSA coherency classification derived from native capabilities.
    pub(crate) coherency_type: u32,
    /// Whether a fine-grain local pool is exposed for this GPU.
    pub(crate) fine_grain_pool: bool,
    /// Whether this target has a PC sampling path; transient query failures
    /// leave the capability visible so later operations can retry.
    pub(crate) pc_sampling_supported: bool,
    /// Current native persisting L2 reservation shared across HSA views.
    pub(crate) persisting_l2_cache_size: Arc<Mutex<usize>>,
}

/// Minimal cleanup contract shared by real and test session owners.
trait RetireSession {
    /// Releases native resources and reports whether cleanup was conclusive.
    fn retire(&mut self) -> bool;
}

impl RetireSession for Session {
    fn retire(&mut self) -> bool {
        self.destroy().is_ok()
    }
}

/// Holds startup resources until publication or rollback finishes.
struct PendingInit<S: RetireSession = Session, G = Gpu> {
    /// Session released after all activated GPU owners.
    session: Option<S>,
    /// Activated GPUs that must be dropped before their session.
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
    /// HSA agent that reports this cache.
    pub(crate) agent: HsaAgent,
    /// Stable NUL-terminated cache name.
    pub(crate) name: Box<[u8]>,
    /// Cache hierarchy level.
    pub(crate) level: u8,
    /// Cache capacity in bytes.
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
    /// Number of native operations currently holding a token.
    active: Mutex<usize>,
    /// Wakes shutdown when the final token is dropped.
    idle: Condvar,
    /// Native cleanup failed after the generation was detached for shutdown.
    cleanup_uncertain: AtomicBool,
}

impl InFlightTracker {
    pub(crate) fn new() -> Self {
        Self {
            active: Mutex::new(0),
            idle: Condvar::new(),
            cleanup_uncertain: AtomicBool::new(false),
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

    /// Retains the whole generation when an unpublished owner cannot be
    /// returned to its registry after native cleanup fails.
    pub(crate) fn quarantine(&self) {
        self.cleanup_uncertain.store(true, Ordering::Release);
    }

    pub(crate) fn is_quarantined(&self) -> bool {
        self.cleanup_uncertain.load(Ordering::Acquire)
    }
}

/// One registered operation that must finish before runtime teardown.
pub(crate) struct InFlightToken {
    /// Counter decremented when this operation leaves its native work.
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

/// Keeps an unlocked native operation in one HSA generation until its owner
/// is either published back into that generation or safely retired.
pub(crate) struct InFlightGeneration {
    generation: u64,
    token: InFlightToken,
}

impl InFlightGeneration {
    pub(crate) fn enter(generation: u64, tracker: &Arc<InFlightTracker>) -> Result<Self, Status> {
        let token = tracker.enter().ok_or(OUT_OF_RESOURCES)?;
        Ok(Self { generation, token })
    }

    pub(crate) fn is_current(&self, lifecycle: &HsaLifecycle) -> bool {
        lifecycle.active_generation() == Some(self.generation)
    }

    /// Prevents shutdown from dropping dependencies when an unpublished owner
    /// cannot be returned to its original generation.
    pub(crate) fn quarantine(&self) {
        self.token.tracker.quarantine();
    }
}

/// Complete process-global HSA state.
///
/// Every integer handle and public pointer accepted by this frontend must map
/// to an owner in this structure. Removing an entry transfers that owner to the
/// entry point performing destruction, allowing callbacks, blocking cleanup,
/// and worker joins to happen after the global mutex is released.
pub(crate) struct HsaState {
    /// Process logging sink retained by work outside the registry lock.
    pub(crate) logging: Arc<LogOutput>,
    /// Next opaque HSA handle value.
    pub(crate) next_handle: u64,
    /// HSA queue handles and their frontend teardown dependencies.
    pub(crate) queue_state: QueueState,
    /// HSA code-object and executable handles around rocloader owners.
    pub(crate) loader: LoaderState,
    /// HSA image and sampler descriptor handles.
    pub(crate) image_state: ImageState,
    /// HSA memory handles and metadata around rocddi allocation owners.
    pub(crate) memory_state: MemoryState,
    /// HSA signal handles, backing, events, and asynchronous registrations.
    pub(crate) signal_state: SignalState,
    /// HSA PC sampling handles and native cleanup state.
    pub(crate) pc_sampling_state: PcSamplingState,
    /// Host memory capacity reported by the CPU agent.
    pub(crate) host_memory_bytes: usize,
    /// System page size used by host memory pools.
    pub(crate) host_page_size: usize,
    /// Stable HSA name of the host CPU agent.
    pub(crate) host_name: Box<str>,
    /// Compute-unit count reported for the CPU agent.
    pub(crate) host_compute_units: u32,
    /// Whether all exposed agents support the full HSA profile.
    pub(crate) full_profile: bool,
    /// Latest reported VM fault details for system event queries.
    pub(crate) vm_fault_details: Option<(HsaAgent, u64, u32)>,
    /// Registered process-level system event callbacks.
    pub(crate) system_event_handlers: Vec<(SystemEventHandler, CallbackArg)>,
    /// Native system event worker joined during final teardown.
    pub(crate) system_event_worker: Option<JoinHandle<()>>,
    /// Shared stop request observed by those workers.
    pub(crate) stop_workers: Arc<AtomicBool>,
    /// Counts native work that must finish before session release.
    pub(crate) inflight: Arc<InFlightTracker>,
    /// CPU and GPU cache records exposed through HSA queries.
    pub(crate) caches: Vec<Cache>,
    /// Activated GPU agents and their native owners.
    pub(crate) gpus: Vec<Gpu>,
    /// Native session retained until all GPU and memory owners are gone.
    pub(crate) session: Session,
    /// Lifetime policy for the KFD context used by this runtime generation.
    /// Memory-pool choices that require the primary context use this value.
    pub(crate) lifetime: DriverContextLifetime,
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

impl HsaState {
    pub(crate) fn running_on_worker(&self) -> bool {
        self.signal_state.running_on_worker()
            || self.memory_state.running_on_worker()
            || self
                .system_event_worker
                .as_ref()
                .is_some_and(|worker| worker.thread().id() == thread::current().id())
    }

    pub(crate) fn request_stop(&self) {
        self.logging.stop();
        self.stop_workers.store(true, Ordering::Release);
    }

    pub(crate) fn create() -> Result<Self, InitFailure> {
        let host_page_size =
            usize::try_from(rocddi::memory::host::host_page_size().map_err(map_error)?)
                .map_err(|_| ERROR)?;
        if host_page_size < 4096 || !host_page_size.is_power_of_two() {
            return Err(ERROR.into());
        }
        let topology = host_topology::snapshot();
        let host_memory_bytes = topology.memory_bytes.ok_or(ERROR)?;
        let host = topology.cpu.clone().unwrap_or_else(|| CpuInfo {
            name: "CPU".to_owned(),
            compute_units: 0,
        });
        // The HSA lifecycle gate admits one initialization at a time. A first
        // GPU activation attempt may retain the primary KFD VM even if it
        // fails, so later generations must select a secondary context.
        let lifetime = if PRIMARY_CONTEXT_USED.load(Ordering::Acquire) {
            DriverContextLifetime::Session
        } else {
            DriverContextLifetime::Process
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
            &topology,
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
        host_topology: &HostTopologySnapshot,
        lifetime: DriverContextLifetime,
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
        let host_caches = selected_cpu_caches(host_topology);
        let mut caches = host_caches
            .records
            .iter()
            .filter(|cache| {
                cache.info.kind == CpuCacheKind::Data && cache.first_shared_cpu() == Some(cache.cpu)
            })
            .map(|cache| {
                Cache::new(
                    cpu_agent,
                    host.name.as_bytes(),
                    cache.info.level,
                    u32::try_from(cache.info.size_bytes).unwrap_or(u32::MAX),
                )
            })
            .collect::<Vec<_>>();
        pending
            .gpus
            .try_reserve(endpoints.len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        for mut endpoint in endpoints {
            if endpoint.gpu().is_none() {
                continue;
            }
            session
                .populate_endpoint(&mut endpoint)
                .map_err(map_error)?;
            let info = endpoint.gpu().copied().ok_or(ERROR)?;
            let agent = HsaAgent {
                handle: GPU_AGENT_BASE + pending.gpus.len() as u64,
            };
            let name = gpu_agent_name(&info);
            for cache in endpoint
                .caches()
                .iter()
                .filter(|cache| rocddi::device::gpu::is_compute_data_cache(cache))
            {
                caches.push(Cache::new(
                    agent,
                    name.as_bytes(),
                    cache.level,
                    u32::try_from(cache.size_bytes).unwrap_or(u32::MAX),
                ));
            }
            let asic_family_id = info.drm_asic_family_id.unwrap_or(info.asic_family_id);
            let timestamp_frequency_hz = info.counter_frequency_hz.unwrap_or(0);
            if lifetime == DriverContextLifetime::Process {
                // Activation may retain a primary VM even when it returns an
                // error. A later generation must use a private context.
                PRIMARY_CONTEXT_USED.store(true, Ordering::Release);
            }
            let device = session.activate(&endpoint).map_err(map_error)?;
            let product_name = endpoint
                .product_name
                .clone()
                .unwrap_or_else(|| "AMD Radeon Graphics".to_owned());
            let mmio_remap = device.gpu().and_then(|gpu| gpu.map_mmio_remap()).ok();
            let hdp_flush = hdp_flush_pointers(
                mmio_remap
                    .as_ref()
                    .and_then(|mapping| mapping.info().host_address),
            );
            let fine_grain_pool = info.hive_id != 0;
            let pc_sampling_supported = crate::pc_sampling::session::supported_gpu(info)
                && pc_sampling_supported_by_query(device.gpu().and_then(|gpu| {
                    gpu.pc_sampling_configurations(PcSamplingConfigurationSet::Supported)
                }));
            pending.gpus.push(Gpu {
                endpoint,
                info,
                device,
                name: name.into_boxed_str(),
                product_name: product_name.into_boxed_str(),
                asic_family_id,
                timestamp_frequency_hz,
                hdp_flush,
                _mmio_remap: mmio_remap,
                coherency_type: AMD_COHERENCY_TYPE_NONCOHERENT,
                fine_grain_pool,
                pc_sampling_supported,
                persisting_l2_cache_size: Arc::new(Mutex::new(0)),
            });
        }
        let gpus = std::mem::take(&mut pending.gpus);
        let session = pending.session.take().ok_or(ERROR)?;
        Ok(Self {
            logging: Arc::new(LogOutput::new()),
            next_handle: 0x4853_4101_0000_0000,
            queue_state: QueueState::new(),
            loader: LoaderState::new(),
            image_state: ImageState::new(),
            memory_state: MemoryState::new(),
            signal_state: SignalState::new(lifetime),
            host_memory_bytes,
            host_page_size,
            host_name: host.name.into_boxed_str(),
            host_compute_units: host.compute_units,
            full_profile,
            vm_fault_details: None,
            system_event_handlers: Vec::new(),
            system_event_worker: None,
            stop_workers: Arc::new(AtomicBool::new(false)),
            inflight: Arc::new(InFlightTracker::new()),
            pc_sampling_state: PcSamplingState::new(),
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
        if self.system_event_worker.is_some() {
            return SUCCESS;
        }
        let Some(device) = self.gpus.first().map(|gpu| gpu.device.clone()) else {
            return OUT_OF_RESOURCES;
        };
        let subscription = match subscribe(&device) {
            Ok(subscription) => subscription,
            Err(_) => return OUT_OF_RESOURCES,
        };
        let stop = self.stop_workers.clone();
        let worker = match thread::Builder::new()
            .name("rocddi-system-events".into())
            .spawn(move || system_event_worker(subscription, &stop))
        {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        self.system_event_worker = Some(worker);
        SUCCESS
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
        let gpu = self.gpus.get(index)?;
        super::isa_name(
            gpu.info.gfx_major,
            gpu.info.gfx_minor,
            gpu.info.gfx_stepping,
            gpu.info.asic_revision,
            variant,
        )
        .map(|_| (index, variant))
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
        let gpu = &self.gpus[index];
        let exposed = match kind {
            crate::memory::GPU_POOL_FINE => gpu.fine_grain_pool,
            crate::memory::GPU_POOL_EXTENDED => {
                gpu.info.gfx_major == 9 && matches!(gpu.info.gfx_minor, 4 | 5)
            }
            _ => true,
        };
        exposed.then_some((index, kind))
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
        if self.inflight.is_quarantined() {
            std::mem::forget(self);
            return INVALID_RUNTIME_STATE;
        }
        notify_system_shutdown(&self.system_event_handlers);
        if let Some(worker) = self.system_event_worker.take() {
            let _ = worker.join();
        }
        self.memory_state.join_workers();
        self.signal_state.join_worker();
        if self.memory_state.is_quarantined() {
            // A copy with unproved native retirement can still reach runtime
            // allocations and signals. Keep their entire ownership graph live.
            self.pc_sampling_state.quarantine_all();
            std::mem::forget(self);
            return INVALID_RUNTIME_STATE;
        }
        if !self.pc_sampling_state.shutdown() {
            std::mem::forget(self);
            return INVALID_RUNTIME_STATE;
        }
        if !self.queue_state.shutdown() {
            std::mem::forget(self);
            return INVALID_RUNTIME_STATE;
        }
        let signal_status = self.signal_state.destroy_events();
        self.image_state.clear();
        self.loader.clear();
        self.memory_state.clear();
        self.signal_state.clear_storage();
        self.pc_sampling_state.clear_retired();
        self.gpus.clear();
        let session_status = self.session.destroy().map_or_else(map_error, |()| SUCCESS);
        if signal_status == SUCCESS {
            session_status
        } else {
            signal_status
        }
    }
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
    let counters = clock_counters(device).map_err(map_error)?;
    Ok((
        translate_gpu_tick(counters, start)?,
        translate_gpu_tick(counters, end)?,
    ))
}

pub(crate) fn translate_gpu_tick(
    counters: rocddi::device::gpu::profiling::ClockCounters,
    tick: u64,
) -> Result<u64, Status> {
    if counters.system_frequency == 0 || counters.gpu_frequency == 0 {
        return Err(ERROR);
    }
    let scaled = |delta: u64| {
        u64::try_from(
            u128::from(delta) * u128::from(counters.system_frequency)
                / u128::from(counters.gpu_frequency),
        )
        .unwrap_or(u64::MAX)
    };
    Ok(if tick >= counters.gpu {
        counters.system.wrapping_add(scaled(tick - counters.gpu))
    } else {
        counters.system.wrapping_sub(scaled(counters.gpu - tick))
    })
}

/// Process registry phase and generation during initialization or teardown.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LifecyclePhase {
    /// No HSA runtime is published.
    Inactive,
    /// A new generation is acquiring native resources.
    Starting(u64),
    /// The generation is invoking tool initialization callbacks.
    LoadingTools(u64),
    /// The generation is available to HSA callers.
    Active(u64),
    /// Final shutdown is retiring the generation.
    Stopping(u64),
    /// Failed native cleanup prevents a safe new generation.
    Quarantined,
}

/// Process-wide gate for HSA generation transitions.
pub(crate) struct HsaLifecycle {
    /// Published HSA adapter state for the active generation.
    state: Option<HsaState>,
    /// Successful `hsa_init` calls awaiting shutdown for the active generation.
    references: u32,
    /// Last generation number assigned by initialization.
    generation: u64,
    /// Current transition phase under the registry mutex.
    phase: LifecyclePhase,
}

impl HsaLifecycle {
    const fn new() -> Self {
        Self {
            state: None,
            references: 0,
            generation: 0,
            phase: LifecyclePhase::Inactive,
        }
    }

    pub(crate) fn as_ref(&self) -> Option<&HsaState> {
        self.state.as_ref()
    }

    pub(crate) fn as_mut(&mut self) -> Option<&mut HsaState> {
        self.state.as_mut()
    }

    /// Returns the generation of the currently published runtime.
    pub(crate) fn active_generation(&self) -> Option<u64> {
        match (self.phase, self.state.as_ref()) {
            (
                LifecyclePhase::LoadingTools(generation) | LifecyclePhase::Active(generation),
                Some(_),
            ) => Some(generation),
            _ => None,
        }
    }

    /// Returns None after adding a reference to an already active runtime.
    pub(crate) fn begin_init(&mut self) -> Result<Option<u64>, Status> {
        match self.phase {
            LifecyclePhase::Active(_) => {
                if self.state.is_none() {
                    return Err(ERROR);
                }
                if self.references == i32::MAX as u32 {
                    return Err(0x100c);
                }
                self.references += 1;
                Ok(None)
            }
            LifecyclePhase::Inactive => {
                self.generation = self.generation.checked_add(1).ok_or(ERROR)?;
                self.phase = LifecyclePhase::Starting(self.generation);
                Ok(Some(self.generation))
            }
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
            | LifecyclePhase::Stopping(_)
            | LifecyclePhase::Quarantined => Err(INVALID_RUNTIME_STATE),
        }
    }

    pub(crate) fn publish_init(
        &mut self,
        generation: u64,
        runtime: HsaState,
    ) -> Result<(), Status> {
        if self.phase != LifecyclePhase::Starting(generation) {
            return Err(ERROR);
        }
        self.state = Some(runtime);
        self.references = 1;
        // HSA queries may run from a tool's OnLoad, but recursive init and
        // shutdown must remain rejected until the tool table is published.
        self.phase = LifecyclePhase::LoadingTools(generation);
        Ok(())
    }

    pub(crate) fn finish_init(&mut self, generation: u64) -> Result<(), Status> {
        if self.phase != LifecyclePhase::LoadingTools(generation) || self.state.is_none() {
            return Err(ERROR);
        }
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
        self.references = 0;
        Ok(())
    }

    /// Returns None after releasing one of several active references.
    pub(crate) fn begin_shutdown(&mut self) -> Result<Option<(HsaState, u64)>, Status> {
        match self.phase {
            LifecyclePhase::Active(generation) => {
                if self.state.is_none() {
                    return Err(ERROR);
                }
                if self.references > 1 {
                    self.references -= 1;
                    return Ok(None);
                }
                let runtime = self.state.take().ok_or(ERROR)?;
                self.references = 0;
                self.phase = LifecyclePhase::Stopping(generation);
                Ok(Some((runtime, generation)))
            }
            LifecyclePhase::Inactive => Err(NOT_INITIALIZED),
            LifecyclePhase::Starting(_)
            | LifecyclePhase::LoadingTools(_)
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

/// Quarantines a generation if initialization or shutdown unwinds.
pub(crate) struct LifecycleTransition {
    /// Generation phase that will be quarantined if this guard unwinds.
    phase: LifecyclePhase,
    /// Cleared once the transition finishes successfully.
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
        let mut registry = LIFECYCLE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if registry.phase == self.phase
            || matches!(
                (self.phase, registry.phase),
                (LifecyclePhase::Starting(expected), LifecyclePhase::LoadingTools(current))
                    if expected == current
            )
        {
            registry.phase = LifecyclePhase::Quarantined;
        }
    }
}

pub(crate) static LIFECYCLE: Mutex<HsaLifecycle> = Mutex::new(HsaLifecycle::new());
pub(crate) static VM_FAULT_CONDVAR: Condvar = Condvar::new();

pub(crate) fn lock() -> Result<MutexGuard<'static, HsaLifecycle>, Status> {
    LIFECYCLE.lock().map_err(|_| ERROR)
}

#[allow(clippy::needless_pass_by_value)]
pub(crate) fn map_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
        // Busy describes a live native dependency. Only queue-handle
        // validation in the queue adapter may return INVALID_QUEUE.
        rocddi::ErrorKind::Busy => RESOURCE_BUSY,
        _ => ERROR,
    }
}

pub(crate) fn boundary(operation: impl FnOnce() -> Status) -> Status {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(ERROR)
}

pub(crate) fn initialized_mut(runtime: &mut HsaLifecycle) -> Result<&mut HsaState, Status> {
    runtime.as_mut().ok_or(NOT_INITIALIZED)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rocddi::device::gpu::profiling::ClockCounters;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

    #[test]
    fn precise_clock_sampling_keeps_the_shortest_complete_sample() {
        let mut samples = [
            Ok::<_, &str>((10, Duration::from_micros(40))),
            Ok((20, Duration::from_micros(3))),
            Ok((30, Duration::from_micros(12))),
            Ok((40, Duration::from_micros(1))),
        ]
        .into_iter();
        assert_eq!(best_clock_sample(4, || samples.next().unwrap()), Ok(40));

        let mut calls = 0;
        assert_eq!(
            best_clock_sample(4, || {
                calls += 1;
                if calls == 2 {
                    Err("late ioctl failure")
                } else {
                    Ok((17, Duration::from_micros(2)))
                }
            }),
            Ok(17)
        );
        assert_eq!(calls, 2);
    }

    #[test]
    fn native_busy_reports_resource_contention_across_frontends() {
        let busy = || rocddi::Error::Operation {
            kind: rocddi::ErrorKind::Busy,
            detail: "injected live native dependency",
        };
        assert_eq!(map_error(busy()), RESOURCE_BUSY);
        assert_ne!(map_error(busy()), INVALID_QUEUE);
    }

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
        let mut registry = HsaLifecycle::new();
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
        let mut registry = HsaLifecycle::new();
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
        for (cause, extra_reason) in [
            (GpuMemoryFaultCause::None, 0),
            (GpuMemoryFaultCause::SramEcc, AMD_MEMORY_FAULT_SRAM_ECC),
            (GpuMemoryFaultCause::DramEcc, AMD_MEMORY_FAULT_DRAM_ECC),
            (GpuMemoryFaultCause::Hang, AMD_MEMORY_FAULT_HANG),
        ] {
            let event = memory_fault_event(
                agent,
                GpuMemoryFault {
                    endpoint_id: Some([42; 16]),
                    virtual_address: 0x5678_9000,
                    page_not_present: true,
                    read_only: true,
                    no_execute: true,
                    imprecise: true,
                    cause,
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
    fn hardware_exception_events_match_the_amd_extension_layout() {
        let agent = HsaAgent { handle: 0x1234 };
        for (cause, expected_cause) in [
            (GpuResetCause::Hang, AMD_HW_EXCEPTION_CAUSE_GPU_HANG),
            (GpuResetCause::Ecc, AMD_HW_EXCEPTION_CAUSE_ECC),
        ] {
            let event = hardware_exception_event(
                agent,
                GpuHardwareException {
                    endpoint_id: Some([42; 16]),
                    scope: rocddi::device::event::GpuResetScope::WholeGpu,
                    memory_lost: false,
                    cause,
                },
            );
            assert_eq!(event.event_type, AMD_GPU_HW_EXCEPTION_EVENT);
            assert_eq!(event.payload[0], agent.handle);
            assert_eq!(
                event.payload[1],
                u64::from(AMD_HW_EXCEPTION_RESET_TYPE_OTHER) | (u64::from(expected_cause) << 32)
            );
            assert_eq!(event.payload[2], 0);
        }
    }

    #[test]
    fn gpu_ticks_translate_around_the_correlated_system_sample() {
        let counters = ClockCounters {
            gpu: 100,
            host: 0,
            system: 1_000,
            system_frequency: 1_000_000_000,
            gpu_frequency: 100_000_000,
        };
        assert_eq!(translate_gpu_tick(counters, 110), Ok(1_100));
        assert_eq!(translate_gpu_tick(counters, 90), Ok(900));
        assert_eq!(
            translate_gpu_tick(
                ClockCounters {
                    gpu_frequency: 200_000_000,
                    ..counters
                },
                110,
            ),
            Ok(1_050)
        );
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
    fn cpu_cache_source_prefers_numa_zero_and_falls_back_to_system_scan() {
        use rocddi::topology::host::{CpuCache, HostDomain};

        let cache = |cpu| CpuCache {
            cpu,
            shared_cpus: None,
            info: rocddi::topology::CacheInfo::new(1, 32 * 1024, None, CpuCacheKind::Data),
        };
        let mut topology = HostTopologySnapshot {
            memory_bytes: None,
            cpu: None,
            online_cpus: None,
            caches: HostCaches {
                records: vec![cache(7)],
                complete: true,
            },
            domains: vec![HostDomain {
                numa_node: 2,
                cpus: None,
                caches: HostCaches {
                    records: vec![cache(2)],
                    complete: true,
                },
            }],
            domains_complete: true,
        };
        assert_eq!(selected_cpu_caches(&topology).records[0].cpu, 7);

        topology.domains.push(HostDomain {
            numa_node: 0,
            cpus: None,
            caches: HostCaches {
                records: vec![cache(0)],
                complete: true,
            },
        });
        assert_eq!(selected_cpu_caches(&topology).records[0].cpu, 0);
        topology.domains[1].caches.records.clear();
        assert!(selected_cpu_caches(&topology).records.is_empty());
    }

    #[test]
    fn cache_iteration_sorts_levels_without_changing_handles() {
        let agent = HsaAgent { handle: CPU_AGENT };
        let other = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let caches = [
            Cache::new(agent, b"CPU", 3, 100),
            Cache::new(agent, b"CPU", 1, 100),
            Cache::new(agent, b"CPU", 2, 100),
            Cache::new(agent, b"CPU", 1, 100),
            Cache::new(other, b"GPU", 1, 100),
        ];
        let handles = cache_handles_in_level_order(&caches, agent)
            .into_iter()
            .map(|cache| cache.handle)
            .collect::<Vec<_>>();
        assert_eq!(
            handles,
            [CACHE_BASE + 1, CACHE_BASE + 3, CACHE_BASE + 2, CACHE_BASE]
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

    #[test]
    #[ignore = "requires a qualified GPU and KFD runtime"]
    fn hsa_runtime_reinitializes_after_shutdown() {
        for _ in 0..3 {
            assert_eq!(crate::hsa_init(), SUCCESS);
            assert_eq!(crate::hsa_shut_down(), SUCCESS);
        }
    }
}
