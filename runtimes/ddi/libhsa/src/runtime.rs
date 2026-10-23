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
use rocddi::topology::host::{CpuCacheKind, CpuInfo, HostCaches, HostTopologySnapshot};
use rocddi::topology::{Endpoint, GpuInfo, TopologySnapshot};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::image_abi::ImageState;
use crate::isa::isa_name;
use crate::loader::LoaderState;
use crate::memory::MemoryState;
use crate::pc_sampling::PcSamplingState;
use crate::queue::QueueState;
use crate::signal::SignalState;

mod agent;
mod clock;
mod events;
pub(crate) use agent::{Cache, Gpu, cache_handles_in_level_order};
use agent::{
    cache_sizes, decode_cache_handle, decode_gpu_pool_handle, full_profile_platform,
    gpu_agent_name, hdp_flush_pointers, pc_sampling_supported_by_query, selected_cpu_caches,
};
pub(crate) use events::CallbackScope;
use events::notify_system_shutdown;
#[cfg(test)]
use events::{hardware_exception_event, memory_fault_event, notify_system_event};
mod lifecycle;
#[cfg(test)]
use clock::best_clock_sample;
pub(crate) use clock::{clock_counters, translate_gpu_interval, translate_gpu_tick};
mod logging;
#[cfg(test)]
pub(crate) use lifecycle::{HsaLifecycle, LIFECYCLE};
pub(crate) use lifecycle::{
    InFlightGeneration, InFlightToken, InFlightTracker, LifecycleTransition, VM_FAULT_CONDVAR,
    boundary, initialized_mut, lock, map_error,
};
#[cfg(test)]
use logging::logging_flag_enabled;
pub(crate) use logging::{LogOutput, LogWork, LogWriteScope};

// KFD retains a primary process VM after its last descriptor closes. Later
// HSA generations therefore need a private context to acquire a fresh VM.
/// Records whether KFD’s process-primary VM has already been used.
static PRIMARY_CONTEXT_USED: AtomicBool = AtomicBool::new(false);

/// Initialization error with evidence that native rollback also failed.
pub(crate) struct InitFailure {
    /// HSA status returned to the caller.
    pub(crate) status: Status,
    /// Native state was retained because cleanup could not be proved.
    pub(crate) cleanup_failed: bool,
}

impl From<Status> for InitFailure {
    /// Wraps an HSA status when rollback itself has not failed.
    fn from(status: Status) -> Self {
        Self {
            status,
            cleanup_failed: false,
        }
    }
}

/// Minimal cleanup contract shared by real and test session owners.
trait RetireSession {
    /// Releases native resources and reports whether cleanup was conclusive.
    fn retire(&mut self) -> bool;
}

impl RetireSession for Session {
    /// Destroys the native session and reports whether cleanup succeeded.
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
    /// Drops activated GPUs before retiring the pending native session.
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
    /// Rolls back unpublished GPUs and their native session.
    fn drop(&mut self) {
        self.cleanup();
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
    next_handle: u64,
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
    system_event_handlers: Vec<(SystemEventHandler, CallbackArg)>,
    /// Native system event worker joined during final teardown.
    pub(crate) system_event_worker: Option<JoinHandle<()>>,
    /// Shared stop request observed by those workers.
    pub(crate) stop_workers: Arc<AtomicBool>,
    /// Counts native work that must finish before session release.
    pub(crate) inflight: Arc<InFlightTracker>,
    /// CPU and GPU cache records exposed through HSA queries.
    pub(crate) caches: Vec<Cache>,
    /// Passive node, cache, and reported-route facts for this HSA generation.
    pub(crate) topology: TopologySnapshot,
    /// Activated GPU agents and their native owners.
    pub(crate) gpus: Vec<Gpu>,
    /// Native session retained until all GPU and memory owners are gone.
    pub(crate) session: Session,
    /// Lifetime policy for the KFD context used by this runtime generation.
    /// Memory-pool choices that require the primary context use this value.
    pub(crate) lifetime: DriverContextLifetime,
}

/// Schedules cleanup after in-flight native work has released its borrows.
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
    /// Checks whether the caller is one of this runtime’s callback workers.
    pub(crate) fn running_on_worker(&self) -> bool {
        self.signal_state.running_on_worker()
            || self.memory_state.running_on_worker()
            || self
                .system_event_worker
                .as_ref()
                .is_some_and(|worker| worker.thread().id() == thread::current().id())
    }

    /// Requests all runtime workers to stop before teardown.
    pub(crate) fn request_stop(&self) {
        self.logging.stop();
        self.stop_workers.store(true, Ordering::Release);
    }

    /// Discovers host and GPU agents and constructs process HSA state.
    pub(crate) fn create() -> Result<Self, InitFailure> {
        let host_page_size =
            usize::try_from(rocddi::memory::host::host_page_size().map_err(map_error)?)
                .map_err(|_| ERROR)?;
        if host_page_size < 4096 || !host_page_size.is_power_of_two() {
            return Err(ERROR.into());
        }
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
        let topology = match pending
            .session
            .as_ref()
            .ok_or(ERROR)
            .and_then(|session| session.topology_snapshot().map_err(map_error))
        {
            Ok(topology) => topology,
            Err(status) => {
                return Err(InitFailure {
                    status,
                    cleanup_failed: !pending.cleanup(),
                });
            }
        };
        let Some(host_memory_bytes) = topology.host.memory_bytes else {
            return Err(InitFailure {
                status: ERROR,
                cleanup_failed: !pending.cleanup(),
            });
        };
        let host = topology.host.cpu.clone().unwrap_or_else(|| CpuInfo {
            name: "CPU".to_owned(),
            compute_units: 0,
        });
        match Self::create_with_session(
            &mut pending,
            host_page_size,
            host_memory_bytes,
            host,
            topology,
            lifetime,
        ) {
            Ok(runtime) => Ok(runtime),
            Err(status) => Err(InitFailure {
                status,
                cleanup_failed: !pending.cleanup(),
            }),
        }
    }

    /// Builds HSA state from a session retained through initialization.
    fn create_with_session(
        pending: &mut PendingInit,
        host_page_size: usize,
        host_memory_bytes: usize,
        host: CpuInfo,
        topology: TopologySnapshot,
        lifetime: DriverContextLifetime,
    ) -> Result<Self, Status> {
        let session = pending.session.as_ref().ok_or(ERROR)?;
        let endpoints = topology
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.gpu().is_some_and(|gpu| gpu.queues.aql))
            .cloned()
            .collect::<Vec<_>>();
        if endpoints.is_empty() {
            return Err(OUT_OF_RESOURCES);
        }
        let full_profile =
            full_profile_platform(endpoints.iter().map(|endpoint| endpoint.local_memory_bytes));
        let cpu_agent = HsaAgent { handle: CPU_AGENT };
        let host_caches = selected_cpu_caches(&topology.host);
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
            for cache in topology
                .caches_for_endpoint(&endpoint)
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
            let xnack_enabled = if info.gfx_major == 9 {
                device
                    .gpu()
                    .and_then(|gpu| gpu.xnack_enabled())
                    .map_err(map_error)?
            } else {
                false
            };
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
                xnack_enabled,
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
            topology,
            gpus,
            session,
            lifetime,
        })
    }

    /// Allocates handle for runtime use.
    pub(crate) fn allocate_handle(&mut self) -> Result<u64, Status> {
        let handle = self.next_handle;
        self.next_handle = self.next_handle.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(handle)
    }

    /// Returns the GPU index encoded in a valid HSA agent handle.
    pub(crate) fn gpu_index(&self, agent: HsaAgent) -> Option<usize> {
        let offset = agent.handle.checked_sub(GPU_AGENT_BASE)?;
        let index = usize::try_from(offset).ok()?;
        (index < self.gpus.len()).then_some(index)
    }

    /// Checks whether a handle names the host CPU or a discovered GPU.
    pub(crate) fn is_agent(&self, agent: HsaAgent) -> bool {
        agent.handle == CPU_AGENT || self.gpu_index(agent).is_some()
    }

    /// Returns the cache index encoded in a valid HSA cache handle.
    pub(crate) fn cache_index(&self, cache: HsaCache) -> Option<usize> {
        decode_cache_handle(cache, self.caches.len())
    }

    /// Returns cache sizes by level for the requested agent.
    pub(crate) fn agent_cache_sizes(&self, agent: HsaAgent) -> [u32; 4] {
        cache_sizes(&self.caches, agent)
    }

    /// Decodes an ISA handle into GPU index and variant.
    pub(crate) fn isa_parts(&self, isa: HsaIsa) -> Option<(usize, u64)> {
        let offset = isa.handle.checked_sub(ISA_BASE)?;
        let index = usize::try_from(offset / ISA_COUNT_PER_GPU).ok()?;
        let variant = offset % ISA_COUNT_PER_GPU;
        let gpu = self.gpus.get(index)?;
        isa_name(&gpu.info, gpu.xnack_enabled, variant).map(|_| (index, variant))
    }

    /// Returns the GPU index associated with a valid ISA handle.
    pub(crate) fn isa_index(&self, isa: HsaIsa) -> Option<usize> {
        self.isa_parts(isa).map(|(index, _)| index)
    }

    /// Finds the ISA handle associated with a wavefront handle.
    pub(crate) fn wavefront_isa(&self, wavefront: HsaWavefront) -> Option<HsaIsa> {
        let offset = wavefront.handle.checked_sub(WAVEFRONT_BASE)?;
        let isa = HsaIsa {
            handle: ISA_BASE.checked_add(offset)?,
        };
        self.isa_parts(isa).map(|_| isa)
    }

    /// Builds a synthetic memory-pool handle for a GPU and pool kind.
    pub(crate) fn gpu_pool(index: usize, kind: u64) -> HsaMemoryPool {
        HsaMemoryPool {
            handle: GPU_POOL_BASE + (index as u64) * 0x10 + kind,
        }
    }

    /// Decodes GPU pool into its runtime components.
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

    /// Prepares log before publication or execution.
    pub(crate) fn prepare_log(&self, flag: u32, arguments: Arguments<'_>) -> Option<LogWork> {
        self.logging.prepare(&self.inflight, flag, arguments)
    }

    /// Stops this session.
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
