// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Constructs an unpublished HSA session and owns failed startup rollback.

use super::*;
use crate::agent::{Cache, Gpu};
use crate::agent::{
    cpu_agents, full_profile_platform, gpu_agent_name, hdp_flush_pointers, host_caches_for_cpu,
    pc_sampling_supported_by_query,
};
use crate::config;
use rocddi::device::gpu::profiling::PcSamplingConfigurationSet;
#[cfg(target_os = "linux")]
use rocddi::driver::linux_kfd::{KfdConfiguration, XnackModeRequest};
use rocddi::topology::host::{CpuCacheKind, CpuInfo};
use rocddi::topology::{CacheAssociations, MemoryBankKind};

/// Startup failure together with the owner of any unfinished rollback.
pub(crate) struct InitFailure {
    /// Original HSA initialization status, preserved while rollback is pending.
    pub(crate) status: Status,
    /// Cleanup progress must not erase a retryable native retirement owner.
    pub(crate) rollback: InitRollback,
}

/// Ownership remaining after an unpublished HSA session has rolled back.
pub(crate) enum InitRollback {
    /// All startup resources have retired.
    Complete,
    /// Frontend owners are gone; only native session retirement remains.
    NativeRetry(SessionRetirement),
    /// An uncertain frontend rollback retained its dependency graph.
    Uncertain,
}

impl From<Status> for InitFailure {
    fn from(status: Status) -> Self {
        Self {
            status,
            rollback: InitRollback::Complete,
        }
    }
}

/// Holds startup resources until publication or rollback finishes.
pub(super) struct PendingInit<G = Gpu> {
    /// Trap publication owners retained through startup rollback.
    pub(super) debugger: Option<PendingDebugger>,
    /// Session released after all activated GPU owners.
    pub(super) native_session: Option<NativeSession>,
    /// Activated GPUs that must be dropped before their native session.
    pub(super) gpus: Vec<G>,
}

impl<G> PendingInit<G> {
    /// Closes admission and releases frontend borrowers before native retirement.
    pub(super) fn cleanup(&mut self) -> InitRollback {
        let Some(native_session) = self.native_session.take() else {
            return InitRollback::Complete;
        };
        // An early error or unwind must retain every owner still reachable by
        // native work. Extract owners only after their release proof succeeds.
        let native_session = RetainOnDrop::new(native_session);
        let debugger = RetainOnDrop::new(self.debugger.take());
        let gpus = RetainOnDrop::new(std::mem::take(&mut self.gpus));
        if native_session.begin_retirement().is_err() {
            return InitRollback::Uncertain;
        }
        // Normal traps remain owned by the shared process context. Uncertain
        // trap publication queues retain their own native dependencies.
        drop(debugger.into_inner());
        drop(gpus.into_inner());
        let mut retirement = native_session.into_inner().into_retirement();
        match retirement.try_close() {
            RetirementOutcome::Closed => InitRollback::Complete,
            RetirementOutcome::Busy | RetirementOutcome::Failed(_) => {
                InitRollback::NativeRetry(retirement)
            }
        }
    }
}

impl<G> Drop for PendingInit<G> {
    fn drop(&mut self) {
        // If unwinding prevents returning a typed result, SessionRetirement's
        // destructor retains unfinished native ownership until process exit.
        self.cleanup();
    }
}

impl HsaSession {
    /// Discovers host and GPU agents and constructs process HSA state.
    pub(crate) fn create() -> Result<Self, InitFailure> {
        let host_page_size =
            usize::try_from(rocddi::memory::host::host_page_size().map_err(map_error)?)
                .map_err(|_| ERROR)?;
        if host_page_size < 4096 || !host_page_size.is_power_of_two() {
            return Err(ERROR.into());
        }
        // Clean HSA generations rejoin the exact retained native process VM.
        // Native process ownership validates identity, loss and prior disable;
        // the HSA lifecycle gate prevents rejoining an uncertain generation.
        let native_session =
            NativeSession::new(DriverContextLifetime::Process).map_err(map_error)?;
        let mut pending = PendingInit {
            debugger: None,
            native_session: Some(native_session),
            gpus: Vec::new(),
        };
        let topology = match pending
            .native_session
            .as_ref()
            .ok_or(ERROR)
            .and_then(|native_session| native_session.topology_graph_snapshot().map_err(map_error))
        {
            Ok(topology) => topology,
            Err(status) => {
                return Err(InitFailure {
                    status,
                    rollback: pending.cleanup(),
                });
            }
        };
        let host_memory_bytes = topology.host.memory_bytes.unwrap_or_else(|| {
            topology
                .memory_banks
                .iter()
                .filter(|bank| bank.kind == MemoryBankKind::System)
                .filter(|bank| topology.nodes[bank.node.0].cpu.is_some())
                .filter_map(|bank| usize::try_from(bank.capacity_bytes).ok())
                .fold(0usize, usize::saturating_add)
        });
        if host_memory_bytes == 0 {
            return Err(InitFailure {
                status: ERROR,
                rollback: pending.cleanup(),
            });
        }
        let host = topology.host.cpu.clone().unwrap_or(CpuInfo {
            name: None,
            compute_units: 0,
        });
        match Self::create_with_native_session(
            &mut pending,
            host_page_size,
            host_memory_bytes,
            &host,
            topology,
        ) {
            Ok(runtime) => Ok(runtime),
            Err(status) => Err(InitFailure {
                status,
                rollback: pending.cleanup(),
            }),
        }
    }

    /// Builds HSA state from a native session retained through initialization.
    fn create_with_native_session(
        pending: &mut PendingInit,
        host_page_size: usize,
        host_memory_bytes: usize,
        host: &CpuInfo,
        topology: TopologyGraphSnapshot,
    ) -> Result<Self, Status> {
        #[cfg(target_os = "linux")]
        {
            let xnack = std::env::var("HSA_XNACK")
                .ok()
                .and_then(|value| match value.as_str() {
                    "0" => Some(XnackModeRequest::Prefer(false)),
                    "1" => Some(XnackModeRequest::Prefer(true)),
                    _ => None,
                });
            let configuration = KfdConfiguration {
                xnack,
                debugger: Some(PendingDebugger::registration()?),
            };
            pending
                .native_session
                .as_mut()
                .ok_or(ERROR)?
                .configure(configuration)
                .map_err(map_error)?;
        }
        pending.debugger = Some(PendingDebugger::default());
        let native_session = pending.native_session.as_ref().ok_or(ERROR)?;
        let discovered = topology
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.gpu().is_some())
            .collect::<Vec<_>>();
        let ids = discovered
            .iter()
            .map(|endpoint| endpoint.gpu().and_then(|gpu| gpu.unique_id))
            .collect::<Vec<_>>();
        let selection = std::env::var_os("ROCR_VISIBLE_DEVICES");
        let selection = selection.as_ref().map(|value| value.to_string_lossy());
        let endpoints = config::visible_gpu_indices(&ids, selection.as_deref())
            .into_iter()
            .filter_map(|index| {
                let endpoint = discovered[index];
                endpoint
                    .gpu()
                    .is_some_and(config::supports_execution)
                    .then(|| endpoint.clone())
            })
            .collect::<Vec<_>>();
        if endpoints.is_empty() && selection.is_none() {
            return Err(OUT_OF_RESOURCES);
        }
        let full_profile = endpoints.is_empty()
            || full_profile_platform(endpoints.iter().map(|endpoint| endpoint.local_memory_bytes));
        let cpus = cpu_agents(&topology, host, host_memory_bytes);
        let system_alloc_max_bytes = cpus
            .iter()
            .map(|cpu| cpu.memory_bytes & !(host_page_size - 1))
            .fold(0usize, usize::saturating_add);
        let mut caches = Vec::new();
        for (index, cpu) in cpus.iter().enumerate() {
            let agent = HsaAgent {
                handle: CPU_AGENT + index as u64,
            };
            let native_caches = cpu.node.into_iter().flat_map(|node| {
                topology.caches.iter().filter(move |cache| {
                    cache.node == node
                        && cache.info.associations.contains(CacheAssociations::CPU)
                        && matches!(cache.info.kind, CpuCacheKind::Data | CpuCacheKind::Other)
                })
            });
            let start = caches.len();
            for cache in native_caches {
                caches.push(Cache::new(
                    agent,
                    cpu.name.as_bytes(),
                    cache.info.level,
                    u32::try_from(cache.info.size_bytes).unwrap_or(u32::MAX),
                ));
            }
            if caches.len() == start {
                // KFD commonly reports no CPU cache records. Use OS caches only
                // when the KFD CPU ID identifies a domain, or there is one CPU agent.
                let selected = host_caches_for_cpu(
                    &topology.host,
                    cpu.node
                        .and_then(|node| topology.nodes[node.0].cpu)
                        .and_then(|facts| facts.core_id_base),
                    cpus.len(),
                );
                if let Some(host_caches) = selected {
                    let records = &host_caches.records;
                    for cache in records.iter().filter(|cache| {
                        cache.info.kind == CpuCacheKind::Data
                            && cache.first_shared_cpu() == Some(cache.cpu)
                    }) {
                        caches.push(Cache::new(
                            agent,
                            cpu.name.as_bytes(),
                            cache.info.level,
                            u32::try_from(cache.info.size_bytes).unwrap_or(u32::MAX),
                        ));
                    }
                }
            }
        }
        pending
            .gpus
            .try_reserve(endpoints.len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        let scratch_options = crate::queue::ScratchOptions::from_env();
        for mut endpoint in endpoints {
            if endpoint.gpu().is_none() {
                continue;
            }
            native_session
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
            let device = native_session.activate(&endpoint).map_err(map_error)?;
            pending
                .debugger
                .as_mut()
                .ok_or(ERROR)?
                .install(&device, &AtomicBool::new(false))?;
            let xnack_enabled =
                if info.gfx_major == 9 || (info.gfx_major == 10 && info.gfx_minor == 1) {
                    device
                        .gpu()
                        .and_then(|gpu| gpu.xnack_enabled())
                        .map_err(map_error)?
                } else {
                    false
                };
            let mmio_remap = device.gpu().and_then(|gpu| gpu.map_mmio_remap()).ok();
            let hdp_flush = hdp_flush_pointers(
                mmio_remap
                    .as_ref()
                    .and_then(|mapping| mapping.info().host_address),
            );
            let fine_grain_pool = info.hive_id != 0
                || std::env::var_os("HSA_FORCE_FINE_GRAIN_PCIE").is_some_and(|value| value == "1");
            let pc_sampling_supported = crate::pc_sampling::session::supported_gpu(&info)
                && pc_sampling_supported_by_query(device.gpu().and_then(|gpu| {
                    gpu.pc_sampling_configurations(PcSamplingConfigurationSet::Supported)
                }));
            pending.gpus.push(Gpu {
                scratch: Arc::new(scratch_options.for_gpu(&info)),
                global_cu_mask: None,
                cp_caps: Arc::new(Mutex::new(None)),
                endpoint,
                xnack_enabled,
                device,
                name: name.into_boxed_str(),
                hdp_flush,
                _mmio_remap: mmio_remap,
                coherency_type: AMD_COHERENCY_TYPE_NONCOHERENT,
                fine_grain_pool,
                pc_sampling_supported,
                persisting_l2_cache_size: Arc::new(Mutex::new(0)),
            });
        }
        let counts = pending
            .gpus
            .iter()
            .map(|gpu| gpu.info().compute_unit_count)
            .collect::<Vec<_>>();
        let masks =
            config::cu_mask::parse(&std::env::var("HSA_CU_MASK").unwrap_or_default(), &counts);
        for (gpu, mask) in pending.gpus.iter_mut().zip(masks) {
            gpu.global_cu_mask = mask;
        }
        let signal_state = SignalState::new(pending.native_session.as_ref().ok_or(ERROR)?)?;
        let native_session = pending.native_session.take().ok_or(ERROR)?;
        let gpus = std::mem::take(&mut pending.gpus);
        let control = SessionControl::new(process_admission().clone());
        let health = control.health.clone();
        Ok(Self {
            agents: Agents {
                host_page_size,
                system_alloc_max_bytes,
                cpus,
                full_profile,
                caches,
                topology,
                gpus,
            },
            config: HsaConfig {
                core_dump: crate::core_dump::Config::from_environment(),
                queue_fault_message_enabled: config::queue_fault_message_enabled(
                    std::env::var("HSA_ENABLE_QUEUE_FAULT_MESSAGE")
                        .ok()
                        .as_deref(),
                ),
                yielding_poll_enabled: std::env::var_os("HSA_ENABLE_YIELDING_POLL")
                    .is_some_and(|value| value == "1"),
                sdma_enabled: config::sdma_enabled(
                    std::env::var("HSA_ENABLE_SDMA").ok().as_deref(),
                ),
                peer_sdma_enabled: config::sdma_enabled(
                    std::env::var("HSA_ENABLE_PEER_SDMA").ok().as_deref(),
                ),
                disable_fragment_allocator: std::env::var_os("HSA_DISABLE_FRAGMENT_ALLOCATOR")
                    .is_some_and(|value| value == "1"),
            },
            control,
            debugger: pending.debugger.take(),
            logging: Arc::new(LogOutput::new()),
            next_handle: 0x4853_4101_0000_0000,
            queue_state: QueueState::new(config::counted_queue_limit(
                std::env::var("GPU_MAX_HW_QUEUES").ok().as_deref(),
            )),
            loader: LoaderState::new(),
            image_state: ImageState::new(),
            memory_state: MemoryState::new(health.clone()),
            signal_state,
            events: SystemEvents::new(),
            pc_sampling_state: PcSamplingState::new(health),
            native_session: Some(native_session),
        })
    }
}
