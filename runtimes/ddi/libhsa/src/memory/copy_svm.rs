// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! SVM policy, asynchronous copies, and native copy route execution.

use super::*;

/// Copies a complete range using the HSA memory owner's selected host/GPU route.
///
/// The caller is already admitted to the HSA generation. This operation retains
/// registry backing and an in-flight lease before releasing the registry lock.
/// An uncertain native outcome quarantines the generation and retains operands.
///
/// # Safety
/// Both ranges must remain valid for `size` bytes through completion. The source
/// must be readable and the destination writable under the HSA memory contract.
/// If native retirement is uncertain, external backing must remain live until
/// conclusive native teardown.
pub(crate) unsafe fn copy(dst: *mut c_void, src: *const c_void, size: usize) -> Result<(), Status> {
    let mut guard = lock()?;
    let runtime = initialized_mut(&mut guard)?;
    if !memory_copy_has_work(dst, src, size)? {
        return Ok(());
    }
    let host_destination = host_copy_address(runtime, dst as usize, size, ACCESS_PERMISSION_WO);
    let host_source = host_copy_address(runtime, src as usize, size, ACCESS_PERMISSION_RO);
    if let (Ok(destination), Ok(source)) = (host_destination, host_source) {
        // SAFETY: The caller retains both ranges. The registry lock retains
        // owned backing through this overlapping host copy.
        unsafe { std::ptr::copy(source as *const u8, destination as *mut u8, size) };
        fence(Ordering::SeqCst);
        return Ok(());
    }
    let resolved = resolve_sync_gpu_copy(
        runtime,
        dst as usize,
        src as usize,
        size,
        host_destination,
        host_source,
    )?;
    let _inflight = runtime.control.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    let mut retained = [None; 2];
    for (slot, base) in retained.iter_mut().zip(resolved.borrowed_memory) {
        if let Some(base) = base {
            if let Err(status) = runtime.memory_state.retain_copy_borrow(base) {
                for base in retained.into_iter().flatten() {
                    runtime.memory_state.release_copy_borrow(base);
                }
                return Err(status);
            }
            *slot = Some(base);
        }
    }
    let stop = runtime.control.operation_stop();
    let health = runtime.control.health.clone();
    drop(guard);

    let (status, may_be_live) = match execute_sync_gpu_copy(resolved.route, &stop) {
        Ok(SyncGpuCopyOutcome::Done) => (SUCCESS, false),
        Ok(SyncGpuCopyOutcome::LoadedDestination { address, bytes }) => {
            (write_loaded_copy_destination(address, &bytes), false)
        }
        Err(failure) => (failure.status, failure.operands_may_be_live),
    };
    if may_be_live {
        health.quarantine();
    }
    release_async_resources(&[], &retained, may_be_live);
    if status == SUCCESS {
        Ok(())
    } else {
        Err(status)
    }
}

/// Converts a pointer and size into a checked SVM address range.
pub(crate) fn svm_range(pointer: *mut c_void, size: usize) -> Result<(u64, u64), Status> {
    if pointer.is_null() || size == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let page_size = rocddi::memory::host::host_page_size().map_err(map_error)?;
    let page_size = usize::try_from(page_size).map_err(|_| INVALID_ARGUMENT)?;
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(ERROR);
    }
    let address = pointer as usize;
    let base = address & !(page_size - 1);
    let end = address
        .checked_add(size)
        .and_then(|end| end.checked_add(page_size - 1))
        .map(|end| end & !(page_size - 1))
        .ok_or(INVALID_ARGUMENT)?;
    let length = end.checked_sub(base).ok_or(INVALID_ARGUMENT)?;
    Ok((base as u64, length as u64))
}

/// Maps an HSA agent to its SVM location identifier.
pub(crate) fn svm_location_from_agent(
    runtime: &HsaSession,
    service: &MemoryAdvice,
    agent: HsaAgent,
    allow_null: bool,
) -> Result<SvmLocation, Status> {
    if agent.handle == 0 && allow_null {
        return Ok(SvmLocation::UnspecifiedOrMixed);
    }
    if runtime.agents.is_cpu_agent(agent) {
        return Ok(SvmLocation::System);
    }
    let index = runtime.agents.gpu_index(agent).ok_or(INVALID_AGENT)?;
    Ok(SvmLocation::Device(
        service
            .target(&runtime.agents.gpus[index].device)
            .map_err(map_error)?,
    ))
}

/// Maps a native SVM location identifier back to an HSA agent.
pub(crate) fn svm_agent_from_location(
    runtime: &HsaSession,
    location: SvmLocation,
) -> Result<HsaAgent, Status> {
    match location {
        SvmLocation::System => Ok(HsaAgent { handle: CPU_AGENT }),
        SvmLocation::UnspecifiedOrMixed => Ok(HsaAgent { handle: 0 }),
        SvmLocation::Device(identity) => runtime
            .agents
            .gpus
            .iter()
            .position(|gpu| gpu.endpoint.id == identity.endpoint_id())
            .map(|index| HsaAgent {
                handle: GPU_AGENT_BASE + index as u64,
            })
            .ok_or(ERROR),
    }
}

/// Selects the provider of an explicit GPU, or requires a unique service.
pub(crate) fn svm_service(
    runtime: &HsaSession,
    agent: Option<HsaAgent>,
) -> Result<MemoryAdvice, Status> {
    let session = runtime.native_session().ok_or(ERROR)?;
    if let Some(agent) = agent {
        if let Some(index) = runtime.agents.gpu_index(agent) {
            return session
                .memory_advice_for(&runtime.agents.gpus[index].device)
                .map_err(map_error);
        }
    }
    session.memory_advice().map_err(map_error)
}

/// Rolls back signal and allocation retention until the worker accepts them.
pub(crate) struct PendingAsyncRetention<'a> {
    /// Signal registry borrowed until copy ownership transfers.
    signals: &'a mut SignalState,
    /// Allocation registry borrowed until copy ownership transfers.
    pub(crate) memory: &'a mut MemoryState,
    /// Signals retained for the pending worker.
    pub(crate) handles: Arc<Vec<HsaSignal>>,
    /// Prefix successfully retained before a possible failure.
    retained: usize,
    /// Allocation bases borrowed until worker ownership is published.
    borrowed_memory: Vec<usize>,
}

impl<'a> PendingAsyncRetention<'a> {
    /// Starts a guard that rolls back retained copy operands unless transferred.
    pub(crate) fn new(
        signals: &'a mut SignalState,
        memory: &'a mut MemoryState,
        handles: Arc<Vec<HsaSignal>>,
    ) -> Result<Self, Status> {
        let mut pending = Self {
            signals,
            memory,
            handles,
            retained: 0,
            borrowed_memory: Vec::new(),
        };
        for signal in pending.handles.iter() {
            pending.signals.retain_async_signal(*signal)?;
            pending.retained += 1;
        }
        Ok(pending)
    }

    /// Retains memory while dependent work may use it.
    pub(crate) fn retain_memory(&mut self, base: usize) -> Result<(), Status> {
        self.borrowed_memory
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        self.memory.retain_copy_borrow(base)?;
        self.borrowed_memory.push(base);
        Ok(())
    }

    /// Transfers retained copy borrows to the asynchronous worker.
    pub(crate) fn transfer_to_worker(&mut self) {
        self.retained = 0;
        self.borrowed_memory.clear();
    }
}

impl MemoryState {
    /// Managed SVM and VMEM subranges keep their complete reservation alive.
    fn copy_borrow_key(&self, address: usize) -> usize {
        self.vmem_reservations
            .range(..=address)
            .next_back()
            .filter(|(_, reservation)| reservation.contains_range(address, 1))
            .map_or(address, |(&base, _)| base)
    }
    /// Retains copy borrow while dependent work may use it.
    pub(crate) fn retain_copy_borrow(&mut self, base: usize) -> Result<(), Status> {
        let base = self.copy_borrow_key(base);
        if !self.async_copy_borrows.contains_key(&base) {
            self.async_copy_borrows
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
        }
        let count = self.async_copy_borrows.entry(base).or_default();
        *count = count.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(())
    }

    /// Releases the retained copy borrow reference.
    pub(crate) fn release_copy_borrow(&mut self, base: usize) {
        let base = self.copy_borrow_key(base);
        if let Some(count) = self.async_copy_borrows.get_mut(&base) {
            *count -= 1;
            if *count == 0 {
                self.async_copy_borrows.remove(&base);
            }
        }
    }

    /// Copies borrowed after validating its source and destination.
    pub(crate) fn copy_borrowed(&self, base: usize) -> bool {
        self.async_copy_borrows
            .get(&self.copy_borrow_key(base))
            .is_some_and(|count| *count != 0)
    }
}

impl Drop for PendingAsyncRetention<'_> {
    /// Releases copy operands that were not transferred to a worker.
    fn drop(&mut self) {
        for signal in self.handles.iter().take(self.retained) {
            self.signals.release_async_signal(*signal);
        }
        for &base in &self.borrowed_memory {
            self.memory.release_copy_borrow(base);
        }
    }
}

/// Releases the retained async resources reference.
pub(crate) fn release_async_resources(
    handles: &[HsaSignal],
    borrowed_memory: &[Option<usize>],
    retain_memory: bool,
) {
    if let Ok(mut guard) = lock() {
        if let Some(runtime) = guard.as_mut() {
            for signal in handles {
                runtime.signal_state.release_async_signal(*signal);
            }
            if !retain_memory {
                for &base in borrowed_memory.iter().flatten() {
                    runtime.memory_state.release_copy_borrow(base);
                }
            }
        }
    }
    // Shutdown removes the runtime from the registry before joining workers;
    // it retains all signal storage until after this worker exits.
}

/// Publishes success or failure through the completion signal value.
fn complete_async_signal_value(signal: &crate::signal::AmdSignal, success: bool) {
    if success {
        signal.value.fetch_sub(1, Ordering::Release);
    } else {
        signal.value.store(-1, Ordering::Release);
    }
}

/// Completes async signal after its pending work has retired.
pub(crate) fn finish_async_signal(completion: HsaSignal, success: bool) {
    if completion.handle == 0 {
        return;
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            complete_async_signal_value(signal, success);
        })
    };
}

/// Completes async copy signal after its pending work has retired.
fn finish_async_copy_signal(
    completion: HsaSignal,
    success: bool,
    profile: Option<AsyncCopyProfile>,
) {
    if completion.handle == 0 {
        return;
    }
    let profile = if success { profile } else { None };
    let mut guard = lock().ok();
    if let Some(runtime) = guard.as_mut().and_then(|guard| guard.as_mut()) {
        runtime.signal_state.set_copy_profile(completion, profile);
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined. The
    // registry lock orders profile publication with the completion update.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            if let Some(profile) = profile {
                signal.start_ts.store(profile.start, Ordering::Relaxed);
                signal.end_ts.store(profile.end, Ordering::Relaxed);
                crate::signal::flush_signal_timestamps(signal);
            }
            complete_async_signal_value(signal, success);
        })
    };
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the worker owns the provider service and stop token for its full asynchronous lifetime"
)]
/// Runs SVM prefetch work after its dependency signals are satisfied.
pub(crate) fn svm_prefetch_worker(
    service: MemoryAdvice,
    stop: Arc<AtomicBool>,
    dependencies: &[HsaSignal],
    address: u64,
    size: u64,
    location: SvmLocation,
) -> bool {
    if !wait_async_dependencies(dependencies, &stop) {
        return false;
    }
    let result = service.prefetch(
        address,
        size,
        location,
        rocddi::memory::advice::PrefetchCompletion::Migration,
    );
    result.is_ok()
}

/// Waits for async dependencies to complete.
pub(crate) fn wait_async_dependencies(dependencies: &[HsaSignal], stop: &AtomicBool) -> bool {
    for &dependency in dependencies {
        loop {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            // SAFETY: Submission retains every dependency until the worker
            // exits; final shutdown joins workers before releasing storage.
            let Some(ready) = (unsafe {
                crate::signal::with_signal(dependency, |signal| {
                    signal.value.load(Ordering::Acquire) == 0
                })
            }) else {
                return false;
            };
            if ready {
                break;
            }
            thread::sleep(Duration::from_micros(20));
        }
    }
    true
}

/// Execution path selected for an asynchronous HSA copy request.
pub(crate) enum AsyncCopyRoute {
    /// Zero-byte copy that only completes dependencies and its signal.
    Noop,
    /// CPU copy performed after dependency signals complete.
    Host {
        /// Destination operand for this copy route.
        destination: HostCopyDestination,
        /// Source operand for this copy route.
        source: HostCopySource,
        /// Number of bytes in the host copy.
        size: usize,
    },
    /// One native SDMA operation on an activated GPU.
    Gpu {
        /// Activated device retained until the asynchronous route completes.
        device: Box<Device>,
        /// Index of the activated GPU used by this route.
        index: usize,
        /// Native GPU copy operation for this route.
        entry: GpuCopyEntry,
        /// Uses a private compute dispatch instead of the native SDMA sequence.
        compute: bool,
    },
    /// Several ordered SDMA operations sharing one native sequence.
    GpuBatch {
        /// Activated device retained until the asynchronous route completes.
        device: Box<Device>,
        /// Index of the activated GPU used by this route.
        index: usize,
        /// Ordered native copy operations sharing one batch.
        entries: Vec<GpuCopyEntry>,
        /// Wait and signal controls surrounding the copy.
        controls: BatchCopyControls,
    },
}

/// One operation within a native GPU copy route.
pub(crate) enum GpuCopyEntry {
    /// Rectangular copy with row and slice strides.
    Rect(CopyRect),
    /// Direct linear copy.
    Linear {
        /// Destination operand for this copy route.
        destination: u64,
        /// Source operand for this copy route.
        source: u64,
        /// Number of bytes copied or exchanged.
        size: u64,
    },
    /// Exchange between two GPU address ranges.
    Swap {
        /// First GPU address in the swap.
        address_a: u64,
        /// Second GPU address in the swap.
        address_b: u64,
        /// Number of bytes copied or exchanged.
        size: u64,
    },
    /// Copy whose addresses may come from GPU-visible pointer slots.
    Indirect {
        /// Destination operand for this copy route.
        destination: u64,
        /// Source operand for this copy route.
        source: u64,
        /// Number of bytes copied or exchanged.
        size: u64,
        /// Whether the source names a GPU-visible pointer slot.
        indirect_source: bool,
        /// Whether the destination names a GPU-visible pointer slot.
        indirect_destination: bool,
    },
    /// Copy from host storage staged for SDMA access.
    FromHost {
        /// Destination operand for this copy route.
        destination: u64,
        /// Source operand for this copy route.
        source: HostCopySource,
    },
    /// Copy to a host destination staged for SDMA access.
    ToHost {
        /// Destination operand for this copy route.
        destination: HostCopyDestination,
        /// Source operand for this copy route.
        source: u64,
        /// Number of bytes copied or exchanged.
        size: usize,
    },
}

/// Optional wait and signal attached to a native batch copy.
#[derive(Clone, Copy, Default)]
pub(crate) struct BatchCopyControls {
    /// GPU comparison that must pass before the batch begins.
    wait: Option<GpuCopyWait>,
    /// GPU atomic notification after the final packet.
    signal: Option<GpuCopySignal>,
}

/// Native route selected for one synchronous HSA copy.
pub(crate) enum SyncGpuCopyRoute {
    /// Two GPU address spaces connected through bounded host staging.
    Staged {
        /// Device that can read the source mapping.
        source_device: Box<Device>,
        /// Device that can write the destination mapping.
        destination_device: Box<Device>,
        /// First source GPU address.
        source: u64,
        /// First destination GPU address.
        destination: u64,
        /// Total transfer size.
        size: usize,
        /// Copy mechanism selected independently for each GPU's staging leg.
        compute: [bool; 2],
    },
    /// Direct copy between GPU-addressable ranges.
    Direct {
        /// Activated device retained through this synchronous copy.
        device: Box<Device>,
        /// Destination operand for this copy route.
        destination: u64,
        /// Source operand for this copy route.
        source: u64,
        /// Number of bytes in this synchronous copy.
        size: u64,
        /// Selects a private compute dispatch for this transfer.
        compute: bool,
    },
    /// Copy from a CPU-only source through staging.
    FromHost {
        /// Activated device retained through this synchronous copy.
        device: Box<Device>,
        /// Destination operand for this copy route.
        destination: u64,
        /// Source operand for this copy route.
        source: HostCopySource,
        /// Selects private compute staging for this transfer.
        compute: bool,
    },
    /// Copy into a CPU-only destination through staging.
    ToHost {
        /// Activated device retained through this synchronous copy.
        device: Box<Device>,
        /// Destination operand for this copy route.
        destination: HostCopyDestination,
        /// Source operand for this copy route.
        source: u64,
        /// Number of bytes in this synchronous copy.
        size: usize,
        /// Selects private compute staging for this transfer.
        compute: bool,
    },
}

/// CPU source storage and when its bytes must be captured.
pub(crate) enum HostCopySource {
    /// Direct host pointer and byte extent.
    Address(usize, usize),
    /// Loader-owned mapping resolved after dependencies finish.
    LoadedAddress(usize, usize),
    /// Owned snapshot of a loader mapping for immediate copy.
    LoadedBytes(Vec<u8>),
}

/// Point at which a loader-owned source is read.
#[derive(Clone, Copy)]
enum HostSourceTiming {
    /// Snapshot before releasing the registry lock.
    Immediate,
    /// Resolve the live loader mapping after dependency completion.
    AfterDependencies,
}

/// CPU destination, including loader mappings needing a later lookup.
pub(crate) enum HostCopyDestination {
    /// Direct host address.
    Address(usize),
    /// Loader-owned address resolved while the registry lock is held.
    LoadedAddress(usize),
}

/// Completion data that must be published after a synchronous GPU copy.
pub(crate) enum SyncGpuCopyOutcome {
    /// Native and host copies completed without deferred publication.
    Done,
    /// Staged bytes to write to a loader-owned mapping.
    LoadedDestination {
        /// Address of the loader-owned destination mapping.
        address: usize,
        /// Bytes staged until the native copy completes.
        bytes: Vec<u8>,
    },
}

/// Frontend status and whether native work may still read its operands.
pub(crate) struct CopyRouteFailure {
    /// HSA status translated from validation or native failure.
    pub(crate) status: Status,
    /// Requires retaining borrowed memory until retirement is proved.
    pub(crate) operands_may_be_live: bool,
}

impl CopyRouteFailure {
    /// Constructs a failure whose native operands are already retired.
    pub(super) fn retired(status: Status) -> Self {
        Self {
            status,
            operands_may_be_live: false,
        }
    }
}

impl From<rocddi::device::gpu::CopyFailure> for CopyRouteFailure {
    /// Preserves native liveness uncertainty while translating the status.
    fn from(failure: rocddi::device::gpu::CopyFailure) -> Self {
        Self {
            status: map_error(failure.error),
            operands_may_be_live: failure.operands_may_be_live,
        }
    }
}

/// Synchronous route and public allocations borrowed by its operands.
pub(crate) struct ResolvedSyncGpuCopy {
    /// Native copy plan selected after pointer validation.
    pub(crate) route: SyncGpuCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    pub(crate) borrowed_memory: [Option<usize>; 2],
}

/// Asynchronous route and public allocations borrowed by its operands.
pub(crate) struct ResolvedAsyncCopy {
    /// Native or host copy plan selected after pointer validation.
    pub(crate) route: AsyncCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    pub(crate) borrowed_memory: [Option<usize>; 2],
}

/// Copies agent after validating its source and destination.
fn copy_agent(
    runtime: &HsaSession,
    requested: HsaAgent,
    description: Option<&PointerDescription>,
) -> Result<HsaAgent, Status> {
    let agent = if requested.handle == 0 {
        description.map_or(HsaAgent { handle: CPU_AGENT }, |description| {
            if description.pointer_type == POINTER_TYPE_SVM && description.owner.handle == 0 {
                HsaAgent { handle: CPU_AGENT }
            } else {
                description.owner
            }
        })
    } else {
        requested
    };
    runtime
        .agents
        .is_agent(agent)
        .then_some(agent)
        .ok_or(INVALID_AGENT)
}

/// Selects copy GPU from the available runtime resources.
pub(crate) fn selected_copy_gpu(
    runtime: &HsaSession,
    description: &PointerDescription,
) -> Option<(usize, HsaAgent)> {
    std::iter::once(description.owner)
        .chain(description.accessible.iter().copied())
        .filter(|agent| description.accessible.contains(agent))
        .find_map(|agent| runtime.agents.gpu_index(agent).map(|index| (index, agent)))
}

/// Copies pointer description after validating its source and destination.
pub(crate) fn copy_pointer_description(
    runtime: &HsaSession,
    pointer: usize,
    size: usize,
) -> Result<Option<PointerDescription>, Status> {
    if pointer.checked_add(size).is_none() {
        return Err(INVALID_ARGUMENT);
    }
    let description = pointer_description_for_range(runtime, pointer, size);
    if let Some(description) = &description {
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return Err(INVALID_ALLOCATION);
        }
        if description.offset_range(pointer, size).is_none() {
            return Err(INVALID_ALLOCATION);
        }
    }
    Ok(description)
}

/// Resolves a GPU address to its allocation and accessible agent.
pub(crate) fn gpu_copy_address(
    runtime: &HsaSession,
    description: Option<&PointerDescription>,
    pointer: usize,
    size: usize,
    gpu: HsaAgent,
    required_access: u32,
) -> Result<(u64, usize), Status> {
    // Repeated locks can share a numeric alias but carry different GPU grants.
    // Select one registration that covers this operation; never merge metadata.
    let locked_description = description
        .filter(|description| description.pointer_type == POINTER_TYPE_LOCKED)
        .and_then(|_| {
            runtime
                .memory_state
                .locked_gpu_description(pointer, size, gpu)
        });
    let description = locked_description.as_ref().or(description);
    let description = description.ok_or(INVALID_ALLOCATION)?;
    if !description.accessible.contains(&gpu) {
        return Err(INVALID_AGENT);
    }
    vmem_copy_access(runtime, description, gpu, required_access)?;
    let offset = description
        .offset_range(pointer, size)
        .ok_or(INVALID_ALLOCATION)?;
    let address = description
        .agent_base
        .checked_add(offset)
        .ok_or(INVALID_ALLOCATION)?;
    Ok((
        u64::try_from(address).map_err(|_| INVALID_ALLOCATION)?,
        description.agent_base,
    ))
}

/// Resolves a caller-owned image backing range to this agent's GPU address.
/// Image data lifetime remains the caller's responsibility under HSA.
pub(crate) fn image_device_address(
    runtime: &HsaSession,
    pointer: usize,
    size: usize,
    agent: HsaAgent,
    permission: u32,
) -> Result<u64, Status> {
    image_copy_borrow(runtime, pointer, size, agent, permission).map(|(address, _)| address)
}

/// Describes the usable pool block without changing public requested-size metadata.
fn image_pool_description(
    description: &PointerDescription,
    allocation: rocddi::memory::device::AllocationInfo,
    pointer: usize,
) -> Option<PointerDescription> {
    // HSA rounds pool requests before allocating through rocddi. `size` is
    // that usable extent; `native_size` may also include provider padding.
    let size = usize::try_from(allocation.size).ok()?;
    if !contains(description.agent_base, size, pointer)
        && !description
            .host_base
            .is_some_and(|host| contains(host, size, pointer))
    {
        return None;
    }
    let mut native = description.clone();
    native.size = size;
    Some(native)
}

/// Resolves image backing and the owner to retain through native retirement.
pub(crate) fn image_copy_borrow(
    runtime: &HsaSession,
    pointer: usize,
    size: usize,
    agent: HsaAgent,
    permission: u32,
) -> Result<(u64, usize), Status> {
    pointer.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    // Image storage includes native row/slice padding. Direct pool allocations
    // guarantee their rounded usable extent even though pointer-info reports
    // the original request. Other allocation kinds retain their exact bounds.
    let description = match runtime
        .memory_state
        .allocations
        .values()
        .find_map(|memory| {
            image_pool_description(&memory.description, memory.allocation.info(), pointer)
        }) {
        Some(description) => Some(description),
        None => copy_pointer_description(runtime, pointer, size)?,
    };
    gpu_copy_address(
        runtime,
        description.as_ref(),
        pointer,
        size,
        agent,
        permission,
    )
    .map(|(address, base)| (address, runtime.memory_state.copy_borrow_key(base)))
}

/// Prepares host copy source before publication or execution.
fn prepare_host_copy_source(
    runtime: &HsaSession,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
    timing: HostSourceTiming,
) -> Result<HostCopySource, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
    {
        if matches!(timing, HostSourceTiming::AfterDependencies) {
            return Ok(HostCopySource::LoadedAddress(pointer, size));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| OUT_OF_RESOURCES)?;
        bytes.resize(size, 0);
        // SAFETY: The registry lock retains this loaded mapping while the
        // complete range is copied into owned host storage.
        unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
        Ok(HostCopySource::LoadedBytes(bytes))
    } else {
        Ok(HostCopySource::Address(host, size))
    }
}

/// Prepares host copy destination before publication or execution.
fn prepare_host_copy_destination(
    runtime: &HsaSession,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
) -> Result<HostCopyDestination, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
    {
        Ok(HostCopyDestination::LoadedAddress(pointer))
    } else {
        Ok(HostCopyDestination::Address(host))
    }
}

/// Checks virtual memory permissions for a copy by the given agent.
pub(crate) fn vmem_copy_access(
    runtime: &HsaSession,
    description: &PointerDescription,
    agent: HsaAgent,
    required_access: u32,
) -> Result<(), Status> {
    if description.pointer_type != POINTER_TYPE_HSA_VMEM {
        return Ok(());
    }
    let mapping = runtime
        .memory_state
        .vmem_mappings
        .get(&description.agent_base)
        .ok_or(INVALID_ALLOCATION)?;
    if mapping
        .access
        .get(&agent.handle)
        .is_none_or(|access| access.permissions & required_access != required_access)
    {
        return Err(INVALID_ALLOCATION);
    }
    Ok(())
}

/// Frontend policy for the ordinary copy, forced SDMA, and batch contracts.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CopyPreference {
    /// Same-agent copies use compute; other routes prefer available SDMA.
    Default,
    /// The engine API requests SDMA, subject to environment overrides.
    PreferSdma,
    /// Rectangles and batch packet operations require native SDMA.
    RequireSdma,
}

/// Applies SDMA overrides before the same-agent compute default.
#[allow(
    clippy::fn_params_excessive_bools,
    reason = "policy overrides and independent native topology facts are exercised as a routing truth table"
)]
fn select_compute(
    sdma_enabled: bool,
    peer_sdma_enabled: bool,
    peer: bool,
    same_gpu: bool,
    native_sdma: bool,
    preference: CopyPreference,
) -> Result<bool, Status> {
    let sdma = sdma_enabled && (!peer || peer_sdma_enabled) && native_sdma;
    if preference == CopyPreference::RequireSdma {
        return if sdma { Ok(false) } else { Err(NOT_SUPPORTED) };
    }
    Ok(!sdma || (same_gpu && preference == CopyPreference::Default))
}

/// Selects a mechanism from immutable generation policy and native capability.
fn copy_uses_compute(
    runtime: &HsaSession,
    index: usize,
    source: HsaAgent,
    destination: HsaAgent,
    preference: CopyPreference,
) -> Result<bool, Status> {
    let peer = source != destination
        && runtime.agents.gpu_index(source).is_some()
        && runtime.agents.gpu_index(destination).is_some();
    let same_gpu = source == destination && runtime.agents.gpu_index(source).is_some();
    let device = &runtime.agents.gpus[index].device;
    let native_sdma = device.gpu().is_ok_and(|gpu| gpu.supports_linear_copy());
    let compute = select_compute(
        runtime.config.sdma_enabled,
        runtime.config.peer_sdma_enabled,
        peer,
        same_gpu,
        native_sdma,
        preference,
    )?;
    if compute && !compute_copy::supported(device) {
        return Err(NOT_SUPPORTED);
    }
    Ok(compute)
}

/// Resolves async copy to a validated runtime resource.
#[allow(
    clippy::too_many_arguments,
    reason = "preserves the HSA source/destination/agent tuple with explicit route selection"
)]
fn resolve_async_copy(
    runtime: &HsaSession,
    destination: usize,
    dst_agent: HsaAgent,
    source: usize,
    src_agent: HsaAgent,
    size: usize,
    copy_gpu_index: Option<usize>,
    preference: CopyPreference,
) -> Result<ResolvedAsyncCopy, Status> {
    if size == 0 {
        if (dst_agent.handle != 0 && !runtime.agents.is_agent(dst_agent))
            || (src_agent.handle != 0 && !runtime.agents.is_agent(src_agent))
        {
            return Err(INVALID_AGENT);
        }
        return Ok(ResolvedAsyncCopy {
            route: AsyncCopyRoute::Noop,
            borrowed_memory: [None, None],
        });
    }
    let dst_description = copy_pointer_description(runtime, destination, size)?;
    let src_description = copy_pointer_description(runtime, source, size)?;
    let dst_agent = copy_agent(runtime, dst_agent, dst_description.as_ref())?;
    let src_agent = copy_agent(runtime, src_agent, src_description.as_ref())?;
    let dst_gpu = runtime.agents.gpu_index(dst_agent);
    let src_gpu = runtime.agents.gpu_index(src_agent);
    if let Some(index) = copy_gpu_index.or(src_gpu.or(dst_gpu)) {
        let compute = copy_uses_compute(runtime, index, src_agent, dst_agent, preference)?;
        let gpu_agent = HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        };
        let gpu_destination = gpu_copy_address(
            runtime,
            dst_description.as_ref(),
            destination,
            size,
            gpu_agent,
            ACCESS_PERMISSION_WO,
        );
        let gpu_source = gpu_copy_address(
            runtime,
            src_description.as_ref(),
            source,
            size,
            gpu_agent,
            ACCESS_PERMISSION_RO,
        );
        let device = Box::new(runtime.agents.gpus[index].device.clone());
        return match (gpu_destination, gpu_source) {
            (Ok((destination, dst_base)), Ok((source, src_base))) => Ok(ResolvedAsyncCopy {
                route: AsyncCopyRoute::Gpu {
                    device,
                    index,
                    compute,
                    entry: GpuCopyEntry::Linear {
                        destination,
                        source,
                        size: u64::try_from(size).map_err(|_| INVALID_ARGUMENT)?,
                    },
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            }),
            (Ok((destination, dst_base)), Err(_)) if src_gpu.is_none() => {
                let host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
                let source = prepare_host_copy_source(
                    runtime,
                    src_description.as_ref(),
                    source,
                    host,
                    size,
                    HostSourceTiming::AfterDependencies,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        compute,
                        entry: GpuCopyEntry::FromHost {
                            destination,
                            source,
                        },
                    },
                    borrowed_memory: [
                        Some(dst_base),
                        src_description.map(|description| description.agent_base),
                    ],
                })
            }
            (Err(_), Ok((source, src_base))) if dst_gpu.is_none() => {
                let host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
                let destination = prepare_host_copy_destination(
                    runtime,
                    dst_description.as_ref(),
                    destination,
                    host,
                    size,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        compute,
                        entry: GpuCopyEntry::ToHost {
                            destination,
                            source,
                            size,
                        },
                    },
                    borrowed_memory: [
                        dst_description.map(|description| description.agent_base),
                        Some(src_base),
                    ],
                })
            }
            (Err(status), _) | (_, Err(status)) => Err(status),
        };
    }
    let destination_host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
    let source_host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
    let destination = prepare_host_copy_destination(
        runtime,
        dst_description.as_ref(),
        destination,
        destination_host,
        size,
    )?;
    let source = prepare_host_copy_source(
        runtime,
        src_description.as_ref(),
        source,
        source_host,
        size,
        HostSourceTiming::AfterDependencies,
    )?;
    Ok(ResolvedAsyncCopy {
        route: AsyncCopyRoute::Host {
            destination,
            source,
            size,
        },
        borrowed_memory: [
            dst_description.map(|description| description.agent_base),
            src_description.map(|description| description.agent_base),
        ],
    })
}

/// Resolves sync GPU copy to a validated runtime resource.
pub(crate) fn resolve_sync_gpu_copy(
    runtime: &HsaSession,
    destination: usize,
    source: usize,
    size: usize,
    host_destination: Result<usize, Status>,
    host_source: Result<usize, Status>,
) -> Result<ResolvedSyncGpuCopy, Status> {
    match (host_destination, host_source) {
        (Err(_), Ok(host_source)) => {
            let description =
                copy_pointer_description(runtime, destination, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (destination, destination_base) = gpu_copy_address(
                runtime,
                Some(&description),
                destination,
                size,
                agent,
                ACCESS_PERMISSION_WO,
            )?;
            let source_description = copy_pointer_description(runtime, source, size)?;
            let source_base = source_description
                .as_ref()
                .map(|description| description.agent_base);
            let source = prepare_host_copy_source(
                runtime,
                source_description.as_ref(),
                source,
                host_source,
                size,
                HostSourceTiming::Immediate,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::FromHost {
                    device: Box::new(runtime.agents.gpus[index].device.clone()),
                    compute: copy_uses_compute(
                        runtime,
                        index,
                        HsaAgent { handle: CPU_AGENT },
                        agent,
                        CopyPreference::Default,
                    )?,
                    destination,
                    source,
                },
                borrowed_memory: [Some(destination_base), source_base],
            })
        }
        (Ok(host_destination), Err(_)) => {
            let description =
                copy_pointer_description(runtime, source, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (source, source_base) = gpu_copy_address(
                runtime,
                Some(&description),
                source,
                size,
                agent,
                ACCESS_PERMISSION_RO,
            )?;
            let destination_description = copy_pointer_description(runtime, destination, size)?;
            let destination_base = destination_description
                .as_ref()
                .map(|description| description.agent_base);
            let destination = prepare_host_copy_destination(
                runtime,
                destination_description.as_ref(),
                destination,
                host_destination,
                size,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::ToHost {
                    device: Box::new(runtime.agents.gpus[index].device.clone()),
                    compute: copy_uses_compute(
                        runtime,
                        index,
                        agent,
                        HsaAgent { handle: CPU_AGENT },
                        CopyPreference::Default,
                    )?,
                    destination,
                    source,
                    size,
                },
                borrowed_memory: [destination_base, Some(source_base)],
            })
        }
        (Err(_), Err(_)) => {
            let resolved = resolve_async_copy(
                runtime,
                destination,
                HsaAgent { handle: 0 },
                source,
                HsaAgent { handle: 0 },
                size,
                None,
                CopyPreference::Default,
            );
            let Ok(resolved) = resolved else {
                let source_description =
                    copy_pointer_description(runtime, source, size)?.ok_or(INVALID_ALLOCATION)?;
                let destination_description = copy_pointer_description(runtime, destination, size)?
                    .ok_or(INVALID_ALLOCATION)?;
                let (source_index, source_agent) =
                    selected_copy_gpu(runtime, &source_description).ok_or(INVALID_AGENT)?;
                let (destination_index, destination_agent) =
                    selected_copy_gpu(runtime, &destination_description).ok_or(INVALID_AGENT)?;
                let (source, source_base) = gpu_copy_address(
                    runtime,
                    Some(&source_description),
                    source,
                    size,
                    source_agent,
                    ACCESS_PERMISSION_RO,
                )?;
                let (destination, destination_base) = gpu_copy_address(
                    runtime,
                    Some(&destination_description),
                    destination,
                    size,
                    destination_agent,
                    ACCESS_PERMISSION_WO,
                )?;
                return Ok(ResolvedSyncGpuCopy {
                    route: SyncGpuCopyRoute::Staged {
                        compute: [
                            copy_uses_compute(
                                runtime,
                                source_index,
                                source_agent,
                                HsaAgent { handle: CPU_AGENT },
                                CopyPreference::Default,
                            )?,
                            copy_uses_compute(
                                runtime,
                                destination_index,
                                HsaAgent { handle: CPU_AGENT },
                                destination_agent,
                                CopyPreference::Default,
                            )?,
                        ],
                        source_device: Box::new(runtime.agents.gpus[source_index].device.clone()),
                        destination_device: Box::new(
                            runtime.agents.gpus[destination_index].device.clone(),
                        ),
                        source,
                        destination,
                        size,
                    },
                    borrowed_memory: [Some(destination_base), Some(source_base)],
                });
            };
            let AsyncCopyRoute::Gpu {
                device,
                compute,
                entry:
                    GpuCopyEntry::Linear {
                        destination,
                        source,
                        size,
                    },
                ..
            } = resolved.route
            else {
                return Err(INVALID_ALLOCATION);
            };
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::Direct {
                    device,
                    compute,
                    destination,
                    source,
                    size,
                },
                borrowed_memory: resolved.borrowed_memory,
            })
        }
        (Ok(_), Ok(_)) => Err(ERROR),
    }
}

/// Copies GPU after validating its source and destination.
fn copy_gpu(device: &Device) -> Result<rocddi::device::gpu::GpuDevice<'_>, CopyRouteFailure> {
    device
        .gpu()
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))
}

/// Copies bytes from a loader-owned source into stable staging memory.
fn loaded_source_snapshot(address: usize, size: usize) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| OUT_OF_RESOURCES)?;
    bytes.resize(size, 0);
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    let host = crate::loader::loaded_host_range(&runtime.loader, address, size)?
        .ok_or(INVALID_ALLOCATION)?;
    // SAFETY: The registry lock retains the complete loaded mapping while the
    // source is snapshotted after the dependency signals have been satisfied.
    unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
    Ok(bytes)
}

/// Runs a closure with a validated host source address.
fn with_host_copy_source<T>(
    source: &HostCopySource,
    copy: impl FnOnce(&[u8]) -> Result<T, CopyRouteFailure>,
) -> Result<T, CopyRouteFailure> {
    let snapshot = if let HostCopySource::LoadedAddress(address, size) = source {
        loaded_source_snapshot(*address, *size).map_err(CopyRouteFailure::retired)?
    } else {
        Vec::new()
    };
    let host = match source {
        HostCopySource::Address(address, size) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            unsafe { std::slice::from_raw_parts(*address as *const u8, *size) }
        }
        HostCopySource::LoadedAddress(_, _) => snapshot.as_slice(),
        HostCopySource::LoadedBytes(bytes) => bytes,
    };
    copy(host)
}

/// Copies from host route after validating its source and destination.
fn copy_from_host_route(
    device: &Device,
    destination: u64,
    source: &HostCopySource,
    stop: &AtomicBool,
) -> Result<(), CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    with_host_copy_source(source, |host| {
        // SAFETY: HSA checked the GPU destination and retains its backing. rocddi
        // copies host bytes into owned staging before native submission.
        unsafe { gpu.copy_from_host(destination, host, stop) }.map_err(Into::into)
    })
}

/// Copies to host destination after validating its source and destination.
fn copy_to_host_destination(
    destination: &HostCopyDestination,
    size: usize,
    copy: impl FnOnce(&mut [u8]) -> Result<(), CopyRouteFailure>,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match destination {
        HostCopyDestination::Address(address) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            let host = unsafe { std::slice::from_raw_parts_mut(*address as *mut u8, size) };
            copy(host)?;
            fence(Ordering::SeqCst);
            Ok(SyncGpuCopyOutcome::Done)
        }
        HostCopyDestination::LoadedAddress(address) => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(size)
                .map_err(|_| CopyRouteFailure::retired(OUT_OF_RESOURCES))?;
            bytes.resize(size, 0);
            copy(&mut bytes)?;
            Ok(SyncGpuCopyOutcome::LoadedDestination {
                address: *address,
                bytes,
            })
        }
    }
}

/// Copies to host route after validating its source and destination.
fn copy_to_host_route(
    device: &Device,
    destination: &HostCopyDestination,
    source: u64,
    size: usize,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    copy_to_host_destination(destination, size, |host| {
        // SAFETY: HSA checked the GPU source and retains its backing. rocddi
        // writes the host buffer only after native retirement.
        unsafe { gpu.copy_to_host(host, source, stop) }.map_err(Into::into)
    })
}

/// Executes a synchronous copy on the selected native route.
pub(crate) fn execute_sync_gpu_copy(
    route: SyncGpuCopyRoute,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match route {
        SyncGpuCopyRoute::Staged {
            source_device,
            destination_device,
            source,
            destination,
            size,
            compute,
        } => {
            let source_gpu = copy_gpu(&source_device)?;
            let destination_gpu = copy_gpu(&destination_device)?;
            staged_copy(size, |offset, bytes| {
                // SAFETY: The frontend retained both checked GPU ranges. The
                // DDI writes bytes only after retiring its native staging, and
                // copies their contents into its own retained upload staging.
                if compute[0] {
                    compute_copy::copy_to_host(&source_device, bytes, source + offset as u64)?;
                } else {
                    unsafe { source_gpu.copy_to_host(bytes, source + offset as u64, stop) }
                        .map_err(CopyRouteFailure::from)?;
                }
                if compute[1] {
                    compute_copy::copy_from_host(
                        &destination_device,
                        destination + offset as u64,
                        bytes,
                    )
                } else {
                    unsafe {
                        destination_gpu.copy_from_host(destination + offset as u64, bytes, stop)
                    }
                    .map_err(CopyRouteFailure::from)
                }
            })?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::Direct {
            device,
            destination,
            source,
            size,
            compute,
        } => {
            if compute {
                compute_copy::copy(&device, destination, source, size)?;
                return Ok(SyncGpuCopyOutcome::Done);
            }
            let gpu = copy_gpu(&device)?;
            // SAFETY: HSA checked both mapped GPU ranges before retaining
            // their backing through this synchronous call.
            unsafe { gpu.copy_linear(destination, source, size, stop) }
                .map_err(CopyRouteFailure::from)?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::FromHost {
            device,
            destination,
            source,
            compute,
        } => {
            if compute {
                with_host_copy_source(&source, |host| {
                    compute_copy::copy_from_host(&device, destination, host)
                })?;
            } else {
                copy_from_host_route(&device, destination, &source, stop)?;
            }
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::ToHost {
            device,
            destination,
            source,
            size,
            compute,
        } => {
            if compute {
                copy_to_host_destination(&destination, size, |host| {
                    compute_copy::copy_to_host(&device, host, source)
                })
            } else {
                copy_to_host_route(&device, &destination, source, size, stop)
            }
        }
    }
}

/// Copies one bounded chunk at a time, retiring its read before its write.
fn staged_copy(
    size: usize,
    mut transfer: impl FnMut(usize, &mut [u8]) -> Result<(), CopyRouteFailure>,
) -> Result<(), CopyRouteFailure> {
    const CHUNK_BYTES: usize = 4 * 1024 * 1024;
    let mut bytes = Vec::new();
    let capacity = size.min(CHUNK_BYTES);
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| CopyRouteFailure::retired(OUT_OF_RESOURCES))?;
    bytes.resize(capacity, 0);
    let mut offset = 0;
    while offset < size {
        let length = (size - offset).min(capacity);
        transfer(offset, &mut bytes[..length])?;
        offset += length;
    }
    Ok(())
}

/// Writes loaded copy destination to the validated destination.
pub(crate) fn write_loaded_copy_destination(address: usize, bytes: &[u8]) -> Status {
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    let host = match crate::loader::loaded_host_range(&runtime.loader, address, bytes.len()) {
        Ok(Some(host)) => host,
        Ok(None) | Err(_) => return INVALID_ALLOCATION,
    };
    // SAFETY: The registry lock retains the complete loaded mapping until
    // the copied bytes and their system-scope ordering are published.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len()) };
    fence(Ordering::SeqCst);
    SUCCESS
}

/// Submits a sequence of native copy entries to the GPU.
fn execute_gpu_entries(
    device: &Device,
    entries: impl ExactSizeIterator<Item = GpuCopyEntry>,
    controls: BatchCopyControls,
    sdma_engine: Option<u32>,
    timed: bool,
    stop: &AtomicBool,
) -> Result<Option<GpuCopyTimestamps>, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    let mut sequence = match sdma_engine {
        Some(engine) => GpuCopySequence::begin_on_sdma_engine(gpu, stop, engine),
        None => GpuCopySequence::begin(gpu, stop),
    }
    .map_err(CopyRouteFailure::from)?;
    if timed {
        sequence.enable_timing().map_err(CopyRouteFailure::from)?;
    }
    let count = entries.len();
    for (ordinal, entry) in entries.enumerate() {
        let options = GpuCopyOptions {
            wait: if ordinal == 0 { controls.wait } else { None },
            signal: if ordinal + 1 == count {
                controls.signal
            } else {
                None
            },
            ..GpuCopyOptions::default()
        };
        let outcome = match entry {
            GpuCopyEntry::Rect(rect) => {
                // SAFETY: Preparation validated every selected row and keeps
                // its mapped backing live until native retirement.
                unsafe { copy_rect_with_options(&mut sequence, rect, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Linear {
                destination,
                source,
                size,
            } => {
                // SAFETY: Copy preparation checked both mapped GPU ranges and
                // retains their owners until native retirement is proved.
                unsafe { sequence.copy_linear_with_options(destination, source, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Swap {
                address_a,
                address_b,
                size,
            } => {
                // SAFETY: Preparation verified read and write access to both
                // ranges on the selected GPU and retained their owners.
                unsafe { sequence.swap_linear_with_options(address_a, address_b, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Indirect {
                destination,
                source,
                size,
                indirect_source,
                indirect_destination,
            } => {
                // SAFETY: Preparation checked access to every direct range and
                // indirect pointer slot. The caller owns the resolved targets
                // through completion, including after an uncertain submission.
                unsafe {
                    sequence.copy_linear_with_options(
                        destination,
                        source,
                        size,
                        GpuCopyOptions {
                            indirect_source,
                            indirect_destination,
                            ..options
                        },
                    )
                }
                .map(|()| SyncGpuCopyOutcome::Done)
                .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::FromHost {
                destination,
                source,
            } => with_host_copy_source(&source, |host| {
                // SAFETY: HSA checked the GPU destination and retains its
                // backing. rocddi owns the staged copy of these host bytes.
                unsafe { sequence.copy_from_host(destination, host) }
                    .map_err(CopyRouteFailure::from)
            })
            .map(|()| SyncGpuCopyOutcome::Done),
            GpuCopyEntry::ToHost {
                destination,
                source,
                size,
            } => copy_to_host_destination(&destination, size, |host| {
                // SAFETY: HSA checked the GPU source and retains its backing.
                // rocddi writes the host slice after native retirement.
                unsafe { sequence.copy_to_host(host, source) }.map_err(CopyRouteFailure::from)
            }),
        }?;
        if let SyncGpuCopyOutcome::LoadedDestination { address, bytes } = outcome {
            let status = write_loaded_copy_destination(address, &bytes);
            if status != SUCCESS {
                return Err(CopyRouteFailure::retired(status));
            }
        }
    }
    if timed {
        sequence
            .finish_timing()
            .map(Some)
            .map_err(CopyRouteFailure::from)
    } else {
        Ok(None)
    }
}

/// Lowers optional box controls onto its first and last row only.
///
/// # Safety
/// Every row must be accessible to the sequence's GPU through retirement.
unsafe fn copy_rect_with_options(
    sequence: &mut GpuCopySequence<'_, '_>,
    rect: CopyRect,
    options: GpuCopyOptions,
) -> Result<(), rocddi::device::gpu::CopyFailure> {
    if options.wait.is_none() && options.signal.is_none() {
        // SAFETY: The caller retained every validated row.
        return unsafe { sequence.copy_rect(rect) };
    }
    for layer in 0..rect.depth {
        for row in 0..rect.height {
            let first = layer == 0 && row == 0;
            let last = layer + 1 == rect.depth && row + 1 == rect.height;
            // SAFETY: Preparation checked each complete rectangle extent. The
            // linear row operations retain identical ordering and ownership.
            unsafe {
                sequence.copy_linear_with_options(
                    rect.destination
                        + u64::from(layer) * rect.destination_slice
                        + u64::from(row) * rect.destination_pitch,
                    rect.source
                        + u64::from(layer) * rect.source_slice
                        + u64::from(row) * rect.source_pitch,
                    rect.width,
                    GpuCopyOptions {
                        wait: if first { options.wait } else { None },
                        signal: if last { options.signal } else { None },
                        ..GpuCopyOptions::default()
                    },
                )
            }?;
        }
    }
    Ok(())
}

/// Reads the host profiling clock without requiring a GPU or native correlation.
fn system_copy_tick() -> Result<u64, CopyRouteFailure> {
    rocddi::host::clock::read().map_err(|error| CopyRouteFailure::retired(map_error(error)))
}

/// Builds an asynchronous copy profile from GPU copy timestamps.
fn gpu_copy_profile(index: usize, ticks: Option<GpuCopyTimestamps>) -> Option<AsyncCopyProfile> {
    ticks.map(|ticks| AsyncCopyProfile {
        clock: AsyncCopyClock::Gpu(index),
        start: ticks.start,
        end: ticks.end,
    })
}

/// Executes a validated host, GPU, or staging copy route.
fn execute_compute_entry(
    device: &Device,
    entry: GpuCopyEntry,
    profile_enabled: bool,
) -> Result<Option<AsyncCopyProfile>, CopyRouteFailure> {
    let start = profile_enabled.then(system_copy_tick).transpose()?;
    let outcome = match entry {
        GpuCopyEntry::Linear {
            destination,
            source,
            size,
        } => {
            compute_copy::copy(device, destination, source, size)?;
            SyncGpuCopyOutcome::Done
        }
        GpuCopyEntry::FromHost {
            destination,
            source,
        } => {
            with_host_copy_source(&source, |host| {
                compute_copy::copy_from_host(device, destination, host)
            })?;
            SyncGpuCopyOutcome::Done
        }
        GpuCopyEntry::ToHost {
            destination,
            source,
            size,
        } => copy_to_host_destination(&destination, size, |host| {
            compute_copy::copy_to_host(device, host, source)
        })?,
        _ => return Err(CopyRouteFailure::retired(NOT_SUPPORTED)),
    };
    if let SyncGpuCopyOutcome::LoadedDestination { address, bytes } = outcome {
        let status = write_loaded_copy_destination(address, &bytes);
        if status != SUCCESS {
            return Err(CopyRouteFailure::retired(status));
        }
    }
    let end = profile_enabled.then(system_copy_tick).transpose()?;
    Ok(start.zip(end).map(|(start, end)| AsyncCopyProfile {
        clock: AsyncCopyClock::System,
        start,
        end,
    }))
}

/// Executes a validated host, GPU, or staging copy route.
fn execute_copy_route(
    route: AsyncCopyRoute,
    sdma_engine: Option<u32>,
    profile_enabled: bool,
    stop: &AtomicBool,
) -> Result<Option<AsyncCopyProfile>, CopyRouteFailure> {
    match route {
        AsyncCopyRoute::Noop => Ok(None),
        AsyncCopyRoute::Host {
            destination,
            source,
            size,
        } => {
            if stop.load(Ordering::Acquire) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            let start = profile_enabled.then(system_copy_tick).transpose()?;
            let needs_loaded_mapping = matches!(&source, HostCopySource::LoadedAddress(..))
                || matches!(&destination, HostCopyDestination::LoadedAddress(..));
            let guard = if needs_loaded_mapping {
                match lock() {
                    Ok(guard) => Some(guard),
                    Err(_) => return Err(CopyRouteFailure::retired(ERROR)),
                }
            } else {
                None
            };
            let runtime = guard.as_ref().and_then(|guard| guard.as_ref());
            let loaded_host = |address| {
                runtime.and_then(|runtime| {
                    crate::loader::loaded_host_range(&runtime.loader, address, size)
                        .ok()
                        .flatten()
                })
            };
            let source_host = match &source {
                HostCopySource::Address(address, _) => *address,
                HostCopySource::LoadedAddress(address, _) => {
                    let Some(host) = loaded_host(*address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
                HostCopySource::LoadedBytes(bytes) => bytes.as_ptr() as usize,
            };
            let destination_host = match destination {
                HostCopyDestination::Address(address) => address,
                HostCopyDestination::LoadedAddress(address) => {
                    let Some(host) = loaded_host(address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
            };
            // SAFETY: The caller keeps ordinary host ranges live through
            // completion. HsaSession-owned memory is borrowed, and the registry
            // lock keeps loaded mappings live for this copy. ptr::copy also
            // permits overlapping ranges.
            unsafe { std::ptr::copy(source_host as *const u8, destination_host as *mut u8, size) };
            fence(Ordering::SeqCst);
            drop(guard);
            let end = profile_enabled.then(system_copy_tick).transpose()?;
            let profile = start.zip(end).map(|(start, end)| AsyncCopyProfile {
                clock: AsyncCopyClock::System,
                start,
                end,
            });
            if profile.is_some_and(|profile| profile.end < profile.start) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            Ok(profile)
        }
        AsyncCopyRoute::Gpu {
            device,
            index,
            entry,
            compute,
        } => {
            if compute {
                execute_compute_entry(&device, entry, profile_enabled)
            } else {
                execute_gpu_entries(
                    &device,
                    std::iter::once(entry),
                    BatchCopyControls::default(),
                    sdma_engine,
                    profile_enabled,
                    stop,
                )
                .map(|ticks| gpu_copy_profile(index, ticks))
            }
        }
        AsyncCopyRoute::GpuBatch {
            device,
            index,
            entries,
            controls,
        } => execute_gpu_entries(
            &device,
            entries.into_iter(),
            controls,
            None,
            profile_enabled,
            stop,
        )
        .map(|ticks| gpu_copy_profile(index, ticks)),
    }
}

pub(crate) mod async_copy;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn image_pool_backing_uses_rounded_extent_without_exposing_provider_padding() {
        let gpu = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let description = PointerDescription {
            pointer_type: POINTER_TYPE_HSA,
            agent_base: 0x20_000,
            host_base: Some(0x10_000),
            size: 384,
            owner: gpu,
            global_flags: POOL_FLAG_COARSE,
            registered: true,
            alloc_flags: POINTER_ALLOC_NONPAGED | POINTER_ALLOC_HOST_ACCESS,
            user_data: 0x1234,
            accessible: vec![gpu],
        };
        let allocation = rocddi::memory::device::AllocationInfo {
            device_address: 0x20_000,
            host_address: Some(0x10_000),
            size: 4096,
            native_size: 8192,
            physical_backing_id: [0; 2],
        };
        for base in [0x10_000, 0x20_000] {
            let native = image_pool_description(&description, allocation, base).unwrap();
            assert_eq!(native.offset_range(base, 512), Some(0));
            assert_eq!(native.offset_range(base + 256, 512), Some(256));
            assert_eq!(native.offset_range(base + 4095, 1), Some(4095));
            assert_eq!(native.offset_range(base, 4097), None);
            assert_eq!(native.offset_range(base + 4095, 2), None);
            assert_eq!(native.offset_range(base, usize::MAX), None);
            assert!(image_pool_description(&description, allocation, base + 4096).is_none());
            assert!(image_pool_description(&description, allocation, base - 1).is_none());
            assert_eq!(native.agent_base, description.agent_base);
            assert_eq!(native.host_base, description.host_base);
            assert_eq!(native.owner.handle, gpu.handle);
            assert_eq!(native.accessible[0].handle, gpu.handle);
            assert_eq!(native.user_data, description.user_data);
            // Ordinary copy routing continues to use the requested extent.
            assert_eq!(description.offset_range(base, 384), Some(0));
            assert_eq!(description.offset_range(base, 512), None);
        }
        let mut info = std::mem::MaybeUninit::<HsaAmdPointerInfo>::uninit();
        // SAFETY: The public writer initializes the complete output structure.
        unsafe {
            write_pointer_info(
                info.as_mut_ptr(),
                size_of::<HsaAmdPointerInfo>() as u32,
                Some(&description),
            );
            assert_eq!(info.assume_init().size_in_bytes, 384);
        }
    }

    #[test]
    fn copy_policy_selects_compute_and_honors_forced_sdma_and_overrides() {
        assert_eq!(
            select_compute(true, true, false, true, true, CopyPreference::Default),
            Ok(true)
        );
        assert_eq!(
            select_compute(true, true, false, true, true, CopyPreference::PreferSdma),
            Ok(false)
        );
        assert_eq!(
            select_compute(true, true, true, false, true, CopyPreference::Default),
            Ok(false)
        );
        assert_eq!(
            select_compute(true, false, true, false, true, CopyPreference::PreferSdma),
            Ok(true)
        );
        assert_eq!(
            select_compute(false, true, false, false, true, CopyPreference::PreferSdma),
            Ok(true)
        );
        assert_eq!(
            select_compute(true, true, false, false, false, CopyPreference::Default),
            Ok(true)
        );
        assert_eq!(
            select_compute(false, true, false, false, true, CopyPreference::RequireSdma),
            Err(NOT_SUPPORTED)
        );
        assert_eq!(
            select_compute(true, false, false, false, true, CopyPreference::RequireSdma),
            Ok(false)
        );
    }

    #[test]
    fn staged_copy_preserves_payload_across_bounded_chunks_and_short_tail() {
        let source: Vec<u8> = (0..9 * 1024 * 1024 + 37)
            .map(|index| (index % 251) as u8)
            .collect();
        let mut destination = vec![0; source.len()];
        let mut chunks = Vec::new();
        assert!(
            staged_copy(source.len(), |offset, bytes| {
                assert!(bytes.len() <= 4 * 1024 * 1024);
                bytes.copy_from_slice(&source[offset..offset + bytes.len()]);
                destination[offset..offset + bytes.len()].copy_from_slice(bytes);
                chunks.push((offset, bytes.len()));
                Ok(())
            })
            .is_ok()
        );
        assert_eq!(source, destination);
        assert!(chunks.len() >= 3);
        assert_eq!(
            chunks.last().map(|&(offset, length)| offset + length),
            Some(source.len())
        );
    }

    #[test]
    fn staged_copy_stops_at_the_failed_leg_and_preserves_liveness_uncertainty() {
        let mut transfers = 0;
        let result = staged_copy(12 * 1024 * 1024, |_, _| {
            transfers += 1;
            if transfers == 2 {
                Err(CopyRouteFailure {
                    status: ERROR,
                    operands_may_be_live: true,
                })
            } else {
                Ok(())
            }
        });
        let failure = result.err().unwrap();
        assert_eq!(transfers, 2);
        assert_eq!(failure.status, ERROR);
        assert!(failure.operands_may_be_live);
        assert!(
            staged_copy(0, |_, _| {
                transfers += 1;
                Ok(())
            })
            .is_ok()
        );
        assert_eq!(transfers, 2);
    }

    #[test]
    fn async_dependencies_wait_for_every_signal_and_observe_cancellation() {
        let first = Arc::new(crate::signal::AmdSignal::user(1));
        let second = Arc::new(crate::signal::AmdSignal::user(1));
        let handles = [
            HsaSignal {
                handle: Arc::as_ptr(&first) as u64,
            },
            HsaSignal {
                handle: Arc::as_ptr(&second) as u64,
            },
        ];
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            sender
                .send(wait_async_dependencies(&handles, &AtomicBool::new(false)))
                .unwrap();
        });
        first.value.store(0, Ordering::Release);
        assert!(receiver.recv_timeout(Duration::from_millis(10)).is_err());
        second.value.store(0, Ordering::Release);
        assert!(receiver.recv_timeout(Duration::from_secs(1)).unwrap());
        worker.join().unwrap();
        assert!(!wait_async_dependencies(&handles, &AtomicBool::new(true)));
    }
}
