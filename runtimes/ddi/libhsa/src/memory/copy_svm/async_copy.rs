// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Async copy submission, batches, rectangles, and engine queries.

use super::*;

/// Enqueued copy with its operands retained until completion signaling.
pub(crate) struct AsyncCopyTask {
    /// Validated host or native copy route.
    route: AsyncCopyRoute,
    /// Optional dedicated SDMA engine selected by the caller.
    sdma_engine: Option<u32>,
    /// Allocation borrows released after native retirement.
    borrowed_memory: Vec<Option<usize>>,
    /// HSA signal decremented when the copy task completes.
    completion: HsaSignal,
}

/// Queues a resolved copy job for execution by a memory worker.
pub(crate) fn enqueue_copy_job(
    runtime: &mut HsaState,
    tasks: Vec<AsyncCopyTask>,
    dependency_slice: &[HsaSignal],
) -> Status {
    for task in &tasks {
        let status = runtime
            .signal_state
            .async_completion_status(task.completion);
        if status != SUCCESS {
            return status;
        }
    }
    runtime.memory_state.reap_finished_workers();
    let mut handles = Vec::new();
    let Some(handle_count) = dependency_slice.len().checked_add(tasks.len()) else {
        return OUT_OF_RESOURCES;
    };
    if handles.try_reserve_exact(handle_count).is_err()
        || runtime.memory_state.workers.try_reserve(1).is_err()
    {
        return OUT_OF_RESOURCES;
    }
    handles.extend_from_slice(dependency_slice);
    handles.extend(tasks.iter().map(|task| task.completion));
    let mut pending = match PendingAsyncRetention::new(
        &mut runtime.signal_state,
        &mut runtime.memory_state,
        Arc::new(handles),
    ) {
        Ok(pending) => pending,
        Err(status) => return status,
    };
    for task in &tasks {
        for &base in &task.borrowed_memory {
            if let Some(base) = base {
                if let Err(status) = pending.retain_memory(base) {
                    return status;
                }
            }
        }
    }
    let profile_enabled = matches!(
        pending.memory.async_copy_profiling,
        AsyncCopyProfiling::Enabled
    );
    let host_clock = if profile_enabled
        && tasks
            .iter()
            .any(|task| matches!(&task.route, AsyncCopyRoute::Host { .. }))
    {
        runtime.gpus.first().map(|gpu| gpu.device.clone())
    } else {
        None
    };
    let worker_handles = pending.handles.clone();
    let quarantine = pending.memory.async_copy_quarantine.clone();
    let stop = runtime.stop_workers.clone();
    let dependency_len = dependency_slice.len();
    let worker = match thread::Builder::new()
        .name("rocddi-hsa-copy".into())
        .spawn(move || {
            let mut execute = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                wait_async_dependencies(&worker_handles[..dependency_len], &stop)
            }))
            .unwrap_or(false);
            for task in tasks {
                let result = if execute {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute_copy_route(
                            task.route,
                            task.sdma_engine,
                            profile_enabled,
                            host_clock.as_ref(),
                            &stop,
                        )
                    }))
                    .unwrap_or(Err(CopyRouteFailure {
                        status: ERROR,
                        operands_may_be_live: true,
                    }))
                } else {
                    Err(CopyRouteFailure::retired(ERROR))
                };
                let (success, may_be_live, profile) = match result {
                    Ok(profile) => (true, false, profile),
                    Err(failure) => (false, failure.operands_may_be_live, None),
                };
                if may_be_live {
                    quarantine.store(true, Ordering::Release);
                }
                // Release only this operation's backing before publishing its
                // completion. Later operations keep their own borrows.
                release_async_resources(&[], &task.borrowed_memory, may_be_live);
                finish_async_copy_signal(task.completion, success, profile);
                execute = success;
            }
            release_async_resources(&worker_handles, &[], false);
        }) {
        Ok(worker) => worker,
        Err(_) => return OUT_OF_RESOURCES,
    };
    for completion in pending.handles.iter().skip(dependency_len) {
        pending.signals.set_copy_profile(*completion, None);
    }
    pending.transfer_to_worker();
    pending.memory.workers.push(worker);
    SUCCESS
}

/// Resolves and queues an asynchronous copy with retained operands.
pub(crate) fn enqueue_async_copy(
    runtime: &mut HsaState,
    resolved: ResolvedAsyncCopy,
    dependency_slice: &[HsaSignal],
    completion: HsaSignal,
    sdma_engine: Option<u32>,
) -> Status {
    enqueue_copy_job(
        runtime,
        vec![AsyncCopyTask {
            route: resolved.route,
            sdma_engine,
            borrowed_memory: resolved.borrowed_memory.to_vec(),
            completion,
        }],
        dependency_slice,
    )
}

#[allow(clippy::too_many_arguments)]
/// Validates caller inputs and submits an asynchronous copy.
pub(crate) unsafe fn submit_async_copy(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
    engine: Option<u32>,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if dst.is_null()
            || src.is_null()
            || completion.handle == 0
            || size > isize::MAX as usize
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.signal_state.owns_signal(completion) {
            return INVALID_SIGNAL;
        }
        if engine.is_some_and(|engine| engine.count_ones() != 1 || engine > u16::MAX.into()) {
            return INVALID_ARGUMENT;
        }
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if size == 0 {
            if !runtime.is_agent(dst_agent) || !runtime.is_agent(src_agent) {
                return INVALID_AGENT;
            }
            if let Some(mask) = engine {
                match copy_engine_mask(runtime, dst_agent, src_agent) {
                    Ok(available) if available & mask != 0 => {}
                    Ok(_) => return INVALID_ARGUMENT,
                    Err(status) => return status,
                }
            }
            return SUCCESS;
        }
        let resolved = match resolve_async_copy(
            runtime,
            dst as usize,
            dst_agent,
            src as usize,
            src_agent,
            size,
            None,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        let sdma_engine = match (engine, &resolved.route) {
            (Some(mask), AsyncCopyRoute::Gpu { index, .. }) => {
                let available = match copy_engine_mask_for_gpu(runtime, *index) {
                    Ok(available) => available,
                    Err(status) => return status,
                };
                if available & mask == 0 {
                    return INVALID_ARGUMENT;
                }
                Some(mask.trailing_zeros())
            }
            (Some(_), _) => return NOT_SUPPORTED,
            (None, _) => None,
        };
        enqueue_async_copy(runtime, resolved, dependency_slice, completion, sdma_engine)
    })
}

/// One validated input operation in an HSA async copy batch.
#[derive(Clone, Copy)]
struct BatchCopyEntry {
    /// Destination address supplied by the caller.
    destination: usize,
    /// Agent expected to reach the destination.
    destination_agent: HsaAgent,
    /// Source address supplied by the caller.
    source: usize,
    /// Agent expected to reach the source.
    source_agent: HsaAgent,
    /// Number of bytes transferred by this operation.
    size: usize,
    /// Packet shape and address mode.
    mode: BatchCopyMode,
}

/// Native packet shape selected for a batch entry.
#[derive(Clone, Copy)]
enum BatchCopyMode {
    /// Ordinary linear copy.
    Linear,
    /// Exchange two ranges.
    Swap,
    /// Resolve source or destination through GPU pointer slots.
    Indirect {
        /// Whether the source address comes from the indirect entry list.
        source: bool,
        /// Whether the destination address comes from the indirect entry list.
        destination: bool,
    },
}

/// Selects the copy mode encoded in a batch operation.
fn batch_copy_mode(operation: u16) -> Result<BatchCopyMode, Status> {
    match operation {
        0 | 1 => Ok(BatchCopyMode::Linear),
        2 => Ok(BatchCopyMode::Swap),
        3 => Ok(BatchCopyMode::Indirect {
            source: true,
            destination: false,
        }),
        4 => Ok(BatchCopyMode::Indirect {
            source: false,
            destination: true,
        }),
        5 => Ok(BatchCopyMode::Indirect {
            source: true,
            destination: true,
        }),
        _ => Err(INVALID_ARGUMENT),
    }
}

/// The C caller owns the array for the duration of the entry point. Alignment
/// and Rust slice-size limits are checked before forming a shared reference.
#[allow(unsafe_code)]
pub(crate) unsafe fn batch_slice<'a, T>(
    pointer: *const T,
    count: usize,
) -> Result<&'a [T], Status> {
    if pointer.is_null()
        || (pointer as usize) % align_of::<T>() != 0
        || count > isize::MAX as usize / size_of::<T>()
    {
        return Err(INVALID_ARGUMENT);
    }
    // SAFETY: The HSA caller supplies count initialized, readable elements.
    Ok(unsafe { std::slice::from_raw_parts(pointer, count) })
}

#[allow(unsafe_code)]
/// Decodes copy entries from the caller’s batch descriptor.
unsafe fn batch_copy_entries(op: HsaAmdMemoryCopyOp) -> Result<Vec<BatchCopyEntry>, Status> {
    let mode = batch_copy_mode(op.operation)?;
    if matches!(mode, BatchCopyMode::Indirect { .. }) && op.entry_count > 1024 {
        return Err(INVALID_ARGUMENT);
    }
    let count = usize::from(op.entry_count.max(1));
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    if op.entry_count == 0 {
        if op.operation == 1 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: The scalar members of the public C unions are active here.
        let destination_agent = unsafe { op.destination_agent.agent };
        let size = unsafe { op.size.bytes };
        if op.secondary_size
            != if matches!(mode, BatchCopyMode::Swap) {
                size
            } else {
                0
            }
        {
            return Err(INVALID_ARGUMENT);
        }
        entries.push(BatchCopyEntry {
            destination: op.destination as usize,
            destination_agent,
            source: op.source as usize,
            source_agent: op.source_agent,
            size,
            mode,
        });
    } else {
        if op.secondary_size != 0 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: A multi-entry descriptor selects pointer, agent, and size
        // lists. The caller keeps those arrays readable through this call.
        let destinations = unsafe { batch_slice(op.destination.cast::<*mut c_void>(), count)? };
        let agents = unsafe { batch_slice(op.destination_agent.list, count)? };
        match op.operation {
            0 => {
                // SAFETY: LINEAR with a positive entry count selects the list
                // members of both source and size unions.
                let sources = unsafe { batch_slice(op.source.cast::<*mut c_void>(), count)? };
                let sizes = unsafe { batch_slice(op.size.list, count)? };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: sources[index] as usize,
                        source_agent: op.source_agent,
                        size: sizes[index],
                        mode,
                    });
                }
            }
            1 => {
                // SAFETY: BROADCAST keeps the scalar source and size members.
                let size = unsafe { op.size.bytes };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: op.source as usize,
                        source_agent: op.source_agent,
                        size,
                        mode,
                    });
                }
            }
            2..=5 => {
                let sources = unsafe { batch_slice(op.source.cast::<*mut c_void>(), count)? };
                let sizes = unsafe { batch_slice(op.size.list, count)? };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: sources[index] as usize,
                        source_agent: op.source_agent,
                        size: sizes[index],
                        mode,
                    });
                }
            }
            _ => return Err(INVALID_ARGUMENT),
        }
    }
    if entries.iter().any(|entry| {
        entry.source == 0
            || entry.destination == 0
            || entry.size > isize::MAX as usize
            || (entry.size == 0
                && (op.entry_count != 0 || !matches!(entry.mode, BatchCopyMode::Linear)))
    }) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(entries)
}

/// Resolves indirect batch entry to a validated runtime resource.
fn resolve_indirect_batch_entry(
    runtime: &HsaState,
    entry: &BatchCopyEntry,
    indirect_source: bool,
    indirect_destination: bool,
    selected: Option<usize>,
) -> Result<ResolvedAsyncCopy, Status> {
    if entry.size > 0x3fff_ffe0 {
        return Err(INVALID_ARGUMENT);
    }
    let source_size = if indirect_source { 8 } else { entry.size };
    let destination_size = if indirect_destination { 8 } else { entry.size };
    let source_description = copy_pointer_description(runtime, entry.source, source_size)?;
    let destination_description =
        copy_pointer_description(runtime, entry.destination, destination_size)?;
    let source_agent = copy_agent(runtime, entry.source_agent, source_description.as_ref())?;
    let destination_agent = copy_agent(
        runtime,
        entry.destination_agent,
        destination_description.as_ref(),
    )?;
    let index = selected
        .or(runtime.gpu_index(source_agent))
        .or(runtime.gpu_index(destination_agent))
        .ok_or(INVALID_AGENT)?;
    if runtime.gpus[index].info.gfx_minor != 5 {
        return Err(NOT_SUPPORTED);
    }
    let gpu_agent = HsaAgent {
        handle: GPU_AGENT_BASE + index as u64,
    };
    let (source, source_base) = gpu_copy_address(
        runtime,
        source_description.as_ref(),
        entry.source,
        source_size,
        gpu_agent,
        ACCESS_PERMISSION_RO,
    )?;
    let (destination, destination_base) = gpu_copy_address(
        runtime,
        destination_description.as_ref(),
        entry.destination,
        destination_size,
        gpu_agent,
        if indirect_destination {
            ACCESS_PERMISSION_RO
        } else {
            ACCESS_PERMISSION_WO
        },
    )?;
    if indirect_source && source % 8 != 0 || indirect_destination && destination % 8 != 0 {
        return Err(INVALID_ARGUMENT);
    }
    Ok(ResolvedAsyncCopy {
        route: AsyncCopyRoute::Gpu {
            device: Box::new(runtime.gpus[index].device.clone()),
            index,
            entry: GpuCopyEntry::Indirect {
                destination,
                source,
                size: u64::try_from(entry.size).map_err(|_| INVALID_ARGUMENT)?,
                indirect_source,
                indirect_destination,
            },
        },
        borrowed_memory: [Some(destination_base), Some(source_base)],
    })
}

/// Resolves batch copy entries to a validated runtime resource.
fn resolve_batch_copy_entries(
    runtime: &HsaState,
    entries: &[BatchCopyEntry],
) -> Result<(AsyncCopyRoute, Vec<Option<usize>>), Status> {
    let mut prepared_entries = Vec::new();
    prepared_entries
        .try_reserve_exact(entries.len())
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut borrowed_memory = Vec::new();
    borrowed_memory
        .try_reserve_exact(
            entries
                .len()
                .checked_mul(4)
                .and_then(|count| count.checked_add(2))
                .ok_or(OUT_OF_RESOURCES)?,
        )
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut selected: Option<(usize, Box<Device>)> = None;
    for entry in entries {
        if entry.size != 0
            && runtime.gpu_index(entry.source_agent).is_none()
            && runtime.gpu_index(entry.destination_agent).is_none()
        {
            return Err(INVALID_AGENT);
        }
        let selected_index = selected.as_ref().map(|(index, _)| *index);
        let resolved = match entry.mode {
            BatchCopyMode::Indirect {
                source,
                destination,
            } => resolve_indirect_batch_entry(runtime, entry, source, destination, selected_index)?,
            BatchCopyMode::Linear | BatchCopyMode::Swap => resolve_async_copy(
                runtime,
                entry.destination,
                entry.destination_agent,
                entry.source,
                entry.source_agent,
                entry.size,
                selected_index,
            )?,
        };
        let (device, index, prepared) = match resolved.route {
            AsyncCopyRoute::Gpu {
                device,
                index,
                entry,
            } => (device, index, entry),
            AsyncCopyRoute::Noop if entries.len() == 1 => {
                return Ok((AsyncCopyRoute::Noop, borrowed_memory));
            }
            _ => return Err(INVALID_AGENT),
        };
        let prepared = if matches!(entry.mode, BatchCopyMode::Swap) {
            let GpuCopyEntry::Linear {
                destination,
                source,
                size,
            } = prepared
            else {
                return Err(NOT_SUPPORTED);
            };
            if runtime.gpus[index].info.gfx_minor != 5 {
                return Err(NOT_SUPPORTED);
            }
            let end_a = source.checked_add(size).ok_or(INVALID_ARGUMENT)?;
            let end_b = destination.checked_add(size).ok_or(INVALID_ARGUMENT)?;
            if source % 32 != 0 || destination % 32 != 0 || (source < end_b && destination < end_a)
            {
                return Err(INVALID_ARGUMENT);
            }
            let reverse = resolve_async_copy(
                runtime,
                entry.source,
                entry.source_agent,
                entry.destination,
                entry.destination_agent,
                entry.size,
                Some(index),
            )?;
            if !matches!(
                reverse.route,
                AsyncCopyRoute::Gpu {
                    index: reverse_index,
                    entry: GpuCopyEntry::Linear { .. },
                    ..
                } if reverse_index == index
            ) {
                return Err(NOT_SUPPORTED);
            }
            borrowed_memory.extend(reverse.borrowed_memory);
            GpuCopyEntry::Swap {
                address_a: source,
                address_b: destination,
                size,
            }
        } else {
            prepared
        };
        // A multi-entry operation uses the source GPU, or its first
        // destination GPU for host sources. Later peer destinations must be
        // accessible from that same GPU before one ordered DDI submission.
        selected.get_or_insert((index, device));
        prepared_entries.push(prepared);
        borrowed_memory.extend(resolved.borrowed_memory);
    }
    let (index, device) = selected.ok_or(INVALID_ARGUMENT)?;
    Ok((
        AsyncCopyRoute::GpuBatch {
            device,
            index,
            entries: prepared_entries,
            controls: BatchCopyControls::default(),
        },
        borrowed_memory,
    ))
}

/// Copy tasks staged for publication as one async batch.
pub(crate) struct PreparedBatchCopy {
    /// Tasks to enqueue after every entry validates.
    pub(crate) tasks: Vec<AsyncCopyTask>,
}

/// Resolves the address used by a batch wait or signal operation.
fn batch_control_address(
    runtime: &HsaState,
    pointer: usize,
    gpu: HsaAgent,
    permission: u32,
) -> Result<(u64, usize), Status> {
    let description = copy_pointer_description(runtime, pointer, 8)?;
    let (address, base) =
        gpu_copy_address(runtime, description.as_ref(), pointer, 8, gpu, permission)?;
    if address % 8 != 0 {
        return Err(INVALID_ARGUMENT);
    }
    Ok((address, base))
}

#[allow(unsafe_code)]
/// Prepares batch copy before publication or execution.
pub(crate) unsafe fn prepare_batch_copy(
    runtime: &HsaState,
    operations: &[HsaAmdMemoryCopyOp],
) -> Result<PreparedBatchCopy, Status> {
    if operations.is_empty() {
        return Err(INVALID_ARGUMENT);
    }
    let mut tasks = Vec::new();
    tasks
        .try_reserve_exact(operations.len())
        .map_err(|_| OUT_OF_RESOURCES)?;
    for &op in operations {
        if op.version != 1
            || op.operation > 5
            || op.reserved != [0]
            || op.wait.reserved != 0
            || op.wait.function > 6
            || op.wait.scope > 2
            || op.signal.reserved != 0
            || op.signal.operation > 3
            || op.signal.scope > 2
        {
            return Err(INVALID_ARGUMENT);
        }
        if !runtime.signal_state.owns_signal(op.completion_signal) {
            return Err(INVALID_SIGNAL);
        }
        if op.traffic_class != 0 {
            return Err(NOT_SUPPORTED);
        }
        if (op.wait.function == 0
            && (!op.wait.address.is_null()
                || op.wait.value != 0
                || op.wait.mask != 0
                || op.wait.scope != 0 && op.operation < 3))
            || (op.wait.function != 0 && op.wait.address.is_null())
            || (op.signal.operation == 0
                && (op.signal.scope != 0 || !op.signal.address.is_null() || op.signal.data != 0))
            || (op.signal.operation != 0 && op.signal.address.is_null())
        {
            return Err(INVALID_ARGUMENT);
        }
        if op.operation == 1 && runtime.gpu_index(op.source_agent).is_none() {
            return Err(INVALID_AGENT);
        }
        // SAFETY: Each operation selects the documented scalar or array union
        // members. All array values are copied before the caller returns.
        let entries = unsafe { batch_copy_entries(op)? };
        let (mut route, mut bases) = resolve_batch_copy_entries(runtime, &entries)?;
        if let AsyncCopyRoute::GpuBatch {
            index,
            entries,
            controls,
            ..
        } = &mut route
        {
            if (op.wait.function != 0 || op.signal.operation != 0)
                && (runtime.gpus[*index].info.gfx_minor != 5
                    || entries.iter().any(|entry| {
                        !matches!(
                            entry,
                            GpuCopyEntry::Linear { .. }
                                | GpuCopyEntry::Swap { .. }
                                | GpuCopyEntry::Indirect { .. }
                        )
                    }))
            {
                return Err(NOT_SUPPORTED);
            }
            let gpu = HsaAgent {
                handle: GPU_AGENT_BASE + *index as u64,
            };
            if op.wait.function != 0 {
                let (address, base) = batch_control_address(
                    runtime,
                    op.wait.address as usize,
                    gpu,
                    ACCESS_PERMISSION_RO,
                )?;
                controls.wait = Some(GpuCopyWait {
                    address,
                    function: u8::try_from(op.wait.function).map_err(|_| INVALID_ARGUMENT)?,
                    reference: op.wait.value,
                    mask: op.wait.mask,
                });
                bases.push(Some(base));
            }
            if op.signal.operation != 0 {
                let (address, base) = batch_control_address(
                    runtime,
                    op.signal.address as usize,
                    gpu,
                    ACCESS_PERMISSION_RW,
                )?;
                controls.signal = Some(GpuCopySignal {
                    address,
                    operation: match op.signal.operation {
                        1 => GpuCopySignalOperation::Write,
                        2 => GpuCopySignalOperation::Add,
                        3 => GpuCopySignalOperation::Subtract,
                        _ => return Err(INVALID_ARGUMENT),
                    },
                    data: op.signal.data,
                });
                bases.push(Some(base));
            }
        } else if op.wait.function != 0 || op.signal.operation != 0 {
            return Err(INVALID_ARGUMENT);
        }
        tasks.push(AsyncCopyTask {
            route,
            sdma_engine: None,
            borrowed_memory: bases,
            completion: op.completion_signal,
        });
    }
    Ok(PreparedBatchCopy { tasks })
}

/// Computes the address range covered by a pitched copy.
pub(crate) fn pitched_copy_range(
    pitched: HsaPitchedPtr,
    offset: HsaDim3,
    range: HsaDim3,
) -> Option<(usize, usize)> {
    let base = pitched.base as usize;
    let pitch = pitched.pitch;
    let slice = pitched.slice;
    let width = range.x as usize;
    let row_end = (offset.x as usize).checked_add(width)?;
    if base == 0 || (base | pitch | slice | offset.x as usize | width) & 3 != 0 || row_end > pitch {
        return None;
    }
    let first = (offset.z as usize)
        .checked_mul(slice)?
        .checked_add((offset.y as usize).checked_mul(pitch)?)?
        .checked_add(offset.x as usize)?;
    let extent = ((range.z - 1) as usize)
        .checked_mul(slice)?
        .checked_add(((range.y - 1) as usize).checked_mul(pitch)?)?
        .checked_add(width)?;
    let address = base.checked_add(first)?;
    address.checked_add(extent)?;
    Some((address, extent))
}

/// Copies engine mask for GPU after validating its source and destination.
fn copy_engine_mask_for_gpu(runtime: &HsaState, index: usize) -> Result<u32, Status> {
    let gpu = runtime.gpus[index].device.gpu().map_err(map_error)?;
    if !gpu.supports_linear_copy() {
        return Ok(0);
    }
    let count = gpu.info().queues.sdma_engine_count.min(16);
    let valid_ids = (1_u32 << count) - 1;
    Ok(gpu.available_sdma_engines().map_err(map_error)? & valid_ids)
}

/// Copies engine mask after validating its source and destination.
pub(crate) fn copy_engine_mask(
    runtime: &HsaState,
    dst: HsaAgent,
    src: HsaAgent,
) -> Result<u32, Status> {
    let Some(index) = runtime.gpu_index(src).or_else(|| runtime.gpu_index(dst)) else {
        return Ok(0);
    };
    copy_engine_mask_for_gpu(runtime, index)
}
