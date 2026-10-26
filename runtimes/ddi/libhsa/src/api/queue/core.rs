// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA queue lifecycle, configuration, and profiling C entry points.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::queue::*;
use crate::runtime::{
    CallbackScope, InFlightGeneration, boundary, initialized_mut, lock, map_error,
};
use crate::signal::AsyncCopyClock;
use rocddi::device::gpu::queue::{
    AqlControlRegion, QueuePriority, QueueRingMemory, SdmaEngineSelection,
};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a hardware-backed HSA queue.
pub unsafe extern "C" fn hsa_queue_create(
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    callback: QueueErrorCallback,
    data: *mut c_void,
    private_segment_size: u32,
    _group_segment_size: u32,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        if queue.is_null()
            || size == 0
            || !size.is_power_of_two()
            || !matches!(
                queue_type,
                QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE | QUEUE_TYPE_COOPERATIVE
            )
        {
            return INVALID_ARGUMENT;
        }
        let mut created_log = None;
        // SAFETY: The HSA caller supplies writable queue output and retains callback data.
        let status = unsafe {
            create_hardware_queue(
                None,
                agent,
                size,
                queue_type,
                QueuePriority::Normal,
                callback,
                // SAFETY: The C caller retains and synchronizes callback data
                // through queue destruction and the last event callback.
                CallbackArg::new(data),
                private_segment_size,
                QueueRingMemory::System,
                None,
                queue,
                &mut created_log,
            )
        };
        let log = created_log.and_then(|(id, public)| {
            lock().ok().and_then(|guard| {
                guard.as_ref().and_then(|runtime| {
                    runtime.prepare_log(
                        AMD_LOG_FLAG_INFO,
                        format_args!(
                            "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={}",
                            agent.handle,
                            size.max(64)
                        ),
                    )
                })
            })
        });
        if let Some(log) = log {
            log.write();
        }
        status
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_counted_queue_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Acquires a counted queue backed by a shared hardware queue.
pub unsafe extern "C" fn hsa_amd_counted_queue_acquire(
    agent: HsaAgent,
    queue_type: u32,
    priority: u32,
    callback: QueueErrorCallback,
    data: *mut c_void,
    _flags: u64,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        let Ok(_creation_gate) = COUNTED_CREATE_GATE.lock() else {
            return ERROR;
        };
        let pool_key = (agent.handle, priority);
        let (generation, create_new, counted_queue_size, existing, native_priority) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(generation) = guard.active_generation() else {
                return NOT_INITIALIZED;
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            if queue.is_null() {
                return INVALID_ARGUMENT;
            }
            let Some(native_priority) = queue_priority(priority) else {
                return INVALID_ARGUMENT;
            };
            if queue_type != QUEUE_TYPE_MULTI {
                return INVALID_QUEUE_CREATION;
            }
            if runtime.gpu_index(agent).is_none() {
                return INVALID_AGENT;
            }
            if runtime.queue_state.counted_queues.try_reserve(1).is_err()
                || runtime
                    .queue_state
                    .counted_queue_pools
                    .try_reserve(1)
                    .is_err()
            {
                return OUT_OF_RESOURCES;
            }
            let pool = runtime
                .queue_state
                .counted_queue_pools
                .entry(pool_key)
                .or_default();
            let create_new = pool.len() < runtime.queue_state.counted_queue_limit;
            if create_new && pool.try_reserve(1).is_err() {
                return OUT_OF_RESOURCES;
            }
            (
                generation,
                create_new,
                runtime.queue_state.counted_queue_size,
                least_used_counted_queue(pool),
                native_priority,
            )
        };
        let mut created_log = None;
        let hardware_queue = if create_new {
            let mut created = std::ptr::null_mut();
            // SAFETY: The stack output is writable; the callback data remains
            // owned by the HSA caller until the resulting queue is destroyed.
            let status = unsafe {
                create_hardware_queue(
                    Some(generation),
                    agent,
                    counted_queue_size,
                    queue_type,
                    native_priority,
                    callback,
                    // SAFETY: The C caller retains the callback argument.
                    CallbackArg::new(data),
                    0,
                    QueueRingMemory::System,
                    None,
                    &raw mut created,
                    &mut created_log,
                )
            };
            if status != SUCCESS {
                return OUT_OF_RESOURCES;
            }
            created as usize
        } else {
            let Some(existing) = existing else {
                return OUT_OF_RESOURCES;
            };
            existing
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => {
                if create_new {
                    // SAFETY: The newly created internal queue was not returned.
                    unsafe { hsa_queue_destroy(hardware_queue as *mut HsaQueue) };
                }
                return status;
            }
        };
        if guard.active_generation() != Some(generation) {
            drop(guard);
            if create_new {
                // SAFETY: No public counted handle was returned to the caller.
                unsafe { hsa_queue_destroy(hardware_queue as *mut HsaQueue) };
            }
            return INVALID_RUNTIME_STATE;
        }
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if create_new {
            let Some(record) = runtime.queue_state.queues.get_mut(&hardware_queue) else {
                return ERROR;
            };
            record.counted_pool_key = Some(pool_key);
            runtime
                .queue_state
                .counted_queue_pools
                .entry(pool_key)
                .or_default()
                .push(CountedHardwareQueue {
                    queue: hardware_queue,
                    use_count: 0,
                });
            // SAFETY: The native AQL control mapping is live and its properties
            // word is aligned through the complete counted queue lifetime.
            unsafe {
                let address = AqlControlRegion::properties_address_at(hardware_queue);
                (&*(address as *const AtomicU32))
                    .fetch_or(AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
            }
        }
        let Some(entry) = runtime
            .queue_state
            .counted_queue_pools
            .get_mut(&pool_key)
            .and_then(|pool| pool.iter_mut().find(|entry| entry.queue == hardware_queue))
        else {
            return ERROR;
        };
        let Some(use_count) = entry.use_count.checked_add(1) else {
            return OUT_OF_RESOURCES;
        };
        // SAFETY: The selected hardware queue remains owned by the pool until
        // the runtime generation is shut down.
        let mut counted = unsafe { CountedQueue::new(hardware_queue, pool_key) };
        let public = counted.public_pointer();
        let key = public as usize;
        entry.use_count = use_count;
        runtime.queue_state.released_counted_queues.remove(&key);
        runtime.queue_state.counted_queues.insert(key, counted);
        // SAFETY: The caller supplied writable output storage.
        unsafe { queue.write(public) };
        let log = created_log.and_then(|(id, public)| {
            runtime.prepare_log(
                AMD_LOG_FLAG_INFO,
                format_args!(
                    "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={counted_queue_size}",
                    agent.handle
                ),
            )
        });
        drop(guard);
        if let Some(log) = log {
            log.write();
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_counted_queue_release")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases the counted queue resource.
pub unsafe extern "C" fn hsa_amd_counted_queue_release(queue: *mut HsaQueue) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if queue.is_null() {
            return INVALID_ARGUMENT;
        }
        let key = queue as usize;
        let Some(counted) = runtime.queue_state.counted_queues.remove(&key) else {
            return ERROR;
        };
        let Some(entry) = runtime
            .queue_state
            .counted_queue_pools
            .get_mut(&counted.pool_key)
            .and_then(|pool| {
                pool.iter_mut()
                    .find(|entry| entry.queue == counted.hardware_queue)
            })
        else {
            runtime.queue_state.counted_queues.insert(key, counted);
            return ERROR;
        };
        let Some(use_count) = entry.use_count.checked_sub(1) else {
            runtime.queue_state.counted_queues.insert(key, counted);
            return ERROR;
        };
        entry.use_count = use_count;
        runtime.queue_state.released_counted_queues.insert(key);
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_soft_queue_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a host-managed queue with direct index control.
pub unsafe extern "C" fn hsa_soft_queue_create(
    region: HsaRegion,
    size: u32,
    queue_type: u32,
    features: u32,
    doorbell_signal: HsaSignal,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        if queue.is_null()
            || size == 0
            || !size.is_power_of_two()
            || !matches!(queue_type, QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE)
            || features == 0
            || features & !(QUEUE_FEATURE_KERNEL_DISPATCH | QUEUE_FEATURE_AGENT_DISPATCH) != 0
            || doorbell_signal.handle == 0
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let pool = HsaMemoryPool {
            handle: region.handle,
        };
        if runtime.decode_cpu_pool(pool).is_none() && runtime.decode_gpu_pool(pool).is_none() {
            return INVALID_REGION;
        }
        if !runtime.signal_state.owns_signal(doorbell_signal) {
            return INVALID_SIGNAL;
        }
        if runtime.queue_state.soft_queues.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let count = size as usize;
        let mut ring = Vec::new();
        if ring.try_reserve_exact(count).is_err() {
            return OUT_OF_RESOURCES;
        }
        let mut invalid_packet = [0_u8; AQL_PACKET_BYTES];
        invalid_packet[0] = 1;
        for _ in 0..count {
            ring.push(SoftPacket(invalid_packet));
        }
        let mut control = Box::new(SoftQueueControl([0_u64; 32]));
        let public = control.0.as_mut_ptr().cast::<HsaQueue>();
        let id = match runtime.queue_state.allocate_id() {
            Ok(id) => id,
            Err(status) => return status,
        };
        // The queue publishes this handle in its header. Retain its backing
        // even if the caller destroys the public signal first.
        if let Err(status) = runtime.signal_state.retain_async_signal(doorbell_signal) {
            return status;
        }
        // SAFETY: The boxed control record is aligned for HsaQueue and the
        // fixed index offsets used by the base queue atomics.
        unsafe {
            public.write(HsaQueue {
                queue_type,
                features,
                base_address: ring.as_mut_ptr().cast(),
                doorbell_signal,
                size,
                reserved: 0,
                id,
            });
        }
        runtime.queue_state.soft_queues.insert(
            public as usize,
            SoftQueue {
                _control: control,
                _ring: ring,
                doorbell_signal,
                active: true,
            },
        );
        // SAFETY: The caller supplied writable output storage and the control
        // allocation remains stable until hsa_queue_destroy.
        unsafe { queue.write(public) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates an AMD queue from an engine-specific descriptor.
pub unsafe extern "C" fn hsa_amd_queue_create(
    agent: HsaAgent,
    descriptors: *mut HsaAmdQueueCreateDesc,
    descriptor_count: u32,
) -> Status {
    boundary(|| {
        if descriptors.is_null() || descriptor_count == 0 {
            return INVALID_ARGUMENT;
        }
        let generation = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if runtime.gpu_index(agent).is_none() {
                return INVALID_AGENT;
            }
            let Some(generation) = guard.active_generation() else {
                return INVALID_RUNTIME_STATE;
            };
            generation
        };
        let mut first_error = SUCCESS;
        let mut logs = Vec::new();
        for index in 0..descriptor_count as usize {
            // SAFETY: The caller supplies initialized input fields and writable
            // output storage. queue may still be uninitialized.
            let pointer = unsafe { descriptors.add(index) };
            let input = unsafe { read_queue_create_input(pointer) };
            // SAFETY: The caller supplies this writable output-only field.
            let output = unsafe { &raw mut (*pointer).queue };
            let mut fail = |status| {
                // SAFETY: Each descriptor has writable output storage.
                unsafe { output.write(std::ptr::null_mut()) };
                if first_error == SUCCESS {
                    first_error = status;
                }
            };
            if input.version != AMD_QUEUE_CREATE_DESC_VERSION
                || input.queue_size_bytes == 0
                || !input.queue_size_bytes.is_power_of_two()
                || input.priority > AMD_QUEUE_PRIORITY_HIGH
                || input.traffic_class != 0
                || input.reserved_header.iter().any(|byte| *byte != 0)
                || input.reserved.iter().any(|byte| *byte != 0)
            {
                fail(INVALID_ARGUMENT);
                continue;
            }
            let known_flags =
                AMD_QUEUE_CREATE_DEVICE_MEM_RING | AMD_QUEUE_CREATE_DEVICE_MEM_DESCRIPTOR;
            if input.flags & !known_flags != 0 {
                fail(INVALID_ARGUMENT);
                continue;
            }
            if input.flags & AMD_QUEUE_CREATE_DEVICE_MEM_DESCRIPTOR != 0 {
                fail(INVALID_QUEUE_CREATION);
                continue;
            }
            let ring_memory = if input.flags & AMD_QUEUE_CREATE_DEVICE_MEM_RING != 0 {
                QueueRingMemory::HostVisibleLocal
            } else {
                QueueRingMemory::System
            };
            if input.engine_type == AMD_QUEUE_ENGINE_SDMA {
                if input.priority != AMD_QUEUE_PRIORITY_NORMAL || input.callback.is_some() {
                    fail(INVALID_QUEUE_CREATION);
                    continue;
                }
                // SAFETY: engine_type selects the SDMA arm of the C union.
                let sdma = unsafe { (&raw const (*pointer).engine.sdma).read() };
                if sdma.reserved.iter().any(|word| *word != 0) {
                    fail(INVALID_ARGUMENT);
                    continue;
                }
                let selection = if sdma.engine_id == u32::MAX {
                    SdmaEngineSelection::Any
                } else {
                    SdmaEngineSelection::Id(sdma.engine_id)
                };
                // SAFETY: The caller supplies this writable output-only field.
                unsafe { output.write(std::ptr::null_mut()) };
                let mut created_log = None;
                // SAFETY: The descriptor's output remains writable for this
                // call. SDMA packet production begins only after publication.
                let status = unsafe {
                    create_sdma_queue(
                        Some(generation),
                        agent,
                        input.queue_size_bytes,
                        selection,
                        ring_memory,
                        output,
                        &mut created_log,
                    )
                };
                if status != SUCCESS {
                    fail(status);
                } else if let Some((id, public, engine)) = created_log {
                    if let Some(log) = lock().ok().and_then(|guard| guard.as_ref().and_then(|runtime| runtime.prepare_log(
                        AMD_LOG_FLAG_INFO,
                        format_args!(
                            "created SDMA queue id={id} agent=0x{:x} address=0x{public:x} engine={engine} bytes={}",
                            agent.handle, input.queue_size_bytes
                        ),
                    ))) {
                        if logs.try_reserve(1).is_ok() {
                            logs.push(log);
                        }
                    }
                }
                continue;
            }
            if input.engine_type != AMD_QUEUE_ENGINE_COMPUTE {
                fail(if input.engine_type == AMD_QUEUE_ENGINE_AIE {
                    INVALID_QUEUE_CREATION
                } else {
                    INVALID_ARGUMENT
                });
                continue;
            }
            // SAFETY: engine_type selects the compute arm of the C union.
            let compute = unsafe { (&raw const (*pointer).engine.compute).read() };
            if input.queue_size_bytes % AQL_PACKET_BYTES as u32 != 0
                || !matches!(
                    compute.queue_type,
                    QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE | QUEUE_TYPE_COOPERATIVE
                )
                || (compute.cu_mask_count == 0) != compute.cu_mask.is_null()
                || compute.cu_mask_count % 32 != 0
                || compute.reserved.iter().any(|word| *word != 0)
            {
                fail(INVALID_ARGUMENT);
                continue;
            }
            let Some(priority) = queue_priority(input.priority) else {
                fail(INVALID_ARGUMENT);
                continue;
            };
            // SAFETY: The mask pointer and count were validated above. Own the
            // words before queue output can overwrite any aliased input.
            let cu_mask = match unsafe { copy_queue_create_mask(compute) } {
                Ok(mask) => mask,
                Err(status) => {
                    fail(status);
                    continue;
                }
            };
            // SAFETY: The caller supplies this writable output-only field.
            unsafe { output.write(std::ptr::null_mut()) };
            let packet_count = input.queue_size_bytes / AQL_PACKET_BYTES as u32;
            let mut created_log = None;
            // SAFETY: The admitted descriptor supplies writable output and retained callback data.
            let status = unsafe {
                create_hardware_queue(
                    Some(generation),
                    agent,
                    packet_count,
                    compute.queue_type,
                    priority,
                    input.callback,
                    // SAFETY: The C caller retains and synchronizes callback data
                    // through queue destruction and the last event callback.
                    CallbackArg::new(input.callback_data),
                    compute.private_segment_size,
                    ring_memory,
                    cu_mask,
                    output,
                    &mut created_log,
                )
            };
            if status != SUCCESS {
                fail(status);
            } else if let Some((id, public)) = created_log {
                if let Some(log) = lock().ok().and_then(|guard| guard.as_ref().and_then(|runtime| runtime.prepare_log(
                    AMD_LOG_FLAG_INFO,
                    format_args!(
                        "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={packet_count}",
                        agent.handle
                    ),
                ))) {
                    if logs.try_reserve(1).is_ok() {
                        logs.push(log);
                    }
                }
            }
        }
        for log in logs {
            log.write();
        }
        first_error
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_cu_set_mask")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Sets the compute-unit mask used by a queue.
pub unsafe extern "C" fn hsa_amd_queue_cu_set_mask(
    queue: *const HsaQueue,
    bit_count: u32,
    mask: *const u32,
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
        if counted_pool_key(runtime, queue).is_some() {
            return INVALID_QUEUE;
        }
        let Some(hardware_key) = hardware_queue_key(runtime, queue) else {
            return INVALID_QUEUE;
        };
        if bit_count % 32 != 0 || (bit_count != 0 && mask.is_null()) {
            return INVALID_ARGUMENT;
        }
        let Some(agent) = runtime
            .queue_state
            .queues
            .get(&hardware_key)
            .map(|record| record.agent)
        else {
            return INVALID_QUEUE;
        };
        let Some(gpu_index) = runtime.gpu_index(agent) else {
            return INVALID_QUEUE;
        };
        let selected = if bit_count == 0 {
            let compute_units = runtime.gpus[gpu_index].info().compute_unit_count;
            full_cu_mask(compute_units)
        } else {
            // SAFETY: The ABI requires bit_count / 32 readable words. Own them
            // before native queue control can mutate caller-visible storage.
            match unsafe { copy_cu_mask(mask, bit_count as usize / 32) } {
                Ok(selected) => selected,
                Err(status) => return status,
            }
        };
        let Some(record) = runtime.queue_state.queues.get_mut(&hardware_key) else {
            return INVALID_QUEUE;
        };
        match record.native.set_cu_mask(&selected) {
            Ok(()) => {
                record.cu_mask = selected;
                SUCCESS
            }
            Err(error) => map_error(error),
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_cu_get_mask")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Copies the queue compute-unit mask into caller storage.
pub unsafe extern "C" fn hsa_amd_queue_cu_get_mask(
    queue: *const HsaQueue,
    bit_count: u32,
    mask: *mut u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(hardware_key) = hardware_queue_key(runtime, queue) else {
            return INVALID_QUEUE;
        };
        if bit_count == 0 || bit_count % 32 != 0 {
            return INVALID_ARGUMENT;
        }
        let Some(record) = runtime.queue_state.queues.get(&hardware_key) else {
            return INVALID_QUEUE;
        };
        let Some(gpu_index) = runtime.gpu_index(record.agent) else {
            return INVALID_QUEUE;
        };
        let enabled = if record.cu_mask.is_empty() {
            full_cu_mask(runtime.gpus[gpu_index].info().compute_unit_count)
        } else {
            record.cu_mask.clone()
        };
        let output_words = bit_count as usize / 32;
        let copied = output_words.min(enabled.len());
        // SAFETY: The caller supplied output_words writable entries, which
        // need not contain initialized u32 values. Enabled is separate storage.
        unsafe {
            std::ptr::write_bytes(mask, 0, output_words);
            std::ptr::copy_nonoverlapping(enabled.as_ptr(), mask, copied);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_set_priority")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Changes a queue’s native scheduling priority.
pub unsafe extern "C" fn hsa_amd_queue_set_priority(queue: *mut HsaQueue, priority: u32) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if queue.is_null() {
            return INVALID_ARGUMENT;
        }
        if runtime
            .queue_state
            .sdma_queues
            .contains_key(&(queue as usize))
        {
            // This entry point applies only to compute queues.
            return INVALID_QUEUE;
        }
        if counted_pool_key(runtime, queue).is_some() {
            return INVALID_QUEUE;
        }
        let Some(hardware_key) = hardware_queue_key(runtime, queue) else {
            return INVALID_QUEUE;
        };
        let Some(priority) = queue_priority(priority) else {
            return INVALID_ARGUMENT;
        };
        let Some(record) = runtime.queue_state.queues.get_mut(&hardware_key) else {
            return INVALID_QUEUE;
        };
        record
            .native
            .set_priority(priority)
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_inactivate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Stops a queue from accepting further packet submissions.
pub unsafe extern "C" fn hsa_queue_inactivate(queue: *mut HsaQueue) -> Status {
    boundary(|| {
        if queue.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if let Some(intercept) = runtime
            .queue_state
            .intercept_queues
            .get(&(queue as usize))
            .cloned()
        {
            if CallbackScope::active() {
                return INVALID_RUNTIME_STATE;
            }
            drop(guard);
            if let Err(status) = intercept.stop() {
                return status;
            }
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            return runtime.queue_state.queues.get_mut(&intercept.lower).map_or(
                INVALID_QUEUE,
                |record| {
                    record.inactivated = true;
                    record
                        .native
                        .inactivate()
                        .map_or_else(map_error, |()| SUCCESS)
                },
            );
        }
        if let Some(hardware_key) = hardware_queue_key(runtime, queue) {
            if let Some(record) = runtime.queue_state.queues.get_mut(&hardware_key) {
                // A shared cooperative queue must not be returned by a later
                // create once any holder starts inactivating it.
                record.inactivated = true;
                return record
                    .native
                    .inactivate()
                    .map_or_else(map_error, |()| SUCCESS);
            }
        }
        if let Some(record) = runtime.queue_state.sdma_queues.get_mut(&(queue as usize)) {
            return record
                .native
                .inactivate()
                .map_or_else(map_error, |()| SUCCESS);
        }
        if let Some(record) = runtime.queue_state.soft_queues.get_mut(&(queue as usize)) {
            record.inactivate();
            return SUCCESS;
        }
        INVALID_QUEUE
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Stops a queue worker and retires its native queue.
pub unsafe extern "C" fn hsa_queue_destroy(queue: *mut HsaQueue) -> Status {
    boundary(|| {
        #[allow(
            clippy::large_enum_variant,
            reason = "queue teardown transfers owners on the stack without allocating"
        )]
        enum RemovedQueue {
            Aql(Queue),
            Sdma(SdmaQueue),
        }

        if queue.is_null() {
            return INVALID_ARGUMENT;
        }
        if CallbackScope::active() {
            return INVALID_RUNTIME_STATE;
        }
        let intercept = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            runtime
                .queue_state
                .intercept_queues
                .get(&(queue as usize))
                .cloned()
                .map(|owner| runtime.inflight.enter().map(|token| (owner, token)))
        };
        if let Some(intercept) = intercept {
            let Some((intercept, _in_flight)) = intercept else {
                return INVALID_RUNTIME_STATE;
            };
            return crate::queue_intercept::destroy(&intercept);
        }
        let key = queue as usize;
        let (mut record, cooperative_agent, in_flight) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(generation) = guard.active_generation() else {
                return NOT_INITIALIZED;
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let cooperative_agent = if let Some(record) = runtime.queue_state.queues.get_mut(&key) {
                if record.public_handle != key {
                    return INVALID_QUEUE;
                }
                if record.counted_pool_key.is_some() {
                    return INVALID_QUEUE;
                }
                if record.cooperative_refs > 1 {
                    record.cooperative_refs -= 1;
                    return SUCCESS;
                }
                (record.cooperative_refs == 1).then_some(record.agent.handle)
            } else {
                None
            };
            let in_flight = match InFlightGeneration::enter(generation, &runtime.inflight) {
                Ok(in_flight) => in_flight,
                Err(status) => return status,
            };
            if let Some(agent) = cooperative_agent {
                if runtime
                    .queue_state
                    .cooperative_teardown
                    .try_reserve(1)
                    .is_err()
                {
                    return OUT_OF_RESOURCES;
                }
                runtime.queue_state.cooperative_teardown.insert(agent);
            }
            let record = if let Some(record) = runtime.queue_state.queues.remove(&key) {
                RemovedQueue::Aql(record)
            } else if let Some(record) = runtime.queue_state.sdma_queues.remove(&key) {
                RemovedQueue::Sdma(record)
            } else {
                return if let Some(soft) = runtime.queue_state.soft_queues.remove(&key) {
                    runtime
                        .signal_state
                        .release_async_signal(soft.doorbell_signal);
                    SUCCESS
                } else {
                    INVALID_QUEUE
                };
            };
            (record, cooperative_agent, in_flight)
        };
        let result = match &mut record {
            RemovedQueue::Aql(queue) => destroy_runtime_queue(queue),
            RemovedQueue::Sdma(queue) => destroy_runtime_sdma_queue(queue),
        };
        match result {
            Ok(()) => {
                let engine = match record {
                    RemovedQueue::Aql(_) => "AQL",
                    RemovedQueue::Sdma(_) => "SDMA",
                };
                let log = lock().ok().and_then(|mut guard| {
                    in_flight.is_current(&guard).then_some(())?;
                    guard.as_mut().and_then(|runtime| {
                        if let Some(agent) = cooperative_agent {
                            runtime.queue_state.cooperative_teardown.remove(&agent);
                        }
                        runtime.prepare_log(
                            AMD_LOG_FLAG_INFO,
                            format_args!("destroyed {engine} queue address=0x{key:x}"),
                        )
                    })
                });
                if let Some(log) = log {
                    log.write();
                }
                SUCCESS
            }
            Err(error) => {
                let status = map_error(error);
                let Ok(mut guard) = lock() else {
                    in_flight.quarantine();
                    std::mem::forget(record);
                    return status;
                };
                if !in_flight.is_current(&guard) {
                    in_flight.quarantine();
                    std::mem::forget(record);
                    return status;
                }
                let Some(runtime) = guard.as_mut() else {
                    in_flight.quarantine();
                    std::mem::forget(record);
                    return status;
                };
                // A native teardown may partially release an ID. Never
                // replace a queue created under the same public key while
                // this owner was outside the registry lock.
                if runtime.queue_state.queues.contains_key(&key)
                    || runtime.queue_state.sdma_queues.contains_key(&key)
                {
                    in_flight.quarantine();
                    std::mem::forget(record);
                    return INVALID_RUNTIME_STATE;
                }
                if let Some(agent) = cooperative_agent {
                    runtime.queue_state.cooperative_teardown.remove(&agent);
                }
                match record {
                    RemovedQueue::Aql(record) => {
                        runtime.queue_state.queues.insert(key, record);
                    }
                    RemovedQueue::Sdma(record) => {
                        runtime.queue_state.sdma_queues.insert(key, record);
                    }
                }
                status
            }
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_profiling_set_profiler_enabled")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Enables or disables dispatch profiling on a queue.
pub unsafe extern "C" fn hsa_amd_profiling_set_profiler_enabled(
    queue: *mut HsaQueue,
    enable: i32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if !queue_known(runtime, queue) {
            return INVALID_QUEUE;
        }
        let queue = hardware_queue_key(runtime, queue).unwrap_or(queue as usize);
        // SAFETY: Queue validation proves the public control mapping is live.
        let properties = unsafe {
            let address = AqlControlRegion::properties_address_at(queue);
            &*(address as *const AtomicU32)
        };
        if enable == 0 {
            properties.fetch_and(!AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
        } else {
            properties.fetch_or(AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the requested queue property or native status.
pub unsafe extern "C" fn hsa_amd_queue_get_info(
    queue: *mut HsaQueue,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        let input_key = queue as usize;
        if runtime
            .queue_state
            .released_counted_queues
            .contains(&input_key)
        {
            return INVALID_ARGUMENT;
        }
        if let Some(record) = runtime.queue_state.sdma_queues.get(&input_key) {
            let info = record.native.info();
            // SAFETY: Each supported attribute writes its public C type.
            unsafe {
                match attribute {
                    AMD_QUEUE_INFO_AGENT => value.cast::<HsaAgent>().write(record.agent),
                    AMD_QUEUE_INFO_DOORBELL_ID => {
                        value.cast::<u64>().write(info.doorbell_host_address as u64)
                    }
                    QUEUE_INFO_USE_COUNT => value.cast::<u32>().write(u32::MAX),
                    QUEUE_INFO_HW_ID => value.cast::<u32>().write(record.hardware_id),
                    AMD_QUEUE_INFO_ENGINE_TYPE => {
                        value.cast::<u32>().write(u32::from(AMD_QUEUE_ENGINE_SDMA));
                    }
                    AMD_QUEUE_INFO_SDMA_ENGINE_ID => {
                        let Some(engine_id) = info.sdma_engine_id else {
                            return ERROR;
                        };
                        value.cast::<u32>().write(engine_id);
                    }
                    AMD_QUEUE_INFO_READ_POINTER => {
                        value.cast::<u64>().write(info.read_index_device_address);
                    }
                    AMD_QUEUE_INFO_WRITE_POINTER => {
                        value.cast::<u64>().write(info.write_index_device_address);
                    }
                    _ => return INVALID_ARGUMENT,
                }
            }
            return SUCCESS;
        }
        let counted = runtime.queue_state.counted_queues.get(&input_key);
        let hardware_key = runtime
            .queue_state
            .intercept_queues
            .get(&input_key)
            .map_or_else(
                || counted.map_or(input_key, |queue| queue.hardware_queue),
                |queue| queue.lower,
            );
        let Some(record) = runtime.queue_state.queues.get(&hardware_key) else {
            return INVALID_QUEUE;
        };
        // SAFETY: Each match arm writes the public type for the queried attribute.
        unsafe {
            match attribute {
                AMD_QUEUE_INFO_AGENT => value.cast::<HsaAgent>().write(record.agent),
                AMD_QUEUE_INFO_DOORBELL_ID => value
                    .cast::<u64>()
                    .write(record.native.info().doorbell_host_address as u64),
                QUEUE_INFO_USE_COUNT => {
                    let pool_key =
                        counted.map_or(record.counted_pool_key, |queue| Some(queue.pool_key));
                    let use_count = if let Some(pool_key) = pool_key {
                        runtime
                            .queue_state
                            .counted_queue_pools
                            .get(&pool_key)
                            .and_then(|pool| pool.iter().find(|entry| entry.queue == hardware_key))
                            .map(|entry| entry.use_count)
                            .filter(|count| *count != 0)
                            .ok_or(INVALID_ARGUMENT)
                    } else {
                        Ok(u32::MAX)
                    };
                    match use_count {
                        Ok(use_count) => value.cast::<u32>().write(use_count),
                        Err(status) => return status,
                    }
                }
                QUEUE_INFO_HW_ID => value.cast::<u32>().write(record.hardware_id),
                AMD_QUEUE_INFO_PREFETCH_DISPATCH_MAJOR
                | AMD_QUEUE_INFO_PREFETCH_DISPATCH_MINOR
                | AMD_QUEUE_INFO_PREFETCH_BARRIER_MAJOR
                | AMD_QUEUE_INFO_PREFETCH_BARRIER_MINOR => value.cast::<u8>().write(u8::MAX),
                AMD_QUEUE_INFO_PREFETCH_RING_BUFFER => value.cast::<u64>().write(0),
                AMD_QUEUE_INFO_PROPERTIES => value.cast::<[u8; 8]>().write([0; 8]),
                AMD_QUEUE_INFO_VM_FAULT_STATUS => value.cast::<bool>().write(record.vm_faulted),
                AMD_QUEUE_INFO_VM_FAULT_ADDRESS => {
                    value.cast::<u64>().write(record.vm_fault_address)
                }
                AMD_QUEUE_INFO_VM_FAULT_REASON => value.cast::<u32>().write(record.vm_fault_reason),
                AMD_QUEUE_INFO_ENGINE_TYPE => value
                    .cast::<u32>()
                    .write(u32::from(AMD_QUEUE_ENGINE_COMPUTE)),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_signal_external_semaphore")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Enqueues a signal operation for an external semaphore.
pub unsafe extern "C" fn hsa_amd_queue_signal_external_semaphore(
    queue: *mut HsaQueue,
    semaphore: HsaAmdExternalSemaphore,
    _value: u64,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if queue.is_null() {
            return INVALID_QUEUE;
        }
        if semaphore.handle == 0 {
            return INVALID_ARGUMENT;
        }
        if hardware_queue_key(runtime, queue).is_none() {
            return INVALID_QUEUE;
        }
        NOT_SUPPORTED
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_wait_external_semaphore")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits for queue external semaphore under the requested wait policy.
pub unsafe extern "C" fn hsa_amd_queue_wait_external_semaphore(
    queue: *mut HsaQueue,
    semaphore: HsaAmdExternalSemaphore,
    _value: u64,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if queue.is_null() {
            return INVALID_QUEUE;
        }
        if semaphore.handle == 0 {
            return INVALID_ARGUMENT;
        }
        if hardware_queue_key(runtime, queue).is_none() {
            return INVALID_QUEUE;
        }
        NOT_SUPPORTED
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_profiling_get_dispatch_time")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes the recorded start and end timestamps of a dispatch.
pub unsafe extern "C" fn hsa_amd_profiling_get_dispatch_time(
    agent: HsaAgent,
    signal: HsaSignal,
    time: *mut ProfilingTime,
) -> Status {
    boundary(|| {
        if time.is_null() {
            return INVALID_ARGUMENT;
        }
        let (device, _call, start, end) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            if !runtime.signal_state.owns_signal(signal) {
                return INVALID_SIGNAL;
            }
            // SAFETY: owns_signal validated the handle while the runtime lock
            // prevents destruction of its backing storage.
            let Some((start, end)) = (unsafe {
                crate::signal::with_signal(signal, |signal| {
                    (
                        signal.start_ts.load(Ordering::Acquire),
                        signal.end_ts.load(Ordering::Acquire),
                    )
                })
            }) else {
                return INVALID_SIGNAL;
            };
            let Some(token) = runtime.inflight.enter() else {
                return OUT_OF_RESOURCES;
            };
            (runtime.gpus[index].device.clone(), token, start, end)
        };
        let (start, end) = match crate::runtime::translate_gpu_interval(&device, start, end) {
            Ok(interval) => interval,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { time.write(ProfilingTime { start, end }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_profiling_get_async_copy_time")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Writes timestamps recorded for an asynchronous copy.
pub unsafe extern "C" fn hsa_amd_profiling_get_async_copy_time(
    signal: HsaSignal,
    time: *mut ProfilingTime,
) -> Status {
    boundary(|| {
        let (profile, device) = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if time.is_null() {
                return INVALID_ARGUMENT;
            }
            if !runtime.signal_state.owns_signal(signal) {
                return INVALID_SIGNAL;
            }
            // SAFETY: owns_signal validated the handle while the registry
            // lock prevents destruction of its backing storage.
            if !unsafe {
                crate::signal::with_signal(signal, |signal| {
                    signal.value.load(Ordering::Acquire) <= 0
                })
            }
            .unwrap_or(false)
            {
                return ERROR;
            }
            let Some(profile) = runtime.signal_state.copy_profile(signal) else {
                return ERROR;
            };
            let device = match profile.clock {
                AsyncCopyClock::System => None,
                AsyncCopyClock::Gpu(index) => {
                    let Some(gpu) = runtime.gpus.get(index) else {
                        return ERROR;
                    };
                    let Some(token) = runtime.inflight.enter() else {
                        return OUT_OF_RESOURCES;
                    };
                    Some((gpu.device.clone(), token))
                }
            };
            (profile, device)
        };
        let (start, end) = if let Some((device, _call)) = device {
            match crate::runtime::translate_gpu_interval(&device, profile.start, profile.end) {
                Ok(interval) => interval,
                Err(status) => return status,
            }
        } else {
            (profile.start, profile.end)
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { time.write(ProfilingTime { start, end }) };
        SUCCESS
    })
}
