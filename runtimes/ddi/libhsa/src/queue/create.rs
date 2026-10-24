// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native queue creation and publication transactions.

use super::*;

/// Preserves one cooperative queue per agent while its native creation runs.
static COOPERATIVE_CREATE_GATE: Mutex<()> = Mutex::new(());

/// # Safety
/// `queue` must point to writable output storage for the complete call. The
/// callback and its data must remain valid until native queue teardown and
/// completion of the final event callback.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn create_hardware_queue(
    expected_generation: Option<u64>,
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    priority: QueuePriority,
    callback: QueueErrorCallback,
    data: CallbackArg,
    private_segment_size: u32,
    ring_memory: QueueRingMemory,
    cu_mask: Option<Vec<u32>>,
    queue: *mut *mut HsaQueue,
    created_log: &mut Option<(u64, usize)>,
) -> Status {
    struct Prepared {
        generation: u64,
        id: u64,
        device: Device,
        gpu: GpuInfo,
        event: Arc<QueueEventOwner>,
        host_page_size: usize,
        stop: Arc<AtomicBool>,
        tracker: Arc<crate::runtime::InFlightTracker>,
        _token: crate::runtime::InFlightToken,
    }

    let cooperative = queue_type == QUEUE_TYPE_COOPERATIVE;
    let _cooperative_gate = if cooperative {
        match COOPERATIVE_CREATE_GATE.lock() {
            Ok(guard) => Some(guard),
            Err(_) => return ERROR,
        }
    } else {
        None
    };
    let (size, callback, data, private_segment_size) = if cooperative {
        (
            16_384 / AQL_PACKET_BYTES as u32,
            None,
            // SAFETY: A null callback argument has no referent to retain.
            unsafe { CallbackArg::new(std::ptr::null_mut()) },
            0,
        )
    } else {
        (size.max(64), callback, data, private_segment_size)
    };
    let Some(ring_size_bytes) = u64::from(size).checked_mul(AQL_PACKET_BYTES as u64) else {
        return INVALID_QUEUE_CREATION;
    };
    let prepared = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(generation) = guard.active_generation() else {
            return NOT_INITIALIZED;
        };
        if expected_generation.is_some_and(|expected| expected != generation) {
            return INVALID_RUNTIME_STATE;
        }
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        if cooperative {
            if runtime.gpus[index].info.gws_count == 0 {
                return INVALID_QUEUE_CREATION;
            }
            if runtime
                .queue_state
                .cooperative_teardown
                .contains(&agent.handle)
            {
                return OUT_OF_RESOURCES;
            }
            if let Some((public, record)) = runtime
                .queue_state
                .queues
                .iter_mut()
                .find(|(_, record)| record.agent == agent && record.cooperative_refs != 0)
            {
                if record.inactivated
                    || record.teardown_started
                    || !record.event_alive.load(Ordering::Acquire)
                {
                    return OUT_OF_RESOURCES;
                }
                if ring_memory != record.ring_memory {
                    return INVALID_QUEUE_CREATION;
                }
                let Some(refs) = record.cooperative_refs.checked_add(1) else {
                    return OUT_OF_RESOURCES;
                };
                if priority != QueuePriority::Normal {
                    if let Err(error) = record.native.set_priority(priority) {
                        return map_error(error);
                    }
                }
                if let Some(mask) = cu_mask.as_ref() {
                    if let Err(error) = record.native.set_cu_mask(mask) {
                        return map_error(error);
                    }
                    record.cu_mask.clone_from(mask);
                }
                record.cooperative_refs = refs;
                // SAFETY: The shared public queue remains in the registry.
                unsafe { queue.write(*public as *mut HsaQueue) };
                return SUCCESS;
            }
        }
        let id = match runtime.queue_state.allocate_id() {
            Ok(id) => id,
            Err(status) => return status,
        };
        if private_segment_size != u32::MAX && private_segment_size > MAX_PRIVATE_SEGMENT_BYTES {
            return OUT_OF_RESOURCES;
        }
        let event = match QueueEventSignal::shared_event(runtime) {
            Ok(event) => event,
            Err(status) => return status,
        };
        if runtime.queue_state.queues.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let tracker = runtime.inflight.clone();
        let Some(token) = tracker.enter() else {
            return OUT_OF_RESOURCES;
        };
        Prepared {
            generation,
            id,
            device: runtime.gpus[index].device.clone(),
            gpu: runtime.gpus[index].info,
            event,
            host_page_size: runtime.host_page_size,
            stop: runtime.stop_workers.clone(),
            tracker,
            _token: token,
        }
    };

    let inactive_signal = match QueueEventSignal::create(
        prepared.event.clone(),
        &prepared.device,
        prepared.host_page_size,
    ) {
        Ok(signal) => Arc::new(signal),
        Err(status) => return status,
    };
    let (scratch_allocation, scratch) =
        if private_segment_size == 0 || private_segment_size == u32::MAX {
            (None, None)
        } else {
            match allocate_scratch(
                &prepared.device,
                prepared.gpu,
                private_segment_size,
                prepared.gpu.wavefront_size,
            ) {
                Ok((allocation, scratch)) => (Some(allocation), Some(scratch)),
                Err(status) => return status,
            }
        };
    let mut doorbell = Box::new(AmdSignal::doorbell(0, 0));
    let event_alive = Arc::new(AtomicBool::new(true));
    let worker_alive = event_alive.clone();
    let worker_signal = inactive_signal.clone();
    let saved_cu_mask = cu_mask.unwrap_or_default();
    let mut creation_guard = UnpublishedQueueGuard::new(&prepared.tracker);
    let native = match prepared.device.gpu().and_then(|gpu| {
        // SAFETY: The owner tuple retains all external GPU addresses
        // through native creation, including an ambiguous CREATE result.
        unsafe {
            gpu.create_queue_with_dependencies(
                QueueRequest {
                    ring_size_bytes,
                    parameters: QueueParameters::Aql {
                        producer_mode: if queue_type == QUEUE_TYPE_SINGLE {
                            QueueProducerMode::Single
                        } else {
                            QueueProducerMode::Multiple
                        },
                        ring_memory,
                        global_work_sync: cooperative,
                        inactive_signal: Some(inactive_signal.handle()),
                        error_event: Some(inactive_signal.error_event()),
                        scratch,
                    },
                    priority,
                    device_producer: false,
                },
                QueueBacking {
                    inactive_signal,
                    scratch: scratch_allocation,
                    uncertain_scratch: Vec::new(),
                },
            )
        }
    }) {
        Ok(created) => created,
        Err(error) => {
            if !matches!(error, rocddi::Error::QueueBackingMayBeLive { .. }) {
                creation_guard.disarm();
            }
            return if cooperative && error.kind() == rocddi::ErrorKind::Busy {
                OUT_OF_RESOURCES
            } else if error.kind() == rocddi::ErrorKind::Unsupported {
                INVALID_QUEUE_CREATION
            } else {
                map_error(error)
            };
        }
    };
    // The native owner retains its external backing if setup unwinds.
    let mut pending = native;
    if !saved_cu_mask.is_empty() {
        if let Err(error) = pending.set_cu_mask(&saved_cu_mask) {
            return abandon_unpublished_retained(pending, map_error(error), &mut creation_guard);
        }
    }
    let info = pending.info();
    let Some(control) = info.aql_control else {
        return abandon_unpublished_retained(pending, INVALID_QUEUE_CREATION, &mut creation_guard);
    };
    if control.alignment() < 256
        || control.writable_prefix_bytes() < std::mem::size_of::<HsaQueue>()
        || control.read_index_host_address() != info.read_index_host_address
        || control.write_index_host_address() != info.write_index_host_address
    {
        return abandon_unpublished_retained(pending, ERROR, &mut creation_guard);
    }
    let public = control.host_base();
    *doorbell = AmdSignal::doorbell(info.doorbell_host_address, public);
    let doorbell_handle = (&raw mut *doorbell) as usize as u64;
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => {
            return abandon_unpublished_retained(pending, status, &mut creation_guard);
        }
    };
    if guard.active_generation() != Some(prepared.generation) {
        drop(guard);
        return abandon_unpublished_retained(pending, INVALID_RUNTIME_STATE, &mut creation_guard);
    }
    let Some(runtime) = guard.as_mut() else {
        drop(guard);
        return abandon_unpublished_retained(pending, NOT_INITIALIZED, &mut creation_guard);
    };
    if runtime.queue_state.queues.try_reserve(1).is_err()
        || runtime.queue_state.queues.contains_key(&public)
    {
        drop(guard);
        return abandon_unpublished_retained(pending, OUT_OF_RESOURCES, &mut creation_guard);
    }
    // SAFETY: The public control pointer is unique in the registry, and
    // rocddi retains this AQL control allocation until native teardown.
    unsafe {
        let header = public as *mut HsaQueue;
        (*header).queue_type = queue_type;
        (*header).features = QUEUE_FEATURE_KERNEL_DISPATCH;
        (*header).base_address = info.ring_host_address as *mut c_void;
        (*header).doorbell_signal = HsaSignal {
            handle: doorbell_handle,
        };
        (*header).size = size;
        (*header).reserved = 0;
        (*header).id = prepared.id;
    }
    runtime.queue_state.queues.insert(
        public,
        Queue {
            native: pending,
            public_handle: public,
            scratch_event_visibility: ScratchEventVisibility::Unpublished,
            _doorbell: doorbell,
            event_alive,
            event_worker: None,
            agent,
            hardware_id: u32::try_from(prepared.id).unwrap_or(u32::MAX),
            counted_pool_key: None,
            cooperative_refs: u32::from(cooperative),
            ring_memory,
            inactivated: false,
            cu_mask: saved_cu_mask,
            callback,
            callback_data: data,
            vm_faulted: false,
            vm_fault_address: 0,
            vm_fault_reason: 0,
            teardown_started: false,
        },
    );
    runtime.queue_state.released_counted_queues.remove(&public);
    let event_worker = thread::Builder::new()
        .name("rocddi-queue-events".into())
        .spawn(move || queue_event_worker(&prepared.stop, &worker_alive, &worker_signal, public));
    let Ok(event_worker) = event_worker else {
        let Some(mut record) = runtime.queue_state.queues.remove(&public) else {
            prepared.tracker.quarantine();
            return ERROR;
        };
        drop(guard);
        if destroy_runtime_queue(&mut record).is_err() {
            std::mem::forget(record);
        } else {
            creation_guard.disarm();
        }
        return OUT_OF_RESOURCES;
    };
    let Some(record) = runtime.queue_state.queues.get_mut(&public) else {
        prepared.tracker.quarantine();
        return ERROR;
    };
    record.event_worker = Some(event_worker);
    record.scratch_event_visibility = ScratchEventVisibility::Published;
    // SAFETY: The caller supplied writable output storage, and the registry
    // now owns the complete queue and its event worker.
    unsafe { queue.write(public as *mut HsaQueue) };
    *created_log = Some((prepared.id, public));
    creation_guard.disarm();
    SUCCESS
}

/// Quarantines an HSA generation if native creation unwinds or leaves an
/// unpublished queue whose cleanup cannot be proven complete.
struct UnpublishedQueueGuard<'a> {
    /// In-flight tracker retained until queue publication or rollback.
    tracker: &'a crate::runtime::InFlightTracker,
    /// Whether dropping this guard must retire an unpublished queue.
    armed: bool,
}

impl<'a> UnpublishedQueueGuard<'a> {
    /// Arms rollback for an unpublished queue and retains its tracker.
    fn new(tracker: &'a crate::runtime::InFlightTracker) -> Self {
        Self {
            tracker,
            armed: true,
        }
    }

    /// Prevents this creation guard from retiring the published queue.
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for UnpublishedQueueGuard<'_> {
    /// Quarantines the native generation if publication never disarmed rollback.
    fn drop(&mut self) {
        if self.armed {
            self.tracker.quarantine();
        }
    }
}

/// Releases a queue that failed before it entered the public registry.
fn abandon_unpublished_tracked<D: 'static>(
    native: rocddi::device::gpu::queue::Queue,
    dependencies: D,
    status: Status,
    guard: &mut UnpublishedQueueGuard<'_>,
) -> Status {
    // SAFETY: No public producer has observed the queue. The dependencies
    // retain every external GPU address until native teardown succeeds.
    match unsafe { native.abandon_unpublished_with_dependencies(dependencies) } {
        Ok(()) => {
            guard.disarm();
            status
        }
        Err(error) => map_error(error),
    }
}

/// Releases retained queue resources after a failed publication.
fn abandon_unpublished_retained(
    native: RetainedQueue<QueueBacking>,
    status: Status,
    guard: &mut UnpublishedQueueGuard<'_>,
) -> Status {
    // SAFETY: No public producer has observed the queue. The native owner
    // retains all external GPU addresses through uncertain teardown.
    match unsafe { native.abandon_unpublished() } {
        Ok(()) => {
            guard.disarm();
            status
        }
        Err(error) => map_error(error),
    }
}

/// Validates the requested SDMA engine against the queue transport.
fn validated_sdma_engine(info: QueueTransport, size: u32) -> Option<u32> {
    let engine_id = info.sdma_engine_id?;
    (info.ring_size_bytes == u64::from(size)
        && info.ring_host_address != 0
        && info.index_unit_bytes == 1
        && !info.read_index_wraps
        && info.read_index_width == QueueAccessWidth::Bits64
        && info.write_index_width == QueueAccessWidth::Bits64
        && info.doorbell_width == QueueAccessWidth::Bits64
        && info.read_index_host_address != 0
        && info.read_index_host_address % std::mem::align_of::<AtomicU64>() == 0
        && info.write_index_host_address != 0
        && info.write_index_host_address % std::mem::align_of::<AtomicU64>() == 0
        && info.read_index_host_address != info.write_index_host_address
        && info.read_index_device_address != 0
        && info.write_index_device_address != 0
        && info.doorbell_host_address != 0
        && info.doorbell_host_address % std::mem::align_of::<u64>() == 0
        && info
            .doorbell_device_address
            .is_some_and(|address| address != 0))
    .then_some(engine_id)
}

/// # Safety
/// `queue` addresses writable output storage. The caller serializes native
/// queue publication and destruction with all producers.
pub(super) unsafe fn create_sdma_queue(
    expected_generation: Option<u64>,
    agent: HsaAgent,
    size_bytes: u32,
    selection: SdmaEngineSelection,
    ring_memory: QueueRingMemory,
    queue: *mut *mut HsaQueue,
    created_log: &mut Option<(u64, usize, u32)>,
) -> Status {
    let (generation, id, device, tracker, _token) = {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(generation) = guard.active_generation() else {
            return NOT_INITIALIZED;
        };
        if expected_generation.is_some_and(|expected| expected != generation) {
            return INVALID_RUNTIME_STATE;
        }
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        let id = match runtime.queue_state.allocate_id() {
            Ok(id) => id,
            Err(status) => return status,
        };
        if runtime.queue_state.sdma_queues.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let tracker = runtime.inflight.clone();
        let Some(token) = tracker.enter() else {
            return OUT_OF_RESOURCES;
        };
        (
            generation,
            id,
            runtime.gpus[index].device.clone(),
            tracker,
            token,
        )
    };
    let mut control = Box::new(SdmaQueueControl {
        read_index_host_address: 0,
        write_index_host_address: 0,
        _padding: [0; SDMA_QUEUE_HANDLE_BIT - 2 * std::mem::size_of::<usize>()],
        header: [0; std::mem::size_of::<HsaQueue>()],
    });
    let mut doorbell = Box::new(AmdSignal::doorbell(0, 0));
    let mut creation_guard = UnpublishedQueueGuard::new(&tracker);
    let native = match device.gpu().and_then(|gpu| {
        // SAFETY: SDMA creation passes no external GPU addresses. The queue
        // remains unpublished until its transport has been validated.
        unsafe {
            gpu.create_queue(QueueRequest {
                ring_size_bytes: u64::from(size_bytes),
                parameters: QueueParameters::SdmaByEngine {
                    selection,
                    ring_memory,
                },
                priority: QueuePriority::Normal,
                device_producer: true,
            })
        }
    }) {
        Ok(native) => native,
        Err(error) if error.kind() == rocddi::ErrorKind::Unsupported => {
            creation_guard.disarm();
            return INVALID_QUEUE_CREATION;
        }
        Err(error) => {
            if !matches!(error, rocddi::Error::QueueBackingMayBeLive { .. }) {
                creation_guard.disarm();
            }
            return map_error(error);
        }
    };
    let info = native.info();
    let Some(engine_id) = validated_sdma_engine(info, size_bytes) else {
        return abandon_unpublished_tracked(
            native,
            (),
            INVALID_QUEUE_CREATION,
            &mut creation_guard,
        );
    };
    let public = control.header.as_mut_ptr().cast::<HsaQueue>();
    let public_key = public as usize;
    control.read_index_host_address = info.read_index_host_address;
    control.write_index_host_address = info.write_index_host_address;
    *doorbell = AmdSignal::doorbell(info.doorbell_host_address, public_key);
    // SAFETY: The public header has HsaQueue alignment and remains owned by
    // the SDMA record until successful native queue destruction.
    unsafe {
        public.write(HsaQueue {
            queue_type: QUEUE_TYPE_SINGLE,
            features: 0,
            base_address: info.ring_host_address as *mut c_void,
            doorbell_signal: HsaSignal {
                handle: (&raw mut *doorbell) as usize as u64,
            },
            size: size_bytes,
            reserved: 0,
            id,
        });
    }
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return abandon_unpublished_tracked(native, (), status, &mut creation_guard),
    };
    if guard.active_generation() != Some(generation) {
        drop(guard);
        return abandon_unpublished_tracked(native, (), INVALID_RUNTIME_STATE, &mut creation_guard);
    }
    let Some(runtime) = guard.as_mut() else {
        drop(guard);
        return abandon_unpublished_tracked(native, (), NOT_INITIALIZED, &mut creation_guard);
    };
    if runtime.queue_state.sdma_queues.try_reserve(1).is_err()
        || runtime.queue_state.sdma_queues.contains_key(&public_key)
    {
        drop(guard);
        return abandon_unpublished_tracked(native, (), OUT_OF_RESOURCES, &mut creation_guard);
    }
    runtime.queue_state.sdma_queues.insert(
        public_key,
        SdmaQueue {
            native,
            _control: control,
            _doorbell: doorbell,
            agent,
            hardware_id: u32::try_from(id).unwrap_or(u32::MAX),
            teardown_started: false,
        },
    );
    // SAFETY: The registry owns the queue before the caller sees this header.
    unsafe { queue.write(public) };
    *created_log = Some((id, public_key, engine_id));
    creation_guard.disarm();
    SUCCESS
}
