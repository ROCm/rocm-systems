//! HSA queue objects, public ring layout, scratch, and error-event delivery.
//!
//! Hardware queues retain their rocddi queue, ring allocation, index storage,
//! doorbell signal, scratch backing, and event-worker dependencies as one
//! teardown unit. Counted queues add shared acquisition accounting without
//! changing the public `hsa_queue_t` layout. Soft queues implement only the
//! host-visible index semantics that do not require a native GPU queue.
//!
//! Atomic accessors operate directly on ABI-defined fields, so their offsets
//! and memory orderings are part of the compatibility contract. Destruction
//! first prevents new observation, then stops workers, then releases native
//! resources; partial failure keeps enough state for a safe retry.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use rocddi::device::Device;
use rocddi::gpu::queue::{
    QueueErrorEvent, QueueParameters, QueuePriority, QueueProducerMode, QueueRequest, QueueScratch,
};
use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};
use rocddi::topology::GpuInfo;

use crate::ffi::*;
use crate::runtime::{boundary, initialized_mut, lock, map_error};
use crate::signal::AmdSignal;

const WRITE_INDEX_OFFSET: usize = 56;
const READ_INDEX_OFFSET: usize = 128;
const QUEUE_PROPERTIES_OFFSET: usize = 180;
const AQL_PACKET_BYTES: usize = 64;
const AQL_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
const AQL_PACKET_TYPE_MASK: u16 = 0xff;
const AQL_PRIVATE_SEGMENT_SIZE_OFFSET: usize = 24;
const MAX_PRIVATE_SEGMENT_BYTES: u32 = 262_128;
const SCRATCH_ALIGNMENT: u64 = 256;
const GPU_PAGE_BYTES: u64 = 4096;

/// Dedicated device-visible signals used to stop a queue and report errors.
struct QueueEventSignal {
    _allocation: Allocation,
    shared_event: Arc<QueueSharedEvent>,
    inactive_address: usize,
    error_address: usize,
}

/// Process-shared KFD event and mailbox backing queue event signals.
pub(crate) struct QueueSharedEvent {
    event: rocddi::gpu::event::linux::SignalEvent,
    mailbox: usize,
    event_id: u32,
}

impl QueueEventSignal {
    fn create(runtime: &mut crate::runtime::Runtime, device_index: usize) -> Result<Self, Status> {
        let event = if let Some(event) = &runtime.queue_event {
            event.clone()
        } else {
            let (event, mailbox, event_id) =
                runtime.create_signal_event().ok_or(OUT_OF_RESOURCES)?;
            let event = Arc::new(QueueSharedEvent {
                event,
                mailbox,
                event_id,
            });
            runtime.queue_event = Some(event.clone());
            event
        };
        let allocation = runtime.gpus[device_index]
            .device
            .allocate(
                MemoryKind::System,
                GPU_PAGE_BYTES,
                GPU_PAGE_BYTES,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(map_error)?;
        let info = allocation.info();
        let address = info.host_address.ok_or(OUT_OF_RESOURCES)?;
        if info.device_address != address as u64 {
            return Err(OUT_OF_RESOURCES);
        }
        let error_address = address + std::mem::size_of::<AmdSignal>();
        // SAFETY: The allocation is writable, naturally page-aligned, and has
        // space for two disjoint signals retained through native destruction.
        unsafe {
            (address as *mut AmdSignal).write(AmdSignal::interrupt(
                0,
                event.mailbox,
                event.event_id,
            ));
            (error_address as *mut AmdSignal).write(AmdSignal::interrupt(
                0,
                event.mailbox,
                event.event_id,
            ));
        }
        Ok(Self {
            _allocation: allocation,
            shared_event: event,
            inactive_address: address,
            error_address,
        })
    }

    fn handle(&self) -> u64 {
        self.inactive_address as u64
    }

    fn error_event(&self) -> QueueErrorEvent {
        self.shared_event
            .event
            .queue_error_event((self.error_address + std::mem::offset_of!(AmdSignal, value)) as u64)
    }

    fn load_inactive(&self) -> SignalValue {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.inactive_address as *const AmdSignal) }
            .value
            .load(Ordering::Acquire)
    }

    fn load_error(&self) -> SignalValue {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.error_address as *const AmdSignal) }
            .value
            .load(Ordering::Acquire)
    }

    fn release_queue(&self) {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.inactive_address as *const AmdSignal) }
            .value
            .store(0, Ordering::Release);
    }

    fn release_error(&self) {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.error_address as *const AmdSignal) }
            .value
            .store(0, Ordering::Release);
    }
}

/// One directly published hardware queue and all teardown dependencies.
///
/// `teardown_started` makes destruction resumable: once native inactivation
/// succeeds, a later retry proceeds directly to final queue destruction.
pub(crate) struct Queue {
    pub(crate) native: rocddi::gpu::queue::Queue,
    _doorbell: Box<AmdSignal>,
    inactive_signal: Arc<QueueEventSignal>,
    event_alive: Arc<AtomicBool>,
    event_worker: Option<JoinHandle<()>>,
    scratch: Option<Allocation>,
    agent: HsaAgent,
    hardware_id: u32,
    counted_pool_key: Option<(u64, u32)>,
    cu_mask: Vec<u32>,
    callback: QueueErrorCallback,
    callback_data: usize,
    teardown_started: bool,
}

fn stop_queue_event_worker(alive: &AtomicBool, worker: &mut Option<JoinHandle<()>>) {
    alive.store(false, Ordering::Release);
    if let Some(worker) = worker.take() {
        let _ = worker.join();
    }
}

pub(crate) fn destroy_runtime_queue(queue: &mut Queue) -> Result<(), rocddi::Error> {
    stop_queue_event_worker(&queue.event_alive, &mut queue.event_worker);
    if !queue.teardown_started {
        queue.native.inactivate()?;
        queue.teardown_started = true;
    }
    queue.native.destroy()
}

// Native and soft queue controls expose 128-byte-aligned public handles.
// Put counted handles 64 bytes past that boundary so index operations can
// identify them without consulting the registry or a thread-local cache.
const COUNTED_QUEUE_HANDLE_BIT: usize = 64;

#[repr(C, align(128))]
/// Stable storage for the public prefix copied from a shared hardware queue.
struct CountedQueuePublic {
    hardware_queue: usize,
    _padding: [u8; COUNTED_QUEUE_HANDLE_BIT - std::mem::size_of::<usize>()],
    header: [u8; std::mem::size_of::<HsaQueue>()],
}

const _: () = assert!(std::mem::offset_of!(CountedQueuePublic, header) == COUNTED_QUEUE_HANDLE_BIT);

/// Logical counted-queue handle borrowing one pooled hardware queue.
pub(crate) struct CountedQueue {
    public: Box<CountedQueuePublic>,
    pub(crate) hardware_queue: usize,
    pub(crate) pool_key: (u64, u32),
}

impl CountedQueue {
    unsafe fn new(hardware_queue: usize, pool_key: (u64, u32)) -> Self {
        // SAFETY: The caller retains the native queue and its initialized
        // public header for the complete lifetime of this logical handle.
        let source = unsafe { &*(hardware_queue as *const HsaQueue) };
        let mut public = Box::new(CountedQueuePublic {
            hardware_queue,
            _padding: [0; COUNTED_QUEUE_HANDLE_BIT - std::mem::size_of::<usize>()],
            header: [0; std::mem::size_of::<HsaQueue>()],
        });
        // SAFETY: The header starts at a 64-byte-aligned offset and has not
        // been initialized as another type.
        unsafe {
            public
                .header
                .as_mut_ptr()
                .cast::<HsaQueue>()
                .write(HsaQueue {
                    queue_type: source.queue_type,
                    features: source.features,
                    base_address: source.base_address,
                    doorbell_signal: source.doorbell_signal,
                    size: source.size,
                    reserved: source.reserved,
                    id: source.id,
                });
        }
        Self {
            public,
            hardware_queue,
            pool_key,
        }
    }

    fn public_pointer(&mut self) -> *mut HsaQueue {
        self.public.header.as_mut_ptr().cast()
    }
}

/// Hardware queue retained by a counted-queue pool with its live borrower count.
pub(crate) struct CountedHardwareQueue {
    pub(crate) queue: usize,
    pub(crate) use_count: u32,
}

#[repr(C, align(64))]
/// Aligned packet storage for a host-only soft queue.
struct SoftPacket([u8; AQL_PACKET_BYTES]);

/// Host-only queue whose indices remain valid without a native GPU transport.
pub(crate) struct SoftQueue {
    _control: Box<SoftQueueControl>,
    _ring: Vec<SoftPacket>,
    active: bool,
}

#[repr(C, align(128))]
struct SoftQueueControl([u64; 32]);

impl SoftQueue {
    fn inactivate(&mut self) {
        self.active = false;
    }
}

/// Validated scratch allocation geometry derived from one queue request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScratchPlan {
    byte_length: u64,
    maximum_private_segment_byte_length: u32,
    maximum_wave_count: u32,
}

fn scratch_plan(
    gpu: GpuInfo,
    private_segment_size: u32,
    lanes_per_wave: u32,
) -> Result<ScratchPlan, Status> {
    if private_segment_size == 0 || private_segment_size > MAX_PRIVATE_SEGMENT_BYTES {
        return Err(OUT_OF_RESOURCES);
    }
    if (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping) != (12, 0, 1)
        || gpu.wavefront_size != 32
        || gpu.xcc_count != 1
        || !matches!(lanes_per_wave, 32 | 64)
    {
        return Err(OUT_OF_RESOURCES);
    }
    let lane_alignment = u32::try_from(SCRATCH_ALIGNMENT)
        .ok()
        .and_then(|alignment| alignment.checked_div(lanes_per_wave))
        .filter(|alignment| *alignment != 0)
        .ok_or(OUT_OF_RESOURCES)?;
    let aligned_private = private_segment_size
        .checked_add(lane_alignment - 1)
        .map(|size| size / lane_alignment * lane_alignment)
        .ok_or(OUT_OF_RESOURCES)?;
    let engines = gpu
        .shader_engine_count_per_xcc
        .checked_mul(gpu.xcc_count)
        .filter(|engines| *engines != 0)
        .ok_or(OUT_OF_RESOURCES)?;
    let maximum_waves = gpu
        .compute_unit_count
        .checked_mul(gpu.maximum_scratch_wave_count_per_compute_unit)
        .ok_or(OUT_OF_RESOURCES)?;
    let wave_bytes = u64::from(aligned_private)
        .checked_mul(u64::from(lanes_per_wave))
        .ok_or(OUT_OF_RESOURCES)?;
    let representable_bytes = u64::from(u32::MAX) / GPU_PAGE_BYTES * GPU_PAGE_BYTES;
    let representable_waves = u32::try_from(representable_bytes / wave_bytes)
        .unwrap_or(u32::MAX)
        .min(maximum_waves);
    let maximum_wave_count = representable_waves / engines * engines;
    if maximum_wave_count == 0 {
        return Err(OUT_OF_RESOURCES);
    }
    let required = wave_bytes
        .checked_mul(u64::from(maximum_wave_count))
        .ok_or(OUT_OF_RESOURCES)?;
    let byte_length = required
        .checked_add(GPU_PAGE_BYTES - 1)
        .map(|size| size / GPU_PAGE_BYTES * GPU_PAGE_BYTES)
        .filter(|size| u32::try_from(*size).is_ok())
        .ok_or(OUT_OF_RESOURCES)?;
    Ok(ScratchPlan {
        byte_length,
        maximum_private_segment_byte_length: aligned_private,
        maximum_wave_count,
    })
}

fn allocate_scratch(
    device: &Device,
    gpu: GpuInfo,
    private_segment_size: u32,
    lanes_per_wave: u32,
) -> Result<(Allocation, QueueScratch), Status> {
    let plan = scratch_plan(gpu, private_segment_size, lanes_per_wave)?;
    let allocation = device
        .gpu()
        .and_then(|gpu| gpu.allocate_queue_scratch(plan.byte_length))
        .map_err(map_error)?;
    let scratch = QueueScratch {
        device_address: allocation.info().device_address,
        byte_length: plan.byte_length,
        maximum_private_segment_byte_length: plan.maximum_private_segment_byte_length,
        maximum_wave_count: plan.maximum_wave_count,
    };
    Ok((allocation, scratch))
}

fn dispatch_private_segment(queue: &Queue) -> Option<u32> {
    let info = queue.native.info();
    let packet_count = info.ring_size_bytes / AQL_PACKET_BYTES as u64;
    if packet_count == 0 || !packet_count.is_power_of_two() {
        return None;
    }
    if info.read_index_host_address % std::mem::align_of::<AtomicU64>() != 0
        || info.write_index_host_address % std::mem::align_of::<AtomicU64>() != 0
    {
        return None;
    }
    // SAFETY: Native queue transport retains aligned 64-bit AQL indices.
    let read =
        unsafe { &*(info.read_index_host_address as *const AtomicU64) }.load(Ordering::Acquire);
    // SAFETY: Native queue transport retains aligned 64-bit AQL indices.
    let write =
        unsafe { &*(info.write_index_host_address as *const AtomicU64) }.load(Ordering::Acquire);
    let pending = write
        .saturating_sub(read)
        .saturating_add(1)
        .min(packet_count);
    for offset in 0..pending {
        let slot = (read + offset) & (packet_count - 1);
        let byte_offset = usize::try_from(slot).ok()?.checked_mul(AQL_PACKET_BYTES)?;
        let packet = info.ring_host_address.checked_add(byte_offset)?;
        // SAFETY: Queue transport retains the aligned packet ring. An acquire
        // load of the published header makes the packet body visible.
        let header = unsafe { &*(packet as *const AtomicU16) }.load(Ordering::Acquire);
        if header & AQL_PACKET_TYPE_MASK != AQL_PACKET_TYPE_KERNEL_DISPATCH {
            continue;
        }
        // SAFETY: The stopped queue retains this complete 64-byte dispatch
        // packet and the private-segment field is naturally aligned.
        let private =
            unsafe { ((packet + AQL_PRIVATE_SEGMENT_SIZE_OFFSET) as *const u32).read_volatile() };
        if private != 0 {
            return Some(private);
        }
    }
    None
}

struct PendingCallback {
    callback: unsafe extern "C" fn(Status, *mut HsaQueue, *mut c_void),
    status: Status,
    queue: usize,
    data: usize,
}

struct QueueEventOutcome {
    callback: Option<PendingCallback>,
    rearm: bool,
}

fn queue_error_status(error: u64) -> Status {
    if error & 2 != 0 {
        INCOMPATIBLE_ARGUMENTS
    } else if error & 4 != 0 {
        INVALID_ALLOCATION
    } else if error & 8 != 0 {
        INVALID_CODE_OBJECT
    } else if error & 16 != 0 {
        MEMORY_FAULT
    } else if error & (32 | 256) != 0 {
        INVALID_PACKET_FORMAT
    } else if error & 64 != 0 {
        INVALID_ARGUMENT
    } else if error & 128 != 0 {
        OUT_OF_REGISTERS
    } else if error & 0x2000_0000 != 0 {
        MEMORY_APERTURE_VIOLATION
    } else if error & 0x4000_0000 != 0 {
        ILLEGAL_INSTRUCTION
    } else if error & 0x8000_0000 != 0 {
        EXCEPTION
    } else {
        ERROR
    }
}

fn pending_callback(
    runtime: &crate::runtime::Runtime,
    key: usize,
    status: Status,
) -> Option<PendingCallback> {
    let queue = runtime.queues.get(&key)?;
    Some(PendingCallback {
        callback: queue.callback?,
        status,
        queue: key,
        data: queue.callback_data,
    })
}

fn handle_queue_event(
    runtime: &mut crate::runtime::Runtime,
    key: usize,
    observed: SignalValue,
) -> Option<QueueEventOutcome> {
    let error = observed as u64;
    if error & 0x401 == 0 {
        return Some(QueueEventOutcome {
            callback: pending_callback(runtime, key, queue_error_status(error)),
            rearm: false,
        });
    }
    let (index, private_segment_size) = {
        let queue = runtime.queues.get(&key)?;
        let index = runtime.gpu_index(queue.agent)?;
        let Some(private_segment_size) = dispatch_private_segment(queue) else {
            return Some(QueueEventOutcome {
                callback: pending_callback(runtime, key, ERROR),
                rearm: false,
            });
        };
        (index, private_segment_size)
    };
    let lanes_per_wave = if error & 0x400 != 0 { 32 } else { 64 };
    let result = allocate_scratch(
        &runtime.gpus[index].device,
        runtime.gpus[index].info,
        private_segment_size,
        lanes_per_wave,
    )
    .and_then(|(allocation, scratch)| {
        let queue = runtime.queues.get_mut(&key).ok_or(INVALID_QUEUE)?;
        queue.native.set_scratch(scratch).map_err(map_error)?;
        queue.scratch = Some(allocation);
        queue.inactive_signal.release_queue();
        Ok(())
    });
    Some(match result {
        Ok(()) => QueueEventOutcome {
            callback: None,
            rearm: true,
        },
        Err(status) => QueueEventOutcome {
            callback: pending_callback(runtime, key, status),
            rearm: false,
        },
    })
}

fn queue_event_worker(
    stop: &AtomicBool,
    alive: &AtomicBool,
    signal: &QueueEventSignal,
    key: usize,
) {
    let mut inactive_armed = true;
    let mut error_armed = true;
    while !stop.load(Ordering::Acquire) && alive.load(Ordering::Acquire) {
        let error = signal.load_error();
        let callback = if error != 0 && error_armed {
            error_armed = false;
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let Some(runtime) = guard.as_mut() else {
                return;
            };
            if !runtime.queues.contains_key(&key) {
                return;
            }
            let callback = pending_callback(runtime, key, queue_error_status(error as u64));
            signal.release_error();
            callback
        } else {
            if error == 0 {
                error_armed = true;
            }
            let observed = signal.load_inactive();
            if observed == 0 {
                inactive_armed = true;
                None
            } else if inactive_armed {
                inactive_armed = false;
                let mut guard = match lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                let Some(runtime) = guard.as_mut() else {
                    return;
                };
                if !runtime.queues.contains_key(&key) {
                    return;
                }
                handle_queue_event(runtime, key, observed).and_then(|outcome| {
                    inactive_armed = outcome.rearm;
                    outcome.callback
                })
            } else {
                None
            }
        };
        if let Some(callback) = callback {
            // SAFETY: HSA requires callback and data to remain valid through
            // queue destruction; the worker only invokes it for a live record.
            unsafe {
                (callback.callback)(
                    callback.status,
                    callback.queue as *mut HsaQueue,
                    callback.data as *mut c_void,
                )
            };
        }
        thread::sleep(Duration::from_micros(20));
    }
}

fn hardware_queue_key(runtime: &crate::runtime::Runtime, queue: *const HsaQueue) -> Option<usize> {
    let key = queue as usize;
    runtime.counted_queues.get(&key).map_or_else(
        || runtime.queues.contains_key(&key).then_some(key),
        |queue| Some(queue.hardware_queue),
    )
}

fn counted_pool_key(
    runtime: &crate::runtime::Runtime,
    queue: *const HsaQueue,
) -> Option<(u64, u32)> {
    let key = queue as usize;
    runtime.counted_queues.get(&key).map_or_else(
        || {
            runtime
                .queues
                .get(&key)
                .and_then(|queue| queue.counted_pool_key)
        },
        |queue| Some(queue.pool_key),
    )
}

fn queue_known(runtime: &crate::runtime::Runtime, queue: *const HsaQueue) -> bool {
    !queue.is_null()
        && (hardware_queue_key(runtime, queue).is_some()
            || runtime.soft_queues.contains_key(&(queue as usize)))
}

fn least_used_counted_queue(pool: &[CountedHardwareQueue]) -> Option<usize> {
    pool.iter()
        .min_by_key(|entry| entry.use_count)
        .map(|entry| entry.queue)
}

fn queue_priority(priority: u32) -> Option<QueuePriority> {
    match priority {
        AMD_QUEUE_PRIORITY_LOW => Some(QueuePriority::Low),
        AMD_QUEUE_PRIORITY_NORMAL => Some(QueuePriority::Normal),
        AMD_QUEUE_PRIORITY_HIGH => Some(QueuePriority::High),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn create_hardware_queue(
    runtime: &mut crate::runtime::Runtime,
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    priority: QueuePriority,
    callback: QueueErrorCallback,
    data: *mut c_void,
    private_segment_size: u32,
    cu_mask: Option<&[u32]>,
    queue: *mut *mut HsaQueue,
) -> Status {
    let Some(index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let ring_size_bytes = match u64::from(size).checked_mul(AQL_PACKET_BYTES as u64) {
        Some(size) => size,
        None => return INVALID_QUEUE_CREATION,
    };
    let id = match runtime.allocate_queue_id() {
        Ok(id) => id,
        Err(status) => return status,
    };
    if private_segment_size != u32::MAX && private_segment_size > MAX_PRIVATE_SEGMENT_BYTES {
        return OUT_OF_RESOURCES;
    }
    let inactive_signal = match QueueEventSignal::create(runtime, index) {
        Ok(signal) => Arc::new(signal),
        Err(status) => return status,
    };
    let (scratch_allocation, scratch) =
        if private_segment_size == 0 || private_segment_size == u32::MAX {
            (None, None)
        } else {
            match allocate_scratch(
                &runtime.gpus[index].device,
                runtime.gpus[index].info,
                private_segment_size,
                runtime.gpus[index].info.wavefront_size,
            ) {
                Ok((allocation, scratch)) => (Some(allocation), Some(scratch)),
                Err(status) => return status,
            }
        };
    let mut native = match runtime.gpus[index].device.gpu().and_then(|gpu| {
        gpu.create_queue(QueueRequest {
            ring_size_bytes,
            parameters: QueueParameters::Aql {
                producer_mode: if queue_type == QUEUE_TYPE_SINGLE {
                    QueueProducerMode::Single
                } else {
                    QueueProducerMode::Multiple
                },
                inactive_signal: Some(inactive_signal.handle()),
                error_event: Some(inactive_signal.error_event()),
                scratch,
            },
            priority,
            device_producer: false,
        })
    }) {
        Ok(queue) => queue,
        Err(error) => return map_error(error),
    };
    if let Some(cu_mask) = cu_mask {
        if let Err(error) = native.set_cu_mask(cu_mask) {
            let _ = native.destroy();
            return map_error(error);
        }
    }
    let info = native.info();
    if info.write_index_host_address < WRITE_INDEX_OFFSET
        || info.read_index_host_address < READ_INDEX_OFFSET
    {
        return ERROR;
    }
    let public = info.write_index_host_address - WRITE_INDEX_OFFSET;
    if info.read_index_host_address - READ_INDEX_OFFSET != public || public % 128 != 0 {
        return ERROR;
    }
    let mut doorbell = Box::new(AmdSignal::doorbell(info.doorbell_host_address, public));
    let doorbell_handle = (&raw mut *doorbell) as usize as u64;
    // SAFETY: rocddi's AQL queue control allocation is the public
    // amd_queue_v2_t layout and remains exclusively owned by native.
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
        (*header).id = id;
    }
    let hardware_id = u32::try_from(id).unwrap_or(u32::MAX);
    runtime.queues.insert(
        public,
        Queue {
            native,
            _doorbell: doorbell,
            inactive_signal: inactive_signal.clone(),
            event_alive: Arc::new(AtomicBool::new(true)),
            event_worker: None,
            scratch: scratch_allocation,
            agent,
            hardware_id,
            counted_pool_key: None,
            cu_mask: cu_mask.map_or_else(Vec::new, <[u32]>::to_vec),
            callback,
            callback_data: data as usize,
            teardown_started: false,
        },
    );
    runtime.released_counted_queues.remove(&public);
    let stop = runtime.stop_workers.clone();
    let alive = runtime.queues[&public].event_alive.clone();
    let Ok(event_worker) = thread::Builder::new()
        .name("rocddi-queue-events".into())
        .spawn(move || queue_event_worker(&stop, &alive, &inactive_signal, public))
    else {
        let Some(mut record) = runtime.queues.remove(&public) else {
            return ERROR;
        };
        if destroy_runtime_queue(&mut record).is_err() {
            std::mem::forget(record);
        }
        return OUT_OF_RESOURCES;
    };
    let Some(record) = runtime.queues.get_mut(&public) else {
        return ERROR;
    };
    record.event_worker = Some(event_worker);
    // SAFETY: The caller supplied writable output storage.
    unsafe { queue.write(public as *mut HsaQueue) };
    runtime.log(
        AMD_LOG_FLAG_INFO,
        format_args!(
            "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={size}",
            agent.handle
        ),
    );
    SUCCESS
}

#[unsafe(no_mangle)]
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
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        create_hardware_queue(
            runtime,
            agent,
            size,
            queue_type,
            QueuePriority::Normal,
            callback,
            data,
            private_segment_size,
            None,
            queue,
        )
    })
}

#[unsafe(no_mangle)]
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
        let Some(native_priority) = queue_priority(priority) else {
            return INVALID_ARGUMENT;
        };
        if queue_type != QUEUE_TYPE_MULTI {
            return INVALID_QUEUE_CREATION;
        }
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        if runtime.counted_queues.try_reserve(1).is_err()
            || runtime.counted_queue_pools.try_reserve(1).is_err()
        {
            return OUT_OF_RESOURCES;
        }

        let pool_key = (agent.handle, priority);
        let create_new = runtime
            .counted_queue_pools
            .get(&pool_key)
            .map_or(0, Vec::len)
            < runtime.counted_queue_limit;
        if create_new
            && runtime
                .counted_queue_pools
                .entry(pool_key)
                .or_default()
                .try_reserve(1)
                .is_err()
        {
            return OUT_OF_RESOURCES;
        }

        let hardware_queue = if create_new {
            let mut created = std::ptr::null_mut();
            let counted_queue_size = runtime.counted_queue_size;
            let status = create_hardware_queue(
                runtime,
                agent,
                counted_queue_size,
                queue_type,
                native_priority,
                callback,
                data,
                0,
                None,
                &raw mut created,
            );
            if status != SUCCESS {
                return OUT_OF_RESOURCES;
            }
            let hardware_queue = created as usize;
            let Some(record) = runtime.queues.get_mut(&hardware_queue) else {
                return ERROR;
            };
            record.counted_pool_key = Some(pool_key);
            runtime
                .counted_queue_pools
                .entry(pool_key)
                .or_default()
                .push(CountedHardwareQueue {
                    queue: hardware_queue,
                    use_count: 0,
                });
            // SAFETY: create_hardware_queue retains a complete public queue
            // control mapping with an aligned properties word.
            unsafe {
                (*created
                    .cast::<u8>()
                    .add(QUEUE_PROPERTIES_OFFSET)
                    .cast::<AtomicU32>())
                .fetch_or(AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
            }
            hardware_queue
        } else {
            let Some(hardware_queue) = runtime
                .counted_queue_pools
                .get(&pool_key)
                .and_then(|pool| least_used_counted_queue(pool))
            else {
                return OUT_OF_RESOURCES;
            };
            hardware_queue
        };

        let Some(entry) = runtime
            .counted_queue_pools
            .get_mut(&pool_key)
            .and_then(|pool| pool.iter_mut().find(|entry| entry.queue == hardware_queue))
        else {
            return ERROR;
        };
        let Some(use_count) = entry.use_count.checked_add(1) else {
            return OUT_OF_RESOURCES;
        };
        // SAFETY: The selected hardware queue remains owned by the counted
        // pool until runtime shutdown.
        let mut counted = unsafe { CountedQueue::new(hardware_queue, pool_key) };
        let public = counted.public_pointer();
        let key = public as usize;
        entry.use_count = use_count;
        runtime.released_counted_queues.remove(&key);
        runtime.counted_queues.insert(key, counted);
        // SAFETY: The caller supplied writable output storage.
        unsafe { queue.write(public) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
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
        let Some(counted) = runtime.counted_queues.remove(&key) else {
            return ERROR;
        };
        let Some(entry) = runtime
            .counted_queue_pools
            .get_mut(&counted.pool_key)
            .and_then(|pool| {
                pool.iter_mut()
                    .find(|entry| entry.queue == counted.hardware_queue)
            })
        else {
            runtime.counted_queues.insert(key, counted);
            return ERROR;
        };
        let Some(use_count) = entry.use_count.checked_sub(1) else {
            runtime.counted_queues.insert(key, counted);
            return ERROR;
        };
        entry.use_count = use_count;
        runtime.released_counted_queues.insert(key);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
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
        if !matches!(
            pool.handle,
            CPU_POOL_FINE | CPU_POOL_EXTENDED | CPU_POOL_KERNARG | CPU_POOL_COARSE
        ) && runtime.decode_gpu_pool(pool).is_none()
        {
            return INVALID_REGION;
        }
        if !runtime.owns_signal(doorbell_signal) {
            return INVALID_SIGNAL;
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
        let id = match runtime.allocate_queue_id() {
            Ok(id) => id,
            Err(status) => return status,
        };
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
        runtime.soft_queues.insert(
            public as usize,
            SoftQueue {
                _control: control,
                _ring: ring,
                active: true,
            },
        );
        // SAFETY: The caller supplied writable output storage and the control
        // allocation remains stable until hsa_queue_destroy.
        unsafe { queue.write(public) };
        SUCCESS
    })
}

fn full_cu_mask(compute_units: u32) -> Vec<u32> {
    let mut mask = vec![u32::MAX; compute_units.div_ceil(32) as usize];
    if let Some(last) = mask.last_mut() {
        let tail = compute_units % 32;
        if tail != 0 {
            *last = (1_u32 << tail) - 1;
        }
    }
    mask
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_queue_create(
    agent: HsaAgent,
    descriptors: *mut HsaAmdQueueCreateDesc,
    descriptor_count: u32,
) -> Status {
    boundary(|| {
        if descriptors.is_null() || descriptor_count == 0 {
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
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        // SAFETY: The caller supplies descriptor_count writable descriptors.
        let descriptors =
            unsafe { std::slice::from_raw_parts_mut(descriptors, descriptor_count as usize) };
        let mut first_error = SUCCESS;
        for descriptor in descriptors {
            descriptor.queue = std::ptr::null_mut();
            let mut fail = |status| {
                if first_error == SUCCESS {
                    first_error = status;
                }
            };
            if descriptor.version != AMD_QUEUE_CREATE_DESC_VERSION
                || descriptor.queue_size_bytes == 0
                || !descriptor.queue_size_bytes.is_power_of_two()
                || descriptor.priority > AMD_QUEUE_PRIORITY_HIGH
                || descriptor.traffic_class != 0
                || descriptor.reserved_header.iter().any(|byte| *byte != 0)
                || descriptor.reserved.iter().any(|byte| *byte != 0)
            {
                fail(INVALID_ARGUMENT);
                continue;
            }
            let known_flags =
                AMD_QUEUE_CREATE_DEVICE_MEM_RING | AMD_QUEUE_CREATE_DEVICE_MEM_DESCRIPTOR;
            if descriptor.flags & !known_flags != 0 {
                fail(INVALID_ARGUMENT);
                continue;
            }
            if descriptor.flags != 0 {
                fail(INVALID_QUEUE_CREATION);
                continue;
            }
            if descriptor.engine_type != AMD_QUEUE_ENGINE_COMPUTE {
                fail(
                    if matches!(
                        descriptor.engine_type,
                        AMD_QUEUE_ENGINE_SDMA | AMD_QUEUE_ENGINE_AIE
                    ) {
                        INVALID_QUEUE_CREATION
                    } else {
                        INVALID_ARGUMENT
                    },
                );
                continue;
            }
            // SAFETY: engine_type selects the compute arm of the C union.
            let compute = unsafe { descriptor.engine.compute };
            if descriptor.queue_size_bytes % AQL_PACKET_BYTES as u32 != 0
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
            let Some(priority) = queue_priority(descriptor.priority) else {
                fail(INVALID_ARGUMENT);
                continue;
            };
            let cu_mask = if compute.cu_mask_count == 0 {
                None
            } else {
                // SAFETY: The descriptor supplies one u32 per 32 requested bits.
                Some(unsafe {
                    std::slice::from_raw_parts(compute.cu_mask, compute.cu_mask_count as usize / 32)
                })
            };
            let packet_count = descriptor.queue_size_bytes / AQL_PACKET_BYTES as u32;
            let status = create_hardware_queue(
                runtime,
                agent,
                packet_count,
                compute.queue_type,
                priority,
                descriptor.callback,
                descriptor.callback_data,
                compute.private_segment_size,
                cu_mask,
                &raw mut descriptor.queue,
            );
            if status != SUCCESS {
                fail(status);
            }
        }
        first_error
    })
}

#[unsafe(no_mangle)]
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
        let Some(agent) = runtime.queues.get(&hardware_key).map(|record| record.agent) else {
            return INVALID_QUEUE;
        };
        let Some(gpu_index) = runtime.gpu_index(agent) else {
            return INVALID_QUEUE;
        };
        let all_enabled;
        let selected = if bit_count == 0 {
            let compute_units = runtime.gpus[gpu_index].info.compute_unit_count;
            all_enabled = full_cu_mask(compute_units);
            all_enabled.as_slice()
        } else {
            // SAFETY: The ABI requires bit_count / 32 readable words.
            unsafe { std::slice::from_raw_parts(mask, bit_count as usize / 32) }
        };
        let Some(record) = runtime.queues.get_mut(&hardware_key) else {
            return INVALID_QUEUE;
        };
        match record.native.set_cu_mask(selected) {
            Ok(()) => {
                record.cu_mask.clear();
                record.cu_mask.extend_from_slice(selected);
                SUCCESS
            }
            Err(error) => map_error(error),
        }
    })
}

#[unsafe(no_mangle)]
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
        let Some(record) = runtime.queues.get(&hardware_key) else {
            return INVALID_QUEUE;
        };
        let Some(gpu_index) = runtime.gpu_index(record.agent) else {
            return INVALID_QUEUE;
        };
        let enabled = if record.cu_mask.is_empty() {
            full_cu_mask(runtime.gpus[gpu_index].info.compute_unit_count)
        } else {
            record.cu_mask.clone()
        };
        let output_words = bit_count as usize / 32;
        // SAFETY: The caller supplied output_words writable entries.
        let output = unsafe { std::slice::from_raw_parts_mut(mask, output_words) };
        output.fill(0);
        let copied = output.len().min(enabled.len());
        output[..copied].copy_from_slice(&enabled[..copied]);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
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
        if counted_pool_key(runtime, queue).is_some() {
            return INVALID_QUEUE;
        }
        let Some(hardware_key) = hardware_queue_key(runtime, queue) else {
            return INVALID_QUEUE;
        };
        let Some(priority) = queue_priority(priority) else {
            return INVALID_ARGUMENT;
        };
        let Some(record) = runtime.queues.get_mut(&hardware_key) else {
            return INVALID_QUEUE;
        };
        record
            .native
            .set_priority(priority)
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[unsafe(no_mangle)]
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
        if let Some(hardware_key) = hardware_queue_key(runtime, queue) {
            if let Some(record) = runtime.queues.get_mut(&hardware_key) {
                return record
                    .native
                    .inactivate()
                    .map_or_else(map_error, |()| SUCCESS);
            }
        }
        if let Some(record) = runtime.soft_queues.get_mut(&(queue as usize)) {
            record.inactivate();
            return SUCCESS;
        }
        INVALID_QUEUE
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_destroy(queue: *mut HsaQueue) -> Status {
    boundary(|| {
        if queue.is_null() {
            return INVALID_ARGUMENT;
        }
        let key = queue as usize;
        let mut record = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            if runtime
                .queues
                .get(&key)
                .is_some_and(|record| record.counted_pool_key.is_some())
            {
                return INVALID_QUEUE;
            }
            let Some(record) = runtime.queues.remove(&key) else {
                return if runtime.soft_queues.remove(&key).is_some() {
                    SUCCESS
                } else {
                    INVALID_QUEUE
                };
            };
            record
        };
        match destroy_runtime_queue(&mut record) {
            Ok(()) => {
                if let Ok(guard) = lock() {
                    if let Some(runtime) = guard.as_ref() {
                        runtime.log(
                            AMD_LOG_FLAG_INFO,
                            format_args!("destroyed AQL queue address=0x{key:x}"),
                        );
                    }
                }
                SUCCESS
            }
            Err(error) => {
                let status = map_error(error);
                let Ok(mut guard) = lock() else {
                    std::mem::forget(record);
                    return status;
                };
                let Some(runtime) = guard.as_mut() else {
                    std::mem::forget(record);
                    return status;
                };
                runtime.queues.insert(key, record);
                status
            }
        }
    })
}

#[inline]
unsafe fn queue_index(queue: *const HsaQueue, offset: usize) -> Option<&'static AtomicU64> {
    if queue.is_null() {
        return None;
    }
    let hardware = if queue as usize & COUNTED_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live counted handle points at CountedQueuePublic::header,
        // whose preceding word holds the stable hardware queue address.
        unsafe {
            (*queue
                .cast::<u8>()
                .sub(COUNTED_QUEUE_HANDLE_BIT)
                .cast::<CountedQueuePublic>())
            .hardware_queue
        }
    } else {
        queue as usize
    };
    // SAFETY: The caller retains a live queue handle. Native and soft queue
    // controls contain the index at the public ABI offset; counted handles
    // retain their hardware queue through the counted pool.
    Some(unsafe { &*(hardware as *const u8).add(offset).cast::<AtomicU64>() })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_read_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe { queue_index(queue, READ_INDEX_OFFSET) }
        .map_or(0, |index| index.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_read_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe { queue_index(queue, READ_INDEX_OFFSET) }
        .map_or(0, |index| index.load(Ordering::Acquire))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_read_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_read_index_scacquire(queue) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_write_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe { queue_index(queue, WRITE_INDEX_OFFSET) }
        .map_or(0, |index| index.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_write_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe { queue_index(queue, WRITE_INDEX_OFFSET) }
        .map_or(0, |index| index.load(Ordering::Acquire))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_write_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_write_index_scacquire(queue) }
}

unsafe fn queue_store(queue: *const HsaQueue, offset: usize, value: u64, order: Ordering) {
    // SAFETY: The caller owns a live queue while accessing its index.
    if let Some(index) = unsafe { queue_index(queue, offset) } {
        index.store(value, order);
    }
}

unsafe fn queue_compare_exchange(
    queue: *const HsaQueue,
    expected: u64,
    value: u64,
    success: Ordering,
    failure: Ordering,
) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe { queue_index(queue, WRITE_INDEX_OFFSET) }.map_or(0, |index| {
        index
            .compare_exchange(expected, value, success, failure)
            .unwrap_or_else(|observed| observed)
    })
}

unsafe fn queue_add(queue: *const HsaQueue, value: u64, order: Ordering) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe { queue_index(queue, WRITE_INDEX_OFFSET) }
        .map_or(0, |index| index.fetch_add(value, order))
}

macro_rules! queue_store_entry {
    ($name:ident, $offset:expr, $order:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, value: u64) {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_store(queue, $offset, value, $order) };
        }
    };
}

macro_rules! queue_compare_exchange_entry {
    ($name:ident, $success:expr, $failure:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, expected: u64, value: u64) -> u64 {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_compare_exchange(queue, expected, value, $success, $failure) }
        }
    };
}

macro_rules! queue_add_entry {
    ($name:ident, $order:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, value: u64) -> u64 {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_add(queue, value, $order) }
        }
    };
}

queue_store_entry!(
    hsa_queue_store_write_index_relaxed,
    WRITE_INDEX_OFFSET,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_write_index_screlease,
    WRITE_INDEX_OFFSET,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_write_index_release,
    WRITE_INDEX_OFFSET,
    Ordering::Release
);

queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_scacq_screl,
    Ordering::AcqRel,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_acq_rel,
    Ordering::AcqRel,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_scacquire,
    Ordering::Acquire,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_acquire,
    Ordering::Acquire,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_relaxed,
    Ordering::Relaxed,
    Ordering::Relaxed
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_screlease,
    Ordering::Release,
    Ordering::Relaxed
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_release,
    Ordering::Release,
    Ordering::Relaxed
);

queue_add_entry!(hsa_queue_add_write_index_scacq_screl, Ordering::AcqRel);
queue_add_entry!(hsa_queue_add_write_index_acq_rel, Ordering::AcqRel);
queue_add_entry!(hsa_queue_add_write_index_scacquire, Ordering::Acquire);
queue_add_entry!(hsa_queue_add_write_index_acquire, Ordering::Acquire);
queue_add_entry!(hsa_queue_add_write_index_relaxed, Ordering::Relaxed);
queue_add_entry!(hsa_queue_add_write_index_screlease, Ordering::Release);
queue_add_entry!(hsa_queue_add_write_index_release, Ordering::Release);

queue_store_entry!(
    hsa_queue_store_read_index_relaxed,
    READ_INDEX_OFFSET,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_read_index_screlease,
    READ_INDEX_OFFSET,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_read_index_release,
    READ_INDEX_OFFSET,
    Ordering::Release
);

#[unsafe(no_mangle)]
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
            &*(queue as *mut HsaQueue)
                .cast::<u8>()
                .add(QUEUE_PROPERTIES_OFFSET)
                .cast::<AtomicU32>()
        };
        if enable == 0 {
            properties.fetch_and(!AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
        } else {
            properties.fetch_or(AMD_QUEUE_PROPERTIES_ENABLE_PROFILING, Ordering::Release);
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
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
        if runtime.released_counted_queues.contains(&input_key) {
            return INVALID_ARGUMENT;
        }
        let counted = runtime.counted_queues.get(&input_key);
        let hardware_key = counted.map_or(input_key, |queue| queue.hardware_queue);
        let Some(record) = runtime.queues.get(&hardware_key) else {
            return INVALID_QUEUE;
        };
        // SAFETY: Each match arm writes the public type for the queried attribute.
        unsafe {
            match attribute {
                AMD_QUEUE_INFO_AGENT => value.cast::<HsaAgent>().write(record.agent),
                AMD_QUEUE_INFO_DOORBELL_ID => value.cast::<u64>().write(record.hardware_id.into()),
                QUEUE_INFO_USE_COUNT => {
                    let pool_key =
                        counted.map_or(record.counted_pool_key, |queue| Some(queue.pool_key));
                    let use_count = if let Some(pool_key) = pool_key {
                        runtime
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
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_profiling_get_dispatch_time(
    agent: HsaAgent,
    signal: HsaSignal,
    time: *mut ProfilingTime,
) -> Status {
    boundary(|| {
        if time.is_null() {
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
        let Some(index) = runtime.gpu_index(agent) else {
            return INVALID_AGENT;
        };
        // SAFETY: Callers pass a completion signal that remains live while its
        // profiling fields are queried.
        let Some(signal) = (unsafe { crate::signal::signal_ref(signal) }) else {
            return INVALID_SIGNAL;
        };
        let start = signal.start_ts.load(Ordering::Acquire);
        let end = signal.end_ts.load(Ordering::Acquire);
        let (start, end) = match runtime.translate_gpu_interval(index, start, end) {
            Ok(interval) => interval,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { time.write(ProfilingTime { start, end }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_profiling_get_async_copy_time(
    signal: HsaSignal,
    time: *mut ProfilingTime,
) -> Status {
    boundary(|| {
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
        if !runtime.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        // SAFETY: owns_signal validated this live slab slot.
        let signal = unsafe { &*(signal.handle as usize as *const AmdSignal) };
        let start = signal.start_ts.load(Ordering::Acquire);
        let end = signal.end_ts.load(Ordering::Acquire);
        if start == 0 && end == 0 {
            return ERROR;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { time.write(ProfilingTime { start, end }) };
        SUCCESS
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_a_queue_joins_its_event_worker() {
        let alive = Arc::new(AtomicBool::new(true));
        let entered = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let worker_alive = alive.clone();
        let worker_entered = entered.clone();
        let worker_finished = finished.clone();
        let mut worker = Some(thread::spawn(move || {
            worker_entered.store(true, Ordering::Release);
            while worker_alive.load(Ordering::Acquire) {
                thread::yield_now();
            }
            thread::sleep(Duration::from_millis(2));
            worker_finished.store(true, Ordering::Release);
        }));
        let deadline = std::time::Instant::now() + Duration::from_millis(100);
        while !entered.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            thread::yield_now();
        }

        stop_queue_event_worker(&alive, &mut worker);

        assert!(worker.is_none());
        assert!(finished.load(Ordering::Acquire));
    }

    #[test]
    fn async_copy_profiling_entry_point_matches_the_public_abi() {
        let _: unsafe extern "C" fn(HsaSignal, *mut ProfilingTime) -> Status =
            hsa_amd_profiling_get_async_copy_time;
    }

    #[test]
    fn external_semaphore_queue_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(*mut HsaQueue, HsaAmdExternalSemaphore, u64) -> Status =
            hsa_amd_queue_signal_external_semaphore;
        let _: unsafe extern "C" fn(*mut HsaQueue, HsaAmdExternalSemaphore, u64) -> Status =
            hsa_amd_queue_wait_external_semaphore;
    }

    #[test]
    fn counted_queue_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(
            HsaAgent,
            u32,
            u32,
            QueueErrorCallback,
            *mut c_void,
            u64,
            *mut *mut HsaQueue,
        ) -> Status = hsa_amd_counted_queue_acquire;
        let _: unsafe extern "C" fn(*mut HsaQueue) -> Status = hsa_amd_counted_queue_release;
    }

    #[test]
    fn counted_queue_pool_selects_the_first_least_used_hardware_queue() {
        let pool = [
            CountedHardwareQueue {
                queue: 0x1000,
                use_count: 3,
            },
            CountedHardwareQueue {
                queue: 0x2000,
                use_count: 1,
            },
            CountedHardwareQueue {
                queue: 0x3000,
                use_count: 1,
            },
        ];
        assert_eq!(least_used_counted_queue(&pool), Some(0x2000));
        assert_eq!(least_used_counted_queue(&[]), None);
    }

    #[test]
    fn counted_queue_handles_are_unique_copies_of_the_hardware_header() {
        let hardware = Box::new(HsaQueue {
            queue_type: QUEUE_TYPE_MULTI,
            features: QUEUE_FEATURE_KERNEL_DISPATCH,
            base_address: 0x1234usize as *mut c_void,
            doorbell_signal: HsaSignal { handle: 0x5678 },
            size: 16_384,
            reserved: 0,
            id: 42,
        });
        let hardware_key = (&raw const *hardware) as usize;
        // SAFETY: hardware remains live while both public headers are copied.
        let mut first = unsafe { CountedQueue::new(hardware_key, (1, 2)) };
        // SAFETY: hardware remains live while both public headers are copied.
        let mut second = unsafe { CountedQueue::new(hardware_key, (1, 2)) };
        let first = first.public_pointer();
        let second = second.public_pointer();
        assert_ne!(first, second);
        assert_eq!(first as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
        assert_eq!(second as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
        // SAFETY: Both pointers refer to live CountedQueue-owned headers.
        unsafe {
            assert_eq!((*first).base_address, hardware.base_address);
            assert_eq!(
                (*first).doorbell_signal.handle,
                hardware.doorbell_signal.handle
            );
            assert_eq!((*first).size, hardware.size);
            assert_eq!((*first).id, hardware.id);
            assert_eq!((*second).base_address, hardware.base_address);
            assert_eq!(
                (*second).doorbell_signal.handle,
                hardware.doorbell_signal.handle
            );
            assert_eq!((*second).size, hardware.size);
            assert_eq!((*second).id, hardware.id);
        }
    }

    #[repr(C, align(128))]
    struct PublicQueueStorage([u8; 256]);

    #[test]
    fn soft_queue_control_uses_the_direct_index_path() {
        let mut control = Box::new(SoftQueueControl([0; 32]));
        let public = control.0.as_mut_ptr().cast::<HsaQueue>();
        assert_eq!(public as usize % 128, 0);
        // SAFETY: The aligned control allocation retains the read index word.
        unsafe {
            public
                .cast::<u8>()
                .add(READ_INDEX_OFFSET)
                .cast::<AtomicU64>()
                .write(AtomicU64::new(31));
            assert_eq!(hsa_queue_load_read_index_relaxed(public), 31);
        }
    }

    #[test]
    fn counted_queue_index_operations_reach_the_hardware_control() {
        let mut hardware = PublicQueueStorage([0; 256]);
        let hardware_queue = hardware.0.as_mut_ptr().cast::<HsaQueue>();
        // SAFETY: The storage is aligned and retains the public header and
        // both atomic index words throughout this test.
        unsafe {
            hardware_queue.write(HsaQueue {
                queue_type: QUEUE_TYPE_MULTI,
                features: QUEUE_FEATURE_KERNEL_DISPATCH,
                base_address: std::ptr::null_mut(),
                doorbell_signal: HsaSignal { handle: 0 },
                size: 64,
                reserved: 0,
                id: 7,
            });
            let read = hardware
                .0
                .as_mut_ptr()
                .add(READ_INDEX_OFFSET)
                .cast::<AtomicU64>();
            let write = hardware
                .0
                .as_mut_ptr()
                .add(WRITE_INDEX_OFFSET)
                .cast::<AtomicU64>();
            read.write(AtomicU64::new(19));
            write.write(AtomicU64::new(23));

            let mut counted = CountedQueue::new(hardware_queue as usize, (1, 2));
            let public = counted.public_pointer();
            assert_eq!(hardware_queue as usize % 128, 0);
            assert_eq!(public as usize % 128, COUNTED_QUEUE_HANDLE_BIT);
            assert_eq!(public as usize % 64, 0);
            assert_eq!(hsa_queue_load_read_index_relaxed(public), 19);
            assert_eq!(hsa_queue_load_read_index_scacquire(public), 19);
            assert_eq!(hsa_queue_add_write_index_relaxed(public, 2), 23);
            assert_eq!((&*write).load(Ordering::Relaxed), 25);
            assert_eq!(hsa_queue_load_write_index_relaxed(public), 25);
        }
    }

    fn gfx1201() -> GpuInfo {
        GpuInfo {
            gfx_major: 12,
            gfx_minor: 0,
            gfx_stepping: 1,
            wavefront_size: 32,
            compute_unit_count: 64,
            maximum_wave_count_per_compute_unit: 32,
            maximum_scratch_wave_count_per_compute_unit: 32,
            xcc_count: 1,
            shader_engine_count_per_xcc: 2,
            ..GpuInfo::default()
        }
    }

    #[test]
    fn amd_queue_create_descriptor_matches_the_public_x86_64_abi() {
        assert_eq!(std::mem::size_of::<HsaAmdComputeQueueParams>(), 32);
        assert_eq!(std::mem::size_of::<HsaAmdSdmaQueueParams>(), 32);
        assert_eq!(std::mem::size_of::<HsaAmdQueueEngineParams>(), 32);
        assert_eq!(std::mem::size_of::<HsaAmdQueueCreateDesc>(), 96);
        assert_eq!(std::mem::align_of::<HsaAmdQueueCreateDesc>(), 8);
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, version), 0);
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, flags), 2);
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, engine_type), 4);
        assert_eq!(
            std::mem::offset_of!(HsaAmdQueueCreateDesc, queue_size_bytes),
            8
        );
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, priority), 12);
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, callback), 16);
        assert_eq!(
            std::mem::offset_of!(HsaAmdQueueCreateDesc, callback_data),
            24
        );
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, queue), 32);
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, engine), 40);
        assert_eq!(
            std::mem::offset_of!(HsaAmdQueueCreateDesc, traffic_class),
            72
        );
        assert_eq!(std::mem::offset_of!(HsaAmdQueueCreateDesc, reserved), 76);
    }

    #[test]
    fn gfx1201_dynamic_scratch_covers_every_native_wave_slot() {
        assert_eq!(
            scratch_plan(gfx1201(), 1024, 32),
            Ok(ScratchPlan {
                byte_length: 64 * 1024 * 1024,
                maximum_private_segment_byte_length: 1024,
                maximum_wave_count: 2048,
            })
        );
    }

    #[test]
    fn scratch_plan_aligns_per_lane_size_and_rejects_unqualified_targets() {
        assert_eq!(
            scratch_plan(gfx1201(), 1025, 32),
            Ok(ScratchPlan {
                byte_length: 1032 * 32 * 2048,
                maximum_private_segment_byte_length: 1032,
                maximum_wave_count: 2048,
            })
        );

        let mut unsupported = gfx1201();
        unsupported.gfx_stepping = 0;
        assert_eq!(scratch_plan(unsupported, 1024, 32), Err(OUT_OF_RESOURCES));
        assert_eq!(scratch_plan(gfx1201(), 0, 32), Err(OUT_OF_RESOURCES));
        assert_eq!(
            scratch_plan(gfx1201(), MAX_PRIVATE_SEGMENT_BYTES + 1, 32),
            Err(OUT_OF_RESOURCES)
        );
        assert_eq!(scratch_plan(gfx1201(), 1024, 16), Err(OUT_OF_RESOURCES));
    }

    #[test]
    fn gfx1201_wave64_scratch_covers_twice_the_wave32_storage() {
        assert_eq!(
            scratch_plan(gfx1201(), 1024, 64),
            Ok(ScratchPlan {
                byte_length: 128 * 1024 * 1024,
                maximum_private_segment_byte_length: 1024,
                maximum_wave_count: 2048,
            })
        );
    }

    #[test]
    fn firmware_queue_errors_map_to_public_status_codes() {
        assert_eq!(queue_error_status(2), INCOMPATIBLE_ARGUMENTS);
        assert_eq!(queue_error_status(4), INVALID_ALLOCATION);
        assert_eq!(queue_error_status(8), INVALID_CODE_OBJECT);
        assert_eq!(queue_error_status(16), MEMORY_FAULT);
        assert_eq!(queue_error_status(32), INVALID_PACKET_FORMAT);
        assert_eq!(queue_error_status(64), INVALID_ARGUMENT);
        assert_eq!(queue_error_status(128), OUT_OF_REGISTERS);
        assert_eq!(queue_error_status(0x2000_0000), MEMORY_APERTURE_VIOLATION);
        assert_eq!(queue_error_status(0x4000_0000), ILLEGAL_INSTRUCTION);
        assert_eq!(queue_error_status(0x8000_0000), EXCEPTION);
    }

    #[test]
    fn base_atomic_entry_points_preserve_queue_indices() {
        let mut storage = PublicQueueStorage([0; 256]);
        let queue = storage.0.as_mut_ptr().cast::<HsaQueue>();

        // SAFETY: The aligned backing storage contains initialized atomic
        // index fields at the public amd_queue_v2_t offsets used below.
        unsafe {
            storage
                .0
                .as_mut_ptr()
                .add(WRITE_INDEX_OFFSET)
                .cast::<AtomicU64>()
                .write(AtomicU64::new(0));
            storage
                .0
                .as_mut_ptr()
                .add(READ_INDEX_OFFSET)
                .cast::<AtomicU64>()
                .write(AtomicU64::new(0));

            hsa_queue_store_write_index_relaxed(queue, 4);
            assert_eq!(hsa_queue_load_write_index_acquire(queue), 4);
            assert_eq!(hsa_queue_add_write_index_relaxed(queue, 3), 4);
            assert_eq!(hsa_queue_cas_write_index_scacquire(queue, 6, 9), 7);
            assert_eq!(hsa_queue_load_write_index_relaxed(queue), 7);
            assert_eq!(hsa_queue_cas_write_index_scacq_screl(queue, 7, 9), 7);
            assert_eq!(hsa_queue_load_write_index_scacquire(queue), 9);

            hsa_queue_store_write_index_release(queue, 11);
            assert_eq!(hsa_queue_load_write_index_relaxed(queue), 11);
            hsa_queue_store_read_index_screlease(queue, 5);
            assert_eq!(hsa_queue_load_read_index_acquire(queue), 5);
            hsa_queue_store_read_index_release(queue, 8);
            assert_eq!(hsa_queue_load_read_index_relaxed(queue), 8);
        }
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn repeated_queue_index_access_does_not_wait_for_the_registry_mutex() {
        use std::sync::mpsc;

        let mut storage = PublicQueueStorage([0; 256]);
        // SAFETY: This aligned storage contains a live atomic at the public
        // write-index offset until the scoped worker exits.
        unsafe {
            storage
                .0
                .as_mut_ptr()
                .add(WRITE_INDEX_OFFSET)
                .cast::<AtomicU64>()
                .write(AtomicU64::new(42));
        }
        let pointer = storage.0.as_ptr() as usize;
        let (ready_tx, ready_rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(move || {
                // SAFETY: The scoped owner keeps the synthetic queue storage
                // live for the read.
                let queue = pointer as *const HsaQueue;
                ready_tx.send(()).unwrap();
                go_rx.recv().unwrap();
                done_tx
                    .send(unsafe { hsa_queue_load_write_index_relaxed(queue) })
                    .unwrap();
            });
            ready_rx.recv().unwrap();
            let guard = crate::runtime::RUNTIME.lock().unwrap();
            go_tx.send(()).unwrap();
            let value = done_rx.recv_timeout(Duration::from_millis(200));
            drop(guard);
            assert_eq!(value.unwrap(), 42);
        });
    }
}
