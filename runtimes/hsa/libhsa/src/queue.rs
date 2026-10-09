// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA queue objects, public ring layout, scratch, and error-event delivery.
//!
//! Hardware queues retain their rocddi queue, ring allocation, index storage,
//! doorbell signal, scratch backing, and event-worker dependencies as one
//! teardown unit. Counted queues add shared acquisition accounting without
//! changing the public `hsa_queue_t` layout. SDMA queues use a separate public
//! header that borrows their native byte-index pointers. Soft queues implement
//! only the host-visible index semantics that do not require a native GPU queue.
//!
//! Atomic accessors use ABI-defined offsets for AQL and soft queues and native
//! index mappings for SDMA. Destruction first prevents new observation, then
//! stops workers, then releases native resources; partial failure keeps enough
//! state for a safe retry.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use rocddi::device::Device;
use rocddi::gpu::queue::{
    QueueAccessWidth, QueueErrorEvent, QueueParameters, QueuePriority, QueueProducerMode,
    QueueRequest, QueueRingMemory, QueueScratch, QueueTransport, SdmaEngineSelection,
};
use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};
use rocddi::topology::GpuInfo;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{CallbackScope, VM_FAULT_CONDVAR, boundary, initialized_mut, lock, map_error};
use crate::signal::{AmdSignal, AsyncCopyClock};

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
    event: crate::platform::event::SignalEvent,
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
                GPU_PAGE_BYTES.max(runtime.host_page_size as u64),
                runtime.host_page_size as u64,
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
    // A failed control update may have published the candidate address.
    // Retain it through native destruction even if a later update succeeds.
    uncertain_scratch: Vec<Allocation>,
    pub(crate) agent: HsaAgent,
    hardware_id: u32,
    counted_pool_key: Option<(u64, u32)>,
    cooperative_refs: u32,
    ring_memory: QueueRingMemory,
    inactivated: bool,
    cu_mask: Vec<u32>,
    callback: QueueErrorCallback,
    callback_data: CallbackArg,
    pub(crate) vm_faulted: bool,
    pub(crate) vm_fault_address: u64,
    pub(crate) vm_fault_reason: u32,
    teardown_started: bool,
}

fn stop_queue_event_worker(
    alive: &AtomicBool,
    worker: &mut Option<JoinHandle<()>>,
) -> Result<(), rocddi::Error> {
    alive.store(false, Ordering::Release);
    if worker
        .as_ref()
        .is_some_and(|handle| handle.thread().id() == thread::current().id())
    {
        return Err(rocddi::Error::Operation {
            kind: rocddi::ErrorKind::Busy,
            detail: "queue event worker cannot join itself",
        });
    }
    if let Some(worker) = worker.take() {
        let _ = worker.join();
    }
    Ok(())
}

fn destroy_native_queue(
    native: &mut rocddi::gpu::queue::Queue,
    teardown_started: &mut bool,
) -> Result<(), rocddi::Error> {
    if !*teardown_started {
        native.inactivate()?;
        *teardown_started = true;
    }
    // SAFETY: Inactivation stopped GPU queue consumption. The HSA caller must
    // not publish through this queue concurrently with its destruction.
    unsafe { native.destroy() }
}

pub(crate) fn destroy_runtime_queue(queue: &mut Queue) -> Result<(), rocddi::Error> {
    stop_queue_event_worker(&queue.event_alive, &mut queue.event_worker)?;
    destroy_native_queue(&mut queue.native, &mut queue.teardown_started)
}

pub(crate) fn destroy_runtime_sdma_queue(queue: &mut SdmaQueue) -> Result<(), rocddi::Error> {
    destroy_native_queue(&mut queue.native, &mut queue.teardown_started)
}

fn abandon_unpublished_queue<D: 'static>(
    native: rocddi::gpu::queue::Queue,
    dependencies: D,
    status: Status,
) -> Status {
    // SAFETY: This queue has no public producer. The owner tuple retains every
    // external GPU address that a GPU engine could still reach after failed
    // cleanup.
    unsafe { native.abandon_unpublished_with_dependencies(dependencies) }
        .map_or_else(map_error, |()| status)
}

// Native and soft queue controls expose at least 128-byte-aligned handles.
// Put counted handles 64 bytes past that boundary so index operations can
// identify them without consulting the registry. SDMA controls put the public
// header 128 bytes into a 256-byte allocation for the same reason.
const COUNTED_QUEUE_HANDLE_BIT: usize = 64;
const SDMA_QUEUE_HANDLE_BIT: usize = 128;

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
    doorbell_signal: HsaSignal,
    active: bool,
}

#[repr(C, align(256))]
struct SoftQueueControl([u64; 32]);

impl SoftQueue {
    fn inactivate(&mut self) {
        self.active = false;
    }
}

/// Public SDMA header and borrowed native index mappings. The header is
/// separate from rocddi's byte-index control page, which has no HSA prefix.
#[repr(C, align(256))]
struct SdmaQueueControl {
    read_index_host_address: usize,
    write_index_host_address: usize,
    _padding: [u8; SDMA_QUEUE_HANDLE_BIT - 2 * std::mem::size_of::<usize>()],
    header: [u8; std::mem::size_of::<HsaQueue>()],
}

const _: () = assert!(std::mem::offset_of!(SdmaQueueControl, header) == SDMA_QUEUE_HANDLE_BIT);

/// One SDMA public handle borrowing the ring and indices of its native owner.
pub(crate) struct SdmaQueue {
    pub(crate) native: rocddi::gpu::queue::Queue,
    _control: Box<SdmaQueueControl>,
    _doorbell: Box<AmdSignal>,
    agent: HsaAgent,
    hardware_id: u32,
    teardown_started: bool,
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
    data: CallbackArg,
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

fn pending_error_callback(
    runtime: &mut crate::runtime::Runtime,
    key: usize,
    status: Status,
) -> Option<PendingCallback> {
    if status == MEMORY_FAULT {
        // Publish the affected queue so a running system event worker can
        // attach the process fault's address and reason.
        let details = runtime.vm_fault_details;
        let queue = runtime.queues.get_mut(&key)?;
        queue.vm_faulted = true;
        if let Some((_, address, reason)) = details.filter(|(agent, _, _)| *agent == queue.agent) {
            queue.vm_fault_address = address;
            queue.vm_fault_reason = reason;
        }
        if runtime.system_event_worker_started {
            VM_FAULT_CONDVAR.notify_all();
            return None;
        }
        // Without a system event handler, the queue callback is the only
        // fault notification available to the application.
    }
    pending_callback(runtime, key, status)
}

/// Publishes a new owner only after the control update completes. The
/// candidate is retained before the control update, so failure or
/// unwind cannot release backing that the command processor may have observed.
fn update_scratch_owner<T>(
    current: &mut Option<T>,
    uncertain: &mut Vec<T>,
    candidate: T,
    update: impl FnOnce() -> Result<(), Status>,
) -> Result<(), Status> {
    uncertain.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
    uncertain.push(candidate);
    update()?;
    *current = uncertain.pop();
    Ok(())
}

fn handle_queue_event(
    runtime: &mut crate::runtime::Runtime,
    key: usize,
    observed: SignalValue,
) -> Option<QueueEventOutcome> {
    let error = observed as u64;
    if error & 0x401 == 0 {
        return Some(QueueEventOutcome {
            callback: pending_error_callback(runtime, key, queue_error_status(error)),
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
        update_scratch_owner(
            &mut queue.scratch,
            &mut queue.uncertain_scratch,
            allocation,
            || {
                // SAFETY: The command processor's inactive signal reports a
                // stopped queue's scratch fault. Both old and candidate
                // allocations remain owned if the update fails or unwinds.
                unsafe { queue.native.set_scratch(scratch) }.map_err(map_error)
            },
        )?;
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
            let callback = pending_error_callback(runtime, key, queue_error_status(error as u64));
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
            let _scope = CallbackScope::enter();
            // SAFETY: HSA requires callback and data to remain valid through
            // queue destruction; the worker only invokes it for a live record.
            unsafe {
                (callback.callback)(
                    callback.status,
                    callback.queue as *mut HsaQueue,
                    callback.data.as_ptr(),
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
/// # Safety
/// `queue` must point to writable output storage for the complete call. The
/// callback and its data must remain valid until native queue teardown and
/// completion of the final event callback.
unsafe fn create_hardware_queue(
    runtime: &mut crate::runtime::Runtime,
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
    let Some(index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let cooperative = queue_type == QUEUE_TYPE_COOPERATIVE;
    if cooperative {
        if runtime.gpus[index].info.gws_count == 0 {
            return INVALID_QUEUE_CREATION;
        }
        if runtime.cooperative_teardown.contains(&agent.handle) {
            return OUT_OF_RESOURCES;
        }
        if let Some((public, record)) = runtime
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
            if let Some(mask) = cu_mask {
                if let Err(error) = record.native.set_cu_mask(&mask) {
                    return map_error(error);
                }
                record.cu_mask = mask;
            }
            record.cooperative_refs = refs;
            // SAFETY: This queue is still registered and the caller supplied
            // writable output for the returned shared public handle.
            unsafe { queue.write(*public as *mut HsaQueue) };
            return SUCCESS;
        }
    }
    // ROCr's cooperative queue uses one internal 16 KiB AQL ring without
    // caller-specific scratch or callback state. AMD queue descriptors can
    // still adjust its priority and CU mask.
    let (size, callback, data, private_segment_size) = if cooperative {
        (
            16_384 / AQL_PACKET_BYTES as u32,
            None,
            // SAFETY: A null callback argument has no referent to retain.
            unsafe { CallbackArg::new(std::ptr::null_mut()) },
            0,
        )
    } else {
        (size, callback, data, private_segment_size)
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
    // From CREATE onward the command processor may retain both external
    // addresses. Stage every allocation needed for public registration before
    // that call so successful creation can be published without another
    // allocation.
    if runtime.queues.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let mut doorbell = Box::new(AmdSignal::doorbell(0, 0));
    let event_alive = Arc::new(AtomicBool::new(true));
    let worker_alive = event_alive.clone();
    let worker_signal = inactive_signal.clone();
    let stop = runtime.stop_workers.clone();
    let saved_cu_mask = cu_mask.unwrap_or_default();
    let (native, (inactive_signal, scratch_allocation)) =
        match runtime.gpus[index].device.gpu().and_then(|gpu| {
            // SAFETY: The supplied owner tuple retains the AQL event
            // signals and scratch backing through native creation and, on
            // success, remains with the queue until its destruction.
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
                    (inactive_signal, scratch_allocation),
                )
            }
        }) {
            Ok(created) => created,
            Err(error) => {
                return if cooperative && error.kind() == rocddi::ErrorKind::Busy {
                    OUT_OF_RESOURCES
                } else if error.kind() == rocddi::ErrorKind::Unsupported {
                    INVALID_QUEUE_CREATION
                } else {
                    map_error(error)
                };
            }
        };
    // An unwind in frontend setup must not drop backing still reachable by
    // this unpublished native queue. Explicit rejection uses the native
    // abandonment transaction below to release it only after DESTROY succeeds.
    let mut pending = ManuallyDrop::new((native, inactive_signal, scratch_allocation));
    if !saved_cu_mask.is_empty() {
        if let Err(error) = pending.0.set_cu_mask(&saved_cu_mask) {
            let (native, inactive_signal, scratch_allocation) = ManuallyDrop::into_inner(pending);
            return abandon_unpublished_queue(
                native,
                (inactive_signal, scratch_allocation),
                map_error(error),
            );
        }
    }
    let info = pending.0.info();
    if info.write_index_host_address < WRITE_INDEX_OFFSET
        || info.read_index_host_address < READ_INDEX_OFFSET
    {
        let (native, inactive_signal, scratch_allocation) = ManuallyDrop::into_inner(pending);
        return abandon_unpublished_queue(native, (inactive_signal, scratch_allocation), ERROR);
    }
    let public = info.write_index_host_address - WRITE_INDEX_OFFSET;
    if info.read_index_host_address - READ_INDEX_OFFSET != public
        || public % 256 != 0
        || runtime.queues.contains_key(&public)
    {
        let (native, inactive_signal, scratch_allocation) = ManuallyDrop::into_inner(pending);
        return abandon_unpublished_queue(native, (inactive_signal, scratch_allocation), ERROR);
    }
    *doorbell = AmdSignal::doorbell(info.doorbell_host_address, public);
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
    let (native, inactive_signal, scratch_allocation) = ManuallyDrop::into_inner(pending);
    runtime.queues.insert(
        public,
        Queue {
            native,
            _doorbell: doorbell,
            inactive_signal,
            event_alive,
            event_worker: None,
            scratch: scratch_allocation,
            uncertain_scratch: Vec::new(),
            agent,
            hardware_id,
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
    runtime.released_counted_queues.remove(&public);
    let Ok(event_worker) = thread::Builder::new()
        .name("rocddi-queue-events".into())
        .spawn(move || queue_event_worker(&stop, &worker_alive, &worker_signal, public))
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
    *created_log = Some((id, public));
    SUCCESS
}

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
unsafe fn create_sdma_queue(
    runtime: &mut crate::runtime::Runtime,
    agent: HsaAgent,
    size_bytes: u32,
    selection: SdmaEngineSelection,
    ring_memory: QueueRingMemory,
    queue: *mut *mut HsaQueue,
    created_log: &mut Option<(u64, usize, u32)>,
) -> Status {
    let Some(index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let id = match runtime.allocate_queue_id() {
        Ok(id) => id,
        Err(status) => return status,
    };
    if runtime.sdma_queues.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let mut control = Box::new(SdmaQueueControl {
        read_index_host_address: 0,
        write_index_host_address: 0,
        _padding: [0; SDMA_QUEUE_HANDLE_BIT - 2 * std::mem::size_of::<usize>()],
        header: [0; std::mem::size_of::<HsaQueue>()],
    });
    let mut doorbell = Box::new(AmdSignal::doorbell(0, 0));
    let native = match runtime.gpus[index].device.gpu().and_then(|gpu| {
        // SAFETY: SDMA creation passes no external GPU addresses. The queue
        // stays unpublished until its complete transport is validated below.
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
            return INVALID_QUEUE_CREATION;
        }
        Err(error) => return map_error(error),
    };
    let info = native.info();
    let Some(engine_id) = validated_sdma_engine(info, size_bytes) else {
        return abandon_unpublished_queue(native, (), INVALID_QUEUE_CREATION);
    };
    let public = control.header.as_mut_ptr().cast::<HsaQueue>();
    let public_key = public as usize;
    if runtime.sdma_queues.contains_key(&public_key) {
        return abandon_unpublished_queue(native, (), ERROR);
    }
    control.read_index_host_address = info.read_index_host_address;
    control.write_index_host_address = info.write_index_host_address;
    *doorbell = AmdSignal::doorbell(info.doorbell_host_address, public_key);
    // SAFETY: The public header has HsaQueue alignment and remains owned by
    // this record until successful native queue destruction.
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
    runtime.sdma_queues.insert(
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
    // SAFETY: The caller supplied writable output storage and the registry
    // now owns the native queue and both public allocations.
    unsafe { queue.write(public) };
    *created_log = Some((id, public_key, engine_id));
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
        let mut created_log = None;
        // SAFETY: The HSA caller supplies writable queue output and retains callback data.
        let status = unsafe {
            create_hardware_queue(
                runtime,
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
            runtime.prepare_log(
                AMD_LOG_FLAG_INFO,
                format_args!(
                    "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={size}",
                    agent.handle
                ),
            )
        });
        drop(guard);
        if let Some(log) = log {
            log.write();
        }
        status
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

        let mut created_log = None;
        let hardware_queue = if create_new {
            let mut created = std::ptr::null_mut();
            let counted_queue_size = runtime.counted_queue_size;
            // SAFETY: This stack output is writable and the HSA caller retains callback data.
            let status = unsafe {
                create_hardware_queue(
                    runtime,
                    agent,
                    counted_queue_size,
                    queue_type,
                    native_priority,
                    callback,
                    // SAFETY: The C caller retains and synchronizes callback data
                    // through queue destruction and the last event callback.
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
        let log = created_log.and_then(|(id, public)| {
            runtime.prepare_log(
                AMD_LOG_FLAG_INFO,
                format_args!(
                    "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={}",
                    agent.handle, runtime.counted_queue_size
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
        if runtime.soft_queues.try_reserve(1).is_err() {
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
        let id = match runtime.allocate_queue_id() {
            Ok(id) => id,
            Err(status) => return status,
        };
        // The queue publishes this handle in its header. Retain its backing
        // even if the caller destroys the public signal first.
        if let Err(status) = runtime.retain_async_signal(doorbell_signal) {
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
        runtime.soft_queues.insert(
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

struct QueueCreateInput {
    version: u16,
    flags: u16,
    engine_type: u8,
    reserved_header: [u8; 3],
    queue_size_bytes: u32,
    priority: u32,
    callback: QueueErrorCallback,
    callback_data: *mut c_void,
    traffic_class: u32,
    reserved: [u8; 20],
}

/// # Safety
/// The caller supplies a readable descriptor with initialized input fields.
/// Its output-only queue field may be uninitialized and is never read here.
unsafe fn read_queue_create_input(pointer: *const HsaAmdQueueCreateDesc) -> QueueCreateInput {
    // SAFETY: Raw reads avoid referencing the uninitialized output field.
    unsafe {
        QueueCreateInput {
            version: (&raw const (*pointer).version).read(),
            flags: (&raw const (*pointer).flags).read(),
            engine_type: (&raw const (*pointer).engine_type).read(),
            reserved_header: (&raw const (*pointer).reserved_header).read(),
            queue_size_bytes: (&raw const (*pointer).queue_size_bytes).read(),
            priority: (&raw const (*pointer).priority).read(),
            callback: (&raw const (*pointer).callback).read(),
            callback_data: (&raw const (*pointer).callback_data).read(),
            traffic_class: (&raw const (*pointer).traffic_class).read(),
            reserved: (&raw const (*pointer).reserved).read(),
        }
    }
}

/// # Safety
/// The caller supplies a nonzero `count` of readable, aligned words at a
/// non-null `pointer`.
unsafe fn copy_cu_mask(pointer: *const u32, count: usize) -> Result<Vec<u32>, Status> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    // SAFETY: The caller keeps the validated mask readable through this copy.
    owned.extend_from_slice(unsafe { std::slice::from_raw_parts(pointer, count) });
    Ok(owned)
}

/// # Safety
/// A nonzero mask count is a multiple of 32 and its pointer addresses that
/// many readable words. The caller validates both fields first.
unsafe fn copy_queue_create_mask(
    compute: HsaAmdComputeQueueParams,
) -> Result<Option<Vec<u32>>, Status> {
    if compute.cu_mask_count == 0 {
        return Ok(None);
    }
    let count = compute.cu_mask_count as usize / 32;
    // SAFETY: The caller validated the non-null mask and its word count.
    unsafe { copy_cu_mask(compute.cu_mask, count) }.map(Some)
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
                        runtime,
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
                    if let Some(log) = runtime.prepare_log(
                        AMD_LOG_FLAG_INFO,
                        format_args!(
                            "created SDMA queue id={id} agent=0x{:x} address=0x{public:x} engine={engine} bytes={}",
                            agent.handle, input.queue_size_bytes
                        ),
                    ) {
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
                    runtime,
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
                if let Some(log) = runtime.prepare_log(
                    AMD_LOG_FLAG_INFO,
                    format_args!(
                        "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={packet_count}",
                        agent.handle
                    ),
                ) {
                    if logs.try_reserve(1).is_ok() {
                        logs.push(log);
                    }
                }
            }
        }
        drop(guard);
        for log in logs {
            log.write();
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
        let selected = if bit_count == 0 {
            let compute_units = runtime.gpus[gpu_index].info.compute_unit_count;
            full_cu_mask(compute_units)
        } else {
            // SAFETY: The ABI requires bit_count / 32 readable words. Own them
            // before native queue control can mutate caller-visible storage.
            match unsafe { copy_cu_mask(mask, bit_count as usize / 32) } {
                Ok(selected) => selected,
                Err(status) => return status,
            }
        };
        let Some(record) = runtime.queues.get_mut(&hardware_key) else {
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
        if runtime.sdma_queues.contains_key(&(queue as usize)) {
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
                // A shared cooperative queue must not be returned by a later
                // create once any holder starts inactivating it.
                record.inactivated = true;
                return record
                    .native
                    .inactivate()
                    .map_or_else(map_error, |()| SUCCESS);
            }
        }
        if let Some(record) = runtime.sdma_queues.get_mut(&(queue as usize)) {
            return record
                .native
                .inactivate()
                .map_or_else(map_error, |()| SUCCESS);
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
        let key = queue as usize;
        let (mut record, cooperative_agent) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let cooperative_agent = if let Some(record) = runtime.queues.get_mut(&key) {
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
            if let Some(agent) = cooperative_agent {
                if runtime.cooperative_teardown.try_reserve(1).is_err() {
                    return OUT_OF_RESOURCES;
                }
                runtime.cooperative_teardown.insert(agent);
            }
            let record = if let Some(record) = runtime.queues.remove(&key) {
                RemovedQueue::Aql(record)
            } else if let Some(record) = runtime.sdma_queues.remove(&key) {
                RemovedQueue::Sdma(record)
            } else {
                return if let Some(soft) = runtime.soft_queues.remove(&key) {
                    runtime.release_async_signal(soft.doorbell_signal);
                    SUCCESS
                } else {
                    INVALID_QUEUE
                };
            };
            (record, cooperative_agent)
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
                    guard.as_mut().and_then(|runtime| {
                        if let Some(agent) = cooperative_agent {
                            runtime.cooperative_teardown.remove(&agent);
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
                    std::mem::forget(record);
                    return status;
                };
                let Some(runtime) = guard.as_mut() else {
                    std::mem::forget(record);
                    return status;
                };
                if let Some(agent) = cooperative_agent {
                    runtime.cooperative_teardown.remove(&agent);
                }
                match record {
                    RemovedQueue::Aql(record) => {
                        runtime.queues.insert(key, record);
                    }
                    RemovedQueue::Sdma(record) => {
                        runtime.sdma_queues.insert(key, record);
                    }
                }
                status
            }
        }
    })
}

/// Applies one atomic index operation without exporting a queue reference.
///
/// # Safety
/// The public queue handle and its index mapping must remain live throughout
/// `operation`; the caller must prevent concurrent destruction.
#[inline]
unsafe fn with_queue_index<R>(
    queue: *const HsaQueue,
    offset: usize,
    operation: impl for<'a> FnOnce(&'a AtomicU64) -> R,
) -> Option<R> {
    if queue.is_null() {
        return None;
    }
    let address = if queue as usize & COUNTED_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live counted handle points at CountedQueuePublic::header,
        // whose preceding word holds the stable hardware queue address.
        let hardware = unsafe {
            (*queue
                .cast::<u8>()
                .sub(COUNTED_QUEUE_HANDLE_BIT)
                .cast::<CountedQueuePublic>())
            .hardware_queue
        };
        hardware.checked_add(offset)?
    } else if queue as usize & SDMA_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live SDMA header begins 128 bytes into its private
        // control. The private words borrow rocddi's aligned index mappings.
        let control = unsafe {
            &*queue
                .cast::<u8>()
                .sub(SDMA_QUEUE_HANDLE_BIT)
                .cast::<SdmaQueueControl>()
        };
        match offset {
            READ_INDEX_OFFSET => control.read_index_host_address,
            WRITE_INDEX_OFFSET => control.write_index_host_address,
            _ => return None,
        }
    } else {
        (queue as usize).checked_add(offset)?
    };
    // SAFETY: The caller retains a live queue handle. AQL and soft controls
    // contain indices at the ABI offsets; counted and SDMA handles resolve to
    // the native index mapping retained by their queue owner.
    let index = unsafe { &*(address as *const AtomicU64) };
    Some(operation(index))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_read_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, READ_INDEX_OFFSET, |index| {
            index.load(Ordering::Relaxed)
        })
    }
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_read_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, READ_INDEX_OFFSET, |index| {
            index.load(Ordering::Acquire)
        })
    }
    .unwrap_or(0)
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
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.load(Ordering::Relaxed)
        })
    }
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_write_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.load(Ordering::Acquire)
        })
    }
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_queue_load_write_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_write_index_scacquire(queue) }
}

unsafe fn queue_store(queue: *const HsaQueue, offset: usize, value: u64, order: Ordering) {
    // SAFETY: The caller owns a live queue while accessing its index.
    let _ = unsafe { with_queue_index(queue, offset, |index| index.store(value, order)) };
}

unsafe fn queue_compare_exchange(
    queue: *const HsaQueue,
    expected: u64,
    value: u64,
    success: Ordering,
    failure: Ordering,
) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index
                .compare_exchange(expected, value, success, failure)
                .unwrap_or_else(|observed| observed)
        })
    }
    .unwrap_or(0)
}

unsafe fn queue_add(queue: *const HsaQueue, value: u64, order: Ordering) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.fetch_add(value, order)
        })
    }
    .unwrap_or(0)
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
        if let Some(record) = runtime.sdma_queues.get(&input_key) {
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
        let counted = runtime.counted_queues.get(&input_key);
        let hardware_key = counted.map_or(input_key, |queue| queue.hardware_queue);
        let Some(record) = runtime.queues.get(&hardware_key) else {
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
            if !runtime.owns_signal(signal) {
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

#[unsafe(no_mangle)]
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
            if !runtime.owns_signal(signal) {
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
            let Some(profile) = runtime
                .async_signal_refs
                .get(&(signal.handle as usize))
                .and_then(|record| record.copy_profile)
            else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct DropCount(Arc<AtomicUsize>);

    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Release);
        }
    }

    #[test]
    fn failed_scratch_update_retains_both_old_and_candidate_owners() {
        let old_drops = Arc::new(AtomicUsize::new(0));
        let new_drops = Arc::new(AtomicUsize::new(0));
        let mut current = Some(DropCount(old_drops.clone()));
        let mut uncertain = Vec::new();
        assert_eq!(
            update_scratch_owner(
                &mut current,
                &mut uncertain,
                DropCount(new_drops.clone()),
                || Err(ERROR),
            ),
            Err(ERROR)
        );
        assert_eq!(old_drops.load(Ordering::Acquire), 0);
        assert_eq!(new_drops.load(Ordering::Acquire), 0);
        assert_eq!(uncertain.len(), 1);
        drop(current);
        drop(uncertain);
        assert_eq!(old_drops.load(Ordering::Acquire), 1);
        assert_eq!(new_drops.load(Ordering::Acquire), 1);
    }

    #[test]
    #[allow(clippy::panic)]
    fn scratch_update_unwind_keeps_candidate_owned() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut current = None;
        let mut uncertain = Vec::new();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = update_scratch_owner(
                &mut current,
                &mut uncertain,
                DropCount(drops.clone()),
                || -> Result<(), Status> { panic!("injected scratch update unwind") },
            );
        }));
        assert!(outcome.is_err());
        assert_eq!(drops.load(Ordering::Acquire), 0);
        assert_eq!(uncertain.len(), 1);
        drop(uncertain);
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

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

        assert!(stop_queue_event_worker(&alive, &mut worker).is_ok());

        assert!(worker.is_none());
        assert!(finished.load(Ordering::Acquire));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn queue_event_worker_keeps_its_join_handle_for_later_cleanup() {
        let alive = Arc::new(AtomicBool::new(true));
        let worker_slot = Arc::new(std::sync::Mutex::new(None));
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker_alive = alive.clone();
        let worker_slot_clone = worker_slot.clone();
        let worker = thread::spawn(move || {
            let result = loop {
                let mut slot = worker_slot_clone.lock().unwrap();
                if slot.is_some() {
                    break stop_queue_event_worker(&worker_alive, &mut slot);
                }
                drop(slot);
                thread::yield_now();
            };
            sender.send(result.map_err(|error| error.kind())).unwrap();
        });
        *worker_slot.lock().unwrap() = Some(worker);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            Err(rocddi::ErrorKind::Busy)
        );
        assert!(!alive.load(Ordering::Acquire));
        let worker = worker_slot.lock().unwrap().take().unwrap();
        worker.join().unwrap();
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
            assert_eq!((*write).load(Ordering::Relaxed), 25);
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
    fn queue_create_input_does_not_read_the_uninitialized_output() {
        let mut descriptor = std::mem::MaybeUninit::<HsaAmdQueueCreateDesc>::uninit();
        let pointer = descriptor.as_mut_ptr();
        // SAFETY: Every input field is initialized below. The reader only
        // copies those fields, leaving the queue output untouched.
        unsafe {
            (&raw mut (*pointer).version).write(AMD_QUEUE_CREATE_DESC_VERSION);
            (&raw mut (*pointer).flags).write(0);
            (&raw mut (*pointer).engine_type).write(AMD_QUEUE_ENGINE_COMPUTE);
            (&raw mut (*pointer).reserved_header).write([0; 3]);
            (&raw mut (*pointer).queue_size_bytes).write(4096);
            (&raw mut (*pointer).priority).write(AMD_QUEUE_PRIORITY_NORMAL);
            (&raw mut (*pointer).callback).write(None);
            (&raw mut (*pointer).callback_data).write(std::ptr::null_mut());
            (&raw mut (*pointer).engine).write(HsaAmdQueueEngineParams {
                compute: HsaAmdComputeQueueParams {
                    cu_mask: std::ptr::null(),
                    queue_type: QUEUE_TYPE_MULTI,
                    private_segment_size: 0,
                    cu_mask_count: 0,
                    reserved: [0; 3],
                },
            });
            (&raw mut (*pointer).traffic_class).write(0);
            (&raw mut (*pointer).reserved).write([0; 20]);
            let input = read_queue_create_input(pointer);
            assert_eq!(input.version, AMD_QUEUE_CREATE_DESC_VERSION);
            assert_eq!(input.queue_size_bytes, 4096);
        }
    }

    #[test]
    fn queue_create_copies_a_mask_inside_the_descriptor() -> Result<(), Status> {
        let mut descriptor = HsaAmdQueueCreateDesc {
            version: AMD_QUEUE_CREATE_DESC_VERSION,
            flags: 0,
            engine_type: AMD_QUEUE_ENGINE_COMPUTE,
            reserved_header: [0; 3],
            queue_size_bytes: 4096,
            priority: AMD_QUEUE_PRIORITY_NORMAL,
            callback: None,
            callback_data: std::ptr::null_mut(),
            queue: std::ptr::null_mut(),
            engine: HsaAmdQueueEngineParams { reserved: [0; 32] },
            traffic_class: 0,
            reserved: [0; 20],
        };
        descriptor.engine = HsaAmdQueueEngineParams {
            compute: HsaAmdComputeQueueParams {
                cu_mask: &raw const descriptor.queue_size_bytes,
                queue_type: QUEUE_TYPE_MULTI,
                private_segment_size: 0,
                cu_mask_count: 32,
                reserved: [0; 3],
            },
        };
        // SAFETY: The mask points to the initialized u32 field in this live
        // descriptor. No mutable reference to the descriptor is active.
        let copied = unsafe { copy_queue_create_mask(descriptor.engine.compute) }?;
        // SAFETY: The output field is writable and separate from the owned mask.
        unsafe { (&raw mut descriptor.queue).write(std::ptr::null_mut()) };
        assert_eq!(copied, Some(vec![4096]));
        Ok(())
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
