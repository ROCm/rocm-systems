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
//! Atomic accessors project typed AQL control fields for hardware and soft
//! queues, and use native index mappings for SDMA. Destruction first prevents
//! new observation, then stops workers and releases native resources. Partial
//! failure keeps enough state for a safe retry.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use rocddi::device::Device;
use rocddi::device::gpu::queue::{
    AqlControlRegion, MAX_PRIVATE_SEGMENT_BYTES, QueueAccessWidth, QueueErrorEvent,
    QueueParameters, QueuePriority, QueueProducerMode, QueueRequest, QueueRingMemory, QueueScratch,
    QueueTransport, RetainedQueue, SdmaEngineSelection, plan_scratch,
};
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocddi::topology::GpuInfo;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::queue_intercept::{InterceptQueue, QueueNotifier};
use crate::runtime::{CallbackScope, VM_FAULT_CONDVAR, initialized_mut, lock, map_error};
use crate::signal::AmdSignal;

mod create;
mod event;
pub(crate) use create::{create_hardware_queue, create_sdma_queue};
pub(crate) mod index;
#[cfg(test)]
use crate::api::queue::core::*;
use event::queue_event_worker;
#[cfg(test)]
use event::{queue_error_status, update_scratch_owner};

/// AQL packet size in bytes, checked against the public C header by ABI tests.
pub(crate) const AQL_PACKET_BYTES: usize = std::mem::size_of::<HsaKernelDispatchPacket>();
/// AQL packet type for a kernel dispatch.
const AQL_PACKET_TYPE_KERNEL_DISPATCH: u16 = 2;
/// Bits containing the AQL packet type.
const AQL_PACKET_TYPE_MASK: u16 = 0xff;
/// GPU page size in bytes.
const GPU_PAGE_BYTES: u64 = 4096;

/// Serializes counted-pool decisions while native queue creation is unlocked.
pub(crate) static COUNTED_CREATE_GATE: Mutex<()> = Mutex::new(());

/// Dedicated device-visible signals used to stop a queue and report errors.
struct QueueEventSignal {
    /// GPU-visible backing for the inactive and error signals.
    _allocation: Allocation,
    /// Native event and mailbox shared by process queues.
    shared_event: Arc<QueueEventOwner>,
    /// CPU address of the queue inactive signal.
    inactive_address: usize,
    /// CPU address of the queue error signal.
    error_address: usize,
}

/// Process-shared KFD event and mailbox backing queue event signals.
pub(crate) struct QueueEventOwner {
    /// Native event owner retained through queue callback delivery.
    event: crate::platform::event::SignalEvent,
    /// Mailbox address written by device-visible signals.
    mailbox: usize,
    /// Native event ID paired with that mailbox.
    event_id: u32,
}

impl QueueEventSignal {
    /// Finds or creates the shared queue error event for a native device.
    fn shared_event(
        runtime: &mut crate::runtime::HsaState,
    ) -> Result<Arc<QueueEventOwner>, Status> {
        let event = if let Some(event) = &runtime.queue_state.queue_event {
            event.clone()
        } else {
            let (event, mailbox, event_id) =
                runtime.create_signal_event().ok_or(OUT_OF_RESOURCES)?;
            let event = Arc::new(QueueEventOwner {
                event,
                mailbox,
                event_id,
            });
            runtime.queue_state.queue_event = Some(event.clone());
            event
        };
        Ok(event)
    }

    /// Creates an error event and registers the queue with its worker.
    fn create(
        event: Arc<QueueEventOwner>,
        device: &Device,
        host_page_size: usize,
    ) -> Result<Self, Status> {
        let allocation = device
            .allocate(
                MemoryKind::System,
                GPU_PAGE_BYTES.max(host_page_size as u64),
                host_page_size as u64,
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

    /// Returns the native event handle published to the queue.
    fn handle(&self) -> u64 {
        self.inactive_address as u64
    }

    /// Returns a clone of the event used for queue errors.
    fn error_event(&self) -> QueueErrorEvent {
        // SAFETY: This owner retains the complete device-visible signal.
        let value =
            unsafe { std::ptr::addr_of!((*(self.error_address as *const AmdSignal)).value) as u64 };
        self.shared_event.event.queue_error_event(value)
    }

    /// Reads the signal indicating that queue submission has stopped.
    fn load_inactive(&self) -> SignalValue {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.inactive_address as *const AmdSignal) }
            .value
            .load(Ordering::Acquire)
    }

    /// Reads the signal containing the latest native queue error.
    fn load_error(&self) -> SignalValue {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.error_address as *const AmdSignal) }
            .value
            .load(Ordering::Acquire)
    }

    /// Releases the retained queue reference.
    fn release_queue(&self) {
        // SAFETY: This owner retains the initialized signal allocation.
        unsafe { &*(self.inactive_address as *const AmdSignal) }
            .value
            .store(0, Ordering::Release);
    }

    /// Releases the retained error reference.
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
    /// Direct native queue and all retained transport backing.
    pub(crate) native: RetainedQueue<QueueBacking>,
    /// Public or intercept-proxy queue pointer returned to callers.
    pub(crate) public_handle: usize,
    /// Controls when profiler callbacks may mention this queue.
    scratch_event_visibility: ScratchEventVisibility,
    /// Retains the doorbell signal published in the HSA queue header.
    _doorbell: Box<AmdSignal>,
    /// Stop flag observed by the queue event worker.
    event_alive: Arc<AtomicBool>,
    /// Worker joined before the native queue is destroyed.
    event_worker: Option<JoinHandle<()>>,
    /// HSA agent that owns the queue.
    pub(crate) agent: HsaAgent,
    /// Native queue ID used by KFD event reporting.
    pub(crate) hardware_id: u32,
    /// Counted-queue pool to which this queue may be returned.
    pub(crate) counted_pool_key: Option<(u64, u32)>,
    /// Cooperative handles borrowing this native queue.
    pub(crate) cooperative_refs: u32,
    /// Queue ring placement requested at creation.
    ring_memory: QueueRingMemory,
    /// Native execution has been stopped successfully.
    pub(crate) inactivated: bool,
    /// Current compute-unit mask for scratch sizing and updates.
    pub(crate) cu_mask: Vec<u32>,
    /// Caller callback for queue errors.
    callback: QueueErrorCallback,
    /// Borrowed callback argument retained through event delivery.
    callback_data: CallbackArg,
    /// Whether an observed event reported a VM fault.
    pub(crate) vm_faulted: bool,
    /// Last observed GPU fault address.
    pub(crate) vm_fault_address: u64,
    /// Last observed native fault reason flags.
    pub(crate) vm_fault_reason: u32,
    /// Native destruction has begun and can only be resumed.
    teardown_started: bool,
}

/// External GPU addresses retained by the native queue through retirement.
pub(crate) struct QueueBacking {
    /// Inactive signal owned by this queue backing.
    inactive_signal: Arc<QueueEventSignal>,
    /// Optional scratch retained by this queue backing.
    scratch: Option<Allocation>,
    /// Uncertain scratch owned by this queue backing.
    uncertain_scratch: Vec<Allocation>,
}

impl Queue {
    /// Publishes a proxy handle only after its control and forwarding worker
    /// are ready. Queue event callbacks read these fields under the runtime
    /// lock, so they cannot observe a partially installed proxy.
    pub(crate) fn attach_intercept(
        &mut self,
        public_handle: usize,
        callback: QueueErrorCallback,
        callback_data: CallbackArg,
    ) {
        self.public_handle = public_handle;
        self.callback = callback;
        self.callback_data = callback_data;
    }
}

/// Whether scratch lifecycle events may use a public queue handle.
#[derive(Clone, Copy, Eq, PartialEq)]
enum ScratchEventVisibility {
    /// Failed construction has no public queue handle for tool callbacks.
    Unpublished,
    /// Published scratch releases are reported even for initial scratch.
    Published,
}

/// Stops queue event worker.
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

/// Destroys native queue and releases its owned resources.
fn destroy_native_queue(
    native: &mut rocddi::device::gpu::queue::Queue,
    teardown_started: &mut bool,
    callback: &mut dyn FnMut(),
) -> Result<(), rocddi::Error> {
    if !*teardown_started {
        native.inactivate()?;
        *teardown_started = true;
    }
    // SAFETY: Inactivation stopped GPU queue consumption. The HSA caller must
    // not publish through this queue concurrently with its destruction.
    unsafe { native.destroy_with_retained_transport(callback) }
}

/// Destroys runtime queue and releases its owned resources.
pub(crate) fn destroy_runtime_queue(queue: &mut Queue) -> Result<(), rocddi::Error> {
    stop_queue_event_worker(&queue.event_alive, &mut queue.event_worker)?;
    let public = queue.public_handle as *const HsaQueue;
    let published = queue.scratch_event_visibility == ScratchEventVisibility::Published;
    let mut release_scratch = |backing: &mut QueueBacking| {
        if let Some(scratch) = backing.scratch.take() {
            if published {
                crate::tool_events::free_start(public);
                drop(scratch);
                crate::tool_events::free_end(public);
            } else {
                drop(scratch);
            }
        }
    };
    if !queue.teardown_started {
        queue.native.inactivate()?;
        queue.teardown_started = true;
    }
    // SAFETY: Inactivation stopped queue consumption, and the HSA caller has
    // stopped publication. The retained bundle stays live through teardown.
    unsafe {
        queue
            .native
            .destroy_with_retained_transport(&mut release_scratch)
    }
}

/// Destroys runtime SDMA queue and releases its owned resources.
pub(crate) fn destroy_runtime_sdma_queue(queue: &mut SdmaQueue) -> Result<(), rocddi::Error> {
    destroy_native_queue(&mut queue.native, &mut queue.teardown_started, &mut || {})
}

// Native and soft queue controls expose at least 128-byte-aligned handles.
// Put counted handles 64 bytes past that boundary so index operations can
// identify them without consulting the registry. SDMA controls put the public
// header 128 bytes into a 256-byte allocation for the same reason.
/// Tag bit distinguishing counted queue handles.
const COUNTED_QUEUE_HANDLE_BIT: usize = 64;
/// Tag bit distinguishing SDMA queue handles.
const SDMA_QUEUE_HANDLE_BIT: usize = 128;

/// Stable storage for the public prefix copied from a shared hardware queue.
#[repr(C, align(128))]
struct CountedQueuePublic {
    /// Native queue pointer shared by this logical handle.
    hardware_queue: usize,
    /// Keeps the public header at the counted-handle tag offset.
    _padding: [u8; COUNTED_QUEUE_HANDLE_BIT - std::mem::size_of::<usize>()],
    /// HSA queue prefix copied from the native hardware queue.
    header: [u8; std::mem::size_of::<HsaQueue>()],
}

const _: () = assert!(std::mem::offset_of!(CountedQueuePublic, header) == COUNTED_QUEUE_HANDLE_BIT);

/// Logical counted-queue handle borrowing one pooled hardware queue.
pub(crate) struct CountedQueue {
    /// Stable public header allocated for this logical handle.
    public: Box<CountedQueuePublic>,
    /// Shared native queue borrowed from the counted pool.
    pub(crate) hardware_queue: usize,
    /// Agent and priority identifying the pool entry.
    pub(crate) pool_key: (u64, u32),
}

impl CountedQueue {
    /// Wraps a borrowed counted hardware queue with its pool key.
    pub(crate) unsafe fn new(hardware_queue: usize, pool_key: (u64, u32)) -> Self {
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

    /// Returns the stable address of the public HSA queue record.
    pub(crate) fn public_pointer(&mut self) -> *mut HsaQueue {
        self.public.header.as_mut_ptr().cast()
    }
}

/// Hardware queue retained by a counted-queue pool with its live borrower count.
pub(crate) struct CountedHardwareQueue {
    /// Public pointer of the pooled native queue.
    pub(crate) queue: usize,
    /// Logical counted handles currently borrowing that queue.
    pub(crate) use_count: u32,
}

/// Aligned packet storage for a host-only soft queue.
#[repr(C, align(64))]
pub(crate) struct SoftPacket(pub(crate) [u8; AQL_PACKET_BYTES]);

/// Host-only queue whose indices remain valid without a native GPU transport.
pub(crate) struct SoftQueue {
    /// Stable host-only control page.
    pub(crate) _control: Box<SoftQueueControl>,
    /// Host-only AQL packet ring.
    pub(crate) _ring: Vec<SoftPacket>,
    /// Signal used as this queue's public doorbell.
    pub(crate) doorbell_signal: HsaSignal,
    /// Whether host publication may still advance the queue.
    pub(crate) active: bool,
}

/// Fixed-size control page used by a host-only soft queue.
#[repr(C, align(256))]
pub(crate) struct SoftQueueControl(pub(crate) [u64; 32]);

impl SoftQueue {
    /// Marks the public queue inactive before native destruction.
    pub(crate) fn inactivate(&mut self) {
        self.active = false;
    }
}

/// Public SDMA header and borrowed native index mappings. The header is
/// separate from rocddi's byte-index control page, which has no HSA prefix.
#[repr(C, align(256))]
struct SdmaQueueControl {
    /// Host address of the native SDMA read index.
    read_index_host_address: usize,
    /// Host address of the native SDMA write index.
    write_index_host_address: usize,
    /// Keeps the public header at the SDMA-handle tag offset.
    _padding: [u8; SDMA_QUEUE_HANDLE_BIT - 2 * std::mem::size_of::<usize>()],
    /// HSA queue prefix exposed to callers.
    header: [u8; std::mem::size_of::<HsaQueue>()],
}

const _: () = assert!(std::mem::offset_of!(SdmaQueueControl, header) == SDMA_QUEUE_HANDLE_BIT);

/// One SDMA public handle borrowing the ring and indices of its native owner.
pub(crate) struct SdmaQueue {
    /// Direct native queue and its retained transport backing.
    pub(crate) native: rocddi::device::gpu::queue::Queue,
    /// Public HSA control prefix and borrowed index addresses.
    _control: Box<SdmaQueueControl>,
    /// Doorbell signal retained while producers may use it.
    _doorbell: Box<AmdSignal>,
    /// HSA GPU agent that owns this queue.
    pub(crate) agent: HsaAgent,
    /// Native queue ID used for error reporting.
    pub(crate) hardware_id: u32,
    /// Native destruction progress retained across retries.
    pub(crate) teardown_started: bool,
}

/// HSA queue handles and policy kept for one initialized HSA generation.
///
/// Native queue execution and transport backing live in rocddi. This state
/// tracks the public HSA headers, optional interception, and counted handles.
pub(crate) struct QueueState {
    /// Next public queue identity.
    next_queue_id: u64,
    /// Shared native event used for queue error notifications.
    queue_event: Option<Arc<QueueEventOwner>>,
    /// Public AQL queues keyed by queue pointer.
    pub(crate) queues: HashMap<usize, Queue>,
    /// Tool interception proxies keyed by public queue pointer.
    pub(crate) intercept_queues: HashMap<usize, Arc<InterceptQueue>>,
    /// Optional internal hook called for newly created queues.
    pub(crate) internal_queue_notifier: Option<(QueueNotifier, CallbackArg)>,
    /// Agent identities with cooperative queue teardown in progress.
    pub(crate) cooperative_teardown: HashSet<u64>,
    /// Public SDMA queues keyed by queue pointer.
    pub(crate) sdma_queues: HashMap<usize, SdmaQueue>,
    /// Software queues keyed by queue pointer.
    pub(crate) soft_queues: HashMap<usize, SoftQueue>,
    /// Counted queue handles keyed by public queue pointer.
    pub(crate) counted_queues: HashMap<usize, CountedQueue>,
    /// Idle counted hardware queues grouped by agent and priority.
    pub(crate) counted_queue_pools: HashMap<(u64, u32), Vec<CountedHardwareQueue>>,
    /// Public counted queue pointers already released by the caller.
    pub(crate) released_counted_queues: HashSet<usize>,
    /// Maximum counted queues accepted by this HSA generation.
    pub(crate) counted_queue_limit: usize,
    /// Ring size selected for counted hardware queues.
    pub(crate) counted_queue_size: u32,
}

impl QueueState {
    /// Initializes empty queue registries and the next queue ID.
    pub(crate) fn new() -> Self {
        Self {
            next_queue_id: 0,
            queue_event: None,
            queues: HashMap::new(),
            intercept_queues: HashMap::new(),
            internal_queue_notifier: None,
            cooperative_teardown: HashSet::new(),
            sdma_queues: HashMap::new(),
            soft_queues: HashMap::new(),
            counted_queues: HashMap::new(),
            counted_queue_pools: HashMap::new(),
            released_counted_queues: HashSet::new(),
            counted_queue_limit: 4,
            counted_queue_size: 16_384,
        }
    }

    /// Allocates ID for runtime use.
    pub(crate) fn allocate_id(&mut self) -> Result<u64, Status> {
        let id = self.next_queue_id;
        self.next_queue_id = id.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(id)
    }

    /// Stops callback admission and detaches queue workers while queue backing
    /// remains published for legacy tool unload callbacks.
    pub(crate) fn take_callback_workers(
        &mut self,
    ) -> (Vec<JoinHandle<()>>, Vec<Arc<InterceptQueue>>) {
        let mut workers = Vec::new();
        for queue in self.queues.values_mut() {
            queue.event_alive.store(false, Ordering::Release);
            if let Some(worker) = queue.event_worker.take() {
                workers.push(worker);
            }
        }
        let intercepts = self.intercept_queues.values().cloned().collect();
        (workers, intercepts)
    }

    /// Stops every public queue before its native backing can be released.
    /// An uncertain native outcome requires retaining the whole HSA generation
    /// because a queue may still reach its memory and signal objects.
    pub(crate) fn shutdown(&mut self) -> bool {
        let mut intercept_stop_failed = false;
        for intercept in self.intercept_queues.values() {
            intercept_stop_failed |= intercept.stop().is_err();
        }
        if intercept_stop_failed {
            return false;
        }
        let mut queue_cleanup_uncertain = false;
        for (_, mut queue) in self.queues.drain() {
            if destroy_runtime_queue(&mut queue).is_err() {
                std::mem::forget(queue);
                queue_cleanup_uncertain = true;
            }
        }
        for (_, mut queue) in self.sdma_queues.drain() {
            if destroy_runtime_sdma_queue(&mut queue).is_err() {
                std::mem::forget(queue);
                queue_cleanup_uncertain = true;
            }
        }
        if queue_cleanup_uncertain {
            return false;
        }
        self.counted_queues.clear();
        self.intercept_queues.clear();
        self.counted_queue_pools.clear();
        self.released_counted_queues.clear();
        self.queue_event = None;
        self.soft_queues.clear();
        true
    }
}

/// Allocates scratch for runtime use.
fn allocate_scratch(
    device: &Device,
    gpu: GpuInfo,
    private_segment_size: u32,
    lanes_per_wave: u32,
) -> Result<(Allocation, QueueScratch), Status> {
    let plan = plan_scratch(gpu, private_segment_size, lanes_per_wave).map_err(map_error)?;
    let allocation = device
        .gpu()
        .and_then(|gpu| gpu.allocate_queue_scratch(plan.byte_length))
        .map_err(map_error)?;
    let scratch = QueueScratch {
        device_address: allocation.info().device_address,
        byte_length: plan.byte_length,
        maximum_private_segment_byte_length: plan.maximum_private_segment_byte_length,
        lanes_per_wave: plan.lanes_per_wave,
        maximum_wave_count: plan.maximum_wave_count,
    };
    Ok((allocation, scratch))
}

/// Reads the dispatch private-segment requirement from a native queue.
fn dispatch_private_segment(queue: &Queue) -> Option<(u32, u64)> {
    let info = queue.native.info();
    let packet_count = info.ring_size_bytes / AQL_PACKET_BYTES as u64;
    if packet_count == 0
        || !packet_count.is_power_of_two()
        || info.ring_host_address % std::mem::align_of::<HsaKernelDispatchPacket>() != 0
    {
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
        let slot = usize::try_from(slot).ok()?;
        let byte_offset = slot.checked_mul(AQL_PACKET_BYTES)?;
        let packet_address = info.ring_host_address.checked_add(byte_offset)?;
        let packet = packet_address as *const HsaKernelDispatchPacket;
        // SAFETY: Queue transport retains the aligned packet ring. An acquire
        // load of the published header makes the packet body visible.
        let header = unsafe { &*std::ptr::addr_of!((*packet).header).cast::<AtomicU16>() }
            .load(Ordering::Acquire);
        if header & AQL_PACKET_TYPE_MASK != AQL_PACKET_TYPE_KERNEL_DISPATCH {
            continue;
        }
        // SAFETY: The stopped queue retains this complete dispatch packet;
        // its named field is aligned and published before the header.
        let private = unsafe { std::ptr::addr_of!((*packet).private_segment_size).read_volatile() };
        if private != 0 {
            return Some((private, read + offset));
        }
    }
    None
}

/// Finds the native queue key corresponding to a public queue pointer.
pub(crate) fn hardware_queue_key(
    runtime: &crate::runtime::HsaState,
    queue: *const HsaQueue,
) -> Option<usize> {
    let key = queue as usize;
    if let Some(intercept) = runtime.queue_state.intercept_queues.get(&key) {
        return Some(intercept.lower);
    }
    runtime.queue_state.counted_queues.get(&key).map_or_else(
        || runtime.queue_state.queues.contains_key(&key).then_some(key),
        |queue| Some(queue.hardware_queue),
    )
}

/// Builds the pool key used to share counted hardware queues.
pub(crate) fn counted_pool_key(
    runtime: &crate::runtime::HsaState,
    queue: *const HsaQueue,
) -> Option<(u64, u32)> {
    let key = queue as usize;
    runtime.queue_state.counted_queues.get(&key).map_or_else(
        || {
            runtime
                .queue_state
                .queues
                .get(&key)
                .and_then(|queue| queue.counted_pool_key)
        },
        |queue| Some(queue.pool_key),
    )
}

/// Checks whether a public queue pointer is still registered.
pub(crate) fn queue_known(runtime: &crate::runtime::HsaState, queue: *const HsaQueue) -> bool {
    !queue.is_null()
        && (hardware_queue_key(runtime, queue).is_some()
            || runtime
                .queue_state
                .soft_queues
                .contains_key(&(queue as usize)))
}

/// Selects the least-used live hardware queue in a counted pool.
pub(crate) fn least_used_counted_queue(pool: &[CountedHardwareQueue]) -> Option<usize> {
    pool.iter()
        .min_by_key(|entry| entry.use_count)
        .map(|entry| entry.queue)
}

/// Converts an HSA priority value to the native queue priority.
pub(crate) fn queue_priority(priority: u32) -> Option<QueuePriority> {
    match priority {
        AMD_QUEUE_PRIORITY_LOW => Some(QueuePriority::Low),
        AMD_QUEUE_PRIORITY_NORMAL => Some(QueuePriority::Normal),
        AMD_QUEUE_PRIORITY_HIGH => Some(QueuePriority::High),
        _ => None,
    }
}

/// Builds a CU mask with all reported compute units enabled.
pub(crate) fn full_cu_mask(compute_units: u32) -> Vec<u32> {
    let mut mask = vec![u32::MAX; compute_units.div_ceil(32) as usize];
    if let Some(last) = mask.last_mut() {
        let tail = compute_units % 32;
        if tail != 0 {
            *last = (1_u32 << tail) - 1;
        }
    }
    mask
}

/// Initialized input prefix copied from an extensible queue descriptor.
pub(crate) struct QueueCreateInput {
    /// Version of the AMD queue creation descriptor.
    pub(crate) version: u16,
    /// Queue ring and descriptor memory placement flags.
    pub(crate) flags: u16,
    /// Queue engine requested by the caller.
    pub(crate) engine_type: u8,
    /// Reserved storage retained for the native layout.
    pub(crate) reserved_header: [u8; 3],
    /// Requested queue ring capacity in bytes.
    pub(crate) queue_size_bytes: u32,
    /// Scheduling priority requested for the new queue.
    pub(crate) priority: u32,
    /// Asynchronous error callback, if provided by the caller.
    pub(crate) callback: QueueErrorCallback,
    /// Opaque argument passed to the error callback.
    pub(crate) callback_data: *mut c_void,
    /// Traffic class passed to the native queue transport.
    pub(crate) traffic_class: u32,
    /// Reserved storage retained for the native layout.
    pub(crate) reserved: [u8; 20],
}

/// # Safety
/// The caller supplies a readable descriptor with initialized input fields.
/// Its output-only queue field may be uninitialized and is never read here.
pub(crate) unsafe fn read_queue_create_input(
    pointer: *const HsaAmdQueueCreateDesc,
) -> QueueCreateInput {
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
pub(crate) unsafe fn copy_cu_mask(pointer: *const u32, count: usize) -> Result<Vec<u32>, Status> {
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
pub(crate) unsafe fn copy_queue_create_mask(
    compute: HsaAmdComputeQueueParams,
) -> Result<Option<Vec<u32>>, Status> {
    if compute.cu_mask_count == 0 {
        return Ok(None);
    }
    let count = compute.cu_mask_count as usize / 32;
    // SAFETY: The caller validated the non-null mask and its word count.
    unsafe { copy_cu_mask(compute.cu_mask, count) }.map(Some)
}

#[cfg(test)]
mod tests;
