// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Tool-owned AQL interception over a separately retained hardware queue.
//!
//! The public queue exposes coherent host and device visible control and packet
//! storage. A worker observes device doorbells; host doorbell stores also drain
//! synchronously. Rewriters see one published packet at a time and may emit
//! zero or more packets through their writer. Forwarding serializes all output
//! into the underlying queue and retries when its ring is full.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rocddi::device::Device;
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{CallbackScope, boundary, initialized_mut, lock, map_error};
use crate::signal::AmdSignal;

const PACKET_BYTES: usize = 64;
const CONTROL_BYTES: u64 = 4096;
// amd_queue_v2_t: 256-byte queue prefix and 128 per-XCC 16-byte index pairs.
const CONTROL_ABI_BYTES: usize = 2304;
const READ_INDEX_OFFSET: usize = 128;
const WRITE_INDEX_OFFSET: usize = 56;
const SIGNAL_OFFSET: usize = 3072;
const PACKET_TYPE_MASK: u16 = 0xff;
const PACKET_TYPE_INVALID: u16 = 1;
const PACKET_TYPE_VENDOR: u16 = 0;
const INTERCEPT_MARKER_FORMAT: u8 = 0xfe;

const _: () = assert!(CONTROL_ABI_BYTES <= SIGNAL_OFFSET);
const _: () = assert!(SIGNAL_OFFSET + std::mem::size_of::<AmdSignal>() <= CONTROL_BYTES as usize);

type PacketWriter = unsafe extern "C" fn(*const c_void, u64);
type InterceptHandler = unsafe extern "C" fn(*const c_void, u64, u64, *mut c_void, PacketWriter);
pub(crate) type QueueNotifier = unsafe extern "C" fn(*const HsaQueue, HsaAgent, *mut c_void);

/// One aligned AQL packet copied between the proxy and lower queue.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Packet([u8; PACKET_BYTES]);

/// Vendor packet that invokes a tool callback on the forwarding worker.
#[repr(C)]
struct InterceptMarker {
    /// Vendor packet header published to the proxy ring.
    header: u16,
    /// Marker format discriminator within vendor packets.
    format: u8,
    /// Reserved bytes preserving the 64-byte packet layout.
    reserved: [u8; 5],
    /// Tool callback invoked by the forwarding worker.
    callback: Option<unsafe extern "C" fn(*const InterceptMarker, *mut HsaQueue, u64)>,
    /// Callback arguments stored in the packet.
    user_data: [u64; 6],
}

const _: () = assert!(std::mem::size_of::<InterceptMarker>() == PACKET_BYTES);

impl Packet {
    fn is_marker(&self) -> bool {
        u16::from_le_bytes([self.0[0], self.0[1]]) & PACKET_TYPE_MASK == PACKET_TYPE_VENDOR
            && self.0[2] == INTERCEPT_MARKER_FORMAT
    }
}

/// One registered packet rewrite callback and its borrowed argument.
#[derive(Clone, Copy)]
struct Interceptor {
    /// Tool callback that can rewrite one published packet.
    callback: InterceptHandler,
    /// Caller-owned argument kept alive by tool registration.
    data: CallbackArg,
}

/// Stack state exposed to a synchronous tool packet-writer callback.
struct RewriteFrame {
    /// Snapshot of callbacks that may still run for this packet.
    interceptors: *const Interceptor,
    /// Number of entries in that callback snapshot.
    interceptor_count: usize,
    /// Next callback index, or none when output reaches the lower queue.
    next: Option<usize>,
    /// Public queue index of the packet being rewritten.
    packet_index: u64,
    /// Destination for emitted packets after all callbacks run.
    output: *mut VecDeque<Packet>,
    /// Failure flag shared by nested rewrite frames.
    failed: *mut bool,
}

thread_local! {
    static REWRITE_FRAME: Cell<*mut RewriteFrame> = const { Cell::new(ptr::null_mut()) };
}

unsafe extern "C" fn packet_writer(packets: *const c_void, count: u64) {
    REWRITE_FRAME.with(|slot| {
        let frame = slot.get();
        if frame.is_null() {
            return;
        }
        // SAFETY: A frame is installed only for a synchronous tool callback
        // on this thread. Its stack storage and output remain live until the
        // callback returns.
        let frame = unsafe { &mut *frame };
        let Some(count) = usize::try_from(count).ok() else {
            // SAFETY: The frame retains this writable failure flag.
            unsafe { *frame.failed = true };
            return;
        };
        if packets.is_null() && count != 0 || count.checked_mul(PACKET_BYTES).is_none() {
            // SAFETY: The frame retains this writable failure flag.
            unsafe { *frame.failed = true };
            return;
        }
        if let Some(index) = frame.next {
            // SAFETY: The callback snapshot has interceptor_count entries and
            // its storage is retained for the whole rewrite chain.
            let interceptor = unsafe { *frame.interceptors.add(index) };
            let mut nested = RewriteFrame {
                interceptors: frame.interceptors,
                interceptor_count: frame.interceptor_count,
                next: index.checked_sub(1),
                packet_index: frame.packet_index,
                output: frame.output,
                failed: frame.failed,
            };
            let previous = slot.replace(&raw mut nested);
            let _scope = CallbackScope::enter();
            // SAFETY: Tool registration supplies the callback and data. The
            // packet span is borrowed only through this synchronous call.
            unsafe {
                (interceptor.callback)(
                    packets,
                    count as u64,
                    frame.packet_index,
                    interceptor.data.as_ptr(),
                    packet_writer,
                );
            }
            slot.set(previous);
        } else {
            // SAFETY: The frame owns the output queue exclusively while the
            // rewrite chain runs. The tool supplies count readable packets.
            let output = unsafe { &mut *frame.output };
            if output.try_reserve(count).is_err() {
                // SAFETY: The frame retains this writable failure flag.
                unsafe { *frame.failed = true };
                return;
            }
            for index in 0..count {
                // SAFETY: The tool supplies count consecutive 64-byte AQL
                // packets. An unaligned read accepts any valid C packet array.
                output.push_back(unsafe { packets.cast::<Packet>().add(index).read_unaligned() });
            }
        }
    });
}

/// Serialized forwarding cursor and packets waiting for lower-ring space.
struct InterceptState {
    /// Next public packet index to inspect.
    next_packet: u64,
    /// Rewritten packets not yet published to the lower queue.
    pending: VecDeque<Packet>,
}

/// The proxy queue, its coherent public storage, and its forwarding worker.
pub(crate) struct InterceptQueue {
    /// Retains the public control page exposed to queue producers.
    _control: Allocation,
    /// Retains the public packet ring exposed to queue producers.
    _ring: Allocation,
    /// Hardware queue receiving rewritten packets.
    pub(crate) lower: usize,
    /// Public proxy queue pointer returned to callers.
    pub(crate) public: usize,
    /// Number of packets in the power-of-two proxy ring.
    size: u32,
    /// CPU address of the proxy packet ring.
    ring_address: usize,
    /// CPU address of the proxy doorbell signal.
    signal_address: usize,
    /// Stop flag observed by the forwarding worker.
    active: AtomicBool,
    /// Worker joined before proxy backing is released.
    worker: Mutex<Option<JoinHandle<()>>>,
    /// Serialized read cursor and packets awaiting lower-ring space.
    state: Mutex<InterceptState>,
    /// Snapshot source for packet rewrite callbacks.
    interceptors: Mutex<Arc<Vec<Interceptor>>>,
    /// Error callback for failures reported by the proxy.
    callback: QueueErrorCallback,
    /// Caller-owned error callback argument.
    callback_data: CallbackArg,
}

impl InterceptQueue {
    fn create(
        device: &Device,
        lower: usize,
        size: u32,
        callback: QueueErrorCallback,
        callback_data: CallbackArg,
    ) -> Result<Arc<Self>, Status> {
        let ring_bytes = u64::from(size)
            .checked_mul(PACKET_BYTES as u64)
            .ok_or(INVALID_QUEUE_CREATION)?;
        let ring_bytes_usize = usize::try_from(ring_bytes).map_err(|_| INVALID_QUEUE_CREATION)?;
        let permissions = DeviceAccess::READ | DeviceAccess::WRITE;
        let control = device
            .allocate(
                MemoryKind::System,
                CONTROL_BYTES,
                CONTROL_BYTES,
                permissions,
            )
            .map_err(map_error)?;
        let ring = device
            .allocate(
                MemoryKind::System,
                ring_bytes.max(CONTROL_BYTES),
                CONTROL_BYTES,
                permissions,
            )
            .map_err(map_error)?;
        let control_info = control.info();
        let ring_info = ring.info();
        let public = control_info.host_address.ok_or(OUT_OF_RESOURCES)?;
        let ring_address = ring_info.host_address.ok_or(OUT_OF_RESOURCES)?;
        if control_info.device_address != public as u64
            || ring_info.device_address != ring_address as u64
            || control_info.size < CONTROL_BYTES
            || ring_info.size < ring_bytes
            || public % CONTROL_BYTES as usize != 0
            || ring_address % PACKET_BYTES != 0
        {
            return Err(OUT_OF_RESOURCES);
        }
        // SAFETY: Both owned mappings are writable and have the requested
        // extents. The lower queue remains live during proxy construction.
        unsafe {
            ptr::write_bytes(public as *mut u8, 0, CONTROL_BYTES as usize);
            ptr::copy_nonoverlapping(lower as *const u8, public as *mut u8, CONTROL_ABI_BYTES);
            ptr::write_bytes(ring_address as *mut u8, 0, ring_bytes_usize);
            for index in 0..size as usize {
                ((ring_address + index * PACKET_BYTES) as *mut AtomicU16)
                    .as_ref()
                    .ok_or(ERROR)?
                    .store(PACKET_TYPE_INVALID, Ordering::Relaxed);
            }
            (*(public as *mut HsaQueue)).base_address = ring_address as *mut c_void;
            (*(public as *mut HsaQueue)).doorbell_signal = HsaSignal {
                handle: (public + SIGNAL_OFFSET) as u64,
            };
            ((public + READ_INDEX_OFFSET) as *mut AtomicU64)
                .as_ref()
                .ok_or(ERROR)?
                .store(0, Ordering::Relaxed);
            ((public + WRITE_INDEX_OFFSET) as *mut AtomicU64)
                .as_ref()
                .ok_or(ERROR)?
                .store(0, Ordering::Relaxed);
        }
        let owner = Arc::new(Self {
            _control: control,
            _ring: ring,
            lower,
            public,
            size,
            ring_address,
            signal_address: public + SIGNAL_OFFSET,
            active: AtomicBool::new(true),
            worker: Mutex::new(None),
            state: Mutex::new(InterceptState {
                next_packet: 0,
                pending: VecDeque::new(),
            }),
            interceptors: Mutex::new(Arc::new(Vec::new())),
            callback,
            callback_data,
        });
        // SAFETY: The control allocation contains an aligned, uninitialized
        // signal slot. No producer can observe it until the queue is returned.
        unsafe {
            (owner.signal_address as *mut AmdSignal).write(AmdSignal::intercept_doorbell(
                owner.public,
                Arc::as_ptr(&owner) as usize,
            ));
        }
        Ok(owner)
    }

    pub(crate) fn start(self: &Arc<Self>) -> Result<(), Status> {
        let mut worker_slot = self.worker.lock().map_err(|_| ERROR)?;
        let owner = self.clone();
        let worker = thread::Builder::new()
            .name("rocddi-intercept-queue".into())
            .spawn(move || owner.work())
            .map_err(|_| OUT_OF_RESOURCES)?;
        *worker_slot = Some(worker);
        Ok(())
    }

    pub(crate) fn stop(&self) -> Result<(), Status> {
        self.active.store(false, Ordering::Release);
        let mut worker = self.worker.lock().map_err(|_| ERROR)?;
        if worker
            .as_ref()
            .is_some_and(|worker| worker.thread().id() == thread::current().id())
        {
            return Err(INVALID_RUNTIME_STATE);
        }
        if let Some(worker) = worker.take() {
            let _ = worker.join();
        }
        Ok(())
    }

    pub(crate) fn register(&self, callback: InterceptHandler, data: CallbackArg) -> Status {
        if !self.active.load(Ordering::Acquire) {
            return INVALID_QUEUE;
        }
        let Ok(mut interceptors) = self.interceptors.lock() else {
            return ERROR;
        };
        let mut next = Vec::new();
        if next.try_reserve(interceptors.len() + 1).is_err() {
            return OUT_OF_RESOURCES;
        }
        next.extend_from_slice(&interceptors);
        next.push(Interceptor { callback, data });
        *interceptors = Arc::new(next);
        SUCCESS
    }

    fn work(&self) {
        while self.active.load(Ordering::Acquire) {
            self.drain();
            thread::sleep(Duration::from_micros(20));
        }
    }

    fn drain(&self) {
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        let result = self.drain_locked();
        if let Err(status) = result {
            if self.active.swap(false, Ordering::AcqRel) {
                if let Some(callback) = self.callback {
                    let _scope = CallbackScope::enter();
                    // SAFETY: The queue and callback argument remain owned through
                    // worker join, including this final callback.
                    unsafe {
                        callback(
                            status,
                            self.public as *mut HsaQueue,
                            self.callback_data.as_ptr(),
                        )
                    };
                }
            }
        }
    }

    fn drain_locked(&self) -> Result<(), Status> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(()),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(ERROR),
        };
        while self.active.load(Ordering::Acquire) {
            self.forward_pending(&mut state)?;
            if !state.pending.is_empty() {
                return Ok(());
            }
            // SAFETY: The public control allocation retains both aligned
            // indices. The signal slot is retained by the same allocation.
            let write = unsafe { &*((self.public + WRITE_INDEX_OFFSET) as *const AtomicU64) }
                .load(Ordering::Acquire);
            let doorbell = unsafe { &*(self.signal_address as *const AmdSignal) }
                .value
                .load(Ordering::Acquire);
            if doorbell < 0 {
                return Ok(());
            }
            let end = write
                .min((doorbell as u64).saturating_add(1))
                .min(state.next_packet.saturating_add(u64::from(self.size)));
            if state.next_packet >= end {
                return Ok(());
            }
            let slot = (state.next_packet & u64::from(self.size - 1)) as usize;
            let address = self.ring_address + slot * PACKET_BYTES;
            // SAFETY: The retained ring contains size aligned packet slots.
            // The producer publishes the packet body before its header.
            let header = unsafe { &*(address as *const AtomicU16) }.load(Ordering::Acquire);
            if header & PACKET_TYPE_MASK == PACKET_TYPE_INVALID {
                return Ok(());
            }
            // SAFETY: The acquired non-invalid header grants this consumer
            // exclusive observation until it advances the public read index.
            let packet = unsafe { (address as *const Packet).read() };
            let snapshot = self.interceptors.lock().map_err(|_| ERROR)?.clone();
            let mut failed = false;
            if snapshot.is_empty() {
                state.pending.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
                state.pending.push_back(packet);
            } else {
                let mut frame = RewriteFrame {
                    interceptors: snapshot.as_ptr(),
                    interceptor_count: snapshot.len(),
                    next: snapshot.len().checked_sub(2),
                    packet_index: state.next_packet,
                    output: &raw mut state.pending,
                    failed: &raw mut failed,
                };
                let previous = REWRITE_FRAME.with(|slot| slot.replace(&raw mut frame));
                let interceptor = snapshot[snapshot.len() - 1];
                let _scope = CallbackScope::enter();
                // SAFETY: The packet and callback snapshot remain live for
                // this synchronous tool invocation and its writer chain.
                unsafe {
                    (interceptor.callback)(
                        (&raw const packet).cast(),
                        1,
                        state.next_packet,
                        interceptor.data.as_ptr(),
                        packet_writer,
                    );
                }
                REWRITE_FRAME.with(|slot| slot.set(previous));
            }
            if failed {
                state.pending.clear();
                return Err(OUT_OF_RESOURCES);
            }
            // SAFETY: This slot is still owned by the proxy consumer. Release
            // invalidation lets its producer reuse it after the read advance.
            unsafe { &*(address as *const AtomicU16) }
                .store(PACKET_TYPE_INVALID, Ordering::Release);
            state.next_packet += 1;
            // SAFETY: The control allocation retains this aligned read index.
            unsafe { &*((self.public + READ_INDEX_OFFSET) as *const AtomicU64) }
                .store(state.next_packet, Ordering::Release);
        }
        Ok(())
    }

    fn forward_pending(&self, state: &mut InterceptState) -> Result<(), Status> {
        while self.active.load(Ordering::Acquire) {
            let markers = state
                .pending
                .iter()
                .take_while(|packet| packet.is_marker())
                .count();
            // SAFETY: The lower hardware queue is retained until this worker
            // has stopped. Its public control and ring remain mapped.
            let lower = unsafe { &*(self.lower as *const HsaQueue) };
            let write_index = unsafe { &*((self.lower + WRITE_INDEX_OFFSET) as *const AtomicU64) };
            let read_index = unsafe { &*((self.lower + READ_INDEX_OFFSET) as *const AtomicU64) };
            let write = write_index.load(Ordering::Acquire);
            let Some(packet) = state.pending.get(markers).copied() else {
                for _ in 0..markers {
                    if let Some(marker) = state.pending.pop_front() {
                        self.invoke_marker(marker, write);
                    }
                }
                return Ok(());
            };
            let read = read_index.load(Ordering::Acquire);
            if write.wrapping_sub(read) >= u64::from(self.size) {
                return Ok(());
            }
            if write_index
                .compare_exchange_weak(
                    write,
                    write.wrapping_add(1),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            for _ in 0..markers {
                let Some(marker_packet) = state.pending.pop_front() else {
                    return Err(ERROR);
                };
                self.invoke_marker(marker_packet, write);
            }
            let slot = (write & u64::from(self.size - 1)) as usize;
            let address = lower.base_address as usize + slot * PACKET_BYTES;
            // SAFETY: This worker reserved one native ring slot. Publish the
            // body before its header, then ring the retained MMIO doorbell.
            unsafe {
                ptr::copy_nonoverlapping(
                    packet.0.as_ptr().add(2),
                    (address + 2) as *mut u8,
                    PACKET_BYTES - 2,
                );
                (&*(address as *const AtomicU16)).store(
                    u16::from_le_bytes([packet.0[0], packet.0[1]]),
                    Ordering::Release,
                );
                crate::signal::hsa_signal_store_screlease(
                    lower.doorbell_signal,
                    i64::from_ne_bytes(write.to_ne_bytes()),
                );
            }
            state.pending.pop_front();
        }
        Ok(())
    }

    fn invoke_marker(&self, packet: Packet, write: u64) {
        // SAFETY: The packet has the 64-byte public marker ABI and the lower
        // queue remains live until this worker stops.
        let marker = unsafe { &*((&raw const packet).cast::<InterceptMarker>()) };
        if let Some(callback) = marker.callback {
            let _scope = CallbackScope::enter();
            // SAFETY: The marker callback is a synchronous tool function.
            unsafe { callback(marker, self.lower as *mut HsaQueue, write) };
        }
    }
}

/// Called by the signal store path while the caller retains the queue handle.
///
/// # Safety
/// `owner` must be the stable owner address installed in a live intercept
/// doorbell. The caller serializes queue destruction with signal stores.
pub(crate) unsafe fn ring(owner: usize) {
    if owner != 0 {
        // SAFETY: The owning queue remains live for this public signal store.
        unsafe { &*(owner as *const InterceptQueue) }.drain();
    }
}

/// Stops marker callbacks before removing the wrapped queue from the runtime.
/// A failed native destroy restores both owners for a later retry.
pub(crate) fn destroy(owner: &InterceptQueue) -> Status {
    if let Err(status) = owner.stop() {
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
    let Some(registered) = runtime.queue_state.intercept_queues.remove(&owner.public) else {
        return INVALID_QUEUE;
    };
    let Some(mut lower) = runtime.queue_state.queues.remove(&owner.lower) else {
        runtime
            .queue_state
            .intercept_queues
            .insert(owner.public, registered);
        return ERROR;
    };
    drop(guard);
    let result = crate::queue::destroy_runtime_queue(&mut lower).map_err(map_error);
    if let Err(status) = result {
        let Ok(mut guard) = lock() else {
            std::mem::forget(lower);
            std::mem::forget(registered);
            return status;
        };
        let Some(runtime) = guard.as_mut() else {
            std::mem::forget(lower);
            std::mem::forget(registered);
            return status;
        };
        runtime.queue_state.queues.insert(owner.lower, lower);
        runtime
            .queue_state
            .intercept_queues
            .insert(owner.public, registered);
        return status;
    }
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_queue_intercept_create(
    agent: HsaAgent,
    size: u32,
    queue_type: u32,
    callback: QueueErrorCallback,
    data: *mut c_void,
    private_segment_size: u32,
    group_segment_size: u32,
    queue: *mut *mut HsaQueue,
) -> Status {
    boundary(|| {
        if queue.is_null()
            || size < 4
            || !size.is_power_of_two()
            || !matches!(queue_type, QUEUE_TYPE_MULTI | QUEUE_TYPE_SINGLE)
        {
            return INVALID_ARGUMENT;
        }
        let mut lower = ptr::null_mut();
        // SAFETY: The stack output is writable. The lower queue is unpublished
        // until the proxy is ready, so it starts without a user callback.
        let status = unsafe {
            crate::queue::hsa_queue_create(
                agent,
                size,
                queue_type,
                None,
                ptr::null_mut(),
                private_segment_size,
                group_segment_size,
                &raw mut lower,
            )
        };
        if status != SUCCESS {
            return status;
        }
        let created = (|| -> Result<Arc<InterceptQueue>, Status> {
            let mut guard = lock()?;
            let runtime = initialized_mut(&mut guard)?;
            let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
            runtime
                .queue_state
                .intercept_queues
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
            // SAFETY: The C caller retains callback data until queue
            // destruction and the last callback.
            let owner = InterceptQueue::create(
                &runtime.gpus[index].device,
                lower as usize,
                size,
                callback,
                unsafe { CallbackArg::new(data) },
            )?;
            if !runtime.queue_state.queues.contains_key(&(lower as usize)) {
                return Err(ERROR);
            }
            owner.start()?;
            let lower_record = runtime
                .queue_state
                .queues
                .get_mut(&(lower as usize))
                .ok_or(ERROR)?;
            // SAFETY: The C caller retains callback data through the last
            // queue callback and the proxy remains owned until teardown.
            lower_record
                .attach_intercept(owner.public, callback, unsafe { CallbackArg::new(data) });
            runtime
                .queue_state
                .intercept_queues
                .insert(owner.public, owner.clone());
            Ok(owner)
        })();
        match created {
            Ok(owner) => {
                // SAFETY: The caller supplied writable output storage and the
                // proxy control is retained until queue destruction.
                unsafe { queue.write(owner.public as *mut HsaQueue) };
                SUCCESS
            }
            Err(status) => {
                // SAFETY: This call alone owns the unpublished lower queue.
                // If shutdown already owns it, its runtime stop releases it.
                let _ = unsafe { crate::queue::hsa_queue_destroy(lower) };
                status
            }
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_queue_intercept_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_queue_intercept_register(
    queue: *mut HsaQueue,
    callback: Option<InterceptHandler>,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        let Some(owner) = runtime
            .queue_state
            .intercept_queues
            .get(&(queue as usize))
            .cloned()
        else {
            return INVALID_QUEUE;
        };
        drop(guard);
        // SAFETY: The C caller retains callback data until interception and
        // the final queue callback have completed.
        owner.register(callback, unsafe { CallbackArg::new(data) })
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_runtime_queue_create_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_runtime_queue_create_register(
    callback: Option<QueueNotifier>,
    data: *mut c_void,
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
        if runtime.queue_state.internal_queue_notifier.is_some() {
            return ERROR;
        }
        // SAFETY: This is only retained and delivered for future hidden
        // runtime queues. The caller owns its callback data lifetime.
        runtime.queue_state.internal_queue_notifier =
            callback.map(|callback| (callback, unsafe { CallbackArg::new(data) }));
        SUCCESS
    })
}
