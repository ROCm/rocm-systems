// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA signals, IPC signal storage, waits, groups, and asynchronous handlers.
//!
//! The first 64 bytes of each native signal follow the public AMD signal layout
//! used by queues and device-visible storage. HsaState-owned slabs keep ordinary
//! signal addresses stable, while dedicated rocddi allocations back interrupt
//! and IPC-capable signals. Atomic operations preserve the ordering requested by
//! each ABI entry point rather than strengthening all accesses indiscriminately.
//!
//! One dispatcher serializes asynchronous callbacks. Registration and shutdown
//! communicate through a channel so callbacks and worker joins occur without
//! holding the process-global runtime lock.

use std::collections::{HashMap, hash_map::Entry};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering, fence};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::platform::event::{SignalEvent, SignalEventAddressing, SignalEventService};
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocddi::session::DriverContextLifetime;

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::memory::SignalBatchPool;
use crate::runtime::{CallbackScope, HsaState, InFlightToken, lock, map_error};

pub(crate) mod asynchronous;
mod ipc;
use asynchronous::AsyncDispatcher;
#[cfg(test)]
use asynchronous::*;
pub(crate) mod wait;
#[cfg(test)]
use crate::api::signal::wait::*;
#[cfg(test)]
use crate::api::signal::{atomic::*, core::*};
use ipc::{ImportedIpcSignal, OwnedIpcSignal};
pub(crate) use wait::set_system_frequency;
#[cfg(test)]
use wait::*;

/// Signal size in bytes.
const SIGNAL_BYTES: usize = 64;
/// Number of ordinary signals retained in one stable-address slab.
const SIGNALS_PER_SLAB: usize = 1024;
/// Maximum number of native events kept for signal reuse.
const MAX_POOLED_SIGNAL_EVENTS: usize = 256;
/// Fast path indicating whether any batch needs device-write readback.
static HAS_FLUSH_BATCHES: AtomicBool = AtomicBool::new(false);
/// Address ranges of live signal batches requiring HDP flushes.
static FLUSH_BATCHES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Locks the address ranges whose device writes require an HDP readback.
fn flush_batches() -> MutexGuard<'static, Vec<(usize, usize)>> {
    FLUSH_BATCHES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Checks whether an address lies in a signal batch needing HDP flushes.
pub(crate) fn batch_needs_flush(address: usize) -> bool {
    HAS_FLUSH_BATCHES.load(Ordering::Acquire)
        && flush_batches()
            .iter()
            .any(|&(start, end)| start <= address && address < end)
}

/// Reads back a posted device write without overwriting a later update.
fn flush_device_write(value: &AtomicI64) {
    // KFD's public VRAM mapping can post PCIe writes through HDP. Read back
    // the existing store; repeating it could overwrite an agent's update.
    fence(Ordering::SeqCst);
    let _ = value.load(Ordering::Relaxed);
}

/// Stores a signal value and flushes posted device writes.
pub(crate) fn flush_device_store(value: &AtomicI64, written: SignalValue) {
    fence(Ordering::Release);
    value.store(written, Ordering::Relaxed);
    flush_device_write(value);
}

/// Flushes a signal store when its batch uses device-visible backing.
pub(crate) fn flush_signal_store(signal: &AmdSignal) {
    if batch_needs_flush(std::ptr::from_ref(signal) as usize) {
        flush_device_write(&signal.value);
    }
}

/// Flushes device-written profiling timestamps before the host reads them.
pub(crate) fn flush_signal_timestamps(signal: &AmdSignal) {
    if batch_needs_flush(std::ptr::from_ref(signal) as usize) {
        fence(Ordering::SeqCst);
        let _ = signal.end_ts.load(Ordering::Relaxed);
    }
}

/// Public AMD signal record shared with host code and, when applicable, a GPU.
///
/// Field offsets and the 64-byte extent are ABI requirements. The signal kind
/// determines whether updates are ordinary atomics, queue doorbells, or
/// interrupt-capable event notifications.
#[repr(C, align(64))]
pub(crate) struct AmdSignal {
    /// Signal kind, invalidated atomically while lock-free readers inspect it.
    pub(crate) kind: AtomicI64,
    /// Value updated by host atomics or device signal operations.
    pub(crate) value: AtomicI64,
    /// Native event mailbox or private intercept owner address.
    event_mailbox_ptr: u64,
    /// Native event ID for interrupt-backed signals.
    event_id: u32,
    /// Reserved ABI word.
    reserved1: u32,
    /// GPU start timestamp for profiling queries.
    pub(crate) start_ts: AtomicU64,
    /// GPU end timestamp for profiling queries.
    pub(crate) end_ts: AtomicU64,
    /// Queue pointer for doorbell signals.
    pub(crate) queue_ptr: usize,
    /// Reserved ABI words.
    reserved3: [u32; 2],
}

impl AmdSignal {
    /// Returns the native event ID for an interrupt-backed signal.
    pub(crate) fn event_id(&self) -> u32 {
        self.event_id
    }

    /// Builds an ordinary host-visible HSA signal record.
    pub(crate) fn user(value: SignalValue) -> Self {
        Self::user_with_event(value, 0, 0)
    }

    /// Builds an interrupt-backed signal with its native event address and ID.
    pub(crate) fn interrupt(value: SignalValue, event_mailbox_ptr: usize, event_id: u32) -> Self {
        Self::user_with_event(value, event_mailbox_ptr as u64, event_id)
    }

    /// Initializes the public signal layout for an ordinary or event signal.
    fn user_with_event(value: SignalValue, event_mailbox_ptr: u64, event_id: u32) -> Self {
        Self {
            kind: AtomicI64::new(AMD_SIGNAL_KIND_USER),
            value: AtomicI64::new(value),
            event_mailbox_ptr,
            event_id,
            reserved1: 0,
            start_ts: AtomicU64::new(0),
            end_ts: AtomicU64::new(0),
            queue_ptr: 0,
            reserved3: [0; 2],
        }
    }

    /// Builds a queue doorbell signal containing the queue address.
    pub(crate) fn doorbell(address: usize, queue: usize) -> Self {
        Self {
            kind: AtomicI64::new(AMD_SIGNAL_KIND_DOORBELL),
            value: AtomicI64::new(i64::try_from(address).unwrap_or(0)),
            event_mailbox_ptr: 0,
            event_id: 0,
            reserved1: 0,
            start_ts: AtomicU64::new(0),
            end_ts: AtomicU64::new(0),
            queue_ptr: queue,
            reserved3: [0; 2],
        }
    }

    /// Builds a private intercept doorbell associated with its owner.
    pub(crate) fn intercept_doorbell(queue: usize, owner: usize) -> Self {
        Self {
            kind: AtomicI64::new(AMD_SIGNAL_KIND_LEGACY_DOORBELL),
            value: AtomicI64::new(-1),
            // This signal is private to the intercept queue. The host store
            // path uses this owner while the queue remains live; device stores
            // update value and are collected by the queue worker.
            event_mailbox_ptr: owner as u64,
            event_id: 0,
            reserved1: 0,
            start_ts: AtomicU64::new(0),
            end_ts: AtomicU64::new(0),
            queue_ptr: queue,
            reserved3: [0; 2],
        }
    }

    /// Returns the private owner address recorded in an intercept doorbell.
    pub(crate) fn intercept_owner(&self) -> usize {
        self.event_mailbox_ptr as usize
    }
}

const _: () = {
    assert!(size_of::<AmdSignal>() == SIGNAL_BYTES);
    assert!(std::mem::offset_of!(AmdSignal, kind) == 0);
    assert!(std::mem::offset_of!(AmdSignal, value) == 8);
};

/// Clock domain of a completed asynchronous copy's raw timestamps.
#[derive(Clone, Copy)]
pub(crate) enum AsyncCopyClock {
    /// System clock counter shared with native clock queries.
    System,
    /// GPU clock counter for the indexed activated GPU.
    Gpu(usize),
}

/// Timestamps and clock provenance retained outside the public signal ABI.
#[derive(Clone, Copy)]
pub(crate) struct AsyncCopyProfile {
    /// Clock domain used by both raw timestamps.
    pub(crate) clock: AsyncCopyClock,
    /// Native copy start timestamp.
    pub(crate) start: u64,
    /// Native copy end timestamp.
    pub(crate) end: u64,
}

/// Dispatcher and copy worker references retain signal storage after the
/// public handle dies. A completed copy keeps its timing until this signal is
/// reused or destroyed, even after the worker releases its reference.
pub(crate) struct AsyncSignalRecord {
    /// Internal worker or dispatcher references to this signal slot.
    references: u32,
    /// Sticky retirement flag shared with existing waiters.
    retired: Arc<AtomicBool>,
    /// Completed copy timing retained for profiling queries.
    pub(crate) copy_profile: Option<AsyncCopyProfile>,
}

impl AsyncSignalRecord {
    /// Restores a pooled signal slot to its reusable state.
    fn revive(&mut self) {
        if self.retired.load(Ordering::Acquire) {
            // Existing waiters must keep observing the prior retirement even
            // when a new IPC attachment publishes the same numeric handle.
            self.retired = Arc::new(AtomicBool::new(false));
            self.copy_profile = None;
        }
    }
}

/// Stable device-visible storage from which ordinary signals are carved.
pub(crate) struct SignalSlab {
    /// GPU-visible allocation retaining every signal slot in this slab.
    _allocation: Allocation,
    /// CPU address of the first signal slot.
    host: usize,
    /// Index of the next never-used slot.
    next: usize,
}

/// One contiguous native allocation retained by its public handle and members.
#[allow(dead_code, reason = "the backing is held for its lifetime, not read")]
enum SignalBatchBacking {
    /// Dedicated native allocation backing an interrupt-capable signal.
    Native(Allocation),
    #[cfg(test)]
    /// In-memory signal records used to test batch ownership without KFD.
    Test(Box<[AmdSignal]>),
}

/// Owns native signal backing and tracks free and live slots.
struct SignalBatchStorage {
    /// Native allocation retained for the lifetime of this batch.
    _backing: SignalBatchBacking,
    /// Number of batch slots that have not been allocated.
    remaining: usize,
    /// Whether the public batch handle is still valid.
    handle_live: bool,
    /// Async copy and SVM completion require a host atomic decrement.
    host_atomics_supported: bool,
    /// Optional flush range retained by this signal batch storage.
    flush_range: Option<(usize, usize)>,
}

impl Drop for SignalBatchStorage {
    /// Removes this batch from the set of HDP-flush address ranges.
    fn drop(&mut self) {
        if let Some(range) = self.flush_range {
            let mut ranges = flush_batches();
            ranges.retain(|&registered| registered != range);
            HAS_FLUSH_BATCHES.store(!ranges.is_empty(), Ordering::Release);
        }
    }
}

impl SignalSlab {
    /// Allocates native backing for a device-visible signal batch.
    fn create(runtime: &HsaState) -> Result<Self, Status> {
        let (gpu, other_gpus) = runtime.gpus.split_first().ok_or(OUT_OF_RESOURCES)?;
        let peers = other_gpus
            .iter()
            .map(|peer| &peer.device)
            .collect::<Vec<_>>();
        let allocation = gpu
            .device
            .allocate_with_peers(
                &peers,
                MemoryKind::System,
                (SIGNAL_BYTES * SIGNALS_PER_SLAB).max(runtime.host_page_size) as u64,
                runtime.host_page_size as u64,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(crate::runtime::map_error)?;
        let host = allocation.info().host_address.ok_or(OUT_OF_RESOURCES)?;
        Ok(Self {
            _allocation: allocation,
            host,
            next: 0,
        })
    }

    /// Checks whether the slab has an unused signal slot.
    fn has_space(&self) -> bool {
        self.next < SIGNALS_PER_SLAB
    }

    /// Places an initialized signal in the next free batch slot.
    unsafe fn allocate(&mut self, signal: AmdSignal) -> HsaSignal {
        let address = self.host + self.next * SIGNAL_BYTES;
        self.next += 1;
        // SAFETY: Each bump-allocated slot is aligned, writable, unique, and
        // contained in the live mapped allocation retained by this slab.
        unsafe { (address as *mut AmdSignal).write(signal) };
        HsaSignal {
            handle: address as u64,
        }
    }

    /// Checks whether a live signal occupies this batch address.
    fn owns_live(&self, address: usize) -> bool {
        let end = self.host + self.next * SIGNAL_BYTES;
        if address < self.host || address >= end || (address - self.host) % SIGNAL_BYTES != 0 {
            return false;
        }
        // SAFETY: The bounds and alignment checks identify an initialized slot
        // in this slab, whose allocation remains mapped for the slab lifetime.
        unsafe { &*(address as *const AmdSignal) }
            .kind
            .load(Ordering::Acquire)
            != AMD_SIGNAL_KIND_INVALID
    }
}

/// HSA signal storage and public handle ownership for one generation.
///
/// rocddi owns the native allocations and KFD events stored here. This state
/// tracks HSA signal slots, IPC attachments, waiters, and callback references.
pub(crate) struct SignalState {
    /// Backing slabs for ordinary HSA signals.
    signal_slabs: Vec<SignalSlab>,
    /// Signal slots available for reuse.
    recycled_signals: Vec<usize>,
    /// Contiguous batch backing and each member's owning batch ID.
    batches: HashMap<u64, SignalBatchStorage>,
    /// Batch members owned by this signal state.
    batch_members: HashMap<usize, u64>,
    /// Native page preparation and first-event transaction.
    event_service: SignalEventService,
    /// Native events attached to public interrupt signals.
    interrupt_signals: HashMap<usize, Option<SignalEvent>>,
    /// Native signal events available for reuse.
    signal_event_pool: Vec<SignalEvent>,
    /// IPC signal exports owned by this process.
    owned_ipc_signals: HashMap<usize, OwnedIpcSignal>,
    /// IPC signals imported from another process.
    imported_ipc_signals: HashMap<usize, ImportedIpcSignal>,
    /// Async-handler references keeping signal slots alive.
    async_signal_refs: HashMap<usize, AsyncSignalRecord>,
    /// Public signal groups and their member signals.
    signal_groups: HashMap<u64, Vec<HsaSignal>>,
    /// Background dispatcher for asynchronous signal handlers.
    async_dispatcher: Option<AsyncDispatcher>,
}

impl SignalState {
    /// Reports whether a signal uses native interrupt backing.
    pub(crate) fn is_interrupt_signal(&self, signal: HsaSignal) -> bool {
        self.interrupt_signals
            .contains_key(&(signal.handle as usize))
    }

    /// Retains the members of a newly allocated public signal group.
    pub(crate) fn insert_group(&mut self, handle: u64, signals: Vec<HsaSignal>) {
        self.signal_groups.insert(handle, signals);
    }

    /// Releases a public signal group by handle.
    pub(crate) fn remove_group(&mut self, handle: u64) -> bool {
        self.signal_groups.remove(&handle).is_some()
    }

    /// Destroys batch and releases its owned resources.
    pub(crate) fn destroy_batch(&mut self, batch: HsaAmdSignalBatch) -> Status {
        let Some(storage) = self.batches.get_mut(&batch.handle) else {
            return INVALID_ARGUMENT;
        };
        if !storage.handle_live {
            return INVALID_ARGUMENT;
        }
        storage.handle_live = false;
        if storage.remaining == 0 {
            self.batches.remove(&batch.handle);
        }
        SUCCESS
    }

    /// Initializes signal registries and stores the native context lifetime.
    pub(crate) fn new(lifetime: DriverContextLifetime) -> Self {
        Self {
            signal_slabs: Vec::new(),
            recycled_signals: Vec::new(),
            batches: HashMap::new(),
            batch_members: HashMap::new(),
            event_service: SignalEventService::new(lifetime),
            interrupt_signals: HashMap::new(),
            signal_event_pool: Vec::new(),
            owned_ipc_signals: HashMap::new(),
            imported_ipc_signals: HashMap::new(),
            async_signal_refs: HashMap::new(),
            signal_groups: HashMap::new(),
            async_dispatcher: None,
        }
    }

    /// Releases HSA signal records after workers and native queues have stopped.
    pub(crate) fn clear_storage(&mut self) {
        self.signal_groups.clear();
        self.async_signal_refs.clear();
        self.imported_ipc_signals.clear();
        self.owned_ipc_signals.clear();
        self.recycled_signals.clear();
        self.batch_members.clear();
        self.batches.clear();
        self.signal_slabs.clear();
    }
}

impl HsaState {
    /// Ensures signal slab exists before use.
    fn ensure_signal_slab(&mut self) -> Result<(), Status> {
        if self
            .signal_state
            .signal_slabs
            .last()
            .is_none_or(|slab| !slab.has_space())
        {
            self.signal_state
                .signal_slabs
                .push(SignalSlab::create(self)?);
        }
        Ok(())
    }

    /// Allocates signal for runtime use.
    fn allocate_signal(&mut self, signal: AmdSignal) -> Result<HsaSignal, Status> {
        if let Some(address) = self.signal_state.recycled_signals.pop() {
            // SAFETY: The slot was invalidated after its final asynchronous
            // registration retired. Its slab remains mapped until shutdown.
            unsafe { (address as *mut AmdSignal).write(signal) };
            return Ok(HsaSignal {
                handle: address as u64,
            });
        }
        self.ensure_signal_slab()?;
        let slab = self
            .signal_state
            .signal_slabs
            .last_mut()
            .ok_or(OUT_OF_RESOURCES)?;
        // SAFETY: The selected slab has one unused aligned slot.
        Ok(unsafe { slab.allocate(signal) })
    }

    /// Creates signal batch and records its ownership.
    pub(crate) fn create_signal_batch(
        &mut self,
        values: &[SignalValue],
        interrupt: bool,
        pool: SignalBatchPool,
        consumers: &[HsaAgent],
    ) -> Result<(HsaAmdSignalBatch, Vec<HsaSignal>), Status> {
        let bytes = values
            .len()
            .checked_mul(SIGNAL_BYTES)
            .ok_or(OUT_OF_RESOURCES)?;
        let page = self.host_page_size;
        let allocation_bytes = bytes
            .checked_add(page - 1)
            .map(|size| size & !(page - 1))
            .ok_or(OUT_OF_RESOURCES)?;
        let (owner_index, kind, flush) = match pool {
            SignalBatchPool::System => (0, MemoryKind::System, false),
            SignalBatchPool::Device {
                index,
                extended,
                flush,
            } => (
                index,
                MemoryKind::DeviceLocal {
                    host_visible: true,
                    coherent: extended,
                    extended_coherent: extended,
                    uncached: false,
                    contiguous: false,
                },
                flush,
            ),
        };
        if flush {
            flush_batches()
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
        }
        let owner = self
            .gpus
            .get(owner_index)
            .ok_or(OUT_OF_RESOURCES)?
            .device
            .clone();
        let peers = self
            .gpus
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != owner_index)
            .filter(|(index, _)| {
                matches!(pool, SignalBatchPool::System)
                    || consumers.is_empty()
                    || consumers
                        .iter()
                        .any(|agent| agent.handle == GPU_AGENT_BASE + *index as u64)
            })
            .map(|(_, gpu)| gpu.device.clone())
            .collect::<Vec<_>>();
        let peer_refs = peers.iter().collect::<Vec<_>>();
        let allocation = owner
            .allocate_with_peers(
                &peer_refs,
                kind,
                allocation_bytes as u64,
                page as u64,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(map_error)?;
        let info = allocation.info();
        let host = info.host_address.ok_or(OUT_OF_RESOURCES)?;
        let end = host.checked_add(bytes).ok_or(OUT_OF_RESOURCES)?;
        if info.device_address != host as u64
            || info.size < bytes as u64
            || host % SIGNAL_BYTES != 0
        {
            return Err(OUT_OF_RESOURCES);
        }
        self.signal_state
            .batches
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        self.signal_state
            .batch_members
            .try_reserve(values.len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        if interrupt {
            self.signal_state
                .interrupt_signals
                .try_reserve(values.len())
                .map_err(|_| OUT_OF_RESOURCES)?;
        }
        let mut signals = Vec::new();
        signals
            .try_reserve_exact(values.len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        let batch = HsaAmdSignalBatch {
            handle: self.allocate_handle()?,
        };
        for (index, &value) in values.iter().enumerate() {
            let address = host + index * SIGNAL_BYTES;
            let event = if interrupt {
                self.acquire_signal_event()
            } else {
                None
            };
            let storage = match &event {
                Some((_, mailbox, id)) => AmdSignal::interrupt(value, *mailbox, *id),
                None if interrupt => AmdSignal::interrupt(value, 0, 0),
                None => AmdSignal::user(value),
            };
            // SAFETY: The unique aligned slot is inside the live allocation.
            unsafe { (address as *mut AmdSignal).write(storage) };
            if interrupt {
                self.signal_state
                    .interrupt_signals
                    .insert(address, event.map(|(event, _, _)| event));
            }
            self.signal_state
                .batch_members
                .insert(address, batch.handle);
            signals.push(HsaSignal {
                handle: address as u64,
            });
        }
        let flush_range = if flush {
            let last = signals.last().ok_or(OUT_OF_RESOURCES)?;
            // SAFETY: The initialized last slot remains live in allocation.
            let last_signal = unsafe { &*(last.handle as usize as *const AmdSignal) };
            flush_device_store(&last_signal.value, values[values.len() - 1]);
            flush_batches().push((host, end));
            HAS_FLUSH_BATCHES.store(true, Ordering::Release);
            Some((host, end))
        } else {
            None
        };
        self.signal_state.batches.insert(
            batch.handle,
            SignalBatchStorage {
                _backing: SignalBatchBacking::Native(allocation),
                remaining: values.len(),
                handle_live: true,
                host_atomics_supported: !flush,
                flush_range,
            },
        );
        Ok((batch, signals))
    }

    /// Creates signal event and records its ownership.
    pub(crate) fn create_signal_event(
        &mut self,
    ) -> Option<(crate::platform::event::SignalEvent, usize, u32)> {
        let (gpu, rest) = self.gpus.split_first()?;
        let peers = rest.iter().map(|peer| &peer.device).collect::<Vec<_>>();
        let (event, mailbox) = self
            .signal_state
            .event_service
            .create(&gpu.device, &peers, SignalEventAddressing::Identity)
            .ok()?;
        let id = event.info().kfd_event_id;
        Some((event, mailbox.host, id))
    }

    /// Takes a pooled event or creates one for an interrupt signal.
    fn acquire_signal_event(&mut self) -> Option<(SignalEvent, usize, u32)> {
        // Reuse the KFD event after its previous signal has been destroyed.
        // The stable process event page supplies the same mailbox address.
        if let Some(event) = self.signal_state.signal_event_pool.last() {
            let gpu = self.gpus.first()?;
            let mailbox = self
                .signal_state
                .event_service
                .mailbox(&gpu.device, event)
                .ok()?;
            let id = event.info().kfd_event_id;
            let event = self.signal_state.signal_event_pool.pop()?;
            return Some((event, mailbox.host, id));
        }
        self.create_signal_event()
    }

    /// Creates signal and records its ownership.
    pub(crate) fn create_signal(
        &mut self,
        value: SignalValue,
        interrupt: bool,
    ) -> Result<HsaSignal, Status> {
        if !interrupt {
            return self.allocate_signal(AmdSignal::user(value));
        }
        self.signal_state
            .interrupt_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        let event = self.acquire_signal_event();
        let storage = event.as_ref().map_or_else(
            || AmdSignal::interrupt(value, 0, 0),
            |(_, mailbox, event_id)| AmdSignal::interrupt(value, *mailbox, *event_id),
        );
        let signal = self.allocate_signal(storage)?;
        let owner = event.map(|(event, _, _)| event);
        match self
            .signal_state
            .interrupt_signals
            .entry(signal.handle as usize)
        {
            Entry::Vacant(entry) => {
                entry.insert(owner);
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(signal)
    }
}

impl SignalState {
    /// Publish or clear copy timing only while the runtime registry is locked.
    pub(crate) fn set_copy_profile(
        &mut self,
        signal: HsaSignal,
        profile: Option<AsyncCopyProfile>,
    ) {
        if let Some(record) = self.async_signal_refs.get_mut(&(signal.handle as usize)) {
            record.copy_profile = profile;
        }
    }

    /// Return timing retained with a completed asynchronous signal.
    pub(crate) fn copy_profile(&self, signal: HsaSignal) -> Option<AsyncCopyProfile> {
        self.async_signal_refs
            .get(&(signal.handle as usize))
            .and_then(|record| record.copy_profile)
    }

    /// Checks whether the frontend owns the supplied signal handle.
    pub(crate) fn owns_signal(&self, signal: HsaSignal) -> bool {
        let address = signal.handle as usize;
        if self
            .async_signal_refs
            .get(&address)
            .is_some_and(|record| record.retired.load(Ordering::Acquire))
        {
            return false;
        }
        self.owned_ipc_signals.contains_key(&address)
            || self.batch_members.contains_key(&address)
            || self
                .imported_ipc_signals
                .get(&address)
                .is_some_and(|imported| imported.references != 0)
            || self.signal_slabs.iter().any(|slab| slab.owns_live(address))
    }

    /// Reports whether a live signal has notification backing for callbacks.
    pub(crate) fn supports_async_handler(&self, signal: HsaSignal) -> bool {
        let address = signal.handle as usize;
        self.interrupt_signals.contains_key(&address)
            || self.owned_ipc_signals.contains_key(&address)
            || self.imported_ipc_signals.contains_key(&address)
    }

    /// Rejects batch backings that cannot perform the worker's atomic decrement.
    pub(crate) fn async_completion_status(&self, signal: HsaSignal) -> Status {
        if !self.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        let Some(batch_id) = self.batch_members.get(&(signal.handle as usize)) else {
            return SUCCESS;
        };
        match self.batches.get(batch_id) {
            Some(batch) if batch.host_atomics_supported => SUCCESS,
            Some(_) => NOT_SUPPORTED,
            None => INVALID_SIGNAL,
        }
    }

    /// Retains async signal while dependent work may use it.
    pub(crate) fn retain_async_signal(&mut self, signal: HsaSignal) -> Result<(), Status> {
        let address = signal.handle as usize;
        if !self.async_signal_refs.contains_key(&address) {
            self.async_signal_refs
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
        }
        match self.async_signal_refs.entry(address) {
            Entry::Vacant(entry) => {
                entry.insert(AsyncSignalRecord {
                    references: 1,
                    retired: Arc::new(AtomicBool::new(false)),
                    copy_profile: None,
                });
            }
            Entry::Occupied(mut entry) => {
                let record = entry.get_mut();
                record.references = record.references.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
            }
        }
        Ok(())
    }

    /// Releases the retained async signal reference.
    pub(crate) fn release_async_signal(&mut self, signal: HsaSignal) {
        let address = signal.handle as usize;
        let Some(record) = self.async_signal_refs.get_mut(&address) else {
            return;
        };
        if record.references == 0 {
            return;
        }
        record.references -= 1;
        if record.references != 0 {
            return;
        }
        if record.retired.load(Ordering::Acquire) {
            if self.finish_signal_storage(address) == SUCCESS {
                self.async_signal_refs.remove(&address);
            }
        } else if record.copy_profile.is_none() {
            self.async_signal_refs.remove(&address);
        }
    }

    /// Destroys signal and releases its owned resources.
    pub(crate) fn destroy_signal(&mut self, signal: HsaSignal) -> Status {
        let address = signal.handle as usize;
        if !self.owns_signal(signal) {
            if self.async_signal_refs.get(&address).is_some_and(|record| {
                record.references == 0 && record.retired.load(Ordering::Acquire)
            }) {
                let status = self.finish_signal_storage(address);
                if status == SUCCESS {
                    self.async_signal_refs.remove(&address);
                }
                return status;
            }
            return INVALID_SIGNAL;
        }
        if let Some(imported) = self.imported_ipc_signals.get_mut(&address) {
            if imported.references > 1 {
                imported.references -= 1;
                return SUCCESS;
            }
        }
        if self
            .async_signal_refs
            .get(&address)
            .is_some_and(|record| record.references == 0)
        {
            let status = self.finish_signal_storage(address);
            if status == SUCCESS {
                self.async_signal_refs.remove(&address);
            }
            return status;
        }
        if let Some(record) = self.async_signal_refs.get_mut(&address) {
            // Keep the mapped IPC
            // allocation and any interrupt event live until the last handler
            // returns false, even after the public reference is destroyed.
            record.retired.store(true, Ordering::Release);
            record.copy_profile = None;
            if let Some(imported) = self.imported_ipc_signals.get_mut(&address) {
                imported.references = 0;
            }
            return SUCCESS;
        }
        self.finish_signal_storage(address)
    }

    /// Completes signal storage after its pending work has retired.
    fn finish_signal_storage(&mut self, address: usize) -> Status {
        if let Some(mut imported) = self.imported_ipc_signals.remove(&address) {
            return match imported.allocation.free() {
                Ok(()) => SUCCESS,
                Err(error) => {
                    self.imported_ipc_signals.insert(address, imported);
                    map_error(error)
                }
            };
        }
        if let Some(mut owned) = self.owned_ipc_signals.remove(&address) {
            return match owned.allocation.free() {
                Ok(()) => SUCCESS,
                Err(error) => {
                    self.owned_ipc_signals.insert(address, owned);
                    map_error(error)
                }
            };
        }
        if let Some(mut event) = self.interrupt_signals.remove(&address).flatten() {
            if self.signal_event_pool.len() < MAX_POOLED_SIGNAL_EVENTS
                && self.signal_event_pool.try_reserve(1).is_ok()
            {
                self.signal_event_pool.push(event);
            } else if let Err(error) = event.destroy() {
                self.interrupt_signals.insert(address, Some(event));
                return map_error(error);
            }
        }
        if let Some(batch_id) = self.batch_members.remove(&address) {
            // Native backing is retained until both independent ownership
            // paths have retired, including deferred async signal references.
            // SAFETY: Membership identifies a live initialized batch slot.
            unsafe { &*(address as *const AmdSignal) }
                .kind
                .store(AMD_SIGNAL_KIND_INVALID, Ordering::Release);
            if let Some(batch) = self.batches.get_mut(&batch_id) {
                batch.remaining -= 1;
                if batch.remaining == 0 && !batch.handle_live {
                    self.batches.remove(&batch_id);
                }
            }
            return SUCCESS;
        }
        // Slab storage remains mapped until shutdown. A later creation may
        // reuse the slot once all asynchronous registrations have retired.
        let recycle = self.recycled_signals.try_reserve(1).is_ok();
        // SAFETY: The caller validated this initialized live slab slot.
        unsafe { &*(address as *const AmdSignal) }
            .kind
            .store(AMD_SIGNAL_KIND_INVALID, Ordering::Release);
        if recycle {
            self.recycled_signals.push(address);
        }
        SUCCESS
    }

    /// Destroys events and releases its owned resources.
    pub(crate) fn destroy_events(&mut self) -> Status {
        let mut status = SUCCESS;
        for (_, event) in self.interrupt_signals.drain() {
            if let Some(mut event) = event {
                if let Err(error) = event.destroy() {
                    if status == SUCCESS {
                        status = map_error(error);
                    }
                }
            }
        }
        for mut event in self.signal_event_pool.drain(..) {
            if let Err(error) = event.destroy() {
                if status == SUCCESS {
                    status = map_error(error);
                }
            }
        }
        let unresolved_page = match self.event_service.release() {
            Ok(unresolved) => unresolved,
            Err(error) => {
                if status == SUCCESS {
                    status = map_error(error);
                }
                false
            }
        };
        // An unsuccessful CREATE_EVENT cannot tell us whether KFD installed
        // the page. Another runtime would have its address but not its handle.
        // Quarantine this process instead of reinitializing over that state.
        if unresolved_page && status == SUCCESS {
            status = INVALID_RUNTIME_STATE;
        }
        status
    }
}

/// Rejects a null or duplicate consumer list before signal creation.
pub(crate) unsafe fn valid_consumers(num_consumers: u32, consumers: *const HsaAgent) -> bool {
    if num_consumers == 0 {
        return true;
    }
    if consumers.is_null() {
        return false;
    }
    // SAFETY: The caller supplies num_consumers readable handles.
    let consumers = unsafe { std::slice::from_raw_parts(consumers, num_consumers as usize) };
    !consumers
        .iter()
        .enumerate()
        .any(|(index, agent)| consumers[..index].contains(agent))
}

/// Checks whether AMD signal attributes require a consumer list.
fn amd_signal_uses_consumer_list(attributes: u64) -> bool {
    attributes & (AMD_SIGNAL_AMD_GPU_ONLY | AMD_SIGNAL_IPC) == 0
}

/// Decides whether signal attributes require native interrupt backing.
pub(crate) unsafe fn signal_uses_interrupt(
    cpu_count: usize,
    num_consumers: u32,
    consumers: *const HsaAgent,
    attributes: u64,
) -> Result<bool, Status> {
    if !amd_signal_uses_consumer_list(attributes) {
        return Ok(false);
    }
    // SAFETY: The public ABI promises num_consumers readable handles when nonzero.
    if !unsafe { valid_consumers(num_consumers, consumers) } {
        return Err(INVALID_ARGUMENT);
    }
    if num_consumers == 0 {
        return Ok(true);
    }
    // SAFETY: valid_consumers established this complete readable array.
    let consumers = unsafe { std::slice::from_raw_parts(consumers, num_consumers as usize) };
    Ok(consumers.iter().any(|agent| {
        agent
            .handle
            .checked_sub(CPU_AGENT)
            .is_some_and(|index| index < cpu_count as u64)
    }))
}

/// Finds the record behind one live public signal handle without exporting a
/// reference whose lifetime could outlive the handle's backing.
///
/// # Safety
/// The handle must identify an aligned initialized signal whose owner remains
/// mapped for every use of the returned pointer. Public callers retain signals
/// through their operations; async registrations are retained by the runtime.
unsafe fn signal_ptr(signal: HsaSignal) -> Option<NonNull<AmdSignal>> {
    let pointer = NonNull::new(signal.handle as usize as *mut AmdSignal)?;
    // SAFETY: HSA signal handles created by this frontend are aligned pointers
    // to AmdSignal records retained by the runtime or their owning queue.
    let live = unsafe { pointer.as_ref().kind.load(Ordering::Acquire) };
    (live != AMD_SIGNAL_KIND_INVALID).then_some(pointer)
}

/// Runs one operation against a live signal without returning a borrowed
/// reference. The callback result cannot contain the temporary signal borrow.
///
/// # Safety
/// The caller must keep the signal storage live and synchronize destruction
/// for the complete callback, including any wait performed by the callback.
pub(crate) unsafe fn with_signal<R>(
    signal: HsaSignal,
    operation: impl for<'a> FnOnce(&'a AmdSignal) -> R,
) -> Option<R> {
    // SAFETY: The caller keeps the storage live throughout the operation.
    let pointer = unsafe { signal_ptr(signal) }?;
    // SAFETY: No reference escapes this call and the caller retains the owner.
    Some(operation(unsafe { pointer.as_ref() }))
}

/// Evaluates an HSA signal wait condition against the observed value.
fn condition_met(condition: u32, observed: SignalValue, compare: SignalValue) -> bool {
    match condition {
        SIGNAL_CONDITION_EQ => observed == compare,
        SIGNAL_CONDITION_NE => observed != compare,
        SIGNAL_CONDITION_LT => observed < compare,
        SIGNAL_CONDITION_GTE => observed >= compare,
        _ => false,
    }
}

/// Returns the public value address for an eligible signal.
pub(crate) fn signal_value_pointer(
    signal: &AmdSignal,
    interrupt: bool,
) -> Result<*mut SignalValue, Status> {
    if signal.kind.load(Ordering::Acquire) == AMD_SIGNAL_KIND_INVALID {
        return Err(INVALID_SIGNAL);
    }
    if signal.kind.load(Ordering::Acquire) != AMD_SIGNAL_KIND_USER || interrupt {
        return Err(INVALID_ARGUMENT);
    }
    Ok((&raw const signal.value).cast::<SignalValue>().cast_mut())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
