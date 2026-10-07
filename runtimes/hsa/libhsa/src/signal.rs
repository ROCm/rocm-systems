// SPDX-License-Identifier: MIT

//! HSA signals, IPC signal storage, waits, groups, and asynchronous handlers.
//!
//! The first 64 bytes of each native signal follow the public AMD signal layout
//! used by queues and device-visible storage. Runtime-owned slabs keep ordinary
//! signal addresses stable, while dedicated rocddi allocations back interrupt
//! and IPC-capable signals. Atomic operations preserve the ordering requested by
//! each ABI entry point rather than strengthening all accesses indiscriminately.
//!
//! One dispatcher serializes asynchronous callbacks. Registration and shutdown
//! communicate through a channel so callbacks and worker joins occur without
//! holding the process-global runtime lock.

use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering, fence};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::platform::event::{SignalEvent, SignalEventPage, create_signal_event};
use crate::platform::memory::{self as linux_interop, KfdIpcMemoryHandle};
use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};

use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{
    CallbackScope, InFlightToken, Runtime, boundary, initialized_mut, lock, map_error,
};

const SIGNAL_BYTES: usize = 64;
const SIGNALS_PER_SLAB: usize = 1024;
const MAX_POOLED_SIGNAL_EVENTS: usize = 256;
const BLOCKED_SPIN_BUDGET: Duration = Duration::from_micros(200);
const SIGNAL_EVENT_PAGE_BYTES: u64 = 4096 * 8;
const IPC_SIGNAL_ALLOCATION_BYTES: u64 = 4096;
const SHARED_SIGNAL_ID: u64 = 0x71fc_ca6a_3d5d_5276;
static PROCESS_SIGNAL_EVENT_PAGE: OnceLock<usize> = OnceLock::new();
static SYSTEM_FREQUENCY_HZ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_system_frequency(frequency: u64) {
    SYSTEM_FREQUENCY_HZ.store(frequency, Ordering::Release);
}

fn system_frequency() -> u64 {
    let cached = SYSTEM_FREQUENCY_HZ.load(Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    let retained = lock().ok().and_then(|guard| {
        guard.as_ref().and_then(|runtime| {
            let gpu = runtime.gpus.first()?;
            let token = runtime.inflight.enter()?;
            Some((gpu.device.clone(), token))
        })
    });
    let Some((device, _call)) = retained else {
        return 0;
    };
    let frequency = device
        .gpu()
        .and_then(|gpu| gpu.clock_counters())
        .ok()
        .map_or(0, |counters| counters.system_frequency);
    if frequency == 0 {
        return 0;
    }
    // Shutdown can clear the cache while the ioctl runs. Publish only for
    // the same active native VM, while holding the registry lock.
    let Ok(guard) = lock() else {
        return 0;
    };
    if guard
        .as_ref()
        .and_then(|runtime| runtime.gpus.first())
        .is_some_and(|gpu| gpu.device.shares_address_domain(&device))
    {
        SYSTEM_FREQUENCY_HZ.store(frequency, Ordering::Release);
        frequency
    } else {
        0
    }
}

fn timeout_elapsed(elapsed: Duration, timeout_hint: u64, frequency: u64) -> bool {
    timeout_hint != u64::MAX
        && (frequency == 0
            || elapsed.as_nanos().saturating_mul(u128::from(frequency))
                >= u128::from(timeout_hint) * 1_000_000_000)
}

#[repr(C, align(64))]
/// Public AMD signal record shared with host code and, when applicable, a GPU.
///
/// Field offsets and the 64-byte extent are ABI requirements. The signal kind
/// determines whether updates are ordinary atomics, queue doorbells, or
/// interrupt-capable event notifications.
pub(crate) struct AmdSignal {
    // Destruction invalidates this field while lock-free readers inspect it.
    // Keep its ABI offset and width while avoiding a host data race.
    pub(crate) kind: AtomicI64,
    pub(crate) value: AtomicI64,
    event_mailbox_ptr: u64,
    event_id: u32,
    reserved1: u32,
    pub(crate) start_ts: AtomicU64,
    pub(crate) end_ts: AtomicU64,
    pub(crate) queue_ptr: usize,
    reserved3: [u32; 2],
}

impl AmdSignal {
    pub(crate) fn user(value: SignalValue) -> Self {
        Self::user_with_event(value, 0, 0)
    }

    pub(crate) fn interrupt(value: SignalValue, event_mailbox_ptr: usize, event_id: u32) -> Self {
        Self::user_with_event(value, event_mailbox_ptr as u64, event_id)
    }

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
}

const _: () = {
    assert!(size_of::<AmdSignal>() == SIGNAL_BYTES);
    assert!(std::mem::offset_of!(AmdSignal, kind) == 0);
    assert!(std::mem::offset_of!(AmdSignal, value) == 8);
};

#[repr(C, align(64))]
/// Cross-process signal record stored in shareable system memory.
struct SharedSignal {
    signal: AmdSignal,
    sdma_start_ts: u64,
    core_signal: u64,
    id: u64,
    reserved: [u8; 8],
    sdma_end_ts: u64,
    reserved2: [u8; 24],
}

impl SharedSignal {
    fn ipc(value: SignalValue) -> Self {
        Self {
            signal: AmdSignal::user(value),
            sdma_start_ts: 0,
            core_signal: 0,
            id: SHARED_SIGNAL_ID,
            reserved: [0; 8],
            sdma_end_ts: 0,
            reserved2: [0; 24],
        }
    }

    fn is_ipc(&self) -> bool {
        self.core_signal == 0 && self.id == SHARED_SIGNAL_ID
    }
}

const _: () = assert!(size_of::<SharedSignal>() == 128);

/// Original IPC signal allocation retained until its exporting handle dies.
pub(crate) struct OwnedIpcSignal {
    allocation: Allocation,
}

/// Imported IPC signal mapping and its local public-reference count.
pub(crate) struct ImportedIpcSignal {
    allocation: Allocation,
    ipc_handle: [u32; 8],
    references: u32,
}

/// Work accepted by the single asynchronous callback dispatcher.
#[derive(Clone, Copy)]
enum AsyncRequest {
    Function {
        callback: unsafe extern "C" fn(*mut c_void),
        arg: CallbackArg,
    },
    Signal {
        signal: HsaSignal,
        condition: u32,
        compare_value: SignalValue,
        handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
        arg: CallbackArg,
    },
}

/// Persistent signal condition re-evaluated by the dispatcher.
struct AsyncSignalHandler {
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
    arg: CallbackArg,
}

/// Clock domain of a completed asynchronous copy's raw timestamps.
#[derive(Clone, Copy)]
pub(crate) enum AsyncCopyClock {
    System,
    Gpu(usize),
}

/// Timestamps and clock provenance retained outside the public signal ABI.
#[derive(Clone, Copy)]
pub(crate) struct AsyncCopyProfile {
    pub(crate) clock: AsyncCopyClock,
    pub(crate) start: u64,
    pub(crate) end: u64,
}

/// Dispatcher and copy worker references retain signal storage after the
/// public handle dies. A completed copy keeps its timing until this signal is
/// reused or destroyed, even after the worker releases its reference.
pub(crate) struct AsyncSignalRecord {
    references: u32,
    retired: Arc<AtomicBool>,
    pub(crate) copy_profile: Option<AsyncCopyProfile>,
}

impl AsyncSignalRecord {
    fn revive(&mut self) {
        if self.retired.load(Ordering::Acquire) {
            // Existing waiters must keep observing the prior retirement even
            // when a new IPC attachment publishes the same numeric handle.
            self.retired = Arc::new(AtomicBool::new(false));
            self.copy_profile = None;
        }
    }
}

/// Sending half of the process-wide serialized callback worker.
pub(crate) struct AsyncDispatcher {
    sender: Sender<AsyncRequest>,
}

impl AsyncDispatcher {
    fn send(&self, request: AsyncRequest) -> Status {
        self.sender.send(request).map_or(ERROR, |()| SUCCESS)
    }
}

/// Stable device-visible storage from which ordinary signals are carved.
pub(crate) struct SignalSlab {
    _allocation: Allocation,
    host: usize,
    next: usize,
}

impl SignalSlab {
    fn create(runtime: &Runtime) -> Result<Self, Status> {
        let gpu = runtime.gpus.first().ok_or(OUT_OF_RESOURCES)?;
        let allocation = gpu
            .device
            .allocate(
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

    fn has_space(&self) -> bool {
        self.next < SIGNALS_PER_SLAB
    }

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

fn retry_unconfirmed_event_page<T, E>(
    first: Result<T, E>,
    page_offered: bool,
    confirmed: bool,
    retry_without_page: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    match first {
        Ok(event) => Ok(event),
        Err(_) if page_offered && !confirmed => retry_without_page(),
        Err(error) => Err(error),
    }
}

impl Runtime {
    fn enqueue_async(&mut self, request: AsyncRequest) -> Status {
        if self.async_dispatcher.is_none() {
            let (sender, receiver) = std::sync::mpsc::channel();
            let stop = self.stop_workers.clone();
            let worker = match thread::Builder::new()
                .name("rocddi-async-events".into())
                .spawn(move || run_async_dispatcher(&receiver, &stop))
            {
                Ok(worker) => worker,
                Err(_) => return OUT_OF_RESOURCES,
            };
            self.async_dispatcher = Some(AsyncDispatcher { sender });
            self.workers.push(worker);
        }
        self.async_dispatcher
            .as_ref()
            .map_or(ERROR, |dispatcher| dispatcher.send(request))
    }

    fn ensure_signal_slab(&mut self) -> Result<(), Status> {
        if self
            .signal_slabs
            .last()
            .is_none_or(|slab| !slab.has_space())
        {
            self.signal_slabs.push(SignalSlab::create(self)?);
        }
        Ok(())
    }

    fn allocate_signal(&mut self, signal: AmdSignal) -> Result<HsaSignal, Status> {
        if let Some(address) = self.recycled_signals.pop() {
            // SAFETY: The slot was invalidated after its final asynchronous
            // registration retired. Its slab remains mapped until shutdown.
            unsafe { (address as *mut AmdSignal).write(signal) };
            return Ok(HsaSignal {
                handle: address as u64,
            });
        }
        self.ensure_signal_slab()?;
        let slab = self.signal_slabs.last_mut().ok_or(OUT_OF_RESOURCES)?;
        // SAFETY: The selected slab has one unused aligned slot.
        Ok(unsafe { slab.allocate(signal) })
    }

    fn ensure_signal_event_page(&mut self) -> Result<(usize, bool), Status> {
        if let Some(host) = PROCESS_SIGNAL_EVENT_PAGE.get().copied() {
            return Ok((host, false));
        }
        if self.signal_event_page.is_none() {
            let allocation = {
                let (gpu, peers) = self.gpus.split_first().ok_or(OUT_OF_RESOURCES)?;
                let peers = peers.iter().map(|peer| &peer.device).collect::<Vec<_>>();
                gpu.device
                    .allocate_with_peers(
                        &peers,
                        MemoryKind::System,
                        SIGNAL_EVENT_PAGE_BYTES.max(self.host_page_size as u64),
                        self.host_page_size as u64,
                        DeviceAccess::READ | DeviceAccess::WRITE,
                    )
                    .map_err(map_error)?
            };
            let info = allocation.info();
            let host = info.host_address.ok_or(OUT_OF_RESOURCES)?;
            if info.device_address != host as u64 || info.size < SIGNAL_EVENT_PAGE_BYTES {
                return Err(OUT_OF_RESOURCES);
            }
            self.signal_event_page = Some(SignalEventPage::new(allocation));
        }
        let host = self
            .signal_event_page
            .as_ref()
            .and_then(SignalEventPage::host_address)
            .ok_or(OUT_OF_RESOURCES)?;
        Ok((host, true))
    }

    pub(crate) fn create_signal_event(
        &mut self,
    ) -> Option<(crate::platform::event::SignalEvent, usize, u32)> {
        let (page_host, install_page) = self.ensure_signal_event_page().ok()?;
        let gpu = self.gpus.first()?;
        let device = gpu.device.gpu().ok()?;
        let first = if install_page {
            create_signal_event(device, Some(self.signal_event_page.as_mut()?))
        } else {
            create_signal_event(device, None)
        };
        let page_offered = self
            .signal_event_page
            .as_ref()
            .is_some_and(SignalEventPage::was_offered);
        // An attempted CREATE_EVENT can install the page before reporting an
        // error. Retry without its handle only after the ioctl was dispatched.
        let created = retry_unconfirmed_event_page(
            first,
            install_page && page_offered,
            self.signal_event_page_confirmed,
            || create_signal_event(device, None),
        );
        if page_offered
            && PROCESS_SIGNAL_EVENT_PAGE
                .set(page_host)
                .is_err_and(|published| published != page_host)
        {
            if let Ok(mut event) = created {
                let _ = event.destroy();
            }
            return None;
        }
        let event = created.ok()?;
        self.signal_event_page_confirmed = true;
        let info = event.info();
        let offset = usize::try_from(info.event_page_slot_index)
            .ok()?
            .checked_mul(8)?;
        let mailbox = page_host.checked_add(offset)?;
        Some((event, mailbox, info.kfd_event_id))
    }

    fn acquire_signal_event(&mut self) -> Option<(SignalEvent, usize, u32)> {
        // Reuse the KFD event after its previous signal has been destroyed.
        // The stable process event page supplies the same mailbox address.
        if let Some(event) = self.signal_event_pool.pop() {
            let page = *PROCESS_SIGNAL_EVENT_PAGE.get()?;
            let info = event.info();
            let offset = usize::try_from(info.event_page_slot_index)
                .ok()?
                .checked_mul(8)?;
            let mailbox = page.checked_add(offset)?;
            return Some((event, mailbox, info.kfd_event_id));
        }
        self.create_signal_event()
    }

    pub(crate) fn create_signal(
        &mut self,
        value: SignalValue,
        interrupt: bool,
    ) -> Result<HsaSignal, Status> {
        if !interrupt {
            return self.allocate_signal(AmdSignal::user(value));
        }
        self.interrupt_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        let event = self.acquire_signal_event();
        let storage = event.as_ref().map_or_else(
            || AmdSignal::interrupt(value, 0, 0),
            |(_, mailbox, event_id)| AmdSignal::interrupt(value, *mailbox, *event_id),
        );
        let signal = self.allocate_signal(storage)?;
        let owner = event.map(|(event, _, _)| event);
        match self.interrupt_signals.entry(signal.handle as usize) {
            Entry::Vacant(entry) => {
                entry.insert(owner);
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(signal)
    }

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
            || self
                .imported_ipc_signals
                .get(&address)
                .is_some_and(|imported| imported.references != 0)
            || self.signal_slabs.iter().any(|slab| slab.owns_live(address))
    }

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

    fn create_ipc_signal(&mut self, value: SignalValue) -> Result<HsaSignal, Status> {
        let allocation = {
            let (gpu, peers) = self.gpus.split_first().ok_or(OUT_OF_RESOURCES)?;
            let peers = peers.iter().map(|peer| &peer.device).collect::<Vec<_>>();
            gpu.device
                .allocate_with_peers(
                    &peers,
                    MemoryKind::System,
                    IPC_SIGNAL_ALLOCATION_BYTES.max(self.host_page_size as u64),
                    self.host_page_size as u64,
                    DeviceAccess::READ | DeviceAccess::WRITE,
                )
                .map_err(map_error)?
        };
        let info = allocation.info();
        let host = info.host_address.ok_or(OUT_OF_RESOURCES)?;
        if info.device_address != host as u64 {
            return Err(OUT_OF_RESOURCES);
        }
        self.owned_ipc_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        // SAFETY: The dedicated allocation is writable, page-sized, and retained
        // by owned_ipc_signals for the complete public signal lifetime.
        unsafe {
            std::ptr::write_bytes(host as *mut u8, 0, IPC_SIGNAL_ALLOCATION_BYTES as usize);
            (host as *mut SharedSignal).write(SharedSignal::ipc(value));
        }
        match self.owned_ipc_signals.entry(host) {
            Entry::Vacant(entry) => {
                entry.insert(OwnedIpcSignal { allocation });
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(HsaSignal {
            handle: host as u64,
        })
    }

    fn export_ipc_signal(&self, signal: HsaSignal) -> Result<[u32; 8], Status> {
        let address = signal.handle as usize;
        if !self.owns_signal(signal) {
            return Err(INVALID_ARGUMENT);
        }
        let allocation = self
            .owned_ipc_signals
            .get(&address)
            .map(|signal| &signal.allocation)
            .or_else(|| {
                self.imported_ipc_signals
                    .get(&address)
                    .map(|signal| &signal.allocation)
            })
            .ok_or(INVALID_ARGUMENT)?;
        linux_interop::export_kfd_ipc_memory(allocation)
            .map(KfdIpcMemoryHandle::words)
            .map_err(map_error)
    }

    unsafe fn attach_ipc_signal(&mut self, words: [u32; 8]) -> Result<HsaSignal, Status> {
        if let Some((address, imported)) = self
            .imported_ipc_signals
            .iter_mut()
            .find(|(_, signal)| signal.ipc_handle == words)
        {
            imported.references = imported.references.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
            let address = *address;
            if let Some(record) = self.async_signal_refs.get_mut(&address) {
                if record.references == 0 && record.retired.load(Ordering::Acquire) {
                    self.async_signal_refs.remove(&address);
                } else {
                    record.revive();
                }
            }
            return Ok(HsaSignal {
                handle: address as u64,
            });
        }

        let allocation = {
            let devices = self.gpus.iter().map(|gpu| &gpu.device).collect::<Vec<_>>();
            linux_interop::import_kfd_ipc_memory(
                &self.session,
                &devices,
                &devices,
                KfdIpcMemoryHandle::from_words(words),
                IPC_SIGNAL_ALLOCATION_BYTES,
            )
            .map_err(|error| match error.kind() {
                rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                    INVALID_ARGUMENT
                }
                rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
                _ => ERROR,
            })?
        };
        let info = allocation.info();
        let host = info.host_address.ok_or(INVALID_ARGUMENT)?;
        if info.device_address != host as u64 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: A successful IPC import mapped at least one page at host.
        let shared = unsafe { &*(host as *const SharedSignal) };
        if !shared.is_ipc() {
            return Err(INVALID_ARGUMENT);
        }
        self.imported_ipc_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        match self.imported_ipc_signals.entry(host) {
            Entry::Vacant(entry) => {
                entry.insert(ImportedIpcSignal {
                    allocation,
                    ipc_handle: words,
                    references: 1,
                });
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(HsaSignal {
            handle: host as u64,
        })
    }

    fn destroy_signal(&mut self, signal: HsaSignal) -> Status {
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
            // ROCr also retains each async registration. Keep the mapped IPC
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

    pub(crate) fn destroy_signal_events(&mut self) -> Status {
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
        let unresolved_page = self
            .signal_event_page
            .as_ref()
            .is_some_and(SignalEventPage::was_offered)
            && !self.signal_event_page_confirmed;
        if let Some(mut page) = self.signal_event_page.take() {
            if let Err(error) = page.retain_for_process() {
                if status == SUCCESS {
                    status = map_error(error);
                }
            }
        }
        // An unsuccessful CREATE_EVENT cannot tell us whether KFD installed
        // the page. Another runtime would have its address but not its handle.
        // Quarantine this process instead of reinitializing over that state.
        if unresolved_page && status == SUCCESS {
            status = INVALID_RUNTIME_STATE;
        }
        status
    }
}

unsafe fn valid_consumers(num_consumers: u32, consumers: *const HsaAgent) -> bool {
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

fn amd_signal_uses_consumer_list(attributes: u64) -> bool {
    attributes & (AMD_SIGNAL_AMD_GPU_ONLY | AMD_SIGNAL_IPC) == 0
}

unsafe fn signal_uses_interrupt(
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
    Ok(consumers.iter().any(|agent| agent.handle == CPU_AGENT))
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

fn condition_met(condition: u32, observed: SignalValue, compare: SignalValue) -> bool {
    match condition {
        SIGNAL_CONDITION_EQ => observed == compare,
        SIGNAL_CONDITION_NE => observed != compare,
        SIGNAL_CONDITION_LT => observed < compare,
        SIGNAL_CONDITION_GTE => observed >= compare,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_create(
    initial_value: SignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    signal: *mut HsaSignal,
) -> Status {
    boundary(|| {
        if signal.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The public ABI promises num_consumers readable handles when nonzero.
        let interrupt = match unsafe { signal_uses_interrupt(num_consumers, consumers, 0) } {
            Ok(interrupt) => interrupt,
            Err(status) => return status,
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let created = match runtime.create_signal(initial_value, interrupt) {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(created) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_create(
    initial_value: SignalValue,
    num_consumers: u32,
    consumers: *const HsaAgent,
    attributes: u64,
    signal: *mut HsaSignal,
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
        if signal.is_null() {
            return INVALID_ARGUMENT;
        }
        // Default/IPC signals do not use the consumer list. In particular, do
        // not inspect a pointer the caller is allowed to omit for these modes.
        // SAFETY: The public ABI promises num_consumers readable handles when
        // the selected signal kind consumes this list.
        let interrupt = match unsafe { signal_uses_interrupt(num_consumers, consumers, attributes) }
        {
            Ok(interrupt) => interrupt,
            Err(status) => return status,
        };
        let created = match if attributes & AMD_SIGNAL_IPC != 0 {
            runtime.create_ipc_signal(initial_value)
        } else {
            runtime.create_signal(initial_value, interrupt)
        } {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(created) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_destroy(signal: HsaSignal) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        runtime.destroy_signal(signal)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ipc_signal_create(
    signal: HsaSignal,
    handle: *mut HsaAmdIpcSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let words = match runtime.export_ipc_signal(signal) {
            Ok(words) => words,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { handle.write(HsaAmdIpcSignal { handle: words }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ipc_signal_attach(
    handle: *const HsaAmdIpcSignal,
    signal: *mut HsaSignal,
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
        if handle.is_null() || signal.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied a readable IPC handle.
        let words = unsafe { handle.read() }.handle;
        // SAFETY: attach_ipc_signal validates the imported mapping and shared header.
        let attached = match unsafe { runtime.attach_ipc_signal(words) } {
            Ok(signal) => signal,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { signal.write(attached) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_load_relaxed(signal: HsaSignal) -> SignalValue {
    // SAFETY: Callers must supply a live HSA signal handle.
    unsafe { with_signal(signal, |signal| signal.value.load(Ordering::Relaxed)) }.unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_load_scacquire(signal: HsaSignal) -> SignalValue {
    // SAFETY: Callers must supply a live HSA signal handle.
    unsafe { with_signal(signal, |signal| signal.value.load(Ordering::Acquire)) }.unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_load_acquire(signal: HsaSignal) -> SignalValue {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_signal_load_scacquire(signal) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_store_relaxed(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Callers must supply a live HSA signal handle.
    let _ = unsafe {
        with_signal(signal, |signal| {
            if signal.kind.load(Ordering::Acquire) == AMD_SIGNAL_KIND_DOORBELL {
                let address = signal.value.load(Ordering::Relaxed) as usize;
                // SAFETY: A queue doorbell signal contains its live MMIO mapping.
                rocddi::gpu::queue::ring_doorbell(address, value as u64);
            } else {
                signal.value.store(value, Ordering::Relaxed);
            }
        })
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_silent_store_relaxed(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Same operation is sufficient for polling-backed signals.
    unsafe { hsa_signal_store_relaxed(signal, value) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_silent_store_screlease(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Same operation is sufficient for polling-backed signals.
    unsafe { hsa_signal_store_screlease(signal, value) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_store_screlease(signal: HsaSignal, value: SignalValue) {
    // SAFETY: Callers must supply a live HSA signal handle.
    let _ = unsafe {
        with_signal(signal, |signal| {
            if signal.kind.load(Ordering::Acquire) == AMD_SIGNAL_KIND_DOORBELL {
                fence(Ordering::Release);
                let address = signal.value.load(Ordering::Relaxed) as usize;
                // SAFETY: A queue doorbell signal contains its live MMIO mapping.
                rocddi::gpu::queue::ring_doorbell(address, value as u64);
            } else {
                signal.value.store(value, Ordering::Release);
            }
        })
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_store_release(signal: HsaSignal, value: SignalValue) {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent release spelling.
    unsafe { hsa_signal_store_screlease(signal, value) };
}

unsafe fn signal_exchange(signal: HsaSignal, value: SignalValue, order: Ordering) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe { with_signal(signal, |signal| signal.value.swap(value, order)) }.unwrap_or(0)
}

unsafe fn signal_compare_exchange(
    signal: HsaSignal,
    expected: SignalValue,
    value: SignalValue,
    success: Ordering,
    failure: Ordering,
) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe {
        with_signal(signal, |signal| {
            signal
                .value
                .compare_exchange(expected, value, success, failure)
                .unwrap_or_else(|observed| observed)
        })
    }
    .unwrap_or(0)
}

unsafe fn signal_fetch_update(
    signal: HsaSignal,
    value: SignalValue,
    order: Ordering,
    operation: fn(&AtomicI64, SignalValue, Ordering),
) {
    // SAFETY: Callers must supply a live HSA user signal.
    let _ = unsafe { with_signal(signal, |signal| operation(&signal.value, value, order)) };
}

fn signal_add(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_add(operand, order);
}

fn signal_subtract(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_sub(operand, order);
}

fn signal_and(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_and(operand, order);
}

fn signal_or(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_or(operand, order);
}

fn signal_xor(value: &AtomicI64, operand: SignalValue, order: Ordering) {
    value.fetch_xor(operand, order);
}

macro_rules! signal_exchange_entry {
    ($name:ident, $order:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(signal: HsaSignal, value: SignalValue) -> SignalValue {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_exchange(signal, value, $order) }
        }
    };
}

macro_rules! signal_compare_exchange_entry {
    ($name:ident, $success:expr, $failure:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            signal: HsaSignal,
            expected: SignalValue,
            value: SignalValue,
        ) -> SignalValue {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_compare_exchange(signal, expected, value, $success, $failure) }
        }
    };
}

macro_rules! signal_fetch_entry {
    ($name:ident, $order:expr, $operation:path) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(signal: HsaSignal, value: SignalValue) {
            // SAFETY: The public entry point preserves the HSA signal contract.
            unsafe { signal_fetch_update(signal, value, $order, $operation) };
        }
    };
}

signal_exchange_entry!(hsa_signal_exchange_scacq_screl, Ordering::AcqRel);
signal_exchange_entry!(hsa_signal_exchange_acq_rel, Ordering::AcqRel);
signal_exchange_entry!(hsa_signal_exchange_scacquire, Ordering::Acquire);
signal_exchange_entry!(hsa_signal_exchange_acquire, Ordering::Acquire);
signal_exchange_entry!(hsa_signal_exchange_relaxed, Ordering::Relaxed);
signal_exchange_entry!(hsa_signal_exchange_screlease, Ordering::Release);
signal_exchange_entry!(hsa_signal_exchange_release, Ordering::Release);

signal_compare_exchange_entry!(
    hsa_signal_cas_scacq_screl,
    Ordering::AcqRel,
    Ordering::Acquire
);
signal_compare_exchange_entry!(hsa_signal_cas_acq_rel, Ordering::AcqRel, Ordering::Acquire);
signal_compare_exchange_entry!(
    hsa_signal_cas_scacquire,
    Ordering::Acquire,
    Ordering::Acquire
);
signal_compare_exchange_entry!(hsa_signal_cas_acquire, Ordering::Acquire, Ordering::Acquire);
signal_compare_exchange_entry!(hsa_signal_cas_relaxed, Ordering::Relaxed, Ordering::Relaxed);
signal_compare_exchange_entry!(
    hsa_signal_cas_screlease,
    Ordering::Release,
    Ordering::Relaxed
);
signal_compare_exchange_entry!(hsa_signal_cas_release, Ordering::Release, Ordering::Relaxed);

signal_fetch_entry!(hsa_signal_add_scacq_screl, Ordering::AcqRel, signal_add);
signal_fetch_entry!(hsa_signal_add_acq_rel, Ordering::AcqRel, signal_add);
signal_fetch_entry!(hsa_signal_add_scacquire, Ordering::Acquire, signal_add);
signal_fetch_entry!(hsa_signal_add_acquire, Ordering::Acquire, signal_add);
signal_fetch_entry!(hsa_signal_add_relaxed, Ordering::Relaxed, signal_add);
signal_fetch_entry!(hsa_signal_add_screlease, Ordering::Release, signal_add);
signal_fetch_entry!(hsa_signal_add_release, Ordering::Release, signal_add);

signal_fetch_entry!(
    hsa_signal_subtract_scacq_screl,
    Ordering::AcqRel,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_acq_rel,
    Ordering::AcqRel,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_scacquire,
    Ordering::Acquire,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_acquire,
    Ordering::Acquire,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_relaxed,
    Ordering::Relaxed,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_screlease,
    Ordering::Release,
    signal_subtract
);
signal_fetch_entry!(
    hsa_signal_subtract_release,
    Ordering::Release,
    signal_subtract
);

signal_fetch_entry!(hsa_signal_and_scacq_screl, Ordering::AcqRel, signal_and);
signal_fetch_entry!(hsa_signal_and_acq_rel, Ordering::AcqRel, signal_and);
signal_fetch_entry!(hsa_signal_and_scacquire, Ordering::Acquire, signal_and);
signal_fetch_entry!(hsa_signal_and_acquire, Ordering::Acquire, signal_and);
signal_fetch_entry!(hsa_signal_and_relaxed, Ordering::Relaxed, signal_and);
signal_fetch_entry!(hsa_signal_and_screlease, Ordering::Release, signal_and);
signal_fetch_entry!(hsa_signal_and_release, Ordering::Release, signal_and);

signal_fetch_entry!(hsa_signal_or_scacq_screl, Ordering::AcqRel, signal_or);
signal_fetch_entry!(hsa_signal_or_acq_rel, Ordering::AcqRel, signal_or);
signal_fetch_entry!(hsa_signal_or_scacquire, Ordering::Acquire, signal_or);
signal_fetch_entry!(hsa_signal_or_acquire, Ordering::Acquire, signal_or);
signal_fetch_entry!(hsa_signal_or_relaxed, Ordering::Relaxed, signal_or);
signal_fetch_entry!(hsa_signal_or_screlease, Ordering::Release, signal_or);
signal_fetch_entry!(hsa_signal_or_release, Ordering::Release, signal_or);

signal_fetch_entry!(hsa_signal_xor_scacq_screl, Ordering::AcqRel, signal_xor);
signal_fetch_entry!(hsa_signal_xor_acq_rel, Ordering::AcqRel, signal_xor);
signal_fetch_entry!(hsa_signal_xor_scacquire, Ordering::Acquire, signal_xor);
signal_fetch_entry!(hsa_signal_xor_acquire, Ordering::Acquire, signal_xor);
signal_fetch_entry!(hsa_signal_xor_relaxed, Ordering::Relaxed, signal_xor);
signal_fetch_entry!(hsa_signal_xor_screlease, Ordering::Release, signal_xor);
signal_fetch_entry!(hsa_signal_xor_release, Ordering::Release, signal_xor);

unsafe fn signal_wait(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    order: Ordering,
) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe {
        with_signal(signal, |signal| {
            // The common satisfied path needs only one atomic load. Read the clock
            // and system frequency only once the caller actually has to wait.
            let observed = signal.value.load(order);
            if condition_met(condition, observed, compare_value) {
                return observed;
            }
            let start = Instant::now();
            let frequency = system_frequency();
            loop {
                let observed = signal.value.load(order);
                if condition_met(condition, observed, compare_value) {
                    return observed;
                }
                if timeout_hint != u64::MAX
                    && timeout_elapsed(start.elapsed(), timeout_hint, frequency)
                {
                    return observed;
                }
                wait_pause(wait_state_hint, &start);
            }
        })
    }
    .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_wait_scacquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Acquire,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_wait_relaxed(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Relaxed,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_wait_acquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe {
        hsa_signal_wait_scacquire(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_group_create(
    signal_count: u32,
    signals: *const HsaSignal,
    consumer_count: u32,
    consumers: *const HsaAgent,
    group: *mut HsaSignalGroup,
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
        if signal_count == 0
            || signals.is_null()
            || group.is_null()
            || (consumer_count != 0 && consumers.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplies signal_count readable handles.
        let signals = unsafe { std::slice::from_raw_parts(signals, signal_count as usize) };
        if signals.iter().any(|signal| !runtime.owns_signal(*signal)) {
            return INVALID_SIGNAL;
        }
        if consumer_count != 0 {
            // SAFETY: The caller supplies consumer_count readable handles.
            let consumers =
                unsafe { std::slice::from_raw_parts(consumers, consumer_count as usize) };
            if consumers.iter().any(|agent| !runtime.is_agent(*agent)) {
                return INVALID_AGENT;
            }
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        let mut stored = Vec::new();
        if stored.try_reserve_exact(signals.len()).is_err() {
            return OUT_OF_RESOURCES;
        }
        stored.extend_from_slice(signals);
        runtime.signal_groups.insert(handle, stored);
        // SAFETY: The caller supplied writable output storage.
        unsafe { group.write(HsaSignalGroup { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_signal_group_destroy(group: HsaSignalGroup) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.signal_groups.remove(&group.handle).is_some() {
            SUCCESS
        } else {
            INVALID_SIGNAL_GROUP
        }
    })
}

unsafe fn signal_group_wait_any(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    let signals = {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if conditions.is_null() || compare_values.is_null() || signal.is_null() || value.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(signals) = runtime.signal_groups.get(&group.handle) else {
            return INVALID_SIGNAL_GROUP;
        };
        signals.clone()
    };
    // SAFETY: The group retains signals.len() handles and the caller supplies
    // matching condition/value arrays and writable outputs.
    let index = unsafe {
        hsa_amd_signal_wait_any(
            signals.len() as u32,
            signals.as_ptr(),
            conditions,
            compare_values,
            u64::MAX,
            wait_state_hint,
            value,
        )
    };
    let Some(satisfied) = signals.get(index as usize) else {
        return INVALID_ARGUMENT;
    };
    // SAFETY: The caller supplied writable output storage.
    unsafe { signal.write(*satisfied) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_group_wait_any_relaxed(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_group_wait_any_scacquire(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    let status = unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    };
    if status == SUCCESS {
        fence(Ordering::Acquire);
    }
    status
}

struct WaitInputs {
    signals: Vec<HsaSignal>,
    conditions: Vec<u32>,
    values: Vec<SignalValue>,
}

unsafe fn copy_wait_input<T: Copy>(pointer: *const T, count: usize) -> Option<Vec<T>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if pointer.is_null()
        || count
            .checked_mul(size_of::<T>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return None;
    }
    let mut copied = Vec::new();
    copied.try_reserve_exact(count).ok()?;
    // SAFETY: The caller supplies count aligned, readable elements. The
    // borrow ends before any result pointer can be written by the wait.
    copied.extend_from_slice(unsafe { std::slice::from_raw_parts(pointer, count) });
    Some(copied)
}

unsafe fn signal_wait_inputs(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
) -> Option<WaitInputs> {
    let count = signal_count as usize;
    // SAFETY: The public ABI supplies readable arrays for the duration of each
    // copy. Own them before a satisfying-value output can alias any input.
    Some(WaitInputs {
        signals: unsafe { copy_wait_input(signals, count) }?,
        conditions: unsafe { copy_wait_input(conditions, count) }?,
        values: unsafe { copy_wait_input(values, count) }?,
    })
}

fn wait_pause(wait_state_hint: u32, start: &Instant) {
    // A brief active phase avoids scheduler latency for short GPU completions
    // even when the caller allows a blocked wait. Longer blocked waits yield.
    if wait_state_hint == WAIT_STATE_ACTIVE
        || (wait_state_hint == WAIT_STATE_BLOCKED && start.elapsed() < BLOCKED_SPIN_BUDGET)
    {
        std::hint::spin_loop();
    } else {
        thread::yield_now();
    }
}

/// One validated signal; `retained` keeps its slot stable even after destroy.
struct WaitEntry {
    handle: HsaSignal,
    signal: Option<NonNull<AmdSignal>>,
    retired: Option<Arc<AtomicBool>>,
    retained: bool,
}

struct WaitRegistration {
    // Release retained slots before the token permits shutdown to unmap them.
    entries: Vec<WaitEntry>,
    _token: InFlightToken,
    stop: Arc<AtomicBool>,
}

impl Drop for WaitRegistration {
    fn drop(&mut self) {
        if let Ok(mut guard) = lock() {
            if let Some(runtime) = guard.as_mut() {
                release_wait_entries(runtime, &mut self.entries);
            }
        }
    }
}

fn release_wait_entries(runtime: &mut Runtime, entries: &mut [WaitEntry]) {
    for entry in entries {
        if entry.retained {
            entry.signal = None;
            entry.retired = None;
            runtime.release_async_signal(entry.handle);
            entry.retained = false;
        }
    }
}

/// # Safety
/// The caller must hold the runtime registry lock until every accepted signal
/// is retained, and must keep the returned registration alive through the wait.
unsafe fn prepare_multi_wait(
    runtime: &mut Runtime,
    handles: &[HsaSignal],
) -> Option<WaitRegistration> {
    let token = runtime.inflight.enter()?;
    // SAFETY: The runtime lock protects every accepted handle while its
    // storage reference is acquired, and the references are retained below.
    let mut entries =
        unsafe { collect_wait_signals(handles, |signal| runtime.owns_signal(signal)) };
    for index in 0..entries.len() {
        if entries[index].signal.is_none() {
            continue;
        }
        let handle = entries[index].handle;
        if runtime.retain_async_signal(handle).is_err() {
            release_wait_entries(runtime, &mut entries);
            return None;
        }
        entries[index].retained = true;
        let Some(record) = runtime.async_signal_refs.get(&(handle.handle as usize)) else {
            release_wait_entries(runtime, &mut entries);
            return None;
        };
        entries[index].retired = Some(record.retired.clone());
    }
    Some(WaitRegistration {
        entries,
        _token: token,
        stop: runtime.stop_workers.clone(),
    })
}

/// Selects live signals before a wait without touching invalid handles.
///
/// # Safety
/// Every handle accepted by `owns` must remain backed while the returned
/// entries are used. The caller must synchronize their eventual release.
unsafe fn collect_wait_signals(
    signals: &[HsaSignal],
    mut owns: impl FnMut(HsaSignal) -> bool,
) -> Vec<WaitEntry> {
    signals
        .iter()
        .map(|signal| {
            let reference = if owns(*signal) {
                // SAFETY: The caller guarantees the accepted handle's backing.
                unsafe { signal_ptr(*signal) }
            } else {
                None
            };
            WaitEntry {
                handle: *signal,
                signal: reference,
                retired: None,
                retained: false,
            }
        })
        .collect()
}

unsafe fn signal_wait_any_open(
    signals: &[WaitEntry],
    conditions: &[u32],
    values: &[SignalValue],
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
    stop: &AtomicBool,
) -> u32 {
    if signals.iter().all(|entry| entry.signal.is_none()) {
        return u32::MAX;
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        if stop.load(Ordering::Acquire) {
            return u32::MAX;
        }
        for (index, entry) in signals.iter().enumerate() {
            let Some(signal) = entry.signal else {
                continue;
            };
            if entry
                .retired
                .as_ref()
                .is_some_and(|retired| retired.load(Ordering::Acquire))
            {
                return u32::MAX;
            }
            // SAFETY: The registration retains this slot until the wait exits.
            let observed = unsafe { signal.as_ref().value.load(Ordering::Relaxed) };
            if condition_met(conditions[index], observed, values[index]) {
                if !satisfying_value.is_null() {
                    // SAFETY: The caller supplied optional writable output storage.
                    unsafe { satisfying_value.write(observed) };
                }
                return u32::try_from(index).unwrap_or(u32::MAX);
            }
        }
        if timeout_hint != u64::MAX && timeout_elapsed(start.elapsed(), timeout_hint, frequency) {
            return u32::MAX;
        }
        wait_pause(wait_state_hint, &start);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_wait_any(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_any_open(
            &registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_value,
            &registration.stop,
        )
    }
}

unsafe fn signal_wait_all_open(
    pending: &mut [WaitEntry],
    conditions: &[u32],
    values: &[SignalValue],
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
    stop: &AtomicBool,
) -> u32 {
    if !satisfying_values.is_null() {
        // SAFETY: The caller supplied signal_count writable output entries.
        // ROCr publishes zero for invalid and not-yet-satisfied signals even
        // when the wait ultimately times out.
        unsafe { std::ptr::write_bytes(satisfying_values, 0, pending.len()) };
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        if stop.load(Ordering::Acquire) {
            return u32::MAX;
        }
        let mut remaining = false;
        for (index, entry) in pending.iter_mut().enumerate() {
            let Some(signal_ref) = entry.signal else {
                continue;
            };
            if entry
                .retired
                .as_ref()
                .is_some_and(|retired| retired.load(Ordering::Acquire))
            {
                return u32::MAX;
            }
            // SAFETY: The registration retains this slot until the wait exits.
            let observed = unsafe { signal_ref.as_ref().value.load(Ordering::Relaxed) };
            if condition_met(conditions[index], observed, values[index]) {
                if !satisfying_values.is_null() {
                    // SAFETY: The caller supplied signal_count writable entries.
                    unsafe { satisfying_values.add(index).write(observed) };
                }
                entry.signal = None;
            } else {
                remaining = true;
            }
        }
        if !remaining {
            return 0;
        }
        if timeout_hint != u64::MAX && timeout_elapsed(start.elapsed(), timeout_hint, frequency) {
            return u32::MAX;
        }
        wait_pause(wait_state_hint, &start);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_wait_all(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let mut registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_all_open(
            &mut registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_values,
            &registration.stop,
        )
    }
}

fn signal_value_pointer(signal: &AmdSignal, interrupt: bool) -> Result<*mut SignalValue, Status> {
    if signal.kind.load(Ordering::Acquire) == AMD_SIGNAL_KIND_INVALID {
        return Err(INVALID_SIGNAL);
    }
    if signal.kind.load(Ordering::Acquire) != AMD_SIGNAL_KIND_USER || interrupt {
        return Err(INVALID_ARGUMENT);
    }
    Ok((&raw const signal.value).cast::<SignalValue>().cast_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_value_pointer(
    signal: HsaSignal,
    value: *mut *mut SignalValue,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        let interrupt = runtime
            .interrupt_signals
            .contains_key(&(signal.handle as usize));
        // SAFETY: owns_signal validated the handle while the runtime lock
        // prevents destruction of its backing storage.
        let Some(pointer) =
            (unsafe { with_signal(signal, |signal| signal_value_pointer(signal, interrupt)) })
        else {
            return INVALID_SIGNAL;
        };
        let pointer = match pointer {
            Ok(pointer) => pointer,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { value.write(pointer) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_get_event_id(
    signal: HsaSignal,
    event_id: *mut u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if event_id.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        // SAFETY: owns_signal validated this live slab slot and the caller
        // supplied writable output storage.
        unsafe { event_id.write((*((signal.handle as usize) as *const AmdSignal)).event_id) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_async_function(
    callback: AsyncFunction,
    arg: *mut c_void,
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
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        runtime.enqueue_async(AsyncRequest::Function {
            callback,
            arg: unsafe { CallbackArg::new(arg) },
        })
    })
}

fn accept_async_request(request: AsyncRequest, handlers: &mut Vec<AsyncSignalHandler>) {
    match request {
        AsyncRequest::Function { callback, arg } => {
            let _scope = CallbackScope::enter();
            // SAFETY: The public contract requires callback and arg to remain
            // valid until the callback executes.
            unsafe { callback(arg.as_ptr()) };
        }
        AsyncRequest::Signal {
            signal,
            condition,
            compare_value,
            handler,
            arg,
        } => handlers.push(AsyncSignalHandler {
            signal,
            condition,
            compare_value,
            handler,
            arg,
        }),
    }
}

fn drain_async_requests(
    receiver: &Receiver<AsyncRequest>,
    handlers: &mut Vec<AsyncSignalHandler>,
    stop: &std::sync::atomic::AtomicBool,
) -> bool {
    loop {
        if stop.load(Ordering::Acquire) {
            return false;
        }
        match receiver.try_recv() {
            Ok(request) => {
                if stop.load(Ordering::Acquire) {
                    return false;
                }
                accept_async_request(request, handlers);
            }
            Err(TryRecvError::Empty) => return true,
            Err(TryRecvError::Disconnected) => return false,
        }
    }
}

fn run_async_dispatcher(receiver: &Receiver<AsyncRequest>, stop: &std::sync::atomic::AtomicBool) {
    let mut handlers = Vec::new();
    while !stop.load(Ordering::Acquire) {
        if handlers.is_empty() {
            match receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(request) => {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    accept_async_request(request, &mut handlers);
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        if !drain_async_requests(receiver, &mut handlers, stop) {
            return;
        }

        let mut invoked = false;
        let mut index = 0;
        while index < handlers.len() {
            if stop.load(Ordering::Acquire) {
                return;
            }
            let registration = &handlers[index];
            // SAFETY: The runtime retains each registered signal until this
            // worker releases its last handler reference.
            let observed = unsafe { hsa_signal_load_scacquire(registration.signal) };
            if stop.load(Ordering::Acquire) {
                return;
            }
            if !condition_met(registration.condition, observed, registration.compare_value) {
                index += 1;
                continue;
            }
            invoked = true;
            let _scope = CallbackScope::enter();
            // SAFETY: HSA requires the callback and argument to remain valid
            // while the registration is active.
            let keep = unsafe { (registration.handler)(observed, registration.arg.as_ptr()) };
            if keep {
                index += 1;
            } else {
                let removed = handlers.remove(index);
                if let Ok(mut guard) = lock() {
                    if let Some(runtime) = guard.as_mut() {
                        runtime.release_async_signal(removed.signal);
                    }
                }
            }
        }

        if stop.load(Ordering::Acquire) {
            return;
        }
        if invoked {
            thread::yield_now();
        } else {
            match receiver.recv_timeout(Duration::from_micros(20)) {
                Ok(request) => {
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    accept_async_request(request, &mut handlers);
                }
                Err(RecvTimeoutError::Timeout) => (),
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_signal_async_handler(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    handler: SignalHandler,
    arg: *mut c_void,
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
        let Some(handler) = handler else {
            return INVALID_ARGUMENT;
        };
        if !matches!(
            condition,
            SIGNAL_CONDITION_EQ | SIGNAL_CONDITION_NE | SIGNAL_CONDITION_LT | SIGNAL_CONDITION_GTE
        ) {
            return INVALID_ARGUMENT;
        }
        if !runtime.owns_signal(signal) {
            return INVALID_SIGNAL;
        }
        let address = signal.handle as usize;
        if !runtime.interrupt_signals.contains_key(&address)
            && !runtime.owned_ipc_signals.contains_key(&address)
            && !runtime.imported_ipc_signals.contains_key(&address)
        {
            return INVALID_SIGNAL;
        }
        if let Err(status) = runtime.retain_async_signal(signal) {
            return status;
        }
        let status = runtime.enqueue_async(AsyncRequest::Signal {
            signal,
            condition,
            compare_value,
            handler,
            arg: unsafe { CallbackArg::new(arg) },
        });
        if status != SUCCESS {
            runtime.release_async_signal(signal);
        }
        status
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn first_event_retries_only_after_the_page_was_offered() {
        let mut retries = 0;
        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), true, false, || {
            retries += 1;
            Ok(7)
        });
        assert_eq!(result, Ok(7));
        assert_eq!(retries, 1);

        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), false, false, || {
            retries += 1;
            Ok(8)
        });
        assert_eq!(result, Err(()));
        assert_eq!(retries, 1);

        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), true, true, || {
            retries += 1;
            Ok(9)
        });
        assert_eq!(result, Err(()));
        assert_eq!(retries, 1);
    }

    #[test]
    #[ignore = "requires a qualified GPU and KFD runtime"]
    fn destroyed_signal_storage_is_reused() {
        assert_eq!(crate::hsa_init(), SUCCESS);
        let mut signal = HsaSignal { handle: 0 };
        let mut previous = 0;
        for value in 0..10_000 {
            // SAFETY: The test supplies writable output storage and owns each
            // returned signal until destruction below.
            assert_eq!(
                unsafe { hsa_signal_create(value, 0, std::ptr::null(), &raw mut signal) },
                SUCCESS
            );
            assert_eq!(unsafe { hsa_signal_load_relaxed(signal) }, value);
            if value != 0 {
                assert_eq!(signal.handle, previous);
            }
            previous = signal.handle;
            // SAFETY: No other work retains this signal.
            assert_eq!(unsafe { hsa_signal_destroy(signal) }, SUCCESS);
        }
        assert_eq!(crate::hsa_shut_down(), SUCCESS);
    }

    unsafe extern "C" fn count_rearmed_handler(_value: SignalValue, arg: *mut c_void) -> bool {
        // SAFETY: The test passes a live AtomicU64 for the worker lifetime.
        let calls = unsafe { &*(arg.cast::<AtomicU64>()) };
        calls.fetch_add(1, Ordering::Relaxed) < 2
    }

    struct AsyncSerializationProbe {
        active: AtomicU64,
        maximum_active: AtomicU64,
        completed: AtomicU64,
    }

    unsafe extern "C" fn serialized_async_function(arg: *mut c_void) {
        // SAFETY: The test passes a live probe for the dispatcher lifetime.
        let probe = unsafe { &*(arg.cast::<AsyncSerializationProbe>()) };
        let active = probe.active.fetch_add(1, Ordering::AcqRel) + 1;
        probe.maximum_active.fetch_max(active, Ordering::AcqRel);
        thread::sleep(Duration::from_millis(2));
        probe.active.fetch_sub(1, Ordering::AcqRel);
        probe.completed.fetch_add(1, Ordering::Release);
    }

    struct StopDispatcherProbe {
        stop: std::sync::atomic::AtomicBool,
        later_calls: AtomicU64,
    }

    unsafe extern "C" fn stop_dispatcher_from_callback(arg: *mut c_void) {
        // SAFETY: The probe remains live until the dispatcher returns.
        let probe = unsafe { &*(arg.cast::<StopDispatcherProbe>()) };
        probe.stop.store(true, Ordering::Release);
    }

    unsafe extern "C" fn count_later_callback(arg: *mut c_void) {
        // SAFETY: The probe remains live until the dispatcher returns.
        let probe = unsafe { &*(arg.cast::<StopDispatcherProbe>()) };
        probe.later_calls.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn dispatcher_does_not_invoke_queued_callbacks_after_callback_requests_stop() {
        let probe = StopDispatcherProbe {
            stop: std::sync::atomic::AtomicBool::new(false),
            later_calls: AtomicU64::new(0),
        };
        // SAFETY: The probe outlives the dispatcher and uses atomic fields.
        let argument = unsafe { CallbackArg::new((&raw const probe).cast_mut().cast()) };
        let (sender, receiver) = std::sync::mpsc::channel();
        sender
            .send(AsyncRequest::Function {
                callback: stop_dispatcher_from_callback,
                arg: argument,
            })
            .unwrap();
        sender
            .send(AsyncRequest::Function {
                callback: count_later_callback,
                arg: argument,
            })
            .unwrap();
        drop(sender);

        run_async_dispatcher(&receiver, &probe.stop);

        assert!(probe.stop.load(Ordering::Acquire));
        assert_eq!(probe.later_calls.load(Ordering::Relaxed), 0);
    }

    unsafe extern "C" fn shutdown_from_async_callback(arg: *mut c_void) {
        // SAFETY: The test retains this atomic until the dispatcher exits.
        let status = unsafe { &*(arg.cast::<AtomicU64>()) };
        status.store(u64::from(crate::hsa_shut_down()), Ordering::Release);
    }

    #[test]
    fn async_callback_without_runtime_reports_not_initialized() {
        let status = AtomicU64::new(0);
        let (sender, receiver) = std::sync::mpsc::channel();
        sender
            .send(AsyncRequest::Function {
                callback: shutdown_from_async_callback,
                // SAFETY: The atomic remains live until dispatch returns.
                arg: unsafe { CallbackArg::new((&raw const status).cast_mut().cast()) },
            })
            .unwrap();
        drop(sender);
        let stop = std::sync::atomic::AtomicBool::new(false);
        run_async_dispatcher(&receiver, &stop);
        assert_eq!(status.load(Ordering::Acquire), u64::from(NOT_INITIALIZED));
    }

    struct ShutdownSignalProbe {
        status: AtomicU64,
        stop: std::sync::atomic::AtomicBool,
    }

    unsafe extern "C" fn shutdown_from_signal_handler(
        _value: SignalValue,
        arg: *mut c_void,
    ) -> bool {
        // SAFETY: The test retains the probe until the dispatcher joins.
        let probe = unsafe { &*(arg.cast::<ShutdownSignalProbe>()) };
        probe
            .status
            .store(u64::from(crate::hsa_shut_down()), Ordering::Release);
        probe.stop.store(true, Ordering::Release);
        false
    }

    #[test]
    fn async_signal_handler_without_runtime_reports_not_initialized() {
        let storage = AmdSignal::user(1);
        let signal = HsaSignal {
            handle: (&raw const storage) as u64,
        };
        let probe = std::sync::Arc::new(ShutdownSignalProbe {
            status: AtomicU64::new(0),
            stop: std::sync::atomic::AtomicBool::new(false),
        });
        let (sender, receiver) = std::sync::mpsc::channel();
        sender
            .send(AsyncRequest::Signal {
                signal,
                condition: SIGNAL_CONDITION_EQ,
                compare_value: 1,
                handler: shutdown_from_signal_handler,
                // SAFETY: The Arc remains live until the dispatcher joins.
                arg: unsafe { CallbackArg::new(std::sync::Arc::as_ptr(&probe).cast_mut().cast()) },
            })
            .unwrap();
        let worker_probe = probe.clone();
        let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_probe.stop));
        worker.join().unwrap();
        drop(sender);
        assert_eq!(
            probe.status.load(Ordering::Acquire),
            u64::from(NOT_INITIALIZED)
        );
    }

    #[test]
    fn async_handler_rearms_while_condition_remains_satisfied() {
        let storage = AmdSignal::user(1);
        let signal = HsaSignal {
            handle: (&raw const storage) as u64,
        };
        let calls = AtomicU64::new(0);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::channel();
        sender
            .send(AsyncRequest::Signal {
                signal,
                condition: SIGNAL_CONDITION_EQ,
                compare_value: 1,
                handler: count_rearmed_handler,
                // SAFETY: The atomic remains live until the dispatcher joins.
                arg: unsafe { CallbackArg::new((&raw const calls).cast_mut().cast()) },
            })
            .unwrap();
        let worker_stop = stop.clone();
        let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_stop));

        let timeout = if cfg!(miri) {
            Duration::from_secs(5)
        } else {
            Duration::from_millis(100)
        };
        let deadline = std::time::Instant::now() + timeout;
        while calls.load(Ordering::Relaxed) < 3 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        worker.join().unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn async_callbacks_execute_serially() {
        let probe = AsyncSerializationProbe {
            active: AtomicU64::new(0),
            maximum_active: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::channel();
        // SAFETY: The probe outlives the dispatcher and uses atomic fields.
        let argument = unsafe { CallbackArg::new((&raw const probe).cast_mut().cast()) };
        for _ in 0..2 {
            sender
                .send(AsyncRequest::Function {
                    callback: serialized_async_function,
                    arg: argument,
                })
                .unwrap();
        }
        let worker_stop = stop.clone();
        let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_stop));

        let timeout = if cfg!(miri) {
            Duration::from_secs(5)
        } else {
            Duration::from_millis(100)
        };
        let deadline = std::time::Instant::now() + timeout;
        while probe.completed.load(Ordering::Acquire) < 2 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        worker.join().unwrap();

        assert_eq!(probe.completed.load(Ordering::Acquire), 2);
        assert_eq!(probe.maximum_active.load(Ordering::Acquire), 1);
    }

    #[test]
    fn lock_free_signal_lookup_observes_atomic_invalidation() {
        let storage = std::sync::Arc::new(AmdSignal::user(7));
        let handle = HsaSignal {
            handle: std::sync::Arc::as_ptr(&storage) as u64,
        };
        let observed_live = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader_storage = storage.clone();
        let reader_observed = observed_live.clone();
        let reader = thread::spawn(move || {
            loop {
                // SAFETY: Both threads retain the signal allocation until join.
                if unsafe { signal_ptr(handle) }.is_none() {
                    break;
                }
                reader_observed.store(true, Ordering::Release);
                thread::yield_now();
            }
            drop(reader_storage);
        });
        while !observed_live.load(Ordering::Acquire) {
            thread::yield_now();
        }
        storage
            .kind
            .store(AMD_SIGNAL_KIND_INVALID, Ordering::Release);
        reader.join().unwrap();
        // SAFETY: The allocation remains live through this final lookup.
        assert!(unsafe { signal_ptr(handle) }.is_none());
    }

    #[test]
    fn amd_signal_helpers_match_the_public_abi() {
        let _: unsafe extern "C" fn(HsaSignal, *mut u32) -> Status = hsa_amd_signal_get_event_id;
        let _: unsafe extern "C" fn(AsyncFunction, *mut c_void) -> Status = hsa_amd_async_function;
        let _: unsafe extern "C" fn(HsaSignal, *mut HsaAmdIpcSignal) -> Status =
            hsa_amd_ipc_signal_create;
        let _: unsafe extern "C" fn(*const HsaAmdIpcSignal, *mut HsaSignal) -> Status =
            hsa_amd_ipc_signal_attach;
    }

    #[test]
    fn ipc_signal_storage_matches_the_rocr_shared_layout() {
        assert_eq!(size_of::<HsaAmdIpcSignal>(), 32);
        assert_eq!(align_of::<HsaAmdIpcSignal>(), 4);
        assert_eq!(size_of::<SharedSignal>(), 128);
        assert_eq!(align_of::<SharedSignal>(), 64);
        assert_eq!(std::mem::offset_of!(SharedSignal, signal), 0);
        assert_eq!(std::mem::offset_of!(SharedSignal, sdma_start_ts), 64);
        assert_eq!(std::mem::offset_of!(SharedSignal, core_signal), 72);
        assert_eq!(std::mem::offset_of!(SharedSignal, id), 80);
        assert_eq!(std::mem::offset_of!(SharedSignal, sdma_end_ts), 96);

        let signal = SharedSignal::ipc(17);
        assert!(signal.is_ipc());
        assert_eq!(
            signal.signal.kind.load(Ordering::Relaxed),
            AMD_SIGNAL_KIND_USER
        );
        assert_eq!(signal.signal.value.load(Ordering::Relaxed), 17);
    }

    #[test]
    fn signal_consumers_reject_duplicates() {
        let consumers = [HsaAgent { handle: 1 }, HsaAgent { handle: 1 }];
        // SAFETY: The array contains the declared number of readable handles.
        assert!(!unsafe { valid_consumers(consumers.len() as u32, consumers.as_ptr()) });
        // SAFETY: A zero count permits a null consumer pointer.
        assert!(unsafe { valid_consumers(0, std::ptr::null()) });
        assert!(amd_signal_uses_consumer_list(0));
        assert!(!amd_signal_uses_consumer_list(AMD_SIGNAL_AMD_GPU_ONLY));
        assert!(!amd_signal_uses_consumer_list(AMD_SIGNAL_IPC));
        assert!(!amd_signal_uses_consumer_list(
            AMD_SIGNAL_AMD_GPU_ONLY | AMD_SIGNAL_IPC
        ));
    }

    #[test]
    fn signal_consumers_select_rocr_signal_types() {
        let cpu = [HsaAgent { handle: CPU_AGENT }];
        let gpu = [HsaAgent {
            handle: GPU_AGENT_BASE,
        }];
        // SAFETY: Each nonzero count names a complete readable array.
        unsafe {
            assert_eq!(signal_uses_interrupt(0, std::ptr::null(), 0), Ok(true));
            assert_eq!(signal_uses_interrupt(1, cpu.as_ptr(), 0), Ok(true));
            assert_eq!(signal_uses_interrupt(1, gpu.as_ptr(), 0), Ok(false));
            assert_eq!(
                signal_uses_interrupt(1, std::ptr::null(), AMD_SIGNAL_AMD_GPU_ONLY),
                Ok(false)
            );
            assert_eq!(
                signal_uses_interrupt(1, std::ptr::null(), AMD_SIGNAL_IPC),
                Ok(false)
            );
        }
    }

    #[test]
    fn interrupt_signal_exposes_the_kfd_mailbox_layout() {
        let signal = AmdSignal::interrupt(7, 0x1234_5000, 19);
        assert_eq!(signal.kind.load(Ordering::Relaxed), AMD_SIGNAL_KIND_USER);
        assert_eq!(signal.value.load(Ordering::Relaxed), 7);
        assert_eq!(signal.event_mailbox_ptr, 0x1234_5000);
        assert_eq!(signal.event_id, 19);
    }

    #[test]
    fn value_pointer_requires_a_busy_wait_signal() {
        let user = AmdSignal::user(7);
        let doorbell = AmdSignal::doorbell(0x1000, 0x2000);
        assert_eq!(
            signal_value_pointer(&user, false),
            Ok((&raw const user.value).cast::<SignalValue>().cast_mut())
        );
        assert_eq!(signal_value_pointer(&user, true), Err(INVALID_ARGUMENT));
        assert_eq!(
            signal_value_pointer(&doorbell, false),
            Err(INVALID_ARGUMENT)
        );
    }

    #[test]
    fn signal_group_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(
            u32,
            *const HsaSignal,
            u32,
            *const HsaAgent,
            *mut HsaSignalGroup,
        ) -> Status = hsa_signal_group_create;
        let _: extern "C" fn(HsaSignalGroup) -> Status = hsa_signal_group_destroy;
        let _: unsafe extern "C" fn(
            HsaSignalGroup,
            *const u32,
            *const SignalValue,
            u32,
            *mut HsaSignal,
            *mut SignalValue,
        ) -> Status = hsa_signal_group_wait_any_relaxed;
        let _: unsafe extern "C" fn(
            HsaSignalGroup,
            *const u32,
            *const SignalValue,
            u32,
            *mut HsaSignal,
            *mut SignalValue,
        ) -> Status = hsa_signal_group_wait_any_scacquire;
    }

    #[test]
    fn base_atomic_entry_points_preserve_signal_values() {
        let storage = AmdSignal::user(7);
        let signal = HsaSignal {
            handle: (&raw const storage) as u64,
        };

        // SAFETY: The handle points to live, aligned user-signal storage for
        // the duration of every call below.
        unsafe {
            assert_eq!(hsa_signal_load_acquire(signal), 7);
            hsa_signal_store_release(signal, 11);
            assert_eq!(hsa_signal_exchange_relaxed(signal, 13), 11);

            assert_eq!(hsa_signal_cas_scacq_screl(signal, 12, 17), 13);
            assert_eq!(hsa_signal_load_relaxed(signal), 13);
            assert_eq!(hsa_signal_cas_acq_rel(signal, 13, 17), 13);

            hsa_signal_add_relaxed(signal, 5);
            hsa_signal_subtract_acquire(signal, 2);
            hsa_signal_and_screlease(signal, 0xf);
            hsa_signal_or_acq_rel(signal, 0x20);
            hsa_signal_xor_release(signal, 0x3);
            assert_eq!(hsa_signal_load_scacquire(signal), 0x27);

            hsa_signal_silent_store_screlease(signal, 41);
            assert_eq!(
                hsa_signal_wait_relaxed(signal, SIGNAL_CONDITION_EQ, 41, 0, 0),
                41
            );
            assert_eq!(
                hsa_signal_wait_acquire(signal, SIGNAL_CONDITION_GTE, 40, 0, 0),
                41
            );
        }
    }

    #[test]
    fn wait_timeout_uses_system_timestamp_ticks() {
        let frequency = 100_000_000;
        assert!(!timeout_elapsed(Duration::from_nanos(9), 1, frequency));
        assert!(timeout_elapsed(Duration::from_nanos(10), 1, frequency));
        assert!(!timeout_elapsed(
            Duration::from_millis(999),
            frequency,
            frequency
        ));
        assert!(timeout_elapsed(
            Duration::from_secs(1),
            frequency,
            frequency
        ));
        assert!(!timeout_elapsed(
            Duration::from_secs(1),
            u64::MAX,
            frequency
        ));
    }

    #[test]
    fn wait_many_reports_original_indices_and_satisfying_values() {
        let first = AmdSignal::user(3);
        let second = AmdSignal::user(8);
        let signals = [
            HsaSignal { handle: 0 },
            HsaSignal { handle: 1 },
            HsaSignal {
                handle: (&raw const first) as u64,
            },
            HsaSignal {
                handle: (&raw const second) as u64,
            },
        ];
        let conditions = [
            SIGNAL_CONDITION_EQ,
            SIGNAL_CONDITION_EQ,
            SIGNAL_CONDITION_GTE,
            SIGNAL_CONDITION_LT,
        ];
        let values = [0, 0, 3, 9];
        let mut satisfying = 0;
        let stop = AtomicBool::new(false);

        // SAFETY: The validated handles point to live, aligned signal storage
        // for the duration of each wait; the invalid handle is never read.
        unsafe {
            let mut valid = collect_wait_signals(&signals, |signal| {
                signal.handle == signals[2].handle || signal.handle == signals[3].handle
            });
            assert_eq!(
                signal_wait_any_open(
                    &valid,
                    &conditions,
                    &values,
                    0,
                    0,
                    &raw mut satisfying,
                    &stop
                ),
                2
            );
            assert_eq!(satisfying, 3);

            let mut satisfying_all = [-1; 4];
            assert_eq!(
                signal_wait_all_open(
                    &mut valid,
                    &conditions,
                    &values,
                    0,
                    0,
                    satisfying_all.as_mut_ptr(),
                    &stop,
                ),
                0
            );
            assert_eq!(satisfying_all, [0, 0, 3, 8]);
        }
    }

    #[test]
    fn wait_all_zeroes_unsatisfied_values_on_timeout() {
        let storage = AmdSignal::user(3);
        let signals = [HsaSignal {
            handle: (&raw const storage) as u64,
        }];
        let conditions = [SIGNAL_CONDITION_EQ];
        let values = [4];
        let mut satisfying = [-1];
        let stop = AtomicBool::new(false);

        // SAFETY: The validated signal storage remains live for the wait.
        unsafe {
            let mut valid = collect_wait_signals(&signals, |_| true);
            assert_eq!(
                signal_wait_all_open(
                    &mut valid,
                    &conditions,
                    &values,
                    0,
                    0,
                    satisfying.as_mut_ptr(),
                    &stop,
                ),
                u32::MAX
            );
        }
        assert_eq!(satisfying, [0]);
    }

    #[test]
    fn wait_all_accepts_comparison_values_as_output_storage() -> Result<(), &'static str> {
        let storage = AmdSignal::user(7);
        let signals = [HsaSignal {
            handle: (&raw const storage) as u64,
        }];
        let conditions = [SIGNAL_CONDITION_EQ];
        let mut values = [7];
        let stop = AtomicBool::new(false);

        // SAFETY: The signal storage and input arrays remain live through the
        // copy. The output aliases the comparison array only after the copy.
        unsafe {
            let inputs =
                signal_wait_inputs(1, signals.as_ptr(), conditions.as_ptr(), values.as_ptr())
                    .ok_or("valid wait inputs")?;
            let mut entries = collect_wait_signals(&inputs.signals, |_| true);
            assert_eq!(
                signal_wait_all_open(
                    &mut entries,
                    &inputs.conditions,
                    &inputs.values,
                    0,
                    0,
                    values.as_mut_ptr(),
                    &stop,
                ),
                0
            );
        }
        assert_eq!(values, [7]);
        Ok(())
    }

    #[test]
    fn shutdown_cancels_unbounded_multi_signal_waits() {
        let storage = AmdSignal::user(3);
        let handles = [HsaSignal {
            handle: (&raw const storage) as u64,
        }];
        let conditions = [SIGNAL_CONDITION_EQ];
        let values = [4];
        let stop = AtomicBool::new(true);
        // SAFETY: The accepted handle stays live through both calls.
        let mut signals = unsafe { collect_wait_signals(&handles, |_| true) };
        // SAFETY: Both calls receive valid slices and no output pointer.
        unsafe {
            assert_eq!(
                signal_wait_any_open(
                    &signals,
                    &conditions,
                    &values,
                    u64::MAX,
                    0,
                    std::ptr::null_mut(),
                    &stop,
                ),
                u32::MAX
            );
            assert_eq!(
                signal_wait_all_open(
                    &mut signals,
                    &conditions,
                    &values,
                    u64::MAX,
                    0,
                    std::ptr::null_mut(),
                    &stop,
                ),
                u32::MAX
            );
        }
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn retired_signal_ends_multi_signal_waits() {
        let retired = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_retired = retired.clone();
        let worker_stop = stop.clone();
        let (ready_send, ready_recv) = std::sync::mpsc::channel();
        let (result_send, result_recv) = std::sync::mpsc::channel();
        let worker = thread::spawn(move || {
            let storage = AmdSignal::user(3);
            let handle = HsaSignal {
                handle: (&raw const storage) as u64,
            };
            // SAFETY: This worker owns the signal storage until the wait exits.
            let mut entries = unsafe { collect_wait_signals(&[handle], |_| true) };
            entries[0].retired = Some(worker_retired);
            ready_send.send(()).unwrap();
            let conditions = [SIGNAL_CONDITION_EQ];
            let values = [4];
            // SAFETY: The signal and both input slices remain live for the wait.
            let result = unsafe {
                signal_wait_any_open(
                    &entries,
                    &conditions,
                    &values,
                    u64::MAX,
                    0,
                    std::ptr::null_mut(),
                    &worker_stop,
                )
            };
            result_send.send(result).unwrap();
        });
        ready_recv.recv_timeout(Duration::from_secs(1)).unwrap();
        retired.store(true, Ordering::Release);
        let result = result_recv.recv_timeout(Duration::from_secs(1));
        if result.is_err() {
            stop.store(true, Ordering::Release);
        }
        worker.join().unwrap();
        assert!(matches!(result, Ok(value) if value == u32::MAX));

        stop.store(false, Ordering::Release);
        let storage = AmdSignal::user(3);
        let handle = HsaSignal {
            handle: (&raw const storage) as u64,
        };
        // SAFETY: The signal storage remains live for this immediate wait.
        let mut entries = unsafe { collect_wait_signals(&[handle], |_| true) };
        entries[0].retired = Some(retired);
        let conditions = [SIGNAL_CONDITION_EQ];
        let values = [4];
        // SAFETY: Both input slices remain live for this immediate wait.
        assert_eq!(
            unsafe {
                signal_wait_all_open(
                    &mut entries,
                    &conditions,
                    &values,
                    u64::MAX,
                    0,
                    std::ptr::null_mut(),
                    &stop,
                )
            },
            u32::MAX
        );
    }

    #[test]
    fn ipc_reattach_does_not_clear_a_prior_waiters_retirement() {
        let mut record = AsyncSignalRecord {
            references: 1,
            retired: Arc::new(AtomicBool::new(false)),
            copy_profile: None,
        };
        let prior_waiter = record.retired.clone();
        record.retired.store(true, Ordering::Release);
        record.revive();
        assert!(prior_waiter.load(Ordering::Acquire));
        assert!(!record.retired.load(Ordering::Acquire));
        assert!(!Arc::ptr_eq(&prior_waiter, &record.retired));

        let current_waiter = record.retired.clone();
        record.revive();
        assert!(Arc::ptr_eq(&current_waiter, &record.retired));
    }

    #[test]
    fn amd_multi_waits_require_runtime_initialization() {
        let signal = AmdSignal::user(3);
        let signals = [HsaSignal {
            handle: (&raw const signal) as u64,
        }];
        let conditions = [SIGNAL_CONDITION_EQ];
        let values = [3];
        let mut satisfying = -1;

        // SAFETY: All arrays contain one readable/writable entry and the signal
        // storage stays live. The runtime is intentionally not initialized.
        unsafe {
            assert_eq!(
                hsa_amd_signal_wait_any(
                    1,
                    signals.as_ptr(),
                    conditions.as_ptr(),
                    values.as_ptr(),
                    0,
                    0,
                    &raw mut satisfying,
                ),
                u32::MAX
            );
            assert_eq!(satisfying, -1);
            assert_eq!(
                hsa_amd_signal_wait_all(
                    1,
                    signals.as_ptr(),
                    conditions.as_ptr(),
                    values.as_ptr(),
                    0,
                    0,
                    &raw mut satisfying,
                ),
                u32::MAX
            );
            assert_eq!(satisfying, -1);
        }
    }
}
