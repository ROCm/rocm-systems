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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering, fence};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use rocddi::gpu::event::linux::{SignalEvent, create_signal_event};
use rocddi::memory::interop::linux::{self as linux_interop, KfdIpcMemoryHandle};
use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};

use crate::ffi::*;
use crate::runtime::{Runtime, boundary, initialized_mut, lock, map_error};

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
    let frequency = lock()
        .ok()
        .and_then(|guard| {
            guard.as_ref().and_then(|runtime| {
                runtime.gpus.first().and_then(|gpu| {
                    gpu.device
                        .gpu()
                        .and_then(|gpu| gpu.clock_counters())
                        .ok()
                        .map(|counters| counters.system_frequency)
                })
            })
        })
        .unwrap_or(0);
    if frequency != 0 {
        SYSTEM_FREQUENCY_HZ.store(frequency, Ordering::Release);
    }
    frequency
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
    pub(crate) kind: i64,
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
            kind: AMD_SIGNAL_KIND_USER,
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
            kind: AMD_SIGNAL_KIND_DOORBELL,
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

const _: () = assert!(size_of::<AmdSignal>() == SIGNAL_BYTES);

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
        arg: usize,
    },
    Signal {
        signal: HsaSignal,
        condition: u32,
        compare_value: SignalValue,
        handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
        arg: usize,
    },
}

/// Persistent signal condition re-evaluated by the dispatcher.
struct AsyncSignalHandler {
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    handler: unsafe extern "C" fn(SignalValue, *mut c_void) -> bool,
    arg: usize,
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
                (SIGNAL_BYTES * SIGNALS_PER_SLAB) as u64,
                4096,
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
        unsafe { *(address as *const i64) != AMD_SIGNAL_KIND_INVALID }
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
                        SIGNAL_EVENT_PAGE_BYTES,
                        4096,
                        DeviceAccess::READ | DeviceAccess::WRITE,
                    )
                    .map_err(map_error)?
            };
            let info = allocation.info();
            let host = info.host_address.ok_or(OUT_OF_RESOURCES)?;
            if info.device_address != host as u64 || info.size < SIGNAL_EVENT_PAGE_BYTES {
                return Err(OUT_OF_RESOURCES);
            }
            self.signal_event_page = Some(allocation);
        }
        let host = self
            .signal_event_page
            .as_ref()
            .and_then(|allocation| allocation.info().host_address)
            .ok_or(OUT_OF_RESOURCES)?;
        Ok((host, true))
    }

    pub(crate) fn create_signal_event(
        &mut self,
    ) -> Option<(rocddi::gpu::event::linux::SignalEvent, usize, u32)> {
        let (page_host, install_page) = self.ensure_signal_event_page().ok()?;
        let gpu = self.gpus.first()?;
        let event_page = if install_page {
            Some(self.signal_event_page.as_ref()?)
        } else {
            None
        };
        let mut event = gpu
            .device
            .gpu()
            .and_then(|device| create_signal_event(device, event_page))
            .ok()?;
        if install_page
            && PROCESS_SIGNAL_EVENT_PAGE
                .set(page_host)
                .is_err_and(|published| published != page_host)
        {
            let _ = event.destroy();
            return None;
        }
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
        self.ensure_signal_slab()?;
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
        self.owned_ipc_signals.contains_key(&address)
            || self.imported_ipc_signals.contains_key(&address)
            || self.signal_slabs.iter().any(|slab| slab.owns_live(address))
    }

    fn create_ipc_signal(&mut self, value: SignalValue) -> Result<HsaSignal, Status> {
        let allocation = {
            let (gpu, peers) = self.gpus.split_first().ok_or(OUT_OF_RESOURCES)?;
            let peers = peers.iter().map(|peer| &peer.device).collect::<Vec<_>>();
            gpu.device
                .allocate_with_peers(
                    &peers,
                    MemoryKind::System,
                    IPC_SIGNAL_ALLOCATION_BYTES,
                    IPC_SIGNAL_ALLOCATION_BYTES,
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
            return Ok(HsaSignal {
                handle: *address as u64,
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
        if let Some(imported) = self.imported_ipc_signals.get_mut(&address) {
            if imported.references > 1 {
                imported.references -= 1;
                return SUCCESS;
            }
            let Some(mut imported) = self.imported_ipc_signals.remove(&address) else {
                return ERROR;
            };
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
        if !self.owns_signal(signal) {
            return INVALID_SIGNAL;
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
        // Slab storage remains mapped until shutdown so asynchronous waiters do
        // not race an unmap; the invalid kind prevents subsequent API use.
        // SAFETY: owns_signal validated this live slab slot.
        unsafe { (address as *mut i64).write(AMD_SIGNAL_KIND_INVALID) };
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
        if let Some(page) = self.signal_event_page.take() {
            let host = page.info().host_address;
            if host.is_some_and(|host| PROCESS_SIGNAL_EVENT_PAGE.get() == Some(&host)) {
                if let Err(error) = linux_interop::retain_kfd_signal_event_page_for_process(page) {
                    if status == SUCCESS {
                        status = map_error(error);
                    }
                }
            }
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

pub(crate) unsafe fn signal_ref(signal: HsaSignal) -> Option<&'static AmdSignal> {
    if signal.handle == 0 {
        return None;
    }
    // SAFETY: HSA signal handles created by this frontend are aligned pointers
    // to AmdSignal records retained by the runtime or their owning queue.
    let signal = unsafe { &*(signal.handle as usize as *const AmdSignal) };
    (signal.kind != AMD_SIGNAL_KIND_INVALID).then_some(signal)
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
    unsafe { signal_ref(signal) }.map_or(0, |signal| signal.value.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_load_scacquire(signal: HsaSignal) -> SignalValue {
    // SAFETY: Callers must supply a live HSA signal handle.
    unsafe { signal_ref(signal) }.map_or(0, |signal| signal.value.load(Ordering::Acquire))
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
    let Some(signal) = (unsafe { signal_ref(signal) }) else {
        return;
    };
    if signal.kind == AMD_SIGNAL_KIND_DOORBELL {
        let address = signal.value.load(Ordering::Relaxed) as usize;
        // SAFETY: A queue doorbell signal contains its live MMIO mapping.
        unsafe { (address as *mut u64).write_volatile(value as u64) };
    } else {
        signal.value.store(value, Ordering::Relaxed);
    }
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
    let Some(signal) = (unsafe { signal_ref(signal) }) else {
        return;
    };
    if signal.kind == AMD_SIGNAL_KIND_DOORBELL {
        fence(Ordering::Release);
        let address = signal.value.load(Ordering::Relaxed) as usize;
        // SAFETY: A queue doorbell signal contains its live MMIO mapping.
        unsafe { (address as *mut u64).write_volatile(value as u64) };
    } else {
        signal.value.store(value, Ordering::Release);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_signal_store_release(signal: HsaSignal, value: SignalValue) {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent release spelling.
    unsafe { hsa_signal_store_screlease(signal, value) };
}

unsafe fn signal_exchange(signal: HsaSignal, value: SignalValue, order: Ordering) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe { signal_ref(signal) }.map_or(0, |signal| signal.value.swap(value, order))
}

unsafe fn signal_compare_exchange(
    signal: HsaSignal,
    expected: SignalValue,
    value: SignalValue,
    success: Ordering,
    failure: Ordering,
) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe { signal_ref(signal) }.map_or(0, |signal| {
        signal
            .value
            .compare_exchange(expected, value, success, failure)
            .unwrap_or_else(|observed| observed)
    })
}

unsafe fn signal_fetch_update(
    signal: HsaSignal,
    value: SignalValue,
    order: Ordering,
    operation: fn(&AtomicI64, SignalValue, Ordering),
) {
    // SAFETY: Callers must supply a live HSA user signal.
    if let Some(signal) = unsafe { signal_ref(signal) } {
        operation(&signal.value, value, order);
    }
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
    let Some(signal) = (unsafe { signal_ref(signal) }) else {
        return 0;
    };
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
        if timeout_hint != u64::MAX && timeout_elapsed(start.elapsed(), timeout_hint, frequency) {
            return observed;
        }
        wait_pause(wait_state_hint, &start);
    }
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

unsafe fn signal_wait_inputs<'a>(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
) -> Option<(&'a [HsaSignal], &'a [u32], &'a [SignalValue])> {
    let count = signal_count as usize;
    if count == 0 {
        return Some((&[], &[], &[]));
    }
    if count != 0 && (signals.is_null() || conditions.is_null() || values.is_null()) {
        return None;
    }
    // SAFETY: The caller supplies signal_count readable elements in each array.
    Some(unsafe {
        (
            std::slice::from_raw_parts(signals, count),
            std::slice::from_raw_parts(conditions, count),
            std::slice::from_raw_parts(values, count),
        )
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

fn signal_runtime_is_initialized() -> bool {
    match lock() {
        Ok(runtime) => runtime.is_some(),
        Err(_) => false,
    }
}

unsafe fn signal_wait_any_open(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
) -> u32 {
    let Some((signals, conditions, values)) =
        (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    let valid = signals
        .iter()
        .enumerate()
        .filter_map(|(index, signal)| {
            // SAFETY: NULL and invalidated slab signals are ignored as required.
            unsafe { signal_ref(*signal) }.map(|signal| (index, signal))
        })
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return u32::MAX;
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        for &(index, signal) in &valid {
            let observed = signal.value.load(Ordering::Relaxed);
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
    if !signal_runtime_is_initialized() {
        return u32::MAX;
    }
    // SAFETY: Initialization is established before the public array contract
    // is inspected, matching ROCr's API ordering.
    unsafe {
        signal_wait_any_open(
            signal_count,
            signals,
            conditions,
            values,
            timeout_hint,
            wait_state_hint,
            satisfying_value,
        )
    }
}

unsafe fn signal_wait_all_open(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
) -> u32 {
    let Some((signals, conditions, values)) =
        (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    if !satisfying_values.is_null() {
        // SAFETY: The caller supplied signal_count writable output entries.
        // ROCr publishes zero for invalid and not-yet-satisfied signals even
        // when the wait ultimately times out.
        unsafe { std::ptr::write_bytes(satisfying_values, 0, signals.len()) };
    }
    let mut pending = Vec::with_capacity(signals.len());
    for signal in signals {
        // SAFETY: NULL and invalidated slab signals count as already satisfied.
        if let Some(signal) = unsafe { signal_ref(*signal) } {
            pending.push(Some(signal));
        } else {
            pending.push(None);
        }
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        let mut remaining = false;
        for (index, signal) in pending.iter_mut().enumerate() {
            let Some(signal_ref) = *signal else {
                continue;
            };
            let observed = signal_ref.value.load(Ordering::Relaxed);
            if condition_met(conditions[index], observed, values[index]) {
                if !satisfying_values.is_null() {
                    // SAFETY: The caller supplied signal_count writable entries.
                    unsafe { satisfying_values.add(index).write(observed) };
                }
                *signal = None;
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
    if !signal_runtime_is_initialized() {
        return u32::MAX;
    }
    // SAFETY: Initialization is established before the public array contract
    // is inspected, matching ROCr's API ordering.
    unsafe {
        signal_wait_all_open(
            signal_count,
            signals,
            conditions,
            values,
            timeout_hint,
            wait_state_hint,
            satisfying_values,
        )
    }
}

unsafe fn signal_value_pointer(
    signal: HsaSignal,
    interrupt: bool,
) -> Result<*mut SignalValue, Status> {
    // SAFETY: Callers must supply a live HSA signal handle.
    let Some(signal) = (unsafe { signal_ref(signal) }) else {
        return Err(INVALID_SIGNAL);
    };
    if signal.kind != AMD_SIGNAL_KIND_USER || interrupt {
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
        // SAFETY: Runtime initialization and writable output storage have been
        // established before validating the signal kind.
        let interrupt = runtime
            .interrupt_signals
            .contains_key(&(signal.handle as usize));
        let pointer = match unsafe { signal_value_pointer(signal, interrupt) } {
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
            arg: arg as usize,
        })
    })
}

fn accept_async_request(request: AsyncRequest, handlers: &mut Vec<AsyncSignalHandler>) {
    match request {
        AsyncRequest::Function { callback, arg } => {
            // SAFETY: The public contract requires callback and arg to remain
            // valid until the callback executes.
            unsafe { callback(arg as *mut c_void) };
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
) -> bool {
    loop {
        match receiver.try_recv() {
            Ok(request) => accept_async_request(request, handlers),
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
                Ok(request) => accept_async_request(request, &mut handlers),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        if !drain_async_requests(receiver, &mut handlers) {
            return;
        }

        let mut invoked = false;
        let mut index = 0;
        while index < handlers.len() {
            let registration = &handlers[index];
            // SAFETY: Public signal lifetime rules require registered signals
            // to remain valid while monitoring is active.
            let observed = unsafe { hsa_signal_load_scacquire(registration.signal) };
            if !condition_met(registration.condition, observed, registration.compare_value) {
                index += 1;
                continue;
            }
            invoked = true;
            // SAFETY: HSA requires the callback and argument to remain valid
            // while the registration is active.
            let keep = unsafe { (registration.handler)(observed, registration.arg as *mut c_void) };
            if keep {
                index += 1;
            } else {
                handlers.remove(index);
            }
        }

        if invoked {
            thread::yield_now();
        } else {
            match receiver.recv_timeout(Duration::from_micros(20)) {
                Ok(request) => accept_async_request(request, &mut handlers),
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
        runtime.enqueue_async(AsyncRequest::Signal {
            signal,
            condition,
            compare_value,
            handler,
            arg: arg as usize,
        })
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

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
                arg: (&raw const calls) as usize,
            })
            .unwrap();
        let worker_stop = stop.clone();
        let worker = thread::spawn(move || run_async_dispatcher(&receiver, &worker_stop));

        let deadline = std::time::Instant::now() + Duration::from_millis(100);
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
        let argument = (&raw const probe) as usize;
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

        let deadline = std::time::Instant::now() + Duration::from_millis(100);
        while probe.completed.load(Ordering::Acquire) < 2 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        worker.join().unwrap();

        assert_eq!(probe.completed.load(Ordering::Acquire), 2);
        assert_eq!(probe.maximum_active.load(Ordering::Acquire), 1);
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
        assert_eq!(signal.signal.kind, AMD_SIGNAL_KIND_USER);
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
        assert_eq!(signal.kind, AMD_SIGNAL_KIND_USER);
        assert_eq!(signal.value.load(Ordering::Relaxed), 7);
        assert_eq!(signal.event_mailbox_ptr, 0x1234_5000);
        assert_eq!(signal.event_id, 19);
    }

    #[test]
    fn value_pointer_requires_a_busy_wait_signal() {
        let user = AmdSignal::user(7);
        let doorbell = AmdSignal::doorbell(0x1000, 0x2000);
        // SAFETY: Both handles point to live, aligned signal storage.
        unsafe {
            assert_eq!(
                signal_value_pointer(
                    HsaSignal {
                        handle: (&raw const user) as u64,
                    },
                    false
                ),
                Ok((&raw const user.value).cast::<SignalValue>().cast_mut())
            );
            assert_eq!(
                signal_value_pointer(
                    HsaSignal {
                        handle: (&raw const user) as u64,
                    },
                    true,
                ),
                Err(INVALID_ARGUMENT)
            );
            assert_eq!(
                signal_value_pointer(
                    HsaSignal {
                        handle: (&raw const doorbell) as u64,
                    },
                    false
                ),
                Err(INVALID_ARGUMENT)
            );
        }
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
            HsaSignal {
                handle: (&raw const first) as u64,
            },
            HsaSignal {
                handle: (&raw const second) as u64,
            },
        ];
        let conditions = [
            SIGNAL_CONDITION_EQ,
            SIGNAL_CONDITION_GTE,
            SIGNAL_CONDITION_LT,
        ];
        let values = [0, 3, 9];
        let mut satisfying = 0;

        // SAFETY: All arrays have signal_count entries and nonzero handles point
        // to live, aligned signal storage for the duration of each call.
        unsafe {
            assert_eq!(
                signal_wait_any_open(
                    signals.len() as u32,
                    signals.as_ptr(),
                    conditions.as_ptr(),
                    values.as_ptr(),
                    0,
                    0,
                    &raw mut satisfying,
                ),
                1
            );
            assert_eq!(satisfying, 3);

            let mut satisfying_all = [-1; 3];
            assert_eq!(
                signal_wait_all_open(
                    signals.len() as u32,
                    signals.as_ptr(),
                    conditions.as_ptr(),
                    values.as_ptr(),
                    0,
                    0,
                    satisfying_all.as_mut_ptr(),
                ),
                0
            );
            assert_eq!(satisfying_all, [0, 3, 8]);
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

        // SAFETY: All arrays contain one readable or writable element and the
        // signal storage remains live for the complete wait.
        unsafe {
            assert_eq!(
                signal_wait_all_open(
                    1,
                    signals.as_ptr(),
                    conditions.as_ptr(),
                    values.as_ptr(),
                    0,
                    0,
                    satisfying.as_mut_ptr(),
                ),
                u32::MAX
            );
        }
        assert_eq!(satisfying, [0]);
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
