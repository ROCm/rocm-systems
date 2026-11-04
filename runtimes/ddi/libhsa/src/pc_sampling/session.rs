// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU trap backing and serialized delivery for one PC sampling session.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::size_of;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rocddi::device::Device;
use rocddi::device::gpu::profiling::{PcSampling, PcSamplingMethod, PcSamplingRequest};
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocddi::topology::GpuInfo;

use super::sampling_error;
use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::runtime::{CallbackScope, map_error};
use crate::signal::AmdSignal;

/// Sample size in bytes.
const SAMPLE_BYTES: usize = 64;
/// Metadata size in bytes.
const METADATA_BYTES: usize = 64;
/// Signal size in bytes.
const SIGNAL_BYTES: usize = 64;
/// Max half buffer size in bytes.
const MAX_HALF_BUFFER_BYTES: usize = 4 * 1024 * 1024;
/// Delay between polls of a sample-buffer notification.
const POLL_INTERVAL: Duration = Duration::from_millis(1);
/// Maximum wait for a trap buffer handoff.
const WRITE_TIMEOUT: Duration = Duration::from_millis(100);

// Opaque callback tokens are never reused, even when a later callback runs
// with its context at the same stack address.
/// Monotonic token source for PC sampling copy callbacks.
static NEXT_COPY_TOKEN: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// Active callback context for one PC sampling copy invocation.
    static ACTIVE_COPY: RefCell<Option<CopyContext>> = const { RefCell::new(None) };
}

/// Embedded PC sampling trap program for GFX1200.
const TRAP_GFX1200: &[u8] = include_bytes!("trap_gfx1200.bin");
/// Embedded PC sampling trap program for GFX1250.
const TRAP_GFX1250: &[u8] = include_bytes!("trap_gfx1250.bin");

/// Trap-visible control record for one double-buffered XCC sample stream.
#[repr(C, align(64))]
struct SampleMetadata {
    /// Trap-side sequence used to coordinate buffer handoff.
    ticket: AtomicU64,
    /// Capacity of each sample half in records.
    capacity: u32,
    /// Reserved storage retained for the native layout.
    reserved0: u32,
    /// Number of records written into the first half.
    written0: AtomicU32,
    /// First-half threshold at which a notification is requested.
    watermark0: u32,
    /// GPU address of the first-half notification signal.
    signal0: u64,
    /// Number of records written into the second half.
    written1: AtomicU32,
    /// Second-half notification threshold.
    watermark1: u32,
    /// GPU address of the second-half notification signal.
    signal1: u64,
    /// Reserved storage retained for the native layout.
    reserved1: [u8; 16],
}

/// Trap handler arguments naming the per-XCC sampling buffers.
#[repr(C)]
struct TrapMemory {
    /// GPU address of host-trap sample buffers.
    hosttrap_buffers: u64,
    /// GPU address of stochastic sample buffers.
    stochastic_buffers: u64,
    /// Bytes between corresponding buffers for adjacent XCCs.
    per_xcc_stride: u64,
    /// Reserved storage retained for the native layout.
    reserved: u64,
}

const _: () = {
    assert!(size_of::<SampleMetadata>() == METADATA_BYTES);
    assert!(std::mem::offset_of!(SampleMetadata, capacity) == 8);
    assert!(std::mem::offset_of!(SampleMetadata, written0) == 16);
    assert!(std::mem::offset_of!(SampleMetadata, signal0) == 24);
    assert!(std::mem::offset_of!(SampleMetadata, written1) == 32);
    assert!(std::mem::offset_of!(SampleMetadata, signal1) == 40);
    assert!(size_of::<TrapMemory>() == 32);
    assert!(std::mem::offset_of!(TrapMemory, per_xcc_stride) == 16);
    assert!(size_of::<AmdSignal>() == SIGNAL_BYTES);
};

/// Device allocations retired from a successful native registration. The
/// runtime retains them until its queues have stopped using the old handler.
pub(crate) struct RetiredAssets {
    /// Backing released after old handler users have stopped.
    _allocations: Vec<Allocation>,
}

/// Host view and handoff state for one XCC's sample buffer pair.
struct XccBuffer {
    /// CPU address of the shared metadata and sample allocation.
    host: usize,
    /// Number of samples available in each half.
    capacity: usize,
    /// Byte extent of each sample half.
    half_bytes: usize,
    /// Half currently selected for trap writes.
    selected: usize,
    /// Half and record count awaiting completion of a timed-out swap.
    pending_swap: Option<(usize, usize)>,
    /// CPU addresses of the two notification signals.
    signals: [usize; 2],
}

impl XccBuffer {
    /// Returns the mapped trap metadata for this XCC buffer.
    fn metadata(&self) -> &SampleMetadata {
        // SAFETY: The allocation is mapped and retained by the session.
        unsafe { &*(self.host as *const SampleMetadata) }
    }

    /// Returns the notification signal value for one buffer half.
    fn signal(&self, index: usize) -> &AtomicI64 {
        // SAFETY: Each signal is initialized within the retained allocation.
        &unsafe { &*(self.signals[index] as *const AmdSignal) }.value
    }

    /// Returns the address of one half of the sample storage.
    fn samples(&self, index: usize) -> usize {
        self.host + METADATA_BYTES + index * self.half_bytes
    }
}

/// Native registration, trap backing, and delivery state for one session.
pub(super) struct Resources {
    /// Activated GPU whose VM retains these resources.
    device: Device,
    /// Sampling mode and interval selected by the caller.
    request: PcSamplingRequest,
    /// Native ID retained through cleanup, including an adopted trace.
    trace_id: Option<NonZeroU32>,
    /// Native PC sampling owner once registration succeeds.
    trace: Option<PcSampling>,
    /// All GPU-visible trap code, arguments, and sample backing.
    allocations: Vec<Allocation>,
    /// GPU address of the installed trap handler code.
    code_address: u64,
    /// GPU address of the handler argument record.
    tma_address: u64,
    /// Per-XCC sample buffers and swap state.
    xcc: Vec<XccBuffer>,
    /// Caller callback receiving batches of copied sample records.
    callback:
        unsafe extern "C" fn(*mut c_void, usize, usize, PcSamplingDataCopyCallback, *mut c_void),
    /// Caller-owned callback argument retained through delivery.
    callback_data: CallbackArg,
    /// Pending byte count that triggers a delivery batch.
    threshold: usize,
    /// Sample bytes waiting for the next callback.
    pending: VecDeque<u8>,
    /// Number of samples dropped before the next callback.
    lost: usize,
    /// Whether sampling is currently started.
    active: bool,
    /// Final callback delivery has completed before legacy tool unload.
    callbacks_quiesced: bool,
    /// Keeps backing alive when a handler update may have taken effect.
    trap_may_be_installed: bool,
}

/// Rounds a size up to the required alignment, checking overflow.
fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
}

/// Selects the embedded trap code for a supported GPU generation.
fn trap_code(info: GpuInfo) -> Option<&'static [u8]> {
    match (info.gfx_major, info.gfx_minor, info.gfx_stepping) {
        (12, 0, 0 | 1) => Some(TRAP_GFX1200),
        (12, 5, 0) => Some(TRAP_GFX1250),
        _ => None,
    }
}

/// Checks whether embedded trap code exists for this GPU revision.
pub(crate) fn supported_gpu(info: GpuInfo) -> bool {
    trap_code(info).is_some()
}

impl Resources {
    #[allow(unsafe_code, clippy::too_many_arguments)]
    /// Allocates trap backing and validates a sampling request.
    pub(super) fn prepare(
        device: Device,
        info: GpuInfo,
        request: PcSamplingRequest,
        trace_id: Option<NonZeroU32>,
        buffer_size: usize,
        page_size: usize,
        callback: PcSamplingDataReadyCallback,
        callback_data: CallbackArg,
    ) -> Result<Self, Status> {
        let code = trap_code(info).ok_or(NOT_SUPPORTED)?;
        let callback = callback.ok_or(INVALID_ARGUMENT)?;
        let xcc_count = usize::try_from(info.xcc_count.max(1)).map_err(|_| OUT_OF_RESOURCES)?;
        let share = buffer_size / xcc_count;
        let half_bytes =
            (share / 2).clamp(SAMPLE_BYTES, MAX_HALF_BUFFER_BYTES) / SAMPLE_BYTES * SAMPLE_BYTES;
        let capacity = half_bytes / SAMPLE_BYTES;
        let signal_offset = METADATA_BYTES
            .checked_add(half_bytes.checked_mul(2).ok_or(OUT_OF_RESOURCES)?)
            .ok_or(OUT_OF_RESOURCES)?;
        let stride = align_up(
            signal_offset
                .checked_add(2 * SIGNAL_BYTES)
                .ok_or(OUT_OF_RESOURCES)?,
            256,
        )
        .ok_or(OUT_OF_RESOURCES)?;
        let total = stride.checked_mul(xcc_count).ok_or(OUT_OF_RESOURCES)?;
        let page = u64::try_from(page_size).map_err(|_| OUT_OF_RESOURCES)?;
        let permissions = DeviceAccess::READ | DeviceAccess::WRITE;
        let samples = device
            .allocate(
                MemoryKind::System,
                u64::try_from(total).map_err(|_| OUT_OF_RESOURCES)?,
                page,
                permissions,
            )
            .map_err(map_error)?;
        let sample_info = samples.info();
        let sample_host = sample_info.host_address.ok_or(NOT_SUPPORTED)?;
        let code_allocation = device
            .allocate(
                MemoryKind::System,
                u64::try_from(code.len()).map_err(|_| OUT_OF_RESOURCES)?,
                page,
                permissions | DeviceAccess::EXECUTE,
            )
            .map_err(map_error)?;
        let code_info = code_allocation.info();
        let code_host = code_info.host_address.ok_or(NOT_SUPPORTED)?;
        let tma = device
            .allocate(
                MemoryKind::System,
                u64::try_from(size_of::<TrapMemory>()).map_err(|_| OUT_OF_RESOURCES)?,
                page,
                permissions,
            )
            .map_err(map_error)?;
        let tma_info = tma.info();
        let tma_host = tma_info.host_address.ok_or(NOT_SUPPORTED)?;
        let mut xcc = Vec::new();
        xcc.try_reserve_exact(xcc_count)
            .map_err(|_| OUT_OF_RESOURCES)?;
        for index in 0..xcc_count {
            let offset = index.checked_mul(stride).ok_or(OUT_OF_RESOURCES)?;
            let host = sample_host.checked_add(offset).ok_or(OUT_OF_RESOURCES)?;
            let gpu = sample_info
                .device_address
                .checked_add(u64::try_from(offset).map_err(|_| OUT_OF_RESOURCES)?)
                .ok_or(OUT_OF_RESOURCES)?;
            let signal0 = gpu
                .checked_add(u64::try_from(signal_offset).map_err(|_| OUT_OF_RESOURCES)?)
                .ok_or(OUT_OF_RESOURCES)?;
            let signal1 = signal0
                .checked_add(SIGNAL_BYTES as u64)
                .ok_or(OUT_OF_RESOURCES)?;
            let signal_host0 = host.checked_add(signal_offset).ok_or(OUT_OF_RESOURCES)?;
            let signal_host1 = signal_host0
                .checked_add(SIGNAL_BYTES)
                .ok_or(OUT_OF_RESOURCES)?;
            // SAFETY: All disjoint records fit the mapped allocation, and no
            // GPU trap can access them before handler installation.
            unsafe {
                (host as *mut SampleMetadata).write(SampleMetadata {
                    ticket: AtomicU64::new(0),
                    capacity: u32::try_from(capacity).map_err(|_| OUT_OF_RESOURCES)?,
                    reserved0: 0,
                    written0: AtomicU32::new(0),
                    watermark0: u32::try_from((capacity * 4 / 5).max(1))
                        .map_err(|_| OUT_OF_RESOURCES)?,
                    signal0,
                    written1: AtomicU32::new(0),
                    watermark1: u32::try_from((capacity * 4 / 5).max(1))
                        .map_err(|_| OUT_OF_RESOURCES)?,
                    signal1,
                    reserved1: [0; 16],
                });
                (signal_host0 as *mut AmdSignal).write(AmdSignal::interrupt(1, 0, 0));
                (signal_host1 as *mut AmdSignal).write(AmdSignal::interrupt(1, 0, 0));
                std::ptr::write_bytes((host + METADATA_BYTES) as *mut u8, 0, 2 * half_bytes);
            }
            xcc.push(XccBuffer {
                host,
                capacity,
                half_bytes,
                selected: 0,
                pending_swap: None,
                signals: [signal_host0, signal_host1],
            });
        }
        // SAFETY: Both destination allocations are mapped and large enough.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), code_host as *mut u8, code.len());
            (tma_host as *mut TrapMemory).write(TrapMemory {
                hosttrap_buffers: if request.method == PcSamplingMethod::HostTrap {
                    sample_info.device_address
                } else {
                    0
                },
                stochastic_buffers: if request.method == PcSamplingMethod::Stochastic {
                    sample_info.device_address
                } else {
                    0
                },
                per_xcc_stride: if xcc_count > 1 {
                    u64::try_from(stride).map_err(|_| OUT_OF_RESOURCES)?
                } else {
                    0
                },
                reserved: 0,
            });
        }
        fence(Ordering::SeqCst);
        Ok(Self {
            device,
            request,
            trace_id,
            trace: None,
            allocations: vec![code_allocation, tma, samples],
            code_address: code_info.device_address,
            tma_address: tma_info.device_address,
            xcc,
            callback,
            callback_data,
            threshold: buffer_size,
            pending: VecDeque::new(),
            lost: 0,
            active: false,
            callbacks_quiesced: false,
            trap_may_be_installed: false,
        })
    }

    #[allow(unsafe_code)]
    /// Sets handler for this resources.
    fn set_handler(&self, code: u64, tma: u64) -> Result<(), Status> {
        // SAFETY: `self.allocations` retains code and TMA through successful
        // removal and the runtime retains retired allocations through teardown.
        unsafe {
            self.device
                .gpu()
                .and_then(|gpu| gpu.set_trap_handler(code, tma))
        }
        .map_err(sampling_error)
    }

    /// Registers the trap handler and sampling trace with the GPU.
    fn activate(&mut self) -> Status {
        let trace = self.device.gpu().and_then(|gpu| match self.trace_id {
            Some(id) => gpu.adopt_pc_sampling(self.request, id),
            None => gpu.create_pc_sampling(self.request),
        });
        let trace = match trace {
            Ok(trace) => trace,
            Err(error) => return sampling_error(error),
        };
        self.trace = Some(trace);
        // An ioctl failure can leave the handler installed. Keep its backing
        // alive rather than clearing a slot that another owner may hold.
        self.trap_may_be_installed = true;
        if let Err(status) = self.set_handler(self.code_address, self.tma_address) {
            if self.trace_id.is_some() {
                // A failed create-from-ID must leave the external trace with
                // its original owner, even though handler state is uncertain.
                if let Some(mut trace) = self.trace.take() {
                    if trace.relinquish_adopted().is_err() {
                        std::mem::forget(trace);
                    }
                }
            }
            return status;
        }
        SUCCESS
    }

    /// Starts this session.
    fn start(&mut self) -> Status {
        let Some(trace) = self.trace.as_mut() else {
            return INVALID_RUNTIME_STATE;
        };
        if self.active {
            return SUCCESS;
        }
        // A failed start can still leave the native trace active. Continue
        // draining until a successful stop proves that it is inactive.
        self.active = true;
        trace.start().map_or_else(sampling_error, |()| SUCCESS)
    }

    /// Stops this session.
    fn stop(&mut self) -> Status {
        let Some(trace) = self.trace.as_mut() else {
            return INVALID_RUNTIME_STATE;
        };
        if let Err(error) = trace.stop() {
            return sampling_error(error);
        }
        self.active = false;
        let status = self.drain_remaining();
        if status == SUCCESS {
            self.deliver(true);
        }
        status
    }

    /// Flushes sampled records and delivers any pending callback batch.
    fn flush(&mut self) -> Status {
        let status = self.drain_remaining();
        if status == SUCCESS {
            self.deliver(true);
        }
        status
    }

    /// Stops sampling and retires the native trace and handler.
    fn destroy(&mut self) -> Status {
        if !self.callbacks_quiesced {
            let status = self.stop();
            if status != SUCCESS {
                return status;
            }
        }
        if self.trap_may_be_installed {
            if let Err(status) = self.set_handler(0, 0) {
                return status;
            }
            self.trap_may_be_installed = false;
        }
        let Some(trace) = self.trace.as_mut() else {
            return INVALID_RUNTIME_STATE;
        };
        if let Err(error) = trace.destroy() {
            return sampling_error(error);
        }
        self.trace = None;
        SUCCESS
    }

    /// Drains all.
    fn drain_all(&mut self) -> Status {
        for index in 0..self.xcc.len() {
            let status = self.drain_xcc(index);
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    }

    /// Drains remaining.
    fn drain_remaining(&mut self) -> Status {
        // An earlier timed-out swap can leave the other half selected.
        // Completing two rounds also drains writes that arrived during the
        // first round without leaving the selector on a different half.
        let status = self.drain_all();
        if status != SUCCESS {
            return status;
        }
        self.drain_all()
    }

    #[allow(unsafe_code)]
    /// Drains XCC.
    fn drain_xcc(&mut self, index: usize) -> Status {
        let xcc = &mut self.xcc[index];
        let (old, count) = if let Some(swap) = xcc.pending_swap {
            swap
        } else {
            let old = xcc.selected;
            let next = old ^ 1;
            let ticket = xcc
                .metadata()
                .ticket
                .swap((next as u64) << 63, Ordering::AcqRel);
            xcc.selected = next;
            let observed = ticket & (u64::MAX >> 1);
            let count = observed.min(xcc.capacity as u64) as usize;
            self.lost = self
                .lost
                .saturating_add(observed.saturating_sub(xcc.capacity as u64) as usize);
            xcc.pending_swap = Some((old, count));
            (old, count)
        };
        let written = if old == 0 {
            &xcc.metadata().written0
        } else {
            &xcc.metadata().written1
        };
        let deadline = Instant::now() + WRITE_TIMEOUT;
        while written.load(Ordering::Acquire) < count as u32 {
            if Instant::now() >= deadline {
                return ERROR;
            }
            std::hint::spin_loop();
        }
        let bytes = count * SAMPLE_BYTES;
        if bytes != 0 {
            let limit = self.threshold.saturating_mul(2);
            if self.pending.len().saturating_add(bytes) > limit
                || self.pending.try_reserve(bytes).is_err()
            {
                self.lost = self.lost.saturating_add(count);
            } else {
                // SAFETY: The GPU completion counter follows payload stores,
                // and the ticket swap prevents new writes to this half.
                let samples =
                    unsafe { std::slice::from_raw_parts(xcc.samples(old) as *const u8, bytes) };
                self.pending.extend(samples.iter().copied());
            }
        }
        written.store(0, Ordering::Release);
        xcc.signal(old).store(1, Ordering::Release);
        xcc.pending_swap = None;
        SUCCESS
    }

    /// Delivers accumulated samples when the threshold or force flag is met.
    fn deliver(&mut self, force: bool) {
        if self.callbacks_quiesced {
            return;
        }
        deliver_pending(
            &mut self.pending,
            &mut self.lost,
            self.threshold,
            self.callback,
            self.callback_data.as_ptr(),
            force,
        );
    }

    /// Moves native allocations into a retired owner for safe later release.
    fn retire(&mut self) -> RetiredAssets {
        RetiredAssets {
            _allocations: std::mem::take(&mut self.allocations),
        }
    }
}

impl Drop for Resources {
    /// Retains uncertain trap backing if native handler removal could fail.
    fn drop(&mut self) {
        if self.trap_may_be_installed {
            // Native failure may have published these addresses to a GPU.
            std::mem::forget(std::mem::take(&mut self.allocations));
        }
    }
}

/// Sample bytes available only while the matching callback is active.
struct CopyContext {
    /// Unique token passed to the C client instead of this context's address.
    token: NonZeroUsize,
    /// Start of the batch still readable during the callback.
    bytes: *const u8,
    /// Total byte extent available to the callback.
    size: usize,
    /// Bytes already copied by the callback.
    consumed: usize,
    /// The client requested that delivery stop after this callback.
    aborted: bool,
}

/// Restores the previous thread-local copy context when a callback returns.
struct CopyScope {
    /// Unique callback identity that is never reused.
    token: NonZeroUsize,
    /// Optional previous retained by this copy scope.
    previous: Option<CopyContext>,
}

impl CopyScope {
    /// Installs a temporary callback context for a copy request.
    fn enter(bytes: *const u8, size: usize) -> Option<Self> {
        let token = NEXT_COPY_TOKEN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .ok()
            .and_then(NonZeroUsize::new)?;
        let previous = ACTIVE_COPY
            .try_with(|slot| {
                slot.replace(Some(CopyContext {
                    token,
                    bytes,
                    size,
                    consumed: 0,
                    aborted: false,
                }))
            })
            .ok()?;
        Some(Self { token, previous })
    }

    /// Returns a unique opaque token for the active copy callback.
    fn opaque_token(&self) -> *mut c_void {
        std::ptr::without_provenance_mut(self.token.get())
    }

    /// Returns copy progress and whether the callback completed.
    fn progress(&self) -> Option<(usize, bool)> {
        ACTIVE_COPY
            .try_with(|slot| {
                let active = slot.borrow();
                active
                    .as_ref()
                    .filter(|context| context.token == self.token)
                    .map(|context| (context.consumed, context.aborted))
            })
            .ok()
            .flatten()
    }
}

impl Drop for CopyScope {
    /// Restores the previous thread-local PC sampling copy context.
    fn drop(&mut self) {
        let _ = ACTIVE_COPY.try_with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

#[allow(unsafe_code)]
/// Delivers pending sample bytes through the caller’s copy callback.
fn deliver_pending(
    pending: &mut VecDeque<u8>,
    lost: &mut usize,
    threshold: usize,
    callback: unsafe extern "C" fn(
        *mut c_void,
        usize,
        usize,
        PcSamplingDataCopyCallback,
        *mut c_void,
    ),
    callback_data: *mut c_void,
    force: bool,
) {
    while pending.len() >= threshold || (force && !pending.is_empty()) {
        let available = pending.len().min(threshold);
        let bytes = pending.make_contiguous().as_ptr();
        let Some(copy) = CopyScope::enter(bytes, available) else {
            break;
        };
        let lost_count = std::mem::take(lost);
        let _callback_scope = CallbackScope::enter();
        // SAFETY: Pending bytes and the thread-local copy context remain live
        // through this synchronous callback. The token is invalidated below.
        unsafe {
            callback(
                callback_data,
                available,
                lost_count,
                Some(copy_samples),
                copy.opaque_token(),
            );
        }
        let progress = copy.progress();
        drop(copy);
        let Some((consumed, aborted)) = progress else {
            break;
        };
        pending.drain(..consumed);
        if consumed == 0 || aborted {
            break;
        }
    }
}

#[allow(unsafe_code)]
/// Copies samples after validating its source and destination.
unsafe extern "C" fn copy_samples(
    context: *mut c_void,
    size: usize,
    destination: *mut c_void,
) -> Status {
    if context.is_null() || (size != 0 && destination.is_null()) {
        return INVALID_ARGUMENT;
    }
    ACTIVE_COPY
        .try_with(|slot| {
            let Ok(mut active) = slot.try_borrow_mut() else {
                return INVALID_ARGUMENT;
            };
            let Some(copy) = active.as_mut() else {
                return INVALID_ARGUMENT;
            };
            if copy.token.get() != context as usize || size > copy.size - copy.consumed {
                return INVALID_ARGUMENT;
            }
            if size == 0 {
                copy.aborted = true;
                return SUCCESS;
            }
            if copy.aborted {
                return INVALID_ARGUMENT;
            }
            // SAFETY: The client supplies `size` writable bytes, and the sample
            // bytes remain initialized until the active callback returns.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    copy.bytes.add(copy.consumed),
                    destination.cast::<u8>(),
                    size,
                );
            }
            copy.consumed += size;
            SUCCESS
        })
        .unwrap_or(INVALID_ARGUMENT)
}

/// The callback worker and public control calls share one serialized state.
pub(crate) struct SessionHandle {
    /// HSA agent exposed by this session.
    pub(super) agent: HsaAgent,
    /// Native trace and delivery state shared with the worker.
    resources: Mutex<Resources>,
    /// Stop request observed by the callback worker.
    stop_worker: AtomicBool,
    /// Worker joined before final resource release.
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl SessionHandle {
    /// Creates a synchronized PC sampling session owner.
    pub(super) fn create(
        resources: Resources,
        agent: HsaAgent,
        quarantine: &crate::runtime::GenerationHealth,
    ) -> Result<Arc<Self>, Status> {
        let session = Arc::new(Self {
            agent,
            resources: Mutex::new(resources),
            stop_worker: AtomicBool::new(false),
            worker: Mutex::new(None),
        });
        let weak = Arc::downgrade(&session);
        let worker = thread::Builder::new()
            .name("rocddi-pc-sampling".into())
            .spawn(move || {
                loop {
                    thread::sleep(POLL_INTERVAL);
                    let Some(session) = weak.upgrade() else {
                        break;
                    };
                    if session.stop_worker.load(Ordering::Acquire) {
                        break;
                    }
                    let Ok(mut resources) = session.resources.lock() else {
                        break;
                    };
                    if resources.active && resources.drain_all() == SUCCESS {
                        resources.deliver(false);
                    }
                }
            });
        let Ok(worker) = worker else {
            return Err(OUT_OF_RESOURCES);
        };
        *session
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(worker);
        let (status, trap_may_be_installed) = match session.resources.lock() {
            Ok(mut resources) => {
                let status = resources.activate();
                (status, resources.trap_may_be_installed)
            }
            Err(_) => (ERROR, true),
        };
        if status != SUCCESS {
            if trap_may_be_installed {
                quarantine.quarantine();
            }
            session.quarantine();
            return Err(status);
        }
        Ok(session)
    }

    /// Starts this session.
    pub(crate) fn start(&self) -> Status {
        if self.stop_worker.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
        self.resources
            .lock()
            .map_or(ERROR, |mut resources| resources.start())
    }

    /// Stops this session.
    pub(crate) fn stop(&self) -> Status {
        if self.stop_worker.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
        self.resources
            .lock()
            .map_or(ERROR, |mut resources| resources.stop())
    }

    /// Flushes pending samples through the synchronized session.
    pub(crate) fn flush(&self) -> Status {
        if self.stop_worker.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
        self.resources
            .lock()
            .map_or(ERROR, |mut resources| resources.flush())
    }

    /// Stops callback delivery while native resources and public handles remain
    /// available to a legacy tool's `OnUnload` callback.
    pub(crate) fn quiesce_for_tool_unload(&self) -> Status {
        self.stop_worker.store(true, Ordering::Release);
        let worker = match self.worker.lock() {
            Ok(mut worker) => worker.take(),
            Err(_) => return ERROR,
        };
        if let Some(worker) = worker {
            if worker.join().is_err() {
                return ERROR;
            }
        }
        let Ok(mut resources) = self.resources.lock() else {
            return ERROR;
        };
        let status = resources.stop();
        if status == SUCCESS {
            resources.callbacks_quiesced = true;
        }
        status
    }

    /// Destroys the session and returns assets requiring deferred release.
    pub(crate) fn destroy(&self) -> Result<RetiredAssets, Status> {
        let mut resources = self.resources.lock().map_err(|_| ERROR)?;
        if resources.trace.is_none() && resources.allocations.is_empty() {
            return Ok(RetiredAssets {
                _allocations: Vec::new(),
            });
        }
        let status = resources.destroy();
        if status != SUCCESS {
            return Err(status);
        }
        self.stop_worker.store(true, Ordering::Release);
        drop(resources);
        let worker = self.worker.lock().map_err(|_| ERROR)?.take();
        if let Some(worker) = worker {
            worker.join().map_err(|_| ERROR)?;
        }
        let mut resources = self.resources.lock().map_err(|_| ERROR)?;
        Ok(resources.retire())
    }

    /// Marks resources as retained because native teardown is uncertain.
    pub(crate) fn quarantine(&self) {
        self.stop_worker.store(true, Ordering::Release);
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
}
