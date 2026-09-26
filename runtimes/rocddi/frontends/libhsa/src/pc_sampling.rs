//! HSA PC-sampling sessions and GFX12 trap-buffer delivery.
//!
//! A session owns the native sampling registration together with trap code,
//! double-buffered sample storage, completion signals, and the worker that
//! delivers records to the application callback. Start, stop, flush, and
//! destroy preserve this ownership order so no worker observes released GPU or
//! host storage. Device timestamps are translated from one correlated clock
//! sample before records leave the runtime.
//!
//! The embedded trap program is qualified only for the target described by
//! `trap_handler_gfx12`; unsupported devices are rejected before resources are
//! published.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::{offset_of, size_of, size_of_val};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rocddi::device::Device;
use rocddi::gpu::profiling::{
    ClockCounters, PcSampling, PcSamplingConfiguration, PcSamplingMethod, PcSamplingUnits,
};
use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};

use crate::ffi::*;
use crate::runtime::{boundary, lock, map_error};
use crate::signal::AmdSignal;
use crate::trap_handler_gfx12::GFX12_TRAP_HANDLER;

const METHOD_HOSTTRAP_V1: u32 = 0;
const METHOD_STOCHASTIC_V1: u32 = 1;
const UNITS_MICROSECONDS: u32 = 0;
const UNITS_CLOCK_CYCLES: u32 = 1;
const UNITS_INSTRUCTIONS: u32 = 2;
const SAMPLE_BYTES: usize = 64;
const SIGNAL_BYTES: usize = size_of::<AmdSignal>();
const PAGE_BYTES: usize = 4096;
const BUFFER_SELECTOR: u64 = 1 << 63;
const SAMPLE_COUNT_MASK: u64 = BUFFER_SELECTOR - 1;
const POLL_INTERVAL: Duration = Duration::from_micros(10);

#[repr(C, align(64))]
/// Device-visible producer state for the two alternating sample buffers.
///
/// The trap handler updates the selector and completion counts. Host code uses
/// acquire/release operations before copying bytes or returning a buffer to the
/// producer.
struct SamplingHeader {
    buffer_write_value: AtomicU64,
    buffer_size: u32,
    reserved0: u32,
    buffer_written_value0: AtomicU32,
    buffer_watermark0: u32,
    done_signal0: HsaSignal,
    buffer_written_value1: AtomicU32,
    buffer_watermark1: u32,
    done_signal1: HsaSignal,
    reserved1: [u8; 16],
}

impl SamplingHeader {
    fn new(samples_per_buffer: u32, done_signal0: u64, done_signal1: u64) -> Self {
        let watermark = samples_per_buffer
            .saturating_mul(4)
            .checked_div(5)
            .unwrap_or(0)
            .max(1);
        Self {
            buffer_write_value: AtomicU64::new(0),
            buffer_size: samples_per_buffer,
            reserved0: 0,
            buffer_written_value0: AtomicU32::new(0),
            buffer_watermark0: watermark,
            done_signal0: HsaSignal {
                handle: done_signal0,
            },
            buffer_written_value1: AtomicU32::new(0),
            buffer_watermark1: watermark,
            done_signal1: HsaSignal {
                handle: done_signal1,
            },
            reserved1: [0; 16],
        }
    }
}

#[repr(C)]
/// Device-visible addresses passed to the installed trap handler.
struct TrapMemoryArgument {
    hosttrap_buffers: u64,
    stochastic_buffers: u64,
    per_xcc_stride: u64,
    reserved: u64,
}

const _: () = {
    assert!(size_of::<SamplingHeader>() == 64);
    assert!(offset_of!(SamplingHeader, buffer_write_value) == 0x00);
    assert!(offset_of!(SamplingHeader, buffer_size) == 0x08);
    assert!(offset_of!(SamplingHeader, buffer_written_value0) == 0x10);
    assert!(offset_of!(SamplingHeader, done_signal0) == 0x18);
    assert!(offset_of!(SamplingHeader, buffer_written_value1) == 0x20);
    assert!(offset_of!(SamplingHeader, done_signal1) == 0x28);
    assert!(size_of::<TrapMemoryArgument>() == 32);
};

/// Native device state and allocations that must outlive trap execution.
///
/// Unbinding the trap handler precedes release of every allocation. If unbind
/// fails, this owner remains intact so destruction can be retried safely.
struct TrapResources {
    device: Device,
    _code: Allocation,
    _data: Allocation,
    _signals: Allocation,
    _tma: Allocation,
    bound: bool,
}

impl TrapResources {
    fn create(
        device: &Device,
        method: PcSamplingMethod,
        buffer_size: usize,
        callback: unsafe extern "C" fn(
            *mut c_void,
            usize,
            usize,
            PcSamplingDataCopyCallback,
            *mut c_void,
        ),
        callback_data: *mut c_void,
    ) -> Result<(Self, Arc<SamplingState>), Status> {
        let Some(gpu) = device.endpoint().gpu() else {
            return Err(OUT_OF_RESOURCES);
        };
        if gpu.gfx_major != 12 || gpu.gfx_minor != 0 || gpu.xcc_count != 1 {
            return Err(NOT_SUPPORTED);
        }

        let trap_buffer_bytes = buffer_size / 2;
        let samples_per_buffer =
            u32::try_from(trap_buffer_bytes / SAMPLE_BYTES).map_err(|_| OUT_OF_RESOURCES)?;
        let data_bytes = round_to_page(
            size_of::<SamplingHeader>()
                .checked_add(buffer_size)
                .ok_or(OUT_OF_RESOURCES)?,
        )?;
        let pending_capacity = buffer_size.checked_mul(2).ok_or(OUT_OF_RESOURCES)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(pending_capacity)
            .map_err(|_| OUT_OF_RESOURCES)?;

        let code = device
            .allocate(
                MemoryKind::System,
                PAGE_BYTES as u64,
                PAGE_BYTES as u64,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .map_err(map_error)?;
        let code_info = code.info();
        let code_host = code_info.host_address.ok_or(OUT_OF_RESOURCES)?;
        // SAFETY: The allocation is writable and one page is larger than the
        // complete preassembled trap program.
        unsafe {
            ptr::write_bytes(code_host as *mut u8, 0, PAGE_BYTES);
            ptr::copy_nonoverlapping(
                GFX12_TRAP_HANDLER.as_ptr().cast::<u8>(),
                code_host as *mut u8,
                size_of_val(&GFX12_TRAP_HANDLER),
            );
        }

        let signals = device
            .allocate(
                MemoryKind::System,
                PAGE_BYTES as u64,
                PAGE_BYTES as u64,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(map_error)?;
        let signal_info = signals.info();
        let signal_hosts = [
            signal_info.host_address.ok_or(OUT_OF_RESOURCES)?,
            signal_info
                .host_address
                .ok_or(OUT_OF_RESOURCES)?
                .checked_add(SIGNAL_BYTES)
                .ok_or(OUT_OF_RESOURCES)?,
        ];
        let signal_devices = [
            signal_info.device_address,
            signal_info
                .device_address
                .checked_add(SIGNAL_BYTES as u64)
                .ok_or(OUT_OF_RESOURCES)?,
        ];
        // SAFETY: Both aligned records fit in the dedicated page and remain
        // live for the complete sampling-session lifetime.
        unsafe {
            (signal_hosts[0] as *mut AmdSignal).write(AmdSignal::user(1));
            (signal_hosts[1] as *mut AmdSignal).write(AmdSignal::user(1));
        }

        let data = device
            .allocate(
                MemoryKind::System,
                data_bytes as u64,
                PAGE_BYTES as u64,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(map_error)?;
        let data_info = data.info();
        let data_host = data_info.host_address.ok_or(OUT_OF_RESOURCES)?;
        // SAFETY: The dedicated allocation is writable for its full rounded
        // extent and the header occupies its aligned first cache line.
        unsafe {
            ptr::write_bytes(data_host as *mut u8, 0, data_bytes);
            (data_host as *mut SamplingHeader).write(SamplingHeader::new(
                samples_per_buffer,
                signal_devices[0],
                signal_devices[1],
            ));
        }

        let tma = device
            .allocate(
                MemoryKind::System,
                PAGE_BYTES as u64,
                PAGE_BYTES as u64,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .map_err(map_error)?;
        let tma_info = tma.info();
        let tma_host = tma_info.host_address.ok_or(OUT_OF_RESOURCES)?;
        let (hosttrap_buffers, stochastic_buffers) = match method {
            PcSamplingMethod::HostTrapV1 => (data_info.device_address, 0),
            PcSamplingMethod::StochasticV1 => (0, data_info.device_address),
        };
        // SAFETY: The TMA allocation is writable and large enough for this
        // fixed 32-byte record.
        unsafe {
            ptr::write_bytes(tma_host as *mut u8, 0, PAGE_BYTES);
            (tma_host as *mut TrapMemoryArgument).write(TrapMemoryArgument {
                hosttrap_buffers,
                stochastic_buffers,
                per_xcc_stride: 0,
                reserved: 0,
            });
        }

        let gpu_device = device.gpu().map_err(map_error)?;
        let clock = gpu_device.clock_counters().map_err(map_error)?;
        gpu_device
            .set_trap_handler(code_info.device_address, tma_info.device_address)
            .map_err(map_error)?;

        let state = Arc::new(SamplingState {
            header_host: data_host,
            sample_hosts: [
                data_host + size_of::<SamplingHeader>(),
                data_host + size_of::<SamplingHeader>() + trap_buffer_bytes,
            ],
            signal_hosts,
            samples_per_buffer,
            delivery_size: buffer_size,
            callback,
            callback_data: callback_data as usize,
            clock,
            active: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            worker_error: AtomicU32::new(SUCCESS),
            delivery: Mutex::new(Delivery {
                pending,
                lost_samples: 0,
            }),
        });
        Ok((
            Self {
                device: device.clone(),
                _code: code,
                _data: data,
                _signals: signals,
                _tma: tma,
                bound: true,
            },
            state,
        ))
    }

    fn unbind(&mut self) -> Status {
        if !self.bound {
            return SUCCESS;
        }
        match self.device.gpu().and_then(|gpu| gpu.set_trap_handler(0, 0)) {
            Ok(()) => {
                self.bound = false;
                SUCCESS
            }
            Err(error) => map_error(error),
        }
    }
}

/// Host-side bytes waiting for callback consumption and accumulated loss.
struct Delivery {
    pending: Vec<u8>,
    lost_samples: usize,
}

/// State shared by the API thread and the sampling-delivery worker.
struct SamplingState {
    header_host: usize,
    sample_hosts: [usize; 2],
    signal_hosts: [usize; 2],
    samples_per_buffer: u32,
    delivery_size: usize,
    callback:
        unsafe extern "C" fn(*mut c_void, usize, usize, PcSamplingDataCopyCallback, *mut c_void),
    callback_data: usize,
    clock: ClockCounters,
    active: AtomicBool,
    exit: AtomicBool,
    worker_error: AtomicU32,
    delivery: Mutex<Delivery>,
}

impl SamplingState {
    fn prepare_start(&self) {
        self.exit.store(false, Ordering::Release);
        self.worker_error.store(SUCCESS, Ordering::Release);
        for address in self.signal_hosts {
            signal_at(address).value.store(1, Ordering::Release);
        }
        self.active.store(true, Ordering::Release);
    }

    fn request_exit(&self) {
        self.active.store(false, Ordering::Release);
        self.exit.store(true, Ordering::Release);
    }

    fn drain_and_deliver(&self, flush: bool) -> Status {
        let mut delivery = match self.delivery.lock() {
            Ok(delivery) => delivery,
            Err(_) => return ERROR,
        };
        let status = self.drain_device_buffer(&mut delivery);
        if status != SUCCESS {
            return status;
        }
        self.deliver(&mut delivery, flush);
        SUCCESS
    }

    fn drain_device_buffer(&self, delivery: &mut Delivery) -> Status {
        // SAFETY: TrapResources retains the initialized header allocation while
        // this state and every worker using it are live.
        let header = unsafe { &*(self.header_host as *const SamplingHeader) };
        let selected =
            usize::try_from(header.buffer_write_value.load(Ordering::Acquire) >> 63).unwrap_or(0);
        let old = header
            .buffer_write_value
            .swap(((selected ^ 1) as u64) << 63, Ordering::AcqRel);
        let buffer = usize::try_from(old >> 63).unwrap_or(0);
        let requested = old & SAMPLE_COUNT_MASK;
        let capacity = u64::from(self.samples_per_buffer);
        let sample_count = requested.min(capacity);
        delivery.lost_samples = delivery.lost_samples.saturating_add(
            usize::try_from(requested.saturating_sub(sample_count)).unwrap_or(usize::MAX),
        );

        let written = if buffer == 0 {
            &header.buffer_written_value0
        } else {
            &header.buffer_written_value1
        };
        let expected = u32::try_from(sample_count).unwrap_or(self.samples_per_buffer);
        let mut completed = written.load(Ordering::Acquire);
        while completed < expected && self.active.load(Ordering::Acquire) {
            std::hint::spin_loop();
            completed = written.load(Ordering::Acquire);
        }
        let completed = completed.min(expected);
        delivery.lost_samples = delivery.lost_samples.saturating_add(
            usize::try_from(expected.saturating_sub(completed)).unwrap_or(usize::MAX),
        );

        let available_samples = delivery
            .pending
            .capacity()
            .saturating_sub(delivery.pending.len())
            / SAMPLE_BYTES;
        let accepted_samples = usize::try_from(completed)
            .unwrap_or(usize::MAX)
            .min(available_samples);
        delivery.lost_samples = delivery.lost_samples.saturating_add(
            usize::try_from(completed)
                .unwrap_or(usize::MAX)
                .saturating_sub(accepted_samples),
        );
        let accepted_bytes = accepted_samples * SAMPLE_BYTES;
        if accepted_bytes != 0 {
            let old_length = delivery.pending.len();
            // SAFETY: Capacity was reserved at construction, accepted_bytes is
            // bounded by remaining capacity, and the source buffer stays live
            // and quiescent until its written counter is reset below.
            unsafe {
                delivery.pending.set_len(old_length + accepted_bytes);
                ptr::copy_nonoverlapping(
                    self.sample_hosts[buffer] as *const u8,
                    delivery.pending.as_mut_ptr().add(old_length),
                    accepted_bytes,
                );
            }
            translate_timestamps(&mut delivery.pending[old_length..], self.clock);
        }
        written.store(0, Ordering::Release);
        signal_at(self.signal_hosts[buffer])
            .value
            .store(1, Ordering::Release);
        SUCCESS
    }

    fn deliver(&self, delivery: &mut Delivery, flush: bool) {
        loop {
            let available = delivery.pending.len();
            if available == 0 || (!flush && available < self.delivery_size) {
                break;
            }
            let offered = available.min(self.delivery_size);
            let lost = std::mem::take(&mut delivery.lost_samples);
            let copied = invoke_ready_callback(
                self.callback,
                self.callback_data,
                &delivery.pending[..offered],
                lost,
            );
            if copied != 0 {
                delivery.pending.drain(..copied);
            }
            if copied < offered {
                break;
            }
        }
    }

    fn pending_error(&self) -> Status {
        self.worker_error.load(Ordering::Acquire)
    }
}

fn signal_at(address: usize) -> &'static AmdSignal {
    // SAFETY: Every address is created from a live signal allocation retained
    // by TrapResources until all workers have joined.
    unsafe { &*(address as *const AmdSignal) }
}

fn worker_loop(state: &SamplingState) {
    while !state.exit.load(Ordering::Acquire) {
        let ready = state
            .signal_hosts
            .iter()
            .any(|address| signal_at(*address).value.load(Ordering::Acquire) < 1);
        if ready {
            let status = state.drain_and_deliver(false);
            if status != SUCCESS {
                state.worker_error.store(status, Ordering::Release);
                break;
            }
        } else {
            thread::sleep(POLL_INTERVAL);
        }
    }
}

/// Public sampling-session owner coordinating native and worker lifetimes.
pub(crate) struct PcSamplingSession {
    gpu_index: usize,
    native: Option<PcSampling>,
    resources: Option<TrapResources>,
    state: Arc<SamplingState>,
    worker: Option<JoinHandle<()>>,
    active: bool,
}

impl PcSamplingSession {
    fn start(&mut self) -> Status {
        if self.active {
            return SUCCESS;
        }
        let Some(native) = self.native.as_mut() else {
            return INVALID_ARGUMENT;
        };
        if self.resources.is_none() {
            return INVALID_ARGUMENT;
        }
        self.state.prepare_start();
        let worker_state = self.state.clone();
        let Ok(worker) = thread::Builder::new()
            .name("rocddi-pc-sampling".to_string())
            .spawn(move || worker_loop(&worker_state))
        else {
            self.state.request_exit();
            return OUT_OF_RESOURCES;
        };
        if let Err(error) = native.start() {
            self.state.request_exit();
            let _ = worker.join();
            return map_pc_sampling_error(error);
        }
        self.worker = Some(worker);
        self.active = true;
        SUCCESS
    }

    fn stop(&mut self) -> Status {
        if !self.active {
            return SUCCESS;
        }
        let Some(native) = self.native.as_mut() else {
            return INVALID_ARGUMENT;
        };
        if let Err(error) = native.stop() {
            return map_pc_sampling_error(error);
        }
        self.active = false;
        self.state.request_exit();
        let worker_status = self
            .worker
            .take()
            .map_or(SUCCESS, |worker| worker.join().map_or(ERROR, |()| SUCCESS));
        let flush_status = self.state.drain_and_deliver(true);
        first_error([worker_status, flush_status, self.state.pending_error()])
    }

    fn flush(&self) -> Status {
        let status = self.state.drain_and_deliver(true);
        first_error([status, self.state.pending_error()])
    }

    pub(crate) fn destroy(&mut self) -> Status {
        let stop_status = self.stop();
        if stop_status != SUCCESS {
            return stop_status;
        }
        if let Some(native) = self.native.as_mut() {
            if let Err(error) = native.destroy() {
                return map_pc_sampling_error(error);
            }
        }
        if let Some(resources) = self.resources.as_mut() {
            let status = resources.unbind();
            if status != SUCCESS {
                return status;
            }
        }
        self.resources = None;
        self.native = None;
        SUCCESS
    }
}

fn first_error<const N: usize>(statuses: [Status; N]) -> Status {
    statuses
        .into_iter()
        .find(|status| *status != SUCCESS)
        .unwrap_or(SUCCESS)
}

fn round_to_page(size: usize) -> Result<usize, Status> {
    size.checked_add(PAGE_BYTES - 1)
        .map(|size| size & !(PAGE_BYTES - 1))
        .ok_or(OUT_OF_RESOURCES)
}

fn translate_tick(counters: ClockCounters, tick: u64) -> u64 {
    if counters.system_frequency == 0 {
        return tick;
    }
    let scaled = |delta: u64| {
        u64::try_from(u128::from(delta) * u128::from(counters.system_frequency) / 100_000_000_u128)
            .unwrap_or(u64::MAX)
    };
    if tick >= counters.gpu {
        counters.system.wrapping_add(scaled(tick - counters.gpu))
    } else {
        counters.system.wrapping_sub(scaled(counters.gpu - tick))
    }
}

fn translate_timestamps(samples: &mut [u8], counters: ClockCounters) {
    for sample in samples.chunks_exact_mut(SAMPLE_BYTES) {
        // SAFETY: Every complete sample has an eight-byte timestamp field at
        // byte offset 48. Unaligned access avoids imposing alignment on Vec.
        let timestamp = unsafe { sample.as_ptr().add(48).cast::<u64>().read_unaligned() };
        // SAFETY: The same complete sample provides eight writable bytes.
        unsafe {
            sample
                .as_mut_ptr()
                .add(48)
                .cast::<u64>()
                .write_unaligned(translate_tick(counters, timestamp));
        }
    }
}

#[derive(Clone, Copy)]
struct CopyContext {
    token: usize,
    source: usize,
    length: usize,
    copied: usize,
    stopped: bool,
}

thread_local! {
    static COPY_CONTEXT: RefCell<Option<CopyContext>> = const { RefCell::new(None) };
}

static NEXT_COPY_TOKEN: AtomicUsize = AtomicUsize::new(1);

struct CopyScope {
    token: usize,
    saved: Option<CopyContext>,
    active: bool,
}

impl CopyScope {
    fn enter(source: &[u8]) -> Self {
        let mut token = NEXT_COPY_TOKEN.fetch_add(1, Ordering::Relaxed);
        if token == 0 {
            token = NEXT_COPY_TOKEN.fetch_add(1, Ordering::Relaxed);
        }
        let context = CopyContext {
            token,
            source: source.as_ptr() as usize,
            length: source.len(),
            copied: 0,
            stopped: false,
        };
        let saved = COPY_CONTEXT.with(|slot| slot.replace(Some(context)));
        Self {
            token,
            saved,
            active: true,
        }
    }

    fn callback_data(&self) -> *mut c_void {
        self.token as *mut c_void
    }

    fn finish(mut self) -> usize {
        let current = COPY_CONTEXT.with(|slot| slot.replace(self.saved.take()));
        self.active = false;
        current
            .filter(|context| context.token == self.token)
            .map_or(0, |context| context.copied)
    }
}

impl Drop for CopyScope {
    fn drop(&mut self) {
        if self.active {
            COPY_CONTEXT.with(|slot| {
                slot.replace(self.saved.take());
            });
        }
    }
}

unsafe extern "C" fn data_copy_callback(
    callback_data: *mut c_void,
    data_size: usize,
    destination: *mut c_void,
) -> Status {
    boundary(|| {
        COPY_CONTEXT.with(|slot| {
            let mut slot = match slot.try_borrow_mut() {
                Ok(slot) => slot,
                Err(_) => return ERROR,
            };
            let Some(context) = slot.as_mut() else {
                return INVALID_ARGUMENT;
            };
            if context.token != callback_data as usize || context.stopped {
                return INVALID_ARGUMENT;
            }
            if data_size == 0 {
                context.stopped = true;
                return SUCCESS;
            }
            if destination.is_null() || data_size > context.length.saturating_sub(context.copied) {
                return INVALID_ARGUMENT;
            }
            // SAFETY: The TLS context is live only during the ready callback;
            // bounds were checked and the caller supplies writable destination
            // storage for data_size bytes.
            unsafe {
                ptr::copy_nonoverlapping(
                    (context.source as *const u8).add(context.copied),
                    destination.cast::<u8>(),
                    data_size,
                );
            }
            context.copied += data_size;
            SUCCESS
        })
    })
}

fn invoke_ready_callback(
    callback: unsafe extern "C" fn(
        *mut c_void,
        usize,
        usize,
        PcSamplingDataCopyCallback,
        *mut c_void,
    ),
    callback_data: usize,
    source: &[u8],
    lost_samples: usize,
) -> usize {
    let scope = CopyScope::enter(source);
    // SAFETY: The client supplied the callback at successful session creation.
    // The copy context remains installed on this thread until it returns.
    unsafe {
        callback(
            callback_data as *mut c_void,
            source.len(),
            lost_samples,
            Some(data_copy_callback),
            scope.callback_data(),
        );
    }
    scope.finish()
}

fn public_method(method: PcSamplingMethod) -> u32 {
    match method {
        PcSamplingMethod::HostTrapV1 => METHOD_HOSTTRAP_V1,
        PcSamplingMethod::StochasticV1 => METHOD_STOCHASTIC_V1,
    }
}

fn public_units(units: PcSamplingUnits) -> u32 {
    match units {
        PcSamplingUnits::Microseconds => UNITS_MICROSECONDS,
        PcSamplingUnits::ClockCycles => UNITS_CLOCK_CYCLES,
        PcSamplingUnits::Instructions => UNITS_INSTRUCTIONS,
    }
}

fn native_method(method: u32) -> Option<PcSamplingMethod> {
    match method {
        METHOD_HOSTTRAP_V1 => Some(PcSamplingMethod::HostTrapV1),
        METHOD_STOCHASTIC_V1 => Some(PcSamplingMethod::StochasticV1),
        _ => None,
    }
}

fn native_units(units: u32) -> Option<PcSamplingUnits> {
    match units {
        UNITS_MICROSECONDS => Some(PcSamplingUnits::Microseconds),
        UNITS_CLOCK_CYCLES => Some(PcSamplingUnits::ClockCycles),
        UNITS_INSTRUCTIONS => Some(PcSamplingUnits::Instructions),
        _ => None,
    }
}

fn public_configuration(
    configuration: PcSamplingConfiguration,
) -> Result<HsaPcSamplingConfiguration, Status> {
    Ok(HsaPcSamplingConfiguration {
        method: public_method(configuration.method),
        units: public_units(configuration.units),
        minimum_interval: usize::try_from(configuration.minimum_interval)
            .map_err(|_| OUT_OF_RESOURCES)?,
        maximum_interval: usize::try_from(configuration.maximum_interval)
            .map_err(|_| OUT_OF_RESOURCES)?,
        flags: configuration.flags,
    })
}

fn selected_configuration(
    device: &Device,
    method: u32,
    units: u32,
    interval: usize,
) -> Result<PcSamplingConfiguration, Status> {
    let method = native_method(method).ok_or(INVALID_ARGUMENT)?;
    let units = native_units(units).ok_or(INVALID_ARGUMENT)?;
    let interval = u64::try_from(interval).map_err(|_| INVALID_ARGUMENT)?;
    let configurations = device
        .gpu()
        .and_then(|gpu| gpu.pc_sampling_configurations())
        .map_err(map_pc_sampling_error)?;
    configurations
        .iter()
        .copied()
        .find(|configuration| {
            configuration.method == method
                && configuration.units == units
                && interval >= configuration.minimum_interval
                && interval <= configuration.maximum_interval
                && (configuration.flags & 1 == 0 || interval.is_power_of_two())
        })
        .ok_or(INVALID_ARGUMENT)
}

#[allow(clippy::needless_pass_by_value)]
fn map_pc_sampling_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
        rocddi::ErrorKind::Busy => RESOURCE_BUSY,
        rocddi::ErrorKind::Unsupported => NOT_SUPPORTED,
        _ => ERROR,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_pcs_iterate_configuration(
    agent: HsaAgent,
    configuration_callback: PcSamplingConfigurationCallback,
    callback_data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = configuration_callback else {
            return INVALID_ARGUMENT;
        };
        let device = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            runtime.gpus[index].device.clone()
        };
        let configurations = match device
            .gpu()
            .and_then(|gpu| gpu.pc_sampling_configurations())
        {
            Ok(configurations) => configurations,
            Err(error) => return map_pc_sampling_error(error),
        };
        for configuration in configurations.iter().copied() {
            let configuration = match public_configuration(configuration) {
                Ok(configuration) => configuration,
                Err(status) => return status,
            };
            // SAFETY: The callback is valid by the public API contract and the
            // configuration record remains live for this synchronous call.
            let status = unsafe { callback(&raw const configuration, callback_data) };
            if status == INFO_BREAK {
                return SUCCESS;
            }
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[allow(clippy::too_many_arguments)]
fn create_session(
    trace_id: Option<u32>,
    agent: HsaAgent,
    method: u32,
    units: u32,
    interval: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    client_callback_data: *mut c_void,
    output: *mut HsaPcSampling,
) -> Status {
    if output.is_null()
        || data_ready_callback.is_none()
        || buffer_size == 0
        || buffer_size % (2 * SAMPLE_BYTES) != 0
        || trace_id == Some(0)
    {
        return INVALID_ARGUMENT;
    }
    let callback = match data_ready_callback {
        Some(callback) => callback,
        None => return INVALID_ARGUMENT,
    };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_mut() else {
        return NOT_INITIALIZED;
    };
    let Some(gpu_index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    if runtime.pc_sampling_agents.contains_key(&gpu_index) {
        return RESOURCE_BUSY;
    }
    let configuration =
        match selected_configuration(&runtime.gpus[gpu_index].device, method, units, interval) {
            Ok(configuration) => configuration,
            Err(status) => return status,
        };
    if runtime.pc_sampling.try_reserve(1).is_err()
        || runtime.pc_sampling_agents.try_reserve(1).is_err()
    {
        return OUT_OF_RESOURCES;
    }
    let handle = match runtime.allocate_handle() {
        Ok(handle) => handle,
        Err(status) => return status,
    };
    let (mut resources, state) = match TrapResources::create(
        &runtime.gpus[gpu_index].device,
        configuration.method,
        buffer_size,
        callback,
        client_callback_data,
    ) {
        Ok(resources) => resources,
        Err(status) => return status,
    };
    let native = match runtime.gpus[gpu_index].device.gpu() {
        Ok(gpu) => match trace_id {
            Some(trace_id) => gpu.adopt_pc_sampling(trace_id),
            None => gpu.create_pc_sampling(configuration, interval as u64),
        },
        Err(error) => Err(error),
    };
    let native = match native {
        Ok(native) => native,
        Err(error) => {
            if resources.unbind() != SUCCESS {
                std::mem::forget(resources);
            }
            return map_pc_sampling_error(error);
        }
    };
    runtime.pc_sampling.insert(
        handle,
        Arc::new(Mutex::new(PcSamplingSession {
            gpu_index,
            native: Some(native),
            resources: Some(resources),
            state,
            worker: None,
            active: false,
        })),
    );
    runtime.pc_sampling_agents.insert(gpu_index, handle);
    // SAFETY: The caller supplied writable output storage and no output is
    // published until every session resource has been acquired.
    unsafe { output.write(HsaPcSampling { handle }) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_pcs_create(
    agent: HsaAgent,
    method: u32,
    units: u32,
    interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| {
        create_session(
            None,
            agent,
            method,
            units,
            interval,
            buffer_size,
            data_ready_callback,
            client_callback_data,
            pc_sampling,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_pcs_create_from_id(
    trace_id: u32,
    agent: HsaAgent,
    method: u32,
    units: u32,
    interval: usize,
    _latency: usize,
    buffer_size: usize,
    data_ready_callback: PcSamplingDataReadyCallback,
    client_callback_data: *mut c_void,
    pc_sampling: *mut HsaPcSampling,
) -> Status {
    boundary(|| {
        create_session(
            Some(trace_id),
            agent,
            method,
            units,
            interval,
            buffer_size,
            data_ready_callback,
            client_callback_data,
            pc_sampling,
        )
    })
}

fn find_session(handle: HsaPcSampling) -> Result<Arc<Mutex<PcSamplingSession>>, Status> {
    if handle.handle == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    runtime
        .pc_sampling
        .get(&handle.handle)
        .cloned()
        .ok_or(INVALID_ARGUMENT)
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_start(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| {
        let session = match find_session(pc_sampling) {
            Ok(session) => session,
            Err(status) => return status,
        };
        session.lock().map_or(ERROR, |mut session| session.start())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_stop(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| {
        let session = match find_session(pc_sampling) {
            Ok(session) => session,
            Err(status) => return status,
        };
        session.lock().map_or(ERROR, |mut session| session.stop())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_flush(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| {
        let session = match find_session(pc_sampling) {
            Ok(session) => session,
            Err(status) => return status,
        };
        session.lock().map_or(ERROR, |session| session.flush())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ven_amd_pcs_destroy(pc_sampling: HsaPcSampling) -> Status {
    boundary(|| {
        let session = match find_session(pc_sampling) {
            Ok(session) => session,
            Err(status) => return status,
        };
        let (status, gpu_index) = match session.lock() {
            Ok(mut session) => {
                let gpu_index = session.gpu_index;
                (session.destroy(), gpu_index)
            }
            Err(_) => return ERROR,
        };
        if status != SUCCESS {
            return status;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_mut() else {
            return NOT_INITIALIZED;
        };
        if runtime
            .pc_sampling
            .get(&pc_sampling.handle)
            .is_some_and(|stored| Arc::ptr_eq(stored, &session))
        {
            runtime.pc_sampling.remove(&pc_sampling.handle);
            if runtime.pc_sampling_agents.get(&gpu_index) == Some(&pc_sampling.handle) {
                runtime.pc_sampling_agents.remove(&gpu_index);
            }
        }
        SUCCESS
    })
}

pub(crate) fn extension_table() -> [usize; 7] {
    [
        hsa_ven_amd_pcs_iterate_configuration as *const () as usize,
        hsa_ven_amd_pcs_create as *const () as usize,
        hsa_ven_amd_pcs_create_from_id as *const () as usize,
        hsa_ven_amd_pcs_destroy as *const () as usize,
        hsa_ven_amd_pcs_start as *const () as usize,
        hsa_ven_amd_pcs_stop as *const () as usize,
        hsa_ven_amd_pcs_flush as *const () as usize,
    ]
}

pub(crate) fn destroy_sessions(sessions: HashMap<u64, Arc<Mutex<PcSamplingSession>>>) -> Status {
    let mut status = SUCCESS;
    for (_, session) in sessions {
        let result = session
            .lock()
            .map_or(ERROR, |mut session| session.destroy());
        if result != SUCCESS {
            if status == SUCCESS {
                status = result;
            }
            std::mem::forget(session);
        }
    }
    status
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    struct CallbackCapture {
        bytes: Vec<u8>,
        lost_samples: Vec<usize>,
        calls: usize,
        copy_limit: usize,
        stop_after_copy: bool,
    }

    unsafe extern "C" fn capture_callback(
        data: *mut c_void,
        size: usize,
        lost: usize,
        copy: PcSamplingDataCopyCallback,
        callback_data: *mut c_void,
    ) {
        // SAFETY: Tests pass a live CallbackCapture for the synchronous call.
        let capture = unsafe { &mut *data.cast::<CallbackCapture>() };
        capture.calls += 1;
        capture.lost_samples.push(lost);
        let copy_size = size.min(capture.copy_limit);
        let old_length = capture.bytes.len();
        capture.bytes.resize(old_length + copy_size, 0);
        let copy = copy.unwrap();
        // SAFETY: The destination was resized for copy_size bytes and the
        // callback is invoked within the ready-callback scope.
        assert_eq!(
            unsafe {
                copy(
                    callback_data,
                    copy_size,
                    capture.bytes.as_mut_ptr().add(old_length).cast(),
                )
            },
            SUCCESS
        );
        if capture.stop_after_copy && copy_size != 0 {
            // SAFETY: A zero-sized copy is valid in the active scope.
            assert_eq!(unsafe { copy(callback_data, 0, ptr::null_mut()) }, SUCCESS);
        }
    }

    #[test]
    fn copy_callback_is_scoped_and_supports_partial_copies() {
        let source = [1_u8, 2, 3, 4, 5, 6];
        let mut capture = CallbackCapture {
            bytes: Vec::new(),
            lost_samples: Vec::new(),
            calls: 0,
            copy_limit: 3,
            stop_after_copy: true,
        };
        let copied = invoke_ready_callback(
            capture_callback,
            (&raw mut capture).cast::<c_void>() as usize,
            &source,
            0,
        );
        assert_eq!(copied, 3);
        assert_eq!(capture.bytes, [1, 2, 3]);
        assert_eq!(capture.calls, 1);
        let mut destination = [0_u8; 1];
        // SAFETY: This intentionally probes a callback outside its valid scope.
        assert_eq!(
            unsafe { data_copy_callback(ptr::null_mut(), 1, destination.as_mut_ptr().cast()) },
            INVALID_ARGUMENT
        );
    }

    #[test]
    fn copy_callback_rejects_cross_thread_use() {
        let source = [7_u8];
        let scope = CopyScope::enter(&source);
        let token = scope.callback_data() as usize;
        let status = thread::spawn(move || {
            let mut destination = [0_u8; 1];
            // SAFETY: This intentionally probes the wrong-thread contract.
            unsafe { data_copy_callback(token as *mut c_void, 1, destination.as_mut_ptr().cast()) }
        })
        .join()
        .unwrap();
        assert_eq!(status, INVALID_ARGUMENT);
        assert_eq!(scope.finish(), 0);
    }

    #[test]
    fn device_drain_swaps_buffers_and_accounts_for_overflow() {
        let header = SamplingHeader::new(2, 0, 0);
        let mut samples = [0x22_u8; SAMPLE_BYTES * 4];
        let signal_a = AmdSignal::user(0);
        let signal_b = AmdSignal::user(0);
        header
            .buffer_write_value
            .store(BUFFER_SELECTOR | 3, Ordering::Release);
        header.buffer_written_value1.store(2, Ordering::Release);
        let state = SamplingState {
            header_host: (&raw const header) as usize,
            sample_hosts: [
                samples.as_mut_ptr() as usize,
                // SAFETY: The second half starts within the same live array.
                unsafe { samples.as_mut_ptr().add(SAMPLE_BYTES * 2) as usize },
            ],
            signal_hosts: [
                (&raw const signal_a) as usize,
                (&raw const signal_b) as usize,
            ],
            samples_per_buffer: 2,
            delivery_size: SAMPLE_BYTES * 2,
            callback: capture_callback,
            callback_data: 0,
            clock: ClockCounters {
                gpu: 0,
                host: 0,
                system: 0,
                system_frequency: 0,
            },
            active: AtomicBool::new(true),
            exit: AtomicBool::new(false),
            worker_error: AtomicU32::new(SUCCESS),
            delivery: Mutex::new(Delivery {
                pending: Vec::new(),
                lost_samples: 0,
            }),
        };
        let mut delivery = Delivery {
            pending: Vec::with_capacity(SAMPLE_BYTES * 4),
            lost_samples: 0,
        };

        assert_eq!(state.drain_device_buffer(&mut delivery), SUCCESS);
        assert_eq!(header.buffer_write_value.load(Ordering::Acquire), 0);
        assert_eq!(header.buffer_written_value1.load(Ordering::Acquire), 0);
        assert_eq!(signal_b.value.load(Ordering::Acquire), 1);
        assert_eq!(delivery.pending, [0x22; SAMPLE_BYTES * 2]);
        assert_eq!(delivery.lost_samples, 1);
    }

    #[test]
    fn delivery_retains_bytes_the_callback_does_not_copy() {
        let mut capture = CallbackCapture {
            bytes: Vec::new(),
            lost_samples: Vec::new(),
            calls: 0,
            copy_limit: 0,
            stop_after_copy: true,
        };
        let state = SamplingState {
            header_host: 0,
            sample_hosts: [0; 2],
            signal_hosts: [0; 2],
            samples_per_buffer: 0,
            delivery_size: 4,
            callback: capture_callback,
            callback_data: (&raw mut capture).cast::<c_void>() as usize,
            clock: ClockCounters {
                gpu: 0,
                host: 0,
                system: 0,
                system_frequency: 0,
            },
            active: AtomicBool::new(false),
            exit: AtomicBool::new(false),
            worker_error: AtomicU32::new(SUCCESS),
            delivery: Mutex::new(Delivery {
                pending: Vec::new(),
                lost_samples: 0,
            }),
        };
        let mut delivery = Delivery {
            pending: vec![1, 2, 3, 4, 5, 6],
            lost_samples: 7,
        };

        state.deliver(&mut delivery, false);
        assert_eq!(delivery.pending, [1, 2, 3, 4, 5, 6]);
        assert_eq!(capture.lost_samples, [7]);

        capture.copy_limit = usize::MAX;
        capture.stop_after_copy = false;
        state.deliver(&mut delivery, true);
        assert!(delivery.pending.is_empty());
        assert_eq!(capture.bytes, [1, 2, 3, 4, 5, 6]);
        assert_eq!(capture.lost_samples, [7, 0, 0]);
    }

    #[test]
    fn timestamp_translation_updates_every_complete_sample() {
        let mut samples = [0_u8; SAMPLE_BYTES * 2];
        // SAFETY: Both writes target complete timestamp fields.
        unsafe {
            samples
                .as_mut_ptr()
                .add(48)
                .cast::<u64>()
                .write_unaligned(110);
            samples
                .as_mut_ptr()
                .add(SAMPLE_BYTES + 48)
                .cast::<u64>()
                .write_unaligned(90);
        }
        translate_timestamps(
            &mut samples,
            ClockCounters {
                gpu: 100,
                host: 0,
                system: 1_000,
                system_frequency: 1_000_000_000,
            },
        );
        // SAFETY: Both reads target complete timestamp fields.
        assert_eq!(
            unsafe { samples.as_ptr().add(48).cast::<u64>().read_unaligned() },
            1_100
        );
        assert_eq!(
            unsafe {
                samples
                    .as_ptr()
                    .add(SAMPLE_BYTES + 48)
                    .cast::<u64>()
                    .read_unaligned()
            },
            900
        );
    }

    #[test]
    fn extension_table_contains_every_public_entry_point() {
        assert!(extension_table().into_iter().all(|entry| entry != 0));
        assert_eq!(GFX12_TRAP_HANDLER.len() * size_of::<u32>(), 1920);
    }

    #[test]
    fn entry_points_and_records_match_the_public_x86_64_abi() {
        type Iterate =
            unsafe extern "C" fn(HsaAgent, PcSamplingConfigurationCallback, *mut c_void) -> Status;
        type Create = unsafe extern "C" fn(
            HsaAgent,
            u32,
            u32,
            usize,
            usize,
            usize,
            PcSamplingDataReadyCallback,
            *mut c_void,
            *mut HsaPcSampling,
        ) -> Status;
        type CreateFromId = unsafe extern "C" fn(
            u32,
            HsaAgent,
            u32,
            u32,
            usize,
            usize,
            usize,
            PcSamplingDataReadyCallback,
            *mut c_void,
            *mut HsaPcSampling,
        ) -> Status;
        type Control = extern "C" fn(HsaPcSampling) -> Status;

        let _: Iterate = hsa_ven_amd_pcs_iterate_configuration;
        let _: Create = hsa_ven_amd_pcs_create;
        let _: CreateFromId = hsa_ven_amd_pcs_create_from_id;
        let _: Control = hsa_ven_amd_pcs_destroy;
        let _: Control = hsa_ven_amd_pcs_start;
        let _: Control = hsa_ven_amd_pcs_stop;
        let _: Control = hsa_ven_amd_pcs_flush;
        assert_eq!(size_of::<HsaPcSampling>(), 8);
        assert_eq!(size_of::<HsaPcSamplingConfiguration>(), 32);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, method), 0);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, units), 4);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, minimum_interval), 8);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, maximum_interval), 16);
        assert_eq!(offset_of!(HsaPcSamplingConfiguration, flags), 24);
    }
}
