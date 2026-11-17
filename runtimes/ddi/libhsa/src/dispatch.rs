// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private synchronous AQL dispatches used by HSA image and memory operations.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};

use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocloader::{LoadedImage, Symbol, SymbolKind};

use crate::api::queue::{core::*, index::*};
use crate::api::signal::{atomic::*, core::*, wait::*};
use crate::ffi::*;
use crate::runtime::{GenerationHealth, InFlightToken, RetainOnDrop, lock, map_error};

/// Result after the private queue has attempted explicit retirement.
pub(crate) struct DispatchResult {
    /// Operation error, or retirement error when native cleanup failed.
    pub(crate) status: Status,
    /// External operands must remain live when retirement was inconclusive.
    pub(crate) operands_may_be_live: bool,
}

/// Addresses reachable by the private queue, retained together on uncertainty.
struct Assets<D> {
    /// Allocations and symbols for the complete executable image.
    _image: LoadedImage,
    /// Host-visible kernel argument storage.
    arguments: Allocation,
    /// Stable storage passed to the native queue error callback.
    error: Box<AtomicU32>,
    /// Caller-owned image or copy dependencies.
    dependencies: D,
}

/// Owns one private queue and launches one packet at a time with 64-lane groups.
pub(crate) struct Dispatch<D> {
    /// Queue handle, null before successful creation and after retirement.
    queue: *mut HsaQueue,
    /// Completion signal owned until native queue retirement.
    completion: HsaSignal,
    /// Native dependencies use fail-closed ownership during unwinding.
    assets: Option<RetainOnDrop<Assets<D>>>,
    /// Relocated symbols for each accepted kernel.
    symbols: Vec<Symbol>,
    /// Cancellation requested by runtime shutdown.
    stop: Arc<AtomicBool>,
    /// Prevents generation teardown after uncertain cleanup.
    health: Arc<GenerationHealth>,
    /// Always released, even if assets are retained, so shutdown can progress.
    _inflight: InFlightToken,
    /// Explicit cleanup has run and must not be retried.
    finished: bool,
}

/// Records a native queue fault without calling back into the HSA runtime.
#[allow(unsafe_code)]
unsafe extern "C" fn queue_error(status: Status, _queue: *mut HsaQueue, data: *mut c_void) {
    // SAFETY: Dispatch retains this stable atomic through checked retirement.
    unsafe { &*data.cast::<AtomicU32>() }.store(status, Ordering::Release);
}

impl<D> Dispatch<D> {
    /// Loads a private executable and retains its dependencies before submission.
    #[allow(unsafe_code)]
    pub(crate) fn new(agent: HsaAgent, object: Arc<[u8]>, dependencies: D) -> Result<Self, Status> {
        let (device, page, stop, health, inflight) = {
            let guard = lock()?;
            let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
            let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
            if runtime.operation_stop().load(Ordering::Acquire) {
                return Err(INVALID_RUNTIME_STATE);
            }
            (
                runtime.gpus[index].device.clone(),
                runtime.host_page_size as u64,
                runtime.operation_stop(),
                runtime.health.clone(),
                runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?,
            )
        };
        let (image, mut symbols) =
            rocloader::load_image(&device, object, MemoryKind::System, page, |_| None)
                .map_err(crate::loader::map_loader_error)?;
        symbols.retain(|symbol| symbol.kind == SymbolKind::Kernel);
        let argument_bytes = symbols
            .iter()
            .map(|symbol| symbol.kernarg_size)
            .max()
            .ok_or(ERROR)?;
        let private_bytes = symbols
            .iter()
            .map(|symbol| symbol.private_size)
            .max()
            .ok_or(ERROR)?;
        if u64::from(argument_bytes) > page
            || symbols
                .iter()
                .any(|symbol| u64::from(symbol.kernarg_alignment) > page)
        {
            return Err(NOT_SUPPORTED);
        }
        let arguments = device
            .allocate(MemoryKind::System, page, page, DeviceAccess::READ)
            .map_err(map_error)?;
        if arguments.info().host_address.is_none() {
            return Err(NOT_SUPPORTED);
        }
        // Code publication can itself leave a native queue live. Keep the code
        // image and quarantine its generation before returning that failure.
        if let Err(failure) = device.gpu().map_err(map_error)?.invalidate_code_cache(
            image.device_base(),
            image.size(),
            &stop,
        ) {
            if failure.operands_may_be_live {
                health.quarantine();
                std::mem::forget(image);
            }
            return Err(map_error(failure.error));
        }
        let mut dispatch = Self {
            queue: std::ptr::null_mut(),
            completion: HsaSignal { handle: 0 },
            assets: Some(RetainOnDrop::new(Assets {
                _image: image,
                arguments,
                error: Box::new(AtomicU32::new(SUCCESS)),
                dependencies,
            })),
            symbols,
            stop,
            health,
            _inflight: inflight,
            finished: false,
        };
        // SAFETY: The output signal and queue handles are writable; callback
        // storage is held in assets before queue creation can expose it.
        let status =
            unsafe { hsa_signal_create(1, 0, std::ptr::null(), &raw mut dispatch.completion) };
        if status != SUCCESS {
            return Err(status);
        }
        let Some(assets) = dispatch.assets.as_ref() else {
            return Err(ERROR);
        };
        let callback = (&raw const *assets.error).cast_mut().cast();
        let status = unsafe {
            hsa_queue_create(
                agent,
                256,
                QUEUE_TYPE_SINGLE,
                Some(queue_error),
                callback,
                private_bytes,
                0,
                &raw mut dispatch.queue,
            )
        };
        if status != SUCCESS {
            // create_hardware_queue marks this generation on uncertain native
            // rollback. Its retained callback still needs the atomic owner.
            if dispatch.health.is_quarantined() {
                dispatch.finished = true;
            }
            return Err(status);
        }
        Ok(dispatch)
    }

    /// Borrows the operand owners while preparing a packet's arguments.
    pub(crate) fn dependencies(&self) -> Option<&D> {
        self.assets.as_ref().map(|assets| &assets.dependencies)
    }

    /// Executes a kernel using explicit arguments followed by zero hidden args.
    ///
    /// # Safety
    /// Every address encoded in args must remain device-accessible through
    /// finish, and its owner must be retained in D or on a live-operand result.
    /// The caller must follow the kernel's argument and 64-lane group contract.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn launch(&mut self, name: &str, args: &[u8], work_items: u32) -> Status {
        let Some(assets) = self.assets.as_ref() else {
            return INVALID_RUNTIME_STATE;
        };
        let Some(symbol) = self
            .symbols
            .iter()
            .find(|symbol| symbol.name.strip_suffix(".kd").unwrap_or(&symbol.name) == name)
        else {
            return INVALID_ARGUMENT;
        };
        if self.finished || work_items == 0 || args.len() > symbol.kernarg_size as usize {
            return INVALID_ARGUMENT;
        }
        let Some(grid) = work_items.checked_add(63).map(|count| count & !63) else {
            return INVALID_ARGUMENT;
        };
        if self.stop.load(Ordering::Acquire) {
            return INVALID_RUNTIME_STATE;
        }
        let error = assets.error.load(Ordering::Acquire);
        if error != SUCCESS {
            return error;
        }
        let Some(host) = assets.arguments.info().host_address else {
            return ERROR;
        };
        let address = assets.arguments.info().device_address;
        // SAFETY: One synchronous dispatch at a time uses this owned mapping.
        // The full kernarg segment includes zero initialized hidden arguments.
        unsafe {
            std::ptr::write_bytes(host as *mut u8, 0, symbol.kernarg_size as usize);
            std::ptr::copy_nonoverlapping(args.as_ptr(), host as *mut u8, args.len());
            hsa_signal_store_relaxed(self.completion, 1);
            let index = hsa_queue_add_write_index_relaxed(self.queue, 1);
            let queue = &*self.queue;
            let packet = queue
                .base_address
                .cast::<HsaKernelDispatchPacket>()
                .add((index % u64::from(queue.size)) as usize);
            // The previous completion has retired before this ring is reused.
            packet.write(HsaKernelDispatchPacket {
                header: 1,
                setup: 1,
                workgroup_size_x: 64,
                workgroup_size_y: 1,
                workgroup_size_z: 1,
                reserved0: 0,
                grid_size_x: grid,
                grid_size_y: 1,
                grid_size_z: 1,
                private_segment_size: symbol.private_size,
                group_segment_size: symbol.group_size,
                kernel_object: symbol.address,
                kernarg_address: address as *mut c_void,
                reserved2: 0,
                completion_signal: self.completion,
            });
            // System fences order both the external operands and private args.
            (&*packet.cast::<AtomicU16>()).store(2 | (2 << 9) | (2 << 11), Ordering::Release);
            hsa_signal_store_screlease(
                queue.doorbell_signal,
                i64::from_ne_bytes(index.to_ne_bytes()),
            );
        }
        loop {
            let error = assets.error.load(Ordering::Acquire);
            if error != SUCCESS {
                return error;
            }
            if self.stop.load(Ordering::Acquire) {
                return INVALID_RUNTIME_STATE;
            }
            // A finite wait permits cancellation and queue-error observation.
            let value = unsafe {
                hsa_signal_wait_scacquire(
                    self.completion,
                    SIGNAL_CONDITION_EQ,
                    0,
                    1_000_000,
                    WAIT_STATE_BLOCKED,
                )
            };
            if value == 0 {
                return SUCCESS;
            }
        }
    }

    /// Stops the native queue before releasing any address it could still read.
    #[allow(unsafe_code)]
    fn retire(&mut self, status: Status) -> DispatchResult {
        self.finished = true;
        let retired = self.queue.is_null() || unsafe { hsa_queue_destroy(self.queue) } == SUCCESS;
        if !retired {
            self.health.quarantine();
            return DispatchResult {
                status: ERROR,
                operands_may_be_live: true,
            };
        }
        self.queue = std::ptr::null_mut();
        let signal_status = if self.completion.handle == 0 {
            SUCCESS
        } else {
            unsafe { hsa_signal_destroy(self.completion) }
        };
        self.completion.handle = 0;
        if let Some(assets) = self.assets.take() {
            drop(assets.into_inner());
        }
        DispatchResult {
            status: if status == SUCCESS {
                signal_status
            } else {
                status
            },
            operands_may_be_live: false,
        }
    }

    /// Reports operation and cleanup status after checked native retirement.
    pub(crate) fn finish(mut self, status: Status) -> DispatchResult {
        self.retire(status)
    }
}

impl<D> Drop for Dispatch<D> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.retire(ERROR);
        }
    }
}
