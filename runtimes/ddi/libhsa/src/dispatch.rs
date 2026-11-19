// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Private synchronous AQL dispatches used by HSA image and memory operations.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};

use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocloader::{LoadedImage, Symbol, SymbolKind};

use crate::admission::InFlightToken;
use crate::callback_arg::CallbackArg;
use crate::ffi::*;
use crate::lifecycle::lock;
use crate::queue::index::{QueueIndex, with_queue_index};
use crate::queue::{AqlQueueKind, AqlQueueRequest};
use crate::session::SessionHealth;
use crate::signal;
use crate::status::map_error;
use rocddi::memory::storage::RetainOnDrop;

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
    queue: Option<std::ptr::NonNull<HsaQueue>>,
    /// Completion signal owned until native queue retirement.
    completion: HsaSignal,
    /// Native dependencies use fail-closed ownership during unwinding.
    assets: Option<RetainOnDrop<Assets<D>>>,
    /// Relocated symbols for each accepted kernel.
    symbols: Vec<Symbol>,
    /// Cancellation requested by runtime shutdown.
    stop: Arc<AtomicBool>,
    /// Prevents generation teardown after uncertain cleanup.
    health: Arc<SessionHealth>,
    /// Always released, even if assets are retained, so shutdown can progress.
    _inflight: InFlightToken,
    /// Explicit cleanup has run and must not be retried.
    finished: bool,
}

/// Retains code if publication fails or unwinds with uncertain native users.
struct PendingCode {
    /// The image is transferred only after complete publication.
    image: Option<LoadedImage>,
    /// Uncertain code publication blocks this HSA generation's teardown.
    health: Arc<SessionHealth>,
}

impl PendingCode {
    /// Publishes executable bytes and caches through the loader's single owner.
    fn publish(mut self, stop: &AtomicBool) -> Result<LoadedImage, Status> {
        let image = self.image.as_ref().ok_or(ERROR)?;
        if let Some(publication) = image.publication() {
            publication
                .publish(stop)
                .map_err(crate::loader::map_loader_error)?;
        }
        self.image.take().ok_or(ERROR)
    }
}

impl Drop for PendingCode {
    fn drop(&mut self) {
        if let Some(image) = self.image.take() {
            let can_release = image.can_release();
            release_after_publication(image, can_release, &self.health);
        }
    }
}

/// Applies the publication owner's release proof to its retained backing.
/// Keeping this disposition explicit also covers unwinding during publication.
fn release_after_publication<T>(owner: T, can_release: bool, health: &SessionHealth) {
    let owner = RetainOnDrop::new(owner);
    if can_release {
        drop(owner.into_inner());
    } else {
        health.quarantine();
    }
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
            let index = runtime.agents.gpu_index(agent).ok_or(INVALID_AGENT)?;
            if runtime.control.operation_stop().load(Ordering::Acquire) {
                return Err(INVALID_RUNTIME_STATE);
            }
            (
                runtime.agents.gpus[index].device.clone(),
                runtime.agents.host_page_size as u64,
                runtime.control.operation_stop(),
                runtime.control.health.clone(),
                runtime.control.inflight.enter().ok_or(OUT_OF_RESOURCES)?,
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
        // Publication owns both ordinary images and target entry stubs. Its
        // guard preserves the image through errors and unwinding while native
        // code-cache work might still reference either allocation.
        let image = PendingCode {
            image: Some(image),
            health: health.clone(),
        }
        .publish(&stop)?;
        let mut dispatch = Self {
            queue: None,
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
        // The current admitted operation and the dispatch in-flight token keep
        // the generation alive while its signal and queue owners are published.
        dispatch.completion = {
            let mut guard = lock()?;
            let runtime = crate::lifecycle::initialized_mut(&mut guard)?;
            runtime.create_signal(1, true)?
        };
        let Some(assets) = dispatch.assets.as_ref() else {
            return Err(ERROR);
        };
        let callback = (&raw const *assets.error).cast_mut().cast();
        let request = AqlQueueRequest {
            agent,
            size: 256,
            kind: AqlQueueKind::Single,
            callback: Some(queue_error),
            // SAFETY: Assets retain this atomic until native queue retirement.
            data: unsafe { CallbackArg::new(callback) },
            private_segment_size: private_bytes,
        };
        // SAFETY: The retained assets cover callback code/data through the last
        // native use, including ambiguous queue-creation rollback.
        match unsafe { crate::queue::create(request) } {
            Ok(queue) => dispatch.queue = Some(queue),
            Err(status) => {
                if dispatch.health.is_quarantined() {
                    dispatch.finished = true;
                }
                return Err(status);
            }
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
            if signal::store(self.completion, 1, Ordering::Relaxed).is_none() {
                return INVALID_RUNTIME_STATE;
            }
            let Some(queue) = self.queue else {
                return INVALID_RUNTIME_STATE;
            };
            let Some(index) = with_queue_index(queue.as_ptr(), QueueIndex::Write, |index| {
                index.fetch_add(1, Ordering::Relaxed)
            }) else {
                return INVALID_RUNTIME_STATE;
            };
            let queue = queue.as_ref();
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
            if signal::store(
                queue.doorbell_signal,
                i64::from_ne_bytes(index.to_ne_bytes()),
                Ordering::Release,
            )
            .is_none()
            {
                return INVALID_RUNTIME_STATE;
            }
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
                signal::wait::signal_wait(
                    self.completion,
                    SIGNAL_CONDITION_EQ,
                    0,
                    1_000_000,
                    WAIT_STATE_BLOCKED,
                    Ordering::Acquire,
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
        let retired = self
            .queue
            .is_none_or(|queue| crate::queue::destroy(queue).is_ok());
        if !retired {
            self.health.quarantine();
            return DispatchResult {
                status: ERROR,
                operands_may_be_live: true,
            };
        }
        self.queue = None;
        let signal_status = if self.completion.handle == 0 {
            SUCCESS
        } else {
            match signal::destroy(self.completion) {
                Ok(()) => SUCCESS,
                Err(status) => status,
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Observable native backing used to inject a publication release proof.
    struct Backing(Arc<AtomicBool>);

    impl Drop for Backing {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[test]
    fn failed_publication_retains_backing_until_native_retirement_is_proved() {
        for can_release in [false, true] {
            let released = Arc::new(AtomicBool::new(false));
            let health = SessionHealth::new();
            release_after_publication(Backing(released.clone()), can_release, &health);
            assert_eq!(released.load(Ordering::Acquire), can_release);
            assert_eq!(health.is_quarantined(), !can_release);
        }
    }
}
