// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU-owned SDMA engines, prepared packets and asynchronous retirement.

#![deny(missing_docs)]

use super::{
    abce_builder_build_fence, abce_builder_initialize, abce_copy_metadata_initialize,
    abce_copy_orchestrator_initialize_with_caps, abce_copy_orchestrator_map_copy,
    abce_copy_orchestrator_register_engine, abce_copy_orchestrator_submit,
    abce_engine_affinity_initialize, abce_ring_host_store_fence, abce_ring_initialize,
    abi::{
        ABCE_LINEAR_BATCH_MODE_AUTOMATIC, ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK,
        ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT, ABCE_OP_KIND_FILL, ABCE_OP_KIND_INDIRECT,
        ABCE_OP_KIND_LINEAR, ABCE_OP_KIND_SWAP, ABCE_STATUS_OK, ABCE_STATUS_RESOURCE_EXHAUSTED,
        ABCE_STATUS_UNIMPLEMENTED, abce_builder_config_t, abce_builder_t, abce_copy_metadata_t,
        abce_copy_op_t, abce_copy_orchestrator_t, abce_dep_signal_t, abce_engine_affinity_t,
        abce_isa_version_t, abce_plan_t, abce_platform_caps_t, abce_ring_config_t, abce_ring_t,
        abce_status_t,
    },
};
use crate::device::gpu::GpuDevice;
use crate::device::gpu::queue::{
    KernelCommand, KernelQueue, KernelQueueFormat, Queue, QueueParameters, QueuePriority,
    QueueRequest, QueueRingMemory, SdmaEngineSelection,
};
use crate::memory::{Allocation, DeviceAccess, MemoryKind};
use crate::topology::{GpuCopyTransport, GpuSdmaDialect};
use crate::{Error, ErrorKind};
use std::mem::ManuallyDrop;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const RING_BYTES: u64 = 65536;

/// An engine selector in its native transport namespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineSelector {
    /// KFD hardware engine inside the activated GPU VM.
    Direct(u32),
    /// Kernel logical engine, as reported by `available_sdma_engines`.
    Kernel(u32),
}

impl EngineSelector {
    fn index(self) -> usize {
        match self {
            Self::Direct(n) | Self::Kernel(n) => n as usize,
        }
    }
}

/// A selectable execution resource of one activated GPU, without a queue.
/// A descriptor is neither a separately activated device nor an endpoint.
#[derive(Clone, Copy)]
pub struct SdmaEngine<'gpu> {
    gpu: GpuDevice<'gpu>,
    selector: EngineSelector,
}

impl SdmaEngine<'_> {
    /// Returns the owner-local selector. It cannot identify another GPU's engine.
    #[must_use]
    pub fn selector(&self) -> EngineSelector {
        self.selector
    }
}

/// Batch placement policy. Explicit fan-out requires independently qualified
/// engines and is unavailable on kernel-mediated transports.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Placement {
    /// Allow the planner to choose using batch size and legal engine choices.
    #[default]
    Automatic,
    /// Place the batch on one engine when its operations permit it.
    Serial,
    /// Require the fan-out planning path.
    FanOut,
}

/// Inputs consumed while preparing a native batch. GPU addresses have the
/// lifetime and access requirements of `CopyEngine::prepare_raw`.
#[derive(Clone, Copy, Default)]
pub struct CopyOptions<'a> {
    /// Engine placement policy.
    pub placement: Placement,
    /// Device-visible dependency words, read before the first body.
    pub dependencies: &'a [abce_dep_signal_t],
    /// Optional device addresses for start/end global-clock timestamps.
    pub timestamps: Option<(u64, u64)>,
}

/// Copy service borrowed from one activated GPU. Looking up the service or its
/// engine descriptors allocates no native queues.
pub struct CopyEngine<'gpu> {
    gpu: GpuDevice<'gpu>,
}

impl<'gpu> GpuDevice<'gpu> {
    /// Obtains the ABCE service in this GPU's native address domain.
    ///
    /// # Errors
    /// Reports unsupported when the provider has no qualified copy recipe.
    pub fn copy_engine(self) -> Result<CopyEngine<'gpu>, Error> {
        recipe(self)?;
        Ok(CopyEngine { gpu: self })
    }
}

fn failure(kind: ErrorKind, detail: &'static str) -> Error {
    Error::Operation { kind, detail }
}

fn recipe(gpu: GpuDevice<'_>) -> Result<GpuCopyTransport, Error> {
    gpu.resources.abce_pool.check_process()?;
    let info = gpu.info();
    let transport = info.queues.copy.ok_or_else(|| {
        failure(
            ErrorKind::Unsupported,
            "GPU has no qualified SDMA copy transport",
        )
    })?;
    let dialect = info.target_profile.sdma_dialect();
    let supported = cfg!(target_arch = "x86_64")
        && match transport {
            GpuCopyTransport::User => {
                dialect == Some(GpuSdmaDialect::Gfx9) && info.queues.sdma.is_some()
            }
            GpuCopyTransport::Kernel => {
                matches!(
                    dialect,
                    Some(GpuSdmaDialect::Gfx1200 | GpuSdmaDialect::Gfx1250)
                ) && info.queues.kernel_sdma
                    && info.queues.sdma_system_cache_control
            }
        };
    if !supported {
        return Err(failure(
            ErrorKind::Unsupported,
            "GPU copy recipe is not qualified for native ABCE",
        ));
    }
    Ok(transport)
}

impl<'gpu> CopyEngine<'gpu> {
    /// Enumerates selectable engines without creating queues. Numeric selectors
    /// remain tied to this activated owner, even across matching endpoint IDs.
    ///
    /// # Errors
    /// Reports an unavailable native transport, engine query failure, or inherited owner.
    pub fn engines(&self) -> Result<Vec<SdmaEngine<'gpu>>, Error> {
        let selectors: Vec<_> = match recipe(self.gpu)? {
            GpuCopyTransport::User => {
                let count = self
                    .gpu
                    .info()
                    .queues
                    .sdma_engine_count
                    .saturating_add(self.gpu.info().sdma_xgmi_engine_count)
                    .min(16);
                (0..count).map(EngineSelector::Direct).collect()
            }
            GpuCopyTransport::Kernel => {
                let mask = self.gpu.available_sdma_engines()?;
                (0..32)
                    .filter(|n| mask & (1 << n) != 0)
                    .map(EngineSelector::Kernel)
                    .collect()
            }
        };
        Ok(selectors
            .into_iter()
            .map(|selector| SdmaEngine {
                gpu: self.gpu,
                selector,
            })
            .collect())
    }

    /// Materializes a batch into owned storage and leases native queues. No
    /// payload or synchronization word is published during preparation.
    ///
    /// # Safety
    /// Operation pointers and nested arrays must satisfy the ABCE C contracts
    /// during this call. Every GPU operand, indirect pointer slot, dependency
    /// and timestamp must be mapped with the required access in this execution
    /// GPU's VM, including qualified peer access for foreign residence. Keep
    /// their backing and mappings alive until the returned submission proves
    /// retirement, or until conclusive native teardown. Dropping or forgetting
    /// an accepted submission and a timeout do not end this obligation.
    ///
    /// # Errors
    /// Reports invalid ownership/placement, unqualified fan-out, mapping or
    /// resource failures. An error here proves no payload was accepted.
    #[allow(
        clippy::too_many_lines,
        reason = "one preparation transaction before native acceptance"
    )]
    pub unsafe fn prepare_raw(
        &self,
        engines: &[SdmaEngine<'gpu>],
        operations: &[abce_copy_op_t],
        options: CopyOptions<'_>,
    ) -> Result<PreparedCopy<'gpu>, Error> {
        let transport = recipe(self.gpu)?;
        if engines.is_empty()
            || engines.len() > 16
            || operations.is_empty()
            || operations.len() > 65536
            || options.dependencies.len() > u32::MAX as usize
        {
            return Err(failure(
                ErrorKind::InvalidArgument,
                "invalid native ABCE batch dimensions",
            ));
        }
        if (transport == GpuCopyTransport::Kernel
            && (engines.len() != 1 || options.placement == Placement::FanOut))
            || (engines.len() > 1 && !self.gpu.info().queues.sdma_multi_engine_copy)
        {
            return Err(failure(
                ErrorKind::Unsupported,
                "native multi-engine ABCE has not been qualified for this transport",
            ));
        }
        let mut seen = 0_u32;
        for engine in engines {
            // GpuResources is unique to the activated owner; endpoint/PCI IDs
            // and numeric engine selectors alone cannot establish VM identity.
            if !std::ptr::eq(engine.gpu.resources, self.gpu.resources)
                || engine.selector.index() >= 32
                || seen & (1 << engine.selector.index()) != 0
            {
                return Err(failure(
                    ErrorKind::InvalidArgument,
                    "ABCE engine belongs to another owner or is duplicated",
                ));
            }
            seen |= 1 << engine.selector.index();
        }
        for operation in operations {
            let supported = matches!(operation.kind, ABCE_OP_KIND_LINEAR | ABCE_OP_KIND_FILL)
                || (self.gpu.info().target_profile.sdma_dialect() == Some(GpuSdmaDialect::Gfx1250)
                    && matches!(operation.kind, ABCE_OP_KIND_SWAP | ABCE_OP_KIND_INDIRECT));
            if !supported {
                return Err(failure(
                    ErrorKind::Unsupported,
                    "native ABCE operation has no qualified device recipe",
                ));
            }
        }

        let sync = allocate(self.gpu, 4096, false)?;
        let sync_host = host_address(&sync)?;
        let sync_device = sync.device_address(self.gpu.device())?;
        let mut builder = Box::new(unsafe { std::mem::zeroed::<abce_builder_t>() });
        let isa = abce_isa_version_t {
            major: self.gpu.info().gfx_major,
            minor: self.gpu.info().gfx_minor,
            stepping: self.gpu.info().gfx_stepping,
        };
        let config = abce_builder_config_t {
            use_copy_size_override: false,
            max_linear_copy_size: 0,
            max_fill_size: 0,
        };
        unsafe {
            abce_builder_initialize(isa, &raw const config, &raw mut *builder);
        }
        // Native kernel copies use explicit cache envelopes, including the
        // six-dword gfx1250 GCR recipe already qualified by rocddi.
        builder.packet_caps.use_gcr = transport == GpuCopyTransport::Kernel;
        let caps = abce_platform_caps_t {
            device_atomic_support: engines.len() > 1,
            emit_hdp_flush: false,
            driver_manages_gcr: false,
        };
        let mut orchestrator = Box::new(unsafe { std::mem::zeroed::<abce_copy_orchestrator_t>() });
        unsafe {
            abce_copy_orchestrator_initialize_with_caps(
                &raw const *builder,
                caps,
                &raw mut *orchestrator,
            );
        }
        let mut contexts = Vec::with_capacity(engines.len());
        for (slot, engine) in engines.iter().enumerate() {
            let mut context = self
                .gpu
                .resources
                .abce_pool
                .take(self.gpu, engine.selector)?;
            let mut affinity = unsafe { std::mem::zeroed::<abce_engine_affinity_t>() };
            unsafe {
                abce_engine_affinity_initialize(&raw mut affinity);
            }
            // Kernel logical indices must not enter KFD's physical ranking
            // namespace. Registry slots remain separate from both selectors.
            if let EngineSelector::Direct(id) = engine.selector {
                affinity.hw_engine_id = id;
            }
            unsafe {
                abce_copy_orchestrator_register_engine(
                    &raw mut *orchestrator,
                    native_count(slot)?,
                    &raw mut *context.ring,
                    &raw const affinity,
                );
            }
            contexts.push(context);
        }
        let mut metadata = unsafe { std::mem::zeroed::<abce_copy_metadata_t>() };
        unsafe {
            abce_copy_metadata_initialize(&raw mut metadata);
        }
        metadata.out.value = sync_device as *mut _;
        metadata.out.coordination_scratch = (sync_device + 8) as *mut _;
        metadata.out.sole_update = true;
        metadata.prefer_fused = false;
        metadata.deps = options.dependencies.as_ptr();
        metadata.num_deps = native_count(options.dependencies.len())?;
        metadata.linear_batch_mode = match options.placement {
            Placement::Automatic => ABCE_LINEAR_BATCH_MODE_AUTOMATIC,
            Placement::Serial => ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK,
            Placement::FanOut => ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT,
        };
        if let Some((start, end)) = options.timestamps {
            if start == 0 || end == 0 || (start | end) & 7 != 0 {
                return Err(failure(
                    ErrorKind::InvalidArgument,
                    "timestamps require two aligned GPU addresses",
                ));
            }
            metadata.timestamps.start_value = start as *mut _;
            metadata.timestamps.end_value = end as *mut _;
        }
        let mut plan = unsafe { std::mem::zeroed::<abce_plan_t>() };
        let status = unsafe {
            abce_copy_orchestrator_map_copy(
                &raw mut *orchestrator,
                operations.as_ptr(),
                native_count(operations.len())?,
                &raw const metadata,
                &raw mut plan,
            )
        };
        if status != ABCE_STATUS_OK {
            return Err(map_error(status));
        }
        let mut packets = Vec::with_capacity(plan.num_frames as usize);
        for frame in &mut plan.frames[..plan.num_frames as usize] {
            let context = &mut contexts[frame.engine as usize];
            let suffix = if transport == GpuCopyTransport::User {
                4
            } else {
                0
            };
            let count = frame.bytes as usize / 4;
            let mut words = vec![0_u32; count + suffix];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    frame.packets.cast::<u8>(),
                    words.as_mut_ptr().cast::<u8>(),
                    count * 4,
                );
            }
            if suffix != 0 {
                unsafe {
                    abce_builder_build_fence(
                        &raw const *builder,
                        words.as_mut_ptr().add(count).cast(),
                        context.command_device as *mut u32,
                        1,
                    );
                }
            }
            frame.bytes = native_count(words.len() * 4)?;
            if u64::from(frame.bytes) >= RING_BYTES {
                return Err(failure(
                    ErrorKind::InvalidArgument,
                    "native ABCE frame exceeds queue capacity",
                ));
            }
            if !context.fits_contiguously(u64::from(frame.bytes))? {
                // A fresh queue avoids a wrap-padding wait inside submit.
                // Preparation owns this replacement, before acceptance.
                let replacement = Context::new(self.gpu, context.selector)?;
                *context = replacement;
                orchestrator.engines[frame.engine as usize] = &raw mut *context.ring;
                // The private fence allocation also changed.
                unsafe {
                    abce_builder_build_fence(
                        &raw const *builder,
                        words.as_mut_ptr().add(count).cast(),
                        context.command_device as *mut u32,
                        1,
                    );
                }
            }
            frame.packets = words.as_ptr().cast();
            packets.push(words);
        }
        // Packet addresses use the GPU projection; only the submit-time arm
        // uses the CPU projection. Never infer that these addresses are equal.
        if plan.coordination_signal as u64 == sync_device {
            plan.coordination_signal = sync_host as *mut _;
        } else if plan.coordination_signal as u64 == sync_device + 8 {
            plan.coordination_signal = (sync_host + 8) as *mut _;
        }
        Ok(PreparedCopy {
            work: Work {
                pool: &self.gpu.resources.abce_pool,
                contexts,
                _builder: builder,
                orchestrator,
                plan,
                _packets: packets,
                sync: ManuallyDrop::new(sync),
                sync_host,
                accepting: false,
                accepted: false,
                retired: false,
            },
        })
    }
}

fn native_count(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| {
        failure(
            ErrorKind::InvalidArgument,
            "ABCE dimension exceeds the C ABI limit",
        )
    })
}

fn map_error(status: abce_status_t) -> Error {
    failure(
        if status == ABCE_STATUS_RESOURCE_EXHAUSTED {
            ErrorKind::ResourceExhausted
        } else if status == ABCE_STATUS_UNIMPLEMENTED {
            ErrorKind::Unsupported
        } else {
            ErrorKind::InvalidArgument
        },
        "ABCE rejected the native batch",
    )
}

fn allocate(gpu: GpuDevice<'_>, bytes: u64, executable: bool) -> Result<Allocation, Error> {
    let access = DeviceAccess::READ
        | DeviceAccess::WRITE
        | if executable {
            DeviceAccess::EXECUTE
        } else {
            DeviceAccess::NONE
        };
    gpu.device()
        .allocate(MemoryKind::System, bytes, 4096, access)
}
fn host_address(allocation: &Allocation) -> Result<usize, Error> {
    allocation.info().host_address.ok_or_else(|| {
        failure(
            ErrorKind::Unsupported,
            "native ABCE requires a host-mapped system allocation",
        )
    })
}

enum NativeQueue {
    Direct(Queue),
    Kernel(KernelQueue),
}

struct Context {
    process: crate::os::ProcessIdentity,
    selector: EngineSelector,
    queue: ManuallyDrop<NativeQueue>,
    command: ManuallyDrop<Allocation>,
    command_host: usize,
    command_device: u64,
    ring: Box<abce_ring_t>,
    pending: Option<u64>,
    accepting: bool,
}

// SAFETY: This context has one producer. All raw pointers reference stable
// boxes or mappings owned by its queue/allocation. No caller callbacks or
// borrowed C objects are installed. Pool transfers give exclusive ownership.
unsafe impl Send for Context {}

impl Context {
    fn new(gpu: GpuDevice<'_>, selector: EngineSelector) -> Result<Self, Error> {
        let command = allocate(
            gpu,
            RING_BYTES,
            matches!(selector, EngineSelector::Kernel(_)),
        )?;
        let command_host = host_address(&command)?;
        let command_device = command.device_address(gpu.device())?;
        let mut ring = Box::new(unsafe { std::mem::zeroed::<abce_ring_t>() });
        let queue = match selector {
            EngineSelector::Direct(id) => {
                let queue = unsafe {
                    gpu.create_queue(QueueRequest {
                        ring_size_bytes: RING_BYTES,
                        parameters: QueueParameters::SdmaByEngine {
                            selection: SdmaEngineSelection::Id(id),
                            ring_memory: QueueRingMemory::System,
                        },
                        priority: QueuePriority::Normal,
                        device_producer: false,
                    })
                }?;
                let info = queue.info();
                let mut config = unsafe { std::mem::zeroed::<abce_ring_config_t>() };
                config.base = info.ring_host_address as *mut _;
                config.size = usize::try_from(info.ring_size_bytes).map_err(|_| {
                    failure(
                        ErrorKind::InvalidArgument,
                        "ABCE ring size exceeds host address space",
                    )
                })?;
                config.read_ptr = info.read_index_host_address as *mut _;
                config.write_ptr = info.write_index_host_address as *mut _;
                config.doorbell = info.doorbell_host_address as *mut _;
                unsafe {
                    abce_ring_initialize(
                        &raw const config,
                        None,
                        std::ptr::null_mut(),
                        &raw mut *ring,
                    );
                }
                NativeQueue::Direct(queue)
            }
            EngineSelector::Kernel(id) => {
                NativeQueue::Kernel(gpu.create_kernel_queue(KernelQueueFormat::SdmaOnEngine(id))?)
            }
        };
        Ok(Self {
            process: gpu.resources.abce_pool.identity()?.clone(),
            selector,
            queue: ManuallyDrop::new(queue),
            command: ManuallyDrop::new(command),
            command_host,
            command_device,
            ring,
            pending: None,
            accepting: false,
        })
    }

    fn fits_contiguously(&self, bytes: u64) -> Result<bool, Error> {
        match &*self.queue {
            NativeQueue::Direct(queue) => {
                queue.check()?;
                let (read, write) = queue.progress()?;
                Ok(read == write
                    && write.checked_add(bytes).is_some()
                    && (write & (RING_BYTES - 1)) + bytes < RING_BYTES)
            }
            NativeQueue::Kernel(queue) => {
                Ok(queue.status().terminal.is_none() && self.pending.is_none())
            }
        }
    }

    fn poll(&mut self) -> Result<bool, Error> {
        let Some(pending) = self.pending else {
            return Ok(!self.accepting);
        };
        let retired = match &*self.queue {
            NativeQueue::Direct(queue) => {
                queue.check()?;
                let (read, write) = queue.progress()?;
                unsafe {
                    AtomicU32::from_ptr(self.command_host as *mut u32).load(Ordering::Acquire) == 1
                        && read == write
                        && write == pending
                }
            }
            NativeQueue::Kernel(queue) => {
                let status = queue.refresh_status()?;
                if status.retired_submission >= pending {
                    self.pending = None;
                    self.accepting = false;
                }
                if let Some(kind) = status.terminal {
                    return Err(failure(kind, "accepted kernel ABCE submission failed"));
                }
                status.retired_submission >= pending
            }
        };
        if retired {
            self.pending = None;
            self.accepting = false;
        }
        Ok(retired)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.process.check().is_err() {
            return;
        }
        if self.pending.is_some() || self.accepting {
            return;
        }
        let destroyed = match &mut *self.queue {
            NativeQueue::Direct(queue) => unsafe { queue.destroy() },
            NativeQueue::Kernel(queue) => queue.destroy(),
        };
        if destroyed.is_err() {
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.queue);
            ManuallyDrop::drop(&mut self.command);
        }
    }
}

/// Lazy queue cache in the activated GPU's resource owner. No Device clone is
/// retained here, so the device/service ownership graph has no cycle.
pub(crate) struct CopyPool {
    process: std::io::Result<crate::os::ProcessIdentity>,
    idle: Mutex<[Option<Context>; 32]>,
}

impl Default for CopyPool {
    fn default() -> Self {
        Self {
            process: crate::os::ProcessIdentity::capture(),
            idle: Mutex::new(std::array::from_fn(|_| None)),
        }
    }
}

impl CopyPool {
    fn identity(&self) -> Result<&crate::os::ProcessIdentity, Error> {
        self.process.as_ref().map_err(|_| {
            failure(
                ErrorKind::Unsupported,
                "ABCE process identity could not be captured",
            )
        })
    }

    fn check_process(&self) -> Result<(), Error> {
        self.identity()?.check().map_err(|_| {
            failure(
                ErrorKind::InvalidArgument,
                "inherited ABCE service cannot be used after fork",
            )
        })
    }
    fn take(&self, gpu: GpuDevice<'_>, selector: EngineSelector) -> Result<Context, Error> {
        self.check_process()?;
        let cached = self
            .idle
            .lock()
            .map_err(|_| failure(ErrorKind::Internal, "ABCE pool lock was poisoned"))?
            [selector.index()]
        .take();
        if let Some(context) = cached {
            if context.selector == selector && context.fits_contiguously(0)? {
                return Ok(context);
            }
        }
        Context::new(gpu, selector)
    }
    fn put(&self, context: Context) {
        if self.check_process().is_err() {
            return;
        }
        if let Ok(mut idle) = self.idle.lock() {
            let index = context.selector.index();
            if idle[index].is_none() {
                idle[index] = Some(context);
            }
        }
    }
}

struct Work<'gpu> {
    pool: &'gpu CopyPool,
    contexts: Vec<Context>,
    _builder: Box<abce_builder_t>,
    orchestrator: Box<abce_copy_orchestrator_t>,
    plan: abce_plan_t,
    _packets: Vec<Vec<u32>>,
    sync: ManuallyDrop<Allocation>,
    sync_host: usize,
    accepting: bool,
    accepted: bool,
    retired: bool,
}

impl Drop for Work<'_> {
    fn drop(&mut self) {
        if self.pool.check_process().is_err() {
            return;
        }
        if self.accepting || (self.accepted && !self.retired) {
            // A timeout/error/drop does not revoke GPU access. Context Drop
            // retains command/fence/queue owners; retain shared coordination too.
            return;
        }
        for context in self.contexts.drain(..) {
            self.pool.put(context);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.sync);
        }
    }
}

/// Owned packets and exclusive queue leases. Submission consumes this plan once.
pub struct PreparedCopy<'gpu> {
    work: Work<'gpu>,
}

/// Accepted work, including an ambiguous native acceptance. Error and timeout
/// observations leave all native dependencies retained.
pub struct CopySubmission<'gpu> {
    work: Work<'gpu>,
    terminal: Option<Error>,
}

/// Checked progress of an accepted submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CopyProgress {
    /// Native work may still reference its operands and command storage.
    Pending,
    /// Every accepted queue has crossed its checked retirement frontier.
    Retired,
}

impl<'gpu> PreparedCopy<'gpu> {
    /// Publishes every participating ring before returning, without waiting
    /// for any payload to complete. An error proves no payload was accepted.
    ///
    /// # Errors
    /// Reports rejection before acceptance; uncertain outcomes return a submission owner.
    pub fn submit(mut self) -> Result<CopySubmission<'gpu>, Error> {
        self.work.pool.check_process()?;
        for frame in &self.work.plan.frames[..self.work.plan.num_frames as usize] {
            if !self.work.contexts[frame.engine as usize]
                .fits_contiguously(u64::from(frame.bytes))?
            {
                return Err(failure(
                    ErrorKind::Busy,
                    "ABCE queue is no longer ready for publication",
                ));
            }
        }
        unsafe {
            AtomicU64::from_ptr(self.work.sync_host as *mut u64).store(1, Ordering::Release);
        }
        if matches!(&*self.work.contexts[0].queue, NativeQueue::Kernel(_)) {
            let context = &self.work.contexts[0];
            let frame = &self.work.plan.frames[0];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    frame.packets.cast::<u8>(),
                    context.command_host as *mut u8,
                    frame.bytes as usize,
                );
                abce_ring_host_store_fence();
            }
            self.work.accepting = true;
            self.work.contexts[0].accepting = true;
            // Obtain the queue again after marking the entire bundle retained.
            let NativeQueue::Kernel(queue) = &*self.work.contexts[0].queue else {
                std::process::abort()
            };
            let accepted = unsafe {
                queue.submit(KernelCommand {
                    device_address: self.work.contexts[0].command_device,
                    byte_length: u64::from(self.work.plan.frames[0].bytes),
                })
            };
            self.work.accepting = false;
            self.work.contexts[0].accepting = false;
            self.work.contexts[0].pending = Some(accepted?);
            self.work.accepted = true;
            return Ok(CopySubmission {
                work: self.work,
                terminal: None,
            });
        }
        // All allocations, health checks and capacity checks precede the first
        // reservation. Exclusive leases and contiguous empty rings make the
        // shared multi-ring transaction nonblocking on other producers.
        for context in &mut self.work.contexts {
            unsafe {
                AtomicU32::from_ptr(context.command_host as *mut u32).store(0, Ordering::Release);
            }
            context.accepting = true;
        }
        self.work.accepting = true;
        let status = unsafe {
            abce_copy_orchestrator_submit(&raw mut *self.work.orchestrator, &raw mut self.work.plan)
        };
        self.work.accepted = true;
        self.work.accepting = false;
        if status == ABCE_STATUS_OK {
            for frame in &self.work.plan.frames[..self.work.plan.num_frames as usize] {
                let context = &mut self.work.contexts[frame.engine as usize];
                context.pending = Some(context.ring.control.max_write_index);
                context.accepting = false;
            }
            // Registered but unselected queues have no accepted work.
            for context in &mut self.work.contexts {
                context.accepting = false;
            }
        }
        Ok(CopySubmission {
            work: self.work,
            terminal: (status != ABCE_STATUS_OK).then(|| map_error(status)),
        })
    }
}

impl CopySubmission<'_> {
    /// Whether checked native retirement permits releasing the operands.
    /// Retirement is independent of successful execution; `poll` still reports
    /// a sticky execution failure after a failed submission retires.
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.work.retired
    }
    /// Checks every queue's retirement frontier. A failure keeps the accepted
    /// owner live and is not evidence that operands may be freed.
    ///
    /// # Errors
    /// Returns a sticky native failure, independently of checked retirement.
    pub fn poll(&mut self) -> Result<CopyProgress, &Error> {
        if let Err(error) = self.work.pool.check_process() {
            self.terminal = Some(error);
        }
        if self.work.retired || self.work.pool.check_process().is_err() {
            return match &self.terminal {
                Some(error) => Err(error),
                None => Ok(CopyProgress::Retired),
            };
        }
        let mut retired = true;
        for context in &mut self.work.contexts {
            match context.poll() {
                Ok(done) => retired &= done,
                Err(error) => {
                    retired = false;
                    if self.terminal.is_none() {
                        self.terminal = Some(error);
                    }
                }
            }
        }
        self.work.retired = retired
            || self
                .work
                .contexts
                .iter()
                .all(|c| c.pending.is_none() && !c.accepting);
        match &self.terminal {
            Some(error) => Err(error),
            None => Ok(if self.work.retired {
                CopyProgress::Retired
            } else {
                CopyProgress::Pending
            }),
        }
    }

    /// Waits up to the requested duration. `Pending` leaves the submission and
    /// its dependencies retained for another wait or explicit observation.
    ///
    /// # Errors
    /// Returns the same sticky native failure as `poll`.
    pub fn wait(&mut self, timeout: Duration) -> Result<CopyProgress, &Error> {
        let start = Instant::now();
        loop {
            let _ = self.poll();
            if self.terminal.is_some() {
                return self.poll();
            }
            if self.work.retired {
                return Ok(CopyProgress::Retired);
            }
            if start.elapsed() >= timeout {
                return Ok(CopyProgress::Pending);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;
