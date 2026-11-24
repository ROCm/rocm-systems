// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native Linux KFD connection and call adapters.
//!
//! `Kfd` owns the native descriptor and serializes context selection and runtime
//! enablement on that connection. The private `ioctl` module marshals arguments
//! and dispatches every KFD ioctl through the same production or test boundary.
//!
//! Wrappers own or borrow ioctl records and temporary indirect buffers through
//! each synchronous syscall and preserve outputs on error. Queue rings,
//! registered user memory, event pages, and other referenced backing remain
//! owned by their resource owners until native use ends. Those owners decide
//! whether to retry side-effecting calls; dispatch never silently retries
//! `EINTR`, and mapping operations preserve the returned successful prefix.
//!
//! Mapping owners expose numeric addresses. Replacing a mapping requires an
//! address reservation owned by the Linux OS service. Resource owners must
//! finish native GPU cleanup before releasing the reservation. Inherited
//! owners never issue ioctls or unmap reservations in a fork child.

#![allow(unsafe_code)]

mod ioctl;
pub(super) use ioctl::Call;
#[cfg(test)]
pub(super) use ioctl::IoctlHook;

use crate::memory::storage::TryPush;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::sync::{Mutex, MutexGuard};

use super::errno;
use super::uapi;
use super::{DebuggerRegistration, DebuggerRegistrationMode, XnackModeRequest};
use crate::os::ProcessIdentity;
use crate::os::linux::{
    file::{close_descriptor, close_file},
    invalid_data,
    memory::{Reservation, page_size},
};
use crate::session::DriverContextLifetime;

/// DMA-BUF extent and metadata copied out of one synchronous KFD query.
pub(super) struct DmaBufDetails {
    /// Native size, flags, and owner GPU returned by KFD.
    pub(super) info: uapi::DmaBufInfo,
    /// Driver-owned metadata copied out of the query buffer.
    pub(super) metadata: Vec<u8>,
}

/// An owned KFD endpoint. Kernel calls may run concurrently; operation owners
/// supply their own buffers and serialize mutations of their own resources.
/// Runtime enable and disable are separately serialized here so no session can
/// publish a partially enabled process state.
pub(super) struct Kfd {
    /// Exact native descriptor used for this context.
    file: Option<File>,
    /// Creating process identity checked before native calls.
    process: ProcessIdentity,
    /// Serialized context selection and runtime enablement.
    runtime: Mutex<RuntimeControl>,
    /// Optional unit-test replacement for native ioctls.
    #[cfg(test)]
    hook: Option<IoctlHook>,
}

/// Progress of process-wide KFD runtime enablement and cleanup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeState {
    Disabled,
    EnableInterrupted,
    Enabled,
    CleanupRequired,
}

/// The context belongs to the KFD file description, so a failed selection
/// cannot be replayed when the kernel may already have replaced its owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContextState {
    Primary,
    Secondary,
    SelectionUncertain,
}

/// Context selection and enablement guarded as one KFD transaction.
struct RuntimeControl {
    /// Primary or secondary context selected for this file description.
    context: ContextState,
    /// Last known native runtime enablement state.
    enable: RuntimeState,
    /// Registration carried by the current or pending enable transaction.
    debugger: Option<DebuggerRegistration>,
}

/// Guard retaining exclusive access to one runtime-enable transaction.
pub(super) struct RuntimeGuard<'a> {
    /// Holds exclusive access through the native enable transaction.
    state: MutexGuard<'a, RuntimeControl>,
}

impl Kfd {
    pub(super) fn new(file: File, process: ProcessIdentity) -> Self {
        Self {
            file: Some(file),
            process,
            runtime: Mutex::new(RuntimeControl {
                context: ContextState::Primary,
                enable: RuntimeState::Disabled,
                debugger: None,
            }),
            #[cfg(test)]
            hook: None,
        }
    }

    #[cfg(test)]
    #[allow(clippy::expect_used)]
    pub(super) fn with_hook(file: File, hook: IoctlHook) -> Self {
        Self {
            file: Some(file),
            process: ProcessIdentity::capture().expect("test process identity"),
            runtime: Mutex::new(RuntimeControl {
                context: ContextState::Primary,
                enable: RuntimeState::Disabled,
                debugger: None,
            }),
            hook: Some(hook),
        }
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        self.check_process()?;
        self.disable_runtime()?;
        close_file(&mut self.file)
    }

    pub(super) fn check_process(&self) -> io::Result<()> {
        self.process.check()
    }

    pub(super) fn version(&self) -> io::Result<uapi::Version> {
        let mut version = uapi::Version::default();
        self.call(Call::Version(&mut version))?;
        Ok(version)
    }

    pub(super) fn clock_counters(&self, gpu_id: u32) -> io::Result<uapi::ClockCounters> {
        let mut counters = uapi::ClockCounters {
            gpu_id,
            ..uapi::ClockCounters::default()
        };
        self.call(Call::ClockCounters(&mut counters))?;
        Ok(counters)
    }

    pub(super) fn xnack_enabled(&self) -> io::Result<bool> {
        let mut mode = uapi::XnackMode { xnack_enabled: -1 };
        self.call(Call::XnackMode(&mut mode))?;
        match mode.xnack_enabled {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid_data("KFD returned an invalid XNACK mode")),
        }
    }

    /// Applies a requested mode while serialized with VM acquisition and enable.
    /// Matching active modes need only a query: KFD rejects even an identical
    /// set request while queues exist. A conflicting active mode is never changed.
    pub(super) fn configure_xnack_mode_locked(
        &self,
        runtime: &mut RuntimeGuard<'_>,
        request: XnackModeRequest,
    ) -> io::Result<()> {
        let (enabled, required) = match request {
            XnackModeRequest::Require(enabled) => (enabled, true),
            XnackModeRequest::Prefer(enabled) => (enabled, false),
        };
        if runtime.state.context == ContextState::SelectionUncertain {
            return Err(invalid_data(
                "KFD context selection outcome requires teardown",
            ));
        }
        if self.xnack_enabled()? == enabled {
            return Ok(());
        }
        if runtime.state.enable != RuntimeState::Disabled {
            return if required {
                Err(io::Error::from_raw_os_error(errno::EBUSY))
            } else {
                Ok(())
            };
        }
        let selected = self.call(Call::XnackMode(&mut uapi::XnackMode {
            xnack_enabled: i32::from(enabled),
        }));
        if required {
            selected?;
        }
        // Even an interrupted or rejected set requires a fresh query. A
        // preferred request never turns an unknown effective mode into success.
        if self.xnack_enabled()? != enabled && required {
            return Err(invalid_data("KFD did not select the requested XNACK mode"));
        }
        Ok(())
    }

    pub(super) fn tile_config(&self, gpu_id: u32) -> io::Result<uapi::TileConfig> {
        if gpu_id == 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        // KFD's legacy array outputs are required by the ioctl even though
        // GFX12 AddrLib consumes only GB_ADDR_CONFIG. ROCr offers 40 entries.
        let mut tiles = [0u32; 40];
        let mut macro_tiles = [0u32; 40];
        let mut config = uapi::TileConfig {
            gpu_id,
            ..uapi::TileConfig::default()
        };
        self.call(Call::TileConfig(&mut config, &mut tiles, &mut macro_tiles))?;
        if config.num_tile_configs as usize > tiles.len()
            || config.num_macro_tile_configs as usize > macro_tiles.len()
        {
            return Err(invalid_data(
                "KFD tile configuration exceeds supplied arrays",
            ));
        }
        Ok(config)
    }

    pub(super) fn available_memory(&self, gpu_id: u32) -> io::Result<u64> {
        let mut memory = uapi::AvailableMemory {
            gpu_id,
            ..uapi::AvailableMemory::default()
        };
        self.call(Call::AvailableMemory(&mut memory))?;
        Ok(memory.available)
    }

    pub(super) fn apertures(&self) -> io::Result<Vec<uapi::Aperture>> {
        let mut args = uapi::Apertures::default();
        self.call(Call::Apertures(&mut args, &mut []))?;
        let count = usize::try_from(args.count).map_err(invalid_data)?;
        let mut entries = crate::memory::storage::try_vec_with_capacity(count)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for _ in 0..count {
            entries
                .try_push(uapi::Aperture::default())
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        if count == 0 {
            return Ok(entries);
        }
        self.call(Call::Apertures(&mut args, &mut entries))?;
        let returned = usize::try_from(args.count).map_err(invalid_data)?;
        if returned > entries.len() {
            return Err(invalid_data(
                "KFD returned more apertures than supplied storage",
            ));
        }
        entries.truncate(returned);
        Ok(entries)
    }

    pub(super) fn acquire_vm(&self, render: &File, gpu_id: u32) -> io::Result<()> {
        let mut args = uapi::AcquireVm {
            drm_fd: u32::try_from(render.as_raw_fd()).map_err(invalid_data)?,
            gpu_id,
        };
        self.call(Call::AcquireVm(&mut args))
    }

    pub(super) fn runtime_guard(&self) -> io::Result<RuntimeGuard<'_>> {
        self.check_process()?;
        self.runtime
            .lock()
            .map(|state| RuntimeGuard { state })
            .map_err(|_| invalid_data("KFD runtime lock poisoned"))
    }

    /// SDK-created PC sampling IDs belong to KFD's primary process context.
    pub(super) fn uses_primary_context(&self) -> io::Result<bool> {
        Ok(self.runtime_guard()?.state.context == ContextState::Primary)
    }

    /// Selects the instance-owned context before its first VM or runtime call.
    /// A version or open failure leaves the primary context retryable. An ioctl
    /// failure has no portable proof of whether selection took effect, so only
    /// closing this descriptor can recover from that outcome.
    pub(super) fn prepare_context(&self, lifetime: DriverContextLifetime) -> io::Result<()> {
        let mut runtime = self.runtime_guard()?;
        match (lifetime, runtime.state.context) {
            (DriverContextLifetime::Process, ContextState::Primary)
            | (DriverContextLifetime::Session, ContextState::Secondary) => return Ok(()),
            (_, ContextState::SelectionUncertain) => {
                return Err(invalid_data(
                    "KFD context selection outcome requires teardown",
                ));
            }
            (DriverContextLifetime::Process, ContextState::Secondary) => {
                return Err(invalid_data(
                    "KFD endpoint already owns a secondary context",
                ));
            }
            (DriverContextLifetime::Session, ContextState::Primary) => {}
        }
        if runtime.state.enable != RuntimeState::Disabled {
            return Err(invalid_data(
                "KFD runtime is already active on the primary context",
            ));
        }
        let version = self.version()?;
        if version.major != 1 || version.minor < 19 {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        let result = self.call(Call::CreateProcess(&mut uapi::CreateProcess::default()));
        runtime.state.context = if result.is_ok() {
            ContextState::Secondary
        } else {
            ContextState::SelectionUncertain
        };
        result
    }

    pub(super) fn enable_runtime(&self) -> io::Result<()> {
        let mut runtime = self.runtime_guard()?;
        self.enable_runtime_locked(&mut runtime)
    }

    /// Binds debugger policy to the exact serialized runtime enable transaction.
    /// A native-only session may join an existing registration without replacing
    /// it; an explicit incompatible registration requires the old runtime to end.
    pub(super) fn configure_debugger_locked(
        &self,
        runtime: &mut RuntimeGuard<'_>,
        registration: Option<DebuggerRegistration>,
    ) -> io::Result<bool> {
        self.check_process()?;
        if runtime.state.context == ContextState::SelectionUncertain {
            return Err(invalid_data(
                "KFD context selection outcome requires teardown",
            ));
        }
        if runtime.state.enable == RuntimeState::Disabled {
            runtime.state.debugger = registration;
            return Ok(true);
        }
        if let Some(request) = registration {
            let matches = runtime.state.debugger.is_some_and(|active| {
                (active.address, active.setup_ttmp) == (request.address, request.setup_ttmp)
            });
            if !matches {
                return if request.mode == DebuggerRegistrationMode::Prefer {
                    Ok(false)
                } else {
                    Err(io::Error::from_raw_os_error(errno::EBUSY))
                };
            }
        }
        Ok(true)
    }

    /// Reports the effective registration after activation, including lost races
    /// between advisory configuration and another shared context's activation.
    pub(super) fn debugger_registration_matches(
        &self,
        address: u64,
        setup_ttmp: bool,
    ) -> io::Result<bool> {
        let runtime = self.runtime_guard()?;
        Ok(runtime.state.enable == RuntimeState::Enabled
            && runtime
                .state
                .debugger
                .is_some_and(|active| (active.address, active.setup_ttmp) == (address, setup_ttmp)))
    }

    pub(super) fn enable_runtime_locked(&self, runtime: &mut RuntimeGuard<'_>) -> io::Result<()> {
        if runtime.state.context == ContextState::SelectionUncertain {
            return Err(invalid_data(
                "KFD context selection outcome requires teardown",
            ));
        }
        match runtime.state.enable {
            RuntimeState::Enabled => return Ok(()),
            RuntimeState::CleanupRequired => {
                return Err(invalid_data(
                    "KFD runtime activation outcome requires teardown",
                ));
            }
            RuntimeState::Disabled | RuntimeState::EnableInterrupted => {}
        }
        let result = self.call(Call::RuntimeEnable(&mut uapi::RuntimeEnable {
            mode_mask: 1
                | (u32::from(
                    runtime
                        .state
                        .debugger
                        .is_some_and(|record| record.setup_ttmp),
                ) << 1),
            r_debug: runtime.state.debugger.map_or(0, |record| record.address),
            ..uapi::RuntimeEnable::default()
        }));
        runtime.state.enable = match &result {
            Ok(()) => RuntimeState::Enabled,
            Err(source) if source.raw_os_error() == Some(errno::EINTR) => {
                RuntimeState::EnableInterrupted
            }
            Err(source) if matches!(source.raw_os_error(), Some(errno::EBUSY | errno::EEXIST)) => {
                RuntimeState::Disabled
            }
            Err(_) => RuntimeState::CleanupRequired,
        };
        result
    }

    pub(super) fn disable_runtime(&self) -> io::Result<()> {
        let mut runtime = self.runtime_guard()?;
        if runtime.state.enable == RuntimeState::Disabled {
            runtime.state.debugger = None;
            return Ok(());
        }
        let result = self.call(Call::RuntimeEnable(&mut uapi::RuntimeEnable::default()));
        runtime.state.enable = if result.is_ok() {
            runtime.state.debugger = None;
            RuntimeState::Disabled
        } else {
            RuntimeState::CleanupRequired
        };
        result
    }

    #[cfg(test)]
    pub(super) fn mark_runtime_enabled_for_test(&mut self) {
        let state = self
            .runtime
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.enable = RuntimeState::Enabled;
    }

    /// The output remains with the caller on error, including a handle returned
    /// before a later failure. Exactly one supported backing kind is required.
    pub(super) fn allocate(&self, args: &mut uapi::AllocMemory) -> io::Result<()> {
        let permitted = uapi::VRAM
            | uapi::GTT
            | uapi::USERPTR
            | uapi::WRITABLE
            | uapi::EXECUTABLE
            | uapi::PUBLIC
            | uapi::NO_SUBSTITUTE
            | uapi::COHERENT
            | uapi::UNCACHED
            | uapi::EXT_COHERENT
            | uapi::CONTIGUOUS;
        let kind = args.flags & (uapi::VRAM | uapi::GTT | uapi::USERPTR);
        if args.flags & !permitted != 0 || !matches!(kind, uapi::VRAM | uapi::GTT | uapi::USERPTR) {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::Allocate(args))
    }

    /// Creates non-substitutable writable VRAM at an address owned by the
    /// process scratch pool.
    pub(super) fn allocate_scratch(&self, args: &mut uapi::AllocMemory) -> io::Result<()> {
        if args.va == 0
            || args.size == 0
            || args.va % page_size()? as u64 != 0
            || args.size % page_size()? as u64 != 0
            || args.flags != (uapi::VRAM | uapi::WRITABLE | uapi::NO_SUBSTITUTE)
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::Allocate(args))
    }

    pub(super) fn allocate_mmio(&self, args: &mut uapi::AllocMemory) -> io::Result<()> {
        let page = page_size()? as u64;
        if args.gpu_id == 0
            || args.va == 0
            || args.va % page != 0
            || args.size != page
            || args.mmap_offset != 0
            || args.flags != (uapi::MMIO_REMAP | uapi::WRITABLE | uapi::COHERENT)
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::Allocate(args))
    }

    /// Programs the 64 KiB-granular physical scratch backing base for one GPU.
    pub(super) fn set_scratch_backing_va(&self, gpu_id: u32, address: u64) -> io::Result<()> {
        const SCRATCH_BASE_ALIGNMENT: u64 = 64 * 1024;
        if gpu_id == 0 || address == 0 || address % SCRATCH_BASE_ALIGNMENT != 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::SetScratchBackingVa(&mut uapi::SetScratchBackingVa {
            va_address: address >> 16,
            gpu_id,
            pad: 0,
        }))
    }

    pub(super) fn set_trap_handler(
        &self,
        gpu_id: u32,
        handler_address: u64,
        memory_address: u64,
    ) -> io::Result<()> {
        if gpu_id == 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::SetTrapHandler(&mut uapi::SetTrapHandler {
            tba_address: handler_address,
            tma_address: memory_address,
            gpu_id,
            pad: 0,
        }))
    }

    /// Allocates the process/device doorbell slice at one chosen GPU VA.
    /// The returned handle must be mapped to every device that will ring it.
    pub(super) fn allocate_doorbells(&self, args: &mut uapi::AllocMemory) -> io::Result<()> {
        let required = uapi::DOORBELL | uapi::WRITABLE | uapi::COHERENT | uapi::NO_SUBSTITUTE;
        if args.va == 0 || args.size == 0 || args.flags != required {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::Allocate(args))
    }

    pub(super) fn free(&self, handle: u64) -> io::Result<()> {
        self.call(Call::Free(&mut uapi::FreeMemory { handle }))
    }

    pub(super) fn dma_buf_info(&self, descriptor: i32) -> io::Result<DmaBufDetails> {
        let descriptor =
            u32::try_from(descriptor).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut args = uapi::DmaBufInfo {
            descriptor,
            ..uapi::DmaBufInfo::default()
        };
        let first = self.call(Call::DmaBufInfo(&mut args, &mut []));
        if args.metadata_size == 0 {
            first?;
            return Ok(DmaBufDetails {
                info: args,
                metadata: Vec::new(),
            });
        }
        let count = usize::try_from(args.metadata_size).map_err(invalid_data)?;
        let mut metadata = crate::memory::storage::try_vec_with_capacity(count)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for _ in 0..count {
            metadata
                .try_push(0)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        args.size = 0;
        args.gpu_id = 0;
        args.flags = 0;
        self.call(Call::DmaBufInfo(&mut args, &mut metadata))?;
        let metadata_size = usize::try_from(args.metadata_size).map_err(invalid_data)?;
        if metadata_size > metadata.len() {
            return Err(invalid_data("KFD returned oversized DMA-BUF metadata"));
        }
        metadata.truncate(metadata_size);
        Ok(DmaBufDetails {
            info: args,
            metadata,
        })
    }

    pub(super) fn import_dma_buf(&self, args: &mut uapi::ImportDmaBuf) -> io::Result<()> {
        self.call(Call::ImportDmaBuf(args))
    }

    pub(super) fn export_dma_buf(&self, handle: u64) -> io::Result<File> {
        let mut args = uapi::ExportDmaBuf {
            handle,
            flags: libc::O_CLOEXEC as u32,
            ..uapi::ExportDmaBuf::default()
        };
        let result = self.call(Call::ExportDmaBuf(&mut args));
        if let Err(error) = result {
            if args.descriptor != u32::MAX {
                if let Ok(descriptor) = i32::try_from(args.descriptor) {
                    // A changed output descriptor models a kernel-created
                    // descriptor. Consume it exactly once before returning failure.
                    let _ = close_descriptor(descriptor);
                }
            }
            return Err(error);
        }
        let descriptor = i32::try_from(args.descriptor)
            .map_err(|_| invalid_data("KFD returned an invalid DMA-BUF descriptor"))?;
        // SAFETY: Successful export transfers one newly installed descriptor.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }

    pub(super) fn export_ipc_handle(
        &self,
        handle: u64,
        gpu_id: u32,
        flags: u32,
    ) -> io::Result<[u32; 4]> {
        if handle == 0 || gpu_id == 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut args = uapi::IpcExportHandle {
            handle,
            gpu_id,
            flags,
            ..uapi::IpcExportHandle::default()
        };
        self.call(Call::IpcExportHandle(&mut args))?;
        if args.share_handle == [0; 4] {
            return Err(invalid_data("KFD returned an empty IPC share handle"));
        }
        Ok(args.share_handle)
    }

    pub(super) fn import_ipc_handle(&self, args: &mut uapi::IpcImportHandle) -> io::Result<()> {
        if args.va_addr == 0 || args.share_handle == [0; 4] || args.gpu_id == 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::IpcImportHandle(args))
    }

    pub(super) fn svm_attributes(
        &self,
        address: u64,
        size: u64,
        operation: u32,
        attributes: &mut [uapi::SvmAttribute],
    ) -> io::Result<()> {
        let page_size = page_size()? as u64;
        let attribute_count = u32::try_from(attributes.len())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        if address == 0
            || size == 0
            || address % page_size != 0
            || size % page_size != 0
            || !matches!(operation, uapi::SVM_OP_SET_ATTR | uapi::SVM_OP_GET_ATTR)
            || uapi::svm_request(attributes.len()).is_none()
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut args = uapi::SvmArgs {
            start_address: address,
            size,
            operation,
            attribute_count,
        };
        loop {
            // Legacy SVM validation can race with MMU invalidation or splitting.
            // Like libhsakmt, retry those requests with the same inputs/outputs.
            // EINTR remains visible to the owner; generic ioctls are not replayed.
            match self.call(Call::Svm(&mut args, attributes)) {
                Err(source)
                    if matches!(operation, uapi::SVM_OP_SET_ATTR | uapi::SVM_OP_GET_ATTR)
                        && source.raw_os_error() == Some(errno::EAGAIN) => {}
                result => return result,
            }
        }
    }

    pub(super) fn spm(&self, args: &mut uapi::Spm) -> io::Result<()> {
        if args.gpu_id == 0
            || !matches!(
                args.operation,
                uapi::SPM_OP_ACQUIRE | uapi::SPM_OP_RELEASE | uapi::SPM_OP_SET_DESTINATION
            )
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::Spm(args))
    }

    /// The kernel may update both the records and `num_sample_info`, including
    /// when a query returns ENOSPC. The slice stays borrowed through the ioctl.
    pub(super) fn pc_sample(
        &self,
        args: &mut uapi::PcSampleArgs,
        info: &mut [uapi::PcSampleInfo],
    ) -> io::Result<()> {
        let valid = match args.operation {
            uapi::PC_SAMPLE_QUERY_CAPABILITIES => args.trace_id == 0,
            uapi::PC_SAMPLE_CREATE => {
                args.trace_id == 0
                    && info.len() == 1
                    && info[0].interval != 0
                    && matches!(
                        info[0].method,
                        uapi::PC_SAMPLE_METHOD_HOST_TRAP | uapi::PC_SAMPLE_METHOD_STOCHASTIC
                    )
                    && info[0].unit <= uapi::PC_SAMPLE_UNIT_INSTRUCTIONS
            }
            uapi::PC_SAMPLE_DESTROY | uapi::PC_SAMPLE_START | uapi::PC_SAMPLE_STOP => {
                args.trace_id != 0 && info.is_empty() && args.flags == 0
            }
            _ => false,
        };
        if args.gpu_id == 0 || !valid {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::PcSample(args, info))
    }

    /// A successful ioctl replaces the first 16 input bytes with its output.
    /// An error leaves progress unknown because those bytes may still be input.
    pub(super) fn ais(&self, input: uapi::AisInput) -> io::Result<uapi::AisOutput> {
        if input.handle == 0
            || input.size == 0
            || input.size > crate::memory::interop::linux::AIS_MAX_TRANSFER_BYTES
            || input.descriptor < 0
            || input.file_offset < 0
            || input
                .file_offset
                .checked_add(i64::try_from(input.size).map_err(invalid_data)?)
                .is_none()
            || !matches!(input.operation, uapi::AIS_READ | uapi::AIS_WRITE)
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut args = uapi::AisArgs::new(input);
        self.call(Call::Ais(&mut args))?;
        let output = args.completed_output();
        if output.size_copied > input.size || output.status > 0 {
            return Err(invalid_data("KFD returned invalid AIS progress or status"));
        }
        Ok(output)
    }

    pub(super) fn create_queue(&self, args: &mut uapi::CreateQueue) -> io::Result<()> {
        self.call(Call::CreateQueue(args))
    }

    pub(super) fn destroy_queue(&self, queue_id: u32) -> io::Result<()> {
        self.call(Call::DestroyQueue(&mut uapi::DestroyQueue {
            queue_id,
            pad: 0,
        }))
    }

    pub(super) fn alloc_queue_gws(&self, queue_id: u32) -> io::Result<()> {
        let mut args = uapi::AllocQueueGws {
            queue_id,
            num_gws: 1,
            ..uapi::AllocQueueGws::default()
        };
        self.call(Call::AllocQueueGws(&mut args))
    }

    pub(super) fn update_queue(&self, args: &mut uapi::UpdateQueue) -> io::Result<()> {
        self.call(Call::UpdateQueue(args))
    }

    pub(super) fn set_cu_mask(&self, queue_id: u32, mask: &[u32]) -> io::Result<()> {
        if mask.is_empty() || mask.len() > 32 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let count = u32::try_from(mask.len())
            .ok()
            .and_then(|count| count.checked_mul(32))
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        self.call(Call::SetCuMask(
            &mut uapi::SetCuMask {
                queue_id,
                count,
                mask: mask.as_ptr() as u64,
            },
            mask,
        ))
    }

    /// KFD exposes one MMIO slice for this device and process. Its native owner
    /// serializes the first mapping and retains it for every queue using it.
    pub(super) fn map_doorbells(
        &self,
        offset: u64,
        size: usize,
        bounds: (u64, u64),
    ) -> io::Result<Reservation> {
        self.check_process()?;
        let mut mapping = Reservation::new(size, page_size()?, bounds, false)?;
        // KFD encodes the mapping type in bits 63:62. This is an opaque mmap
        // token, not a nonnegative render-file position. Preserve all bits when
        // passing it through the signed off_t parameter in the libc ABI.
        mapping.map_kfd_file_in_place(
            self.file
                .as_ref()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?,
            i64::from_ne_bytes(offset.to_ne_bytes()),
        )?;
        Ok(mapping)
    }

    pub(super) fn map_mmio(&self, mapping: &mut Reservation, offset: u64) -> io::Result<()> {
        self.check_process()?;
        mapping.map_kfd_file_in_place(
            self.file
                .as_ref()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?,
            i64::from_ne_bytes(offset.to_ne_bytes()),
        )
    }

    /// Preserve the completed prefix even when final synchronization fails.
    /// The owner must retry with the same device list and this updated prefix.
    pub(super) fn transfer(
        &self,
        handle: u64,
        devices: &[u32],
        completed: &mut u32,
        map: bool,
    ) -> io::Result<()> {
        let count = u32::try_from(devices.len()).map_err(invalid_data)?;
        if count == 0 || *completed > count {
            return Err(invalid_data("invalid KFD mapping retry prefix"));
        }
        let mut args = uapi::MapMemory {
            handle,
            count,
            success: *completed,
            ..uapi::MapMemory::default()
        };
        let result = self.call(if map {
            Call::Map(&mut args, devices)
        } else {
            Call::Unmap(&mut args, devices)
        });
        let before = *completed;
        *completed = args.success;
        if args.success < before
            || args.success > count
            || (result.is_ok() && args.success != count)
        {
            return Err(invalid_data("KFD returned an inconsistent mapping prefix"));
        }
        result
    }

    pub(super) fn create_exception_event(
        &self,
        event_type: u32,
        args: &mut uapi::CreateEvent,
    ) -> io::Result<()> {
        if !matches!(event_type, uapi::HW_EXCEPTION | uapi::MEMORY_EXCEPTION) {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        *args = uapi::CreateEvent {
            event_type,
            ..uapi::CreateEvent::default()
        };
        self.call(Call::CreateEvent(args))
    }

    pub(super) fn create_signal_event(
        &self,
        event_page_handle: Option<u64>,
        args: &mut uapi::CreateEvent,
        page_offered: &mut bool,
    ) -> io::Result<()> {
        *page_offered = false;
        if event_page_handle == Some(0) {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        *args = uapi::CreateEvent {
            page_offset: event_page_handle.unwrap_or(0),
            event_type: uapi::SIGNAL_EVENT,
            auto_reset: 1,
            ..uapi::CreateEvent::default()
        };
        self.call_with_page_offer(
            Call::CreateEvent(args),
            event_page_handle.map(|_| page_offered),
        )
    }

    pub(super) fn destroy_event(&self, event_id: u32) -> io::Result<()> {
        self.call(Call::DestroyEvent(&mut uapi::DestroyEvent {
            event_id,
            pad: 0,
        }))
    }

    pub(super) fn reset_event(&self, event_id: u32) -> io::Result<()> {
        self.call(Call::ResetEvent(&mut uapi::ResetEvent { event_id, pad: 0 }))
    }

    pub(super) fn set_event(&self, event_id: u32) -> io::Result<()> {
        self.call(Call::SetEvent(&mut uapi::ResetEvent { event_id, pad: 0 }))
    }

    pub(super) fn wait_signal_event(
        &self,
        event_id: u32,
        last_age: u64,
        memory_exception_id: u32,
        hardware_exception_id: u32,
        timeout_ms: u32,
    ) -> io::Result<Option<(u64, bool)>> {
        let mut events = [
            uapi::EventData {
                payload: [last_age, 0, 0, 0],
                event_id,
                ..uapi::EventData::default()
            },
            uapi::EventData {
                payload: [u64::from(u32::MAX), 0, 0, 0],
                event_id: memory_exception_id,
                ..uapi::EventData::default()
            },
            uapi::EventData {
                payload: [u64::from(u32::MAX), 0, 0, 0],
                event_id: hardware_exception_id,
                ..uapi::EventData::default()
            },
        ];
        let mut args = uapi::WaitEvents {
            count: 3,
            timeout: timeout_ms,
            ..uapi::WaitEvents::default()
        };
        self.call(Call::WaitSignal(&mut args, &mut events))?;
        match args.result {
            uapi::WAIT_COMPLETE => Ok(Some((
                events[0].payload[0],
                events[1..]
                    .iter()
                    .any(|event| event.payload[0] & u64::from(u32::MAX) != u64::from(u32::MAX)),
            ))),
            uapi::WAIT_TIMEOUT => Ok(None),
            _ => Err(invalid_data("KFD signal wait returned an unknown result")),
        }
    }

    /// Blocks until either persistent GPU exception event is signaled.
    pub(super) fn wait_exception_events(
        &self,
        hardware_event_id: u32,
        memory_event_id: u32,
        timeout_ms: u32,
    ) -> io::Result<bool> {
        let mut events = [
            uapi::EventData {
                event_id: hardware_event_id,
                ..uapi::EventData::default()
            },
            uapi::EventData {
                event_id: memory_event_id,
                ..uapi::EventData::default()
            },
        ];
        let mut args = uapi::WaitEvents {
            count: 2,
            timeout: timeout_ms,
            ..uapi::WaitEvents::default()
        };
        self.call(Call::WaitExceptions(&mut args, &mut events))?;
        match args.result {
            uapi::WAIT_COMPLETE => Ok(true),
            uapi::WAIT_TIMEOUT => Ok(false),
            _ => Err(invalid_data(
                "KFD exception wait returned an unknown result",
            )),
        }
    }

    /// Exception events are created without auto-reset. The connection owner
    /// snapshots each payload, then explicitly resets the native event while
    /// serializing all observers of that connection.
    fn poll_exception(&self, event_id: u32) -> io::Result<Option<uapi::EventData>> {
        let mut event = uapi::EventData {
            event_id,
            ..uapi::EventData::default()
        };
        let mut args = uapi::WaitEvents {
            count: 1,
            ..uapi::WaitEvents::default()
        };
        self.call(Call::Wait(&mut args, &mut event))?;
        match args.result {
            uapi::WAIT_TIMEOUT => Ok(None),
            uapi::WAIT_COMPLETE => Ok(Some(event)),
            _ => Err(invalid_data("KFD exception wait did not complete")),
        }
    }

    /// Queries a persistent manual-reset event with timeout zero, copying its
    /// native words without decoding or resetting it. The event owner
    /// serializes this read with observation.
    pub(super) fn exception_payload(&self, event_id: u32) -> io::Result<Option<[u64; 4]>> {
        self.poll_exception(event_id)
            .map(|event| event.map(|event| event.payload))
    }

    pub(super) fn hardware_exception(
        &self,
        event_id: u32,
    ) -> io::Result<Option<HardwareException>> {
        self.poll_exception(event_id).map(|event| {
            event.map(|event| HardwareException {
                gpu_id: payload_u32(&event, 3),
                reset_type: payload_u32(&event, 0),
                memory_lost: payload_u32(&event, 2) != 0,
                reset_cause: payload_u32(&event, 1),
            })
        })
    }

    pub(super) fn memory_exception(&self, event_id: u32) -> io::Result<Option<MemoryException>> {
        self.poll_exception(event_id).map(|event| {
            event.map(|event| MemoryException {
                not_present: payload_u32(&event, 0),
                read_only: payload_u32(&event, 1),
                no_execute: payload_u32(&event, 2),
                imprecise: payload_u32(&event, 3),
                address: event.payload[2],
                gpu_id: payload_u32(&event, 6),
                error_type: payload_u32(&event, 7),
            })
        })
    }
}

/// Decoded hardware exception reported by KFD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HardwareException {
    /// Native GPU that reported the exception.
    pub(super) gpu_id: u32,
    /// Native reset scope encoding.
    pub(super) reset_type: u32,
    /// Whether device memory was reported lost.
    pub(super) memory_lost: bool,
    /// Native reset cause encoding.
    pub(super) reset_cause: u32,
}

/// Decoded memory-exception information returned by KFD event polling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MemoryException {
    /// Accessed page was absent or privileged.
    pub(super) not_present: u32,
    /// Access attempted to write read-only memory.
    pub(super) read_only: u32,
    /// Access attempted to execute non-executable memory.
    pub(super) no_execute: u32,
    /// Fault address may be imprecise.
    pub(super) imprecise: u32,
    /// Reported GPU virtual address.
    pub(super) address: u64,
    /// Native GPU that reported the fault.
    pub(super) gpu_id: u32,
    /// Native exception category.
    pub(super) error_type: u32,
}

fn payload_u32(event: &uapi::EventData, index: usize) -> u32 {
    let bytes = event.payload[index / 2].to_ne_bytes();
    let offset = index % 2 * 4;
    u32::from_ne_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
#[path = "tests/connection.rs"]
mod tests;
