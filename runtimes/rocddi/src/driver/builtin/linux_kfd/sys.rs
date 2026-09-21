//! The audited Linux call boundary for KFD memory and queues.
//!
//! KFD calls are synchronous: the kernel copies the ioctl body and any pointed-to
//! arrays before returning. This module owns those buffers for the whole call,
//! initializes reserved fields, and preserves output even when errno is set.
//! Retrying a side-effecting call is the allocation owner's decision; this
//! boundary never silently retries `EINTR` or discards a successful prefix.
//!
//! Mapping owners expose numeric addresses, not Rust references. Replacing a
//! mapping is permitted only within an address reservation owned by this module.
//! The allocation owner must finish native GPU cleanup before releasing that
//! reservation. An inherited owner never issues ioctls or unmaps in a fork child.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_long, c_ulong, c_void};
use std::fs::File;
use std::io;
use std::mem::{size_of, size_of_val};
use std::os::fd::{AsRawFd, FromRawFd};
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use super::uapi;
use super::util::{check_process, close_descriptor, close_file, page_size};
use crate::session::SessionLifetime;

/// DMA-BUF extent and metadata copied out of one synchronous KFD query.
pub(super) struct DmaBufDetails {
    pub(super) info: uapi::DmaBufInfo,
    pub(super) metadata: crate::host_storage::Buffer<u8>,
}

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn madvise(address: *mut c_void, length: usize, advice: c_int) -> c_int;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
    fn syscall(number: c_long, ...) -> c_long;
}

const PROT_NONE: c_int = 0;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const PROT_READ_WRITE: c_int = 3;
const MAP_SHARED: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MAP_FIXED: c_int = 0x10;
const MAP_FIXED_NOREPLACE: c_int = 0x10_0000;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_NORESERVE: c_int = 0x4000;
const MADV_DONTFORK: c_int = 10;
const O_CLOEXEC: u32 = 0x0008_0000;

/// The typed call also keeps indirect buffers borrowed during fault injection.
/// Test doubles can inspect and modify outputs without following raw pointers.
/// Production dispatch goes straight from this match to the kernel ioctl.
pub(super) enum Call<'a> {
    Version(&'a mut uapi::Version),
    ClockCounters(&'a mut uapi::ClockCounters),
    AvailableMemory(&'a mut uapi::AvailableMemory),
    Apertures(&'a mut uapi::Apertures, &'a mut [uapi::Aperture]),
    AcquireVm(&'a mut uapi::AcquireVm),
    CreateProcess(&'a mut uapi::CreateProcess),
    RuntimeEnable(&'a mut uapi::RuntimeEnable),
    Allocate(&'a mut uapi::AllocMemory),
    Free(&'a mut uapi::FreeMemory),
    Map(&'a mut uapi::MapMemory, &'a [u32]),
    Unmap(&'a mut uapi::MapMemory, &'a [u32]),
    DmaBufInfo(&'a mut uapi::DmaBufInfo, &'a mut [u8]),
    ImportDmaBuf(&'a mut uapi::ImportDmaBuf),
    ExportDmaBuf(&'a mut uapi::ExportDmaBuf),
    IpcImportHandle(&'a mut uapi::IpcImportHandle),
    IpcExportHandle(&'a mut uapi::IpcExportHandle),
    Svm(&'a mut uapi::SvmArgs, &'a mut [uapi::SvmAttribute]),
    Spm(&'a mut uapi::Spm),
    PcSampling(&'a mut uapi::PcSample, &'a mut [uapi::PcSampleInfo]),
    CreateEvent(&'a mut uapi::CreateEvent),
    DestroyEvent(&'a mut uapi::DestroyEvent),
    Wait(&'a mut uapi::WaitEvents, &'a mut uapi::EventData),
    SetScratchBackingVa(&'a mut uapi::SetScratchBackingVa),
    SetTrapHandler(&'a mut uapi::SetTrapHandler),
    CreateQueue(&'a mut uapi::CreateQueue),
    DestroyQueue(&'a mut uapi::DestroyQueue),
    UpdateQueue(&'a mut uapi::UpdateQueue),
    SetCuMask(&'a mut uapi::SetCuMask, &'a [u32]),
}

#[cfg(test)]
pub(super) type IoctlHook = std::sync::Arc<dyn Fn(&mut Call<'_>) -> io::Result<()> + Send + Sync>;

/// An owned KFD endpoint. Kernel calls may run concurrently; operation owners
/// supply their own buffers and serialize mutations of their own resources.
/// Runtime enable and disable are separately serialized here so no session can
/// publish a partially enabled process state.
pub(super) struct Kfd {
    allocator: crate::host_storage::Allocator,
    file: Option<File>,
    process: u32,
    runtime: Mutex<RuntimeControl>,
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

struct RuntimeControl {
    context: ContextState,
    enable: RuntimeState,
}

/// Guard retaining exclusive access to one runtime-enable transaction.
pub(super) struct RuntimeGuard<'a> {
    state: MutexGuard<'a, RuntimeControl>,
}

impl Kfd {
    pub(super) fn new(file: File, allocator: crate::host_storage::Allocator) -> Self {
        Self {
            allocator,
            file: Some(file),
            process: std::process::id(),
            runtime: Mutex::new(RuntimeControl {
                context: ContextState::Primary,
                enable: RuntimeState::Disabled,
            }),
            #[cfg(test)]
            hook: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_hook(file: File, hook: IoctlHook) -> Self {
        Self {
            allocator: crate::host_storage::Allocator::default(),
            file: Some(file),
            process: std::process::id(),
            runtime: Mutex::new(RuntimeControl {
                context: ContextState::Primary,
                enable: RuntimeState::Disabled,
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
        check_process(self.process)
    }

    fn call(&self, mut call: Call<'_>) -> io::Result<()> {
        self.check_process()?;
        let descriptor = self
            .file
            .as_ref()
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?
            .as_raw_fd();
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            return hook(&mut call);
        }
        if let Call::Svm(args, attributes) = &mut call {
            return self.call_svm(descriptor, args, attributes);
        }
        let (request, body) = match &mut call {
            Call::Version(args) => (uapi::GET_VERSION, ptr::from_mut(*args).cast::<c_void>()),
            Call::ClockCounters(args) => (uapi::GET_CLOCK_COUNTERS, ptr::from_mut(*args).cast()),
            Call::AvailableMemory(args) => {
                (uapi::GET_AVAILABLE_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::Apertures(args, entries) => {
                args.pointer = if entries.is_empty() {
                    0
                } else {
                    entries.as_mut_ptr() as u64
                };
                (uapi::GET_APERTURES, ptr::from_mut(*args).cast())
            }
            Call::AcquireVm(args) => (uapi::ACQUIRE_VM, ptr::from_mut(*args).cast()),
            Call::CreateProcess(args) => (uapi::CREATE_PROCESS, ptr::from_mut(*args).cast()),
            Call::RuntimeEnable(args) => (uapi::RUNTIME_ENABLE, ptr::from_mut(*args).cast()),
            Call::Allocate(args) => (uapi::ALLOC_MEMORY, ptr::from_mut(*args).cast()),
            Call::Free(args) => (uapi::FREE_MEMORY, ptr::from_mut(*args).cast()),
            Call::Map(args, devices) => {
                args.devices = devices.as_ptr() as u64;
                (uapi::MAP_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::Unmap(args, devices) => {
                args.devices = devices.as_ptr() as u64;
                (uapi::UNMAP_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::DmaBufInfo(args, metadata) => {
                args.metadata = if metadata.is_empty() {
                    0
                } else {
                    metadata.as_mut_ptr() as u64
                };
                args.metadata_size = u32::try_from(metadata.len()).map_err(invalid_data)?;
                (uapi::GET_DMABUF_INFO, ptr::from_mut(*args).cast())
            }
            Call::ImportDmaBuf(args) => (uapi::IMPORT_DMABUF, ptr::from_mut(*args).cast()),
            Call::ExportDmaBuf(args) => (uapi::EXPORT_DMABUF, ptr::from_mut(*args).cast()),
            Call::IpcImportHandle(args) => (uapi::IPC_IMPORT_HANDLE, ptr::from_mut(*args).cast()),
            Call::IpcExportHandle(args) => (uapi::IPC_EXPORT_HANDLE, ptr::from_mut(*args).cast()),
            Call::Svm(_, _) => return Err(invalid_data("SVM dispatch path was not selected")),
            Call::Spm(args) => (uapi::SPM, ptr::from_mut(*args).cast()),
            Call::PcSampling(args, sample_info) => {
                args.sample_info = if sample_info.is_empty() {
                    0
                } else {
                    sample_info.as_mut_ptr() as u64
                };
                (uapi::PC_SAMPLE, ptr::from_mut(*args).cast())
            }
            Call::CreateEvent(args) => (uapi::CREATE_EVENT, ptr::from_mut(*args).cast()),
            Call::DestroyEvent(args) => (uapi::DESTROY_EVENT, ptr::from_mut(*args).cast()),
            Call::SetScratchBackingVa(args) => {
                (uapi::SET_SCRATCH_BACKING_VA, ptr::from_mut(*args).cast())
            }
            Call::SetTrapHandler(args) => (uapi::SET_TRAP_HANDLER, ptr::from_mut(*args).cast()),
            Call::CreateQueue(args) => (uapi::CREATE_QUEUE, ptr::from_mut(*args).cast()),
            Call::DestroyQueue(args) => (uapi::DESTROY_QUEUE, ptr::from_mut(*args).cast()),
            Call::UpdateQueue(args) => (uapi::UPDATE_QUEUE, ptr::from_mut(*args).cast()),
            Call::SetCuMask(args, mask) => {
                args.mask = mask.as_ptr() as u64;
                (uapi::SET_CU_MASK, ptr::from_mut(*args).cast())
            }
            Call::Wait(args, event) => {
                args.events = ptr::from_mut(*event) as u64;
                (uapi::WAIT_EVENTS, ptr::from_mut(*args).cast())
            }
        };
        // SAFETY: Every variant supplies the repr(C) record belonging to this
        // request. Records and indirect arrays remain exclusively borrowed and
        // valid until ioctl returns; their lengths were set from actual storage.
        // The descriptor is owned here and has been checked against the PID.
        // KFD may update outputs on failure, so nothing rewrites them afterward.
        let result = unsafe { ioctl(descriptor, request as c_ulong, body) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn call_svm(
        &self,
        descriptor: c_int,
        args: &mut uapi::SvmArgs,
        attributes: &mut [uapi::SvmAttribute],
    ) -> io::Result<()> {
        let request = uapi::svm_request(attributes.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let words = 3_usize
            .checked_add(attributes.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut buffer =
            crate::host_storage::Buffer::<u64>::try_with_capacity(words, self.allocator)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for _ in 0..words {
            buffer
                .try_push(0)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        let words = buffer.as_mut_slice().as_mut_ptr();
        let body = words.cast::<u8>();
        // SAFETY: The u64 buffer is eight-byte aligned and exactly covers the
        // 24-byte header followed by every eight-byte flexible-array element.
        unsafe {
            words.cast::<uapi::SvmArgs>().write(*args);
            ptr::copy_nonoverlapping(
                attributes.as_ptr().cast::<u8>(),
                body.add(size_of::<uapi::SvmArgs>()),
                size_of_val(attributes),
            );
        }
        // SAFETY: The request size encodes this complete live buffer. KFD makes
        // one synchronous copy and may update both header and attributes.
        let result = unsafe { ioctl(descriptor, request as c_ulong, body.cast::<c_void>()) };
        // Preserve KFD outputs even when the ioctl reports an error.
        // SAFETY: Both destinations are valid exclusive borrows of the exact
        // initialized record types copied into the aligned buffer above.
        unsafe {
            *args = words.cast::<uapi::SvmArgs>().read();
            ptr::copy_nonoverlapping(
                body.add(size_of::<uapi::SvmArgs>()),
                attributes.as_mut_ptr().cast::<u8>(),
                size_of_val(attributes),
            );
        }
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
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

    pub(super) fn available_memory(&self, gpu_id: u32) -> io::Result<u64> {
        let mut memory = uapi::AvailableMemory {
            gpu_id,
            ..uapi::AvailableMemory::default()
        };
        self.call(Call::AvailableMemory(&mut memory))?;
        Ok(memory.available)
    }

    pub(super) fn apertures(&self) -> io::Result<crate::host_storage::Buffer<uapi::Aperture>> {
        let mut args = uapi::Apertures::default();
        self.call(Call::Apertures(&mut args, &mut []))?;
        let count = usize::try_from(args.count).map_err(invalid_data)?;
        let mut entries = crate::host_storage::Buffer::try_with_capacity(count, self.allocator)
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

    /// Selects the instance-owned context before its first VM or runtime call.
    /// A version or open failure leaves the primary context retryable. An ioctl
    /// failure has no portable proof of whether selection took effect, so only
    /// closing this descriptor can recover from that outcome.
    pub(super) fn prepare_context(&self, lifetime: SessionLifetime) -> io::Result<()> {
        let mut runtime = self.runtime_guard()?;
        match (lifetime, runtime.state.context) {
            (SessionLifetime::Process, ContextState::Primary)
            | (SessionLifetime::Session, ContextState::Secondary) => return Ok(()),
            (_, ContextState::SelectionUncertain) => {
                return Err(invalid_data(
                    "KFD context selection outcome requires teardown",
                ));
            }
            (SessionLifetime::Process, ContextState::Secondary) => {
                return Err(invalid_data(
                    "KFD endpoint already owns a secondary context",
                ));
            }
            (SessionLifetime::Session, ContextState::Primary) => {}
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
            mode_mask: 1,
            ..uapi::RuntimeEnable::default()
        }));
        runtime.state.enable = match &result {
            Ok(()) => RuntimeState::Enabled,
            Err(source) if source.raw_os_error() == Some(4) => RuntimeState::EnableInterrupted,
            Err(source) if matches!(source.raw_os_error(), Some(16 | 17)) => RuntimeState::Disabled,
            Err(_) => RuntimeState::CleanupRequired,
        };
        result
    }

    fn disable_runtime(&mut self) -> io::Result<()> {
        let state = self
            .runtime
            .get_mut()
            .map_err(|_| invalid_data("KFD runtime lock poisoned"))?
            .enable;
        if state == RuntimeState::Disabled {
            return Ok(());
        }
        let result = self.call(Call::RuntimeEnable(&mut uapi::RuntimeEnable::default()));
        self.runtime
            .get_mut()
            .map_err(|_| invalid_data("KFD runtime lock poisoned"))?
            .enable = if result.is_ok() {
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
                metadata: crate::host_storage::Buffer::new(self.allocator),
            });
        }
        let count = usize::try_from(args.metadata_size).map_err(invalid_data)?;
        let mut metadata = crate::host_storage::Buffer::try_with_capacity(count, self.allocator)
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
            flags: O_CLOEXEC,
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
        self.call(Call::Svm(
            &mut uapi::SvmArgs {
                start_address: address,
                size,
                operation,
                attribute_count,
            },
            attributes,
        ))
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

    pub(super) fn pc_sampling_capabilities(
        &self,
        gpu_id: u32,
    ) -> io::Result<crate::host_storage::Buffer<uapi::PcSampleInfo>> {
        if gpu_id == 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut args = uapi::PcSample {
            operation: uapi::PC_SAMPLE_OP_QUERY_CAPABILITIES,
            gpu_id,
            ..uapi::PcSample::default()
        };
        self.call(Call::PcSampling(&mut args, &mut []))?;
        let count = usize::try_from(args.sample_info_count).map_err(invalid_data)?;
        let mut configurations =
            crate::host_storage::Buffer::try_with_capacity(count, self.allocator)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for _ in 0..count {
            configurations
                .try_push(uapi::PcSampleInfo::default())
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        if count == 0 {
            return Ok(configurations);
        }
        self.call(Call::PcSampling(&mut args, &mut configurations))?;
        let returned = usize::try_from(args.sample_info_count).map_err(invalid_data)?;
        if returned > configurations.len() {
            return Err(invalid_data(
                "KFD returned more PC sampling configurations than supplied storage",
            ));
        }
        configurations.truncate(returned);
        Ok(configurations)
    }

    pub(super) fn pc_sampling_create(
        &self,
        gpu_id: u32,
        sample_info: &mut uapi::PcSampleInfo,
        trace_id: &mut u32,
    ) -> io::Result<()> {
        if gpu_id == 0
            || *trace_id != 0
            || sample_info.interval == 0
            || !matches!(
                sample_info.method,
                uapi::PC_SAMPLE_METHOD_HOSTTRAP | uapi::PC_SAMPLE_METHOD_STOCHASTIC
            )
            || !matches!(
                sample_info.sample_type,
                uapi::PC_SAMPLE_TYPE_TIME_US
                    | uapi::PC_SAMPLE_TYPE_CLOCK_CYCLES
                    | uapi::PC_SAMPLE_TYPE_INSTRUCTIONS
            )
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let mut args = uapi::PcSample {
            sample_info_count: 1,
            operation: uapi::PC_SAMPLE_OP_CREATE,
            gpu_id,
            ..uapi::PcSample::default()
        };
        let result = self.call(Call::PcSampling(
            &mut args,
            std::slice::from_mut(sample_info),
        ));
        *trace_id = args.trace_id;
        if result.is_ok() && *trace_id == 0 {
            return Err(invalid_data("KFD returned an invalid PC sampling trace ID"));
        }
        result
    }

    fn pc_sampling_control(&self, gpu_id: u32, trace_id: u32, operation: u32) -> io::Result<()> {
        if gpu_id == 0
            || trace_id == 0
            || !matches!(
                operation,
                uapi::PC_SAMPLE_OP_DESTROY | uapi::PC_SAMPLE_OP_START | uapi::PC_SAMPLE_OP_STOP
            )
        {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.call(Call::PcSampling(
            &mut uapi::PcSample {
                operation,
                gpu_id,
                trace_id,
                ..uapi::PcSample::default()
            },
            &mut [],
        ))
    }

    pub(super) fn pc_sampling_destroy(&self, gpu_id: u32, trace_id: u32) -> io::Result<()> {
        self.pc_sampling_control(gpu_id, trace_id, uapi::PC_SAMPLE_OP_DESTROY)
    }

    pub(super) fn pc_sampling_start(&self, gpu_id: u32, trace_id: u32) -> io::Result<()> {
        self.pc_sampling_control(gpu_id, trace_id, uapi::PC_SAMPLE_OP_START)
    }

    pub(super) fn pc_sampling_stop(&self, gpu_id: u32, trace_id: u32) -> io::Result<()> {
        self.pc_sampling_control(gpu_id, trace_id, uapi::PC_SAMPLE_OP_STOP)
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
        mapping.map_file(
            self.file
                .as_ref()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?,
            i64::from_ne_bytes(offset.to_ne_bytes()),
            PROT_READ_WRITE,
        )?;
        Ok(mapping)
    }

    pub(super) fn map_mmio(&self, mapping: &mut Reservation, offset: u64) -> io::Result<()> {
        self.check_process()?;
        mapping.map_file(
            self.file
                .as_ref()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?,
            i64::from_ne_bytes(offset.to_ne_bytes()),
            PROT_READ_WRITE,
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
    ) -> io::Result<()> {
        if event_page_handle == Some(0) {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        *args = uapi::CreateEvent {
            page_offset: event_page_handle.unwrap_or(0),
            event_type: uapi::SIGNAL_EVENT,
            auto_reset: 1,
            ..uapi::CreateEvent::default()
        };
        self.call(Call::CreateEvent(args))
    }

    pub(super) fn destroy_event(&self, event_id: u32) -> io::Result<()> {
        self.call(Call::DestroyEvent(&mut uapi::DestroyEvent {
            event_id,
            pad: 0,
        }))
    }

    /// Exception events are created without auto-reset. Concurrent zero-timeout
    /// observers therefore cannot consume a terminal notification.
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

    pub(super) fn hardware_memory_lost(&self, event_id: u32) -> io::Result<bool> {
        self.poll_exception(event_id)
            .map(|event| event.is_some_and(|event| payload_u32(&event, 2) != 0))
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

/// Decoded memory-exception information returned by KFD event polling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MemoryException {
    pub(super) not_present: u32,
    pub(super) read_only: u32,
    pub(super) no_execute: u32,
    pub(super) imprecise: u32,
    pub(super) address: u64,
    pub(super) gpu_id: u32,
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

pub(super) fn invalid_data(_detail: impl std::fmt::Display) -> io::Error {
    io::Error::from(io::ErrorKind::InvalidData)
}

/// Owns the entire reservation, including alignment padding. Keeping padding
/// avoids fallible partial unmaps during construction and leaves one range to
/// release after the kernel no longer has a GPU mapping of its usable extent.
pub(super) struct Reservation {
    base: usize,
    length: usize,
    address: usize,
    size: usize,
    process: u32,
    writable: bool,
    restore_on_release: bool,
    #[cfg(test)]
    release_error: Option<i32>,
}

impl Reservation {
    pub(super) fn new(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        host: bool,
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, host, false, std::process::id())
    }

    /// Use a session process ID already checked by `ensure_open` before host
    /// allocation. The retained owner still checks its process on release.
    pub(super) fn new_host_in_process(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        process: u32,
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, true, false, process)
    }

    pub(super) fn new_at(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        address: usize,
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        if address != 0
            && address % alignment == 0
            && address as u64 >= bounds.0
            && address
                .checked_add(size - 1)
                .is_some_and(|last| last as u64 <= bounds.1)
        {
            // SAFETY: MAP_FIXED_NOREPLACE either acquires this exact free range
            // or fails without replacing an existing mapping.
            let mapped = unsafe {
                mmap(
                    address as *mut c_void,
                    size,
                    PROT_NONE,
                    MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE | MAP_FIXED_NOREPLACE,
                    -1,
                    0,
                )
            };
            if mapped as usize == address {
                return Ok(Self {
                    base: address,
                    length: size,
                    address,
                    size,
                    process: std::process::id(),
                    writable: false,
                    restore_on_release: false,
                    #[cfg(test)]
                    release_error: None,
                });
            }
            if mapped as isize != -1 {
                // SAFETY: A nonfailed, unexpected mapping is owned by this
                // call and must not escape while the requested hint falls back.
                let _ = unsafe { munmap(mapped, size) };
            }
        }
        Self::new(size, alignment, bounds, false)
    }

    pub(super) fn new_shared_host(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
    ) -> io::Result<Self> {
        Self::new_with_sharing(size, alignment, bounds, true, true, std::process::id())
    }

    pub(super) fn new_aligned_host(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let length = size
            .checked_add(alignment - page)
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        // SAFETY: This creates a new writable anonymous mapping without
        // replacing existing memory. Its padding is trimmed below.
        let mapped = unsafe {
            mmap(
                ptr::null_mut(),
                length,
                PROT_READ_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE,
                -1,
                0,
            )
        };
        if mapped as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        let base = mapped as usize;
        let Some(address) = base
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
        else {
            // SAFETY: The complete mapping is still owned by this call.
            let _ = unsafe { munmap(mapped, length) };
            return Err(invalid_data(
                "mmap alignment overflows the host address width",
            ));
        };
        let last = address.checked_add(size - 1);
        if (address as u64) < bounds.0 || last.is_none_or(|last| last as u64 > bounds.1) {
            // SAFETY: The complete mapping is still owned by this call.
            let _ = unsafe { munmap(mapped, length) };
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        let prefix = address - base;
        if prefix != 0 {
            // SAFETY: This removes only the leading alignment padding.
            if unsafe { munmap(mapped, prefix) } != 0 {
                let source = io::Error::last_os_error();
                // SAFETY: The complete mapping remains live after failed munmap.
                let _ = unsafe { munmap(mapped, length) };
                return Err(source);
            }
        }
        let end = address + size;
        let mapping_end = base + length;
        if end < mapping_end {
            // SAFETY: This removes only the trailing alignment padding.
            if unsafe { munmap(end as *mut c_void, mapping_end - end) } != 0 {
                let source = io::Error::last_os_error();
                // SAFETY: The aligned extent and trailing padding remain live;
                // the prefix was already released.
                let _ = unsafe { munmap(address as *mut c_void, mapping_end - address) };
                return Err(source);
            }
        }
        Ok(Self {
            base: address,
            length: size,
            address,
            size,
            process: std::process::id(),
            writable: true,
            restore_on_release: false,
            #[cfg(test)]
            release_error: None,
        })
    }

    fn new_with_sharing(
        size: usize,
        alignment: usize,
        bounds: (u64, u64),
        host: bool,
        shared: bool,
        process: u32,
    ) -> io::Result<Self> {
        let page = page_size()?;
        if size == 0 || size % page != 0 || !alignment.is_power_of_two() || alignment < page {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let length = size
            .checked_add(alignment - page)
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let protection = if host { PROT_READ_WRITE } else { PROT_NONE };
        let flags = MAP_ANONYMOUS
            | if shared { MAP_SHARED } else { MAP_PRIVATE }
            | if host { 0 } else { MAP_NORESERVE };
        // SAFETY: This creates a new anonymous mapping without replacing any
        // existing mapping. Length is nonzero, page aligned, and representable;
        // no Rust references are created to the returned memory.
        let mapped = unsafe { mmap(ptr::null_mut(), length, protection, flags, -1, 0) };
        if mapped as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        let base = mapped as usize;
        let address = base
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1));
        let mut reservation = Self {
            base,
            length,
            address: base,
            size,
            process,
            writable: host,
            restore_on_release: false,
            #[cfg(test)]
            release_error: None,
        };
        let Some(address) = address else {
            return Err(invalid_data(
                "mmap alignment overflows the host address width",
            ));
        };
        reservation.address = address;
        let last = address.checked_add(size - 1);
        if (address as u64) < bounds.0 || last.is_none_or(|last| last as u64 > bounds.1) {
            return Err(io::Error::from(io::ErrorKind::OutOfMemory));
        }
        Ok(reservation)
    }

    /// Creates a view into a reservation retained by another owner. If a file
    /// mapping replaces this view, releasing it restores anonymous reserved VA
    /// instead of exposing a hole inside the parent range.
    pub(super) fn view(address: usize, size: usize) -> Self {
        debug_assert_ne!(address, 0);
        debug_assert_ne!(size, 0);
        Self {
            base: address,
            length: 0,
            address,
            size,
            process: std::process::id(),
            writable: false,
            restore_on_release: true,
            #[cfg(test)]
            release_error: None,
        }
    }

    pub(super) fn usable_size(&self) -> usize {
        self.size
    }

    pub(super) fn address(&self) -> usize {
        self.address
    }

    pub(super) fn dont_fork(&self) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 {
            return Err(invalid_data("address reservation is not live"));
        }
        // SAFETY: The live owned mapping covers this exact usable extent.
        // MADV_DONTFORK changes inheritance policy without creating aliases.
        let result = unsafe { madvise(self.address as *mut c_void, self.size, MADV_DONTFORK) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Map the BO over this reservation's usable extent without changing its
    /// address. Keeping the surrounding reservation prevents a failed partial
    /// cleanup from making a still-referenced GPU address available for reuse.
    pub(super) fn map_render(&mut self, render: &File, offset: u64) -> io::Result<()> {
        check_process(self.process)?;
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(render, offset, PROT_READ_WRITE)
    }

    /// Installs a render-node BO mapping without granting CPU access. KFD
    /// scratch allocations require this VMA even though userspace never reads
    /// or writes the backing through the CPU.
    pub(super) fn map_render_inaccessible(&mut self, render: &File, offset: u64) -> io::Result<()> {
        check_process(self.process)?;
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(render, offset, PROT_NONE)
    }

    pub(super) fn map_dma_buf(&mut self, dma_buf: &File) -> io::Result<()> {
        self.map_file(dma_buf, 0, PROT_READ_WRITE)
    }

    pub(super) fn map_dma_buf_with_permissions(
        &mut self,
        dma_buf: &File,
        offset: u64,
        permission_bits: u32,
    ) -> io::Result<()> {
        let protection = match permission_bits {
            0 => PROT_NONE,
            1 => PROT_READ,
            2 => PROT_WRITE,
            3 => PROT_READ_WRITE,
            _ => return Err(io::Error::from(io::ErrorKind::InvalidInput)),
        };
        let offset = i64::try_from(offset).map_err(invalid_data)?;
        self.map_file(dma_buf, offset, protection)
    }

    fn map_file(&mut self, file: &File, offset: i64, protection: c_int) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 && !self.restore_on_release {
            return Err(invalid_data("file mapping has no live address reservation"));
        }
        if offset % i64::try_from(page_size()?).map_err(invalid_data)? != 0 {
            return Err(invalid_data("KFD returned an invalid CPU mapping offset"));
        }
        // SAFETY: MAP_FIXED replaces only the aligned usable extent of our own
        // reservation. The original range is still owned and no Rust references
        // alias it. The caller keeps the allocation and render file alive.
        let mapped = unsafe {
            mmap(
                self.address as *mut c_void,
                self.size,
                protection,
                MAP_SHARED | MAP_FIXED,
                file.as_raw_fd(),
                offset,
            )
        };
        if mapped as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        if self.restore_on_release {
            self.base = self.address;
            self.length = self.size;
        }
        // SAFETY: The successful mapping covers the exact live usable extent.
        // Avoid inheriting device mappings into a fork child that cannot own
        // the corresponding KFD process resources.
        let _ = unsafe { madvise(mapped, self.size, MADV_DONTFORK) };
        self.writable = protection == PROT_READ_WRITE;
        Ok(())
    }

    /// Observe native queue progress using the transport's 64-bit indices.
    /// Read and write positions may change independently; callers that require
    /// a quiescent queue must have already stopped every producer.
    pub(super) fn read_indices(
        &self,
        read_offset: usize,
        write_offset: usize,
    ) -> io::Result<(u64, u64)> {
        self.check_write(read_offset, 8)?;
        self.check_write(write_offset, 8)?;
        if (self.address + read_offset) % 8 != 0 || (self.address + write_offset) % 8 != 0 {
            return Err(invalid_data("unaligned queue pointer page"));
        }
        // SAFETY: This live owned CPU mapping contains two initialized aligned
        // 64-bit queue indices. Access is atomic to match the caller-selected
        // publication protocol; no Rust reference to ordinary u64 is created.
        let read =
            unsafe { &*((self.address + read_offset) as *const std::sync::atomic::AtomicU64) };
        let write =
            unsafe { &*((self.address + write_offset) as *const std::sync::atomic::AtomicU64) };
        Ok((
            read.load(std::sync::atomic::Ordering::Acquire),
            write.load(std::sync::atomic::Ordering::Acquire),
        ))
    }

    /// Expand a ring-relative native read index into one stable monotonic
    /// producer window. The producer is sampled on both sides of the read so a
    /// concurrent wrap cannot select the wrong lap.
    pub(super) fn read_wrapping_indices(
        &self,
        read_offset: usize,
        write_offset: usize,
        read_mask: u64,
    ) -> io::Result<(u64, u64)> {
        self.check_write(read_offset, 8)?;
        self.check_write(write_offset, 8)?;
        if (self.address + read_offset) % 8 != 0 || (self.address + write_offset) % 8 != 0 {
            return Err(invalid_data("unaligned queue pointer page"));
        }
        // SAFETY: This live owned CPU mapping contains two initialized aligned
        // 64-bit queue indices. Device and producer accesses use the same
        // atomic-width protocol, and no ordinary u64 reference is created.
        let read =
            unsafe { &*((self.address + read_offset) as *const std::sync::atomic::AtomicU64) };
        let write =
            unsafe { &*((self.address + write_offset) as *const std::sync::atomic::AtomicU64) };
        loop {
            let first_write = write.load(std::sync::atomic::Ordering::Acquire);
            let native_read = read.load(std::sync::atomic::Ordering::Acquire);
            let second_write = write.load(std::sync::atomic::Ordering::Acquire);
            if first_write == second_write {
                let consumed =
                    first_write.wrapping_sub(first_write.wrapping_sub(native_read) & read_mask);
                return Ok((consumed, first_write));
            }
            std::hint::spin_loop();
        }
    }

    pub(super) fn zero(&mut self) -> io::Result<()> {
        self.check_write(0, self.size)?;
        // SAFETY: The checked live mapping covers exactly size writable bytes.
        // Queue construction has not published it and creates no Rust aliases.
        unsafe { ptr::write_bytes(self.address as *mut u8, 0, self.size) };
        Ok(())
    }

    pub(super) fn write_bytes(&mut self, offset: usize, bytes: &[u8]) -> io::Result<()> {
        self.check_write(offset, bytes.len())?;
        // SAFETY: Both extents are valid and disjoint: input is an ordinary
        // Rust slice, while this private mapping exposes no Rust references.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.address + offset) as *mut u8,
                bytes.len(),
            );
        };
        Ok(())
    }

    pub(super) fn fill_records(&mut self, record: &[u8], count: usize) -> io::Result<()> {
        let size = record
            .len()
            .checked_mul(count)
            .ok_or_else(|| invalid_data("record initialization overflows"))?;
        self.check_write(0, size)?;
        for index in 0..count {
            // SAFETY: The entire destination was checked once above, and each
            // disjoint record occupies its own part of that private mapping.
            unsafe {
                ptr::copy_nonoverlapping(
                    record.as_ptr(),
                    (self.address + index * record.len()) as *mut u8,
                    record.len(),
                );
            };
        }
        Ok(())
    }

    fn check_write(&self, offset: usize, length: usize) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0
            || !self.writable
            || offset.checked_add(length).is_none_or(|end| end > self.size)
        {
            return Err(invalid_data("write exceeds live CPU-visible reservation"));
        }
        Ok(())
    }

    pub(super) fn release(&mut self) -> io::Result<()> {
        check_process(self.process)?;
        if self.length == 0 {
            return Ok(());
        }
        #[cfg(test)]
        if let Some(errno) = self.release_error.take() {
            return Err(io::Error::from_raw_os_error(errno));
        }
        if self.restore_on_release {
            // SAFETY: This exact subrange replaced part of a retained parent
            // reservation. Native GPU cleanup is complete, so replace the BO
            // mapping with inaccessible anonymous VA for future scratch reuse.
            let mapped = unsafe {
                mmap(
                    self.base as *mut c_void,
                    self.length,
                    PROT_NONE,
                    MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE | MAP_FIXED,
                    -1,
                    0,
                )
            };
            if mapped as isize == -1 {
                return Err(io::Error::last_os_error());
            }
        } else {
            // SAFETY: This owner holds the entire live, page-aligned range.
            // Native GPU cleanup must have completed before release.
            let result = unsafe { munmap(self.base as *mut c_void, self.length) };
            if result != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        self.length = 0;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn fail_release_once(&mut self, errno: i32) {
        self.release_error = Some(errno);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // The resource owner forgets this reservation if GPU cleanup fails.
        // Reaching this destructor means only CPU address ownership remains.
        if self.length != 0 || self.restore_on_release {
            let _ = self.release();
        }
    }
}

#[cfg(test)]
#[path = "tests/sys.rs"]
mod tests;

/// Enumerates numeric Linux sysfs directories without an allocator-owned path
/// or libc directory buffer. The caller decides where durable records live.
pub(super) fn numeric_directories(
    path: &std::path::Path,
    visitor: &mut dyn FnMut(u32) -> Result<(), crate::Error>,
) -> Result<(), crate::Error> {
    #[cfg(target_arch = "x86_64")]
    const GETDENTS64: c_long = 217;
    #[cfg(target_arch = "aarch64")]
    const GETDENTS64: c_long = 61;
    let directory = File::open(path)
        .map_err(|source| super::memory::native_error("sysfs directory open", source))?;
    let mut storage = [0u8; 4096];
    loop {
        // SAFETY: getdents64 receives a live directory descriptor and a writable
        // byte array of exactly the advertised length. Records are decoded as
        // bytes below, without assuming alignment of their kernel layout.
        let result = unsafe {
            syscall(
                GETDENTS64,
                directory.as_raw_fd(),
                storage.as_mut_ptr(),
                storage.len(),
            )
        };
        if result < 0 {
            return Err(super::memory::native_error(
                "sysfs directory read",
                io::Error::last_os_error(),
            ));
        }
        if result == 0 {
            return Ok(());
        }
        let length = usize::try_from(result).map_err(|_| {
            super::memory::error(crate::ErrorKind::InvalidData, "invalid directory length")
        })?;
        if length > storage.len() {
            return Err(super::memory::error(
                crate::ErrorKind::InvalidData,
                "oversized directory response",
            ));
        }
        let mut offset = 0;
        while offset < length {
            let record = &storage[offset..length];
            if record.len() < 20 {
                return Err(super::memory::error(
                    crate::ErrorKind::InvalidData,
                    "truncated directory record",
                ));
            }
            let size = usize::from(u16::from_ne_bytes([record[16], record[17]]));
            if size < 20 || size > record.len() {
                return Err(super::memory::error(
                    crate::ErrorKind::InvalidData,
                    "invalid directory record",
                ));
            }
            let name = &record[19..size];
            let end = name.iter().position(|byte| *byte == 0).ok_or_else(|| {
                super::memory::error(crate::ErrorKind::InvalidData, "unterminated directory name")
            })?;
            let name = &name[..end];
            if !name.is_empty() && name.iter().all(u8::is_ascii_digit) {
                let mut number = 0u32;
                for byte in name {
                    number = number
                        .checked_mul(10)
                        .and_then(|n| n.checked_add(u32::from(byte - b'0')))
                        .ok_or_else(|| {
                            super::memory::error(
                                crate::ErrorKind::InvalidData,
                                "directory ordinal overflow",
                            )
                        })?;
                }
                visitor(number)?;
            }
            offset += size;
        }
    }
}
