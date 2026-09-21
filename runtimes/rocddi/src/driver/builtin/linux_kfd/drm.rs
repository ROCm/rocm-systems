//! Minimal DRM calls needed for KFD-bound VM mappings and command submission.
//!
//! This is a deliberately narrow raw-ioctl boundary, not a general libdrm
//! replacement. Every wrapper owns the exact request layout, validates kernel
//! outputs before publication, and leaves retry or rollback policy to the
//! higher-level virtual-memory owner. GEM handles and timeline sync objects are
//! process-local resources and must be closed through the same render file that
//! created or imported them.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_ulong, c_void};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::ptr;

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
}

const fn request(direction: u32, number: u32, size: u32) -> u64 {
    ((direction << 30) | (size << 16) | ((b'd' as u32) << 8) | number) as u64
}

const GEM_CLOSE: u64 = request(1, 0x09, 8);
const PRIME_FD_TO_HANDLE: u64 = request(3, 0x2e, 12);
const AMDGPU_GEM_USERPTR: u64 = request(3, 0x51, 24);
const AMDGPU_GEM_OP: u64 = request(3, 0x50, 24);
const AMDGPU_GEM_LIST_HANDLES: u64 = request(3, 0x59, 16);
const AMDGPU_GEM_VA: u64 = request(1, 0x48, 64);
const AMDGPU_CTX: u64 = request(3, 0x42, 16);
const AMDGPU_CS: u64 = request(3, 0x44, 24);
const AMDGPU_WAIT_CS: u64 = request(3, 0x49, 32);
const SYNCOBJ_CREATE: u64 = request(3, 0xbf, 8);
const SYNCOBJ_DESTROY: u64 = request(3, 0xc0, 8);
const SYNCOBJ_TIMELINE_WAIT: u64 = request(3, 0xca, 48);

const AMDGPU_VA_OP_MAP: u32 = 1;
const AMDGPU_VA_OP_UNMAP: u32 = 2;
const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;
const AMDGPU_VM_MTYPE_UC: u32 = 4 << 5;
const AMDGPU_GEM_USERPTR_VALIDATE: u32 = 1 << 2;
const AMDGPU_GEM_USERPTR_REGISTER: u32 = 1 << 3;
const AMDGPU_GEM_OP_GET_GEM_CREATE_INFO: u32 = 0;
pub(super) const GEM_DOMAIN_GTT: u64 = 1 << 1;
pub(super) const GEM_CREATE_NO_CPU_ACCESS: u64 = 1 << 1;
pub(super) const GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub(super) const GEM_CREATE_COHERENT: u64 = 1 << 13;
pub(super) const GEM_CREATE_UNCACHED: u64 = 1 << 14;
pub(super) const GEM_CREATE_ENCRYPTED: u64 = 1 << 10;
pub(super) const GEM_CREATE_DISCARDABLE: u64 = 1 << 12;
pub(super) const GEM_CREATE_GFX12_DCC: u64 = 1 << 16;
pub(super) const GEM_CREATE_SPARSE: u64 = 1 << 29;
const GEM_LIST_HANDLES_IS_IMPORT: u32 = 1;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
const CLOCK_MONOTONIC: c_int = 1;
const VM_UPDATE_WAIT_NANOSECONDS: i64 = 5_000_000_000;
const CTX_ALLOC: u32 = 1;
const CTX_FREE: u32 = 2;
const CHUNK_IB: u32 = 1;
const CHUNK_SYNCOBJ_TIMELINE_SIGNAL: u32 = 9;

/// DRM hardware IP selected for one opaque native command stream.
pub(super) const HW_IP_COMPUTE: u32 = 1;
/// DRM hardware IP selected for one opaque SDMA command stream.
pub(super) const HW_IP_DMA: u32 = 2;

#[repr(C)]
struct Timespec {
    seconds: i64,
    nanoseconds: i64,
}

#[repr(C)]
#[derive(Default)]
struct GemClose {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    descriptor: i32,
}

#[repr(C)]
#[derive(Default)]
struct GemUserptr {
    address: u64,
    size: u64,
    flags: u32,
    handle: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct GemCreateInfo {
    pub(super) size: u64,
    pub(super) alignment: u64,
    pub(super) domains: u64,
    pub(super) flags: u64,
}

#[repr(C)]
#[derive(Default)]
struct GemOp {
    handle: u32,
    operation: u32,
    value: u64,
    count: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct GemHandleInfo {
    pub(super) handle: u32,
    flags: u32,
    pub(super) size: u64,
    pub(super) domains: u64,
    pub(super) create_flags: u64,
    pub(super) alignment: u64,
}

impl GemHandleInfo {
    pub(super) fn is_imported(self) -> bool {
        self.flags & GEM_LIST_HANDLES_IS_IMPORT != 0
    }
}

#[repr(C)]
#[derive(Default)]
struct GemListHandles {
    entries: u64,
    count: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Default)]
struct GemVa {
    handle: u32,
    pad: u32,
    operation: u32,
    flags: u32,
    address: u64,
    offset: u64,
    size: u64,
    timeline_point: u64,
    timeline_syncobj: u32,
    input_syncobj_count: u32,
    input_syncobjs: u64,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjCreate {
    handle: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjDestroy {
    handle: u32,
    pad: u32,
}

#[repr(C)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nanoseconds: i64,
    count: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nanoseconds: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ContextInput {
    operation: u32,
    flags: u32,
    context_id: u32,
    priority: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ContextOutput {
    context_id: u32,
    pad: u32,
}

#[repr(C)]
union Context {
    input: ContextInput,
    output: ContextOutput,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CommandStreamInput {
    context_id: u32,
    bo_list_handle: u32,
    chunk_count: u32,
    flags: u32,
    chunks: u64,
}

#[repr(C)]
union CommandStream {
    input: CommandStreamInput,
    sequence: u64,
}

#[repr(C)]
struct CommandChunk {
    kind: u32,
    length_dwords: u32,
    data: u64,
}

#[repr(C)]
struct IndirectBuffer {
    pad: u32,
    flags: u32,
    address: u64,
    byte_length: u32,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
}

#[repr(C)]
struct TimelineSignal {
    handle: u32,
    flags: u32,
    point: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CommandWaitInput {
    sequence: u64,
    absolute_deadline_nanoseconds: u64,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    context_id: u32,
}

#[repr(C)]
union CommandWait {
    input: CommandWaitInput,
    busy: u64,
}

fn call<T>(file: &File, request: u64, body: &mut T) -> io::Result<()> {
    // SAFETY: Every caller supplies the exact repr(C) body for this request and
    // keeps it writable for the duration of the synchronous ioctl.
    let result = unsafe {
        ioctl(
            file.as_raw_fd(),
            request as c_ulong,
            ptr::from_mut(body).cast::<c_void>(),
        )
    };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Allocates one private DRM context on the render file bound to the KFD VM.
pub(super) fn create_context(file: &File) -> io::Result<u32> {
    let mut body = Context {
        input: ContextInput {
            operation: CTX_ALLOC,
            ..ContextInput::default()
        },
    };
    call(file, AMDGPU_CTX, &mut body)?;
    // SAFETY: successful context allocation writes the documented output arm.
    let context_id = unsafe { body.output.context_id };
    if context_id == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(context_id)
}

/// Frees one private DRM context after every command stream retires.
pub(super) fn destroy_context(file: &File, context_id: u32) -> io::Result<()> {
    call(
        file,
        AMDGPU_CTX,
        &mut Context {
            input: ContextInput {
                operation: CTX_FREE,
                context_id,
                ..ContextInput::default()
            },
        },
    )
}

/// Submits a caller-owned native indirect buffer without touching its bytes.
///
/// The returned sequence starts at one in a newly allocated DRM context. An
/// `EFAULT` result is ambiguous: DRM can enqueue before ioctl output copyout.
/// The owner must conservatively publish and track that possible submission.
pub(super) fn submit_indirect_buffer(
    file: &File,
    context_id: u32,
    ip_type: u32,
    address: u64,
    byte_length: u32,
    completion_syncobj: u32,
    completion_point: u64,
) -> io::Result<u64> {
    let buffer = IndirectBuffer {
        pad: 0,
        flags: 0,
        address,
        byte_length,
        ip_type,
        ip_instance: 0,
        ring: 0,
    };
    let completion = TimelineSignal {
        handle: completion_syncobj,
        flags: 0,
        point: completion_point,
    };
    let chunks = [
        CommandChunk {
            kind: CHUNK_IB,
            length_dwords: 8,
            data: ptr::from_ref(&buffer) as u64,
        },
        CommandChunk {
            kind: CHUNK_SYNCOBJ_TIMELINE_SIGNAL,
            length_dwords: 4,
            data: ptr::from_ref(&completion) as u64,
        },
    ];
    let chunk_addresses = [
        ptr::from_ref(&chunks[0]) as u64,
        ptr::from_ref(&chunks[1]) as u64,
    ];
    let mut body = CommandStream {
        input: CommandStreamInput {
            context_id,
            chunk_count: 2,
            chunks: chunk_addresses.as_ptr() as u64,
            ..CommandStreamInput::default()
        },
    };
    call(file, AMDGPU_CS, &mut body)?;
    // SAFETY: successful CS submission writes the documented sequence arm.
    Ok(unsafe { body.sequence })
}

/// Performs one DRM fence wait. `None` requests an infinite wait.
pub(super) fn wait_submission(
    file: &File,
    context_id: u32,
    ip_type: u32,
    sequence: u64,
    timeout_nanoseconds: Option<u64>,
) -> io::Result<bool> {
    let deadline = if let Some(timeout) = timeout_nanoseconds {
        let mut now = Timespec {
            seconds: 0,
            nanoseconds: 0,
        };
        // SAFETY: clock_gettime fills this exact Linux timespec record.
        if unsafe { clock_gettime(CLOCK_MONOTONIC, &raw mut now) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let now = u64::try_from(now.seconds)
            .ok()
            .and_then(|seconds| seconds.checked_mul(1_000_000_000))
            .and_then(|value| value.checked_add(u64::try_from(now.nanoseconds).ok()?))
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
        now.saturating_add(timeout).min(i64::MAX as u64)
    } else {
        u64::MAX
    };
    let mut body = CommandWait {
        input: CommandWaitInput {
            sequence,
            absolute_deadline_nanoseconds: deadline,
            ip_type,
            context_id,
            ..CommandWaitInput::default()
        },
    };
    call(file, AMDGPU_WAIT_CS, &mut body)?;
    // SAFETY: successful WAIT_CS writes the documented busy status arm.
    Ok(unsafe { body.busy == 0 })
}

/// Waits for a preattached timeline point; a timeout leaves native work live.
pub(super) fn wait_timeline_point(
    file: &File,
    handle: u32,
    point: u64,
    timeout_nanoseconds: Option<u64>,
) -> io::Result<bool> {
    let deadline = if let Some(timeout) = timeout_nanoseconds {
        let mut now = Timespec {
            seconds: 0,
            nanoseconds: 0,
        };
        // SAFETY: clock_gettime writes the exact Linux timespec record.
        if unsafe { clock_gettime(CLOCK_MONOTONIC, &raw mut now) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let now = now
            .seconds
            .checked_mul(1_000_000_000)
            .and_then(|value| value.checked_add(now.nanoseconds))
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
        now.saturating_add(i64::try_from(timeout).unwrap_or(i64::MAX))
    } else {
        i64::MAX
    };
    let handles = [handle];
    let points = [point];
    let mut body = SyncobjTimelineWait {
        handles: handles.as_ptr() as u64,
        points: points.as_ptr() as u64,
        timeout_nanoseconds: deadline,
        count: 1,
        flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
        first_signaled: 0,
        pad: 0,
        deadline_nanoseconds: 0,
    };
    match call(file, SYNCOBJ_TIMELINE_WAIT, &mut body) {
        Ok(()) => Ok(true),
        Err(error) if matches!(error.raw_os_error(), Some(62 | 110)) => Ok(false),
        Err(error) => Err(error),
    }
}

pub(super) fn create_syncobj(file: &File) -> io::Result<u32> {
    let mut body = SyncobjCreate::default();
    call(file, SYNCOBJ_CREATE, &mut body)?;
    if body.handle == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(body.handle)
}

pub(super) fn destroy_syncobj(file: &File, handle: u32) -> io::Result<()> {
    if handle == 0 {
        return Ok(());
    }
    call(
        file,
        SYNCOBJ_DESTROY,
        &mut SyncobjDestroy { handle, pad: 0 },
    )
}

pub(super) fn import_dma_buf(file: &File, descriptor: i32) -> io::Result<u32> {
    if descriptor < 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let mut body = PrimeHandle {
        descriptor,
        ..PrimeHandle::default()
    };
    call(file, PRIME_FD_TO_HANDLE, &mut body)?;
    if body.handle == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(body.handle)
}

/// Returns immutable creation facts for a handle in this render file.
pub(super) fn gem_create_info(file: &File, handle: u32) -> io::Result<GemCreateInfo> {
    let mut info = GemCreateInfo::default();
    call(
        file,
        AMDGPU_GEM_OP,
        &mut GemOp {
            handle,
            operation: AMDGPU_GEM_OP_GET_GEM_CREATE_INFO,
            value: ptr::from_mut(&mut info) as u64,
            ..GemOp::default()
        },
    )?;
    Ok(info)
}

/// Reports the number of handles, writing the snapshot only when it fits.
pub(super) fn list_gem_handles(file: &File, entries: &mut [GemHandleInfo]) -> io::Result<usize> {
    let count =
        u32::try_from(entries.len()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let mut args = GemListHandles {
        entries: if entries.is_empty() {
            0
        } else {
            entries.as_mut_ptr() as u64
        },
        count,
        ..GemListHandles::default()
    };
    call(file, AMDGPU_GEM_LIST_HANDLES, &mut args)?;
    Ok(args.count as usize)
}

/// A cold activation probe; unsupported ioctls keep IMPORT unadvertised.
pub(super) fn supports_system_dma_buf_import(file: &File) -> bool {
    list_gem_handles(file, &mut []).is_ok()
        && gem_create_info(file, 0).is_err_and(|source| source.raw_os_error() == Some(2))
}

/// Registers a borrowed, page-aligned host extent in this render VM. The
/// caller retains the pages through GEM close and owns the returned handle.
pub(super) fn register_userptr(
    file: &File,
    address: u64,
    size: u64,
    handle: &mut u32,
) -> io::Result<()> {
    let mut body = GemUserptr {
        address,
        size,
        flags: AMDGPU_GEM_USERPTR_REGISTER | AMDGPU_GEM_USERPTR_VALIDATE,
        ..GemUserptr::default()
    };
    let result = call(file, AMDGPU_GEM_USERPTR, &mut body);
    *handle = body.handle;
    result?;
    if *handle == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(())
}

pub(super) fn close_gem(file: &File, handle: u32) -> io::Result<()> {
    if handle == 0 {
        return Ok(());
    }
    call(file, GEM_CLOSE, &mut GemClose { handle, pad: 0 })
}

#[allow(
    clippy::too_many_arguments,
    reason = "keep the raw GEM VA ioctl fields explicit at its only call boundary"
)]
pub(super) fn map(
    file: &File,
    handle: u32,
    address: u64,
    offset: u64,
    size: u64,
    permission_bits: u32,
    timeline_syncobj: u32,
    timeline_point: u64,
) -> io::Result<()> {
    map_with_cache(
        file,
        handle,
        address,
        offset,
        size,
        permission_bits,
        false,
        timeline_syncobj,
        timeline_point,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "keep DRM VA fields explicit at the native registration boundary"
)]
pub(super) fn map_with_cache(
    file: &File,
    handle: u32,
    address: u64,
    offset: u64,
    size: u64,
    permission_bits: u32,
    uncached: bool,
    timeline_syncobj: u32,
    timeline_point: u64,
) -> io::Result<()> {
    if permission_bits & !7 != 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let flags = if permission_bits & 1 != 0 {
        AMDGPU_VM_PAGE_READABLE
    } else {
        0
    } | if permission_bits & 2 != 0 {
        AMDGPU_VM_PAGE_WRITEABLE
    } else {
        0
    } | if permission_bits & 4 != 0 {
        AMDGPU_VM_PAGE_EXECUTABLE
    } else {
        0
    } | if uncached { AMDGPU_VM_MTYPE_UC } else { 0 };
    call(
        file,
        AMDGPU_GEM_VA,
        &mut GemVa {
            handle,
            operation: AMDGPU_VA_OP_MAP,
            flags,
            address,
            offset,
            size,
            timeline_point,
            timeline_syncobj,
            ..GemVa::default()
        },
    )
}

pub(super) fn unmap(
    file: &File,
    handle: u32,
    address: u64,
    offset: u64,
    size: u64,
    timeline_syncobj: u32,
    timeline_point: u64,
) -> io::Result<()> {
    call(
        file,
        AMDGPU_GEM_VA,
        &mut GemVa {
            handle,
            operation: AMDGPU_VA_OP_UNMAP,
            address,
            offset,
            size,
            timeline_point,
            timeline_syncobj,
            ..GemVa::default()
        },
    )
}

pub(super) fn wait(file: &File, handle: u32, point: u64) -> io::Result<()> {
    let mut now = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    // SAFETY: clock_gettime fills this exact Linux timespec record.
    if unsafe { clock_gettime(CLOCK_MONOTONIC, &raw mut now) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let deadline = now
        .seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(now.nanoseconds))
        .and_then(|value| value.checked_add(VM_UPDATE_WAIT_NANOSECONDS))
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
    let handles = [handle];
    let points = [point];
    call(
        file,
        SYNCOBJ_TIMELINE_WAIT,
        &mut SyncobjTimelineWait {
            handles: handles.as_ptr() as u64,
            points: points.as_ptr() as u64,
            timeout_nanoseconds: deadline,
            count: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            first_signaled: 0,
            pad: 0,
            deadline_nanoseconds: 0,
        },
    )
}

const _: () = {
    assert!(std::mem::size_of::<GemClose>() == 8);
    assert!(std::mem::size_of::<PrimeHandle>() == 12);
    assert!(std::mem::size_of::<GemCreateInfo>() == 32);
    assert!(std::mem::size_of::<GemOp>() == 24);
    assert!(std::mem::offset_of!(GemOp, value) == 8);
    assert!(std::mem::size_of::<GemListHandles>() == 16);
    assert!(std::mem::size_of::<GemHandleInfo>() == 40);
    assert!(std::mem::offset_of!(GemHandleInfo, create_flags) == 24);
    assert!(std::mem::size_of::<GemVa>() == 64);
    assert!(std::mem::size_of::<SyncobjCreate>() == 8);
    assert!(std::mem::size_of::<SyncobjDestroy>() == 8);
    assert!(std::mem::size_of::<SyncobjTimelineWait>() == 48);
    assert!(std::mem::size_of::<Context>() == 16);
    assert!(std::mem::size_of::<CommandStream>() == 24);
    assert!(std::mem::size_of::<CommandWait>() == 32);
    assert!(std::mem::size_of::<CommandChunk>() == 16);
    assert!(std::mem::size_of::<IndirectBuffer>() == 32);
    assert!(std::mem::size_of::<TimelineSignal>() == 16);
    assert!(GEM_CLOSE == 0x4008_6409);
    assert!(PRIME_FD_TO_HANDLE == 0xc00c_642e);
    assert!(AMDGPU_GEM_OP == 0xc018_6450);
    assert!(AMDGPU_GEM_LIST_HANDLES == 0xc010_6459);
    assert!(AMDGPU_GEM_VA == 0x4040_6448);
    assert!(SYNCOBJ_CREATE == 0xc008_64bf);
    assert!(SYNCOBJ_DESTROY == 0xc008_64c0);
    assert!(SYNCOBJ_TIMELINE_WAIT == 0xc030_64ca);
    assert!(AMDGPU_CTX == 0xc010_6442);
    assert!(AMDGPU_CS == 0xc018_6444);
    assert!(AMDGPU_WAIT_CS == 0xc020_6449);
};
