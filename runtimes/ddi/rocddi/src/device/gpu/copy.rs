// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linear copy and dword fill through a bounded native SDMA submission context.

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use std::time::Duration;

use crate::device::gpu::GpuDevice;
use crate::device::gpu::kernel_queue::{
    KernelCommand, KernelQueue, KernelQueueFormat, KernelQueueWait,
};
use crate::device::gpu::queue::{
    Queue, QueueParameters, QueuePriority, QueueRequest, QueueRingMemory, SdmaEngineSelection,
    ring_doorbell,
};
use crate::memory::{Allocation, DeviceAccess, MemoryKind};
use crate::topology::{GpuSdmaDialect, GpuTargetProfile};
use crate::{Error, ErrorKind};

mod packet;
use packet::{SdmaCommand, TimingAddresses, encode_gfx9_timestamp};

// GFX12.0 SDMA 7.0.1 uses the OSS5 COPY_LINEAR and GCR packet layouts. Keep the packet
// format here so either frontend can use the same native copy mechanism.
const COPY_PACKET_BYTES: usize = 28;
const SWAP_PACKET_BYTES: usize = 28;
const FILL_PACKET_BYTES: usize = 20;
const GCR_PACKET_BYTES: usize = 20;
const TIMESTAMP_PACKET_BYTES: usize = 12;
const COMMAND_BYTES: usize = GCR_PACKET_BYTES * 2 + COPY_PACKET_BYTES;
#[cfg(test)]
const TIMED_COMMAND_BYTES: usize = COMMAND_BYTES + TIMESTAMP_PACKET_BYTES * 2;
const FILL_COMMAND_BYTES: usize = GCR_PACKET_BYTES * 2 + FILL_PACKET_BYTES;
const GFX1250_GCR_PACKET_BYTES: usize = 24;
const GFX1250_CONTROLLED_COMMAND_BYTES: usize =
    GFX1250_GCR_PACKET_BYTES * 2 + 19 * 4 + TIMESTAMP_PACKET_BYTES * 2;
const MAX_COPY_PACKET_BYTES: u64 = 0x3f_ffe0;
const MAX_INDIRECT_PACKET_BYTES: u64 = 0x3fff_ffe0;
const MAX_FILL_PACKET_BYTES: u64 = 0x3f_ffe0;
const MAX_FILL_PACKET_DWORDS: u64 = MAX_FILL_PACKET_BYTES / 4;
const SDMA_COPY_LINEAR: u32 = 1;
const SDMA_COPY_SWAP: u32 = 1 | (9 << 8);
// The gfx1250 SIGNAL_OPERATION field uses the GL2 64-bit atomic opcodes.
// Atomic swap implements the public raw WRITE operation.
const SDMA_SIGNAL_WRITE64: u32 = 0x67;
const SDMA_SIGNAL_ADD64: u32 = 0x6f;
const SDMA_SIGNAL_SUB64: u32 = 0x70;
const SDMA_CONST_FILL_DWORD: u32 = 0x0b | (2 << 30);
const SDMA_TIMESTAMP_GET_GLOBAL_SYS: u32 = 0x0d | (2 << 8) | (3 << 24);
const SDMA_TIMESTAMP_GET_GLOBAL_GFX9: u32 = 0x0d | (2 << 8);
const SDMA_USER_GCR: u32 = 0x11 | (1 << 8);
const GCR_WRITEBACK: u32 = (1 << 31) | (1 << 22);
const GCR_INVALIDATE: u32 = (1 << 30) | (1 << 25) | (1 << 24) | (1 << 23);
const NATIVE_WAIT_NS: u64 = 50_000_000;
const USER_RING_BYTES: u64 = 4096;
const SDMA_FENCE_BYTES: usize = 16;

/// Packet format selected from the activated GPU target.
#[derive(Clone, Copy)]
enum SdmaProtocol {
    /// SDMA 4 packet layout used by GFX9 KFD user queues.
    Gfx9,
    /// SDMA 7.0 packet layout used by GFX12.0.
    Gfx12_0,
    /// Extended packet layout used by GFX12.5.
    Gfx12_5,
}

/// A gfx1250 SDMA comparison performed before a copy packet.
#[derive(Clone, Copy, Debug)]
pub struct GpuCopyWait {
    /// GPU address of the aligned 64-bit value to poll.
    pub address: u64,
    /// SDMA comparison code, from 1 (less than) through 6 (greater than).
    pub function: u8,
    /// Value compared with the masked memory contents.
    pub reference: u64,
    /// Mask applied to the polled value.
    pub mask: u64,
}

/// Atomic operation performed after a gfx1250 copy packet.
#[derive(Clone, Copy, Debug)]
pub enum GpuCopySignalOperation {
    /// Store the operand by atomic exchange.
    Write,
    /// Add the operand atomically.
    Add,
    /// Subtract the operand atomically.
    Subtract,
}

/// A gfx1250 SDMA atomic notification after a copy packet.
#[derive(Clone, Copy, Debug)]
pub struct GpuCopySignal {
    /// GPU address of the aligned 64-bit atomic target.
    pub address: u64,
    /// Atomic operation on the target.
    pub operation: GpuCopySignalOperation,
    /// Atomic operand.
    pub data: u64,
}

/// Optional gfx1250 packet behavior. Indirect addresses name 64-bit pointer
/// slots; SDMA resolves their contents when the packet executes.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuCopyOptions {
    /// Optional wait before the first packet of a transfer.
    pub wait: Option<GpuCopyWait>,
    /// Optional atomic signal after the last packet of a transfer.
    pub signal: Option<GpuCopySignal>,
    /// Read the source address from a GPU-accessible pointer slot.
    pub indirect_source: bool,
    /// Read the destination address from a GPU-accessible pointer slot.
    pub indirect_destination: bool,
}

impl GpuCopyOptions {
    fn is_plain(self) -> bool {
        self.wait.is_none()
            && self.signal.is_none()
            && !self.indirect_source
            && !self.indirect_destination
    }

    fn validate(self) -> Result<(), CopyFailure> {
        if self.wait.is_some_and(|wait| {
            wait.address == 0 || wait.address % 8 != 0 || !(1..=6).contains(&wait.function)
        }) || self
            .signal
            .is_some_and(|signal| signal.address == 0 || signal.address % 8 != 0)
        {
            return Err(invalid("invalid SDMA copy wait or signal"));
        }
        Ok(())
    }
}

/// Failure of a native copy or fill, including whether the GPU may still reach
/// its operands. A frontend must retain their backing when this is true, even
/// if it has already reported an asynchronous error to its caller.
#[derive(Debug)]
pub struct CopyFailure {
    /// The validation, allocation, submission, or completion failure.
    pub error: Error,
    /// Native retirement was not proved for the last accepted submission.
    pub operands_may_be_live: bool,
}

/// Byte-addressed rectangular copy shape. A slice contains rows separated by
/// `source_pitch` or `destination_pitch`; the corresponding slice stride is in
/// bytes. Bases point at the first byte of the selected rectangle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CopyRect {
    /// First destination byte in the GPU address domain.
    pub destination: u64,
    /// First source byte in the GPU address domain.
    pub source: u64,
    /// Bytes copied per row.
    pub width: u64,
    /// Rows copied per slice.
    pub height: u32,
    /// Number of slices.
    pub depth: u32,
    /// Destination bytes between adjacent rows.
    pub destination_pitch: u64,
    /// Source bytes between adjacent rows.
    pub source_pitch: u64,
    /// Destination bytes between adjacent slices.
    pub destination_slice: u64,
    /// Source bytes between adjacent slices.
    pub source_slice: u64,
}

impl CopyRect {
    /// Describes one contiguous byte range as a single row and slice.
    #[must_use]
    pub const fn linear(destination: u64, source: u64, size: u64) -> Self {
        Self {
            destination,
            source,
            width: size,
            height: 1,
            depth: 1,
            destination_pitch: size,
            source_pitch: size,
            destination_slice: size,
            source_slice: size,
        }
    }

    fn validate(self) -> Result<(), CopyFailure> {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return Ok(());
        }
        if self.destination == 0
            || self.source == 0
            || (self.height > 1
                && (self.destination_pitch < self.width || self.source_pitch < self.width))
        {
            return Err(invalid("invalid SDMA copy rectangle"));
        }
        for (base, pitch, slice) in [
            (
                self.destination,
                self.destination_pitch,
                self.destination_slice,
            ),
            (self.source, self.source_pitch, self.source_slice),
        ] {
            let end = u64::from(self.depth - 1)
                .checked_mul(slice)
                .and_then(|offset| {
                    u64::from(self.height - 1)
                        .checked_mul(pitch)
                        .and_then(|row| offset.checked_add(row))
                })
                .and_then(|offset| offset.checked_add(self.width))
                .and_then(|extent| base.checked_add(extent));
            if end.is_none() {
                return Err(invalid("SDMA copy rectangle exceeds the address space"));
            }
        }
        Ok(())
    }
}

impl CopyFailure {
    fn retired(error: Error) -> Self {
        Self {
            error,
            operands_may_be_live: false,
        }
    }
}

fn invalid(detail: &'static str) -> CopyFailure {
    CopyFailure::retired(Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail,
    })
}

fn poisoned_pool() -> CopyFailure {
    CopyFailure::retired(Error::Operation {
        kind: ErrorKind::Internal,
        detail: "SDMA copy resource pool is poisoned",
    })
}

fn validate_fill_range(destination: u64, count: u64) -> Result<(), CopyFailure> {
    if count == 0 {
        return Ok(());
    }
    let size = count
        .checked_mul(4)
        .ok_or_else(|| invalid("SDMA fill range is too large"))?;
    if destination == 0 || destination % 4 != 0 || destination.checked_add(size).is_none() {
        return Err(invalid("invalid SDMA fill destination"));
    }
    Ok(())
}

/// The GFX9 transport publishes packets directly to a KFD SDMA ring. GFX12
/// retains the DRM-mediated indirect-buffer submission path.
enum CopyQueue {
    Kernel(KernelQueue),
    User(Queue),
}

impl CopyQueue {
    fn healthy(&self) -> bool {
        match self {
            Self::Kernel(queue) => queue.status().terminal.is_none(),
            Self::User(queue) => queue.check().is_ok(),
        }
    }

    #[allow(unsafe_code)]
    fn destroy(&mut self) -> Result<(), Error> {
        match self {
            Self::Kernel(queue) => queue.destroy(),
            // SAFETY: The resource owner has stopped publishing and proved the
            // fence packet retired before it calls this method.
            Self::User(queue) => unsafe { queue.destroy() },
        }
    }
}

/// Native resources for one operation. An unretired submission may still read
/// the indirect buffer or fence address; Drop retains both owners then.
struct CopyResources {
    /// Native submission queue retained until command retirement.
    queue: ManuallyDrop<CopyQueue>,
    /// GFX12 indirect command buffer or GFX9 fence destination.
    command: ManuallyDrop<Allocation>,
    /// CPU mapping of that command buffer.
    host_address: usize,
    /// GPU address submitted to the kernel queue.
    command_address: u64,
    /// Accepted submission awaiting proven retirement.
    pending: Option<u64>,
    /// Submission may be in progress across a native call.
    submitting: bool,
    /// Whether this context has ever accepted a native submission.
    submitted_any: bool,
    /// Optional timestamp destinations for the current sequence.
    timing: Option<TimingState>,
    /// Packet format selected for the activated GPU.
    protocol: SdmaProtocol,
}

/// Timestamp addresses and whether a sequence has issued its first packet.
struct TimingState {
    /// Global-clock destination before the first packet.
    start: u64,
    /// Global-clock destination after the final packet.
    end: u64,
    /// Controls whether the next packet also writes the start timestamp.
    started: bool,
}

impl CopyResources {
    fn new(gpu: &GpuDevice<'_>, format: KernelQueueFormat) -> Result<Self, CopyFailure> {
        let dialect = gpu.info().target_profile.sdma_dialect().ok_or_else(|| {
            CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "no qualified SDMA packet dialect for this GPU",
            })
        })?;
        let gfx9 = dialect == GpuSdmaDialect::Gfx9;
        let command = gpu
            .device()
            .allocate(
                MemoryKind::System,
                4096,
                4096,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .map_err(CopyFailure::retired)?;
        let host_address = command.info().host_address.ok_or_else(|| {
            CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "native command backing has no host mapping",
            })
        })?;
        let command_address = command
            .device_address(gpu.device())
            .map_err(CopyFailure::retired)?;
        let queue = if gfx9 {
            let parameters = match format {
                KernelQueueFormat::Sdma => QueueParameters::Sdma,
                KernelQueueFormat::SdmaOnEngine(engine) => QueueParameters::SdmaByEngine {
                    selection: SdmaEngineSelection::Id(engine),
                    ring_memory: QueueRingMemory::System,
                },
                KernelQueueFormat::Pm4 => return Err(invalid("PM4 is not an SDMA copy format")),
            };
            // SAFETY: SDMA requests contain no external addresses. This owner
            // is the sole producer and retains its ring through destruction.
            #[allow(unsafe_code)]
            let queue = unsafe {
                gpu.create_queue(QueueRequest {
                    ring_size_bytes: USER_RING_BYTES,
                    parameters,
                    priority: QueuePriority::Normal,
                    device_producer: false,
                })
            }
            .map_err(CopyFailure::retired)?;
            CopyQueue::User(queue)
        } else {
            CopyQueue::Kernel(
                gpu.create_kernel_queue(format)
                    .map_err(CopyFailure::retired)?,
            )
        };
        let resources = Self {
            queue: ManuallyDrop::new(queue),
            command: ManuallyDrop::new(command),
            host_address,
            command_address,
            pending: None,
            submitting: false,
            submitted_any: false,
            timing: None,
            protocol: match dialect {
                GpuSdmaDialect::Gfx9 => SdmaProtocol::Gfx9,
                GpuSdmaDialect::Gfx1200 => SdmaProtocol::Gfx12_0,
                GpuSdmaDialect::Gfx1250 => SdmaProtocol::Gfx12_5,
            },
        };
        Ok(resources)
    }

    fn failure(&self, error: Error) -> CopyFailure {
        CopyFailure {
            error,
            operands_may_be_live: self.pending.is_some() || self.submitting,
        }
    }

    fn refresh_retirement(&mut self) {
        let retired = match &*self.queue {
            CopyQueue::Kernel(queue) => self
                .pending
                .is_some_and(|submission| queue.status().retired_submission >= submission),
            CopyQueue::User(queue) => self.pending.is_some_and(|submission| {
                self.fence_word().load(Ordering::Acquire) == 1
                    && queue
                        .progress()
                        .is_ok_and(|(read, write)| read == write && write == submission)
            }),
        };
        if retired {
            self.pending = None;
        }
    }

    #[allow(unsafe_code)]
    fn fence_word(&self) -> &AtomicU32 {
        // SAFETY: The retained system allocation owns an aligned host page.
        unsafe { &*(self.host_address as *const AtomicU32) }
    }

    #[allow(
        unsafe_code,
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        reason = "one bounded producer transaction contains encoding, publication, and retirement"
    )]
    fn submit_user_packet(
        &mut self,
        command: SdmaCommand,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        let timing = self.timing.as_ref().and_then(|state| {
            matches!(command, SdmaCommand::Copy { .. }).then_some(TimingAddresses {
                start: (!state.started).then_some(state.start),
                end: state.end,
            })
        });
        let mut words =
            [0_u32; (COPY_PACKET_BYTES + TIMESTAMP_PACKET_BYTES * 2 + SDMA_FENCE_BYTES) / 4];
        let mut offset = 0;
        if let Some(start) = timing.and_then(|timing| timing.start) {
            encode_gfx9_timestamp(&mut words, offset, start);
            offset += TIMESTAMP_PACKET_BYTES / 4;
        }
        let command_bytes = command.encode(&mut words[offset..], None, SdmaProtocol::Gfx9) as usize;
        offset += command_bytes / 4;
        if let Some(timing) = timing {
            encode_gfx9_timestamp(&mut words, offset, timing.end);
            offset += TIMESTAMP_PACKET_BYTES / 4;
        }
        let fence_offset = offset;
        words[fence_offset] = 5; // SDMA v4 FENCE, 32-bit system write.
        words[fence_offset + 1] = self.command_address as u32;
        words[fence_offset + 2] = (self.command_address >> 32) as u32;
        words[fence_offset + 3] = 1;
        let packet_bytes = (fence_offset + SDMA_FENCE_BYTES / 4) * 4;
        let CopyQueue::User(queue) = &*self.queue else {
            return Err(invalid("KFD SDMA packet requires a user queue"));
        };
        let info = queue.info();
        let write = loop {
            let (read, write) = queue.progress().map_err(|error| self.failure(error))?;
            if read > write {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::DriverContract,
                    detail: "KFD SDMA read index exceeded its write index",
                }));
            }
            if write % 4 != 0 {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::DriverContract,
                    detail: "KFD SDMA write index is not dword aligned",
                }));
            }
            if read == write {
                break write;
            }
            if cancel.load(Ordering::Acquire) {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::Busy,
                    detail: "SDMA copy was cancelled before ring reuse",
                }));
            }
            std::thread::sleep(Duration::from_micros(50));
        };
        let offset = write & (info.ring_size_bytes - 1);
        let padding = if offset + packet_bytes as u64 > info.ring_size_bytes {
            info.ring_size_bytes - offset
        } else {
            0
        };
        let next = write
            .checked_add(padding)
            .and_then(|value| value.checked_add(packet_bytes as u64))
            .ok_or_else(|| invalid("KFD SDMA write index overflowed"))?;
        self.fence_word().store(0, Ordering::Release);
        self.submitting = true;
        let ring = info.ring_host_address as *mut u32;
        // SAFETY: The exclusive producer owns a 4096-byte ring. Queue progress
        // proved it empty, and the command plus optional NOP padding fits.
        unsafe {
            for index in 0..padding as usize / 4 {
                ring.add((offset as usize / 4) + index).write_volatile(0);
            }
            let base = if padding == 0 { offset as usize / 4 } else { 0 };
            for (index, word) in words[..packet_bytes / 4].iter().enumerate() {
                ring.add(base + index).write_volatile(*word);
            }
            fence(Ordering::Release);
            (&*(info.write_index_host_address as *const AtomicU64)).store(next, Ordering::Release);
            ring_doorbell(info.doorbell_host_address, next);
        }
        self.pending = Some(next);
        self.submitting = false;
        self.submitted_any = true;
        if let Some(state) = self.timing.as_mut() {
            if timing.is_some() {
                state.started = true;
            }
        }
        loop {
            if self.fence_word().load(Ordering::Acquire) == 1 {
                // The fence proves operand completion. Wait for the ring read
                // index as well so a following publication or destroy sees an
                // empty queue even when its pointer update trails the write.
                let (read, observed_write) =
                    queue.progress().map_err(|error| self.failure(error))?;
                if read == observed_write && observed_write == next {
                    self.pending = None;
                    return Ok(());
                }
            }
            queue.check().map_err(|error| self.failure(error))?;
            if cancel.load(Ordering::Acquire) {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::Busy,
                    detail: "SDMA copy was cancelled before retirement",
                }));
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    #[allow(unsafe_code)]
    fn submit_packet(
        &mut self,
        command: SdmaCommand,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        if cancel.load(Ordering::Acquire) {
            return Err(self.failure(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "SDMA copy was cancelled",
            }));
        }
        if matches!(&*self.queue, CopyQueue::User(_)) {
            return self.submit_user_packet(command, cancel);
        }
        let timing = self.timing.as_ref().and_then(|timing| {
            matches!(command, SdmaCommand::Copy { .. } | SdmaCommand::Swap { .. }).then_some(
                TimingAddresses {
                    start: (!timing.started).then_some(timing.start),
                    end: timing.end,
                },
            )
        });
        // SAFETY: The allocation owns a writable host mapping of at least one
        // page. The command is at most 148 bytes and no prior submission reaches
        // it before this write.
        let words = unsafe {
            &mut *(self.host_address as *mut [u32; GFX1250_CONTROLLED_COMMAND_BYTES / 4])
        };
        let byte_length = command.encode(words, timing, self.protocol);
        fence(Ordering::Release);
        self.submitting = true;
        // SAFETY: Packet bytes, synchronization, and backing lifetime are
        // established by this resource owner and the caller's operand contract.
        let submission = unsafe {
            let CopyQueue::Kernel(queue) = &mut *self.queue else {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::Internal,
                    detail: "SDMA indirect submission lost its kernel queue",
                }));
            };
            queue.submit(KernelCommand {
                device_address: self.command_address,
                byte_length,
            })
        };
        // The driver contract reports ambiguity as an accepted identity, so
        // an error proves rejection. A panic before this line keeps submitting
        // set, so Drop retains the command and queue backing.
        self.submitting = false;
        let submission = submission.map_err(|error| self.failure(error))?;
        self.submitted_any = true;
        if timing.is_some() {
            if let Some(timing) = self.timing.as_mut() {
                timing.started = true;
            }
        }
        self.pending = Some(submission);
        loop {
            let CopyQueue::Kernel(queue) = &*self.queue else {
                return Err(self.failure(Error::Operation {
                    kind: ErrorKind::Internal,
                    detail: "SDMA indirect wait lost its kernel queue",
                }));
            };
            let wait = queue.wait(submission, NATIVE_WAIT_NS, 0);
            self.refresh_retirement();
            match wait {
                Ok(KernelQueueWait::Retired) => return Ok(()),
                Ok(KernelQueueWait::TimedOut) if !cancel.load(Ordering::Acquire) => {}
                Ok(KernelQueueWait::TimedOut) => {
                    return Err(self.failure(Error::Operation {
                        kind: ErrorKind::Busy,
                        detail: "SDMA copy was cancelled before retirement",
                    }));
                }
                Err(error) => return Err(self.failure(error)),
            }
        }
    }

    fn copy_rect(&mut self, rect: CopyRect, cancel: &AtomicBool) -> Result<(), CopyFailure> {
        if rect.width == 0 || rect.height == 0 || rect.depth == 0 {
            return Ok(());
        }
        for layer in 0..rect.depth {
            for row in 0..rect.height {
                let source_row = rect.source
                    + u64::from(layer) * rect.source_slice
                    + u64::from(row) * rect.source_pitch;
                let destination_row = rect.destination
                    + u64::from(layer) * rect.destination_slice
                    + u64::from(row) * rect.destination_pitch;
                let mut completed = 0;
                while completed < rect.width {
                    let chunk = (rect.width - completed).min(MAX_COPY_PACKET_BYTES) as u32;
                    self.submit_packet(
                        SdmaCommand::Copy {
                            destination: destination_row + completed,
                            source: source_row + completed,
                            size: chunk,
                            options: GpuCopyOptions::default(),
                        },
                        cancel,
                    )?;
                    completed += u64::from(chunk);
                }
            }
        }
        Ok(())
    }

    fn swap_linear(
        &mut self,
        address_a: u64,
        address_b: u64,
        size: u64,
        options: GpuCopyOptions,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        let mut completed = 0;
        while completed < size {
            let chunk = (size - completed).min(MAX_COPY_PACKET_BYTES) as u32;
            self.submit_packet(
                SdmaCommand::Swap {
                    address_a: address_a + completed,
                    address_b: address_b + completed,
                    size: chunk,
                    options: GpuCopyOptions {
                        wait: if completed == 0 { options.wait } else { None },
                        signal: if completed + u64::from(chunk) == size {
                            options.signal
                        } else {
                            None
                        },
                        ..options
                    },
                },
                cancel,
            )?;
            completed += u64::from(chunk);
        }
        Ok(())
    }

    fn copy_linear_controlled(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
        options: GpuCopyOptions,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        let mut completed = 0;
        while completed < size {
            let chunk = if options.indirect_source || options.indirect_destination {
                u32::try_from(size).map_err(|_| invalid("indirect SDMA copy is too large"))?
            } else {
                (size - completed).min(MAX_COPY_PACKET_BYTES) as u32
            };
            self.submit_packet(
                SdmaCommand::Copy {
                    destination: destination + completed,
                    source: source + completed,
                    size: chunk,
                    options: GpuCopyOptions {
                        wait: if completed == 0 { options.wait } else { None },
                        signal: if completed + u64::from(chunk) == size {
                            options.signal
                        } else {
                            None
                        },
                        ..options
                    },
                },
                cancel,
            )?;
            completed += u64::from(chunk);
        }
        Ok(())
    }
}

#[allow(unsafe_code)]
impl Drop for CopyResources {
    fn drop(&mut self) {
        self.refresh_retirement();
        if self.pending.is_some() || self.submitting {
            // The native command may still read this allocation. The caller
            // separately retains its source and destination after this error.
            return;
        }
        if self.queue.destroy().is_err() {
            // Native ownership is uncertain. Keep the fence or indirect buffer
            // alongside the queue transport through process teardown.
            return;
        }
        let _ = self.command.free();
        // SAFETY: Pending native access was ruled out above. These fields are
        // dropped exactly once and remain owned by this resource bundle.
        unsafe {
            ManuallyDrop::drop(&mut self.queue);
            ManuallyDrop::drop(&mut self.command);
        }
    }
}

/// One idle SDMA context per queue format, shared by activated-device clones.
/// Busy callers acquire independent contexts instead of serializing copies.
#[derive(Default)]
pub(crate) struct CopyResourcePool {
    /// Contexts available for reuse by queue format.
    idle: Mutex<CopyResourceSlots>,
}

/// Idle copy contexts indexed by their native SDMA queue format.
#[derive(Default)]
struct CopyResourceSlots {
    /// Context using the driver's default SDMA engine.
    default: Option<CopyResources>,
    /// Contexts pinned to specific SDMA engine indices.
    engines: [Option<CopyResources>; u32::BITS as usize],
}

impl CopyResourceSlots {
    fn for_format(
        &mut self,
        format: KernelQueueFormat,
    ) -> Result<&mut Option<CopyResources>, CopyFailure> {
        match format {
            KernelQueueFormat::Sdma => Ok(&mut self.default),
            KernelQueueFormat::SdmaOnEngine(engine) => self
                .engines
                .get_mut(engine as usize)
                .ok_or_else(|| invalid("SDMA engine index is out of range")),
            KernelQueueFormat::Pm4 => Err(invalid("PM4 is not an SDMA copy format")),
        }
    }
}

impl CopyResourcePool {
    fn take<'a>(
        &'a self,
        gpu: &GpuDevice<'_>,
        format: KernelQueueFormat,
    ) -> Result<CopyResourceLease<'a>, CopyFailure> {
        let cached = self
            .idle
            .lock()
            .map_err(|_| poisoned_pool())?
            .for_format(format)?
            .take();
        let resources = match cached {
            Some(resources) if resources.queue.healthy() => resources,
            Some(_) | None => CopyResources::new(gpu, format)?,
        };
        Ok(CopyResourceLease {
            resources: Some(resources),
            pool: self,
            format,
            reusable: true,
        })
    }

    fn preload(&self, gpu: &GpuDevice<'_>, format: KernelQueueFormat) -> Result<(), CopyFailure> {
        if self
            .idle
            .lock()
            .map_err(|_| poisoned_pool())?
            .for_format(format)?
            .as_ref()
            .is_some_and(|resources| resources.queue.healthy())
        {
            return Ok(());
        }
        let resources = CopyResources::new(gpu, format)?;
        let mut idle = self.idle.lock().map_err(|_| poisoned_pool())?;
        let slot = idle.for_format(format)?;
        let stale = if slot
            .as_ref()
            .is_none_or(|resources| !resources.queue.healthy())
        {
            slot.replace(resources)
        } else {
            None
        };
        drop(idle);
        drop(stale);
        Ok(())
    }
}

/// Checked-out context returned to the pool only after proven retirement.
struct CopyResourceLease<'pool> {
    /// Native command and queue owners while checked out.
    resources: Option<CopyResources>,
    /// Pool that may accept the context after this lease ends.
    pool: &'pool CopyResourcePool,
    /// Slot from which this context was acquired.
    format: KernelQueueFormat,
    /// False after a failure makes reuse unsafe.
    reusable: bool,
}

impl CopyResourceLease<'_> {
    fn discard(&mut self) {
        self.reusable = false;
    }
}

impl Deref for CopyResourceLease<'_> {
    type Target = CopyResources;

    #[allow(
        clippy::expect_used,
        reason = "only Drop can remove a lease's resource"
    )]
    fn deref(&self) -> &Self::Target {
        self.resources
            .as_ref()
            .expect("a live SDMA copy lease owns its resources")
    }
}

impl DerefMut for CopyResourceLease<'_> {
    #[allow(
        clippy::expect_used,
        reason = "only Drop can remove a lease's resource"
    )]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resources
            .as_mut()
            .expect("a live SDMA copy lease owns its resources")
    }
}

impl Drop for CopyResourceLease<'_> {
    fn drop(&mut self) {
        if !self.reusable {
            return;
        }
        let Some(mut resources) = self.resources.take() else {
            return;
        };
        resources.refresh_retirement();
        if resources.pending.is_some() || resources.submitting || !resources.queue.healthy() {
            return;
        }
        resources.timing = None;
        resources.submitted_any = false;
        if let Ok(mut idle) = self.pool.idle.lock() {
            if let Ok(slot) = idle.for_format(self.format) {
                if slot.is_none() {
                    *slot = Some(resources);
                }
            }
        }
    }
}

/// Raw global GPU clock ticks around a native SDMA copy sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuCopyTimestamps {
    /// Global GPU clock when the first copy packet began.
    pub start: u64,
    /// Global GPU clock after the final copy packet completed.
    pub end: u64,
}

/// Host-visible clock values backed by a retained GPU allocation.
struct CopyTiming {
    /// Keeps the timestamp memory mapped until its values are consumed.
    _allocation: Allocation,
    /// CPU address of the native timestamp record.
    host_address: usize,
}

/// An ordered SDMA copy sequence sharing one native queue and command or fence
/// allocation. Each operation retires before the next begins. A failed
/// operation makes the sequence terminal, so an unretired packet cannot be
/// replaced in the shared command allocation.
pub struct GpuCopySequence<'device, 'cancel> {
    /// Activated GPU whose queue and mappings perform the copy.
    gpu: GpuDevice<'device>,
    /// Cancellation flag observed between native submissions.
    cancel: &'cancel AtomicBool,
    /// Checked-out native queue and command buffer.
    resources: CopyResourceLease<'device>,
    /// Temporary GPU-visible allocation for host operands.
    staging: Option<Allocation>,
    /// Optional timestamp backing retained through completion.
    timing: Option<CopyTiming>,
    /// Terminal result that determines operand retention on failure.
    terminal_retention: Option<bool>,
}

impl<'device, 'cancel> GpuCopySequence<'device, 'cancel> {
    /// Acquires the native resources for an ordered copy sequence.
    ///
    /// # Errors
    /// Returns a target-capability or native resource failure.
    pub fn begin(
        gpu: GpuDevice<'device>,
        cancel: &'cancel AtomicBool,
    ) -> Result<Self, CopyFailure> {
        Self::begin_with_format(gpu, cancel, KernelQueueFormat::Sdma)
    }

    /// Acquires an ordered copy sequence on one qualified logical SDMA engine.
    ///
    /// # Errors
    /// Returns a target-capability, unavailable-engine, or native failure.
    pub fn begin_on_sdma_engine(
        gpu: GpuDevice<'device>,
        cancel: &'cancel AtomicBool,
        engine: u32,
    ) -> Result<Self, CopyFailure> {
        Self::begin_with_format(gpu, cancel, KernelQueueFormat::SdmaOnEngine(engine))
    }

    fn begin_with_format(
        gpu: GpuDevice<'device>,
        cancel: &'cancel AtomicBool,
        format: KernelQueueFormat,
    ) -> Result<Self, CopyFailure> {
        if !gpu.supports_linear_copy() {
            return Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "linear SDMA copy is not qualified for this GPU",
            }));
        }
        let resources = gpu.copy_resources(format)?;
        Ok(Self {
            gpu,
            cancel,
            resources,
            staging: None,
            timing: None,
            terminal_retention: None,
        })
    }

    fn active(&self) -> Result<(), CopyFailure> {
        if let Some(operands_may_be_live) = self.terminal_retention {
            return Err(CopyFailure {
                error: Error::Operation {
                    kind: ErrorKind::Busy,
                    detail: "SDMA copy sequence cannot resume after failure",
                },
                operands_may_be_live,
            });
        }
        Ok(())
    }

    fn record(&mut self, result: Result<(), CopyFailure>) -> Result<(), CopyFailure> {
        if let Err(failure) = &result {
            self.terminal_retention = Some(failure.operands_may_be_live);
            self.resources.discard();
        }
        result
    }

    fn release_staging(&mut self, retain: bool) {
        if let Some(staging) = self.staging.take() {
            if retain {
                std::mem::forget(staging);
            }
        }
    }

    /// Enables GPU-clock capture around this sequence's copy packets. Call
    /// before the first copy. The timestamp backing is owned until all native
    /// packets retire, including after a failed wait.
    ///
    /// # Errors
    /// Returns an invalid-state, allocation, or native mapping failure.
    #[allow(unsafe_code)]
    pub fn enable_timing(&mut self) -> Result<(), CopyFailure> {
        self.active()?;
        if self.timing.is_some() || self.resources.submitted_any {
            return Err(invalid(
                "SDMA copy timing must start before the first packet",
            ));
        }
        let (allocation, host_address, start) = match self
            .gpu
            .staging(16, DeviceAccess::READ | DeviceAccess::WRITE)
        {
            Ok(staging) => staging,
            Err(failure) => return self.record(Err(failure)),
        };
        let Some(end) = start.checked_add(8) else {
            return self.record(Err(invalid("SDMA timestamp address overflows")));
        };
        // SAFETY: The new system allocation owns 16 writable host bytes.
        unsafe { std::ptr::write_bytes(host_address as *mut u8, 0, 16) };
        fence(Ordering::Release);
        self.resources.timing = Some(TimingState {
            start,
            end,
            started: false,
        });
        self.timing = Some(CopyTiming {
            _allocation: allocation,
            host_address,
        });
        Ok(())
    }

    /// Returns raw global GPU ticks captured by the first and final copy
    /// packets after every operation in the sequence has retired.
    ///
    /// # Errors
    /// Returns an invalid-state or native timestamp failure.
    #[allow(unsafe_code)]
    pub fn finish_timing(&mut self) -> Result<GpuCopyTimestamps, CopyFailure> {
        self.active()?;
        if !self
            .resources
            .timing
            .as_ref()
            .is_some_and(|timing| timing.started)
        {
            return Err(invalid("SDMA copy timing has no completed copy"));
        }
        let timing = self
            .timing
            .take()
            .ok_or_else(|| invalid("SDMA copy timing has no backing"))?;
        self.resources.timing = None;
        fence(Ordering::Acquire);
        // SAFETY: The timing allocation is at least 16 bytes, page aligned,
        // and native wait proved the final timestamp write retired.
        let (start, end) = unsafe {
            let address = timing.host_address as *const u64;
            (address.read_volatile(), address.add(1).read_volatile())
        };
        if start == 0 || end == 0 || end < start {
            return Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::DriverContract,
                detail: "SDMA copy returned invalid GPU timestamps",
            }));
        }
        Ok(GpuCopyTimestamps { start, end })
    }

    /// Copies a non-overlapping GPU-addressable range after earlier sequence
    /// operations have retired.
    ///
    /// # Safety
    /// Both ranges must be GPU-accessible with source-read and
    /// destination-write permissions. Their backing must remain live until
    /// retirement, or until conclusive teardown after a failure with
    /// `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns range, cancellation, or native failures with the operand
    /// retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_linear(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
    ) -> Result<(), CopyFailure> {
        // SAFETY: A linear range is one row and one slice of the same mapped
        // source and destination backing required by this method.
        unsafe { self.copy_rect(CopyRect::linear(destination, source, size)) }
    }

    /// Copies through a gfx1250 packet with optional wait, signal, and
    /// execution-time address indirection. A wait applies before the first
    /// chunk and a signal after the last chunk.
    ///
    /// # Safety
    /// Direct ranges must remain mapped with their respective read or write
    /// permissions. Indirect addresses name readable, aligned 64-bit GPU
    /// pointer slots; their resolved ranges must remain mapped with the same
    /// permissions through native retirement. Wait and signal addresses must
    /// remain mapped with read and read/write access, respectively. All backing
    /// must survive an uncertain native submission.
    ///
    /// # Errors
    /// Returns an unsupported-target, range, cancellation, or native failure.
    #[allow(unsafe_code)]
    pub unsafe fn copy_linear_with_options(
        &mut self,
        destination: u64,
        source: u64,
        size: u64,
        options: GpuCopyOptions,
    ) -> Result<(), CopyFailure> {
        self.active()?;
        if options.is_plain() {
            // SAFETY: The caller supplied the same direct-range contract.
            return unsafe { self.copy_linear(destination, source, size) };
        }
        if self.gpu.info().target_profile != GpuTargetProfile::Gfx1250 {
            return self.record(Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "controlled SDMA copy is supported only on gfx1250",
            })));
        }
        if let Err(failure) = options.validate() {
            return self.record(Err(failure));
        }
        if size == 0
            || (options.indirect_source || options.indirect_destination)
                && size > MAX_INDIRECT_PACKET_BYTES
            || source == 0
            || destination == 0
            || options.indirect_source && source % 8 != 0
            || options.indirect_destination && destination % 8 != 0
            || source
                .checked_add(if options.indirect_source { 8 } else { size })
                .is_none()
            || destination
                .checked_add(if options.indirect_destination {
                    8
                } else {
                    size
                })
                .is_none()
        {
            return self.record(Err(invalid("invalid controlled SDMA copy range")));
        }
        let result =
            self.resources
                .copy_linear_controlled(destination, source, size, options, self.cancel);
        self.record(result)
    }

    /// Atomically exchanges two disjoint GPU ranges using gfx1250 SDMA.
    /// Each range must remain readable and writable through native retirement.
    ///
    /// # Safety
    /// Both ranges must be mapped to this GPU with read and write access. The
    /// caller retains their backing after an uncertain native submission.
    ///
    /// # Errors
    /// Returns an unsupported-target, range, cancellation, or native failure.
    #[allow(unsafe_code)]
    pub unsafe fn swap_linear(
        &mut self,
        address_a: u64,
        address_b: u64,
        size: u64,
    ) -> Result<(), CopyFailure> {
        // SAFETY: The caller supplied the same two-range contract.
        unsafe {
            self.swap_linear_with_options(address_a, address_b, size, GpuCopyOptions::default())
        }
    }

    /// Exchanges two gfx1250 ranges with optional packet wait and signal.
    ///
    /// # Safety
    /// Both ranges must remain mapped with read and write access. Optional
    /// wait and signal addresses must remain mapped with read and read/write access,
    /// respectively, through retirement or conclusive teardown.
    ///
    /// # Errors
    /// Returns an unsupported-target, range, cancellation, or native failure.
    #[allow(unsafe_code)]
    pub unsafe fn swap_linear_with_options(
        &mut self,
        address_a: u64,
        address_b: u64,
        size: u64,
        options: GpuCopyOptions,
    ) -> Result<(), CopyFailure> {
        self.active()?;
        if self.gpu.info().target_profile != GpuTargetProfile::Gfx1250 {
            return self.record(Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "native SDMA swap is supported only on gfx1250",
            })));
        }
        let end_a = address_a.checked_add(size);
        let end_b = address_b.checked_add(size);
        if options.indirect_source
            || options.indirect_destination
            || options.validate().is_err()
            || size == 0
            || address_a == 0
            || address_b == 0
            || address_a % 32 != 0
            || address_b % 32 != 0
            || end_a.is_none()
            || end_b.is_none()
            || (address_a < end_b.unwrap_or(0) && address_b < end_a.unwrap_or(0))
        {
            return self.record(Err(invalid("invalid SDMA swap ranges")));
        }
        let result = self
            .resources
            .swap_linear(address_a, address_b, size, options, self.cancel);
        self.record(result)
    }

    /// Copies a pitched, non-overlapping GPU-addressable rectangle after
    /// earlier sequence operations have retired. Timing spans the first
    /// through final native copy packet when enabled.
    ///
    /// # Safety
    /// Every selected source and destination row must remain GPU-accessible
    /// with the required permissions. Their backing must remain live until
    /// retirement, or until conclusive teardown after a failure with
    /// `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns shape, cancellation, or native failures with the operand
    /// retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_rect(&mut self, rect: CopyRect) -> Result<(), CopyFailure> {
        self.active()?;
        if let Err(failure) = rect.validate() {
            return self.record(Err(failure));
        }
        let result = self.resources.copy_rect(rect, self.cancel);
        self.record(result)
    }

    /// Copies a host slice to a GPU range. The source is sampled when this
    /// entry executes, after all preceding entries have retired.
    ///
    /// # Safety
    /// The destination must be GPU-mapped and writable through retirement or
    /// conclusive teardown after a failure with `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns staging, range, cancellation, or native failures with the GPU
    /// destination retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_from_host(
        &mut self,
        destination: u64,
        source: &[u8],
    ) -> Result<(), CopyFailure> {
        self.active()?;
        if source.is_empty() {
            return Ok(());
        }
        let (staging, host, device) = match self.gpu.staging(source.len(), DeviceAccess::READ) {
            Ok(staging) => staging,
            Err(failure) => return self.record(Err(failure)),
        };
        self.staging = Some(staging);
        // SAFETY: The staging allocation owns a writable host mapping of at
        // least source.len() bytes. The source slice is live for this call.
        unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), host as *mut u8, source.len()) };
        fence(Ordering::Release);
        // SAFETY: Staging and the caller's destination stay mapped through
        // retirement. An uncertain result retains staging below.
        let result = unsafe { self.copy_linear(destination, device, source.len() as u64) };
        self.release_staging(
            result
                .as_ref()
                .is_err_and(|failure| failure.operands_may_be_live),
        );
        result
    }

    /// Copies a GPU range into a host slice. The destination is written only
    /// after this entry's native retirement is proved.
    ///
    /// # Safety
    /// The source must be GPU-mapped and readable through retirement or
    /// conclusive teardown after a failure with `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns staging, range, cancellation, or native failures with the GPU
    /// source retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_to_host(
        &mut self,
        destination: &mut [u8],
        source: u64,
    ) -> Result<(), CopyFailure> {
        self.active()?;
        if destination.is_empty() {
            return Ok(());
        }
        let (staging, host, device) = match self
            .gpu
            .staging(destination.len(), DeviceAccess::READ | DeviceAccess::WRITE)
        {
            Ok(staging) => staging,
            Err(failure) => return self.record(Err(failure)),
        };
        self.staging = Some(staging);
        // SAFETY: Staging and the caller's source remain mapped until native
        // retirement. An uncertain result retains staging below.
        let result = unsafe { self.copy_linear(device, source, destination.len() as u64) };
        if let Err(failure) = result {
            self.release_staging(failure.operands_may_be_live);
            return Err(failure);
        }
        fence(Ordering::Acquire);
        // SAFETY: The completed GPU copy initialized the full staging range;
        // its host mapping and the destination slice stay live for this call.
        unsafe {
            std::ptr::copy_nonoverlapping(
                host as *const u8,
                destination.as_mut_ptr(),
                destination.len(),
            );
        };
        self.release_staging(false);
        Ok(())
    }
}

impl Drop for GpuCopySequence<'_, '_> {
    fn drop(&mut self) {
        self.resources.refresh_retirement();
        let retain = self.resources.pending.is_some() || self.resources.submitting;
        self.release_staging(retain);
        if let Some(timing) = self.timing.take() {
            if retain {
                std::mem::forget(timing);
            }
        }
    }
}

impl<'device> GpuDevice<'device> {
    fn copy_resources(
        &self,
        format: KernelQueueFormat,
    ) -> Result<CopyResourceLease<'device>, CopyFailure> {
        self.resources.copy_pool.take(self, format)
    }

    /// Prepares default and advertised-engine SDMA queues and command allocations
    /// for later copies or fills on this activated GPU. Subsequent calls leave
    /// healthy idle resources in place; concurrent copies acquire independent
    /// contexts.
    ///
    /// # Errors
    /// Returns an unqualified-target, allocation, or native queue failure.
    pub fn preload_linear_copy(&self) -> Result<(), CopyFailure> {
        if !self.supports_linear_copy() {
            return Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "linear SDMA copy is not qualified for this GPU",
            }));
        }
        let pool = &self.resources.copy_pool;
        pool.preload(self, KernelQueueFormat::Sdma)?;
        let mut engines = self
            .available_sdma_engines()
            .map_err(CopyFailure::retired)?;
        while engines != 0 {
            let engine = engines.trailing_zeros();
            pool.preload(self, KernelQueueFormat::SdmaOnEngine(engine))?;
            engines &= engines - 1;
        }
        Ok(())
    }

    fn staging(
        &self,
        size: usize,
        access: DeviceAccess,
    ) -> Result<(Allocation, usize, u64), CopyFailure> {
        let length = u64::try_from(size).map_err(|_| invalid("staged copy size is too large"))?;
        let alignment = crate::memory::host_page_size().map_err(CopyFailure::retired)?;
        let length = length
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| invalid("staged copy page rounding overflow"))?;
        let staging = self
            .device()
            .allocate(MemoryKind::System, length, alignment, access)
            .map_err(CopyFailure::retired)?;
        let host = staging
            .info()
            .host_address
            .ok_or_else(|| invalid("staged copy allocation has no host mapping"))?;
        let device = staging
            .device_address(self.device())
            .map_err(CopyFailure::retired)?;
        Ok((staging, host, device))
    }

    /// Reports whether this activated GPU has a qualified SDMA packet format
    /// and native submission path.
    #[must_use]
    pub fn supports_linear_copy(&self) -> bool {
        let info = self.info();
        match info.target_profile.sdma_dialect() {
            Some(GpuSdmaDialect::Gfx9) => cfg!(target_arch = "x86_64") && info.queues.sdma,
            Some(GpuSdmaDialect::Gfx1200 | GpuSdmaDialect::Gfx1250) => info.queues.kernel_sdma,
            None => false,
        }
    }

    /// Copies a non-overlapping device-address range through the GFX12.0 SDMA
    /// engine. The caller provides system-coherent operands and keeps them
    /// mapped until completion. Cancellation is checked between bounded native
    /// waits; it cannot revoke an already submitted packet.
    ///
    /// # Safety
    /// Both ranges must remain valid, accessible to this GPU, and carry the
    /// required source-read and destination-write permissions. The caller must
    /// retain their backing after an error with `operands_may_be_live` until
    /// device teardown proves that the GPU cannot reach them.
    ///
    /// # Errors
    /// Returns the native failure and the operand-retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_linear(
        &self,
        destination: u64,
        source: u64,
        size: u64,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        // SAFETY: The caller's range and retention obligations are unchanged
        // when represented as one row and one slice.
        unsafe { self.copy_rect(CopyRect::linear(destination, source, size), cancel) }
    }

    /// Copies a host slice into a GPU range through rocddi-owned staging.
    /// Source bytes are copied to staging before native submission; a native
    /// failure with uncertain retirement retains that staging for teardown.
    ///
    /// # Safety
    /// The destination range must stay GPU-mapped and writable until retirement
    /// or conclusive teardown after `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns allocation, native, or cancellation failures with the GPU
    /// destination retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_from_host(
        &self,
        destination: u64,
        source: &[u8],
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        if source.is_empty() {
            return Ok(());
        }
        let mut sequence = GpuCopySequence::begin(*self, cancel)?;
        // SAFETY: The sequence preserves the caller's range and retention
        // obligations while owning the native queue and staging allocation.
        unsafe { sequence.copy_from_host(destination, source) }
    }

    /// Copies a GPU range into a host slice through rocddi-owned staging.
    /// The host slice is written only after native retirement is proved.
    ///
    /// # Safety
    /// The source range must stay GPU-mapped and readable until retirement or
    /// conclusive teardown after `operands_may_be_live`.
    ///
    /// # Errors
    /// Returns allocation, native, or cancellation failures with the GPU
    /// source retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_to_host(
        &self,
        destination: &mut [u8],
        source: u64,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        if destination.is_empty() {
            return Ok(());
        }
        let mut sequence = GpuCopySequence::begin(*self, cancel)?;
        // SAFETY: The sequence preserves the caller's range and retention
        // obligations while owning the native queue and staging allocation.
        unsafe { sequence.copy_to_host(destination, source) }
    }

    /// Copies a pitched byte rectangle. Each row is split into bounded SDMA
    /// packets.
    ///
    /// # Safety
    /// Every selected source and destination row must remain valid, accessible
    /// to this GPU, and carry the required permissions. The caller must retain
    /// their backing after a failure with `operands_may_be_live` until native
    /// teardown proves that the GPU cannot reach it.
    ///
    /// # Errors
    /// Returns validation or native failure and its retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_rect(&self, rect: CopyRect, cancel: &AtomicBool) -> Result<(), CopyFailure> {
        // SAFETY: The single rectangle has the same operand lifetime and
        // permission requirements as a slice containing that rectangle.
        unsafe { self.copy_rects(std::slice::from_ref(&rect), cancel) }
    }

    /// Copies a sequence of pitched byte rectangles through one native queue
    /// and command allocation. Each row is split into bounded SDMA packets,
    /// and all shapes are checked before the first native submission.
    ///
    /// # Safety
    /// Every selected source and destination row must remain valid, accessible
    /// to this GPU, and carry the required permissions. The caller must retain
    /// all operand backing after a failure with `operands_may_be_live` until
    /// native teardown proves that the GPU cannot reach it.
    ///
    /// # Errors
    /// Returns validation or native failure and its retention requirement.
    #[allow(unsafe_code)]
    pub unsafe fn copy_rects(
        &self,
        rects: &[CopyRect],
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        for rect in rects {
            rect.validate()?;
        }
        if rects
            .iter()
            .all(|rect| rect.width == 0 || rect.height == 0 || rect.depth == 0)
        {
            return Ok(());
        }
        if !self.supports_linear_copy() {
            return Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "linear SDMA copy is not qualified for this GPU",
            }));
        }
        let mut resources = self.copy_resources(KernelQueueFormat::Sdma)?;
        for rect in rects {
            if let Err(failure) = resources.copy_rect(*rect, cancel) {
                resources.discard();
                return Err(failure);
            }
        }
        Ok(())
    }

    /// Fills a GPU-accessible dword range through the GFX1201 SDMA engine.
    /// Cancellation is checked between bounded native waits.
    ///
    /// # Safety
    /// The full destination range must stay mapped and writable by this GPU.
    /// Its backing must remain live after a failure with
    /// `operands_may_be_live` until native teardown proves retirement.
    ///
    /// # Errors
    /// Returns range, native, or cancellation failures with the retention rule.
    #[allow(unsafe_code)]
    pub unsafe fn fill_u32(
        &self,
        destination: u64,
        value: u32,
        count: u64,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        validate_fill_range(destination, count)?;
        if count == 0 {
            return Ok(());
        }
        if !self.supports_linear_copy() {
            return Err(CopyFailure::retired(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "SDMA fill is not qualified for this GPU",
            }));
        }
        let mut resources = self.copy_resources(KernelQueueFormat::Sdma)?;
        let mut completed = 0;
        while completed < count {
            let chunk = (count - completed).min(MAX_FILL_PACKET_DWORDS) as u32;
            let result = resources.submit_packet(
                SdmaCommand::Fill {
                    destination: destination + completed * 4,
                    value,
                    count: chunk,
                },
                cancel,
            );
            if let Err(failure) = result {
                resources.discard();
                return Err(failure);
            }
            completed += u64::from(chunk);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_range_checks_alignment_and_extent_before_native_acquisition() {
        assert!(validate_fill_range(0, 0).is_ok());
        assert!(validate_fill_range(0x1000, 1024).is_ok());
        for (address, count) in [(0, 1), (0x1001, 1), (u64::MAX - 3, 2), (0x1000, u64::MAX)] {
            assert_eq!(
                validate_fill_range(address, count)
                    .err()
                    .map(|failure| failure.error.kind()),
                Some(ErrorKind::InvalidArgument)
            );
        }
    }

    #[test]
    fn pitched_copy_rejects_unrepresentable_rows_before_native_acquisition() {
        let rect = CopyRect {
            destination: 0x1000,
            source: 0x2000,
            width: 32,
            height: 3,
            depth: 2,
            destination_pitch: 64,
            source_pitch: 64,
            destination_slice: 256,
            source_slice: 256,
        };
        assert!(rect.validate().is_ok());
        assert_eq!(
            CopyRect {
                source_pitch: 16,
                ..rect
            }
            .validate()
            .err()
            .map(|failure| failure.error.kind()),
            Some(ErrorKind::InvalidArgument)
        );
        assert_eq!(
            CopyRect {
                destination: u64::MAX - 8,
                ..rect
            }
            .validate()
            .err()
            .map(|failure| failure.error.kind()),
            Some(ErrorKind::InvalidArgument)
        );
        assert_eq!(
            CopyRect {
                source_slice: u64::MAX,
                ..rect
            }
            .validate()
            .err()
            .map(|failure| failure.error.kind()),
            Some(ErrorKind::InvalidArgument)
        );
    }
}
