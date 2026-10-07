// SPDX-License-Identifier: MIT

//! Linear copy and dword fill through a bounded native SDMA submission context.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering, fence};

use crate::gpu::GpuDevice;
use crate::kernel_queue::{KernelCommand, KernelQueue, KernelQueueFormat, KernelQueueWait};
use crate::memory::{Allocation, DeviceAccess, MemoryKind};
use crate::{Error, ErrorKind};

// GFX1201 uses the OSS5 COPY_LINEAR and GCR packet layouts. Keep the packet
// format here so either frontend can use the same native copy mechanism.
const COPY_PACKET_BYTES: usize = 28;
const FILL_PACKET_BYTES: usize = 20;
const GCR_PACKET_BYTES: usize = 20;
const COMMAND_BYTES: usize = GCR_PACKET_BYTES * 2 + COPY_PACKET_BYTES;
const FILL_COMMAND_BYTES: usize = GCR_PACKET_BYTES * 2 + FILL_PACKET_BYTES;
const MAX_COPY_PACKET_BYTES: u64 = 0x3f_ffe0;
const MAX_FILL_PACKET_BYTES: u64 = 0x3f_ffe0;
const MAX_FILL_PACKET_DWORDS: u64 = MAX_FILL_PACKET_BYTES / 4;
const SDMA_COPY_LINEAR: u32 = 1;
const SDMA_CONST_FILL_DWORD: u32 = 0x0b | (2 << 30);
const SDMA_USER_GCR: u32 = 0x11 | (1 << 8);
const GCR_WRITEBACK: u32 = (1 << 31) | (1 << 22);
const GCR_INVALIDATE: u32 = (1 << 30) | (1 << 25) | (1 << 24) | (1 << 23);
const NATIVE_WAIT_NS: u64 = 50_000_000;

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

fn encode_cache_envelope(command: &mut [u32; COMMAND_BYTES / 4], trailing_gcr: usize) {
    *command = [0; COMMAND_BYTES / 4];
    command[0] = SDMA_USER_GCR;
    command[2] = GCR_WRITEBACK | GCR_INVALIDATE;
    command[trailing_gcr] = SDMA_USER_GCR;
    command[trailing_gcr + 2] = GCR_WRITEBACK;
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses are deliberately split into their low and high dwords"
)]
fn encode_copy(command: &mut [u32; COMMAND_BYTES / 4], destination: u64, source: u64, size: u32) {
    // The command allocation is reused only after native retirement. Every
    // word is written, including reserved fields, before the release fence.
    encode_cache_envelope(command, 12);
    command[5] = SDMA_COPY_LINEAR;
    command[6] = size - 1;
    command[8] = source as u32;
    command[9] = (source >> 32) as u32;
    command[10] = destination as u32;
    command[11] = (destination >> 32) as u32;
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the packet address is deliberately split into low and high dwords"
)]
fn encode_fill(command: &mut [u32; COMMAND_BYTES / 4], destination: u64, value: u32, count: u32) {
    encode_cache_envelope(command, 10);
    command[5] = SDMA_CONST_FILL_DWORD;
    command[6] = destination as u32;
    command[7] = (destination >> 32) as u32;
    command[8] = value;
    command[9] = (count - 1) * 4;
}

#[derive(Clone, Copy)]
enum SdmaCommand {
    Copy {
        destination: u64,
        source: u64,
        size: u32,
    },
    Fill {
        destination: u64,
        value: u32,
        count: u32,
    },
}

impl SdmaCommand {
    const fn byte_length(&self) -> u64 {
        match self {
            Self::Copy { .. } => COMMAND_BYTES as u64,
            Self::Fill { .. } => FILL_COMMAND_BYTES as u64,
        }
    }

    fn encode(&self, words: &mut [u32; COMMAND_BYTES / 4]) {
        match *self {
            Self::Copy {
                destination,
                source,
                size,
            } => encode_copy(words, destination, source, size),
            Self::Fill {
                destination,
                value,
                count,
            } => encode_fill(words, destination, value, count),
        }
    }
}

/// Native resources for one operation. An unretired submission may still read
/// the indirect buffer; Drop keeps both owners for process teardown then.
struct CopyResources {
    queue: ManuallyDrop<KernelQueue>,
    command: ManuallyDrop<Allocation>,
    host_address: usize,
    command_address: u64,
    pending: Option<u64>,
    submitting: bool,
}

impl CopyResources {
    fn new(gpu: &GpuDevice<'_>, format: KernelQueueFormat) -> Result<Self, CopyFailure> {
        let queue = gpu
            .create_kernel_queue(format)
            .map_err(CopyFailure::retired)?;
        let command = gpu
            .device()
            .allocate(
                MemoryKind::System,
                4096,
                4096,
                DeviceAccess::READ | DeviceAccess::EXECUTE,
            )
            .map_err(CopyFailure::retired)?;
        let mut resources = Self {
            queue: ManuallyDrop::new(queue),
            command: ManuallyDrop::new(command),
            host_address: 0,
            command_address: 0,
            pending: None,
            submitting: false,
        };
        resources.host_address = resources.command.info().host_address.ok_or_else(|| {
            resources.failure(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "native command backing has no host mapping",
            })
        })?;
        resources.command_address = resources
            .command
            .device_address(gpu.device())
            .map_err(|error| resources.failure(error))?;
        Ok(resources)
    }

    fn failure(&self, error: Error) -> CopyFailure {
        CopyFailure {
            error,
            operands_may_be_live: self.pending.is_some() || self.submitting,
        }
    }

    fn refresh_retirement(&mut self) {
        if self
            .pending
            .is_some_and(|submission| self.queue.status().retired_submission >= submission)
        {
            self.pending = None;
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
        // SAFETY: The allocation owns a writable host mapping of at least one
        // page. The command is at most 68 bytes and no prior submission reaches
        // it before this write.
        let words = unsafe { &mut *(self.host_address as *mut [u32; COMMAND_BYTES / 4]) };
        command.encode(words);
        fence(Ordering::Release);
        self.submitting = true;
        // SAFETY: Packet bytes, synchronization, and backing lifetime are
        // established by this resource owner and the caller's operand contract.
        let submission = unsafe {
            self.queue.submit(KernelCommand {
                device_address: self.command_address,
                byte_length: command.byte_length(),
            })
        };
        // The driver contract reports ambiguity as an accepted identity, so
        // an error proves rejection. A panic before this line keeps submitting
        // set, so Drop retains the command and queue backing.
        self.submitting = false;
        let submission = submission.map_err(|error| self.failure(error))?;
        self.pending = Some(submission);
        loop {
            let wait = self.queue.wait(submission, NATIVE_WAIT_NS, 0);
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
                        },
                        cancel,
                    )?;
                    completed += u64::from(chunk);
                }
            }
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
        let _ = self.queue.destroy();
        let _ = self.command.free();
        // SAFETY: Pending native access was ruled out above. These fields are
        // dropped exactly once and remain owned by this resource bundle.
        unsafe {
            ManuallyDrop::drop(&mut self.queue);
            ManuallyDrop::drop(&mut self.command);
        }
    }
}

/// An ordered GFX1201 SDMA copy sequence sharing one native queue and command
/// allocation. Each operation retires before the next begins. A failed
/// operation makes the sequence terminal, so an unretired packet cannot be
/// replaced in the shared command allocation.
pub struct GpuCopySequence<'device, 'cancel> {
    gpu: GpuDevice<'device>,
    cancel: &'cancel AtomicBool,
    resources: CopyResources,
    staging: Option<Allocation>,
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

    /// Acquires an ordered copy sequence on one selected DRM DMA ring.
    ///
    /// # Errors
    /// Returns a target-capability, unavailable-ring, or native failure.
    pub fn begin_on_sdma_ring(
        gpu: GpuDevice<'device>,
        cancel: &'cancel AtomicBool,
        ring: u32,
    ) -> Result<Self, CopyFailure> {
        Self::begin_with_format(gpu, cancel, KernelQueueFormat::SdmaOnRing(ring))
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
        Ok(Self {
            gpu,
            cancel,
            resources: CopyResources::new(&gpu, format)?,
            staging: None,
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
        self.active()?;
        let rect = CopyRect::linear(destination, source, size);
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
        self.release_staging(self.resources.pending.is_some() || self.resources.submitting);
    }
}

impl GpuDevice<'_> {
    fn staging(
        &self,
        size: usize,
        access: DeviceAccess,
    ) -> Result<(Allocation, usize, u64), CopyFailure> {
        let length = u64::try_from(size).map_err(|_| invalid("staged copy size is too large"))?;
        let alignment = crate::memory::host_page_size().map_err(CopyFailure::retired)?;
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

    /// Reports whether this activated GPU exposes the GFX1201 OSS5 packet
    /// format and bounded native SDMA submission path.
    #[must_use]
    pub fn supports_linear_copy(&self) -> bool {
        let info = self.info();
        (info.gfx_major, info.gfx_minor, info.gfx_stepping) == (12, 0, 1) && info.queues.kernel_sdma
    }

    /// Copies a non-overlapping device-address range through the GFX1201 SDMA
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
        let mut resources = CopyResources::new(self, KernelQueueFormat::Sdma)?;
        for rect in rects {
            resources.copy_rect(*rect, cancel)?;
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
        let mut resources = CopyResources::new(self, KernelQueueFormat::Sdma)?;
        let mut completed = 0;
        while completed < count {
            let chunk = (count - completed).min(MAX_FILL_PACKET_DWORDS) as u32;
            resources.submit_packet(
                SdmaCommand::Fill {
                    destination: destination + completed * 4,
                    value,
                    count: chunk,
                },
                cancel,
            )?;
            completed += u64::from(chunk);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gfx1201_linear_copy_encodes_system_cache_controls_and_full_addresses() {
        let mut command = [u32::MAX; COMMAND_BYTES / 4];
        encode_copy(
            &mut command,
            0x1234_5678_9abc_def0,
            0xfeed_face_cafe_babe,
            4096,
        );
        assert_eq!(command[0], SDMA_USER_GCR);
        assert_eq!(command[2], GCR_WRITEBACK | GCR_INVALIDATE);
        assert_eq!(
            command[5..12],
            [
                1,
                4095,
                0,
                0xcafe_babe,
                0xfeed_face,
                0x9abc_def0,
                0x1234_5678
            ]
        );
        assert_eq!(command[12], SDMA_USER_GCR);
        assert_eq!(command[14], GCR_WRITEBACK);
        assert_eq!(command[16], 0);
    }

    #[test]
    fn gfx1201_fill_encodes_dword_count_and_cache_controls() {
        let mut command = [u32::MAX; COMMAND_BYTES / 4];
        encode_fill(&mut command, 0x1234_5678_9abc_def0, 0xaabb_ccdd, 1024);
        assert_eq!(command[0], SDMA_USER_GCR);
        assert_eq!(command[2], GCR_WRITEBACK | GCR_INVALIDATE);
        assert_eq!(command[5], SDMA_CONST_FILL_DWORD);
        assert_eq!(
            command[6..10],
            [0x9abc_def0, 0x1234_5678, 0xaabb_ccdd, 4092]
        );
        assert_eq!(command[10], SDMA_USER_GCR);
        assert_eq!(command[12], GCR_WRITEBACK);
        assert_eq!(command[14], 0);
    }

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
