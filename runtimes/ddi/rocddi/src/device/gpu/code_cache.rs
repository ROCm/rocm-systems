// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Instruction-cache publication using the AMD AQL PM4 indirect-buffer packet.

use super::queue::{
    QueueParameters, QueuePriority, QueueProducerMode, QueueRequest, QueueRingMemory,
    RetainedQueue, store_ordered_mmio64,
};
use super::{CopyFailure, GpuDevice};
use crate::memory::{DeviceAccess, MemoryKind, host_page_size};
use crate::{Error, ErrorKind};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Separate cache lines between the completion signal and the indirect buffer.
const IB_ENTRY_MARKER_OFFSET: usize = 64;
/// Separate cache line for the write following the cache command.
const IB_EXIT_MARKER_OFFSET: usize = 128;
/// Indirect-buffer byte offset in the retained command allocation.
const INDIRECT_BUFFER_OFFSET: usize = 256;

/// Native AMD completion signal layout; no public HSA handle is registered.
#[repr(C, align(64))]
struct Completion {
    /// `AMD_SIGNAL_KIND_USER`.
    kind: i64,
    /// CP decrements this after completing the PM4 indirect buffer.
    value: AtomicI64,
    /// Unused event, profiling and queue fields.
    reserved: [u64; 6],
}

/// The first condition that ended the completion wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitOutcome {
    Completed,
    Cancelled,
    Deadline,
}

/// Retains native error identity and live backing while annotating retirement.
fn inactivation_failure(mut error: Error, outcome: WaitOutcome) -> CopyFailure {
    if let Error::NativeOperation { operation, .. } = &mut error {
        *operation = match (*operation, outcome) {
            ("AMDKFD_IOC_UPDATE_QUEUE", WaitOutcome::Completed) => {
                "AMDKFD_IOC_UPDATE_QUEUE during code-cache queue inactivation (wait=Completed)"
            }
            ("AMDKFD_IOC_UPDATE_QUEUE", WaitOutcome::Cancelled) => {
                "AMDKFD_IOC_UPDATE_QUEUE during code-cache queue inactivation (wait=Cancelled)"
            }
            ("AMDKFD_IOC_UPDATE_QUEUE", WaitOutcome::Deadline) => {
                "AMDKFD_IOC_UPDATE_QUEUE during code-cache queue inactivation (wait=Deadline)"
            }
            (operation, _) => operation,
        };
    }
    CopyFailure {
        error,
        operands_may_be_live: true,
    }
}

/// Writes one snapshot without replacing the publication or cleanup result.
#[cold]
fn write_failure_diagnostic(
    output: &mut impl Write,
    outcome: WaitOutcome,
    (read, write): (u64, u64),
    completion: i64,
    (entry, exit): (u32, u32),
) -> io::Result<()> {
    writeln!(
        output,
        "rocddi code-cache: wait={outcome:?}; read={read}; write={write}; completion={completion}; ib_entry={entry}; ib_exit={exit}"
    )
}

/// Builds the full-range `ACQUIRE_MEM` cache operation used by `ROCr`'s loader.
fn invalidate_packet(major: u32, minor: u32) -> ([u32; 8], usize) {
    let mut words = [0; 8];
    let count: u32 = if major < 10 { 7 } else { 8 };
    words[0] = 0xc000_5800 | ((count - 2) << 16);
    words[2] = u32::MAX;
    words[3] = 0xff;
    if major < 10 {
        words[1] = (1 << 18) | (1 << 23) | (1 << 27) | (1 << 29);
    } else if major == 12 && minor == 5 {
        // GLI_INV_ALL, GL2_INV and reverse ordering. Bit 9 is reserved on 12.5.
        words[7] = 1 | (1 << 14) | (2 << 16);
    } else {
        words[7] = 1 | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 14);
    }
    (words, count as usize)
}

/// Writes a diagnostic word using `ROCr`'s confirmed, memory-destination recipe.
#[allow(
    clippy::cast_possible_truncation,
    reason = "PM4 carries the aligned 64-bit address in separate low and high words"
)]
fn marker_packet(address: u64) -> [u32; 5] {
    [
        0xc003_3700,          // WRITE_DATA, five dwords.
        (2 << 8) | (1 << 20), // TC_L2 destination, wait for write confirmation.
        address as u32,
        (address >> 32) as u32,
        1,
    ]
}

/// Keeps the normal cache command intact, optionally bracketing it with writes.
fn publication_commands(
    major: u32,
    minor: u32,
    storage_address: u64,
    diagnostics: bool,
) -> ([u32; 18], usize) {
    let (invalidate, count) = invalidate_packet(major, minor);
    let mut words = [0; 18];
    if diagnostics {
        words[..5].copy_from_slice(&marker_packet(
            storage_address + IB_ENTRY_MARKER_OFFSET as u64,
        ));
        words[5..5 + count].copy_from_slice(&invalidate[..count]);
        words[5 + count..10 + count].copy_from_slice(&marker_packet(
            storage_address + IB_EXIT_MARKER_OFFSET as u64,
        ));
        (words, count + 10)
    } else {
        words[..count].copy_from_slice(&invalidate[..count]);
        (words, count)
    }
}

/// Wraps cache commands in the AMD AQL PM4 indirect-buffer packet.
fn publication_packet(
    indirect_buffer_address: u64,
    completion_address: u64,
    command_count: usize,
) -> Result<[u32; 16], CopyFailure> {
    let mut packet = [0_u32; 16];
    // VENDOR_SPECIFIC packet, AMD_AQL_FORMAT_PM4_IB. Cache maintenance is
    // explicit in the IB; match ROCr's NONE/NONE wrapper fence scopes.
    packet[0] = 1 << 16;
    packet[1] = 0xc002_3f00; // INDIRECT_BUFFER, four dwords.
    packet[2] = u32::try_from(indirect_buffer_address & u64::from(u32::MAX))
        .map_err(|_| invalid(ErrorKind::Internal, "IB low word overflow"))?;
    packet[3] = u32::try_from(indirect_buffer_address >> 32)
        .map_err(|_| invalid(ErrorKind::Internal, "IB high word overflow"))?;
    packet[4] = (1 << 23)
        | u32::try_from(command_count)
            .map_err(|_| invalid(ErrorKind::Internal, "IB count overflow"))?;
    packet[5] = 0xa;
    packet[14] = u32::try_from(completion_address & u64::from(u32::MAX))
        .map_err(|_| invalid(ErrorKind::Internal, "signal low word overflow"))?;
    packet[15] = u32::try_from(completion_address >> 32)
        .map_err(|_| invalid(ErrorKind::Internal, "signal high word overflow"))?;
    Ok(packet)
}

/// Wraps an error after all native users of caller resources have retired.
fn retired(error: Error) -> CopyFailure {
    CopyFailure {
        error,
        operands_may_be_live: false,
    }
}

/// Constructs a cache-publication validation or cancellation error.
fn invalid(kind: ErrorKind, detail: &'static str) -> CopyFailure {
    retired(Error::Operation { kind, detail })
}

impl GpuDevice<'_> {
    /// Samples retained resources only after rechecking the native process owner.
    #[cold]
    #[allow(unsafe_code)]
    fn diagnose_wait_failure(
        &self,
        outcome: WaitOutcome,
        queue: &RetainedQueue<(crate::memory::Allocation, crate::device::Device)>,
        completion: &Completion,
        markers: Option<(usize, usize)>,
    ) {
        let Ok(_operation) = self.driver.admit() else {
            return;
        };
        if let Some((entry, exit)) = markers {
            let transport = queue.info();
            // SAFETY: Creation validated aligned 64-bit, nonwrapping AQL
            // indices, and this private owner retains both mappings until
            // after this sample. Direct loads avoid progress()'s exception
            // polling, event resets and sticky device-loss updates.
            let progress = unsafe {
                (
                    (&*(transport.read_index_host_address as *const AtomicU64))
                        .load(Ordering::Acquire),
                    (&*(transport.write_index_host_address as *const AtomicU64))
                        .load(Ordering::Acquire),
                )
            };
            let completion = completion.value.load(Ordering::Acquire);
            // SAFETY: These aligned words occupy disjoint cache lines in the
            // retained writable command allocation. Sample each once, without
            // polling or requiring progress to resume.
            let markers = unsafe {
                (
                    (&*(entry as *const AtomicU32)).load(Ordering::Acquire),
                    (&*(exit as *const AtomicU32)).load(Ordering::Acquire),
                )
            };
            let _ = write_failure_diagnostic(
                &mut std::io::stderr().lock(),
                outcome,
                progress,
                completion,
                markers,
            );
        }
    }

    /// Publishes newly written code to instruction and scalar caches.
    ///
    /// The current mechanism invalidates the full cache range. The supplied
    /// range is validated and describes the caller's executable storage.
    /// Internal packet and completion storage survives any uncertain native
    /// failure; callers must retain their executable on a live-operand error.
    ///
    /// # Errors
    /// Returns validation, allocation, submission, cancellation or retirement
    /// failures. An incomplete invalidation is never reported as success.
    #[allow(
        unsafe_code,
        clippy::too_many_lines,
        reason = "one private queue transaction retains its backing through checked retirement"
    )]
    pub fn invalidate_code_cache(
        &self,
        address: u64,
        byte_length: u64,
        cancel: &AtomicBool,
    ) -> Result<(), CopyFailure> {
        let _operation = self.driver.admit().map_err(retired)?;
        if address == 0 || byte_length == 0 || address.checked_add(byte_length).is_none() {
            return Err(invalid(
                ErrorKind::InvalidArgument,
                "invalid executable cache range",
            ));
        }
        let info = self.info();
        if info.queues.aql.is_none() || !(9..=12).contains(&info.gfx_major) {
            return Err(invalid(
                ErrorKind::Unsupported,
                "target has no AQL code-cache publication recipe",
            ));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(invalid(
                ErrorKind::Busy,
                "code-cache publication cancelled before submission",
            ));
        }
        let page = host_page_size().map_err(retired)?.max(4096);
        // CP fetches the indirect buffer from this page, which also contains
        // the completion signal it writes. The GPU mapping needs both rights.
        let storage = self
            .device()
            .allocate(
                MemoryKind::System,
                page,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
            )
            .map_err(retired)?;
        let host = storage.info().host_address.ok_or_else(|| {
            invalid(
                ErrorKind::Unsupported,
                "cache command needs host-visible backing",
            )
        })?;
        let gpu_address = storage.device_address(self.device()).map_err(retired)?;
        if gpu_address != host as u64 || gpu_address >> 48 != 0 {
            return Err(invalid(
                ErrorKind::Unsupported,
                "cache command requires a shared 48-bit address",
            ));
        }
        let diagnostics =
            std::env::var_os("ROCDDI_CODE_CACHE_DIAGNOSTICS").is_some_and(|value| value == "1");
        let (command, count) =
            publication_commands(info.gfx_major, info.gfx_minor, gpu_address, diagnostics);
        // SAFETY: Completion [0,64), optional marker words at 64 and 128,
        // and up to 18 IB dwords at 256 fit disjointly in this >=4096-byte page.
        unsafe {
            (host as *mut Completion).write(Completion {
                kind: 1,
                value: AtomicI64::new(1),
                reserved: [0; 6],
            });
            if diagnostics {
                ((host + IB_ENTRY_MARKER_OFFSET) as *mut AtomicU32).write(AtomicU32::new(0));
                ((host + IB_EXIT_MARKER_OFFSET) as *mut AtomicU32).write(AtomicU32::new(0));
            }
            std::ptr::copy_nonoverlapping(
                command.as_ptr(),
                (host + INDIRECT_BUFFER_OFFSET) as *mut u32,
                count,
            );
        }
        // SAFETY: This method is the sole producer. The dependency bundle owns
        // every address the packet can read, even if queue creation is partial.
        let mut queue = unsafe {
            self.create_queue_with_dependencies(
                QueueRequest {
                    control_memory: crate::device::gpu::queue::QueueRingMemory::System,
                    ring_size_bytes: 4096,
                    parameters: QueueParameters::Aql {
                        producer_mode: QueueProducerMode::Single,
                        ring_memory: QueueRingMemory::System,
                        global_work_sync: false,
                        inactive_signal: None,
                        error_event: None,
                        scratch: None,
                    },
                    priority: QueuePriority::Normal,
                    device_producer: false,
                },
                (storage, self.device().clone()),
            )
        }
        .map_err(|error| CopyFailure {
            operands_may_be_live: matches!(error, Error::QueueBackingMayBeLive { .. }),
            error,
        })?;
        let transport = queue.info();
        let ib = gpu_address + INDIRECT_BUFFER_OFFSET as u64;
        let packet = publication_packet(ib, gpu_address, count)?;
        // SAFETY: Native transport supplies an aligned writable ring and index;
        // all body and IB stores precede the release header and MMIO doorbell.
        unsafe {
            let ring = transport.ring_host_address as *mut u32;
            std::ptr::copy_nonoverlapping(packet.as_ptr().add(1), ring.add(1), 15);
            (&*ring.cast::<AtomicU32>()).store(packet[0], Ordering::Release);
            (&*(transport.write_index_host_address as *const AtomicU64))
                .store(1, Ordering::Release);
            store_ordered_mmio64(transport.doorbell_host_address, 0);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let outcome = loop {
            // SAFETY: Queue dependencies own the initialized signal page.
            if unsafe { &*(host as *const Completion) }
                .value
                .load(Ordering::Acquire)
                == 0
            {
                break WaitOutcome::Completed;
            }
            if cancel.load(Ordering::Acquire) {
                break WaitOutcome::Cancelled;
            }
            if Instant::now() >= deadline {
                break WaitOutcome::Deadline;
            }
            std::thread::sleep(Duration::from_micros(50));
        };
        if outcome != WaitOutcome::Completed {
            // SAFETY: The retained dependency bundle still owns this signal.
            self.diagnose_wait_failure(
                outcome,
                &queue,
                unsafe { &*(host as *const Completion) },
                diagnostics
                    .then_some((host + IB_ENTRY_MARKER_OFFSET, host + IB_EXIT_MARKER_OFFSET)),
            );
        }
        queue
            .inactivate()
            .map_err(|error| inactivation_failure(error, outcome))?;
        // SAFETY: The sole producer has stopped and CP is inactivated. The
        // retained owner releases its IB and signal only on native success.
        unsafe { queue.destroy() }.map_err(|error| CopyFailure {
            error,
            operands_may_be_live: true,
        })?;
        match outcome {
            WaitOutcome::Completed => Ok(()),
            WaitOutcome::Cancelled => Err(invalid(
                ErrorKind::Busy,
                "code-cache publication did not complete (wait=Cancelled)",
            )),
            WaitOutcome::Deadline => Err(invalid(
                ErrorKind::Busy,
                "code-cache publication did not complete (wait=Deadline)",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Completion, IB_ENTRY_MARKER_OFFSET, IB_EXIT_MARKER_OFFSET, INDIRECT_BUFFER_OFFSET,
        WaitOutcome, inactivation_failure, invalidate_packet, publication_commands,
        publication_packet, write_failure_diagnostic,
    };
    use crate::{Error, ErrorKind};
    use std::io;

    #[test]
    fn publication_packet_matches_rocr_cache_wrapper() -> Result<(), Error> {
        for (command_count, indirect_buffer_control) in [(7, 0x0080_0007), (8, 0x0080_0008)] {
            let packet =
                publication_packet(0x0000_0123_4567_8900, 0x0000_abcd_ef01_2000, command_count)
                    .map_err(|failure| failure.error)?;
            // ROCr's cache invalidation uses ExecutePM4's NONE/NONE defaults.
            assert_eq!(
                packet,
                [
                    0x0001_0000,
                    0xc002_3f00,
                    0x4567_8900,
                    0x0000_0123,
                    indirect_buffer_control,
                    0xa,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0xef01_2000,
                    0x0000_abcd,
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn diagnostic_stream_brackets_the_unchanged_cache_command() -> Result<(), Error> {
        // ROCr amd_gpu_pm4.h encodes WRITE_DATA as opcode 0x37, size 5,
        // destination TC_L2 (2 << 8), confirmation (1 << 20). Its cache
        // commands are GFX9's seven words and GFX10+'s eight words.
        let storage_address = 0x0000_1234_5678_9000;
        for (major, minor, cache) in [
            (
                9,
                0,
                vec![0xc005_5800, 0x2884_0000, u32::MAX, 0xff, 0, 0, 0],
            ),
            (11, 0, vec![0xc006_5800, 0, u32::MAX, 0xff, 0, 0, 0, 0x4381]),
            (
                12,
                5,
                vec![0xc006_5800, 0, u32::MAX, 0xff, 0, 0, 0, 0x24001],
            ),
        ] {
            let (normal, normal_count) = publication_commands(major, minor, storage_address, false);
            assert_eq!(&normal[..normal_count], cache);
            assert!(normal[normal_count..].iter().all(|word| *word == 0));

            let (marked, marked_count) = publication_commands(major, minor, storage_address, true);
            let mut expected = vec![0xc003_3700, 0x0010_0200, 0x5678_9040, 0x0000_1234, 1];
            expected.extend_from_slice(&cache);
            expected.extend_from_slice(&[0xc003_3700, 0x0010_0200, 0x5678_9080, 0x0000_1234, 1]);
            assert_eq!(marked_count, cache.len() + 10);
            assert_eq!(&marked[..marked_count], expected);
            assert!(marked[marked_count..].iter().all(|word| *word == 0));

            let packet = publication_packet(
                storage_address + INDIRECT_BUFFER_OFFSET as u64,
                storage_address,
                marked_count,
            )
            .map_err(|failure| failure.error)?;
            assert_eq!(packet[0], 0x0001_0000);
            assert_eq!(packet[1], 0xc002_3f00);
            assert_eq!(packet[2], 0x5678_9100);
            assert_eq!(packet[3], 0x0000_1234);
            assert_eq!(
                packet[4],
                if major == 9 { 0x0080_0011 } else { 0x0080_0012 }
            );
            assert_eq!(packet[5], 0xa);
            assert_eq!(&packet[6..14], &[0; 8]);
            assert_eq!(packet[14], 0x5678_9000);
            assert_eq!(packet[15], 0x0000_1234);
        }
        Ok(())
    }

    #[test]
    fn diagnostic_storage_is_disjoint_and_fits_the_retained_page() {
        assert_eq!(size_of::<Completion>(), 64);
        assert!(size_of::<Completion>() <= IB_ENTRY_MARKER_OFFSET);
        assert_eq!(IB_ENTRY_MARKER_OFFSET % align_of::<u32>(), 0);
        assert_eq!(IB_EXIT_MARKER_OFFSET % align_of::<u32>(), 0);
        assert!(IB_ENTRY_MARKER_OFFSET + size_of::<u32>() <= IB_EXIT_MARKER_OFFSET);
        assert!(IB_EXIT_MARKER_OFFSET + size_of::<u32>() <= INDIRECT_BUFFER_OFFSET);
        assert_eq!(INDIRECT_BUFFER_OFFSET % align_of::<u32>(), 0);
        assert!(INDIRECT_BUFFER_OFFSET + 18 * size_of::<u32>() <= 4096);
    }

    #[test]
    fn inactivation_context_preserves_native_error_and_live_operands() {
        for outcome in [
            WaitOutcome::Completed,
            WaitOutcome::Cancelled,
            WaitOutcome::Deadline,
        ] {
            let error = crate::os::native_error(
                "AMDKFD_IOC_UPDATE_QUEUE",
                io::Error::from_raw_os_error(13),
            );
            let status = error.native_status();
            let kind = error.kind();
            let failure = inactivation_failure(error, outcome);
            assert!(failure.operands_may_be_live);
            assert_eq!(failure.error.kind(), kind);
            assert_eq!(failure.error.native_status(), status);
            assert!(
                failure
                    .error
                    .to_string()
                    .contains(&format!("wait={outcome:?}"))
            );
        }
        for error in [
            crate::os::native_error("KFD queue inactivation", io::Error::from_raw_os_error(13)),
            Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "queue is not available",
            },
        ] {
            let text = error.to_string();
            let status = error.native_status();
            let kind = error.kind();
            let failure = inactivation_failure(error, WaitOutcome::Deadline);
            assert!(failure.operands_may_be_live);
            assert_eq!(failure.error.to_string(), text);
            assert_eq!(failure.error.kind(), kind);
            assert_eq!(failure.error.native_status(), status);
        }
    }

    #[test]
    fn failure_diagnostic_records_passive_snapshot() -> io::Result<()> {
        let mut output = Vec::new();
        write_failure_diagnostic(&mut output, WaitOutcome::Deadline, (0, 1), 1, (1, 0))?;
        assert_eq!(
            output,
            b"rocddi code-cache: wait=Deadline; read=0; write=1; completion=1; ib_entry=1; ib_exit=0\n"
        );
        output.clear();
        write_failure_diagnostic(&mut output, WaitOutcome::Cancelled, (1, 1), -1, (1, 1))?;
        assert_eq!(
            output,
            b"rocddi code-cache: wait=Cancelled; read=1; write=1; completion=-1; ib_entry=1; ib_exit=1\n"
        );
        let mut full = &mut [0_u8; 0][..];
        assert!(
            write_failure_diagnostic(&mut full, WaitOutcome::Deadline, (0, 1), 1, (0, 0)).is_err()
        );
        Ok(())
    }

    #[test]
    fn cache_packets_match_generation_register_layouts() {
        let (words, size) = invalidate_packet(9, 4);
        assert_eq!(size, 7);
        assert_eq!(
            words,
            [0xc005_5800, 0x2884_0000, 0xffff_ffff, 0xff, 0, 0, 0, 0]
        );
        for (major, minor) in [(10, 1), (10, 3), (11, 0), (11, 5), (12, 0)] {
            let (words, size) = invalidate_packet(major, minor);
            assert_eq!(size, 8);
            assert_eq!(words, [0xc006_5800, 0, 0xffff_ffff, 0xff, 0, 0, 0, 0x4381]);
        }
        assert_eq!(invalidate_packet(12, 5).0[7], 0x24001);
    }
}
