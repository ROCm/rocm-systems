// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Instruction-cache publication using the AMD AQL PM4 indirect-buffer packet.

use super::queue::{
    QueueParameters, QueuePriority, QueueProducerMode, QueueRequest, QueueRingMemory,
    QueueTransport, store_ordered_mmio64,
};
use super::{CopyFailure, GpuDevice};
use crate::memory::{DeviceAccess, MemoryKind, host_page_size};
use crate::{Error, ErrorKind};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

/// Writes a one-time sample without replacing the publication or cleanup error.
#[cold]
fn write_failure_state_diagnostic(
    output: &mut impl Write,
    boundary: &str,
    completed: bool,
    (read, write): (u64, u64),
    completion: i64,
) -> io::Result<()> {
    writeln!(
        output,
        "rocddi code-cache: boundary={boundary}; wait_completed={completed}; read={read}; write={write}; completion={completion}"
    )
}

/// Records numeric addresses supplied by retained owners and cached transport.
#[cold]
fn write_failure_address_diagnostic(
    output: &mut impl Write,
    (executable_address, executable_byte_length): (u64, u64),
    (host, gpu_address, page): (usize, u64, u64),
    (indirect_buffer_address, command_count): (u64, usize),
    transport: QueueTransport,
) -> io::Result<()> {
    let completion_value_offset = std::mem::offset_of!(Completion, value);
    write!(
        output,
        concat!(
            "rocddi code-cache: addresses; ",
            "logical_executable_gpu={executable_address:#x}; logical_executable_size_bytes={executable_byte_length}; ",
            "native_executable_allocation=unavailable(not_retained); ",
            "command_page_cpu={host:#x}; command_page_gpu={gpu_address:#x}; command_page_size_bytes={page}; ",
            "completion_cpu={host:#x}; completion_gpu={gpu_address:#x}; completion_size_bytes={completion_size}; ",
            "completion_value_cpu={completion_value_host:#x}; completion_value_gpu={completion_value_device:#x}; ",
            "ib_cpu={indirect_buffer_host:#x}; ib_gpu={indirect_buffer_address:#x}; ib_words={command_count}; ",
            "ring_cpu={ring_host:#x}; ring_gpu={ring_device:#x}; ring_size_bytes={ring_size}; ",
            "read_index_cpu={read_index_host:#x}; read_index_gpu={read_index_device:#x}; ",
            "write_index_cpu={write_index_host:#x}; write_index_gpu={write_index_device:#x}; ",
            "doorbell_cpu={doorbell_host:#x}; doorbell_gpu="
        ),
        executable_address = executable_address,
        executable_byte_length = executable_byte_length,
        host = host,
        gpu_address = gpu_address,
        page = page,
        completion_size = size_of::<Completion>(),
        completion_value_host = host + completion_value_offset,
        completion_value_device = gpu_address + completion_value_offset as u64,
        indirect_buffer_host = host + 256,
        indirect_buffer_address = indirect_buffer_address,
        command_count = command_count,
        ring_host = transport.ring_host_address,
        ring_device = transport.ring_device_address,
        ring_size = transport.ring_size_bytes,
        read_index_host = transport.read_index_host_address,
        read_index_device = transport.read_index_device_address,
        write_index_host = transport.write_index_host_address,
        write_index_device = transport.write_index_device_address,
        doorbell_host = transport.doorbell_host_address,
    )?;
    match transport.doorbell_device_address {
        Some(address) => writeln!(output, "{address:#x}"),
        None => writeln!(output, "None"),
    }
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
    /// Samples only initialized atomic fields before native backing cleanup.
    #[cold]
    #[allow(unsafe_code)]
    fn diagnose_code_cache_failure(
        &self,
        boundary: &str,
        completed: bool,
        executable_range: (u64, u64),
        command_page: (usize, u64, u64),
        indirect_buffer: (u64, usize),
        transport: QueueTransport,
    ) {
        let Ok(_operation) = self.driver.admit() else {
            return;
        };
        if std::env::var_os("ROCDDI_CODE_CACHE_DIAGNOSTICS").is_none_or(|value| value != "1") {
            return;
        }
        let (host, _, _) = command_page;
        // SAFETY: The private AQL queue retains aligned, nonwrapping 64-bit
        // indices and the initialized completion page through inactivation.
        // Direct acquire loads avoid progress()'s exception polling and resets.
        let (progress, completion) = unsafe {
            (
                (
                    (&*(transport.read_index_host_address as *const AtomicU64))
                        .load(Ordering::Acquire),
                    (&*(transport.write_index_host_address as *const AtomicU64))
                        .load(Ordering::Acquire),
                ),
                (&*(host as *const Completion))
                    .value
                    .load(Ordering::Acquire),
            )
        };
        let mut output = io::stderr().lock();
        let _ =
            write_failure_state_diagnostic(&mut output, boundary, completed, progress, completion);
        let _ = write_failure_address_diagnostic(
            &mut output,
            executable_range,
            command_page,
            indirect_buffer,
            transport,
        );
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
        let storage = self
            .device()
            .allocate(
                MemoryKind::System,
                page,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE,
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
        let (command, count) = invalidate_packet(info.gfx_major, info.gfx_minor);
        // SAFETY: The page contains disjoint, aligned completion and IB records.
        unsafe {
            (host as *mut Completion).write(Completion {
                kind: 1,
                value: AtomicI64::new(1),
                reserved: [0; 6],
            });
            std::ptr::copy_nonoverlapping(command.as_ptr(), (host + 256) as *mut u32, count);
        }
        // SAFETY: This method is the sole producer. The dependency bundle owns
        // every address the packet can read, even if queue creation is partial.
        let mut queue = unsafe {
            self.create_code_cache_queue_with_dependencies(
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
        let ib = gpu_address + 256;
        let mut packet = [0_u32; 16];
        // VENDOR_SPECIFIC packet, AMD_AQL_FORMAT_PM4_IB, system fences.
        packet[0] = (1 << 16) | (2 << 9) | (2 << 11);
        packet[1] = 0xc002_3f00; // INDIRECT_BUFFER, four dwords.
        packet[2] = u32::try_from(ib & u64::from(u32::MAX))
            .map_err(|_| invalid(ErrorKind::Internal, "IB low word overflow"))?;
        packet[3] = u32::try_from(ib >> 32)
            .map_err(|_| invalid(ErrorKind::Internal, "IB high word overflow"))?;
        packet[4] = (1 << 23)
            | u32::try_from(count)
                .map_err(|_| invalid(ErrorKind::Internal, "IB count overflow"))?;
        packet[5] = 0xa;
        packet[14] = u32::try_from(gpu_address & u64::from(u32::MAX))
            .map_err(|_| invalid(ErrorKind::Internal, "signal low word overflow"))?;
        packet[15] = u32::try_from(gpu_address >> 32)
            .map_err(|_| invalid(ErrorKind::Internal, "signal high word overflow"))?;
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
        let completed = loop {
            // SAFETY: Queue dependencies own the initialized signal page.
            if unsafe { &*(host as *const Completion) }
                .value
                .load(Ordering::Acquire)
                == 0
            {
                break true;
            }
            if cancel.load(Ordering::Acquire) || Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_micros(50));
        };
        if !completed {
            queue.diagnose_incomplete_cache_wait();
            self.diagnose_code_cache_failure(
                "wait-incomplete",
                completed,
                (address, byte_length),
                (host, gpu_address, page),
                (ib, count),
                transport,
            );
        }
        queue.inactivate().map_err(|error| {
            self.diagnose_code_cache_failure(
                "inactivation-error",
                completed,
                (address, byte_length),
                (host, gpu_address, page),
                (ib, count),
                transport,
            );
            CopyFailure {
                error,
                operands_may_be_live: true,
            }
        })?;
        // SAFETY: The sole producer has stopped and CP is inactivated. The
        // retained owner releases its IB and signal only on native success.
        unsafe { queue.destroy() }.map_err(|error| CopyFailure {
            error,
            operands_may_be_live: true,
        })?;
        if completed {
            Ok(())
        } else {
            Err(invalid(
                ErrorKind::Busy,
                "code-cache publication did not complete",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QueueTransport, invalidate_packet, write_failure_address_diagnostic,
        write_failure_state_diagnostic,
    };
    use crate::device::gpu::queue::{HostQueuePublication, QueueAccessWidth};
    use std::io;

    #[test]
    fn failure_state_preserves_wait_result_and_independent_sample() -> io::Result<()> {
        let mut output = Vec::new();
        write_failure_state_diagnostic(&mut output, "wait-incomplete", false, (0, 1), 1)?;
        assert_eq!(
            output,
            b"rocddi code-cache: boundary=wait-incomplete; wait_completed=false; read=0; write=1; completion=1\n"
        );
        output.clear();
        // Completion can change after the wait has already ended.
        write_failure_state_diagnostic(&mut output, "inactivation-error", false, (1, 1), 0)?;
        assert_eq!(
            output,
            b"rocddi code-cache: boundary=inactivation-error; wait_completed=false; read=1; write=1; completion=0\n"
        );
        let mut full = &mut [0_u8; 0][..];
        assert!(
            write_failure_state_diagnostic(&mut full, "inactivation-error", true, (1, 1), 0)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn failure_addresses_preserve_retained_cpu_and_device_transport_facts() -> io::Result<()> {
        let mut transport = QueueTransport {
            ring_host_address: 0x1000,
            ring_device_address: 0x2000,
            ring_size_bytes: 4096,
            aql_control: None,
            sdma_engine_id: None,
            read_index_host_address: 0x3000,
            read_index_device_address: 0x4000,
            write_index_host_address: 0x5000,
            write_index_device_address: 0x6000,
            read_index_width: QueueAccessWidth::Bits64,
            write_index_width: QueueAccessWidth::Bits64,
            index_unit_bytes: 64,
            read_index_wraps: false,
            doorbell_host_address: 0x7000,
            doorbell_device_address: None,
            doorbell_width: QueueAccessWidth::Bits64,
            host_publication: HostQueuePublication::Ordered64BitMmio,
        };
        let mut output = Vec::new();
        write_failure_address_diagnostic(
            &mut output,
            (0x9000_0000, 0x1234),
            (0x1000_0000, 0x2000_0000, 8192),
            (0x2000_0100, 7),
            transport,
        )?;
        assert_eq!(
            output,
            concat!(
                "rocddi code-cache: addresses; ",
                "logical_executable_gpu=0x90000000; logical_executable_size_bytes=4660; ",
                "native_executable_allocation=unavailable(not_retained); ",
                "command_page_cpu=0x10000000; command_page_gpu=0x20000000; command_page_size_bytes=8192; ",
                "completion_cpu=0x10000000; completion_gpu=0x20000000; completion_size_bytes=64; ",
                "completion_value_cpu=0x10000008; completion_value_gpu=0x20000008; ",
                "ib_cpu=0x10000100; ib_gpu=0x20000100; ib_words=7; ",
                "ring_cpu=0x1000; ring_gpu=0x2000; ring_size_bytes=4096; ",
                "read_index_cpu=0x3000; read_index_gpu=0x4000; ",
                "write_index_cpu=0x5000; write_index_gpu=0x6000; ",
                "doorbell_cpu=0x7000; doorbell_gpu=None\n"
            )
            .as_bytes()
        );
        transport.doorbell_device_address = Some(0x8000);
        output.clear();
        write_failure_address_diagnostic(
            &mut output,
            (0x9000_0000, 0x1234),
            (0x1000_0000, 0x2000_0000, 4096),
            (0x2000_0100, 8),
            transport,
        )?;
        assert!(output.ends_with(b"doorbell_gpu=0x8000\n"));
        assert!(!output[..output.len() - 1].contains(&b'\n'));
        let mut full = &mut [0_u8; 0][..];
        assert!(
            write_failure_address_diagnostic(
                &mut full,
                (0x9000_0000, 0x1234),
                (0x1000_0000, 0x2000_0000, 4096),
                (0x2000_0100, 7),
                transport,
            )
            .is_err()
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
