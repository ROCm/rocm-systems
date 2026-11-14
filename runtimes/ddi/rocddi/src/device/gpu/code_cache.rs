// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Instruction-cache publication using the AMD AQL PM4 indirect-buffer packet.

use super::queue::{
    QueueParameters, QueuePriority, QueueProducerMode, QueueRequest, QueueRingMemory,
    store_ordered_mmio64,
};
use super::{CopyFailure, GpuDevice};
use crate::memory::{DeviceAccess, MemoryKind, host_page_size};
use crate::{Error, ErrorKind};
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
        queue.inactivate().map_err(|error| CopyFailure {
            error,
            operands_may_be_live: true,
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
    use super::invalidate_packet;

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
