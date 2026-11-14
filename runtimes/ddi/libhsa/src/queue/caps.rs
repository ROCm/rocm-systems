// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! CP capability discovery through a completed, privately owned AQL packet.

use super::*;
use std::sync::atomic::AtomicU32;
use std::time::Instant;

/// Converts CP flags to the bit-indexed public AQL extension array.
fn extension_bits(caps: u32, native_extended_dispatch: bool) -> [u8; 8] {
    let mut bits = [0; 8];
    // The public enum values are indices 1 and 2, not masks 1 and 2.
    if native_extended_dispatch || caps & (1 << 3) != 0 {
        bits[0] |= 1 << 1;
    }
    if caps & (1 << 2) != 0 {
        bits[0] |= 1 << 2;
    }
    bits
}

/// Pins the entire generation if a probe unwinds with uncertain native users.
struct ProbeGuard<'a> {
    /// Native cleanup has not yet been established.
    armed: bool,
    /// Shared generation health retained by runtime teardown.
    health: &'a crate::runtime::GenerationHealth,
}

impl Drop for ProbeGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.health.quarantine();
        }
    }
}

/// Executes one empty barrier before reading CP-written queue capabilities.
fn probe(
    device: &Device,
    page_size: usize,
    stop: &AtomicBool,
    health: &crate::runtime::GenerationHealth,
) -> Result<u32, Status> {
    let allocation = device
        .allocate(
            MemoryKind::System,
            GPU_PAGE_BYTES.max(page_size as u64),
            page_size as u64,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )
        .map_err(map_error)?;
    let signal_info = allocation.info();
    let signal = signal_info.host_address.ok_or(OUT_OF_RESOURCES)?;
    if signal as u64 != signal_info.device_address {
        return Err(OUT_OF_RESOURCES);
    }
    // SAFETY: The page is writable and aligned for the complete signal. The
    // allocation is moved into the queue's retained dependencies below.
    unsafe { (signal as *mut AmdSignal).write(AmdSignal::user(1)) };
    let gpu = device.gpu().map_err(map_error)?;
    let mut guard = ProbeGuard {
        armed: true,
        health,
    };
    // SAFETY: This private single producer owns all packet publication. Its
    // completion signal and native device outlive checked queue destruction.
    let created = unsafe {
        gpu.create_queue_with_dependencies(
            QueueRequest {
                control_memory: rocddi::device::gpu::queue::QueueRingMemory::System,
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
            (allocation, device.clone()),
        )
    };
    let mut queue = match created {
        Ok(queue) => queue,
        Err(error) => {
            if !matches!(error, rocddi::Error::QueueBackingMayBeLive { .. }) {
                guard.armed = false;
            }
            return Err(map_error(error));
        }
    };
    let info = queue.info();
    let result = (|| {
        let control = info.aql_control.ok_or(ERROR)?;
        if stop.load(Ordering::Acquire) {
            return Err(INVALID_RUNTIME_STATE);
        }
        // SAFETY: The native transport validates aligned ring/index/control
        // mappings. No other producer can access this unpublished queue.
        unsafe {
            let packet = info.ring_host_address as *mut u8;
            packet.add(4).write_bytes(0, AQL_PACKET_BYTES - 4);
            packet.add(56).cast::<u64>().write(signal as u64);
            // BARRIER_AND, with system acquire and release fence scopes.
            (&*packet.cast::<AtomicU32>()).store(3 | (2 << 9) | (2 << 11), Ordering::Release);
            (&*(info.write_index_host_address as *const AtomicU64)).store(1, Ordering::Release);
            rocddi::device::gpu::queue::store_ordered_mmio64(info.doorbell_host_address, 0);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            // SAFETY: queue retains the signal page until native destruction.
            let complete = unsafe { &*(signal as *const AmdSignal) }
                .value
                .load(Ordering::Acquire)
                == 0;
            if complete {
                // Completion acquire orders the CP's earlier caps write.
                // SAFETY: The queue retains the full native AQL control page.
                return Ok(unsafe {
                    (AqlControlRegion::caps_address_at(control.host_base()) as *const u32)
                        .read_volatile()
                });
            }
            if stop.load(Ordering::Acquire) || Instant::now() >= deadline {
                return Err(OUT_OF_RESOURCES);
            }
            thread::sleep(Duration::from_micros(50));
        }
    })();
    // A timeout is not retirement and does not cache a negative capability.
    // Inactivation plus checked native destruction can still safely retire it.
    queue.inactivate().map_err(map_error)?;
    // SAFETY: The sole producer has stopped, and inactivation has stopped CP.
    unsafe { queue.destroy() }.map_err(map_error)?;
    guard.armed = false;
    result
}

/// Returns confirmed capabilities, sharing a successful probe per generation.
pub(crate) fn aql_extensions(agent: HsaAgent) -> Result<[u8; 8], Status> {
    let (device, cache, stop, health, page_size, native_extended, _token) = {
        let guard = lock()?;
        let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
        if runtime.cpu_index(agent).is_some() {
            return Ok([0; 8]);
        }
        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
        let gpu = &runtime.gpus[index];
        let info = gpu.info();
        (
            gpu.device.clone(),
            gpu.cp_caps.clone(),
            runtime.operation_stop(),
            runtime.health.clone(),
            runtime.host_page_size,
            info.gfx_major == 12 && info.gfx_minor >= 5,
            runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?,
        )
    };
    let mut cache = cache.lock().map_err(|_| ERROR)?;
    let caps = if let Some(caps) = *cache {
        caps
    } else {
        let caps = probe(&device, page_size, &stop, &health)?;
        *cache = Some(caps);
        caps
    };
    Ok(extension_bits(caps, native_extended))
}

#[cfg(test)]
mod tests {
    use super::extension_bits;

    #[test]
    fn public_properties_use_indices_and_native_dispatch_does_not_imply_hints() {
        assert_eq!(extension_bits(0, false), [0; 8]);
        assert_eq!(extension_bits(1 << 2, false), [4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extension_bits(1 << 3, false), [2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extension_bits(0, true), [2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(extension_bits(12, true), [6, 0, 0, 0, 0, 0, 0, 0]);
    }
}
