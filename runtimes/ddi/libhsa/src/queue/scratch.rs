// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA scratch thresholds and bounded occupancy reduction under memory pressure.

use super::*;

/// Environment policy sampled before a generation activates its devices.
pub(crate) struct ScratchOptions {
    /// Whether completed scratch may be returned to the allocator.
    reclaim: bool,
    /// Whether hardware-supported asynchronous reclamation is requested.
    asynchronous: bool,
    /// Whether allocation failure may reduce concurrent wave occupancy.
    reduce_occupancy: bool,
    /// Optional per-queue threshold for the ordinary use-once protocol.
    single_limit: Option<u64>,
    /// Optional per-queue threshold for asynchronous reclaim.
    async_limit: Option<u64>,
}

impl ScratchOptions {
    /// Samples documented scratch controls once for an HSA generation.
    pub(crate) fn from_env() -> Self {
        let number = |name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
        };
        Self {
            reclaim: std::env::var("HSA_NO_SCRATCH_RECLAIM").as_deref() != Ok("1"),
            asynchronous: std::env::var("HSA_ENABLE_SCRATCH_ASYNC_RECLAIM").as_deref() != Ok("0"),
            reduce_occupancy: std::env::var("HSA_NO_SCRATCH_THREAD_LIMITER").as_deref() != Ok("1"),
            single_limit: number("HSA_SCRATCH_SINGLE_LIMIT"),
            async_limit: number("HSA_SCRATCH_SINGLE_LIMIT_ASYNC"),
        }
    }

    /// Resolves target limits while keeping native qualification separate.
    pub(crate) fn for_gpu(&self, gpu: &GpuInfo) -> ScratchPolicy {
        let supports_async = match (gpu.gfx_major, gpu.gfx_minor) {
            (9, 4) => gpu.packet_processor_firmware_version >= 177,
            (9, 5) => gpu.packet_processor_firmware_version >= 24,
            (12, 5) => true,
            _ => false,
        };
        ScratchPolicy {
            reclaim: self.reclaim,
            asynchronous: self.reclaim && self.asynchronous && supports_async,
            reduce_occupancy: self.reclaim && self.reduce_occupancy,
            single_limit: self
                .single_limit
                .unwrap_or(140 << 20)
                .min(gpu.maximum_scratch_aperture_bytes()),
            async_limit: AtomicU64::new(
                self.async_limit
                    .unwrap_or((3_u64 << 30).saturating_mul(u64::from(gpu.xcc_count)))
                    .min(gpu.maximum_scratch_aperture_bytes()),
            ),
        }
    }
}

/// Target-resolved policy shared by queue workers and the threshold API.
pub(crate) struct ScratchPolicy {
    /// Whether a completed allocation may be reclaimed.
    pub(crate) reclaim: bool,
    /// Whether the CP protocol supports concurrent reclaim.
    pub(crate) asynchronous: bool,
    /// Whether allocation pressure may reduce wave occupancy.
    pub(crate) reduce_occupancy: bool,
    /// Ordinary per-queue use-once threshold.
    pub(crate) single_limit: u64,
    /// Mutable asynchronous threshold in bytes.
    pub(crate) async_limit: AtomicU64,
}

/// Publishes the firmware's use-once bit on an unpublished or stopped queue.
pub(super) fn use_once(queue: &RetainedQueue<QueueBacking>, enabled: bool) {
    if let Some(control) = queue.info().aql_control {
        // SAFETY: The retained native control is aligned and writable. Caller
        // orders this update before releasing the inactive queue signal.
        let properties = unsafe {
            &*(AqlControlRegion::properties_address_at(control.host_base())
                as *const std::sync::atomic::AtomicU32)
        };
        if enabled {
            properties.fetch_or(1 << 4, Ordering::Release);
        } else {
            properties.fetch_and(!(1 << 4), Ordering::Release);
        }
    }
}

/// Allocates with decreasing occupancy, never below the representable wave geometry.
pub(super) fn allocate_with_pressure(
    device: &Device,
    gpu: &GpuInfo,
    private_bytes: u32,
    lanes: u32,
    waves_per_group: u32,
    reduce: bool,
) -> Result<(Allocation, QueueScratch, bool), Status> {
    let mut plan = plan_scratch(*gpu, private_bytes, lanes).map_err(map_error)?;
    let initial_waves = plan.maximum_wave_count;
    let engines = gpu
        .shader_engine_count_per_xcc
        .saturating_mul(gpu.xcc_count);
    loop {
        match device
            .gpu()
            .and_then(|device| device.allocate_queue_scratch(plan.byte_length))
        {
            Ok(allocation) => {
                let scratch = QueueScratch {
                    device_address: allocation.info().device_address,
                    byte_length: plan.byte_length,
                    maximum_private_segment_byte_length: plan.maximum_private_segment_byte_length,
                    lanes_per_wave: lanes,
                    maximum_wave_count: plan.maximum_wave_count,
                };
                return Ok((
                    allocation,
                    scratch,
                    plan.maximum_wave_count != initial_waves,
                ));
            }
            Err(error) if !reduce || error.kind() != rocddi::ErrorKind::ResourceExhausted => {
                return Err(map_error(error));
            }
            Err(_) => {}
        }
        let Some(next) = lower_wave_count(
            plan.maximum_wave_count,
            gpu.compute_unit_count,
            engines,
            gpu.shader_engine_count_per_xcc,
            waves_per_group,
        ) else {
            return Err(OUT_OF_RESOURCES);
        };
        plan.maximum_wave_count = next;
        let bytes = u64::from(next)
            * u64::from(lanes)
            * u64::from(plan.maximum_private_segment_byte_length);
        let alignment = GPU_PAGE_BYTES * u64::from(gpu.xcc_count);
        plan.byte_length = bytes.div_ceil(alignment) * alignment;
    }
}

/// Reduces per-CU wave residency using the dispatch's workgroup width.
fn lower_wave_count(
    current: u32,
    cus: u32,
    engines: u32,
    engines_per_xcc: u32,
    group: u32,
) -> Option<u32> {
    if cus == 0 || engines == 0 || engines_per_xcc == 0 || group == 0 {
        return None;
    }
    let per_cu = current / cus;
    let step = if per_cu <= group && engines_per_xcc < group && group % engines_per_xcc == 0 {
        engines_per_xcc
    } else {
        group
    };
    let next = per_cu.checked_sub(1)? / step * step;
    let waves = next.checked_mul(cus)? / engines * engines;
    (waves > 0 && waves < current).then_some(waves)
}

#[cfg(test)]
mod tests {
    use super::lower_wave_count;

    #[test]
    fn pressure_reduction_preserves_engine_multiples_and_terminates() {
        let mut waves = 2048;
        while let Some(next) = lower_wave_count(waves, 64, 4, 4, 8) {
            assert!(next < waves);
            assert_eq!(next % 4, 0);
            waves = next;
        }
        assert_eq!(waves, 256);
        assert_eq!(lower_wave_count(256, 64, 4, 4, 8), None);
        assert_eq!(lower_wave_count(128, 0, 2, 2, 4), None);
    }
    #[test]
    fn multi_xcc_pressure_uses_per_xcc_steps_below_workgroup_width() {
        assert_eq!(lower_wave_count(8 * 256, 256, 32, 4, 8), Some(4 * 256));
        assert_eq!(lower_wave_count(12 * 256, 256, 32, 4, 16), Some(8 * 256));
        assert_eq!(lower_wave_count(8 * 256, 256, 32, 4, 16), Some(4 * 256));
        assert_eq!(lower_wave_count(4 * 256, 256, 32, 4, 16), None);
    }
}
