// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GFX12 fixed queue scratch geometry shared by GPU frontends.

use crate::topology::GpuInfo;
use crate::{Error, ErrorKind};

/// Largest private segment accepted by the qualified GFX12 queue path.
pub const MAX_PRIVATE_SEGMENT_BYTES: u32 = 262_128;
const SCRATCH_ALIGNMENT: u64 = 256;
const GPU_PAGE_BYTES: u64 = 4096;

/// Validated allocation extent and wave limits for one queue request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScratchPlan {
    /// Total scratch allocation extent in bytes.
    pub byte_length: u64,
    /// Aligned private bytes per lane accepted by the queue.
    pub maximum_private_segment_byte_length: u32,
    /// Execution width whose private segment size this plan reserves.
    pub lanes_per_wave: u32,
    /// Maximum waves covered by the scratch allocation.
    pub maximum_wave_count: u32,
}

fn invalid_scratch() -> Error {
    Error::Operation {
        kind: ErrorKind::ResourceExhausted,
        detail: "queue scratch request exceeds qualified GFX12 geometry",
    }
}

/// Plans a fixed scratch allocation for a qualified GFX12 AQL queue.
///
/// `lanes_per_wave` is the execution width requested by the packet. A wave64
/// packet on a wave32 GPU consumes twice the per-wave storage.
///
/// # Errors
/// Returns resource exhaustion when the target or request cannot be encoded
/// by the current queue scratch path.
pub fn plan_scratch(
    gpu: GpuInfo,
    private_segment_size: u32,
    lanes_per_wave: u32,
) -> Result<ScratchPlan, Error> {
    if private_segment_size == 0 || private_segment_size > MAX_PRIVATE_SEGMENT_BYTES {
        return Err(invalid_scratch());
    }
    if !matches!(
        (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping),
        (12, 0, 0 | 1) | (12, 5, 0)
    ) || gpu.wavefront_size != 32
        || gpu.xcc_count == 0
        || !matches!(lanes_per_wave, 32 | 64)
    {
        return Err(invalid_scratch());
    }
    let lane_alignment = u32::try_from(SCRATCH_ALIGNMENT)
        .ok()
        .and_then(|alignment| alignment.checked_div(lanes_per_wave))
        .filter(|alignment| *alignment != 0)
        .ok_or_else(invalid_scratch)?;
    let aligned_private = private_segment_size
        .checked_add(lane_alignment - 1)
        .map(|size| size / lane_alignment * lane_alignment)
        .ok_or_else(invalid_scratch)?;
    let engines = gpu
        .shader_engine_count_per_xcc
        .checked_mul(gpu.xcc_count)
        .filter(|engines| *engines != 0)
        .ok_or_else(invalid_scratch)?;
    let maximum_waves = gpu
        .compute_unit_count
        .checked_mul(gpu.maximum_scratch_wave_count_per_compute_unit)
        .ok_or_else(invalid_scratch)?;
    let wave_bytes = u64::from(aligned_private)
        .checked_mul(u64::from(lanes_per_wave))
        .ok_or_else(invalid_scratch)?;
    let representable_bytes = u64::from(u32::MAX) / GPU_PAGE_BYTES * GPU_PAGE_BYTES;
    let representable_waves = u32::try_from(representable_bytes / wave_bytes)
        .unwrap_or(u32::MAX)
        .min(maximum_waves);
    let maximum_wave_count = representable_waves / engines * engines;
    if maximum_wave_count == 0 {
        return Err(invalid_scratch());
    }
    let required = wave_bytes
        .checked_mul(u64::from(maximum_wave_count))
        .ok_or_else(invalid_scratch)?;
    let byte_length = required
        .checked_add(GPU_PAGE_BYTES - 1)
        .map(|size| size / GPU_PAGE_BYTES * GPU_PAGE_BYTES)
        .filter(|size| u32::try_from(*size).is_ok())
        .ok_or_else(invalid_scratch)?;
    Ok(ScratchPlan {
        byte_length,
        maximum_private_segment_byte_length: aligned_private,
        lanes_per_wave,
        maximum_wave_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gfx1201() -> GpuInfo {
        GpuInfo {
            gfx_major: 12,
            gfx_minor: 0,
            gfx_stepping: 1,
            wavefront_size: 32,
            compute_unit_count: 64,
            maximum_wave_count_per_compute_unit: 32,
            maximum_scratch_wave_count_per_compute_unit: 32,
            xcc_count: 1,
            shader_engine_count_per_xcc: 2,
            ..GpuInfo::default()
        }
    }

    #[test]
    fn gfx1201_dynamic_scratch_covers_every_native_wave_slot() {
        assert_eq!(
            plan_scratch(gfx1201(), 1024, 32).ok(),
            Some(ScratchPlan {
                byte_length: 64 * 1024 * 1024,
                maximum_private_segment_byte_length: 1024,
                lanes_per_wave: 32,
                maximum_wave_count: 2048,
            })
        );
    }

    #[test]
    fn scratch_plan_aligns_per_lane_size_and_rejects_unqualified_targets() {
        assert_eq!(
            plan_scratch(gfx1201(), 1025, 32).ok(),
            Some(ScratchPlan {
                byte_length: 1032 * 32 * 2048,
                maximum_private_segment_byte_length: 1032,
                lanes_per_wave: 32,
                maximum_wave_count: 2048,
            })
        );

        let mut unsupported = gfx1201();
        unsupported.gfx_stepping = 2;
        assert!(plan_scratch(unsupported, 1024, 32).is_err());
        assert!(plan_scratch(gfx1201(), 0, 32).is_err());
        assert!(plan_scratch(gfx1201(), MAX_PRIVATE_SEGMENT_BYTES + 1, 32).is_err());
        assert!(plan_scratch(gfx1201(), 1024, 16).is_err());
    }

    #[test]
    fn gfx1201_wave64_scratch_covers_twice_the_wave32_storage() {
        assert_eq!(
            plan_scratch(gfx1201(), 1024, 64).ok(),
            Some(ScratchPlan {
                byte_length: 128 * 1024 * 1024,
                maximum_private_segment_byte_length: 1024,
                lanes_per_wave: 64,
                maximum_wave_count: 2048,
            })
        );
    }
}
