// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Queue masks constrained by the generation's process-wide CU policy.

use super::*;

/// Intersects a requested CU mask with process and hardware restrictions.
pub(crate) fn constrain_cu_mask(
    gpu: &GpuInfo,
    global: Option<&[u32]>,
    requested: Option<&[u32]>,
) -> Result<(Vec<u32>, bool), Status> {
    let mut selected =
        requested.map_or_else(|| full_cu_mask(gpu.compute_unit_count), <[u32]>::to_vec);
    let mut clipped = false;
    if let Some(global) = global {
        for (index, word) in selected.iter_mut().enumerate() {
            let allowed = global.get(index).copied().unwrap_or(0);
            clipped |= *word & !allowed != 0;
            *word &= allowed;
        }
    }
    let physical = full_cu_mask(gpu.compute_unit_count);
    selected.resize(physical.len(), 0);
    for (word, physical) in selected.iter_mut().zip(physical) {
        *word &= physical;
    }
    if gpu.gfx_major >= 10
        && !(gpu.gfx_major == 12 && gpu.gfx_minor >= 5)
        && selected
            .iter()
            .any(|word| (word & 0x5555_5555) != ((word >> 1) & 0x5555_5555))
    {
        return Err(INVALID_ARGUMENT);
    }
    Ok((selected, clipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_reset_cannot_enable_cus_disabled_for_the_process() {
        let gpu = GpuInfo {
            gfx_major: 11,
            compute_unit_count: 8,
            ..GpuInfo::default()
        };
        assert_eq!(
            constrain_cu_mask(&gpu, Some(&[0x33]), None),
            Ok((vec![0x33], true))
        );
        assert_eq!(
            constrain_cu_mask(&gpu, Some(&[0x33]), Some(&[0x0f])),
            Ok((vec![3], true))
        );
        assert_eq!(
            constrain_cu_mask(&gpu, None, Some(&[1])),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(
            constrain_cu_mask(&gpu, Some(&[3]), Some(&[12])),
            Ok((vec![0], true))
        );
        assert_eq!(
            constrain_cu_mask(&gpu, None, Some(&[0])),
            Ok((vec![0], false))
        );
    }
}
