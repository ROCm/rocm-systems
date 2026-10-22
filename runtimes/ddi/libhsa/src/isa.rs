// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA ISA names and handle encodings shared by runtime validation and ABI queries.

use crate::ffi::{HsaIsa, HsaWavefront, ISA_BASE, ISA_COUNT_PER_GPU, WAVEFRONT_BASE};

pub(crate) fn gfx_target_name(major: u32, minor: u32, stepping: u32) -> String {
    format!("gfx{major}{minor}{stepping:x}")
}

pub(crate) fn isa_handle(gpu_index: usize, variant: u64) -> HsaIsa {
    HsaIsa {
        handle: ISA_BASE + gpu_index as u64 * ISA_COUNT_PER_GPU + variant,
    }
}

pub(crate) fn isa_name(
    gpu: &rocddi::topology::GpuInfo,
    xnack_enabled: bool,
    variant: u64,
) -> Option<String> {
    let base = match variant {
        0 if (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping) == (12, 5, 0)
            && gpu.asic_revision == 0 =>
        {
            "gfx1250-strict".to_owned()
        }
        0 => gfx_target_name(gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping),
        1 if gpu.gfx_major == 9 && matches!(gpu.gfx_minor, 4 | 5) => "gfx9-4-generic".to_owned(),
        1 if gpu.gfx_major == 9 => return None,
        1 if gpu.gfx_major == 12 && gpu.gfx_minor == 5 => "gfx12-5-generic".to_owned(),
        1 => format!("gfx{}-generic", gpu.gfx_major),
        _ => return None,
    };
    let features = if gpu.gfx_major == 9 {
        format!(
            ":sramecc{}:xnack{}",
            if gpu.sram_ecc_enabled { '+' } else { '-' },
            if xnack_enabled { '+' } else { '-' }
        )
    } else {
        String::new()
    };
    Some(format!("amdgcn-amd-amdhsa--{base}{features}"))
}

pub(crate) fn wavefront_handle(isa: HsaIsa) -> HsaWavefront {
    HsaWavefront {
        handle: WAVEFRONT_BASE + isa.handle.saturating_sub(ISA_BASE),
    }
}
