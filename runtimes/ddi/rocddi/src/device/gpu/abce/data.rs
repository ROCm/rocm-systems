// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Immutable policy callback tables and HDP packet data.

// The C-facing projection keeps C integer conversion and alignment contracts.
// Safety requirements and callback preconditions are specified in the public
// header named by this module; bodies intentionally follow those algorithms.
#![allow(
    clippy::missing_safety_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_ptr_alignment,
    clippy::too_many_lines
)]

#[cfg(all(target_os = "linux", not(test)))]
core::arch::global_asm!(include_str!("data_exports_linux.S"));

// These entry points project caller-owned C objects through raw pointers.
// Initialization/reconfiguration requires exclusive caller access; submission
// accesses individual fields and shared atomic cursors, never whole-object
// exclusive references. Pointer and buffer preconditions are specified by the
// corresponding public C header. The packet/ring tests run both providers.

use super::abi::{SDMA_PKT_HDP_FLUSH, SDMA_PKT_HDP_FLUSH_TAG, abce_sdma_engine_policy_vtable_t};
use super::{
    abce_gfx90a_rank_d2h, abce_gfx90a_rank_h2d, abce_gfx90a_rank_p2p, abce_gfx94x_rank_d2h,
    abce_gfx94x_rank_h2d, abce_gfx94x_rank_p2p, abce_sdma_policy_base_rank_all,
    abce_sdma_policy_base_rank_p2p,
};

// Public C addresses remain live even when LTO folds internal reads.
#[used]
#[unsafe(no_mangle)]
pub static abce_sdma_hdp_flush_cmd: SDMA_PKT_HDP_FLUSH = SDMA_PKT_HDP_FLUSH_TAG {
    DW_0_DATA: 8_u32,
    DW_1_DATA: 0_u32,
    DW_2_DATA: 2_147_483_648_u32,
    DW_3_DATA: 0_u32,
    DW_4_DATA: 0_u32,
    DW_5_DATA: 0_u32,
};
#[used]
#[unsafe(no_mangle)]
pub static abce_sdma_policy_base_vtable: abce_sdma_engine_policy_vtable_t =
    abce_sdma_engine_policy_vtable_t {
        rank_h2d: Some(abce_sdma_policy_base_rank_all),
        rank_d2h: Some(abce_sdma_policy_base_rank_all),
        rank_p2p: Some(abce_sdma_policy_base_rank_p2p),
        rank_local_d2d: Some(abce_sdma_policy_base_rank_all),
    };
pub static abce_gfx94x_sdma_affinity_map: [[i32; 8]; 8] = [
    [0i32, 7i32, 6i32, 1i32, 2i32, 4i32, 5i32, 3i32],
    [7i32, 0i32, 1i32, 5i32, 4i32, 2i32, 3i32, 6i32],
    [5i32, 1i32, 0i32, 6i32, 7i32, 3i32, 2i32, 4i32],
    [1i32, 6i32, 5i32, 0i32, 3i32, 7i32, 4i32, 2i32],
    [2i32, 4i32, 7i32, 3i32, 0i32, 5i32, 6i32, 1i32],
    [4i32, 2i32, 3i32, 7i32, 6i32, 0i32, 1i32, 5i32],
    [5i32, 3i32, 2i32, 4i32, 6i32, 1i32, 0i32, 7i32],
    [3i32, 6i32, 4i32, 2i32, 1i32, 5i32, 7i32, 0i32],
];
#[used]
#[unsafe(no_mangle)]
pub static abce_sdma_policy_gfx94x_vtable: abce_sdma_engine_policy_vtable_t =
    abce_sdma_engine_policy_vtable_t {
        rank_h2d: Some(abce_gfx94x_rank_h2d),
        rank_d2h: Some(abce_gfx94x_rank_d2h),
        rank_p2p: Some(abce_gfx94x_rank_p2p),
        rank_local_d2d: Some(abce_sdma_policy_base_rank_all),
    };
#[used]
#[unsafe(no_mangle)]
pub static abce_sdma_policy_gfx90a_vtable: abce_sdma_engine_policy_vtable_t =
    abce_sdma_engine_policy_vtable_t {
        rank_h2d: Some(abce_gfx90a_rank_h2d),
        rank_d2h: Some(abce_gfx90a_rank_d2h),
        rank_p2p: Some(abce_gfx90a_rank_p2p),
        rank_local_d2d: Some(abce_sdma_policy_base_rank_all),
    };
