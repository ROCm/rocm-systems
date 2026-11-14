// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE planner mechanisms.

// These entry points project caller-owned C objects through raw pointers.
// Initialization/reconfiguration requires exclusive caller access; submission
// accesses individual fields and shared atomic cursors, never whole-object
// exclusive references. Pointer and buffer preconditions are specified by the
// corresponding public C header. The packet/ring tests run both providers.

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

use super::abi::{
    ABCE_ENDPOINT_KIND_HOST, ABCE_EXECUTION_ENGINE_KIND_SDMA,
    ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK, ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT,
    ABCE_MULTICAST_MODE_AUTOMATIC, ABCE_MULTICAST_MODE_FORCE_MULTICAST, ABCE_OP_KIND_BROADCAST,
    ABCE_OP_KIND_COPY_RECT, ABCE_OP_KIND_FILL, ABCE_OP_KIND_INDIRECT, ABCE_OP_KIND_LINEAR,
    ABCE_OP_KIND_MULTICAST, ABCE_OP_KIND_SWAP, ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE,
    ABCE_POLICY_DISPOSITION_NO_PREFERENCE, ABCE_POLICY_DISPOSITION_RANKED,
    ABCE_STATUS_INVALID_ARGUMENT, ABCE_STATUS_INVALID_PLAN,
    ABCE_STATUS_MISSING_COORDINATION_SCRATCH, ABCE_STATUS_NO_ENGINE, ABCE_STATUS_NO_LEGAL_ENGINE,
    ABCE_STATUS_OK, ABCE_STATUS_OUT_OF_RANGE, ABCE_STATUS_RESOURCE_EXHAUSTED,
    ABCE_STATUS_RING_UNAVAILABLE, ABCE_STATUS_TOO_MANY_OPERATIONS, ABCE_STATUS_UNIMPLEMENTED,
    ABCE_TRANSFER_KIND_UNKNOWN, abce_builder_t, abce_copy_metadata_t, abce_copy_op_t,
    abce_copy_orchestrator_t, abce_engine_affinity_t, abce_engine_op_t, abce_engine_policy_fn_t,
    abce_frame_job_t, abce_isa_version_t, abce_op_kind_t, abce_plan_frame_t, abce_plan_t,
    abce_platform_caps_t, abce_policy_disposition_t, abce_policy_result_t, abce_ring_t,
    abce_sdma_engine_policy_t, abce_status_t, abce_submit_pending_t, abce_transfer_kind_t,
};
use super::{
    abce_acquire_packet_scratch, abce_builder_is_gfx125plus, abce_builder_num_copy_packets,
    abce_builder_num_fill_packets, abce_builder_num_rect_packets, abce_builder_num_swap_packets,
    abce_builder_num_wait_signal_swap_packets, abce_builder_requires_gcr,
    abce_builder_swap_alignment, abce_builder_swap_supported, abce_classify_transfer,
    abce_count_trailing_zeros_u64, abce_detect_default_platform_caps,
    abce_engine_affinity_initialize, abce_execution_descriptor_encode, abce_first_engine,
    abce_frame_composer_emit_frame, abce_frame_composer_frame_bytes,
    abce_frame_composer_initialize, abce_frame_composer_signal_count, abce_host_load_relaxed_u64,
    abce_host_store_relaxed_u64, abce_host_store_release_u32, abce_host_store_release_u64,
    abce_popcount_u64, abce_ring_acquire, abce_ring_cancel, abce_ring_fill,
    abce_ring_host_store_fence, abce_ring_padded_size, abce_ring_read_back,
    abce_ring_region_last_byte, abce_ring_release_ordered, abce_sdma_policy_initialize_for_isa,
    abce_sdma_policy_map_engine, abce_sdma_policy_registered_engine,
    abce_sdma_policy_select_heatmap_profile, abce_sdma_policy_set_engine_split,
    abce_sdma_policy_trampoline, abce_status_is_ok, abce_timestamp_surface_enabled,
    abce_transfer_kind_bit,
};

pub unsafe extern "C" fn abce_plan_initialize(out_plan: *mut abce_plan_t) {
    unsafe {
        let _ = libc::memset(
            (out_plan).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_plan_t>() as usize,
        );
        (*(out_plan)).epilogue_required = true;
        (*(out_plan)).failed_op = 4_294_967_295_u32;
    }
}

#[must_use]
pub unsafe extern "C" fn abce_plan_mixed_transfer_kinds(plan: *const abce_plan_t) -> bool {
    unsafe {
        ((*(plan)).transfer_kinds & (((*(plan)).transfer_kinds).wrapping_sub(1_u32))) != (0_u32)
    }
}

#[must_use]
pub unsafe extern "C" fn abce_plan_uniform_transfer_kind(
    plan: *const abce_plan_t,
) -> abce_transfer_kind_t {
    unsafe {
        if (((*(plan)).transfer_kinds) == (0_u32)) || abce_plan_mixed_transfer_kinds(plan) {
            return ABCE_TRANSFER_KIND_UNKNOWN;
        }
        (abce_count_trailing_zeros_u64(u64::from((*(plan)).transfer_kinds))) as abce_transfer_kind_t
    }
}

pub(super) unsafe extern "C" fn abce_env_size_or(
    name: *const core::ffi::c_char,
    fallback: usize,
) -> usize {
    unsafe {
        let raw: *const core::ffi::c_char = (libc::getenv(name)).cast_const();
        if (raw).is_null() || (i32::from(*(raw)) == (0i32)) {
            return fallback;
        }
        let mut end: *mut core::ffi::c_char = core::ptr::null_mut();
        let parsed: u64 = libc::strtoull(raw, core::ptr::addr_of_mut!(end), 0i32);
        if (end) == (raw).cast_mut() {
            return fallback;
        }
        (parsed) as usize
    }
}

pub(super) unsafe extern "C" fn abce_cached_env_size(
    cache: *mut u64,
    name: *const core::ffi::c_char,
    fallback: usize,
) -> usize {
    unsafe {
        let mut value: u64 = abce_host_load_relaxed_u64((cache).cast_const());
        if (value) == (u64::MAX) {
            value = (abce_env_size_or(name, fallback)) as u64;
            abce_host_store_relaxed_u64(cache, value);
        }
        (value) as usize
    }
}

pub(super) unsafe extern "C" fn abce_linear_b2b_max_total() -> usize {
    unsafe {
        static mut cache: u64 = u64::MAX;
        abce_cached_env_size(
            core::ptr::addr_of_mut!(cache),
            c"ABCE_LINEAR_B2B_MAX_TOTAL"
                .as_ptr()
                .cast_mut()
                .cast_const(),
            ((512u32).wrapping_mul(1024u32)) as usize,
        )
    }
}

pub(super) unsafe extern "C" fn abce_broadcast_max_size() -> usize {
    unsafe {
        static mut cache: u64 = u64::MAX;
        abce_cached_env_size(
            core::ptr::addr_of_mut!(cache),
            c"ABCE_BROADCAST_MAX".as_ptr(),
            ((16u32).wrapping_mul(1024u32)) as usize,
        )
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_initialize_with_caps(
    builder: *const abce_builder_t,
    caps: abce_platform_caps_t,
    out_orchestrator: *mut abce_copy_orchestrator_t,
) {
    unsafe {
        let _ = libc::memset(
            (out_orchestrator).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_copy_orchestrator_t>() as usize,
        );
        (*(out_orchestrator)).builder = builder;
        (*(out_orchestrator)).is_gfx125plus = abce_builder_is_gfx125plus(builder);
        (*(out_orchestrator)).caps = caps;
        abce_frame_composer_initialize(
            builder,
            caps,
            core::ptr::addr_of_mut!((*(out_orchestrator)).composer),
        );
        (*(out_orchestrator)).has_policy = abce_sdma_policy_initialize_for_isa(
            (*(builder)).isa,
            core::ptr::addr_of_mut!((*(out_orchestrator)).owned_policy),
        );
        if (*(out_orchestrator)).has_policy {
            (*(out_orchestrator)).policy = Some(abce_sdma_policy_trampoline);
            (*(out_orchestrator)).policy_user_data =
                core::ptr::addr_of_mut!((*(out_orchestrator)).owned_policy)
                    .cast::<core::ffi::c_void>();
        }
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_initialize(
    builder: *const abce_builder_t,
    out_orchestrator: *mut abce_copy_orchestrator_t,
) {
    unsafe {
        abce_copy_orchestrator_initialize_with_caps(
            builder,
            abce_detect_default_platform_caps((*(builder)).isa),
            out_orchestrator,
        );
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_sdma_policy(
    orchestrator: *mut abce_copy_orchestrator_t,
) -> *mut abce_sdma_engine_policy_t {
    unsafe {
        if (*(orchestrator)).has_policy {
            core::ptr::addr_of_mut!((*(orchestrator)).owned_policy)
        } else {
            core::ptr::null_mut()
        }
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_set_engine_policy(
    orchestrator: *mut abce_copy_orchestrator_t,
    policy: abce_engine_policy_fn_t,
    user_data: *mut core::ffi::c_void,
) {
    unsafe {
        (*(orchestrator)).policy = policy;
        (*(orchestrator)).policy_user_data = if (policy).is_some() {
            user_data
        } else {
            core::ptr::null_mut()
        };
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_init_device_profile(
    orchestrator: *mut abce_copy_orchestrator_t,
    total_sdma_engines: u32,
    num_non_xgmi_sdma_engines: u32,
) {
    unsafe {
        if !((*(orchestrator)).has_policy) {
            return;
        }
        let isa: abce_isa_version_t = (*((*(orchestrator)).builder)).isa;
        if (((isa).major) == (9_u32)) && ((((isa).minor) == (4_u32)) || (((isa).minor) == (5_u32)))
        {
            abce_sdma_policy_select_heatmap_profile(
                core::ptr::addr_of_mut!((*(orchestrator)).owned_policy),
                ((isa).minor) as u8,
                (total_sdma_engines) as u8,
            );
        }
        abce_sdma_policy_set_engine_split(
            core::ptr::addr_of_mut!((*(orchestrator)).owned_policy),
            num_non_xgmi_sdma_engines,
        );
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_register_engine(
    orchestrator: *mut abce_copy_orchestrator_t,
    index: u32,
    ring: *mut abce_ring_t,
    affinity: *const abce_engine_affinity_t,
) -> bool {
    unsafe {
        if (((index) >= (16u32)) || (ring).is_null())
            || !(*core::ptr::addr_of_mut!((*(orchestrator)).engines)
                .cast::<*mut abce_ring_t>()
                .wrapping_offset((index) as isize))
            .is_null()
        {
            return false;
        }
        let mut resolved_affinity: abce_engine_affinity_t = { core::mem::zeroed() };
        if (affinity).is_null() {
            abce_engine_affinity_initialize(core::ptr::addr_of_mut!(resolved_affinity));
        } else {
            resolved_affinity = *(affinity);
        }
        if ((resolved_affinity).hw_engine_id) == (4_294_967_295_u32) {
            (resolved_affinity).hw_engine_id = index;
        }
        if ((resolved_affinity).hw_engine_id) >= (16u32) {
            return false;
        }
        (*core::ptr::addr_of_mut!((*(orchestrator)).engines)
            .cast::<*mut abce_ring_t>()
            .wrapping_offset((index) as isize)) = ring;
        (*(orchestrator)).registered_mask |= (1u64).wrapping_shl(u64::from(index) as u32);
        (*core::ptr::addr_of_mut!((*(orchestrator)).affinity)
            .cast::<abce_engine_affinity_t>()
            .wrapping_offset((index) as isize)) = resolved_affinity;
        if (*(orchestrator)).has_policy {
            abce_sdma_policy_map_engine(
                core::ptr::addr_of_mut!((*(orchestrator)).owned_policy),
                (resolved_affinity).hw_engine_id,
                index,
            );
        }
        true
    }
}

#[must_use]
pub unsafe extern "C" fn abce_copy_orchestrator_engine_affinity(
    orchestrator: *const abce_copy_orchestrator_t,
    index: u32,
) -> *const abce_engine_affinity_t {
    unsafe {
        core::ptr::addr_of!(
            (*core::ptr::addr_of!((*(orchestrator)).affinity)
                .cast::<abce_engine_affinity_t>()
                .wrapping_offset((index) as isize))
        )
    }
}

#[must_use]
pub unsafe extern "C" fn abce_copy_orchestrator_describe_execution(
    orchestrator: *const abce_copy_orchestrator_t,
    plan: *const abce_plan_t,
) -> u32 {
    unsafe {
        let mut hw_engine_mask: u32 = 0_u32;
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_27 = true;
            loop {
                if !first_27 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_27 = false;
                if (frame_idx) >= ((*(plan)).num_frames) {
                    break;
                }
                hw_engine_mask |= (1_u32).wrapping_shl(
                    (*core::ptr::addr_of!((*(orchestrator)).affinity)
                        .cast::<abce_engine_affinity_t>()
                        .wrapping_offset(
                            ((*core::ptr::addr_of!((*(plan)).frames)
                                .cast::<abce_plan_frame_t>()
                                .wrapping_offset((frame_idx) as isize))
                            .engine) as isize,
                        ))
                    .hw_engine_id,
                );
            }
        }
        abce_execution_descriptor_encode(
            ABCE_EXECUTION_ENGINE_KIND_SDMA,
            hw_engine_mask,
            abce_plan_uniform_transfer_kind(plan),
            abce_plan_mixed_transfer_kinds(plan),
        )
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_candidates(
    orchestrator: *const abce_copy_orchestrator_t,
    metadata: *const abce_copy_metadata_t,
) -> u64 {
    unsafe {
        if ((*(metadata)).engine_mask) != 0 {
            (*(metadata)).engine_mask & (*(orchestrator)).registered_mask
        } else {
            (*(orchestrator)).registered_mask
        }
    }
}

pub(super) unsafe extern "C" fn abce_engine_at(mask: u64, mut nth: u32) -> u32 {
    {
        {
            let mut engine: u32 = 0_u32;
            let mut first_28 = true;
            loop {
                if !first_28 {
                    engine = (engine).wrapping_add(1u32);
                }
                first_28 = false;
                if (engine) >= (16u32) {
                    break;
                }
                {
                    if (mask & ((1u64).wrapping_shl(u64::from(engine) as u32))) != 0 {
                        if (nth) == (0_u32) {
                            return engine;
                        }
                        nth = (nth).wrapping_sub(1u32);
                    }
                }
            }
        }
        16u32
    }
}

pub(super) unsafe extern "C" fn abce_popcount_engines(mask: u64) -> u32 {
    unsafe { (abce_popcount_u64(mask)) as u32 }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_invoke_policy(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    candidates: u64,
    ranked: *mut u32,
) -> abce_policy_result_t {
    unsafe {
        let mut result: abce_policy_result_t = { core::mem::zeroed() };
        (result).disposition = ABCE_POLICY_DISPOSITION_NO_PREFERENCE;
        (result).count = 0_u32;
        if ((*(orchestrator)).policy).is_none() {
            return result;
        }
        result = (match (*(orchestrator)).policy {
            Some(callback) => callback,
            None => std::process::abort(),
        })(
            (*(orchestrator)).policy_user_data,
            core::ptr::addr_of!((*(op)).src_end),
            core::ptr::addr_of!((*(op)).dst_end),
            candidates,
            ranked,
            16u32,
        );
        if (((result).disposition) as u32) != ABCE_POLICY_DISPOSITION_RANKED {
            (result).count = 0_u32;
            return result;
        }
        let reported_count: u32 = if ((result).count) < (16u32) {
            (result).count
        } else {
            16u32
        };
        let mut emitted: u64 = 0_u64;
        let mut valid_count: u32 = 0_u32;
        {
            let mut rank_idx: u32 = 0_u32;
            let mut first_29 = true;
            loop {
                if !first_29 {
                    rank_idx = (rank_idx).wrapping_add(1u32);
                }
                first_29 = false;
                if (rank_idx) >= (reported_count) {
                    break;
                }
                {
                    let engine: u32 = *(ranked).wrapping_offset((rank_idx) as isize);
                    if (engine) >= (16u32) {
                        continue;
                    }
                    let engine_bit: u64 = (1u64).wrapping_shl(u64::from(engine) as u32);
                    if ((candidates & engine_bit) == 0) || (emitted & engine_bit) != 0 {
                        continue;
                    }
                    emitted |= engine_bit;
                    (*(ranked).wrapping_offset(
                        ({
                            let old = valid_count;
                            valid_count = (valid_count).wrapping_add(1u32);
                            old
                        }) as isize,
                    )) = engine;
                }
            }
        }
        (result).disposition = (if (valid_count) != 0 {
            ABCE_POLICY_DISPOSITION_RANKED as i32
        } else {
            ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE as i32
        }) as abce_policy_disposition_t;
        (result).count = valid_count;
        result
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_best_engine(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    candidates: u64,
) -> u32 {
    unsafe {
        let mut ranked: [u32; 16] = [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result: abce_policy_result_t = abce_copy_orchestrator_invoke_policy(
            orchestrator,
            op,
            candidates,
            core::ptr::addr_of_mut!(ranked).cast::<u32>(),
        );
        if ((result).disposition) == ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE {
            return 16u32;
        }
        if ((result).disposition) == ABCE_POLICY_DISPOSITION_RANKED {
            return *core::ptr::addr_of_mut!(ranked)
                .cast::<u32>()
                .wrapping_offset((0i32) as isize);
        }
        abce_first_engine(candidates)
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_back_to_back_engine(
    orchestrator: *const abce_copy_orchestrator_t,
    first_copy: *const abce_copy_op_t,
    candidates: u64,
) -> u32 {
    unsafe {
        let isa: abce_isa_version_t = (*((*(orchestrator)).builder)).isa;
        if ((*(orchestrator)).has_policy && (((isa).major) == (9_u32)))
            && (((isa).minor) >= (4_u32))
        {
            let reversed_host_engine: u32 = abce_sdma_policy_registered_engine(
                core::ptr::addr_of!((*(orchestrator)).owned_policy),
                1_u32,
            );
            if ((reversed_host_engine) < (16u32))
                && (candidates & ((1_u64).wrapping_shl(u64::from(reversed_host_engine) as u32)))
                    != 0
            {
                return reversed_host_engine;
            }
        }
        abce_copy_orchestrator_best_engine(orchestrator, first_copy, candidates)
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_rank_engine_choices(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    candidates: u64,
    out_choices: *mut u32,
    out_has_engine_preference: *mut bool,
) -> u8 {
    unsafe {
        let mut ranked: [u32; 16] = [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result: abce_policy_result_t = abce_copy_orchestrator_invoke_policy(
            orchestrator,
            op,
            candidates,
            core::ptr::addr_of_mut!(ranked).cast::<u32>(),
        );
        (*(out_has_engine_preference)) = ((result).disposition) == ABCE_POLICY_DISPOSITION_RANKED;
        if ((result).disposition) == ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE {
            return (0i32) as u8;
        }
        let num_candidates: u32 = abce_popcount_engines(candidates);
        let num_choices: u32 = if ((result).disposition) == ABCE_POLICY_DISPOSITION_RANKED {
            if ((result).count) < (16u32) {
                (result).count
            } else {
                16u32
            }
        } else {
            if (num_candidates) < (16u32) {
                num_candidates
            } else {
                16u32
            }
        };
        {
            let mut choice_idx: u32 = 0_u32;
            let mut first_30 = true;
            loop {
                if !first_30 {
                    choice_idx = (choice_idx).wrapping_add(1u32);
                }
                first_30 = false;
                if (choice_idx) >= (num_choices) {
                    break;
                }
                {
                    (*(out_choices).wrapping_offset((choice_idx) as isize)) =
                        if ((result).disposition) == ABCE_POLICY_DISPOSITION_RANKED {
                            *core::ptr::addr_of_mut!(ranked)
                                .cast::<u32>()
                                .wrapping_offset((choice_idx) as isize)
                        } else {
                            abce_engine_at(candidates, choice_idx)
                        };
                }
            }
        }
        (num_choices) as u8
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_select_engine(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    candidates: u64,
    can_fanout: bool,
    max_participants: u32,
    round_robin: *mut u32,
    selected_mask: *mut u64,
    out_ranked_engines: *mut u32,
    out_num_ranked_engines: *mut u8,
    out_has_engine_preference: *mut bool,
) -> u32 {
    unsafe {
        let participant_cap_reached: bool =
            (abce_popcount_engines(*(selected_mask))) >= (max_participants);
        let unselected_candidates: u64 = candidates & !(*(selected_mask));
        let add_participant: bool =
            (can_fanout && !(participant_cap_reached)) && ((unselected_candidates) != (0_u64));
        let allowed: u64 = if add_participant {
            unselected_candidates
        } else if ((*(selected_mask)) != (0_u64)) && (!(can_fanout) || participant_cap_reached) {
            *(selected_mask)
        } else {
            candidates
        };
        (*(out_num_ranked_engines)) = abce_copy_orchestrator_rank_engine_choices(
            orchestrator,
            op,
            candidates,
            out_ranked_engines,
            out_has_engine_preference,
        );
        let mut allowed_ranked_engines: [u32; 16] =
            [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut allowed_has_engine_preference: bool = false;
        let num_allowed_ranked_engines: u8 = abce_copy_orchestrator_rank_engine_choices(
            orchestrator,
            op,
            allowed,
            core::ptr::addr_of_mut!(allowed_ranked_engines).cast::<u32>(),
            core::ptr::addr_of_mut!(allowed_has_engine_preference),
        );
        if i32::from(num_allowed_ranked_engines) == (0i32) {
            return 16u32;
        }
        let engine: u32 = if add_participant || !(can_fanout) {
            *core::ptr::addr_of_mut!(allowed_ranked_engines)
                .cast::<u32>()
                .wrapping_offset((0i32) as isize)
        } else {
            *core::ptr::addr_of_mut!(allowed_ranked_engines)
                .cast::<u32>()
                .wrapping_offset(
                    ({
                        let old = *(round_robin);
                        (*(round_robin)) = (*(round_robin)).wrapping_add(1u32);
                        old
                    } % u32::from(num_allowed_ranked_engines)) as isize,
                )
        };
        if (engine) < (16u32) {
            (*(selected_mask)) |= (1_u64).wrapping_shl(u64::from(engine) as u32);
        }
        engine
    }
}

pub(super) unsafe extern "C" fn abce_pick_balanced_engine(
    legal_mask: u64,
    recommended: u32,
    load: *const u32,
) -> u32 {
    unsafe {
        let mut min_load: u32 = 4_294_967_295_u32;
        {
            let mut engine: u32 = 0_u32;
            let mut first_31 = true;
            loop {
                if !first_31 {
                    engine = (engine).wrapping_add(1u32);
                }
                first_31 = false;
                if (engine) >= (16u32) {
                    break;
                }
                if (legal_mask & ((1_u64).wrapping_shl(u64::from(engine) as u32))) != 0
                    && ((*(load).wrapping_offset((engine) as isize)) < (min_load))
                {
                    min_load = *(load).wrapping_offset((engine) as isize);
                }
            }
        }
        if (min_load) == (4_294_967_295_u32) {
            return 16u32;
        }
        if (((recommended) < (16u32))
            && (legal_mask & ((1_u64).wrapping_shl(u64::from(recommended) as u32))) != 0)
            && ((*(load).wrapping_offset((recommended) as isize)) == (min_load))
        {
            return recommended;
        }
        {
            let mut engine: u32 = 0_u32;
            let mut first_32 = true;
            loop {
                if !first_32 {
                    engine = (engine).wrapping_add(1u32);
                }
                first_32 = false;
                if (engine) >= (16u32) {
                    break;
                }
                if (legal_mask & ((1_u64).wrapping_shl(u64::from(engine) as u32))) != 0
                    && ((*(load).wrapping_offset((engine) as isize)) == (min_load))
                {
                    return engine;
                }
            }
        }
        16u32
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_legal_engine_set(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    candidates: u64,
    out_recommended: *mut u32,
    out_ranked_engines: *mut u32,
    out_num_ranked_engines: *mut u8,
    out_has_engine_preference: *mut bool,
) -> u64 {
    unsafe {
        if (*(orchestrator)).is_gfx125plus {
            (*(out_num_ranked_engines)) = (0i32) as u8;
            {
                let mut engine: u32 = 0_u32;
                let mut first_33 = true;
                loop {
                    if !first_33 {
                        engine = (engine).wrapping_add(1u32);
                    }
                    first_33 = false;
                    if !(((engine) < (16u32)) && (u32::from(*(out_num_ranked_engines)) < (16u32))) {
                        break;
                    }
                    if (candidates & ((1_u64).wrapping_shl(u64::from(engine) as u32))) != 0 {
                        (*(out_ranked_engines).wrapping_offset(
                            ({
                                let old = *(out_num_ranked_engines);
                                (*(out_num_ranked_engines)) =
                                    (*(out_num_ranked_engines)).wrapping_add(1u8);
                                old
                            }) as isize,
                        )) = engine;
                    }
                }
            }
            (*(out_has_engine_preference)) = false;
            (*(out_recommended)) = 16u32;
            return candidates;
        }
        (*(out_num_ranked_engines)) = abce_copy_orchestrator_rank_engine_choices(
            orchestrator,
            op,
            candidates,
            out_ranked_engines,
            out_has_engine_preference,
        );
        let mut legal_mask: u64 = 0_u64;
        {
            let mut choice_idx: u32 = 0_u32;
            let mut first_34 = true;
            loop {
                if !first_34 {
                    choice_idx = (choice_idx).wrapping_add(1u32);
                }
                first_34 = false;
                if (choice_idx) >= u32::from(*(out_num_ranked_engines)) {
                    break;
                }
                legal_mask |= (1_u64).wrapping_shl(u64::from(
                    *(out_ranked_engines).wrapping_offset((choice_idx) as isize),
                ) as u32);
            }
        }
        (*(out_recommended)) = if (*(out_num_ranked_engines)) != 0 {
            *(out_ranked_engines).wrapping_offset((0i32) as isize)
        } else {
            16u32
        };
        legal_mask
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_rank_legal_engines(
    orchestrator: *const abce_copy_orchestrator_t,
    copy: *const abce_copy_op_t,
    out_engines: *mut u32,
    capacity: u32,
) -> u32 {
    unsafe {
        if (out_engines).is_null() || ((capacity) == (0_u32)) {
            return 0_u32;
        }
        let mut ranked: [u32; 16] = [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut recommended: u32 = 16u32;
        let mut num_ranked: u8 = (0i32) as u8;
        let mut has_engine_preference: bool = false;
        let _ = abce_copy_orchestrator_legal_engine_set(
            orchestrator,
            copy,
            (*(orchestrator)).registered_mask,
            core::ptr::addr_of_mut!(recommended),
            core::ptr::addr_of_mut!(ranked).cast::<u32>(),
            core::ptr::addr_of_mut!(num_ranked),
            core::ptr::addr_of_mut!(has_engine_preference),
        );
        let count: u32 = if u32::from(num_ranked) < (capacity) {
            u32::from(num_ranked)
        } else {
            capacity
        };
        {
            let mut rank: u32 = 0_u32;
            let mut first_35 = true;
            loop {
                if !first_35 {
                    rank = (rank).wrapping_add(1u32);
                }
                first_35 = false;
                if (rank) >= (count) {
                    break;
                }
                (*(out_engines).wrapping_offset((rank) as isize)) =
                    *core::ptr::addr_of_mut!(ranked)
                        .cast::<u32>()
                        .wrapping_offset((rank) as isize);
            }
        }
        count
    }
}

pub(super) unsafe extern "C" fn abce_copy_op_max_packets(
    builder: *const abce_builder_t,
    copy: *const abce_copy_op_t,
) -> u64 {
    unsafe {
        'switch: {
            match (*(copy)).kind {
                value
                    if value == ABCE_OP_KIND_LINEAR
                        || value == ABCE_OP_KIND_MULTICAST
                        || value == ABCE_OP_KIND_BROADCAST =>
                {
                    return abce_builder_num_copy_packets(builder, (*(copy)).size);
                }
                value if value == ABCE_OP_KIND_SWAP => {
                    return if (abce_builder_num_swap_packets(builder, (*(copy)).size))
                        > (abce_builder_num_wait_signal_swap_packets(
                            builder,
                            (*(copy)).size,
                            if ((*(copy)).size2) != 0 {
                                (*(copy)).size2
                            } else {
                                (*(copy)).size
                            },
                        )) {
                        abce_builder_num_swap_packets(builder, (*(copy)).size)
                    } else {
                        abce_builder_num_wait_signal_swap_packets(
                            builder,
                            (*(copy)).size,
                            if ((*(copy)).size2) != 0 {
                                (*(copy)).size2
                            } else {
                                (*(copy)).size
                            },
                        )
                    };
                }
                value if value == ABCE_OP_KIND_FILL => {
                    return abce_builder_num_fill_packets(
                        builder,
                        (((*(copy)).size) / (core::mem::size_of::<u32>() as usize) as usize)
                            as usize,
                    );
                }
                value if value == ABCE_OP_KIND_INDIRECT || value == ABCE_OP_KIND_COPY_RECT => {
                    break 'switch;
                }
                _ => {}
            }
        }
        1_u64
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_validate_request(
    orchestrator: *const abce_copy_orchestrator_t,
    copies: *const abce_copy_op_t,
    num_copies: u32,
    metadata: *const abce_copy_metadata_t,
    out_failed_op: *mut u32,
) -> abce_status_t {
    unsafe {
        if ((((copies).is_null() || ((num_copies) == (0_u32)))
            || ((num_copies) > ((64u32).wrapping_mul(1024u32))))
            || (((*(metadata)).out).value).is_null())
            || ((((*(metadata)).num_deps) != (0_u32)) && ((*(metadata)).deps).is_null())
        {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_36 = true;
            loop {
                if !first_36 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_36 = false;
                if (op_idx) >= (num_copies) {
                    break;
                }
                {
                    let copy: *const abce_copy_op_t =
                        core::ptr::addr_of!((*(copies).wrapping_offset((op_idx) as isize)));
                    (*(out_failed_op)) = op_idx;
                    if ((((*(copy)).src_end).kind) == ABCE_ENDPOINT_KIND_HOST)
                        && ((((*(copy)).dst_end).kind) == ABCE_ENDPOINT_KIND_HOST)
                    {
                        return ABCE_STATUS_INVALID_ARGUMENT;
                    }
                    if ((*(copy)).kind) == ABCE_OP_KIND_COPY_RECT {
                        if ((((((*(copy)).rect).is_null()
                            || (((*((*(copy)).rect)).src).base).is_null())
                            || (((*((*(copy)).rect)).dst).base).is_null())
                            || ((((*((*(copy)).rect)).range).x) == (0_u32)))
                            || ((((*((*(copy)).rect)).range).y) == (0_u32)))
                            || ((((*((*(copy)).rect)).range).z) == (0_u32))
                        {
                            return ABCE_STATUS_INVALID_ARGUMENT;
                        }
                        let mut num_rect_packets: u32 = 0_u32;
                        let rect_status: abce_status_t = abce_builder_num_rect_packets(
                            (*(orchestrator)).builder,
                            core::ptr::addr_of!((*((*(copy)).rect)).dst),
                            core::ptr::addr_of!((*((*(copy)).rect)).dst_offset),
                            core::ptr::addr_of!((*((*(copy)).rect)).src),
                            core::ptr::addr_of!((*((*(copy)).rect)).src_offset),
                            core::ptr::addr_of!((*((*(copy)).rect)).range),
                            core::ptr::addr_of_mut!(num_rect_packets),
                        );
                        if !(abce_status_is_ok(rect_status)) {
                            return rect_status;
                        }
                        continue;
                    }
                    if ((*(copy)).size) == (0_usize) {
                        return ABCE_STATUS_INVALID_ARGUMENT;
                    }
                    'switch: {
                        match (*(copy)).kind {
                            value if value == ABCE_OP_KIND_LINEAR => {
                                if ((*(copy)).src).is_null() || ((*(copy)).dst).is_null() {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                break 'switch;
                            }
                            value if value == ABCE_OP_KIND_SWAP => {
                                if ((*(copy)).src).is_null() || ((*(copy)).dst).is_null() {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                if !(abce_builder_swap_supported((*(orchestrator)).builder)) {
                                    return ABCE_STATUS_UNIMPLEMENTED;
                                }
                                let misalignment_mask: usize =
                                    (abce_builder_swap_alignment((*(orchestrator)).builder))
                                        .wrapping_sub(1_usize);
                                if ((((*(copy)).src) as usize | ((*(copy)).dst) as usize)
                                    & misalignment_mask)
                                    != (0_usize)
                                {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                break 'switch;
                            }
                            value if value == ABCE_OP_KIND_INDIRECT => {
                                if !((*(orchestrator)).is_gfx125plus) {
                                    return ABCE_STATUS_UNIMPLEMENTED;
                                }
                                if (((*(copy)).src).is_null() || ((*(copy)).dst).is_null())
                                    || (((*(copy)).size)
                                        > (((1_u64).wrapping_shl((30_u64) as u32)) as usize))
                                {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                break 'switch;
                            }
                            value
                                if value == ABCE_OP_KIND_MULTICAST
                                    || value == ABCE_OP_KIND_BROADCAST =>
                            {
                                if (((*(copy)).src).is_null() || ((*(copy)).dsts).is_null())
                                    || (((*(copy)).num_dsts) == (0_u32))
                                {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                if (((*(copy)).kind) == ABCE_OP_KIND_BROADCAST)
                                    && (((*(copy)).num_dsts) != (2u32))
                                {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                {
                                    let mut dst_idx: u32 = 0_u32;
                                    let mut first_37 = true;
                                    loop {
                                        if !first_37 {
                                            dst_idx = (dst_idx).wrapping_add(1u32);
                                        }
                                        first_37 = false;
                                        if (dst_idx) >= ((*(copy)).num_dsts) {
                                            break;
                                        }
                                        if (*((*(copy)).dsts).wrapping_offset((dst_idx) as isize))
                                            .is_null()
                                        {
                                            return (ABCE_STATUS_INVALID_ARGUMENT as i32)
                                                as abce_status_t;
                                        }
                                    }
                                }
                                break 'switch;
                            }
                            value if value == ABCE_OP_KIND_FILL => {
                                if ((*(copy)).dst).is_null()
                                    || ((((*(copy)).size)
                                        % (core::mem::size_of::<u32>() as usize) as usize)
                                        != ((0i32) as usize))
                                {
                                    return ABCE_STATUS_INVALID_ARGUMENT;
                                }
                                break 'switch;
                            }
                            value if value == ABCE_OP_KIND_COPY_RECT => {
                                break 'switch;
                            }
                            _ => {}
                        }
                    }
                    if (abce_copy_op_max_packets((*(orchestrator)).builder, copy))
                        > u64::from(4_294_967_295_u32)
                    {
                        return ABCE_STATUS_OUT_OF_RANGE;
                    }
                }
            }
        }
        (*(out_failed_op)) = 4_294_967_295_u32;
        ABCE_STATUS_OK
    }
}

pub(super) unsafe extern "C" fn abce_back_to_back_bytes(
    copies: *const abce_copy_op_t,
    num_copies: u32,
) -> usize {
    unsafe {
        let mut total: usize = 0_usize;
        {
            let mut copy_idx: u32 = 0_u32;
            let mut first_38 = true;
            loop {
                if !first_38 {
                    copy_idx = (copy_idx).wrapping_add(1u32);
                }
                first_38 = false;
                if (copy_idx) >= (num_copies) {
                    break;
                }
                {
                    let copy: *const abce_copy_op_t =
                        core::ptr::addr_of!((*(copies).wrapping_offset((copy_idx) as isize)));
                    let writes: u32 = if ((*(copy)).kind) == ABCE_OP_KIND_MULTICAST {
                        (*(copy)).num_dsts
                    } else {
                        1_u32
                    };
                    total = (total).wrapping_add(((*(copy)).size).wrapping_mul((writes) as usize));
                }
            }
        }
        total
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_is_back_to_back_batch(
    orchestrator: *const abce_copy_orchestrator_t,
    copies: *const abce_copy_op_t,
    num_copies: u32,
    metadata: *const abce_copy_metadata_t,
) -> bool {
    unsafe {
        if ((*(metadata)).linear_batch_mode) == ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT {
            return false;
        }
        let decomposed_multicast: bool = (((num_copies) == (1_u32))
            && (((*(copies).wrapping_offset((0i32) as isize)).kind) == ABCE_OP_KIND_MULTICAST))
            && !((*(orchestrator)).is_gfx125plus);
        if !(decomposed_multicast) {
            if (num_copies) <= (1_u32) {
                return false;
            }
            {
                let mut copy_idx: u32 = 0_u32;
                let mut first_39 = true;
                loop {
                    if !first_39 {
                        copy_idx = (copy_idx).wrapping_add(1u32);
                    }
                    first_39 = false;
                    if (copy_idx) >= (num_copies) {
                        break;
                    }
                    if ((*(copies).wrapping_offset((copy_idx) as isize)).kind)
                        != ABCE_OP_KIND_LINEAR
                    {
                        return false;
                    }
                }
            }
        }
        if ((*(metadata)).linear_batch_mode) == ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK {
            return true;
        }
        if (*(orchestrator)).is_gfx125plus {
            return false;
        }
        (abce_back_to_back_bytes(copies, num_copies)) <= (abce_linear_b2b_max_total())
    }
}

pub(super) unsafe extern "C" fn abce_push_engine_op(
    ops: *mut abce_engine_op_t,
    num_ops: *mut u32,
    capacity: u32,
    op: *const abce_engine_op_t,
) -> bool {
    unsafe {
        if (*(num_ops)) >= (capacity) {
            return false;
        }
        (*(ops).wrapping_offset(
            ({
                let old = *(num_ops);
                (*(num_ops)) = (*(num_ops)).wrapping_add(1u32);
                old
            }) as isize,
        )) = *(op);
        true
    }
}

pub(super) unsafe extern "C" fn abce_engine_op_initialize(out_op: *mut abce_engine_op_t) {
    unsafe {
        let _ = libc::memset(
            (out_op).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_engine_op_t>() as usize,
        );
        (*(out_op)).kind = ABCE_OP_KIND_LINEAR;
    }
}

pub(super) unsafe extern "C" fn abce_set_engine_choices(
    op: *mut abce_engine_op_t,
    ranked_engines: *const u32,
    num_ranked_engines: u8,
    has_engine_preference: bool,
) {
    unsafe {
        (*(op)).num_ranked_engines = num_ranked_engines;
        (*(op)).has_engine_preference = has_engine_preference;
        {
            let mut choice_idx: u32 = 0_u32;
            let mut first_40 = true;
            loop {
                if !first_40 {
                    choice_idx = (choice_idx).wrapping_add(1u32);
                }
                first_40 = false;
                if (choice_idx) >= u32::from(num_ranked_engines) {
                    break;
                }
                (*core::ptr::addr_of_mut!((*(op)).ranked_engines)
                    .cast::<u32>()
                    .wrapping_offset((choice_idx) as isize)) =
                    *(ranked_engines).wrapping_offset((choice_idx) as isize);
            }
        }
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_set_fixed_engine(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *mut abce_engine_op_t,
    copy: *const abce_copy_op_t,
    candidates: u64,
    engine: u32,
) {
    unsafe {
        (*(op)).engine = engine;
        let mut has_engine_preference: bool = false;
        (*(op)).num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
            orchestrator,
            copy,
            candidates,
            core::ptr::addr_of_mut!((*(op)).ranked_engines).cast::<u32>(),
            core::ptr::addr_of_mut!(has_engine_preference),
        );
        (*(op)).has_engine_preference = has_engine_preference;
        if i32::from((*(op)).num_ranked_engines) == (0i32) {
            (*core::ptr::addr_of_mut!((*(op)).ranked_engines)
                .cast::<u32>()
                .wrapping_offset((0i32) as isize)) = engine;
            (*(op)).num_ranked_engines = (1i32) as u8;
        }
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_decompose_multi_dst(
    orchestrator: *const abce_copy_orchestrator_t,
    op: *const abce_copy_op_t,
    metadata: *const abce_copy_metadata_t,
    candidates: u64,
    can_fanout: bool,
    max_participants: u32,
    force_back_to_back: bool,
    fixed_engine: u32,
    round_robin: *mut u32,
    selected_mask: *mut u64,
    ops: *mut abce_engine_op_t,
    num_ops: *mut u32,
    capacity: u32,
) -> bool {
    unsafe {
        let use_multicast: bool = (((*(metadata)).multicast_mode)
            == ABCE_MULTICAST_MODE_FORCE_MULTICAST)
            || ((((*(metadata)).multicast_mode) == ABCE_MULTICAST_MODE_AUTOMATIC)
                && (((*(op)).size) <= (((256u32).wrapping_mul(1024u32)) as usize)));
        if (*(orchestrator)).is_gfx125plus && use_multicast {
            let mut ranked_engines: [u32; 16] =
                [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut num_ranked_engines: u8 = (0i32) as u8;
            let mut has_engine_preference: bool = false;
            let multicast_engine: u32 = if force_back_to_back {
                fixed_engine
            } else {
                abce_copy_orchestrator_select_engine(
                    orchestrator,
                    op,
                    candidates,
                    can_fanout,
                    max_participants,
                    round_robin,
                    selected_mask,
                    core::ptr::addr_of_mut!(ranked_engines).cast::<u32>(),
                    core::ptr::addr_of_mut!(num_ranked_engines),
                    core::ptr::addr_of_mut!(has_engine_preference),
                )
            };
            if (multicast_engine) == (16u32) {
                return false;
            }
            if force_back_to_back {
                num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
                    orchestrator,
                    op,
                    candidates,
                    core::ptr::addr_of_mut!(ranked_engines).cast::<u32>(),
                    core::ptr::addr_of_mut!(has_engine_preference),
                );
                if i32::from(num_ranked_engines) == (0i32) {
                    (*core::ptr::addr_of_mut!(ranked_engines)
                        .cast::<u32>()
                        .wrapping_offset((0i32) as isize)) = fixed_engine;
                    num_ranked_engines = (1i32) as u8;
                }
            }
            {
                let mut base: u32 = 0_u32;
                let mut first_41 = true;
                loop {
                    if !first_41 {
                        base = (base).wrapping_add(1024u32);
                    }
                    first_41 = false;
                    if (base) >= ((*(op)).num_dsts) {
                        break;
                    }
                    {
                        let chunk_dsts: u32 = if (((*(op)).num_dsts).wrapping_sub(base)) < (1024u32)
                        {
                            ((*(op)).num_dsts).wrapping_sub(base)
                        } else {
                            1024u32
                        };
                        let mut engine_op: abce_engine_op_t = { core::mem::zeroed() };
                        abce_engine_op_initialize(core::ptr::addr_of_mut!(engine_op));
                        (engine_op).kind = ABCE_OP_KIND_MULTICAST;
                        (engine_op).src = (*(op)).src;
                        (engine_op).dsts =
                            core::ptr::addr_of!((*((*(op)).dsts).wrapping_offset((base) as isize)));
                        (engine_op).num_dsts = chunk_dsts;
                        (engine_op).size = (*(op)).size;
                        (engine_op).engine = multicast_engine;
                        abce_set_engine_choices(
                            core::ptr::addr_of_mut!(engine_op),
                            core::ptr::addr_of_mut!(ranked_engines)
                                .cast::<u32>()
                                .cast_const(),
                            num_ranked_engines,
                            has_engine_preference,
                        );
                        if !(abce_push_engine_op(
                            ops,
                            num_ops,
                            capacity,
                            core::ptr::addr_of_mut!(engine_op).cast_const(),
                        )) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        if !((*(orchestrator)).is_gfx125plus) && (((*(op)).size) < (abce_broadcast_max_size())) {
            let mut ranked_engines: [u32; 16] =
                [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            let mut num_ranked_engines: u8 = (0i32) as u8;
            let mut has_engine_preference: bool = false;
            let bcast_engine: u32 = if force_back_to_back {
                fixed_engine
            } else {
                abce_copy_orchestrator_select_engine(
                    orchestrator,
                    op,
                    candidates,
                    can_fanout,
                    max_participants,
                    round_robin,
                    selected_mask,
                    core::ptr::addr_of_mut!(ranked_engines).cast::<u32>(),
                    core::ptr::addr_of_mut!(num_ranked_engines),
                    core::ptr::addr_of_mut!(has_engine_preference),
                )
            };
            if (bcast_engine) == (16u32) {
                return false;
            }
            if force_back_to_back {
                num_ranked_engines = abce_copy_orchestrator_rank_engine_choices(
                    orchestrator,
                    op,
                    candidates,
                    core::ptr::addr_of_mut!(ranked_engines).cast::<u32>(),
                    core::ptr::addr_of_mut!(has_engine_preference),
                );
                if i32::from(num_ranked_engines) == (0i32) {
                    (*core::ptr::addr_of_mut!(ranked_engines)
                        .cast::<u32>()
                        .wrapping_offset((0i32) as isize)) = fixed_engine;
                    num_ranked_engines = (1i32) as u8;
                }
            }
            {
                let mut base: u32 = 0_u32;
                let mut first_42 = true;
                loop {
                    if !first_42 {
                        base = (base).wrapping_add(2u32);
                    }
                    first_42 = false;
                    if (base) >= ((*(op)).num_dsts) {
                        break;
                    }
                    {
                        let chunk_dsts: u32 = if (((*(op)).num_dsts).wrapping_sub(base)) < (2u32) {
                            ((*(op)).num_dsts).wrapping_sub(base)
                        } else {
                            2u32
                        };
                        let mut engine_op: abce_engine_op_t = { core::mem::zeroed() };
                        abce_engine_op_initialize(core::ptr::addr_of_mut!(engine_op));
                        (engine_op).src = (*(op)).src;
                        (engine_op).dsts =
                            core::ptr::addr_of!((*((*(op)).dsts).wrapping_offset((base) as isize)));
                        (engine_op).size = (*(op)).size;
                        (engine_op).engine = bcast_engine;
                        abce_set_engine_choices(
                            core::ptr::addr_of_mut!(engine_op),
                            core::ptr::addr_of_mut!(ranked_engines)
                                .cast::<u32>()
                                .cast_const(),
                            num_ranked_engines,
                            has_engine_preference,
                        );
                        if (chunk_dsts) == (2u32) {
                            (engine_op).kind = ABCE_OP_KIND_BROADCAST;
                            (engine_op).num_dsts = 2u32;
                        } else {
                            (engine_op).kind = ABCE_OP_KIND_LINEAR;
                            (engine_op).dst = *((*(op)).dsts).wrapping_offset((base) as isize);
                        }
                        if !(abce_push_engine_op(
                            ops,
                            num_ops,
                            capacity,
                            core::ptr::addr_of_mut!(engine_op).cast_const(),
                        )) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        let num_candidates: u32 = abce_popcount_engines(candidates);
        let num_group_engines: u32 = if (max_participants) < (num_candidates) {
            max_participants
        } else {
            num_candidates
        };
        let use_large_copy_grouping: bool = ((*(orchestrator)).is_gfx125plus && can_fanout)
            && (((*(op)).size) >= (((1_u64).wrapping_shl((30_u64) as u32)) as usize));
        {
            let mut dst_idx: u32 = 0_u32;
            let mut first_43 = true;
            loop {
                if !first_43 {
                    dst_idx = (dst_idx).wrapping_add(1u32);
                }
                first_43 = false;
                if (dst_idx) >= ((*(op)).num_dsts) {
                    break;
                }
                {
                    let mut engine_op: abce_engine_op_t = { core::mem::zeroed() };
                    abce_engine_op_initialize(core::ptr::addr_of_mut!(engine_op));
                    (engine_op).kind = ABCE_OP_KIND_LINEAR;
                    (engine_op).src = (*(op)).src;
                    (engine_op).dst = *((*(op)).dsts).wrapping_offset((dst_idx) as isize);
                    (engine_op).size = (*(op)).size;
                    if force_back_to_back {
                        abce_copy_orchestrator_set_fixed_engine(
                            orchestrator,
                            core::ptr::addr_of_mut!(engine_op),
                            op,
                            candidates,
                            fixed_engine,
                        );
                    } else if use_large_copy_grouping {
                        let engine_slot: u32 = (dst_idx / 8u32) % num_group_engines;
                        abce_copy_orchestrator_set_fixed_engine(
                            orchestrator,
                            core::ptr::addr_of_mut!(engine_op),
                            op,
                            candidates,
                            abce_engine_at(candidates, engine_slot),
                        );
                    } else {
                        let mut has_engine_preference: bool = false;
                        (engine_op).engine = abce_copy_orchestrator_select_engine(
                            orchestrator,
                            op,
                            candidates,
                            can_fanout,
                            max_participants,
                            round_robin,
                            selected_mask,
                            core::ptr::addr_of_mut!((engine_op).ranked_engines).cast::<u32>(),
                            core::ptr::addr_of_mut!((engine_op).num_ranked_engines),
                            core::ptr::addr_of_mut!(has_engine_preference),
                        );
                        (engine_op).has_engine_preference = has_engine_preference;
                    }
                    if ((engine_op).engine) == (16u32) {
                        return false;
                    }
                    if !(abce_push_engine_op(
                        ops,
                        num_ops,
                        capacity,
                        core::ptr::addr_of_mut!(engine_op).cast_const(),
                    )) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

pub(super) unsafe extern "C" fn abce_copy_orchestrator_decompose(
    orchestrator: *const abce_copy_orchestrator_t,
    copies: *const abce_copy_op_t,
    num_copies: u32,
    metadata: *const abce_copy_metadata_t,
    candidates: u64,
    force_back_to_back: bool,
    fixed_engine: u32,
    ops: *mut abce_engine_op_t,
    num_ops: *mut u32,
    capacity: u32,
    out_status: *mut abce_status_t,
    out_failed_op: *mut u32,
) -> bool {
    unsafe {
        let num_candidates: u32 = abce_popcount_engines(candidates);
        let can_fanout: bool = (!(force_back_to_back) && ((num_candidates) > (1_u32)))
            && ((*(orchestrator)).caps).device_atomic_support;
        let max_participants: u32 = if force_back_to_back {
            1_u32
        } else if ((*(metadata)).max_engines) == (0_u32) {
            if can_fanout { 16u32 } else { 1_u32 }
        } else {
            if ((*(metadata)).max_engines) < (16u32) {
                (*(metadata)).max_engines
            } else {
                16u32
            }
        };
        let num_group_engines: u32 = if (max_participants) < (num_candidates) {
            max_participants
        } else {
            num_candidates
        };
        let mut use_large_copy_grouping: bool = (*(orchestrator)).is_gfx125plus && can_fanout;
        {
            let mut copy_idx: u32 = 0_u32;
            let mut first_44 = true;
            loop {
                if !first_44 {
                    copy_idx = (copy_idx).wrapping_add(1u32);
                }
                first_44 = false;
                if !(((copy_idx) < (num_copies)) && use_large_copy_grouping) {
                    break;
                }
                {
                    use_large_copy_grouping = (((*(copies).wrapping_offset((copy_idx) as isize))
                        .kind)
                        == ABCE_OP_KIND_LINEAR)
                        && (((*(copies).wrapping_offset((copy_idx) as isize)).size)
                            >= (((1_u64).wrapping_shl((30_u64) as u32)) as usize));
                }
            }
        }
        let mut round_robin: u32 = 0_u32;
        let mut selected_mask: u64 = 0_u64;
        let mut engine_load: [u32; 16] = [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_45 = true;
            loop {
                if !first_45 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_45 = false;
                if (op_idx) >= (num_copies) {
                    break;
                }
                {
                    let copy: *const abce_copy_op_t =
                        core::ptr::addr_of!((*(copies).wrapping_offset((op_idx) as isize)));
                    (*(out_failed_op)) = op_idx;
                    if ((*(copy)).kind) == ABCE_OP_KIND_MULTICAST {
                        if !(abce_copy_orchestrator_decompose_multi_dst(
                            orchestrator,
                            copy,
                            metadata,
                            candidates,
                            can_fanout,
                            max_participants,
                            force_back_to_back,
                            fixed_engine,
                            core::ptr::addr_of_mut!(round_robin),
                            core::ptr::addr_of_mut!(selected_mask),
                            ops,
                            num_ops,
                            capacity,
                        )) {
                            (*(out_status)) = (if (*(num_ops)) >= (capacity) {
                                ABCE_STATUS_TOO_MANY_OPERATIONS as i32
                            } else {
                                ABCE_STATUS_NO_LEGAL_ENGINE as i32
                            }) as abce_status_t;
                            return false;
                        }
                        continue;
                    }
                    let mut engine_op: abce_engine_op_t = { core::mem::zeroed() };
                    abce_engine_op_initialize(core::ptr::addr_of_mut!(engine_op));
                    (engine_op).kind = (*(copy)).kind;
                    (engine_op).src = (*(copy)).src;
                    (engine_op).dst = (*(copy)).dst;
                    (engine_op).dsts = (*(copy)).dsts;
                    (engine_op).num_dsts = (*(copy)).num_dsts;
                    (engine_op).fill_value = (*(copy)).fill_value;
                    (engine_op).size = (*(copy)).size;
                    (engine_op).size2 = (*(copy)).size2;
                    (engine_op).indirect_src = (*(copy)).indirect_src;
                    (engine_op).indirect_dst = (*(copy)).indirect_dst;
                    (engine_op).rect = (*(copy)).rect;
                    if force_back_to_back {
                        abce_copy_orchestrator_set_fixed_engine(
                            orchestrator,
                            core::ptr::addr_of_mut!(engine_op),
                            copy,
                            candidates,
                            fixed_engine,
                        );
                    } else if use_large_copy_grouping {
                        let engine_slot: u32 = (op_idx / 8u32) % num_group_engines;
                        abce_copy_orchestrator_set_fixed_engine(
                            orchestrator,
                            core::ptr::addr_of_mut!(engine_op),
                            copy,
                            candidates,
                            abce_engine_at(candidates, engine_slot),
                        );
                    } else if can_fanout {
                        let mut has_engine_preference: bool = false;
                        let mut recommended: u32 = 16u32;
                        let legal_mask: u64 = abce_copy_orchestrator_legal_engine_set(
                            orchestrator,
                            copy,
                            candidates,
                            core::ptr::addr_of_mut!(recommended),
                            core::ptr::addr_of_mut!((engine_op).ranked_engines).cast::<u32>(),
                            core::ptr::addr_of_mut!((engine_op).num_ranked_engines),
                            core::ptr::addr_of_mut!(has_engine_preference),
                        );
                        (engine_op).has_engine_preference = has_engine_preference;
                        let can_add_participant: bool =
                            (abce_popcount_engines(selected_mask)) < (max_participants);
                        let unselected_legal: u64 = legal_mask & !(selected_mask);
                        let selected_legal: u64 = legal_mask & selected_mask;
                        let allowed_mask: u64 =
                            if can_add_participant && ((unselected_legal) != (0_u64)) {
                                unselected_legal
                            } else {
                                selected_legal
                            };
                        (engine_op).engine = abce_pick_balanced_engine(
                            allowed_mask,
                            recommended,
                            core::ptr::addr_of_mut!(engine_load)
                                .cast::<u32>()
                                .cast_const(),
                        );
                        if ((engine_op).engine) < (16u32) {
                            selected_mask |=
                                (1_u64).wrapping_shl(u64::from((engine_op).engine) as u32);
                            (*core::ptr::addr_of_mut!(engine_load)
                                .cast::<u32>()
                                .wrapping_offset(((engine_op).engine) as isize)) =
                                (*core::ptr::addr_of_mut!(engine_load)
                                    .cast::<u32>()
                                    .wrapping_offset(((engine_op).engine) as isize))
                                .wrapping_add(1u32);
                        }
                    } else {
                        let mut has_engine_preference: bool = false;
                        (engine_op).engine = abce_copy_orchestrator_select_engine(
                            orchestrator,
                            copy,
                            candidates,
                            can_fanout,
                            max_participants,
                            core::ptr::addr_of_mut!(round_robin),
                            core::ptr::addr_of_mut!(selected_mask),
                            core::ptr::addr_of_mut!((engine_op).ranked_engines).cast::<u32>(),
                            core::ptr::addr_of_mut!((engine_op).num_ranked_engines),
                            core::ptr::addr_of_mut!(has_engine_preference),
                        );
                        (engine_op).has_engine_preference = has_engine_preference;
                    }
                    if ((engine_op).engine) == (16u32) {
                        (*(out_status)) = ABCE_STATUS_NO_LEGAL_ENGINE;
                        return false;
                    }
                    if !(abce_push_engine_op(
                        ops,
                        num_ops,
                        capacity,
                        core::ptr::addr_of_mut!(engine_op).cast_const(),
                    )) {
                        (*(out_status)) = ABCE_STATUS_TOO_MANY_OPERATIONS;
                        return false;
                    }
                }
            }
        }
        (*(out_status)) = ABCE_STATUS_OK;
        (*(out_failed_op)) = 4_294_967_295_u32;
        true
    }
}

pub(super) unsafe extern "C" fn abce_initialize_frame_choices(
    frame: *mut abce_plan_frame_t,
    op: *const abce_engine_op_t,
) {
    unsafe {
        (*(frame)).num_ranked_engines = (*(op)).num_ranked_engines;
        (*(frame)).has_engine_preference = (*(op)).has_engine_preference;
        {
            let mut choice_idx: u32 = 0_u32;
            let mut first_46 = true;
            loop {
                if !first_46 {
                    choice_idx = (choice_idx).wrapping_add(1u32);
                }
                first_46 = false;
                if (choice_idx) >= u32::from((*(op)).num_ranked_engines) {
                    break;
                }
                (*core::ptr::addr_of_mut!((*(frame)).ranked_engines)
                    .cast::<u32>()
                    .wrapping_offset((choice_idx) as isize)) =
                    *core::ptr::addr_of!((*(op)).ranked_engines)
                        .cast::<u32>()
                        .wrapping_offset((choice_idx) as isize);
            }
        }
    }
}

pub(super) unsafe extern "C" fn abce_intersect_frame_choices(
    frame: *mut abce_plan_frame_t,
    op: *const abce_engine_op_t,
) {
    unsafe {
        (*(frame)).has_engine_preference =
            (*(frame)).has_engine_preference && (*(op)).has_engine_preference;
        let mut common: [u32; 16] = [0_u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut num_common: u8 = (0i32) as u8;
        {
            let mut frame_choice_idx: u32 = 0_u32;
            let mut first_47 = true;
            loop {
                if !first_47 {
                    frame_choice_idx = (frame_choice_idx).wrapping_add(1u32);
                }
                first_47 = false;
                if !(((frame_choice_idx) < u32::from((*(frame)).num_ranked_engines))
                    && (u32::from(num_common) < (16u32)))
                {
                    break;
                }
                {
                    let candidate: u32 = *core::ptr::addr_of_mut!((*(frame)).ranked_engines)
                        .cast::<u32>()
                        .wrapping_offset((frame_choice_idx) as isize);
                    {
                        let mut op_choice_idx: u32 = 0_u32;
                        let mut first_48 = true;
                        loop {
                            if !first_48 {
                                op_choice_idx = (op_choice_idx).wrapping_add(1u32);
                            }
                            first_48 = false;
                            if (op_choice_idx) >= u32::from((*(op)).num_ranked_engines) {
                                break;
                            }
                            {
                                if (candidate)
                                    == (*core::ptr::addr_of!((*(op)).ranked_engines)
                                        .cast::<u32>()
                                        .wrapping_offset((op_choice_idx) as isize))
                                {
                                    (*core::ptr::addr_of_mut!(common)
                                        .cast::<u32>()
                                        .wrapping_offset(
                                            ({
                                                let old = num_common;
                                                num_common = (num_common).wrapping_add(1u8);
                                                old
                                            }) as isize,
                                        )) = candidate;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        (*(frame)).num_ranked_engines = num_common;
        {
            let mut choice_idx: u32 = 0_u32;
            let mut first_49 = true;
            loop {
                if !first_49 {
                    choice_idx = (choice_idx).wrapping_add(1u32);
                }
                first_49 = false;
                if (choice_idx) >= u32::from(num_common) {
                    break;
                }
                (*core::ptr::addr_of_mut!((*(frame)).ranked_engines)
                    .cast::<u32>()
                    .wrapping_offset((choice_idx) as isize)) = *core::ptr::addr_of_mut!(common)
                    .cast::<u32>()
                    .wrapping_offset((choice_idx) as isize);
            }
        }
    }
}

pub(super) unsafe extern "C" fn abce_build_frames(
    ops: *const abce_engine_op_t,
    num_ops: u32,
    plan: *mut abce_plan_t,
) {
    unsafe {
        let mut seen: u64 = 0_u64;
        let mut engine_to_frame: [u8; 16] =
            [(0i32) as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        (*(plan)).num_frames = 0_u32;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_50 = true;
            loop {
                if !first_50 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_50 = false;
                if (op_idx) >= (num_ops) {
                    break;
                }
                {
                    let engine: u32 = (*(ops).wrapping_offset((op_idx) as isize)).engine;
                    if (seen & ((1u64).wrapping_shl(u64::from(engine) as u32))) == 0 {
                        seen |= (1u64).wrapping_shl(u64::from(engine) as u32);
                        let frame: *mut abce_plan_frame_t = core::ptr::addr_of_mut!(
                            (*core::ptr::addr_of_mut!((*(plan)).frames)
                                .cast::<abce_plan_frame_t>()
                                .wrapping_offset(((*(plan)).num_frames) as isize))
                        );
                        let _ = libc::memset(
                            (frame).cast::<core::ffi::c_void>(),
                            0i32,
                            core::mem::size_of::<abce_plan_frame_t>() as usize,
                        );
                        (*(frame)).engine = engine;
                        abce_initialize_frame_choices(
                            frame,
                            core::ptr::addr_of!((*(ops).wrapping_offset((op_idx) as isize))),
                        );
                        (*(frame)).coordinator = ((*(plan)).num_frames) == (0_u32);
                        (*core::ptr::addr_of_mut!(engine_to_frame)
                            .cast::<u8>()
                            .wrapping_offset((engine) as isize)) = ((*(plan)).num_frames) as u8;
                        (*(plan)).num_frames = ((*(plan)).num_frames).wrapping_add(1u32);
                    } else {
                        abce_intersect_frame_choices(
                            core::ptr::addr_of_mut!(
                                (*core::ptr::addr_of_mut!((*(plan)).frames)
                                    .cast::<abce_plan_frame_t>()
                                    .wrapping_offset(
                                        (*core::ptr::addr_of_mut!(engine_to_frame)
                                            .cast::<u8>()
                                            .wrapping_offset((engine) as isize))
                                            as isize
                                    ))
                            ),
                            core::ptr::addr_of!((*(ops).wrapping_offset((op_idx) as isize))),
                        );
                    }
                    (*core::ptr::addr_of_mut!((*(plan)).frames)
                        .cast::<abce_plan_frame_t>()
                        .wrapping_offset(
                            (*core::ptr::addr_of_mut!(engine_to_frame)
                                .cast::<u8>()
                                .wrapping_offset((engine) as isize))
                                as isize,
                        ))
                    .num_ops = ((*core::ptr::addr_of_mut!((*(plan)).frames)
                        .cast::<abce_plan_frame_t>()
                        .wrapping_offset(
                            (*core::ptr::addr_of_mut!(engine_to_frame)
                                .cast::<u8>()
                                .wrapping_offset((engine) as isize))
                                as isize,
                        ))
                    .num_ops)
                        .wrapping_add(1u32);
                }
            }
        }
    }
}

pub(super) unsafe extern "C" fn abce_make_frame_job(
    ops: *const abce_engine_op_t,
    num_ops: u32,
    metadata: *const abce_copy_metadata_t,
    plan: *const abce_plan_t,
    frame: *const abce_plan_frame_t,
) -> abce_frame_job_t {
    unsafe {
        let mut job: abce_frame_job_t = { core::mem::zeroed() };
        (job).ops = ops;
        (job).num_ops = num_ops;
        (job).metadata = metadata;
        (job).engine = (*(frame)).engine;
        (job).frame_num_ops = (*(frame)).num_ops;
        (job).frame_all_fused = true;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_51 = true;
            loop {
                if !first_51 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_51 = false;
                if (op_idx) >= (num_ops) {
                    break;
                }
                if (((*(ops).wrapping_offset((op_idx) as isize)).engine) == ((*(frame)).engine))
                    && !((*(ops).wrapping_offset((op_idx) as isize)).fused)
                {
                    (job).frame_all_fused = false;
                }
            }
        }
        (job).coordinator = (*(frame)).coordinator;
        (job).multi = (*(plan)).multi;
        (job).start_gate_required = (*(plan)).start_gate_required;
        (job).epilogue_required = (*(plan)).epilogue_required;
        (job).coordination_word = (*(plan)).coordination_signal;
        job
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_map_copy(
    orchestrator: *mut abce_copy_orchestrator_t,
    copies: *const abce_copy_op_t,
    num_copies: u32,
    metadata: *const abce_copy_metadata_t,
    out_plan: *mut abce_plan_t,
) -> abce_status_t {
    unsafe {
        abce_plan_initialize(out_plan);
        let mut status: abce_status_t = abce_copy_orchestrator_validate_request(
            (orchestrator).cast_const(),
            copies,
            num_copies,
            metadata,
            core::ptr::addr_of_mut!((*(out_plan)).failed_op),
        );
        if !(abce_status_is_ok(status)) {
            return status;
        }
        let candidates: u64 =
            abce_copy_orchestrator_candidates((orchestrator).cast_const(), metadata);
        if (candidates) == 0 {
            return ABCE_STATUS_NO_ENGINE;
        }
        (*(out_plan)).execution_descriptor = (*(metadata)).execution_descriptor;
        (*(out_plan)).coordination_readback = ((*(metadata)).out).host_store_readback;
        let mut capacity: u32 = 0_u32;
        {
            let mut copy_idx: u32 = 0_u32;
            let mut first_52 = true;
            loop {
                if !first_52 {
                    copy_idx = (copy_idx).wrapping_add(1u32);
                }
                first_52 = false;
                if (copy_idx) >= (num_copies) {
                    break;
                }
                {
                    let expansion: u32 = if ((*(copies).wrapping_offset((copy_idx) as isize)).kind)
                        == ABCE_OP_KIND_MULTICAST
                    {
                        (*(copies).wrapping_offset((copy_idx) as isize)).num_dsts
                    } else {
                        1_u32
                    };
                    if (expansion) > (((64u32).wrapping_mul(1024u32)).wrapping_sub(capacity)) {
                        (*(out_plan)).failed_op = copy_idx;
                        return ABCE_STATUS_TOO_MANY_OPERATIONS;
                    }
                    capacity = (capacity).wrapping_add(expansion);
                    (*(out_plan)).transfer_kinds |= abce_transfer_kind_bit(abce_classify_transfer(
                        core::ptr::addr_of!(
                            (*(copies).wrapping_offset((copy_idx) as isize)).src_end
                        ),
                        core::ptr::addr_of!(
                            (*(copies).wrapping_offset((copy_idx) as isize)).dst_end
                        ),
                    ));
                }
            }
        }
        let mut inline_ops: [abce_engine_op_t; 16] = { core::mem::zeroed() };
        let mut heap_ops: *mut abce_engine_op_t = core::ptr::null_mut();
        let mut local_ops: *mut abce_engine_op_t =
            core::ptr::addr_of_mut!(inline_ops).cast::<abce_engine_op_t>();
        if (capacity) <= (16u32) {
            let _ = libc::memset(
                core::ptr::addr_of_mut!(inline_ops)
                    .cast::<abce_engine_op_t>()
                    .cast::<core::ffi::c_void>(),
                0i32,
                ((capacity) as usize)
                    .wrapping_mul(core::mem::size_of::<abce_engine_op_t>() as usize),
            );
        } else {
            heap_ops = (libc::calloc(
                (capacity) as usize,
                core::mem::size_of::<abce_engine_op_t>() as usize,
            ))
            .cast::<abce_engine_op_t>();
            if (heap_ops).is_null() {
                return ABCE_STATUS_RESOURCE_EXHAUSTED;
            }
            local_ops = heap_ops;
        }
        let mut num_ops: u32 = 0_u32;
        let force_back_to_back: bool = abce_copy_orchestrator_is_back_to_back_batch(
            (orchestrator).cast_const(),
            copies,
            num_copies,
            metadata,
        );
        let fixed_engine: u32 = if force_back_to_back {
            abce_copy_orchestrator_back_to_back_engine(
                (orchestrator).cast_const(),
                core::ptr::addr_of!((*(copies).wrapping_offset((0i32) as isize))),
                candidates,
            )
        } else {
            16u32
        };
        if force_back_to_back && ((fixed_engine) == (16u32)) {
            libc::free((heap_ops).cast::<core::ffi::c_void>());
            (*(out_plan)).failed_op = 0_u32;
            return ABCE_STATUS_NO_LEGAL_ENGINE;
        }
        if !(abce_copy_orchestrator_decompose(
            (orchestrator).cast_const(),
            copies,
            num_copies,
            metadata,
            candidates,
            force_back_to_back,
            fixed_engine,
            local_ops,
            core::ptr::addr_of_mut!(num_ops),
            capacity,
            core::ptr::addr_of_mut!(status),
            core::ptr::addr_of_mut!((*(out_plan)).failed_op),
        )) {
            libc::free((heap_ops).cast::<core::ffi::c_void>());
            return status;
        }
        if (num_ops) == (0_u32) {
            libc::free((heap_ops).cast::<core::ffi::c_void>());
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        abce_build_frames((local_ops).cast_const(), num_ops, out_plan);
        (*(out_plan)).multi = ((*(out_plan)).num_frames) > (1_u32);
        let use_fused_packets: bool = (*(metadata)).prefer_fused && (*(orchestrator)).is_gfx125plus;
        let mut all_bodies_fused: bool = true;
        {
            let mut op_idx: u32 = 0_u32;
            let mut first_53 = true;
            loop {
                if !first_53 {
                    op_idx = (op_idx).wrapping_add(1u32);
                }
                first_53 = false;
                if (op_idx) >= (num_ops) {
                    break;
                }
                {
                    let kind: abce_op_kind_t =
                        (*(local_ops).wrapping_offset((op_idx) as isize)).kind;
                    if kind == ABCE_OP_KIND_INDIRECT
                        || (use_fused_packets
                            && matches!(
                                kind,
                                ABCE_OP_KIND_LINEAR | ABCE_OP_KIND_MULTICAST | ABCE_OP_KIND_SWAP
                            ))
                    {
                        (*(local_ops).wrapping_offset(op_idx as isize)).fused = true;
                    }
                    all_bodies_fused =
                        all_bodies_fused && (*(local_ops).wrapping_offset((op_idx) as isize)).fused;
                }
            }
        }
        let effective_hdp_flush: bool = ((*(orchestrator)).caps).emit_hdp_flush
            && ((((*(metadata)).coherency).flags).word & 0x1u32) != 0;
        let effective_gcr: bool = ((((((*(metadata)).coherency).flags).word >> 1) & 0x1u32) != 0
            && abce_builder_requires_gcr((*(orchestrator)).builder))
            && !(((*(orchestrator)).caps).driver_manages_gcr);
        let coordinator_has_ordering_work: bool = (((((*(metadata)).num_deps) != (0_u32))
            || abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)))
            || effective_hdp_flush)
            || effective_gcr;
        (*(out_plan)).start_gate_required =
            (*(out_plan)).multi && (!(all_bodies_fused) || coordinator_has_ordering_work);
        (*(out_plan)).epilogue_required = (((!(all_bodies_fused)
            || (*(out_plan)).start_gate_required)
            || abce_timestamp_surface_enabled(core::ptr::addr_of!((*(metadata)).timestamps)))
            || effective_gcr)
            || !(((*(metadata)).out).event_mailbox).is_null();
        (*(out_plan)).fan_in_count = 0_u64;
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_54 = true;
            loop {
                if !first_54 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_54 = false;
                if (frame_idx) >= ((*(out_plan)).num_frames) {
                    break;
                }
                {
                    let job: abce_frame_job_t = abce_make_frame_job(
                        (local_ops).cast_const(),
                        num_ops,
                        metadata,
                        (out_plan).cast_const(),
                        core::ptr::addr_of_mut!(
                            (*core::ptr::addr_of_mut!((*(out_plan)).frames)
                                .cast::<abce_plan_frame_t>()
                                .wrapping_offset((frame_idx) as isize))
                        )
                        .cast_const(),
                    );
                    (*(out_plan)).fan_in_count = ((*(out_plan)).fan_in_count).wrapping_add(
                        abce_frame_composer_signal_count(
                            core::ptr::addr_of_mut!((*(orchestrator)).composer).cast_const(),
                            core::ptr::addr_of!(job),
                        ),
                    );
                }
            }
        }
        let coordinator_drains_fan_in: bool =
            (*(out_plan)).multi && (*(out_plan)).epilogue_required;
        if coordinator_drains_fan_in {
            if (((*(metadata)).out).coordination_scratch).is_null() {
                libc::free((heap_ops).cast::<core::ffi::c_void>());
                return ABCE_STATUS_MISSING_COORDINATION_SCRATCH;
            }
            {};
            if ((*(out_plan)).fan_in_count) > u64::from(4_294_967_295_u32) {
                libc::free((heap_ops).cast::<core::ffi::c_void>());
                return ABCE_STATUS_TOO_MANY_OPERATIONS;
            }
            (*(out_plan)).coordination_signal = ((*(metadata)).out).coordination_scratch;
            (*(out_plan)).coordination_initial_value = (if (*(out_plan)).start_gate_required {
                (1_u64).wrapping_shl((62_u64) as u32)
            } else {
                0_u64
            }) | (*(out_plan)).fan_in_count;
        } else {
            (*(out_plan)).coordination_signal = ((*(metadata)).out).value;
            if (*(out_plan)).multi || (((*(out_plan)).fan_in_count) > (1_u64)) {
                if ((((*(metadata)).out).completion_value) > u64::from(4_294_967_295_u32))
                    || (((*(out_plan)).fan_in_count)
                        > (u64::from(4_294_967_295_u32)
                            .wrapping_sub(((*(metadata)).out).completion_value)))
                {
                    libc::free((heap_ops).cast::<core::ffi::c_void>());
                    return ABCE_STATUS_TOO_MANY_OPERATIONS;
                }
                (*(out_plan)).coordination_initial_value =
                    ((*(out_plan)).fan_in_count).wrapping_add(((*(metadata)).out).completion_value);
            }
        }
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_55 = true;
            loop {
                if !first_55 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_55 = false;
                if (frame_idx) >= ((*(out_plan)).num_frames) {
                    break;
                }
                {
                    let job: abce_frame_job_t = abce_make_frame_job(
                        (local_ops).cast_const(),
                        num_ops,
                        metadata,
                        (out_plan).cast_const(),
                        core::ptr::addr_of_mut!(
                            (*core::ptr::addr_of_mut!((*(out_plan)).frames)
                                .cast::<abce_plan_frame_t>()
                                .wrapping_offset((frame_idx) as isize))
                        )
                        .cast_const(),
                    );
                    let frame_bytes: usize = abce_frame_composer_frame_bytes(
                        core::ptr::addr_of_mut!((*(orchestrator)).composer).cast_const(),
                        core::ptr::addr_of!(job),
                    );
                    if (frame_bytes) > (4_294_967_295_usize) {
                        libc::free((heap_ops).cast::<core::ffi::c_void>());
                        return ABCE_STATUS_TOO_MANY_OPERATIONS;
                    }
                    (*core::ptr::addr_of_mut!((*(out_plan)).frames)
                        .cast::<abce_plan_frame_t>()
                        .wrapping_offset((frame_idx) as isize))
                    .bytes = (frame_bytes) as u32;
                }
            }
        }
        let mut total_bytes: usize = 0_usize;
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_56 = true;
            loop {
                if !first_56 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_56 = false;
                if (frame_idx) >= ((*(out_plan)).num_frames) {
                    break;
                }
                total_bytes = (total_bytes).wrapping_add(
                    ((*core::ptr::addr_of_mut!((*(out_plan)).frames)
                        .cast::<abce_plan_frame_t>()
                        .wrapping_offset((frame_idx) as isize))
                    .bytes) as usize,
                );
            }
        }
        let scratch: *mut core::ffi::c_char = abce_acquire_packet_scratch(total_bytes);
        if (scratch).is_null() && ((total_bytes) > (0_usize)) {
            libc::free((heap_ops).cast::<core::ffi::c_void>());
            return ABCE_STATUS_RESOURCE_EXHAUSTED;
        }
        let mut scratch_offset: usize = 0_usize;
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_57 = true;
            loop {
                if !first_57 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_57 = false;
                if (frame_idx) >= ((*(out_plan)).num_frames) {
                    break;
                }
                {
                    let frame: *mut abce_plan_frame_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!((*(out_plan)).frames)
                            .cast::<abce_plan_frame_t>()
                            .wrapping_offset((frame_idx) as isize))
                    );
                    let frame_buffer: *mut core::ffi::c_char =
                        (scratch).wrapping_add(scratch_offset);
                    let _ = libc::memset(
                        (frame_buffer).cast::<core::ffi::c_void>(),
                        0i32,
                        ((*(frame)).bytes) as usize,
                    );
                    let job: abce_frame_job_t = abce_make_frame_job(
                        (local_ops).cast_const(),
                        num_ops,
                        metadata,
                        (out_plan).cast_const(),
                        (frame).cast_const(),
                    );
                    let emitted: usize = abce_frame_composer_emit_frame(
                        core::ptr::addr_of_mut!((*(orchestrator)).composer).cast_const(),
                        frame_buffer,
                        core::ptr::addr_of!(job),
                    );
                    let _ = emitted;
                    {};
                    (*(frame)).packets = (frame_buffer).cast_const();
                    scratch_offset = (scratch_offset).wrapping_add(((*(frame)).bytes) as usize);
                }
            }
        }
        libc::free((heap_ops).cast::<core::ffi::c_void>());
        (*(out_plan)).valid = true;
        ABCE_STATUS_OK
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_submit(
    orchestrator: *mut abce_copy_orchestrator_t,
    plan: *mut abce_plan_t,
) -> abce_status_t {
    unsafe {
        if !((*(plan)).valid) || (*(plan)).submitted {
            return ABCE_STATUS_INVALID_PLAN;
        }
        let mut pending: [abce_submit_pending_t; 16] = { core::mem::zeroed() };
        let _ = libc::memset(
            core::ptr::addr_of_mut!(pending)
                .cast::<abce_submit_pending_t>()
                .cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<[abce_submit_pending_t; 16]>() as usize,
        );
        let mut acquired_frames: [u8; 16] =
            [(0i32) as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut num_pending: u32 = 0_u32;
        if (((*(plan)).num_frames) == (0_u32)) || (((*(plan)).num_frames) > (16u32)) {
            return ABCE_STATUS_INVALID_PLAN;
        }
        let mut selected_ring_mask: u64 = 0_u64;
        {
            let mut frame_idx: u32 = 0_u32;
            let mut first_58 = true;
            loop {
                if !first_58 {
                    frame_idx = (frame_idx).wrapping_add(1u32);
                }
                first_58 = false;
                if (frame_idx) >= ((*(plan)).num_frames) {
                    break;
                }
                {
                    let frame: *const abce_plan_frame_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!((*(plan)).frames)
                            .cast::<abce_plan_frame_t>()
                            .wrapping_offset((frame_idx) as isize))
                    )
                    .cast_const();
                    if (((((*(frame)).engine) >= (16u32)) || ((*(frame)).packets).is_null())
                        || (((*(frame)).bytes) == (0_u32)))
                        || (selected_ring_mask
                            & ((1_u64).wrapping_shl(u64::from((*(frame)).engine) as u32)))
                            != 0
                    {
                        return ABCE_STATUS_INVALID_PLAN;
                    }
                    selected_ring_mask |= (1_u64).wrapping_shl(u64::from((*(frame)).engine) as u32);
                    let ring: *mut abce_ring_t =
                        *core::ptr::addr_of_mut!((*(orchestrator)).engines)
                            .cast::<*mut abce_ring_t>()
                            .wrapping_offset(((*(frame)).engine) as isize);
                    if (ring).is_null() {
                        return ABCE_STATUS_RING_UNAVAILABLE;
                    }
                    let entry: *mut abce_submit_pending_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!(pending)
                            .cast::<abce_submit_pending_t>()
                            .wrapping_offset((frame_idx) as isize))
                    );
                    (*(entry)).ring = ring;
                    if !(abce_status_is_ok(abce_ring_padded_size(
                        ((*(entry)).ring).cast_const(),
                        (*(frame)).bytes,
                        core::ptr::addr_of_mut!((*(entry)).reserve_bytes),
                    ))) {
                        return ABCE_STATUS_OUT_OF_RANGE;
                    }
                }
            }
        }
        {
            let mut engine_idx: u32 = 0_u32;
            let mut first_59 = true;
            loop {
                if !first_59 {
                    engine_idx = (engine_idx).wrapping_add(1u32);
                }
                first_59 = false;
                if (engine_idx) >= (16u32) {
                    break;
                }
                {
                    {
                        let mut frame_idx: u32 = 0_u32;
                        let mut first_60 = true;
                        loop {
                            if !first_60 {
                                frame_idx = (frame_idx).wrapping_add(1u32);
                            }
                            first_60 = false;
                            if (frame_idx) >= ((*(plan)).num_frames) {
                                break;
                            }
                            {
                                if ((*core::ptr::addr_of_mut!((*(plan)).frames)
                                    .cast::<abce_plan_frame_t>()
                                    .wrapping_offset((frame_idx) as isize))
                                .engine)
                                    != (engine_idx)
                                {
                                    continue;
                                }
                                let entry: *mut abce_submit_pending_t = core::ptr::addr_of_mut!(
                                    (*core::ptr::addr_of_mut!(pending)
                                        .cast::<abce_submit_pending_t>()
                                        .wrapping_offset((frame_idx) as isize))
                                );
                                let acquire_status: abce_status_t = abce_ring_acquire(
                                    (*(entry)).ring,
                                    (*(entry)).reserve_bytes,
                                    core::ptr::addr_of_mut!((*(entry)).reservation),
                                    core::ptr::addr_of_mut!((*(entry)).buffer),
                                );
                                if !(abce_status_is_ok(acquire_status)) {
                                    {
                                        let mut acquired_idx: u32 = 0_u32;
                                        let mut first_61 = true;
                                        loop {
                                            if !first_61 {
                                                acquired_idx = (acquired_idx).wrapping_add(1u32);
                                            }
                                            first_61 = false;
                                            if (acquired_idx) >= (num_pending) {
                                                break;
                                            }
                                            {
                                                let acquired_entry: *mut abce_submit_pending_t = core::ptr::addr_of_mut!(
                                                    (*core::ptr::addr_of_mut!(pending)
                                                        .cast::<abce_submit_pending_t>()
                                                        .wrapping_offset(
                                                            (*core::ptr::addr_of_mut!(
                                                                acquired_frames
                                                            )
                                                            .cast::<u8>()
                                                            .wrapping_offset(
                                                                (acquired_idx) as isize
                                                            ))
                                                                as isize
                                                        ))
                                                );
                                                let _ = abce_ring_cancel(
                                                    (*(acquired_entry)).ring,
                                                    core::ptr::addr_of_mut!(
                                                        (*(acquired_entry)).reservation
                                                    )
                                                    .cast_const(),
                                                );
                                            }
                                        }
                                    }
                                    return ABCE_STATUS_RING_UNAVAILABLE;
                                }
                                (*core::ptr::addr_of_mut!(acquired_frames)
                                    .cast::<u8>()
                                    .wrapping_offset(
                                        ({
                                            let old = num_pending;
                                            num_pending = (num_pending).wrapping_add(1u32);
                                            old
                                        }) as isize,
                                    )) = (frame_idx) as u8;
                            }
                        }
                    }
                }
            }
        }
        let mut stored_engine_inputs: bool = false;
        if (*(plan)).multi || (((*(plan)).fan_in_count) > (1_u64)) {
            abce_host_store_release_u64(
                ((*(plan)).coordination_signal).cast::<u64>(),
                (*(plan)).coordination_initial_value,
            );
            stored_engine_inputs = true;
        }
        if !((*(plan)).execution_descriptor).is_null() {
            abce_host_store_release_u32(
                ((*(plan)).execution_descriptor).cast::<u32>(),
                abce_copy_orchestrator_describe_execution(
                    (orchestrator).cast_const(),
                    (plan).cast_const(),
                ),
            );
            stored_engine_inputs = true;
        }
        let mut copied_with_memcpy: bool = false;
        let mut ring_readback_byte: *const core::ffi::c_void = core::ptr::null();
        {
            let mut pending_idx: u32 = 0_u32;
            let mut first_62 = true;
            loop {
                if !first_62 {
                    pending_idx = (pending_idx).wrapping_add(1u32);
                }
                first_62 = false;
                if (pending_idx) >= (num_pending) {
                    break;
                }
                {
                    let frame: *const abce_plan_frame_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!((*(plan)).frames)
                            .cast::<abce_plan_frame_t>()
                            .wrapping_offset((pending_idx) as isize))
                    )
                    .cast_const();
                    let entry: *mut abce_submit_pending_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!(pending)
                            .cast::<abce_submit_pending_t>()
                            .wrapping_offset((pending_idx) as isize))
                    );
                    abce_ring_fill(
                        ((*(entry)).ring).cast_const(),
                        (*(entry)).buffer,
                        ((*(frame)).packets).cast::<core::ffi::c_void>(),
                        ((*(frame)).bytes) as usize,
                        ((*(entry)).reserve_bytes) as usize,
                    );
                    copied_with_memcpy =
                        copied_with_memcpy || (((*((*(entry)).ring)).config).write).is_none();
                    if ((*((*(entry)).ring)).config).readback_before_doorbell {
                        ring_readback_byte = (abce_ring_region_last_byte(
                            ((*(entry)).buffer).cast_const(),
                            ((*(entry)).reserve_bytes) as usize,
                        ))
                        .cast::<core::ffi::c_void>();
                    }
                }
            }
        }
        if copied_with_memcpy || stored_engine_inputs {
            abce_ring_host_store_fence();
        }
        if !(ring_readback_byte).is_null() {
            abce_ring_read_back(ring_readback_byte);
        }
        if (*(plan)).coordination_readback
            && ((*(plan)).multi || (((*(plan)).fan_in_count) > (1_u64)))
        {
            abce_ring_read_back(((*(plan)).coordination_signal).cast_const());
        }
        {
            let mut acquired_idx: u32 = 0_u32;
            let mut first_63 = true;
            loop {
                if !first_63 {
                    acquired_idx = (acquired_idx).wrapping_add(1u32);
                }
                first_63 = false;
                if (acquired_idx) >= (num_pending) {
                    break;
                }
                {
                    let acquired_entry: *mut abce_submit_pending_t = core::ptr::addr_of_mut!(
                        (*core::ptr::addr_of_mut!(pending)
                            .cast::<abce_submit_pending_t>()
                            .wrapping_offset(
                                (*core::ptr::addr_of_mut!(acquired_frames)
                                    .cast::<u8>()
                                    .wrapping_offset((acquired_idx) as isize))
                                    as isize
                            ))
                    );
                    if !(abce_status_is_ok(abce_ring_release_ordered(
                        (*(acquired_entry)).ring,
                        core::ptr::addr_of_mut!((*(acquired_entry)).reservation).cast_const(),
                    ))) {
                        return ABCE_STATUS_RING_UNAVAILABLE;
                    }
                }
            }
        }
        (*(plan)).submitted = true;
        ABCE_STATUS_OK
    }
}

pub unsafe extern "C" fn abce_copy_orchestrator_dispatch(
    orchestrator: *mut abce_copy_orchestrator_t,
    copies: *const abce_copy_op_t,
    num_copies: u32,
    metadata: *const abce_copy_metadata_t,
) -> abce_status_t {
    unsafe {
        let mut plan: abce_plan_t = { core::mem::zeroed() };
        let status: abce_status_t = abce_copy_orchestrator_map_copy(
            orchestrator,
            copies,
            num_copies,
            metadata,
            core::ptr::addr_of_mut!(plan),
        );
        if !(abce_status_is_ok(status)) {
            return status;
        }
        abce_copy_orchestrator_submit(orchestrator, core::ptr::addr_of_mut!(plan))
    }
}
