// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE policy mechanisms.

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
    ABCE_ENDPOINT_KIND_DEVICE, ABCE_ENDPOINT_KIND_HOST, ABCE_EXECUTION_ENGINE_KIND_NONE,
    ABCE_OP_KIND_LINEAR, ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE,
    ABCE_POLICY_DISPOSITION_NO_PREFERENCE, ABCE_POLICY_DISPOSITION_RANKED,
    ABCE_TRANSFER_KIND_DEVICE_TO_HOST, ABCE_TRANSFER_KIND_HOST_TO_DEVICE,
    ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE, ABCE_TRANSFER_KIND_PEER_TO_PEER,
    ABCE_TRANSFER_KIND_UNKNOWN, abce_copy_endpoint_t, abce_copy_op_t, abce_endpoint_kind_t,
    abce_engine_affinity_t, abce_execution_engine_kind_t, abce_heatmap_profile_t,
    abce_isa_version_t, abce_policy_disposition_t, abce_policy_result_t, abce_sdma_engine_policy_t,
    abce_sdma_engine_policy_vtable_t, abce_topology_data_t, abce_topology_load_result_t,
    abce_transfer_kind_t,
};
use super::{
    abce_count_trailing_zeros_u64, abce_gfx94x_sdma_affinity_map, abce_sdma_policy_gfx90a_vtable,
    abce_sdma_policy_gfx94x_vtable, abce_topology_data_hive_id, abce_topology_data_initialize,
    abce_topology_data_xgmi_physical_id, abce_topology_load_from_kfd,
};

#[must_use]
pub unsafe extern "C" fn abce_copy_endpoint_make(
    kind: abce_endpoint_kind_t,
    device_id: u32,
) -> abce_copy_endpoint_t {
    {
        let mut endpoint: abce_copy_endpoint_t = unsafe { core::mem::zeroed() };
        (endpoint).kind = kind;
        (endpoint).device_id = device_id;
        endpoint
    }
}

#[must_use]
pub unsafe extern "C" fn abce_classify_transfer(
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
) -> abce_transfer_kind_t {
    unsafe {
        if (((*(src)).kind) == ABCE_ENDPOINT_KIND_DEVICE)
            && (((*(dst)).kind) == ABCE_ENDPOINT_KIND_DEVICE)
        {
            let same_device: bool = (((*(src)).device_id) != (4_294_967_295_u32))
                && (((*(src)).device_id) == ((*(dst)).device_id));
            return (if same_device {
                ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE as i32
            } else {
                ABCE_TRANSFER_KIND_PEER_TO_PEER as i32
            }) as abce_transfer_kind_t;
        }
        if (((*(src)).kind) == ABCE_ENDPOINT_KIND_HOST)
            && (((*(dst)).kind) == ABCE_ENDPOINT_KIND_DEVICE)
        {
            return ABCE_TRANSFER_KIND_HOST_TO_DEVICE;
        }
        if (((*(src)).kind) == ABCE_ENDPOINT_KIND_DEVICE)
            && (((*(dst)).kind) == ABCE_ENDPOINT_KIND_HOST)
        {
            return ABCE_TRANSFER_KIND_DEVICE_TO_HOST;
        }
        ABCE_TRANSFER_KIND_UNKNOWN
    }
}

#[must_use]
pub unsafe extern "C" fn abce_transfer_kind_bit(kind: abce_transfer_kind_t) -> u32 {
    (1_u32).wrapping_shl(kind)
}

pub unsafe extern "C" fn abce_engine_affinity_initialize(
    out_affinity: *mut abce_engine_affinity_t,
) {
    unsafe {
        (*(out_affinity)).hw_engine_id = 4_294_967_295_u32;
    }
}

pub(super) unsafe extern "C" fn abce_first_engine(mask: u64) -> u32 {
    unsafe {
        if (mask) != 0 {
            (abce_count_trailing_zeros_u64(mask)) as u32
        } else {
            16u32
        }
    }
}

pub unsafe extern "C" fn abce_copy_op_initialize(out_copy: *mut abce_copy_op_t) {
    unsafe {
        let _ = libc::memset(
            (out_copy).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_copy_op_t>() as usize,
        );
        (*(out_copy)).kind = ABCE_OP_KIND_LINEAR;
        (*(out_copy)).src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, 4_294_967_295_u32);
        (*(out_copy)).dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, 4_294_967_295_u32);
    }
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_encode(
    engine_kind: abce_execution_engine_kind_t,
    instance_mask: u32,
    transfer_kind: abce_transfer_kind_t,
    mixed_kinds: bool,
) -> u32 {
    {
        ((((instance_mask & 65_535_u32).wrapping_shl(0u32))
            | (((transfer_kind) & 7u32).wrapping_shl(16u32)))
            | (if mixed_kinds {
                (1_u32).wrapping_shl(19_u32)
            } else {
                0_u32
            }))
            | (((engine_kind) & 7u32).wrapping_shl(20u32))
    }
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_engine_kind(
    word: u32,
) -> abce_execution_engine_kind_t {
    (((word).wrapping_shr(20u32)) & 7u32) as abce_execution_engine_kind_t
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_valid(word: u32) -> bool {
    unsafe {
        ((abce_execution_descriptor_engine_kind(word)) as u32) != ABCE_EXECUTION_ENGINE_KIND_NONE
    }
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_instance_mask(word: u32) -> u32 {
    ((word).wrapping_shr(0u32)) & 65_535_u32
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_transfer_kind(
    word: u32,
) -> abce_transfer_kind_t {
    (((word).wrapping_shr(16u32)) & 7u32) as abce_transfer_kind_t
}

#[must_use]
pub unsafe extern "C" fn abce_execution_descriptor_mixed_kinds(word: u32) -> bool {
    (word & ((1_u32).wrapping_shl(19_u32))) != (0_u32)
}

#[must_use]
pub unsafe extern "C" fn abce_select_heatmap(
    minor: u8,
    total_sdma: u8,
) -> *const abce_heatmap_profile_t {
    unsafe {
        static kProfiles: [abce_heatmap_profile_t; 3] = [
            abce_heatmap_profile_t {
                minor: (4i32) as u8,
                total_sdma: (16i32) as u8,
                d2h: [
                    (2i32) as u8,
                    (4i32) as u8,
                    (6i32) as u8,
                    (5i32) as u8,
                    (1i32) as u8,
                    (7i32) as u8,
                    (0i32) as u8,
                    (3i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                d2h_count: (8i32) as u8,
                h2d: [
                    (2i32) as u8,
                    (4i32) as u8,
                    (6i32) as u8,
                    (5i32) as u8,
                    (1i32) as u8,
                    (7i32) as u8,
                    (0i32) as u8,
                    (3i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                h2d_count: (8i32) as u8,
            },
            abce_heatmap_profile_t {
                minor: (4i32) as u8,
                total_sdma: (8i32) as u8,
                d2h: [
                    (0i32) as u8,
                    (3i32) as u8,
                    (4i32) as u8,
                    (5i32) as u8,
                    (2i32) as u8,
                    (7i32) as u8,
                    (6i32) as u8,
                    (1i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                d2h_count: (8i32) as u8,
                h2d: [
                    (0i32) as u8,
                    (3i32) as u8,
                    (4i32) as u8,
                    (5i32) as u8,
                    (2i32) as u8,
                    (7i32) as u8,
                    (6i32) as u8,
                    (1i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                h2d_count: (8i32) as u8,
            },
            abce_heatmap_profile_t {
                minor: (5i32) as u8,
                total_sdma: (16i32) as u8,
                d2h: [
                    (1i32) as u8,
                    (0i32) as u8,
                    (2i32) as u8,
                    (3i32) as u8,
                    (4i32) as u8,
                    (5i32) as u8,
                    (6i32) as u8,
                    (7i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                d2h_count: (8i32) as u8,
                h2d: [
                    (1i32) as u8,
                    (0i32) as u8,
                    (2i32) as u8,
                    (3i32) as u8,
                    (4i32) as u8,
                    (5i32) as u8,
                    (6i32) as u8,
                    (7i32) as u8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                h2d_count: (8i32) as u8,
            },
        ];
        {
            let mut profile_idx: usize = 0_usize;
            let mut first_21 = true;
            loop {
                if !first_21 {
                    profile_idx = (profile_idx).wrapping_add(1usize);
                }
                first_21 = false;
                if (profile_idx)
                    >= (core::mem::size_of::<[abce_heatmap_profile_t; 3]>() as usize
                        / core::mem::size_of::<abce_heatmap_profile_t>() as usize)
                {
                    break;
                }
                {
                    if (i32::from(
                        (*core::ptr::addr_of!(kProfiles)
                            .cast::<abce_heatmap_profile_t>()
                            .wrapping_add(profile_idx))
                        .minor,
                    ) == i32::from(minor))
                        && (i32::from(
                            (*core::ptr::addr_of!(kProfiles)
                                .cast::<abce_heatmap_profile_t>()
                                .wrapping_add(profile_idx))
                            .total_sdma,
                        ) == i32::from(total_sdma))
                    {
                        return core::ptr::addr_of!(
                            (*core::ptr::addr_of!(kProfiles)
                                .cast::<abce_heatmap_profile_t>()
                                .wrapping_add(profile_idx))
                        );
                    }
                }
            }
        }
        core::ptr::null()
    }
}

pub unsafe extern "C" fn abce_sdma_policy_emit_all(
    policy: *const abce_sdma_engine_policy_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let count: u32 = if ((*(policy)).num_mapped) < (max) {
            (*(policy)).num_mapped
        } else {
            max
        };
        {
            let mut idx: u32 = 0_u32;
            let mut first_22 = true;
            loop {
                if !first_22 {
                    idx = (idx).wrapping_add(1u32);
                }
                first_22 = false;
                if (idx) >= (count) {
                    break;
                }
                (*(out_hwids).wrapping_offset((idx) as isize)) =
                    *core::ptr::addr_of!((*(policy)).mapped_hwids)
                        .cast::<u8>()
                        .wrapping_offset((idx) as isize);
            }
        }
        count
    }
}

pub unsafe extern "C" fn abce_sdma_policy_emit_xgmi(
    policy: *const abce_sdma_engine_policy_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let mut count: u32 = 0_u32;
        if ((*(policy)).num_non_xgmi) != (0_u32) {
            {
                let mut idx: u32 = 0_u32;
                let mut first_23 = true;
                loop {
                    if !first_23 {
                        idx = (idx).wrapping_add(1u32);
                    }
                    first_23 = false;
                    if !(((idx) < ((*(policy)).num_mapped)) && ((count) < (max))) {
                        break;
                    }
                    if u32::from(
                        *core::ptr::addr_of!((*(policy)).mapped_hwids)
                            .cast::<u8>()
                            .wrapping_offset((idx) as isize),
                    ) >= ((*(policy)).num_non_xgmi)
                    {
                        (*(out_hwids).wrapping_offset(
                            ({
                                let old = count;
                                count = (count).wrapping_add(1u32);
                                old
                            }) as isize,
                        )) = *core::ptr::addr_of!((*(policy)).mapped_hwids)
                            .cast::<u8>()
                            .wrapping_offset((idx) as isize);
                    }
                }
            }
        }
        if (count) != 0 {
            count
        } else {
            abce_sdma_policy_emit_all(policy, out_hwids, max)
        }
    }
}

#[must_use]
pub unsafe extern "C" fn abce_sdma_policy_xgmi_physical_id(
    policy: *const abce_sdma_engine_policy_t,
    device_id: u32,
) -> i32 {
    unsafe {
        abce_topology_data_xgmi_physical_id(core::ptr::addr_of!((*(policy)).topology), device_id)
    }
}

#[must_use]
pub unsafe extern "C" fn abce_sdma_policy_hive_id(
    policy: *const abce_sdma_engine_policy_t,
    device_id: u32,
) -> u64 {
    unsafe { abce_topology_data_hive_id(core::ptr::addr_of!((*(policy)).topology), device_id) }
}

pub unsafe extern "C" fn abce_sdma_policy_copy_hwids(
    src: *const u8,
    src_count: u32,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let count: u32 = if (src_count) < (max) { src_count } else { max };
        {
            let mut idx: u32 = 0_u32;
            let mut first_24 = true;
            loop {
                if !first_24 {
                    idx = (idx).wrapping_add(1u32);
                }
                first_24 = false;
                if (idx) >= (count) {
                    break;
                }
                (*(out_hwids).wrapping_offset((idx) as isize)) =
                    *(src).wrapping_offset((idx) as isize);
            }
        }
        count
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn abce_sdma_policy_base_rank_all(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = src;
        let _ = dst;
        abce_sdma_policy_emit_all(policy, out_hwids, max)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn abce_sdma_policy_base_rank_p2p(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = src;
        let _ = dst;
        abce_sdma_policy_emit_xgmi(policy, out_hwids, max)
    }
}

pub(super) unsafe extern "C" fn abce_gfx94x_rank_h2d(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = src;
        let _ = dst;
        let count: u32 = if ((*(policy)).heatmap).is_null() {
            0_u32
        } else {
            abce_sdma_policy_copy_hwids(
                core::ptr::addr_of!((*((*(policy)).heatmap)).h2d).cast::<u8>(),
                u32::from((*((*(policy)).heatmap)).h2d_count),
                out_hwids,
                max,
            )
        };
        if (count) < (max) {
            (count).wrapping_add(abce_sdma_policy_emit_all(
                policy,
                (out_hwids).wrapping_offset((count) as isize),
                (max).wrapping_sub(count),
            ))
        } else {
            count
        }
    }
}

pub(super) unsafe extern "C" fn abce_gfx94x_rank_d2h(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = src;
        let _ = dst;
        let count: u32 = if ((*(policy)).heatmap).is_null() {
            0_u32
        } else {
            abce_sdma_policy_copy_hwids(
                core::ptr::addr_of!((*((*(policy)).heatmap)).d2h).cast::<u8>(),
                u32::from((*((*(policy)).heatmap)).d2h_count),
                out_hwids,
                max,
            )
        };
        if (count) < (max) {
            (count).wrapping_add(abce_sdma_policy_emit_all(
                policy,
                (out_hwids).wrapping_offset((count) as isize),
                (max).wrapping_sub(count),
            ))
        } else {
            count
        }
    }
}

pub(super) unsafe extern "C" fn abce_gfx94x_rank_p2p(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let src_hive: u64 = abce_sdma_policy_hive_id(policy, (*(src)).device_id);
        let dst_hive: u64 = abce_sdma_policy_hive_id(policy, (*(dst)).device_id);
        if ((src_hive) != 0 && (dst_hive) != 0) && ((src_hive) != (dst_hive)) {
            return 0_u32;
        }
        let mut count: u32 = 0_u32;
        let src_affinity: i32 = abce_sdma_policy_xgmi_physical_id(policy, (*(src)).device_id);
        let dst_affinity: i32 = abce_sdma_policy_xgmi_physical_id(policy, (*(dst)).device_id);
        if (((0i32..8i32).contains(&(src_affinity)) && ((dst_affinity) >= (0i32)))
            && ((dst_affinity) < (8i32)))
            && ((count) < (max))
        {
            (*(out_hwids).wrapping_offset(
                ({
                    let old = count;
                    count = (count).wrapping_add(1u32);
                    old
                }) as isize,
            )) = ((*core::ptr::addr_of!(
                (*core::ptr::addr_of!(abce_gfx94x_sdma_affinity_map)
                    .cast::<[i32; 8]>()
                    .wrapping_offset((src_affinity) as isize))
            )
            .cast::<i32>()
            .wrapping_offset((dst_affinity) as isize))
            .wrapping_mul(2i32)) as u8;
        }
        if (count) < (max) {
            count = (count).wrapping_add(abce_sdma_policy_emit_xgmi(
                policy,
                (out_hwids).wrapping_offset((count) as isize),
                (max).wrapping_sub(count),
            ));
        }
        count
    }
}

pub(super) unsafe extern "C" fn abce_gfx90a_rank_h2d(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = policy;
        let _ = src;
        let _ = dst;
        if (max) == (0_u32) {
            return 0_u32;
        }
        (*(out_hwids).wrapping_offset((0i32) as isize)) = (0i32) as u8;
        1_u32
    }
}

pub(super) unsafe extern "C" fn abce_gfx90a_rank_d2h(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let _ = policy;
        let _ = src;
        let _ = dst;
        if (max) == (0_u32) {
            return 0_u32;
        }
        (*(out_hwids).wrapping_offset((0i32) as isize)) = (1i32) as u8;
        1_u32
    }
}

pub(super) unsafe extern "C" fn abce_gfx90a_rank_p2p(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    out_hwids: *mut u8,
    max: u32,
) -> u32 {
    unsafe {
        let src_hive: u64 = abce_sdma_policy_hive_id(policy, (*(src)).device_id);
        let dst_hive: u64 = abce_sdma_policy_hive_id(policy, (*(dst)).device_id);
        if ((src_hive) != 0 && (dst_hive) != 0) && ((src_hive) != (dst_hive)) {
            return 0_u32;
        }
        abce_sdma_policy_emit_xgmi(policy, out_hwids, max)
    }
}

pub unsafe extern "C" fn abce_sdma_policy_initialize(
    vtable: *const abce_sdma_engine_policy_vtable_t,
    out_policy: *mut abce_sdma_engine_policy_t,
) {
    unsafe {
        let _ = libc::memset(
            (out_policy).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_sdma_engine_policy_t>() as usize,
        );
        (*(out_policy)).vtable = vtable;
        {
            let mut idx: u32 = 0_u32;
            let mut first_25 = true;
            loop {
                if !first_25 {
                    idx = (idx).wrapping_add(1u32);
                }
                first_25 = false;
                if (idx) >= (16u32) {
                    break;
                }
                (*core::ptr::addr_of_mut!((*(out_policy)).hwid_to_reg)
                    .cast::<i32>()
                    .wrapping_offset((idx) as isize)) = (1i32).wrapping_neg();
            }
        }
        abce_topology_data_initialize(core::ptr::addr_of_mut!((*(out_policy)).topology));
    }
}

pub unsafe extern "C" fn abce_sdma_policy_initialize_for_isa(
    isa: abce_isa_version_t,
    out_policy: *mut abce_sdma_engine_policy_t,
) -> bool {
    unsafe {
        if ((isa).major) == (9_u32) {
            if (((isa).minor) == (0_u32)) && (((isa).stepping) == (10_u32)) {
                abce_sdma_policy_initialize(
                    core::ptr::addr_of!(abce_sdma_policy_gfx90a_vtable),
                    out_policy,
                );
                return true;
            }
            if (((isa).minor) == (4_u32)) || (((isa).minor) == (5_u32)) {
                abce_sdma_policy_initialize(
                    core::ptr::addr_of!(abce_sdma_policy_gfx94x_vtable),
                    out_policy,
                );
                return true;
            }
        }
        false
    }
}

pub unsafe extern "C" fn abce_sdma_policy_map_engine(
    policy: *mut abce_sdma_engine_policy_t,
    hw_engine_id: u32,
    reg_index: u32,
) {
    unsafe {
        if (((hw_engine_id) >= (16u32)) || ((reg_index) >= (16u32)))
            || ((*core::ptr::addr_of_mut!((*(policy)).hwid_to_reg)
                .cast::<i32>()
                .wrapping_offset((hw_engine_id) as isize))
                >= (0i32))
        {
            return;
        }
        (*core::ptr::addr_of_mut!((*(policy)).hwid_to_reg)
            .cast::<i32>()
            .wrapping_offset((hw_engine_id) as isize)) = (reg_index) as i32;
        if ((*(policy)).num_mapped) < (16u32) {
            (*core::ptr::addr_of_mut!((*(policy)).mapped_hwids)
                .cast::<u8>()
                .wrapping_offset(
                    ({
                        let old = (*(policy)).num_mapped;
                        (*(policy)).num_mapped = ((*(policy)).num_mapped).wrapping_add(1u32);
                        old
                    }) as isize,
                )) = (hw_engine_id) as u8;
        }
    }
}

pub unsafe extern "C" fn abce_sdma_policy_set_engine_split(
    policy: *mut abce_sdma_engine_policy_t,
    num_non_xgmi: u32,
) {
    unsafe {
        (*(policy)).num_non_xgmi = if (num_non_xgmi) < (16u32) {
            num_non_xgmi
        } else {
            0_u32
        };
    }
}

#[must_use]
pub unsafe extern "C" fn abce_sdma_policy_registered_engine(
    policy: *const abce_sdma_engine_policy_t,
    hw_engine_id: u32,
) -> u32 {
    unsafe {
        if ((hw_engine_id) < (16u32))
            && ((*core::ptr::addr_of!((*(policy)).hwid_to_reg)
                .cast::<i32>()
                .wrapping_offset((hw_engine_id) as isize))
                >= (0i32))
        {
            (*core::ptr::addr_of!((*(policy)).hwid_to_reg)
                .cast::<i32>()
                .wrapping_offset((hw_engine_id) as isize)) as u32
        } else {
            16u32
        }
    }
}

pub unsafe extern "C" fn abce_sdma_policy_select_heatmap_profile(
    policy: *mut abce_sdma_engine_policy_t,
    minor: u8,
    total_sdma: u8,
) {
    unsafe {
        (*(policy)).heatmap = abce_select_heatmap(minor, total_sdma);
    }
}

pub unsafe extern "C" fn abce_sdma_policy_set_topology(
    policy: *mut abce_sdma_engine_policy_t,
    topology: *const abce_topology_data_t,
) {
    unsafe {
        (*(policy)).topology = *(topology);
    }
}

pub unsafe extern "C" fn abce_sdma_policy_load_topology_from_kfd(
    policy: *mut abce_sdma_engine_policy_t,
    base_path: *const core::ffi::c_char,
) -> bool {
    unsafe {
        let mut replacement: abce_topology_data_t = { core::mem::zeroed() };
        let mut result: abce_topology_load_result_t = { core::mem::zeroed() };
        if !(abce_topology_load_from_kfd(
            base_path,
            core::ptr::addr_of_mut!(replacement),
            core::ptr::addr_of_mut!(result),
        )) {
            return false;
        }
        (*(policy)).topology = replacement;
        abce_sdma_policy_select_heatmap_profile(policy, (result).gfx_minor, (result).total_sdma);
        abce_sdma_policy_set_engine_split(policy, u32::from((result).num_non_xgmi_sdma));
        true
    }
}

pub unsafe extern "C" fn abce_sdma_policy_rank_engines(
    policy: *const abce_sdma_engine_policy_t,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    candidate_mask: u64,
    out_engines: *mut u32,
    max: u32,
) -> abce_policy_result_t {
    unsafe {
        let mut hwids: [u8; 16] = { core::mem::zeroed() };
        let mut num_hwids: u32 = 0_u32;
        let mut result: abce_policy_result_t = { core::mem::zeroed() };
        (result).disposition = ABCE_POLICY_DISPOSITION_NO_PREFERENCE;
        (result).count = 0_u32;
        'switch: {
            match (abce_classify_transfer(src, dst)) as u32 {
                value if value == ABCE_TRANSFER_KIND_LOCAL_DEVICE_TO_DEVICE => {
                    num_hwids = (match (*((*(policy)).vtable)).rank_local_d2d {
                        Some(callback) => callback,
                        None => std::process::abort(),
                    })(
                        policy,
                        src,
                        dst,
                        core::ptr::addr_of_mut!(hwids).cast::<u8>(),
                        16u32,
                    );
                    break 'switch;
                }
                value if value == ABCE_TRANSFER_KIND_PEER_TO_PEER => {
                    num_hwids = (match (*((*(policy)).vtable)).rank_p2p {
                        Some(callback) => callback,
                        None => std::process::abort(),
                    })(
                        policy,
                        src,
                        dst,
                        core::ptr::addr_of_mut!(hwids).cast::<u8>(),
                        16u32,
                    );
                    break 'switch;
                }
                value if value == ABCE_TRANSFER_KIND_HOST_TO_DEVICE => {
                    num_hwids = (match (*((*(policy)).vtable)).rank_h2d {
                        Some(callback) => callback,
                        None => std::process::abort(),
                    })(
                        policy,
                        src,
                        dst,
                        core::ptr::addr_of_mut!(hwids).cast::<u8>(),
                        16u32,
                    );
                    break 'switch;
                }
                value if value == ABCE_TRANSFER_KIND_DEVICE_TO_HOST => {
                    num_hwids = (match (*((*(policy)).vtable)).rank_d2h {
                        Some(callback) => callback,
                        None => std::process::abort(),
                    })(
                        policy,
                        src,
                        dst,
                        core::ptr::addr_of_mut!(hwids).cast::<u8>(),
                        16u32,
                    );
                    break 'switch;
                }
                value if value == ABCE_TRANSFER_KIND_UNKNOWN => {
                    break 'switch;
                }
                _ => {}
            }
        }
        if (num_hwids) == (0_u32) {
            (result).disposition = ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE;
            return result;
        }
        let mut count: u32 = 0_u32;
        let mut emitted: u64 = 0_u64;
        {
            let mut idx: u32 = 0_u32;
            let mut first_26 = true;
            loop {
                if !first_26 {
                    idx = (idx).wrapping_add(1u32);
                }
                first_26 = false;
                if !(((idx) < (num_hwids)) && ((count) < (max))) {
                    break;
                }
                {
                    let hw_engine_id: u8 = *core::ptr::addr_of_mut!(hwids)
                        .cast::<u8>()
                        .wrapping_offset((idx) as isize);
                    if (u32::from(hw_engine_id) >= (16u32))
                        || ((*core::ptr::addr_of!((*(policy)).hwid_to_reg)
                            .cast::<i32>()
                            .wrapping_offset((hw_engine_id) as isize))
                            < (0i32))
                    {
                        continue;
                    }
                    let reg_index: u32 = (*core::ptr::addr_of!((*(policy)).hwid_to_reg)
                        .cast::<i32>()
                        .wrapping_offset((hw_engine_id) as isize))
                        as u32;
                    let engine_bit: u64 = (1u64).wrapping_shl(u64::from(reg_index) as u32);
                    if (emitted & engine_bit) != 0 || ((candidate_mask & engine_bit) == 0) {
                        continue;
                    }
                    emitted |= engine_bit;
                    (*(out_engines).wrapping_offset(
                        ({
                            let old = count;
                            count = (count).wrapping_add(1u32);
                            old
                        }) as isize,
                    )) = reg_index;
                }
            }
        }
        (result).disposition = (if (count) != 0 {
            ABCE_POLICY_DISPOSITION_RANKED as i32
        } else {
            ABCE_POLICY_DISPOSITION_NO_LEGAL_ENGINE as i32
        }) as abce_policy_disposition_t;
        (result).count = count;
        result
    }
}

pub(super) unsafe extern "C" fn abce_sdma_policy_trampoline(
    user_data: *mut core::ffi::c_void,
    src: *const abce_copy_endpoint_t,
    dst: *const abce_copy_endpoint_t,
    candidate_mask: u64,
    out_engines: *mut u32,
    max: u32,
) -> abce_policy_result_t {
    unsafe {
        abce_sdma_policy_rank_engines(
            (user_data) as *const abce_sdma_engine_policy_t,
            src,
            dst,
            candidate_mask,
            out_engines,
            max,
        )
    }
}
