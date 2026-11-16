// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE topology mechanisms.

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

use super::abi::abce_topology_data_t;

pub unsafe extern "C" fn abce_topology_data_initialize(out_topology: *mut abce_topology_data_t) {
    unsafe {
        {
            let mut device_id: u32 = 0_u32;
            let mut first_18 = true;
            loop {
                if !first_18 {
                    device_id = (device_id).wrapping_add(1u32);
                }
                first_18 = false;
                if (device_id) >= (16_u32) {
                    break;
                }
                {
                    (*core::ptr::addr_of_mut!((*(out_topology)).xgmi_physical_id)
                        .cast::<i32>()
                        .wrapping_offset((device_id) as isize)) = (1i32).wrapping_neg();
                    (*core::ptr::addr_of_mut!((*(out_topology)).hive_id)
                        .cast::<u64>()
                        .wrapping_offset((device_id) as isize)) = 0_u64;
                    {
                        let mut peer_id: u32 = 0_u32;
                        let mut first_19 = true;
                        loop {
                            if !first_19 {
                                peer_id = (peer_id).wrapping_add(1u32);
                            }
                            first_19 = false;
                            if (peer_id) >= (16_u32) {
                                break;
                            }
                            (*core::ptr::addr_of_mut!(
                                (*core::ptr::addr_of_mut!((*(out_topology)).recommended_mask)
                                    .cast::<[u64; 16]>()
                                    .wrapping_offset((device_id) as isize))
                            )
                            .cast::<u64>()
                            .wrapping_offset((peer_id) as isize)) = 0_u64;
                        }
                    }
                }
            }
        }
    }
}

pub unsafe extern "C" fn abce_topology_data_set_device(
    topology: *mut abce_topology_data_t,
    device_id: u32,
    xgmi_physical_id: i32,
    hive_id: u64,
) {
    unsafe {
        if (device_id) >= (16_u32) {
            return;
        }
        (*core::ptr::addr_of_mut!((*(topology)).xgmi_physical_id)
            .cast::<i32>()
            .wrapping_offset((device_id) as isize)) = xgmi_physical_id;
        (*core::ptr::addr_of_mut!((*(topology)).hive_id)
            .cast::<u64>()
            .wrapping_offset((device_id) as isize)) = hive_id;
    }
}

pub unsafe extern "C" fn abce_topology_data_set_recommended_mask(
    topology: *mut abce_topology_data_t,
    source_device_id: u32,
    destination_device_id: u32,
    mask: u64,
) {
    unsafe {
        if ((source_device_id) >= (16_u32)) || ((destination_device_id) >= (16_u32)) {
            return;
        }
        (*core::ptr::addr_of_mut!(
            (*core::ptr::addr_of_mut!((*(topology)).recommended_mask)
                .cast::<[u64; 16]>()
                .wrapping_offset((source_device_id) as isize))
        )
        .cast::<u64>()
        .wrapping_offset((destination_device_id) as isize)) = mask;
    }
}

#[must_use]
pub unsafe extern "C" fn abce_topology_data_xgmi_physical_id(
    topology: *const abce_topology_data_t,
    device_id: u32,
) -> i32 {
    unsafe {
        if (device_id) < (16_u32) {
            *core::ptr::addr_of!((*(topology)).xgmi_physical_id)
                .cast::<i32>()
                .wrapping_offset((device_id) as isize)
        } else {
            (1i32).wrapping_neg()
        }
    }
}

#[must_use]
pub unsafe extern "C" fn abce_topology_data_hive_id(
    topology: *const abce_topology_data_t,
    device_id: u32,
) -> u64 {
    unsafe {
        if (device_id) < (16_u32) {
            *core::ptr::addr_of!((*(topology)).hive_id)
                .cast::<u64>()
                .wrapping_offset((device_id) as isize)
        } else {
            0_u64
        }
    }
}

#[must_use]
pub unsafe extern "C" fn abce_topology_data_recommended_mask(
    topology: *const abce_topology_data_t,
    source_device_id: u32,
    destination_device_id: u32,
) -> u64 {
    unsafe {
        if ((source_device_id) < (16_u32)) && ((destination_device_id) < (16_u32)) {
            *core::ptr::addr_of!(
                (*core::ptr::addr_of!((*(topology)).recommended_mask)
                    .cast::<[u64; 16]>()
                    .wrapping_offset((source_device_id) as isize))
            )
            .cast::<u64>()
            .wrapping_offset((destination_device_id) as isize)
        } else {
            0_u64
        }
    }
}
