// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE config mechanisms.

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
    ABCE_SDMA_IP_VERSION_OSS4, ABCE_SDMA_IP_VERSION_OSS5, ABCE_SDMA_IP_VERSION_OSS7,
    ABCE_STATUS_INVALID_ARGUMENT, ABCE_STATUS_INVALID_PLAN,
    ABCE_STATUS_MISSING_COORDINATION_SCRATCH, ABCE_STATUS_NO_ENGINE, ABCE_STATUS_NO_LEGAL_ENGINE,
    ABCE_STATUS_NOT_READY, ABCE_STATUS_OK, ABCE_STATUS_OUT_OF_RANGE, ABCE_STATUS_RECT_OUT_OF_RANGE,
    ABCE_STATUS_RESOURCE_EXHAUSTED, ABCE_STATUS_RING_UNAVAILABLE, ABCE_STATUS_TOO_MANY_OPERATIONS,
    ABCE_STATUS_UNIMPLEMENTED, abce_builder_config_t, abce_isa_version_t, abce_packet_caps_t,
    abce_platform_caps_t, abce_status_t,
};

pub(super) unsafe extern "C" fn abce_align_up(value: u64, alignment: u64) -> u64 {
    (((value).wrapping_add(alignment)).wrapping_sub(1_u64)) & !((alignment).wrapping_sub(1_u64))
}

pub(super) unsafe extern "C" fn abce_div_round_up_u64(value: u64, divisor: u64) -> u64 {
    (value / divisor).wrapping_add(i32::from((value % divisor) != (0_u64)) as u64)
}

#[must_use]
pub unsafe extern "C" fn abce_is_power_of_two(value: u64) -> bool {
    ((value) != (0_u64)) && ((value & ((value).wrapping_sub(1_u64))) == (0_u64))
}

pub(super) unsafe extern "C" fn abce_count_trailing_zeros_u64(value: u64) -> i32 {
    (value).trailing_zeros() as i32
}

pub(super) unsafe extern "C" fn abce_popcount_u64(value: u64) -> i32 {
    (value).count_ones() as i32
}

#[must_use]
pub unsafe extern "C" fn abce_status_is_ok(status: abce_status_t) -> bool {
    (status) == ABCE_STATUS_OK
}

#[must_use]
pub unsafe extern "C" fn abce_status_name(status: abce_status_t) -> *const core::ffi::c_char {
    {
        {
            match status {
                value if value == ABCE_STATUS_OK => {
                    return c"OK".as_ptr();
                }
                value if value == ABCE_STATUS_INVALID_ARGUMENT => {
                    return c"INVALID_ARGUMENT".as_ptr();
                }
                value if value == ABCE_STATUS_OUT_OF_RANGE => {
                    return c"OUT_OF_RANGE".as_ptr();
                }
                value if value == ABCE_STATUS_RESOURCE_EXHAUSTED => {
                    return c"RESOURCE_EXHAUSTED".as_ptr();
                }
                value if value == ABCE_STATUS_UNIMPLEMENTED => {
                    return c"UNIMPLEMENTED".as_ptr();
                }
                value if value == ABCE_STATUS_RECT_OUT_OF_RANGE => {
                    return c"RECT_OUT_OF_RANGE".as_ptr();
                }
                value if value == ABCE_STATUS_NO_ENGINE => {
                    return c"NO_ENGINE".as_ptr();
                }
                value if value == ABCE_STATUS_NO_LEGAL_ENGINE => {
                    return c"NO_LEGAL_ENGINE".as_ptr();
                }
                value if value == ABCE_STATUS_TOO_MANY_OPERATIONS => {
                    return c"TOO_MANY_OPERATIONS".as_ptr();
                }
                value if value == ABCE_STATUS_MISSING_COORDINATION_SCRATCH => {
                    return c"MISSING_COORDINATION_SCRATCH"
                        .as_ptr()
                        .cast_mut()
                        .cast_const();
                }
                value if value == ABCE_STATUS_RING_UNAVAILABLE => {
                    return c"RING_UNAVAILABLE".as_ptr();
                }
                value if value == ABCE_STATUS_INVALID_PLAN => {
                    return c"INVALID_PLAN".as_ptr();
                }
                value if value == ABCE_STATUS_NOT_READY => {
                    return c"NOT_READY".as_ptr();
                }
                _ => {}
            }
        }
        c"UNKNOWN".as_ptr()
    }
}

#[must_use]
pub unsafe extern "C" fn abce_detect_packet_caps(isa: abce_isa_version_t) -> abce_packet_caps_t {
    {
        let mut caps: abce_packet_caps_t = unsafe { core::mem::zeroed() };
        (caps).version = ABCE_SDMA_IP_VERSION_OSS4;
        (caps).use_gcr = false;
        (caps).scope_fields = false;
        (caps).gfx125plus = false;
        (caps).max_linear_copy_size = 4_194_272_usize;
        if ((isa).major) == (9_u32) {
            (caps).max_linear_copy_size = (if (((isa).minor) >= (4_u32))
                || ((((isa).minor) == (0_u32)) && (((isa).stepping) == (10_u32)))
            {
                1_073_741_823_i32
            } else {
                4_194_303_i32
            }) as usize;
            return caps;
        }
        if ((isa).major) == (10_u32) {
            (caps).version = ABCE_SDMA_IP_VERSION_OSS5;
            (caps).use_gcr = true;
            (caps).max_linear_copy_size = (if ((isa).minor) < (3_u32) {
                4_194_303_i32
            } else {
                1_073_741_823_i32
            }) as usize;
            return caps;
        }
        if ((((isa).major) == (11_u32)) || (((isa).major) == (12_u32))) && (((isa).minor) < (5_u32))
        {
            (caps).version = ABCE_SDMA_IP_VERSION_OSS5;
            (caps).use_gcr = true;
            (caps).max_linear_copy_size = 1_073_741_823_usize;
            return caps;
        }
        if ((((isa).major) == (11_u32)) || (((isa).major) == (12_u32)))
            && (((isa).minor) >= (5_u32))
        {
            (caps).version = ABCE_SDMA_IP_VERSION_OSS7;
            (caps).scope_fields = true;
            (caps).gfx125plus = (((isa).major) == (12_u32)) && (((isa).minor) >= (5_u32));
            (caps).max_linear_copy_size = 1_073_741_823_usize;
            return caps;
        }
        caps
    }
}

#[must_use]
pub unsafe extern "C" fn abce_detect_default_platform_caps(
    isa: abce_isa_version_t,
) -> abce_platform_caps_t {
    {
        let mut caps: abce_platform_caps_t = unsafe { core::mem::zeroed() };
        (caps).device_atomic_support = !(((((isa).major) == (7_u32))
            && (((isa).minor) == (0_u32)))
            && (((isa).stepping) == (1_u32)));
        (caps).emit_hdp_flush = (((isa).major) >= (9_u32))
            && !((((isa).major) == (10_u32)) && (((isa).minor) == (1_u32)));
        (caps).driver_manages_gcr = false;
        caps
    }
}

pub unsafe extern "C" fn abce_builder_config_initialize(out_config: *mut abce_builder_config_t) {
    unsafe {
        (*(out_config)).use_copy_size_override = true;
        (*(out_config)).max_linear_copy_size = 0_usize;
        (*(out_config)).max_fill_size = 0_usize;
    }
}
