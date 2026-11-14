// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE builder mechanisms.

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
    ABCE_STATUS_INVALID_ARGUMENT, ABCE_STATUS_OK, ABCE_STATUS_RECT_OUT_OF_RANGE, SDMA_PKT_ATOMIC,
    SDMA_PKT_CONSTANT_FILL, SDMA_PKT_COPY_LINEAR, SDMA_PKT_COPY_LINEAR_BROADCAST,
    SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS,
    SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS, SDMA_PKT_COPY_LINEAR_RECT,
    SDMA_PKT_COPY_LINEAR_RECT_GFX12, SDMA_PKT_COPY_LINEAR_SWAP,
    SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS, SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS,
    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS,
    SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS, SDMA_PKT_FENCE,
    SDMA_PKT_FENCE_64B_GFX125PLUS, SDMA_PKT_FENCE_GFX12, SDMA_PKT_GCR, SDMA_PKT_GCR_GFX125PLUS,
    SDMA_PKT_HDP_FLUSH, SDMA_PKT_POLL_MEM_64B_GFX125PLUS, SDMA_PKT_POLL_REGMEM, SDMA_PKT_TIMESTAMP,
    SDMA_PKT_TRAP, SDMA_PKT_WRITE_UNTILED, abce_builder_config_t, abce_builder_t, abce_dim3_t,
    abce_isa_version_t, abce_packet_caps_t, abce_pitched_ptr_t, abce_rect_append_fn_t,
    abce_rect_count_state_t, abce_status_t, abce_wait_signal_params_t,
};
use super::{
    abce_builder_config_initialize, abce_count_trailing_zeros_u64, abce_detect_packet_caps,
    abce_div_round_up_u64, abce_sdma_hdp_flush_cmd,
};

pub(super) unsafe extern "C" fn abce_ptr_low32(ptr: *const core::ffi::c_void) -> u32 {
    ((ptr) as usize) as u32
}

pub(super) unsafe extern "C" fn abce_ptr_high32(ptr: *const core::ffi::c_void) -> u32 {
    ((((ptr) as usize) as u64).wrapping_shr((32_u64) as u32)) as u32
}

pub unsafe extern "C" fn abce_builder_initialize(
    isa: abce_isa_version_t,
    mut config: *const abce_builder_config_t,
    out_builder: *mut abce_builder_t,
) -> abce_status_t {
    unsafe {
        let mut defaults: abce_builder_config_t = { core::mem::zeroed() };
        if (config).is_null() {
            abce_builder_config_initialize(core::ptr::addr_of_mut!(defaults));
            config = core::ptr::addr_of_mut!(defaults).cast_const();
        }
        let packet_caps: abce_packet_caps_t = abce_detect_packet_caps(isa);
        if ((((*(config)).max_linear_copy_size) > ((packet_caps).max_linear_copy_size))
            || ((((*(config)).max_fill_size) % (core::mem::size_of::<u32>() as usize) as usize)
                != ((0i32) as usize)))
            || (((*(config)).max_fill_size) > (4_194_272_usize))
        {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        (*(out_builder)).isa = isa;
        (*(out_builder)).packet_caps = packet_caps;
        (*(out_builder)).scope_fields = (packet_caps).scope_fields;
        (*(out_builder)).is_gfx125plus = (packet_caps).gfx125plus;
        (*(out_builder)).max_copy_size = if ((*(config)).max_linear_copy_size) != 0 {
            (*(config)).max_linear_copy_size
        } else if (*(config)).use_copy_size_override {
            (packet_caps).max_linear_copy_size
        } else {
            4_194_272_usize
        };
        (*(out_builder)).use_extended_count =
            (((*(config)).max_linear_copy_size) != (0_usize)) || (*(config)).use_copy_size_override;
        let max_fill_size: usize = if ((*(config)).max_fill_size) != 0 {
            (*(config)).max_fill_size
        } else {
            4_194_272_usize
        };
        (*(out_builder)).max_fill_dwords =
            ((max_fill_size) / (core::mem::size_of::<u32>() as usize) as usize) as u32;
        ABCE_STATUS_OK
    }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_is_gfx125plus(builder: *const abce_builder_t) -> bool {
    unsafe { (*(builder)).is_gfx125plus }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_max_copy_size(builder: *const abce_builder_t) -> usize {
    unsafe { (*(builder)).max_copy_size }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_swap_supported(builder: *const abce_builder_t) -> bool {
    unsafe {
        (*(builder)).is_gfx125plus
            || (((((*(builder)).isa).major) == (9_u32)) && ((((*(builder)).isa).minor) >= (4_u32)))
    }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_swap_alignment(builder: *const abce_builder_t) -> usize {
    unsafe {
        (if (*(builder)).is_gfx125plus {
            32i32
        } else {
            64i32
        }) as usize
    }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_requires_gcr(builder: *const abce_builder_t) -> bool {
    unsafe { ((*(builder)).packet_caps).use_gcr }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_copy_packets(
    builder: *const abce_builder_t,
    size: usize,
) -> u64 {
    unsafe { abce_div_round_up_u64((size) as u64, ((*(builder)).max_copy_size) as u64) }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_broadcast_packets(
    builder: *const abce_builder_t,
    size: usize,
) -> u64 {
    unsafe { abce_builder_num_copy_packets(builder, size) }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_multicast_packets(
    builder: *const abce_builder_t,
    size: usize,
) -> u64 {
    unsafe { abce_builder_num_copy_packets(builder, size) }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_swap_packets(
    builder: *const abce_builder_t,
    size: usize,
) -> u64 {
    unsafe {
        let max_size: usize = (if (*(builder)).scope_fields && (*(builder)).is_gfx125plus {
            1_073_741_792_i32
        } else {
            1_073_741_760_i32
        }) as usize;
        abce_div_round_up_u64((size) as u64, (max_size) as u64)
    }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_wait_signal_swap_packets(
    builder: *const abce_builder_t,
    size_a: usize,
    size_b: usize,
) -> u64 {
    unsafe {
        let _ = builder;
        abce_div_round_up_u64(
            (if (size_a) > (size_b) { size_a } else { size_b }) as u64,
            1_073_741_792_u64,
        )
    }
}

#[must_use]
pub unsafe extern "C" fn abce_builder_num_fill_packets(
    builder: *const abce_builder_t,
    count: usize,
) -> u64 {
    unsafe { abce_div_round_up_u64((count) as u64, u64::from((*(builder)).max_fill_dwords)) }
}

#[must_use]
pub unsafe extern "C" fn abce_wait_signal_none() -> abce_wait_signal_params_t {
    {
        let mut params: abce_wait_signal_params_t = unsafe { core::mem::zeroed() };
        (params).wait_addr = core::ptr::null_mut();
        (params).signal_addr = core::ptr::null_mut();
        (params).wait_reference = 0_u64;
        (params).wait_mask = u64::MAX;
        (params).wait_first_chunk_only = false;
        (params).wait_function = 3u32;
        (params).signal_operation = 112u32;
        (params).signal_value = 1_u64;
        params
    }
}

pub unsafe extern "C" fn abce_builder_build_fence(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    fence: *mut u32,
    fence_value: u32,
) {
    unsafe {
        {};
        if (((*(builder)).isa).major) >= (12_u32) {
            let pkt: *mut SDMA_PKT_FENCE_GFX12 = (cmd_addr).cast::<SDMA_PKT_FENCE_GFX12>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | 0x5_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 16)) | (0x3_u32 << 16);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 20)) | (0x1_u32 << 20);
            };
            if (*(builder)).scope_fields {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 24)) | (0x3_u32 << 24);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((fence) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((fence) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DATA_UNION).bits).word);
                *word = fence_value;
            };
        } else {
            let pkt: *mut SDMA_PKT_FENCE = (cmd_addr).cast::<SDMA_PKT_FENCE>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | 0x5_u32;
            };
            if (((*(builder)).isa).major) >= (10_u32) {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !(0x7u32 << 16)) | (0x3_u32 << 16);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((fence) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((fence) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DATA_UNION).bits).word);
                *word = fence_value;
            };
        }
    }
}

pub(super) unsafe extern "C" fn abce_builder_form_copy_packet(
    builder: *const abce_builder_t,
    pkt: *mut SDMA_PKT_COPY_LINEAR,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    copy_size: u32,
) {
    unsafe {
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | (0x1_u32 << 0);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word &= !(0xffu32 << 8);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 28)) | (0x1_u32 << 28);
            };
        }
        if (*(builder)).use_extended_count {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).count_ext).word);
                *word =
                    (*word & !0x3fff_ffff_u32) | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
            };
        } else {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).count).word);
                *word =
                    (*word & !0x003f_ffff_u32) | (((copy_size).wrapping_sub(1_u32)) & 0x003f_ffff);
            };
        }
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32(src);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32(src);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((dst).cast_const());
        };
    }
}

pub unsafe extern "C" fn abce_builder_build_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    size: usize,
) {
    unsafe {
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < ((*(builder)).max_copy_size) {
                remaining
            } else {
                (*(builder)).max_copy_size
            }) as u32;
            abce_builder_form_copy_packet(
                builder,
                (cmd_addr).cast::<SDMA_PKT_COPY_LINEAR>(),
                ((dst).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>(),
                ((src).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>(),
                copy_size,
            );
            cmd_addr =
                (cmd_addr).wrapping_add(core::mem::size_of::<SDMA_PKT_COPY_LINEAR>() as usize);
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_broadcast_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    dst1: *mut core::ffi::c_void,
    dst2: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    size: usize,
) {
    unsafe {
        {};
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < ((*(builder)).max_copy_size) {
                remaining
            } else {
                (*(builder)).max_copy_size
            }) as u32;
            let cur_dst1: *mut core::ffi::c_void =
                ((dst1).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            let cur_dst2: *mut core::ffi::c_void =
                ((dst2).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            let cur_src: *const core::ffi::c_void =
                ((src).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            let pkt: *mut SDMA_PKT_COPY_LINEAR_BROADCAST =
                (cmd_addr).cast::<SDMA_PKT_COPY_LINEAR_BROADCAST>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word &= !(0xffu32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 27)) | (0x1_u32 << 27);
            };
            if (*(builder)).use_extended_count {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).count_ext).word);
                    *word = (*word & !0x3fff_ffff_u32)
                        | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
                };
            } else {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).count).word);
                    *word = (*word & !0x003f_ffff_u32)
                        | (((copy_size).wrapping_sub(1_u32)) & 0x003f_ffff);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32(cur_src);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32(cur_src);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_dst1).cast_const());
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_dst1).cast_const());
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST2_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_dst2).cast_const());
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST2_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_dst2).cast_const());
            };
            cmd_addr = (cmd_addr)
                .wrapping_add(core::mem::size_of::<SDMA_PKT_COPY_LINEAR_BROADCAST>() as usize);
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_multicast_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    dsts: *const *mut core::ffi::c_void,
    num_dsts: u32,
    src: *const core::ffi::c_void,
    size: usize,
) {
    unsafe {
        let pkt_bytes: usize = (((5_usize)
            .wrapping_add((2_usize).wrapping_mul((num_dsts) as usize)))
        .wrapping_mul((core::mem::size_of::<u32>() as usize) as usize))
            as usize;
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < ((*(builder)).max_copy_size) {
                remaining
            } else {
                (*(builder)).max_copy_size
            }) as u32;
            let pkt: *mut SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS =
                (cmd_addr).cast::<SDMA_PKT_COPY_LINEAR_MULTICAST_GFX125PLUS>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0xa_u32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).bits).word);
                *word =
                    (*word & !0x3fff_ffff_u32) | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                *word = (*word & !0x3ffu32) | (((num_dsts).wrapping_sub(1_u32)) & 0x3ff);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
            let cur_src: *const core::ffi::c_void =
                ((src).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32(cur_src);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32(cur_src);
            };
            let dst_dw: *mut u32 = (cmd_addr).cast::<u32>().wrapping_offset((5i32) as isize);
            {
                let mut dst_idx: u32 = 0_u32;
                let mut first_1 = true;
                loop {
                    if !first_1 {
                        dst_idx = (dst_idx).wrapping_add(1u32);
                    }
                    first_1 = false;
                    if (dst_idx) >= (num_dsts) {
                        break;
                    }
                    {
                        let cur_dst: *const core::ffi::c_void = (((*(dsts)
                            .wrapping_offset((dst_idx) as isize))
                            as *const core::ffi::c_char)
                            .wrapping_add(cur_size))
                        .cast::<core::ffi::c_void>();
                        (*(dst_dw).wrapping_offset(((dst_idx).wrapping_mul(2_u32)) as isize)) =
                            abce_ptr_low32(cur_dst);
                        (*(dst_dw).wrapping_offset(
                            (((dst_idx).wrapping_mul(2_u32)).wrapping_add(1_u32)) as isize,
                        )) = abce_ptr_high32(cur_dst);
                    }
                }
            }
            cmd_addr = (cmd_addr).wrapping_add(pkt_bytes);
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_multicast_wait_signal_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    dsts: *const *mut core::ffi::c_void,
    num_dsts: u32,
    src: *const core::ffi::c_void,
    size: usize,
    params: *const abce_wait_signal_params_t,
) {
    unsafe {
        let do_wait: bool = !((*(params)).wait_addr).is_null();
        let do_signal: bool = !((*(params)).signal_addr).is_null();
        let mut wait_dws: [u32; 7] = [0_u32, 0, 0, 0, 0, 0, 0];
        let mut signal_dws: [u32; 5] = [0_u32, 0, 0, 0, 0];
        {
            let mut tmpl: SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS =
                { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(tmpl).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS>()
                    as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0xa_u32 << 8);
            };
            if do_wait {
                {
                    let mut first_2 = true;
                    loop {
                        if !first_2 {
                            break;
                        }
                        first_2 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !0x7u32) | (((*(params)).wait_function) & 0x7);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(((*(params)).wait_addr).cast_const()))
                                        .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).wait_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_reference) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_HI_UNION).bits).word
                                );
                                *word = (((*(params)).wait_reference).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_mask) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_HI_UNION).bits).word
                                );
                                *word =
                                    (((*(params)).wait_mask).wrapping_shr((32_u64) as u32)) as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(wait_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((1i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (7_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
            if do_signal {
                {
                    let mut first_3 = true;
                    loop {
                        if !first_3 {
                            break;
                        }
                        first_3 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word =
                                    (*word & !0x7fu32) | (((*(params)).signal_operation) & 0x7f);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(
                                        ((*(params)).signal_addr).cast_const(),
                                    ))
                                    .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).signal_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_LO_UNION).bits).word
                                );
                                *word = ((*(params)).signal_value) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_HI_UNION).bits).word
                                );
                                *word = (((*(params)).signal_value).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(signal_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((14i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (5_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
        }
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < ((*(builder)).max_copy_size) {
                remaining
            } else {
                (*(builder)).max_copy_size
            }) as u32;
            let chunk_wait: bool =
                do_wait && (!((*(params)).wait_first_chunk_only) || ((cur_size) == (0_usize)));
            let chunk_signal: bool = do_signal;
            let out: *mut u32 = (cmd_addr).cast::<u32>();
            let mut num_dwords: u32 = 0_u32;
            let mut header: SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS =
                { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(header).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS>()
                    as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0xa_u32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 30)) | (((i32::from(chunk_wait) as u32) & 0x1) << 30);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word =
                    (*word & !(0x1u32 << 31)) | (((i32::from(chunk_signal) as u32) & 0x1) << 31);
            };
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) =
                *((core::ptr::addr_of_mut!(header)) as *const u32).wrapping_offset((0i32) as isize);
            if chunk_wait {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(wait_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 7]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(7_u32);
            }
            let mut chunk: SDMA_PKT_COPY_LINEAR_MULTICAST_WAITSIGNAL_GFX125PLUS =
                { core::mem::zeroed() };
            ((chunk).COUNT_UNION).DW_8_DATA = 0_u32;
            ((chunk).COPY_PARAMETER_UNION).DW_9_DATA = 0_u32;
            {
                let word = core::ptr::addr_of_mut!((((chunk).COUNT_UNION).bits).word);
                *word =
                    (*word & !0x3fff_ffff_u32) | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
            };
            {
                let word = core::ptr::addr_of_mut!((((chunk).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !0x3ffu32) | (((num_dsts).wrapping_sub(1_u32)) & 0x3ff);
            };
            {
                let word = core::ptr::addr_of_mut!((((chunk).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((chunk).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
            let cur_src: *const core::ffi::c_char =
                (src).cast::<core::ffi::c_char>().wrapping_add(cur_size);
            {
                let word = core::ptr::addr_of_mut!((((chunk).SRC_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_src).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((chunk).SRC_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_src).cast::<core::ffi::c_void>());
            };
            let chunk_words: *const u32 = (core::ptr::addr_of_mut!(chunk)) as *const u32;
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) = *(chunk_words).wrapping_offset((8i32) as isize);
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) = *(chunk_words).wrapping_offset((9i32) as isize);
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) = *(chunk_words).wrapping_offset((10i32) as isize);
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) = *(chunk_words).wrapping_offset((11i32) as isize);
            {
                let mut dst_idx: u32 = 0_u32;
                let mut first_4 = true;
                loop {
                    if !first_4 {
                        dst_idx = (dst_idx).wrapping_add(1u32);
                    }
                    first_4 = false;
                    if (dst_idx) >= (num_dsts) {
                        break;
                    }
                    {
                        let cur_dst: *const core::ffi::c_char = ((*(dsts)
                            .wrapping_offset((dst_idx) as isize))
                            as *const core::ffi::c_char)
                            .wrapping_add(cur_size);
                        (*(out).wrapping_offset(
                            ({
                                let old = num_dwords;
                                num_dwords = (num_dwords).wrapping_add(1u32);
                                old
                            }) as isize,
                        )) = abce_ptr_low32((cur_dst).cast::<core::ffi::c_void>());
                        (*(out).wrapping_offset(
                            ({
                                let old = num_dwords;
                                num_dwords = (num_dwords).wrapping_add(1u32);
                                old
                            }) as isize,
                        )) = abce_ptr_high32((cur_dst).cast::<core::ffi::c_void>());
                    }
                }
            }
            if chunk_signal {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(signal_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 5]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(5_u32);
            }
            cmd_addr = (cmd_addr).wrapping_add(
                ((num_dwords) as usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_swap_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    addr_a: *mut core::ffi::c_void,
    addr_b: *mut core::ffi::c_void,
    size: usize,
) {
    unsafe {
        let use_gfx125plus: bool = (*(builder)).scope_fields && (*(builder)).is_gfx125plus;
        let alignment: usize = abce_builder_swap_alignment(builder);
        let _ = alignment;
        {};
        {};
        let max_copy_size: usize = (if use_gfx125plus {
            1_073_741_792_i32
        } else {
            1_073_741_760_i32
        }) as usize;
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < (max_copy_size) {
                remaining
            } else {
                max_copy_size
            }) as u32;
            let cur_addr_a: *mut core::ffi::c_void =
                ((addr_a).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            let cur_addr_b: *mut core::ffi::c_void =
                ((addr_b).cast::<core::ffi::c_char>().wrapping_add(cur_size))
                    .cast::<core::ffi::c_void>();
            if use_gfx125plus {
                let pkt: *mut SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS =
                    (cmd_addr).cast::<SDMA_PKT_COPY_LINEAR_SWAP_GFX125PLUS>();
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !0xffu32) | (0x1_u32 << 0);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !(0xffu32 << 8)) | (0x9_u32 << 8);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).bits).word);
                    *word = (*word & !0x3fff_ffff_u32)
                        | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).PARAMETER_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                };
                ((*(pkt)).ADDR_A_LO_UNION).DW_3_DATA = abce_ptr_low32((cur_addr_a).cast_const());
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_A_HI_UNION).bits).word);
                    *word = abce_ptr_high32((cur_addr_a).cast_const());
                };
                ((*(pkt)).ADDR_B_LO_UNION).DW_5_DATA = abce_ptr_low32((cur_addr_b).cast_const());
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_B_HI_UNION).bits).word);
                    *word = abce_ptr_high32((cur_addr_b).cast_const());
                };
            } else {
                let pkt: *mut SDMA_PKT_COPY_LINEAR_SWAP =
                    (cmd_addr).cast::<SDMA_PKT_COPY_LINEAR_SWAP>();
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !0xffu32) | (0x1_u32 << 0);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !(0xffu32 << 8)) | (0x9_u32 << 8);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).bits).word);
                    *word = (*word & !0x3fff_ffff_u32)
                        | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
                };
                ((*(pkt)).ADDR_A_LO_UNION).DW_3_DATA = abce_ptr_low32((cur_addr_a).cast_const());
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_A_HI_UNION).bits).word);
                    *word = abce_ptr_high32((cur_addr_a).cast_const());
                };
                ((*(pkt)).ADDR_B_LO_UNION).DW_5_DATA = abce_ptr_low32((cur_addr_b).cast_const());
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_B_HI_UNION).bits).word);
                    *word = abce_ptr_high32((cur_addr_b).cast_const());
                };
            }
            cmd_addr =
                (cmd_addr).wrapping_add(core::mem::size_of::<SDMA_PKT_COPY_LINEAR_SWAP>() as usize);
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub(super) unsafe extern "C" fn abce_max_aligned_element(width: usize) -> i32 {
    unsafe { abce_count_trailing_zeros_u64((width) as u64 | 16u64) }
}

pub unsafe extern "C" fn abce_builder_build_copy_rect(
    builder: *const abce_builder_t,
    append: abce_rect_append_fn_t,
    user_data: *mut core::ffi::c_void,
    dst: *const abce_pitched_ptr_t,
    dst_offset: *const abce_dim3_t,
    src: *const abce_pitched_ptr_t,
    src_offset: *const abce_dim3_t,
    range: *const abce_dim3_t,
) -> abce_status_t {
    unsafe {
        let is_gfx12plus: bool = (((*(builder)).isa).major) >= (12_u32);
        let max_pitch: u32 = (1u32).wrapping_shl((if is_gfx12plus { 16i32 } else { 19i32 }) as u32);
        let max_slice: u64 =
            (1u64).wrapping_shl(((if is_gfx12plus { 32i32 } else { 28i32 }) as u64) as u32);
        let max_x: u32 = (1u32).wrapping_shl((if is_gfx12plus { 16i32 } else { 14i32 }) as u32);
        let max_y: u32 = max_x;
        let max_z: u32 = (1u32).wrapping_shl((if is_gfx12plus { 14i32 } else { 11i32 }) as u32);
        let src_pitch_ele: i32 = abce_max_aligned_element((*(src)).pitch);
        let dst_pitch_ele: i32 = abce_max_aligned_element((*(dst)).pitch);
        let mut max_ele: i32 = if (src_pitch_ele) < (dst_pitch_ele) {
            src_pitch_ele
        } else {
            dst_pitch_ele
        };
        if ((*(range)).z) != (1_u32) {
            let src_slice_ele: i32 = abce_max_aligned_element((*(src)).slice);
            let dst_slice_ele: i32 = abce_max_aligned_element((*(dst)).slice);
            max_ele = if (max_ele)
                < (if (src_slice_ele) < (dst_slice_ele) {
                    src_slice_ele
                } else {
                    dst_slice_ele
                }) {
                max_ele
            } else if (src_slice_ele) < (dst_slice_ele) {
                src_slice_ele
            } else {
                dst_slice_ele
            };
        }
        let range_x_ele: i32 = abce_max_aligned_element(((*(range)).x) as usize);
        let src_off_ele: i32 = abce_max_aligned_element(((*(src_offset)).x % 4_u32) as usize);
        let dst_off_ele: i32 = abce_max_aligned_element(((*(dst_offset)).x % 4_u32) as usize);
        let min_ele: i32 = if (if (max_ele) < (range_x_ele) {
            max_ele
        } else {
            range_x_ele
        }) < (if (src_off_ele) < (dst_off_ele) {
            src_off_ele
        } else {
            dst_off_ele
        }) {
            if (max_ele) < (range_x_ele) {
                max_ele
            } else {
                range_x_ele
            }
        } else {
            if (src_off_ele) < (dst_off_ele) {
                src_off_ele
            } else {
                dst_off_ele
            }
        };
        if ((((*(src)).pitch).wrapping_shr(((min_ele) as usize) as u32)) > ((max_pitch) as usize))
            || ((((*(dst)).pitch).wrapping_shr(((min_ele) as usize) as u32))
                > ((max_pitch) as usize))
        {
            return ABCE_STATUS_RECT_OUT_OF_RANGE;
        }
        if ((*(range)).z) != (1_u32)
            && (((((*(src)).slice).wrapping_shr(((min_ele) as usize) as u32))
                > ((max_slice) as usize))
                || ((((*(dst)).slice).wrapping_shr(((min_ele) as usize) as u32))
                    > ((max_slice) as usize)))
        {
            return ABCE_STATUS_RECT_OUT_OF_RANGE;
        }
        {
            let mut z: u32 = 0_u32;
            let mut first_5 = true;
            loop {
                if !first_5 {
                    z = (z).wrapping_add(max_z);
                }
                first_5 = false;
                if (z) >= ((*(range)).z) {
                    break;
                }
                {
                    {
                        let mut y: u32 = 0_u32;
                        let mut first_6 = true;
                        loop {
                            if !first_6 {
                                y = (y).wrapping_add(max_y);
                            }
                            first_6 = false;
                            if (y) >= ((*(range)).y) {
                                break;
                            }
                            {
                                let mut x: u32 = 0_u32;
                                while (x) < ((*(range)).x) {
                                    let width: u32 = ((*(range)).x).wrapping_sub(x);
                                    let src_tile_ele: i32 = abce_max_aligned_element(
                                        ((((*(src_offset)).x).wrapping_add(x)) % 4_u32) as usize,
                                    );
                                    let dst_tile_ele: i32 = abce_max_aligned_element(
                                        ((((*(dst_offset)).x).wrapping_add(x)) % 4_u32) as usize,
                                    );
                                    let aligned_ele: i32 = if (if (src_tile_ele) < (dst_tile_ele) {
                                        src_tile_ele
                                    } else {
                                        dst_tile_ele
                                    }) < (max_ele)
                                    {
                                        if (src_tile_ele) < (dst_tile_ele) {
                                            src_tile_ele
                                        } else {
                                            dst_tile_ele
                                        }
                                    } else {
                                        max_ele
                                    };
                                    let width_ele: i32 = abce_max_aligned_element((width) as usize);
                                    let mut element: i32 = if (width_ele) < (aligned_ele) {
                                        width_ele
                                    } else {
                                        aligned_ele
                                    };
                                    let mut xcount: u32 = (width).wrapping_shr((element) as u32);
                                    if (xcount) > (max_x) {
                                        element = aligned_ele;
                                        let shifted: u32 = (width).wrapping_shr((element) as u32);
                                        xcount = if (shifted) < (max_x) { shifted } else { max_x };
                                    }
                                    let mut sbase: usize = ((((((*(src)).base) as usize)
                                        .wrapping_add(((*(src_offset)).x) as usize))
                                    .wrapping_add((x) as usize))
                                    .wrapping_add(
                                        ((((*(src_offset)).y).wrapping_add(y)) as usize)
                                            .wrapping_mul((*(src)).pitch),
                                    ))
                                    .wrapping_add(
                                        ((((*(src_offset)).z).wrapping_add(z)) as usize)
                                            .wrapping_mul((*(src)).slice),
                                    );
                                    let mut dbase: usize = ((((((*(dst)).base) as usize)
                                        .wrapping_add(((*(dst_offset)).x) as usize))
                                    .wrapping_add((x) as usize))
                                    .wrapping_add(
                                        ((((*(dst_offset)).y).wrapping_add(y)) as usize)
                                            .wrapping_mul((*(dst)).pitch),
                                    ))
                                    .wrapping_add(
                                        ((((*(dst_offset)).z).wrapping_add(z)) as usize)
                                            .wrapping_mul((*(dst)).slice),
                                    );
                                    let soff: u32 = ((sbase % 4_usize)
                                        .wrapping_shr(((element) as usize) as u32))
                                        as u32;
                                    let doff: u32 = ((dbase % 4_usize)
                                        .wrapping_shr(((element) as usize) as u32))
                                        as u32;
                                    sbase &= (!(3u64)) as usize;
                                    dbase &= (!(3u64)) as usize;
                                    x = (x).wrapping_add((xcount).wrapping_shl((element) as u32));
                                    let rect_y: u32 =
                                        (if (((*(range)).y).wrapping_sub(y)) < (max_y) {
                                            ((*(range)).y).wrapping_sub(y)
                                        } else {
                                            max_y
                                        })
                                        .wrapping_sub(1_u32);
                                    let rect_z: u32 =
                                        (if (((*(range)).z).wrapping_sub(z)) < (max_z) {
                                            ((*(range)).z).wrapping_sub(z)
                                        } else {
                                            max_z
                                        })
                                        .wrapping_sub(1_u32);
                                    if is_gfx12plus {
                                        let pkt: *mut SDMA_PKT_COPY_LINEAR_RECT_GFX12 =
                                            ((match append {
                                                Some(callback) => callback,
                                                None => std::process::abort(),
                                            })(
                                                user_data,
                                                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_RECT>()
                                                    as usize,
                                            ))
                                            .cast::<SDMA_PKT_COPY_LINEAR_RECT_GFX12>();
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !0xffu32) | (0x1_u32 << 0);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !(0xffu32 << 8)) | (0x4_u32 << 8);
                                        };
                                        if (*(builder)).scope_fields {
                                            {
                                                let word = core::ptr::addr_of_mut!(
                                                    (((*(pkt)).HEADER_UNION).bits).word
                                                );
                                                *word = (*word & !(0x1u32 << 28)) | (0x1_u32 << 28);
                                            };
                                        }
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !(0x7u32 << 29))
                                                | ((((element) as u32) & 0x7) << 29);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_ADDR_LO_UNION).bits).word
                                            );
                                            *word = (sbase) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_ADDR_HI_UNION).bits).word
                                            );
                                            *word =
                                                ((sbase).wrapping_shr((32_usize) as u32)) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0xffffu32) | ((soff) & 0xffff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_2_UNION).bits).word
                                            );
                                            *word = (*word & !(0xffffu32 << 16))
                                                | (((((((*(src)).pitch)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32))
                                                    & 0xffff)
                                                    << 16);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_3_UNION).bits).word
                                            );
                                            *word = if ((*(range)).z) == (1_u32) {
                                                0_u32
                                            } else {
                                                ((((*(src)).slice)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32)
                                            };
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_ADDR_LO_UNION).bits).word
                                            );
                                            *word = (dbase) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_ADDR_HI_UNION).bits).word
                                            );
                                            *word =
                                                ((dbase).wrapping_shr((32_usize) as u32)) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0xffffu32) | ((doff) & 0xffff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_2_UNION).bits).word
                                            );
                                            *word = (*word & !(0xffffu32 << 16))
                                                | (((((((*(dst)).pitch)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32))
                                                    & 0xffff)
                                                    << 16);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_3_UNION).bits).word
                                            );
                                            *word = if ((*(range)).z) == (1_u32) {
                                                0_u32
                                            } else {
                                                ((((*(dst)).slice)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32)
                                            };
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0xffffu32)
                                                | (((xcount).wrapping_sub(1_u32)) & 0xffff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !(0xffffu32 << 16))
                                                | (((rect_y) & 0xffff) << 16);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_2_UNION).gfx12).word
                                            );
                                            *word = (*word & !0x3fffu32) | ((rect_z) & 0x3fff);
                                        };
                                        if (*(builder)).scope_fields {
                                            {
                                                let word = core::ptr::addr_of_mut!(
                                                    (((*(pkt)).RECT_PARAMETER_2_UNION).gfx125plus)
                                                        .word
                                                );
                                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                                            };
                                            {
                                                let word = core::ptr::addr_of_mut!(
                                                    (((*(pkt)).RECT_PARAMETER_2_UNION).gfx125plus)
                                                        .word
                                                );
                                                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
                                            };
                                        }
                                    } else {
                                        let pkt: *mut SDMA_PKT_COPY_LINEAR_RECT = ((match append {
                                            Some(callback) => callback,
                                            None => std::process::abort(),
                                        })(
                                            user_data,
                                            core::mem::size_of::<SDMA_PKT_COPY_LINEAR_RECT>()
                                                as usize,
                                        ))
                                        .cast::<SDMA_PKT_COPY_LINEAR_RECT>();
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !0xffu32) | (0x1_u32 << 0);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !(0xffu32 << 8)) | (0x4_u32 << 8);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).HEADER_UNION).bits).word
                                            );
                                            *word = (*word & !(0x7u32 << 29))
                                                | ((((element) as u32) & 0x7) << 29);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_ADDR_LO_UNION).bits).word
                                            );
                                            *word = (sbase) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_ADDR_HI_UNION).bits).word
                                            );
                                            *word =
                                                ((sbase).wrapping_shr((32_usize) as u32)) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0x3fffu32) | ((soff) & 0x3fff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_2_UNION).bits).word
                                            );
                                            *word = (*word & !(0x7ffffu32 << 13))
                                                | (((((((*(src)).pitch)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32))
                                                    & 0x7ffff)
                                                    << 13);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).SRC_PARAMETER_3_UNION).bits).word
                                            );
                                            *word = (*word & !0x0fff_ffff_u32)
                                                | ((if ((*(range)).z) == (1_u32) {
                                                    0_u32
                                                } else {
                                                    ((((*(src)).slice)
                                                        .wrapping_shr(((element) as usize) as u32))
                                                        as u32)
                                                        .wrapping_sub(1_u32)
                                                }) & 0x0fff_ffff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_ADDR_LO_UNION).bits).word
                                            );
                                            *word = (dbase) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_ADDR_HI_UNION).bits).word
                                            );
                                            *word =
                                                ((dbase).wrapping_shr((32_usize) as u32)) as u32;
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0x3fffu32) | ((doff) & 0x3fff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_2_UNION).bits).word
                                            );
                                            *word = (*word & !(0x7ffffu32 << 13))
                                                | (((((((*(dst)).pitch)
                                                    .wrapping_shr(((element) as usize) as u32))
                                                    as u32)
                                                    .wrapping_sub(1_u32))
                                                    & 0x7ffff)
                                                    << 13);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).DST_PARAMETER_3_UNION).bits).word
                                            );
                                            *word = (*word & !0x0fff_ffff_u32)
                                                | ((if ((*(range)).z) == (1_u32) {
                                                    0_u32
                                                } else {
                                                    ((((*(dst)).slice)
                                                        .wrapping_shr(((element) as usize) as u32))
                                                        as u32)
                                                        .wrapping_sub(1_u32)
                                                }) & 0x0fff_ffff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !0x3fffu32)
                                                | (((xcount).wrapping_sub(1_u32)) & 0x3fff);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_1_UNION).bits).word
                                            );
                                            *word = (*word & !(0x3fffu32 << 16))
                                                | (((rect_y) & 0x3fff) << 16);
                                        };
                                        {
                                            let word = core::ptr::addr_of_mut!(
                                                (((*(pkt)).RECT_PARAMETER_2_UNION).bits).word
                                            );
                                            *word = (*word & !0x7ffu32) | ((rect_z) & 0x7ff);
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ABCE_STATUS_OK
    }
}

pub(super) unsafe extern "C" fn abce_rect_count_append(
    user_data: *mut core::ffi::c_void,
    packet_bytes: usize,
) -> *mut core::ffi::c_void {
    unsafe {
        let state: *mut abce_rect_count_state_t = (user_data).cast::<abce_rect_count_state_t>();
        let _ = packet_bytes;
        (*(state)).num_packets = ((*(state)).num_packets).wrapping_add(1u32);
        core::ptr::addr_of_mut!((*(state)).scratch)
            .cast::<core::ffi::c_char>()
            .cast::<core::ffi::c_void>()
    }
}

pub unsafe extern "C" fn abce_builder_num_rect_packets(
    builder: *const abce_builder_t,
    dst: *const abce_pitched_ptr_t,
    dst_offset: *const abce_dim3_t,
    src: *const abce_pitched_ptr_t,
    src_offset: *const abce_dim3_t,
    range: *const abce_dim3_t,
    out_num_packets: *mut u32,
) -> abce_status_t {
    unsafe {
        let mut state: abce_rect_count_state_t = { core::mem::zeroed() };
        (state).num_packets = 0_u32;
        let status: abce_status_t = abce_builder_build_copy_rect(
            builder,
            Some(abce_rect_count_append),
            core::ptr::addr_of_mut!(state).cast::<core::ffi::c_void>(),
            dst,
            dst_offset,
            src,
            src_offset,
            range,
        );
        (*(out_num_packets)) = (state).num_packets;
        status
    }
}

pub unsafe extern "C" fn abce_builder_build_fill(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    ptr: *mut core::ffi::c_void,
    value: u32,
    mut count: usize,
) {
    unsafe {
        let mut cur_ptr: *mut core::ffi::c_char = (ptr).cast::<core::ffi::c_char>();
        let mut pkt: *mut SDMA_PKT_CONSTANT_FILL = (cmd_addr).cast::<SDMA_PKT_CONSTANT_FILL>();
        while (count) > (0_usize) {
            let fill_count: u32 = (if (count) < (((*(builder)).max_fill_dwords) as usize) {
                count
            } else {
                ((*(builder)).max_fill_dwords) as usize
            }) as u32;
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | 0xb_u32;
            };
            if (*(builder)).scope_fields {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 24)) | (0x3_u32 << 24);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 30)) | (0x2_u32 << 30);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_ptr) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DST_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_ptr) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DATA_UNION).bits).word);
                *word = value;
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).COUNT_UNION).bits).word);
                *word = (*word & !0x003f_ffff_u32)
                    | ((((((fill_count).wrapping_sub(1_u32)) as usize)
                        .wrapping_mul(core::mem::size_of::<u32>() as usize))
                        as u32) as u32
                        & 0x003f_ffff);
            };
            pkt = (pkt).wrapping_add(1);
            cur_ptr = (cur_ptr).wrapping_add(
                ((fill_count) as usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            count = (count).wrapping_sub((fill_count) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_poll_compare(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    reference: u32,
    mask: u32,
    function: u32,
) {
    unsafe {
        let pkt: *mut SDMA_PKT_POLL_REGMEM = (cmd_addr).cast::<SDMA_PKT_POLL_REGMEM>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0x8_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 31)) | (0x1_u32 << 31);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x7u32 << 28)) | (((function) & 0x7) << 28);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).VALUE_UNION).bits).word);
            *word = reference;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).MASK_UNION).bits).word);
            *word = mask;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DW5_UNION).bits).word);
            *word = (*word & !0xffffu32) | 0x4_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DW5_UNION).bits).word);
            *word = (*word & !(0xfffu32 << 16)) | (0xfff_u32 << 16);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DW5_UNION).bits).word);
                *word = (*word & !(0x3u32 << 28)) | (0x3_u32 << 28);
            };
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_poll(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    reference: u32,
    mask: u32,
) {
    unsafe {
        abce_builder_build_poll_compare(builder, cmd_addr, addr, reference, mask, 3u32);
    }
}

pub unsafe extern "C" fn abce_builder_build_poll_64b_compare(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    reference: u64,
    mask: u64,
    function: u32,
) {
    unsafe {
        let pkt: *mut SDMA_PKT_POLL_MEM_64B_GFX125PLUS =
            (cmd_addr).cast::<SDMA_PKT_POLL_MEM_64B_GFX125PLUS>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0x8_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0xffu32 << 8)) | (0x5_u32 << 8);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x7u32 << 28)) | (((function) & 0x7) << 28);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
            *word = (*word & !(0x1fff_ffff_u32 << 3))
                | ((((abce_ptr_low32((addr).cast_const())).wrapping_shr(3_u32)) & 0x1fff_ffff)
                    << 3);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).REFERENCE_LO_UNION).bits).word);
            *word = (reference) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).REFERENCE_HI_UNION).bits).word);
            *word = ((reference).wrapping_shr((32_u64) as u32)) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).MASK_LO_UNION).bits).word);
            *word = (mask) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).MASK_HI_UNION).bits).word);
            *word = ((mask).wrapping_shr((32_u64) as u32)) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 20)) | (0x1_u32 << 20);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DW7_UNION).bits).word);
            *word &= !(0xffu32 << 16);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).DW7_UNION).bits).word);
                *word = (*word & !(0x3u32 << 28)) | (0x3_u32 << 28);
            };
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_poll_64b(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    reference: u64,
    mask: u64,
) {
    unsafe {
        let () =
            abce_builder_build_poll_64b_compare(builder, cmd_addr, addr, reference, mask, 3u32);
    }
}

pub unsafe extern "C" fn abce_builder_build_fence_64b(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    fence_addr: *mut core::ffi::c_void,
    fence_value: u64,
) {
    unsafe {
        let pkt: *mut SDMA_PKT_FENCE_64B_GFX125PLUS =
            (cmd_addr).cast::<SDMA_PKT_FENCE_64B_GFX125PLUS>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0x5_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0xffu32 << 8)) | (0x2_u32 << 8);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x3u32 << 16)) | (0x3_u32 << 16);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 20)) | (0x1_u32 << 20);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 24)) | (0x3_u32 << 24);
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
            *word = (*word & !(0x1fff_ffff_u32 << 3))
                | ((((abce_ptr_low32((fence_addr).cast_const())).wrapping_shr(3_u32))
                    & 0x1fff_ffff)
                    << 3);
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((fence_addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DATA_LO_UNION).bits).word);
            *word = (fence_value) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).DATA_HI_UNION).bits).word);
            *word = ((fence_value).wrapping_shr((32_u64) as u32)) as u32;
        };
    }
}

pub unsafe extern "C" fn abce_builder_build_wait_signal_copy(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    size: usize,
    params: *const abce_wait_signal_params_t,
) {
    unsafe {
        let do_wait: bool = !((*(params)).wait_addr).is_null();
        let do_signal: bool = !((*(params)).signal_addr).is_null();
        let mut wait_dws: [u32; 7] = [0_u32, 0, 0, 0, 0, 0, 0];
        let mut signal_dws: [u32; 5] = [0_u32, 0, 0, 0, 0];
        {
            let mut tmpl: SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS = { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(tmpl).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS>() as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word &= !(0xffu32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
            if do_wait {
                {
                    let mut first_7 = true;
                    loop {
                        if !first_7 {
                            break;
                        }
                        first_7 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !0x7u32) | (((*(params)).wait_function) & 0x7);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(((*(params)).wait_addr).cast_const()))
                                        .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).wait_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_reference) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_HI_UNION).bits).word
                                );
                                *word = (((*(params)).wait_reference).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_mask) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_HI_UNION).bits).word
                                );
                                *word =
                                    (((*(params)).wait_mask).wrapping_shr((32_u64) as u32)) as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(wait_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((1i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (7_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
            if do_signal {
                {
                    let mut first_8 = true;
                    loop {
                        if !first_8 {
                            break;
                        }
                        first_8 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word =
                                    (*word & !0x7fu32) | (((*(params)).signal_operation) & 0x7f);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(
                                        ((*(params)).signal_addr).cast_const(),
                                    ))
                                    .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).signal_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_LO_UNION).bits).word
                                );
                                *word = ((*(params)).signal_value) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_HI_UNION).bits).word
                                );
                                *word = (((*(params)).signal_value).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(signal_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((14i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (5_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
        }
        let mut cur_size: usize = 0_usize;
        while (cur_size) < (size) {
            let remaining: usize = (size).wrapping_sub(cur_size);
            let copy_size: u32 = (if (remaining) < ((*(builder)).max_copy_size) {
                remaining
            } else {
                (*(builder)).max_copy_size
            }) as u32;
            let chunk_wait: bool =
                do_wait && (!((*(params)).wait_first_chunk_only) || ((cur_size) == (0_usize)));
            let chunk_signal: bool = do_signal;
            let out: *mut u32 = (cmd_addr).cast::<u32>();
            let mut num_dwords: u32 = 0_u32;
            let mut header: SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS = { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(header).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS>() as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word &= !(0xffu32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 30)) | (((i32::from(chunk_wait) as u32) & 0x1) << 30);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word =
                    (*word & !(0x1u32 << 31)) | (((i32::from(chunk_signal) as u32) & 0x1) << 31);
            };
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) =
                *((core::ptr::addr_of_mut!(header)) as *const u32).wrapping_offset((0i32) as isize);
            if chunk_wait {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(wait_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 7]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(7_u32);
            }
            let cur_src: *const core::ffi::c_char =
                (src).cast::<core::ffi::c_char>().wrapping_add(cur_size);
            let cur_dst: *mut core::ffi::c_char =
                (dst).cast::<core::ffi::c_char>().wrapping_add(cur_size);
            let mut body: SDMA_PKT_COPY_LINEAR_WAITSIGNAL_GFX125PLUS = { core::mem::zeroed() };
            ((body).COPY_COUNT_UNION).DW_8_DATA = 0_u32;
            {
                let word = core::ptr::addr_of_mut!((((body).COPY_COUNT_UNION).bits).word);
                *word =
                    (*word & !0x3fff_ffff_u32) | (((copy_size).wrapping_sub(1_u32)) & 0x3fff_ffff);
            };
            ((body).COPY_PARAMETER_UNION).DW_9_DATA = 0_u32;
            {
                let word = core::ptr::addr_of_mut!((((body).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((body).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
            {
                let word = core::ptr::addr_of_mut!((((body).SRC_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_src).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((body).SRC_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_src).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((body).DST_ADDR_LO_UNION).bits).word);
                *word = abce_ptr_low32((cur_dst) as *const core::ffi::c_void);
            };
            {
                let word = core::ptr::addr_of_mut!((((body).DST_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32((cur_dst) as *const core::ffi::c_void);
            };
            let _ = libc::memcpy(
                ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                (((core::ptr::addr_of_mut!(body)) as *const u32).wrapping_offset((8i32) as isize))
                    .cast::<core::ffi::c_void>(),
                (6_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            num_dwords = (num_dwords).wrapping_add(6_u32);
            if chunk_signal {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(signal_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 5]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(5_u32);
            }
            cmd_addr = (cmd_addr).wrapping_add(
                ((num_dwords) as usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            cur_size = (cur_size).wrapping_add((copy_size) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_wait_signal_indirect_copy(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    size: usize,
    indirect_src: bool,
    indirect_dst: bool,
    params: *const abce_wait_signal_params_t,
) {
    unsafe {
        let _ = builder;
        let do_wait: bool = !((*(params)).wait_addr).is_null();
        let do_signal: bool = !((*(params)).signal_addr).is_null();
        let mut tmpl: SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS = { core::mem::zeroed() };
        let _ = libc::memset(
            core::ptr::addr_of_mut!(tmpl).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<SDMA_PKT_COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX125PLUS>() as usize,
        );
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | (0x1_u32 << 0);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word &= !(0xffu32 << 8);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 20)) | (((i32::from(indirect_src) as u32) & 0x1) << 20);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 21)) | (((i32::from(indirect_dst) as u32) & 0x1) << 21);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 30)) | (((i32::from(do_wait) as u32) & 0x1) << 30);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
            *word = (*word & !(0x1u32 << 31)) | (((i32::from(do_signal) as u32) & 0x1) << 31);
        };
        if do_wait {
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_FUNCTION_UNION).bits).word);
                *word = (*word & !0x7u32) | 0x3_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_FUNCTION_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_ADDR_LO_UNION).bits).word);
                *word = (*word & !(0x1fff_ffff_u32 << 3))
                    | ((((abce_ptr_low32(((*(params)).wait_addr).cast_const()))
                        .wrapping_shr(3_u32))
                        & 0x1fff_ffff)
                        << 3);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32(((*(params)).wait_addr).cast_const());
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_REFERENCE_LO_UNION).bits).word);
                *word = ((*(params)).wait_reference) as u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_REFERENCE_HI_UNION).bits).word);
                *word = (((*(params)).wait_reference).wrapping_shr((32_u64) as u32)) as u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_MASK_LO_UNION).bits).word);
                *word = ((*(params)).wait_mask) as u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).WAIT_MASK_HI_UNION).bits).word);
                *word = (((*(params)).wait_mask).wrapping_shr((32_u64) as u32)) as u32;
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((tmpl).COPY_COUNT_UNION).bits).word);
            *word =
                (*word & !0x3fff_ffff_u32) | ((((size) as u32).wrapping_sub(1_u32)) & 0x3fff_ffff);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).COPY_PARAMETER_UNION).bits).word);
            *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).COPY_PARAMETER_UNION).bits).word);
            *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).COPY_PARAMETER_UNION).bits).word);
            *word = (*word & !(0x3u32 << 10)) | (0x3_u32 << 10);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).SRC_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32(src);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).SRC_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32(src);
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).DST_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((tmpl).DST_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((dst).cast_const());
        };
        if do_signal {
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_OPERATION_UNION).bits).word);
                *word = (*word & !0x7fu32) | 0x70_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_OPERATION_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_ADDR_LO_UNION).bits).word);
                *word = (*word & !(0x1fff_ffff_u32 << 3))
                    | ((((abce_ptr_low32(((*(params)).signal_addr).cast_const()))
                        .wrapping_shr(3_u32))
                        & 0x1fff_ffff)
                        << 3);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_ADDR_HI_UNION).bits).word);
                *word = abce_ptr_high32(((*(params)).signal_addr).cast_const());
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_DATA_LO_UNION).bits).word);
                *word = 1_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).SIGNAL_DATA_HI_UNION).bits).word);
                *word = 0_u32;
            };
        }
        let tmpl_words: *const u32 = (core::ptr::addr_of_mut!(tmpl)) as *const u32;
        let out: *mut u32 = (cmd_addr).cast::<u32>();
        let mut num_dwords: u32 = 0_u32;
        (*(out).wrapping_offset(
            ({
                let old = num_dwords;
                num_dwords = (num_dwords).wrapping_add(1u32);
                old
            }) as isize,
        )) = *(tmpl_words).wrapping_offset((0i32) as isize);
        if do_wait {
            let _ = libc::memcpy(
                ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                ((tmpl_words).wrapping_offset((1i32) as isize)).cast::<core::ffi::c_void>(),
                (7_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            num_dwords = (num_dwords).wrapping_add(7_u32);
        }
        let _ = libc::memcpy(
            ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
            ((tmpl_words).wrapping_offset((8i32) as isize)).cast::<core::ffi::c_void>(),
            (6_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
        );
        num_dwords = (num_dwords).wrapping_add(6_u32);
        if do_signal {
            let _ = libc::memcpy(
                ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                ((tmpl_words).wrapping_offset((14i32) as isize)).cast::<core::ffi::c_void>(),
                (5_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_wait_signal_swap(
    builder: *const abce_builder_t,
    mut cmd_addr: *mut core::ffi::c_char,
    addr_a: *mut core::ffi::c_void,
    addr_b: *mut core::ffi::c_void,
    size_a: usize,
    size_b: usize,
    params: *const abce_wait_signal_params_t,
) {
    unsafe {
        let _ = builder;
        let do_wait: bool = !((*(params)).wait_addr).is_null();
        let do_signal: bool = !((*(params)).signal_addr).is_null();
        let max_copy_size: usize = 1_073_741_792_usize;
        let mut wait_dws: [u32; 7] = [0_u32, 0, 0, 0, 0, 0, 0];
        let mut signal_dws: [u32; 5] = [0_u32, 0, 0, 0, 0];
        {
            let mut tmpl: SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS = { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(tmpl).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS>() as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((tmpl).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0x9_u32 << 8);
            };
            if do_wait {
                {
                    let mut first_9 = true;
                    loop {
                        if !first_9 {
                            break;
                        }
                        first_9 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !0x7u32) | (((*(params)).wait_function) & 0x7);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_FUNCTION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(((*(params)).wait_addr).cast_const()))
                                        .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).wait_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_reference) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_REFERENCE_HI_UNION).bits).word
                                );
                                *word = (((*(params)).wait_reference).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_LO_UNION).bits).word
                                );
                                *word = ((*(params)).wait_mask) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).WAIT_MASK_HI_UNION).bits).word
                                );
                                *word =
                                    (((*(params)).wait_mask).wrapping_shr((32_u64) as u32)) as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(wait_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((1i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (7_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
            if do_signal {
                {
                    let mut first_10 = true;
                    loop {
                        if !first_10 {
                            break;
                        }
                        first_10 = false;
                        {
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word =
                                    (*word & !0x7fu32) | (((*(params)).signal_operation) & 0x7f);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_OPERATION_UNION).bits).word
                                );
                                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_LO_UNION).bits).word
                                );
                                *word = (*word & !(0x1fff_ffff_u32 << 3))
                                    | ((((abce_ptr_low32(
                                        ((*(params)).signal_addr).cast_const(),
                                    ))
                                    .wrapping_shr(3_u32))
                                        & 0x1fff_ffff)
                                        << 3);
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_ADDR_HI_UNION).bits).word
                                );
                                *word = abce_ptr_high32(((*(params)).signal_addr).cast_const());
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_LO_UNION).bits).word
                                );
                                *word = ((*(params)).signal_value) as u32;
                            };
                            {
                                let word = core::ptr::addr_of_mut!(
                                    (((tmpl).SIGNAL_DATA_HI_UNION).bits).word
                                );
                                *word = (((*(params)).signal_value).wrapping_shr((32_u64) as u32))
                                    as u32;
                            };
                            let _ = libc::memcpy(
                                core::ptr::addr_of_mut!(signal_dws)
                                    .cast::<u32>()
                                    .cast::<core::ffi::c_void>(),
                                (((core::ptr::addr_of_mut!(tmpl)) as *const u32)
                                    .wrapping_offset((14i32) as isize))
                                .cast::<core::ffi::c_void>(),
                                (5_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
                            );
                        }
                    }
                }
            }
        }
        let mut cur_a: usize = 0_usize;
        let mut cur_b: usize = 0_usize;
        while ((cur_a) < (size_a)) || ((cur_b) < (size_b)) {
            let remaining_a: usize = (size_a).wrapping_sub(cur_a);
            let remaining_b: usize = (size_b).wrapping_sub(cur_b);
            let chunk_a: u32 = (if (remaining_a) < (max_copy_size) {
                remaining_a
            } else {
                max_copy_size
            }) as u32;
            let chunk_b: u32 = (if (remaining_b) < (max_copy_size) {
                remaining_b
            } else {
                max_copy_size
            }) as u32;
            let chunk: u32 = if (chunk_a) > (chunk_b) {
                chunk_a
            } else {
                chunk_b
            };
            let chunk_wait: bool = do_wait
                && (!((*(params)).wait_first_chunk_only)
                    || (((cur_a) == (0_usize)) && ((cur_b) == (0_usize))));
            let chunk_signal: bool = do_signal;
            let ptr_a: *const core::ffi::c_char =
                ((addr_a) as *const core::ffi::c_char).wrapping_add(cur_a);
            let ptr_b: *const core::ffi::c_char =
                ((addr_b) as *const core::ffi::c_char).wrapping_add(cur_b);
            let out: *mut u32 = (cmd_addr).cast::<u32>();
            let mut num_dwords: u32 = 0_u32;
            let mut header: SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS =
                { core::mem::zeroed() };
            let _ = libc::memset(
                core::ptr::addr_of_mut!(header).cast::<core::ffi::c_void>(),
                0i32,
                core::mem::size_of::<SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS>() as usize,
            );
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | (0x1_u32 << 0);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0x9_u32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word = (*word & !(0x1u32 << 30)) | (((i32::from(chunk_wait) as u32) & 0x1) << 30);
            };
            {
                let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
                *word =
                    (*word & !(0x1u32 << 31)) | (((i32::from(chunk_signal) as u32) & 0x1) << 31);
            };
            (*(out).wrapping_offset(
                ({
                    let old = num_dwords;
                    num_dwords = (num_dwords).wrapping_add(1u32);
                    old
                }) as isize,
            )) =
                *((core::ptr::addr_of_mut!(header)) as *const u32).wrapping_offset((0i32) as isize);
            if chunk_wait {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(wait_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 7]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(7_u32);
            }
            let mut body: SDMA_PKT_COPY_LINEAR_SWAP_WAITSIGNAL_GFX125PLUS = { core::mem::zeroed() };
            ((body).COUNT_UNION).DW_8_DATA = 0_u32;
            {
                let word = core::ptr::addr_of_mut!((((body).COUNT_UNION).bits).word);
                *word = (*word & !0x3fff_ffff_u32) | (((chunk).wrapping_sub(1_u32)) & 0x3fff_ffff);
            };
            ((body).COPY_PARAMETER_UNION).DW_9_DATA = 0_u32;
            {
                let word = core::ptr::addr_of_mut!((((body).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 26)) | (0x3_u32 << 26);
            };
            {
                let word = core::ptr::addr_of_mut!((((body).COPY_PARAMETER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 18)) | (0x3_u32 << 18);
            };
            {
                let word = core::ptr::addr_of_mut!((((body).ADDR_A_LO_UNION).bits).word);
                *word = abce_ptr_low32((ptr_a).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((body).ADDR_A_HI_UNION).bits).word);
                *word = abce_ptr_high32((ptr_a).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((body).ADDR_B_LO_UNION).bits).word);
                *word = abce_ptr_low32((ptr_b).cast::<core::ffi::c_void>());
            };
            {
                let word = core::ptr::addr_of_mut!((((body).ADDR_B_HI_UNION).bits).word);
                *word = abce_ptr_high32((ptr_b).cast::<core::ffi::c_void>());
            };
            let _ = libc::memcpy(
                ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                (((core::ptr::addr_of_mut!(body)) as *const u32).wrapping_offset((8i32) as isize))
                    .cast::<core::ffi::c_void>(),
                (6_usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            num_dwords = (num_dwords).wrapping_add(6_u32);
            if chunk_signal {
                let _ = libc::memcpy(
                    ((out).wrapping_offset((num_dwords) as isize)).cast::<core::ffi::c_void>(),
                    core::ptr::addr_of_mut!(signal_dws).cast::<u32>() as *const core::ffi::c_void,
                    core::mem::size_of::<[u32; 5]>() as usize,
                );
                num_dwords = (num_dwords).wrapping_add(5_u32);
            }
            cmd_addr = (cmd_addr).wrapping_add(
                ((num_dwords) as usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
            );
            cur_a = (cur_a).wrapping_add((chunk_a) as usize);
            cur_b = (cur_b).wrapping_add((chunk_b) as usize);
        }
    }
}

pub unsafe extern "C" fn abce_builder_build_write(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    dst: *mut core::ffi::c_void,
    data: *const u32,
    num_dwords: u32,
) {
    unsafe {
        let _ = builder;
        {};
        let mut header: SDMA_PKT_WRITE_UNTILED = { core::mem::zeroed() };
        let _ = libc::memset(
            core::ptr::addr_of_mut!(header).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<SDMA_PKT_WRITE_UNTILED>() as usize,
        );
        {
            let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0x2_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((header).HEADER_UNION).bits).word);
            *word &= !(0xffu32 << 8);
        };
        {
            let word = core::ptr::addr_of_mut!((((header).DST_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((header).DST_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((header).DW_3_UNION).bits).word);
            *word = (*word & !0xfffffu32) | (((num_dwords).wrapping_sub(1_u32)) & 0xfffff);
        };
        let _ = libc::memcpy(
            (cmd_addr).cast::<core::ffi::c_void>(),
            (core::ptr::addr_of_mut!(header)) as *const core::ffi::c_void,
            core::mem::size_of::<SDMA_PKT_WRITE_UNTILED>() as usize,
        );
        let _ = libc::memcpy(
            ((cmd_addr).wrapping_add(core::mem::size_of::<SDMA_PKT_WRITE_UNTILED>() as usize))
                .cast::<core::ffi::c_void>(),
            (data).cast::<core::ffi::c_void>(),
            ((num_dwords) as usize).wrapping_mul(core::mem::size_of::<u32>() as usize),
        );
    }
}

pub unsafe extern "C" fn abce_builder_build_fill_byte(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    dst: *mut core::ffi::c_void,
    value: u8,
) {
    unsafe {
        let mut pkt: SDMA_PKT_CONSTANT_FILL = { core::mem::zeroed() };
        let _ = libc::memset(
            core::ptr::addr_of_mut!(pkt).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<SDMA_PKT_CONSTANT_FILL>() as usize,
        );
        {
            let word = core::ptr::addr_of_mut!((((pkt).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0xb_u32;
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((pkt).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 24)) | (0x3_u32 << 24);
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((pkt).HEADER_UNION).bits).word);
            *word &= !(0x3u32 << 30);
        };
        {
            let word = core::ptr::addr_of_mut!((((pkt).DST_ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((pkt).DST_ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((dst).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((pkt).DATA_UNION).bits).word);
            *word = u32::from(value);
        };
        {
            let word = core::ptr::addr_of_mut!((((pkt).COUNT_UNION).bits).word);
            *word &= !0x003f_ffff_u32;
        };
        let _ = libc::memcpy(
            (cmd_addr).cast::<core::ffi::c_void>(),
            (core::ptr::addr_of_mut!(pkt)) as *const core::ffi::c_void,
            core::mem::size_of::<SDMA_PKT_CONSTANT_FILL>() as usize,
        );
    }
}

pub unsafe extern "C" fn abce_builder_build_atomic(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    operation: u32,
    value: u64,
) {
    unsafe {
        let pkt: *mut SDMA_PKT_ATOMIC = (cmd_addr).cast::<SDMA_PKT_ATOMIC>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0xa_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0x7fu32 << 25)) | (((operation) & 0x7f) << 25);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 20)) | (0x3_u32 << 20);
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((addr).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_DATA_LO_UNION).bits).word);
            *word = (value) as u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).SRC_DATA_HI_UNION).bits).word);
            *word = ((value).wrapping_shr((32_u64) as u32)) as u32;
        };
    }
}

pub unsafe extern "C" fn abce_builder_build_atomic_add(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
    value: u64,
) {
    unsafe {
        abce_builder_build_atomic(builder, cmd_addr, addr, 47u32, value);
    }
}

pub unsafe extern "C" fn abce_builder_build_atomic_decrement(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    addr: *mut core::ffi::c_void,
) {
    unsafe {
        abce_builder_build_atomic_add(builder, cmd_addr, addr, u64::MAX);
    }
}

pub unsafe extern "C" fn abce_builder_build_get_global_timestamp(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    write_address: *mut core::ffi::c_void,
) {
    unsafe {
        let pkt: *mut SDMA_PKT_TIMESTAMP = (cmd_addr).cast::<SDMA_PKT_TIMESTAMP>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0xd_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !(0xffu32 << 8)) | (0x2_u32 << 8);
        };
        if (*(builder)).scope_fields {
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0x3u32 << 24)) | (0x3_u32 << 24);
            };
        }
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_LO_UNION).bits).word);
            *word = abce_ptr_low32((write_address).cast_const());
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).ADDR_HI_UNION).bits).word);
            *word = abce_ptr_high32((write_address).cast_const());
        };
    }
}

pub unsafe extern "C" fn abce_builder_build_trap(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    event_id: u32,
) {
    unsafe {
        let _ = builder;
        let pkt: *mut SDMA_PKT_TRAP = (cmd_addr).cast::<SDMA_PKT_TRAP>();
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
            *word = (*word & !0xffu32) | 0x6_u32;
        };
        {
            let word = core::ptr::addr_of_mut!((((*(pkt)).INT_CONTEXT_UNION).bits).word);
            *word = (*word & !0x0fff_ffff_u32) | ((event_id) & 0x0fff_ffff);
        };
    }
}

pub unsafe extern "C" fn abce_builder_build_hdp_flush(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
) {
    unsafe {
        let _ = builder;
        {};
        let _ = libc::memcpy(
            (cmd_addr).cast::<core::ffi::c_void>(),
            core::ptr::addr_of!(abce_sdma_hdp_flush_cmd).cast::<core::ffi::c_void>(),
            core::mem::size_of::<SDMA_PKT_HDP_FLUSH>() as usize,
        );
    }
}

pub unsafe extern "C" fn abce_builder_build_gcr(
    builder: *const abce_builder_t,
    cmd_addr: *mut core::ffi::c_char,
    invalidate: bool,
) {
    unsafe {
        {};
        if (*(builder)).is_gfx125plus {
            let pkt: *mut SDMA_PKT_GCR_GFX125PLUS = (cmd_addr).cast::<SDMA_PKT_GCR_GFX125PLUS>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | 0x11_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0x1_u32 << 8);
            };
            if invalidate {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD3_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 4)) | (0x1_u32 << 4);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD3_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 14)) | (0x1_u32 << 14);
                };
            } else {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD3_UNION).bits).word);
                    *word = (*word & !(0x3u32 << 4)) | (0x1_u32 << 4);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD3_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 15)) | (0x1_u32 << 15);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).WORD3_UNION).bits).word);
                *word &= !(0x3u32 << 11);
            };
        } else {
            let pkt: *mut SDMA_PKT_GCR = (cmd_addr).cast::<SDMA_PKT_GCR>();
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !0xffu32) | 0x11_u32;
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).HEADER_UNION).bits).word);
                *word = (*word & !(0xffu32 << 8)) | (0x1_u32 << 8);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                *word = (*word & !(0x1u32 << 31)) | (0x1_u32 << 31);
            };
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                *word = (*word & !(0x1u32 << 22)) | (0x1_u32 << 22);
            };
            if invalidate {
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 30)) | (0x1_u32 << 30);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 25)) | (0x1_u32 << 25);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 24)) | (0x1_u32 << 24);
                };
                {
                    let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                    *word = (*word & !(0x1u32 << 23)) | (0x1_u32 << 23);
                };
            }
            {
                let word = core::ptr::addr_of_mut!((((*(pkt)).WORD2_UNION).bits).word);
                *word &= !(0x3u32 << 27);
            };
        }
    }
}
