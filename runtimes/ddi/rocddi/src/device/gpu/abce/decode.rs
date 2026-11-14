// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE decode mechanisms.

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

use super::abi::{abce_decoded_packet_t, abce_isa_version_t};

#[must_use]
pub unsafe extern "C" fn abce_decoded_packet_make(
    name: *const core::ffi::c_char,
    dwords: u32,
    op: u32,
    sub_op: u32,
) -> abce_decoded_packet_t {
    {
        let mut packet: abce_decoded_packet_t = unsafe { core::mem::zeroed() };
        (packet).name = name;
        (packet).dwords = dwords;
        (packet).op = op;
        (packet).sub_op = sub_op;
        packet
    }
}

#[must_use]
pub unsafe extern "C" fn abce_decode_sdma_packet(
    words: *const u32,
    remaining_dwords: u32,
    isa: abce_isa_version_t,
) -> abce_decoded_packet_t {
    unsafe {
        if (words).is_null() || ((remaining_dwords) == (0_u32)) {
            return abce_decoded_packet_make(c"UNKNOWN".as_ptr(), 1_u32, 0_u32, 0_u32);
        }
        let header: u32 = *(words).wrapping_offset((0i32) as isize);
        let op: u32 = header & 255u32;
        let sub_op: u32 = ((header).wrapping_shr(8_u32)) & 255u32;
        let gfx125plus: bool = (((isa).major) == (12_u32)) && (((isa).minor) >= (5_u32));
        let mut packet: abce_decoded_packet_t =
            abce_decoded_packet_make(c"UNKNOWN".as_ptr(), 1_u32, op, sub_op);
        if ((op) == (0u32)) && ((sub_op) == (0_u32)) {
            packet = abce_decoded_packet_make(
                c"NOP".as_ptr(),
                (((header).wrapping_shr(16u32)) & 16_383_u32).wrapping_add(1u32),
                op,
                sub_op,
            );
        } else if (op) == (1u32) {
            let wait: bool = (header & ((1u32).wrapping_shl(30_u32))) != (0_u32);
            let signal: bool = (header & ((1u32).wrapping_shl(31_u32))) != (0_u32);
            if (sub_op) == (4u32) {
                packet = abce_decoded_packet_make(c"COPY_LINEAR_RECT".as_ptr(), 13_u32, op, sub_op);
            } else if (sub_op) == (9u32) {
                packet = abce_decoded_packet_make(
                    (if gfx125plus && (wait || signal) {
                        c"COPY_LINEAR_SWAP_WAITSIGNAL".as_ptr().cast_mut()
                    } else {
                        c"COPY_LINEAR_SWAP".as_ptr().cast_mut()
                    })
                    .cast_const(),
                    ((7_u32).wrapping_add(if gfx125plus && wait { 7u32 } else { 0u32 }))
                        .wrapping_add(if gfx125plus && signal { 5u32 } else { 0u32 }),
                    op,
                    sub_op,
                );
            } else if (sub_op) == (10u32) {
                let parameter_dword: u32 = if gfx125plus && wait { 9u32 } else { 2u32 };
                let num_destinations: u32 = (if (remaining_dwords) > (parameter_dword) {
                    (*(words).wrapping_offset((parameter_dword) as isize)) & 1023u32
                } else {
                    0_u32
                })
                .wrapping_add(1_u32);
                packet = abce_decoded_packet_make(
                    (if gfx125plus && (wait || signal) {
                        c"COPY_LINEAR_MULTICAST_WAITSIGNAL".as_ptr().cast_mut()
                    } else {
                        c"COPY_LINEAR_MULTICAST".as_ptr().cast_mut()
                    })
                    .cast_const(),
                    (((5_u32).wrapping_add((2_u32).wrapping_mul(num_destinations)))
                        .wrapping_add(if gfx125plus && wait { 7u32 } else { 0u32 }))
                    .wrapping_add(if gfx125plus && signal { 5u32 } else { 0u32 }),
                    op,
                    sub_op,
                );
            } else if (sub_op) == (0u32) {
                let indirect: bool = (header
                    & (((1u32).wrapping_shl(20_u32)) | ((1u32).wrapping_shl(21_u32))))
                    != (0_u32);
                let broadcast: bool = (header & ((1u32).wrapping_shl(27_u32))) != (0_u32);
                if gfx125plus && (wait || signal) {
                    packet = abce_decoded_packet_make(
                        (if indirect {
                            c"COPY_LINEAR_WAITSIGNAL_INDIRECT".as_ptr().cast_mut()
                        } else {
                            c"COPY_LINEAR_WAITSIGNAL".as_ptr().cast_mut()
                        })
                        .cast_const(),
                        ((7_u32).wrapping_add(if wait { 7u32 } else { 0u32 }))
                            .wrapping_add(if signal { 5u32 } else { 0u32 }),
                        op,
                        sub_op,
                    );
                } else if broadcast {
                    packet = abce_decoded_packet_make(
                        c"COPY_LINEAR_BROADCAST".as_ptr(),
                        9_u32,
                        op,
                        sub_op,
                    );
                } else {
                    packet = abce_decoded_packet_make(
                        (if indirect {
                            c"COPY_LINEAR_INDIRECT".as_ptr().cast_mut()
                        } else {
                            c"COPY_LINEAR".as_ptr().cast_mut()
                        })
                        .cast_const(),
                        7_u32,
                        op,
                        sub_op,
                    );
                }
            }
        } else if ((op) == (2u32)) && ((sub_op) == (0u32)) {
            packet = abce_decoded_packet_make(
                c"WRITE_LINEAR".as_ptr(),
                if (remaining_dwords) > (3_u32) {
                    ((4u32)
                        .wrapping_add((*(words).wrapping_offset((3i32) as isize)) & 1_048_575_u32))
                    .wrapping_add(1u32)
                } else {
                    4u32
                },
                op,
                sub_op,
            );
        } else if (op) == (5u32) {
            packet = abce_decoded_packet_make(
                (if (sub_op) == (2u32) {
                    c"FENCE_64".as_ptr().cast_mut()
                } else {
                    c"FENCE".as_ptr().cast_mut()
                })
                .cast_const(),
                if (sub_op) == (2u32) { 5u32 } else { 4u32 },
                op,
                sub_op,
            );
        } else if (op) == (6u32) {
            packet = abce_decoded_packet_make(c"TRAP".as_ptr(), 2_u32, op, sub_op);
        } else if (op) == (8u32) {
            let hdp_flush: bool = (((((((remaining_dwords) >= (6_u32))
                && ((*(words).wrapping_offset((0i32) as isize)) == (8u32)))
                && ((*(words).wrapping_offset((1i32) as isize)) == (0_u32)))
                && ((*(words).wrapping_offset((2i32) as isize)) == (2_147_483_648_u32)))
                && ((*(words).wrapping_offset((3i32) as isize)) == (0_u32)))
                && ((*(words).wrapping_offset((4i32) as isize)) == (0_u32)))
                && ((*(words).wrapping_offset((5i32) as isize)) == (0_u32));
            packet = abce_decoded_packet_make(
                (if hdp_flush {
                    c"HDP_FLUSH".as_ptr().cast_mut()
                } else if (sub_op) == (5u32) {
                    c"POLL_MEM_64".as_ptr().cast_mut()
                } else {
                    c"POLL_REGMEM".as_ptr().cast_mut()
                })
                .cast_const(),
                if (sub_op) == (5u32) { 8u32 } else { 6u32 },
                op,
                sub_op,
            );
        } else if (op) == (10u32) {
            packet = abce_decoded_packet_make(c"ATOMIC".as_ptr(), 8_u32, op, sub_op);
        } else if (op) == (11u32) {
            packet = abce_decoded_packet_make(c"CONST_FILL".as_ptr(), 5_u32, op, sub_op);
        } else if (op) == (13u32) {
            packet = abce_decoded_packet_make(c"TIMESTAMP_GET_GLOBAL".as_ptr(), 3_u32, op, sub_op);
        } else if (op) == (17u32) {
            packet = abce_decoded_packet_make(
                c"GCR".as_ptr(),
                if gfx125plus { 6u32 } else { 5u32 },
                op,
                sub_op,
            );
        }
        if ((packet).dwords) > (remaining_dwords) {
            (packet).name = c"TRUNCATED".as_ptr();
            (packet).dwords = remaining_dwords;
        }
        packet
    }
}
