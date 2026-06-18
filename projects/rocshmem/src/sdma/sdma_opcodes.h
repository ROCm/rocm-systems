/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDMA opcode and sub-opcode constants shared across all
 * packet-structure headers.  Top-level opcodes are common to
 * every SDMA generation; OSS7.0-specific sub-opcodes are always
 * defined but only used when SDMA_IS_OSS7 is set.
 *
 * SDMA_IS_OSS7 is detected at compile time from the HIP architecture
 * macro __gfx950__ (MI350X / CDNA4 and later OSS7 parts).
 */

#pragma once

/* ---- OSS7 detection: set by HIP compiler for each device arch ---- */
#if defined(__gfx950__)
#  define SDMA_IS_OSS7 1
#else
#  define SDMA_IS_OSS7 0
#endif

/* ---- Top-level opcodes (all SDMA generations) ----------- */
const unsigned int SDMA_OP_NOP = 0;
const unsigned int SDMA_OP_COPY = 1;
const unsigned int SDMA_OP_WRITE = 2;

const unsigned int SDMA_OP_FENCE = 5;
const unsigned int SDMA_OP_TRAP = 6;
const unsigned int SDMA_OP_POLL_REGMEM = 8;
const unsigned int SDMA_OP_TIMESTAMP = 13;
const unsigned int SDMA_OP_ATOMIC = 10;
const unsigned int SDMA_OP_CONST_FILL = 11;

/* ---- Pre-OSS7 sub-opcodes ------------------------------- */
const unsigned int SDMA_SUBOP_COPY_LINEAR = 0;
const unsigned int SDMA_SUBOP_COPY_LINEAR_SUB_WINDOW = 36;

const unsigned int SDMA_SUBOP_WRITE_LINEAR = 0;
const unsigned int SDMA_ATOMIC_ADD64 = 47;

/* ---- OSS7.0 (MI4 / MI350X+) sub-opcodes and operations -- */
const unsigned int SDMA_SIGNAL_OP_ADD64_MI4 = 111;
const unsigned int SDMA_WAIT_FUNC_GEQ_MI4 = 5;
const unsigned int SDMA_SUBOP_COPY_LINEAR_PHY_MI4 = 0x8;
const unsigned int SDMA_SUBOP_COPY_PAGE_TRANSFER_MI4 = 0xc;
const unsigned int SDMA_SUBOP_COPY_LINEAR_MULTICAST_MI4 = 0xa;
const unsigned int SDMA_SUBOP_COPY_LINEAR_WAIT_SIGNAL_MI4 = 0x0;
const unsigned int SDMA_SUBOP_COPY_MULTICAST_WAIT_SIGNAL_MI4 = 0xa;
const unsigned int SDMA_SUBOP_COPY_SWAP_WAIT_SIGNAL_MI4 = 0x9;
const unsigned int SDMA_SUBOP_WRITE_LINEAR_MI4 = 0x0;
const unsigned int SDMA_SUBOP_FENCE_MI4 = 0x0;
const unsigned int SDMA_SUBOP_FENCE_COND_INT_MI4 = 0x1;
const unsigned int SDMA_SUBOP_FENCE_64B_MI4 = 0x2;
const unsigned int SDMA_SUBOP_POLL_MEM_64B_MI4 = 0x5;
const unsigned int SDMA_SUBOP_CONSTANT_FILL_MI4 = 0x0;
const unsigned int SDMA_SUBOP_CONSTANT_FILL_PAGE_MI4 = 0x4;
