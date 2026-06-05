////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
////////////////////////////////////////////////////////////////////////////////////

// SwapCopy Shader
// Exchanges contents of two buffers A and B using compute shader.
// After execution: A contains original B, B contains original A.
// All threads in workgroup participate in parallel swap.
//
// Algorithm:
//   1. Vectorized phase: Each thread swaps 16 bytes at global_id*16
//      - Bounds checked against swap_size & ~0xF
//   2. Tail phase: Threads with global_id < (swap_size % 16) swap 1 byte each
//      - Handles non-16-byte-aligned sizes
//
// Kernel Arguments (20 bytes):
//   [DW 0-1]  Address A (64-bit)
//   [DW 2-3]  Address B (64-bit)
//   [DW 4  ]  Swap size in bytes (32-bit)
//
// Register allocation:
//   s[0:1]  = kernarg_segment_ptr
//   s2      = workgroup_id (GFX9) / unused (GFX12 uses ttmp9)
//   s[4:5]  = addr_a
//   s[6:7]  = addr_b
//   s8      = swap_size
//   s9      = num_workitems (64 per workgroup) - unused
//   s10     = stride - unused in single-iteration version
//
//   v0      = global thread offset (global_id)
//   v1      = saved global_id
//   v[2:3]  = vectorized: addr_a + global_id*16 / tail: byte data
//   v[4:5]  = vectorized: addr_b + global_id*16
//   v[6:7]  = vectorized: end address / tail: addr_a tail pointer
//   v[8:9]  = tail: addr_b tail pointer
//   v[8:11] = vectorized: data from buffer A (16 bytes)
//   v[10:11]= tail: end address
//   v[12:15]= vectorized: data from buffer B (16 bytes)

.text

.macro V_ADD_CO_U32 vdst, src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
     v_add_co_u32        \vdst, vcc_lo, \src0, \vsrc1
  .elseif (.amdgcn.gfx_generation_number >= 9)
    v_add_co_u32        \vdst, vcc, \src0, \vsrc1
  .else
    v_add_u32           \vdst, vcc, \src0, \vsrc1
  .endif
.endm

.macro V_ADD_CO_CI_U32 vdst, src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    v_add_co_ci_u32     \vdst, vcc_lo, \src0, \vsrc1, vcc_lo
  .elseif (.amdgcn.gfx_generation_number >= 9)
    v_addc_co_u32       \vdst, vcc, \src0, \vsrc1, vcc
  .else
    v_addc_u32          \vdst, vcc, \src0, \vsrc1, vcc
  .endif
.endm

.macro V_CMP_LT_U64 src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    v_cmp_lt_u64        vcc_lo, \src0, \vsrc1
  .else
    v_cmp_lt_u64        vcc, \src0, \vsrc1
  .endif
.endm

.macro S_AND_B64 sdst, src0, ssrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    s_and_b32           \sdst, \src0, \ssrc1
  .else
    s_and_b64           \sdst, \src0, \ssrc1
  .endif
.endm

// Non-carry add - used when we don't need carry propagation
.macro V_ADD_NC_U32 vdst, src0, vsrc1
  .if (.amdgcn.gfx_generation_number >= 10)
    v_add_nc_u32        \vdst, \src0, \vsrc1
  .else
    v_add_u32           \vdst, \src0, \vsrc1
  .endif
.endm

// GFX12.5 (gfx1250) uses new wait instructions
.macro S_WAITCNT_KMCNT
  .if (.amdgcn.gfx_generation_number == 12 && .amdgcn.gfx_generation_minor >= 5)
    s_wait_kmcnt        0
  .else
    s_waitcnt           lgkmcnt(0)
  .endif
.endm

.macro S_WAITCNT_LOADCNT
  .if (.amdgcn.gfx_generation_number == 12 && .amdgcn.gfx_generation_minor >= 5)
    s_wait_loadcnt      0
  .else
    s_waitcnt           vmcnt(0)
  .endif
.endm

// GFX12.5 uses s_wait_storecnt instead of s_waitcnt expcnt(0)
.macro S_WAITCNT_STORECNT
  .if (.amdgcn.gfx_generation_number == 12 && .amdgcn.gfx_generation_minor >= 5)
    s_wait_storecnt     0
  .else
    s_waitcnt           expcnt(0)
  .endif
.endm

.set kSwapCopyVecWidth, 4
.set kSwapCopyUnroll, 2

.set kSwapCopyNumSGPRs, 16
.set kSwapCopyNumVGPRs, 16

.set SwapCopyRsrc1SGPRs, (kSwapCopyNumSGPRs - 1) / 8
  .if SwapCopyRsrc1SGPRs < 0
    .set SwapCopyRsrc1SGPRs, 0
  .endif

.set SwapCopyRsrc1VGPRs, (kSwapCopyNumVGPRs - 1) / 4
  .if SwapCopyRsrc1VGPRs < 0
    .set SwapCopyRsrc1VGPRs, 0
  .endif

.p2align 8

SwapCopy:

    compute_pgm_rsrc1_sgprs = SwapCopyRsrc1SGPRs
    compute_pgm_rsrc1_vgprs = SwapCopyRsrc1VGPRs
    compute_pgm_rsrc2_user_sgpr = 2
    compute_pgm_rsrc2_tgid_x_en = 1
    enable_sgpr_kernarg_segment_ptr = 1

    // Load kernel arguments
    // s[4:5] = addr_a, s[6:7] = addr_b, s8 = swap_size
    s_load_dwordx4      s[4:7], s[0:1], 0x0
    s_load_dword        s8, s[0:1], 0x10
    S_WAITCNT_KMCNT

    // Compute global thread ID = workgroup_id * 64 + local_id
    // TTMP9 register usage per architecture:
    //   GFX9:   s2 via compute_pgm_rsrc2_tgid_x_en (ttmp9 has dispatch_grid_y)
    //   GFX10+: ttmp9 has workgroup_x (SPI initialized)
    .if (.amdgcn.gfx_generation_number >= 10)
      s_lshl_b32        s2, ttmp9, 0x6      // s2 = workgroup_id * 64
    .else
      s_lshl_b32        s2, s2, 0x6         // s2 = workgroup_id * 64
    .endif
    V_ADD_CO_U32        v0, s2, v0          // v0 = global_id

    // Save global_id for later use
    v_mov_b32           v1, v0

    // s9 = workgroup size (64 threads) - unused now
    s_mov_b32           s9, 64

    // s10 = stride (unused in single-iteration version)
    s_lshl_b32          s10, s9, 4

    // v[2:3] = addr_a + global_id*16
    v_lshlrev_b32       v2, 4, v0
    v_mov_b32           v3, s5
    V_ADD_CO_U32        v2, v2, s4
    V_ADD_CO_CI_U32     v3, v3, 0x0

    // v[4:5] = addr_b + global_id*16
    v_lshlrev_b32       v4, 4, v0
    v_mov_b32           v5, s7
    V_ADD_CO_U32        v4, v4, s6
    V_ADD_CO_CI_U32     v5, v5, 0x0

    // Compute end address for bounds checking: v[6:7] = addr_a + (swap_size & ~0xF)
    // Align swap_size down to 16 bytes since we operate on 16-byte chunks
    v_and_b32           v6, 0xFFFFFFF0, s8
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v6, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // Bounds check: skip vectorized swap if v[2:3] >= v[6:7]
    // This handles dispatch grid rounding (grid rounded up to multiple of 64)
    V_CMP_LT_U64        v[2:3], v[6:7]
    s_cbranch_vccz      L_SWAP_VEC_DONE

    // Mask off threads that are out of bounds for vectorized phase
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_saveexec_b32 s3, vcc_lo
    .else
      s_and_saveexec_b64 s[2:3], vcc
    .endif

    // Each thread swaps its 16-byte chunk
    flat_load_dwordx4   v[8:11], v[2:3]
    flat_load_dwordx4   v[12:15], v[4:5]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[2:3], v[12:15]
    flat_store_dwordx4  v[4:5], v[8:11]
    S_WAITCNT_STORECNT

    // Restore EXEC for tail phase
    .if (.amdgcn.gfx_generation_number >= 10)
      s_mov_b32         exec_lo, s3
    .else
      s_mov_b64         exec, s[2:3]
    .endif

L_SWAP_VEC_DONE:
    // Tail Loop: Byte-by-byte swap for remaining bytes (swap_size % 16)
    // Only threads with local_id < (swap_size % 16) participate
    // Use saved global_id (v1) and v0 for local_id

    // v[6:7] = addr_a + (swap_size & ~0xF) + local_id (reuse v6:7)
    v_and_b32           v6, 0xFFFFFFF0, s8       // aligned_size
    V_ADD_NC_U32        v6, v6, v1               // add global_id (v1 saved earlier)
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v6, s4               // v6:7 = addr_a + aligned_size + global_id
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // v[8:9] = addr_b + (swap_size & ~0xF) + local_id
    v_and_b32           v8, 0xFFFFFFF0, s8       // aligned_size
    V_ADD_NC_U32        v8, v8, v1               // add global_id
    v_mov_b32           v9, s7
    V_ADD_CO_U32        v8, v8, s6               // v8:9 = addr_b + aligned_size + global_id
    V_ADD_CO_CI_U32     v9, v9, 0x0

    // v[10:11] = addr_a + swap_size (end address for tail)
    v_mov_b32           v10, s8                  // swap_size
    v_mov_b32           v11, s5
    V_ADD_CO_U32        v10, v10, s4
    V_ADD_CO_CI_U32     v11, v11, 0x0

    // Only process if within tail region: v[6:7] < v[10:11]
    V_CMP_LT_U64        v[6:7], v[10:11]
    s_cbranch_vccz      L_SWAP_DONE

    // Mask off threads outside tail region
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         exec_lo, exec_lo, vcc_lo
    .else
      s_and_b64         exec, exec, vcc
    .endif

    // Byte swap: load from both, store swapped
    // v2 = byte from A, v3 = byte from B (reusing registers)
    flat_load_ubyte     v2, v[6:7]
    flat_load_ubyte     v3, v[8:9]
    S_WAITCNT_LOADCNT
    flat_store_byte     v[6:7], v3
    flat_store_byte     v[8:9], v2
    S_WAITCNT_STORECNT

L_SWAP_DONE:
    s_endpgm
