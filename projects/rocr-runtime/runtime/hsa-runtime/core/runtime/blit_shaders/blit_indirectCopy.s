////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

// IndirectCopy Shader
// Copies data between source and destination where either or both addresses
// are resolved via indirection (pointer-to-pointer). This enables GPU kernels
// to specify copy addresses dynamically without CPU intervention.
//
// Kernel Arguments (28 bytes):
//   [DW 0-1]  src_addr (64-bit) - Direct address or pointer to actual source
//   [DW 2-3]  dst_addr (64-bit) - Direct address or pointer to actual dest
//   [DW 4  ]  copy_size (32-bit) - Bytes to copy
//   [DW 5  ]  num_workitems (32-bit) - Total number of workitems
//   [DW 6  ]  indirect_flags (32-bit) - bit0: src indirect, bit1: dst indirect
//
// Register allocation:
//   s[0:1]  = kernarg_segment_ptr
//   s2      = workgroup_id (GFX9) / unused (GFX12 uses ttmp9)
//   s[4:5]  = src_addr (resolved)
//   s[6:7]  = dst_addr (resolved)
//   s8      = copy_size
//   s9      = num_workitems
//   s10     = indirect_flags
//   s11     = scratch
//   s12     = stride (num_workitems * 16)
//
//   v0      = global thread offset
//   v[2:3]  = current src pointer
//   v[4:5]  = current dst pointer
//   v[6:7]  = end address (src + aligned_size)
//   v[8:11] = data (16 bytes for vectorized copy)
//   v1, v12 = scratch

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

// Non-carry add (doesn't use/modify VCC)
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

.set kIndirectCopyVecWidth, 4
.set kIndirectCopyUnroll, 4

.set kIndirectCopyNumSGPRs, 16
.set kIndirectCopyNumVGPRs, 16

.set IndirectCopyRsrc1SGPRs, (kIndirectCopyNumSGPRs - 1) / 8
  .if IndirectCopyRsrc1SGPRs < 0
    .set IndirectCopyRsrc1SGPRs, 0
  .endif

.set IndirectCopyRsrc1VGPRs, (kIndirectCopyNumVGPRs - 1) / 4
  .if IndirectCopyRsrc1VGPRs < 0
    .set IndirectCopyRsrc1VGPRs, 0
  .endif

.p2align 8

IndirectCopy:

    compute_pgm_rsrc1_sgprs = IndirectCopyRsrc1SGPRs
    compute_pgm_rsrc1_vgprs = IndirectCopyRsrc1VGPRs
    compute_pgm_rsrc2_user_sgpr = 2
    compute_pgm_rsrc2_tgid_x_en = 1
    enable_sgpr_kernarg_segment_ptr = 1

    // Load kernel arguments
    // s[4:5] = src_addr, s[6:7] = dst_addr, s8 = copy_size, s9 = num_workitems, s10 = flags
    s_load_dwordx4      s[4:7], s[0:1], 0x0
    s_load_dwordx2      s[8:9], s[0:1], 0x10
    s_load_dword        s10, s[0:1], 0x18
    S_WAITCNT_KMCNT

    // =====================================================
    // Phase 0: Resolve indirect addresses
    // =====================================================

    // Check if source is indirect (bit 0 of flags)
    s_and_b32           s11, s10, 1
    s_cbranch_scc0      L_SRC_DIRECT
    // Source is indirect - dereference s[4:5] to get actual source address
    s_load_dwordx2      s[4:5], s[4:5], 0x0
    S_WAITCNT_KMCNT
L_SRC_DIRECT:

    // Check if destination is indirect (bit 1 of flags)
    s_and_b32           s11, s10, 2
    s_cbranch_scc0      L_DST_DIRECT
    // Destination is indirect - dereference s[6:7] to get actual dest address
    s_load_dwordx2      s[6:7], s[6:7], 0x0
    S_WAITCNT_KMCNT
L_DST_DIRECT:

    // Now s[4:5] = actual src, s[6:7] = actual dst

    // =====================================================
    // Phase 1: Setup for copy
    // =====================================================

    // Compute workitem id (single workgroup dispatch)
    v_mov_b32           v0, v0

    // s12 = num_workitems * 16 (stride for vectorized loop)
    // s13 = 3 * s12 (for unroll bounds adjustment)
    s_lshl_b32          s12, s9, 4
    s_lshl_b32          s13, s9, 4
    s_mul_i32           s13, s13, 3

    // v[2:3] = src_addr + thread_offset * 16
    v_lshlrev_b32       v1, 4, v0
    v_mov_b32           v3, s5
    V_ADD_CO_U32        v2, v1, s4
    V_ADD_CO_CI_U32     v3, v3, 0x0

    // v[4:5] = dst_addr + thread_offset * 16
    v_mov_b32           v5, s7
    V_ADD_CO_U32        v4, v1, s6
    V_ADD_CO_CI_U32     v5, v5, 0x0

    // v[6:7] = src_addr + (copy_size & ~0xF) (end of vectorized region)
    v_and_b32           v6, 0xFFFFFFF0, s8
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v6, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // =====================================================
    // Phase 2a: Unrolled vectorized loop (4 iterations per check)
    // Adjusted end: v[6:7] - 3*stride ensures 4 iterations are safe
    // =====================================================

L_INDIRECT_VEC_UNROLL:
    // Compute adjusted_end = end - 3*stride in v[12:13] (must be 64-bit aligned for gfx1250)
    // Recompute each iteration since v[8:11] are clobbered by data loads
    v_mov_b32           v12, s13            // v12 = 3*stride
    v_mov_b32           v13, 0
    // v[12:13] = v[6:7] - v[12:13] (end - 3*stride)
    .if (.amdgcn.gfx_generation_number >= 10)
      v_sub_co_u32      v12, vcc_lo, v6, v12
      v_sub_co_ci_u32   v13, vcc_lo, v7, v13, vcc_lo
    .elseif (.amdgcn.gfx_generation_number >= 9)
      v_sub_co_u32      v12, vcc, v6, v12
      v_subb_co_u32     v13, vcc, v7, v13, vcc
    .else
      v_sub_u32         v12, vcc, v6, v12
      v_subb_u32        v13, vcc, v7, v13, vcc
    .endif

    // Check if 4 full iterations fit: ptr < end - 3*stride
    V_CMP_LT_U64        v[2:3], v[12:13]
    s_cbranch_vccz      L_INDIRECT_VEC_SINGLE

    // Unrolled vectorized copy (4 iterations, bounds guaranteed safe)
    flat_load_dwordx4   v[8:11], v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[4:5], v[8:11]

    V_ADD_CO_U32        v2, v2, s12
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s12
    V_ADD_CO_CI_U32     v5, v5, 0x0

    flat_load_dwordx4   v[8:11], v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[4:5], v[8:11]

    V_ADD_CO_U32        v2, v2, s12
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s12
    V_ADD_CO_CI_U32     v5, v5, 0x0

    flat_load_dwordx4   v[8:11], v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[4:5], v[8:11]

    V_ADD_CO_U32        v2, v2, s12
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s12
    V_ADD_CO_CI_U32     v5, v5, 0x0

    flat_load_dwordx4   v[8:11], v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[4:5], v[8:11]

    V_ADD_CO_U32        v2, v2, s12
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s12
    V_ADD_CO_CI_U32     v5, v5, 0x0

    s_branch            L_INDIRECT_VEC_UNROLL

    // =====================================================
    // Phase 2b: Single-iteration cleanup (handles remaining 0-3 iterations)
    // =====================================================

L_INDIRECT_VEC_SINGLE:
    V_CMP_LT_U64        v[2:3], v[6:7]
    s_cbranch_vccz      L_INDIRECT_VEC_DONE

    flat_load_dwordx4   v[8:11], v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[4:5], v[8:11]

    V_ADD_CO_U32        v2, v2, s12
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s12
    V_ADD_CO_CI_U32     v5, v5, 0x0

    s_branch            L_INDIRECT_VEC_SINGLE

L_INDIRECT_VEC_DONE:

    // =====================================================
    // Phase 3: Tail loop (byte-by-byte for remainder)
    // Start at: (copy_size & ~0xF) + thread_offset
    // This handles the remaining 0-15 bytes after vectorized phase
    // =====================================================

    // v1 = copy_size & ~0xF (aligned_size = start of tail region)
    v_and_b32           v1, 0xFFFFFFF0, s8

    // v1 = aligned_size + thread_offset
    V_ADD_NC_U32        v1, v1, v0

    // v[2:3] = src_addr + aligned_size + thread_offset
    v_mov_b32           v3, s5
    V_ADD_CO_U32        v2, v1, s4
    V_ADD_CO_CI_U32     v3, v3, 0x0

    // v[4:5] = dst_addr + aligned_size + thread_offset
    v_mov_b32           v5, s7
    V_ADD_CO_U32        v4, v1, s6
    V_ADD_CO_CI_U32     v5, v5, 0x0

    // v[6:7] = src_addr + copy_size (end address)
    v_mov_b32           v6, s8
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v6, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

L_INDIRECT_BYTE_LOOP:
    V_CMP_LT_U64        v[2:3], v[6:7]
    s_cbranch_vccz      L_INDIRECT_BYTE_DONE
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         exec_lo, exec_lo, vcc_lo
    .else
      s_and_b64         exec, exec, vcc
    .endif

    flat_load_ubyte     v1, v[2:3]
    S_WAITCNT_LOADCNT
    flat_store_byte     v[4:5], v1

    V_ADD_CO_U32        v2, v2, s9
    V_ADD_CO_CI_U32     v3, v3, 0x0
    V_ADD_CO_U32        v4, v4, s9
    V_ADD_CO_CI_U32     v5, v5, 0x0

    s_branch            L_INDIRECT_BYTE_LOOP

L_INDIRECT_BYTE_DONE:
    s_endpgm
