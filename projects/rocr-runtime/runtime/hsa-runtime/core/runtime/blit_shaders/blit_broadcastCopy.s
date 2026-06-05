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

// BroadcastCopy Shader
// Copies a single source buffer to multiple destinations using compute shader.
// Each workgroup handles one destination, all threads in workgroup copy in parallel.
//
// Kernel Arguments (24 bytes):
//   [DW 0-1]  Source address (64-bit)
//   [DW 2-3]  Destination list address (64-bit) - array of destination pointers
//   [DW 4  ]  Number of destinations (32-bit)
//   [DW 5  ]  Copy size per destination (32-bit)
//   [DW 6  ]  Total number of workitems (32-bit)

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

.set kBroadcastCopyVecWidth, 4
.set kBroadcastCopyUnroll, 4

.set kBroadcastCopyNumSGPRs, 24

// VGPR allocation differs by architecture:
// - GFX9: Uses v16 for local_id to avoid clobber by flat_load_dwordx4 v[12:15]
//   due to different instruction execution ordering. Requires 20 VGPRs.
// - GFX10+: Uses v14 for local_id since it's consumed before flat_load_dwordx4.
//   Requires 16 VGPRs.
.if (.amdgcn.gfx_generation_number >= 10)
  .set kBroadcastCopyNumVGPRs, 16
  .set LOCAL_ID_VREG, 14
.else
  .set kBroadcastCopyNumVGPRs, 20
  .set LOCAL_ID_VREG, 16
.endif

.set BroadcastCopyRsrc1SGPRs, (kBroadcastCopyNumSGPRs - 1) / 8
  .if BroadcastCopyRsrc1SGPRs < 0
    .set BroadcastCopyRsrc1SGPRs, 0
  .endif

.set BroadcastCopyRsrc1VGPRs, (kBroadcastCopyNumVGPRs - 1) / 4
  .if BroadcastCopyRsrc1VGPRs < 0
    .set BroadcastCopyRsrc1VGPRs, 0
  .endif

.p2align 8

BroadcastCopy:

    compute_pgm_rsrc1_sgprs = BroadcastCopyRsrc1SGPRs
    compute_pgm_rsrc1_vgprs = BroadcastCopyRsrc1VGPRs
    compute_pgm_rsrc2_user_sgpr = 2
    compute_pgm_rsrc2_tgid_x_en = 1
    enable_sgpr_kernarg_segment_ptr = 1

    // Load kernel arguments
    // s[4:5] = src_addr, s[6:7] = dst_list_addr, s[8:9] = {num_destinations, copy_size}, s10 = num_workitems
    s_load_dwordx4      s[4:7], s[0:1], 0x0
    s_load_dwordx2      s[8:9], s[0:1], 0x10
    s_load_dword        s10, s[0:1], 0x18
    S_WAITCNT_KMCNT

    // Save local work-item ID (v0 at kernel entry contains local_id)
    // Register selection is architecture-dependent:
    // - GFX9: Use v16 to avoid clobbering by flat_load_dwordx4 v[12:15]
    // - GFX10+: Use v14 (consumed before flat_load_dwordx4 executes)
    .if (.amdgcn.gfx_generation_number >= 10)
      v_mov_b32         v14, v0
    .else
      v_mov_b32         v16, v0
    .endif

    // Get workgroup ID (destination index)
    // TTMP9 register usage per architecture:
    //   GFX9:   s2 via compute_pgm_rsrc2_tgid_x_en (ttmp9 has dispatch_grid_y)
    //   GFX10+: ttmp9 has workgroup_x (SPI initialized)
    .if (.amdgcn.gfx_generation_number >= 10)
      v_mov_b32         v0, ttmp9
    .else
      v_mov_b32         v0, s2
    .endif

    // Load destination address from dst_list[workgroup_id]
    // v1 = workgroup_id * 8 (each address is 8 bytes)
    v_lshlrev_b32       v1, 3, v0
    v_mov_b32           v3, s7
    V_ADD_CO_U32        v2, v1, s6
    V_ADD_CO_CI_U32     v3, v3, 0x0

    flat_load_dwordx2   v[4:5], v[2:3]
    S_WAITCNT_LOADCNT

    // For broadcast: each workgroup copies FULL data to its destination
    // Use local_id (v14 on GFX10+, v16 on GFX9) for src/dst addressing
    // This ensures all workgroups read the same source data

    // Main Loop: 16-byte vectorized copy
    // s12 = 64 * 16 = 1024 (stride for workgroup, not total workitems)
    // s13 = 3 * s12 = 3072 (for unroll bounds adjustment)
    s_mov_b32           s12, 64 * 16
    s_mov_b32           s13, 64 * 16 * 3

    // v[6:7] = src_addr + local_id * 16
    .if (.amdgcn.gfx_generation_number >= 10)
      v_lshlrev_b32     v1, 4, v14
    .else
      v_lshlrev_b32     v1, 4, v16
    .endif
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v1, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // v[8:9] = dst_addr + local_id * 16
    V_ADD_CO_U32        v8, v1, v4
    V_ADD_CO_CI_U32     v9, v5, 0x0

    // v[10:11] = src_addr + (copy_size & ~0xF) (end of vectorized region)
    v_and_b32           v10, 0xFFFFFFF0, s9
    v_mov_b32           v11, s5
    V_ADD_CO_U32        v10, v10, s4
    V_ADD_CO_CI_U32     v11, v11, 0x0

    // Phase 1: Unrolled loop (4 iterations per check)
    // Adjusted end: v[10:11] - 3*stride ensures 4 iterations are always safe
    // We compare against (end - 3*stride) so when ptr < adjusted_end,
    // ptr, ptr+stride, ptr+2*stride, ptr+3*stride are all < end

    // Compute adjusted_end = end - 3*stride in v[0:1] (must be 64-bit aligned for gfx1250)
    // Use v_sub for 64-bit subtraction
    v_mov_b32           v0, s13             // v0 = 3*stride
    v_mov_b32           v1, 0
    // v[0:1] = v[10:11] - v[0:1] (end - 3*stride)
    .if (.amdgcn.gfx_generation_number >= 10)
      v_sub_co_u32      v0, vcc_lo, v10, v0
      v_sub_co_ci_u32   v1, vcc_lo, v11, v1, vcc_lo
    .elseif (.amdgcn.gfx_generation_number >= 9)
      v_sub_co_u32      v0, vcc, v10, v0
      v_subb_co_u32     v1, vcc, v11, v1, vcc
    .else
      v_sub_u32         v0, vcc, v10, v0
      v_subb_u32        v1, vcc, v11, v1, vcc
    .endif

L_BROADCAST_VEC_UNROLL:
    // Check if 4 full iterations fit: ptr < end - 3*stride
    V_CMP_LT_U64        v[6:7], v[0:1]
    s_cbranch_vccz      L_BROADCAST_VEC_SINGLE

    // Unrolled vectorized copy (4 iterations, bounds guaranteed safe)
    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_VEC_UNROLL

    // Phase 2: Single-iteration cleanup (handles remaining 0-3 iterations)
L_BROADCAST_VEC_SINGLE:
    V_CMP_LT_U64        v[6:7], v[10:11]
    s_cbranch_vccz      L_BROADCAST_VEC_DONE

    flat_load_dwordx4   v[12:15], v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[8:9], v[12:15]

    V_ADD_CO_U32        v6, v6, s12
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, s12
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_VEC_SINGLE

L_BROADCAST_VEC_DONE:

    // Tail Loop: Byte-by-byte copy for remaining bytes after vectorized portion
    // Start at: (copy_size & ~0xF) + local_id
    // v1 = copy_size & ~0xF (start of tail region)
    v_and_b32           v1, 0xFFFFFFF0, s9

    // v[6:7] = src_addr + (copy_size & ~0xF) + local_id
    .if (.amdgcn.gfx_generation_number >= 10)
      V_ADD_NC_U32      v1, v1, v14      // v1 = aligned_size + local_id
    .else
      V_ADD_NC_U32      v1, v1, v16      // v1 = aligned_size + local_id
    .endif
    v_mov_b32           v7, s5
    V_ADD_CO_U32        v6, v1, s4
    V_ADD_CO_CI_U32     v7, v7, 0x0

    // v[8:9] = dst_addr + (copy_size & ~0xF) + local_id
    V_ADD_CO_U32        v8, v1, v4
    V_ADD_CO_CI_U32     v9, v5, 0x0

    // v[10:11] = src_addr + copy_size (end address)
    v_mov_b32           v10, s9
    v_mov_b32           v11, s5
    V_ADD_CO_U32        v10, v10, s4
    V_ADD_CO_CI_U32     v11, v11, 0x0

L_BROADCAST_BYTE_LOOP:
    V_CMP_LT_U64        v[6:7], v[10:11]
    s_cbranch_vccz      L_BROADCAST_BYTE_DONE
    .if (.amdgcn.gfx_generation_number >= 10)
      s_and_b32         exec_lo, exec_lo, vcc_lo
    .else
      s_and_b64         exec, exec, vcc
    .endif

    flat_load_ubyte     v1, v[6:7]
    S_WAITCNT_LOADCNT
    flat_store_byte     v[8:9], v1

    // Stride by 64 (workgroup size), not total workitems
    V_ADD_CO_U32        v6, v6, 64
    V_ADD_CO_CI_U32     v7, v7, 0x0
    V_ADD_CO_U32        v8, v8, 64
    V_ADD_CO_CI_U32     v9, v9, 0x0

    s_branch            L_BROADCAST_BYTE_LOOP

L_BROADCAST_BYTE_DONE:
    s_endpgm
