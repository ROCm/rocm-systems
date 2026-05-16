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
//   s9      = num_workitems (64 per workgroup)
//   s10     = stride (num_workitems * 16 for vectorized loop)
//
//   v0      = global thread offset
//   v[2:3]  = current addr_a pointer (64-bit aligned for gfx1250)
//   v[4:5]  = current addr_b pointer (64-bit aligned for gfx1250)
//   v[6:7]  = end address (addr_a + aligned_size) (64-bit aligned for gfx1250)
//   v[8:11] = data from buffer A (16 bytes) (64-bit aligned for gfx1250)
//   v[12:15]= data from buffer B (16 bytes) (64-bit aligned for gfx1250)
//   v1      = saved local_id

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
    //   GFX9 (MI300):  s2 via compute_pgm_rsrc2_tgid_x_en (ttmp9 has dispatch_grid_y)
    //   GFX10+:        ttmp9 has workgroup_x (SPI initialized)
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

    // Each thread swaps its 16-byte chunk
    flat_load_dwordx4   v[8:11], v[2:3]
    flat_load_dwordx4   v[12:15], v[4:5]
    S_WAITCNT_LOADCNT
    flat_store_dwordx4  v[2:3], v[12:15]
    flat_store_dwordx4  v[4:5], v[8:11]
    S_WAITCNT_STORECNT
    s_endpgm
