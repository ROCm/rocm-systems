////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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
//
///////////////////////////////////////////////////////////////////////////////////////

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

.p2align 8

CopyAligned:
.set kCopyAlignedVecWidth, 4
compute_pgm_rsrc2_user_sgpr = 2
compute_pgm_rsrc2_tgid_x_en = 1
enable_sgpr_kernarg_segment_ptr = 1

.set kCopyAlignedUnroll, 1
.set kCopyAlignedNumSGPRs, 32
.set kCopyAlignedNumVGPRs, (8 + (kCopyAlignedUnroll * kCopyAlignedVecWidth))
.set CopyAlignedRsrc1SGPRs, (kCopyAlignedNumSGPRs - 1)/8
.set CopyAlignedRsrc1VGPRs, (kCopyAlignedNumVGPRs - 1)/4

compute_pgm_rsrc1_sgprs = CopyAlignedRsrc1SGPRs
compute_pgm_rsrc1_vgprs = CopyAlignedRsrc1VGPRs


  s_load_dwordx4  s[4:7], s[0:1], 0x0
  s_load_dwordx4  s[8:11], s[0:1], 0x10
  s_load_dwordx4  s[12:15], s[0:1], 0x20
  s_load_dwordx4  s[16:19], s[0:1], 0x30
  s_load_dwordx4  s[20:23], s[0:1], 0x40
  s_load_dword    s24, s[0:1], 0x50
  s_waitcnt                lgkmcnt(0)

  .if (.amdgcn.gfx_generation_number == 12)
    s_lshl_b32              s2, ttmp9, 0x6
  .else
    s_lshl_b32              s2, s2, 0x6
  .endif

    V_ADD_CO_U32            v0, s2, v0

    v_mov_b32               v3, s5
    V_ADD_CO_U32            v2, v0, s4
    V_ADD_CO_CI_U32         v3, v3, 0x0


    v_mov_b32               v5, s7
    V_ADD_CO_U32            v4, v0, s6
    V_ADD_CO_CI_U32         v5, v5, 0x0

  L_COPY_ALIGNED_PHASE_1_LOOP:

    V_CMP_LT_U64            v[2:3], s[8:9]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_1_DONE
    s_and_b64               exec, exec, vcc


    FLAT_LOAD_UBYTE         v1, v[2:3]
    s_waitcnt               vmcnt(0)
    V_ADD_CO_U32            v2, v2, s24
    V_ADD_CO_CI_U32         v3, v3, 0x0


    FLAT_STORE_BYTE         v[4:5], v1
    V_ADD_CO_U32            v4, v4, s24
    V_ADD_CO_CI_U32         v5, v5, 0x0

    s_branch                L_COPY_ALIGNED_PHASE_1_LOOP

  L_COPY_ALIGNED_PHASE_1_DONE:

    s_mov_b64               exec, 0xFFFFFFFFFFFFFFFF

.if kCopyAlignedVecWidth == 4
      s_lshl_b32            s25, s24, 0x4
  .else
      s_lshl_b32            s25, s24, 0x2
  .endif

  .if kCopyAlignedVecWidth == 4
    v_lshlrev_b32          v1, 0x4, v0
  .else
    v_lshlrev_b32          v1, 0x2, v0
  .endif


    v_mov_b32               v3, s9
    V_ADD_CO_U32            v2, v1, s8
    V_ADD_CO_CI_U32         v3, v3, 0x0

    v_mov_b32               v5, s11
    V_ADD_CO_U32            v4, v1, s10
    V_ADD_CO_CI_U32         v5, v5, 0x0

  L_COPY_ALIGNED_PHASE_2_LOOP:

    V_CMP_LT_U64            v[2:3], s[12:13]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_2_DONE

.macro mCopyAlignedPhase2Load iter iter_end
    .if kCopyAlignedVecWidth == 4
      flat_load_dwordx4    v[8 + (\iter * 4):8 + (\iter * 4) + 3], v[2:3]
    .else
      flat_load_dword      v[8 + \iter], v[2:3]
    .endif

    V_ADD_CO_U32           v2, v2, s25
    V_ADD_CO_CI_U32        v3, v3, 0x0

    .if (\iter_end - \iter)
      mCopyAlignedPhase2Load (\iter + 1), \iter_end
    .endif
.endm

mCopyAlignedPhase2Load 0, (kCopyAlignedUnroll - 1)

  s_waitcnt                vmcnt(0)

.macro mCopyAlignedPhase2Store iter iter_end
    .if kCopyAlignedVecWidth == 4
      flat_store_dwordX4   v[4:5], v[8 + (\iter * 4):8 + (\iter * 4) + 3]
    .else
      flat_store_dword     v[4:5], v[8 + \iter]
    .endif

	V_ADD_CO_U32         v4, v4, s25
	V_ADD_CO_CI_U32      v5, v5, 0x0


    .if (\iter_end - \iter)
      mCopyAlignedPhase2Store (\iter + 1), \iter_end
    .endif
.endm

mCopyAlignedPhase2Store 0, (kCopyAlignedUnroll - 1)

  s_branch                L_COPY_ALIGNED_PHASE_2_LOOP

  L_COPY_ALIGNED_PHASE_2_DONE:

    s_lshl_b32              s25, s24, 0x2

    v_lshlrev_b32           v1, 0x2, v0
    v_mov_b32               v3, s13
    V_ADD_CO_U32            v2, v1, s12
    V_ADD_CO_CI_U32         v3, v3, 0x0

    v_mov_b32               v5, s15
    V_ADD_CO_U32            v4, v1, s14
    V_ADD_CO_CI_U32         v5, v5, 0x0

  L_COPY_ALIGNED_PHASE_3_LOOP:

    V_CMP_LT_U64            v[2:3], s[16:17]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_3_DONE
    s_and_b64               exec, exec, vcc


    FLAT_LOAD_DWORD         v1, v[2:3]
    V_ADD_CO_U32            v2, v2, s25
    V_ADD_CO_CI_U32         v3, v3, 0x0
    s_waitcnt               vmcnt(0)


    flat_store_dword        v[4:5], v1
    V_ADD_CO_U32            v4, v4, s25
    V_ADD_CO_CI_U32         v5, v5, 0x0

    s_branch                L_COPY_ALIGNED_PHASE_3_LOOP

  L_COPY_ALIGNED_PHASE_3_DONE:

    s_mov_b64               exec, 0xFFFFFFFFFFFFFFFF

    v_mov_b32               v3, s17
    V_ADD_CO_U32            v2, v0, s16
    V_ADD_CO_CI_U32         v3, v3, 0x0

    v_mov_b32               v5, s19
    V_ADD_CO_U32            v4, v0, s18
    V_ADD_CO_CI_U32         v5, v5, 0x0

    V_CMP_LT_U64            v[2:3], s[20:21]
    s_cbranch_vccz          L_COPY_ALIGNED_PHASE_4_DONE
    s_and_b64               exec, exec, vcc

    FLAT_LOAD_UBYTE         v1, v[2:3]
    s_waitcnt               vmcnt(0)

    FLAT_STORE_BYTE         v[4:5], v1

  L_COPY_ALIGNED_PHASE_4_DONE:
    s_endpgm

