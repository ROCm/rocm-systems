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
////////////////////////////////////////////////////////////////////////////////////

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

.set kFillVecWidth, 4
.set kFillUnroll, 1

.set kFillNumSGPRs, 13
.set kFillNumVGPRs, 4 + kFillUnroll

.set FillRsrc1SGPRs , (kFillNumSGPRs - 1) / 8
  .if FillRsrc1SGPRs  < 0
    .set FillRsrc1SGPRs , 0
  .endif

.set FillRsrc1VGPRs , (kFillNumVGPRs - 1) / 4
  .if FillRsrc1VGPRs  < 0
    .set FillRsrc1VGPRs , 0
  .endif

.p2align 8

Fill:

    compute_pgm_rsrc1_sgprs = FillRsrc1SGPRs
    compute_pgm_rsrc1_vgprs = FillRsrc1VGPRs
    compute_pgm_rsrc2_user_sgpr = 2
    compute_pgm_rsrc2_tgid_x_en = 1
    enable_sgpr_kernarg_segment_ptr = 1

    s_load_dwordx4  s[4:7], s[0:1], 0x0
    s_load_dwordx4  s[8:11], s[0:1], 0x10
    s_waitcnt       lgkmcnt(0)

   .if (.amdgcn.gfx_generation_number == 12)
     s_lshl_b32      s2, ttmp9, 0x6
   .else
     s_lshl_b32      s2, s2, 0x6
   .endif

    V_ADD_CO_U32     v0, s2, v0

.macro mFillPattern iter iter_end
    v_mov_b32              v[4 + \iter], s10

    .if (\iter_end - \iter)
      mFillPattern (\iter + 1), \iter_end
    .endif
  .endm

  mFillPattern 0, (kFillVecWidth - 1)

  .if kFillVecWidth == 4
      s_lshl_b32            s12, s11, 0x4
  .else
      s_lshl_b32            s12, s11, 0x2
  .endif


  .if kFillVecWidth == 4
    v_lshlrev_b32          v1, 0x4, v0
  .else
    v_lshlrev_b32          v1, 0x2, v0
  .endif

   v_mov_b32               v3, s5
   V_ADD_CO_U32            v2, v1, s4
   V_ADD_CO_CI_U32         v3, v3, 0x0

  L_FILL_PHASE_1_LOOP:

    V_CMP_LT_U64            v[2:3], s[6:7]
    s_cbranch_vccz          L_FILL_PHASE_1_DONE

.macro mFillPhase1 iter iter_end
    .if kFillVecWidth == 4
      flat_store_dwordx4   v[2:3], v[4:7]
    .else
      flat_store_dword     v[2:3], v4
    .endif

     V_ADD_CO_U32          v2, v2, s12
     V_ADD_CO_CI_U32       v3, v3, 0x0

    .if \iter < \iter_end
      mFillPhase1 (\iter + 1), \iter_end
    .endif
.endm

mFillPhase1 0, kFillUnroll - 1

  s_branch                L_FILL_PHASE_1_LOOP

  L_FILL_PHASE_1_DONE:

    s_lshl_b32              s12, s11, 0x2

    v_lshlrev_b32           v1, 0x2, v0
    v_mov_b32               v3, s7
    V_ADD_CO_U32            v2, v1, s6
    V_ADD_CO_CI_U32         v3, v3, 0x0

  L_FILL_PHASE_2_LOOP:

    V_CMP_LT_U64            v[2:3], s[8:9]
    s_cbranch_vccz          L_FILL_PHASE_2_DONE
    s_and_b64               exec, exec, vcc


    flat_store_dword        v[2:3], v4
    V_ADD_CO_U32            v2, v2, s12
    V_ADD_CO_CI_U32         v3, v3, 0x0

    s_branch                L_FILL_PHASE_2_LOOP

  L_FILL_PHASE_2_DONE:
    s_endpgm



