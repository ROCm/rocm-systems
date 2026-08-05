// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Native CDNA3/gfx942 Triton golden fixture generated from the same source and
// constexprs as the corresponding gfx950 DBT input fixture.
//
// Recorded fixture metadata:
//   kernel: matmul_async_buffer_load_lds
//   target: gfx942, wavefront_size: 64
//   Triton metadata shared: 8192
//   num_warps: 4
//   num_stages: 3

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_async_buffer_load_lds    ; -- Begin function matmul_async_buffer_load_lds
	.p2align	8
	.type	matmul_async_buffer_load_lds,@function
matmul_async_buffer_load_lds:           ; @matmul_async_buffer_load_lds
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.5:
	.file	1 "/tmp" "generate_rocjitsu_triton_fixtures.py"
	.loc	1 48 0 prologue_end             ; generate_rocjitsu_triton_fixtures.py:48:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.6:
.LBB0_0:
.Ltmp1:
	.loc	1 39 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:39:22 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_ashr_i32 s0, s12, 31
	s_lshr_b32 s0, s0, 26
	s_add_i32 s0, s12, s0
	s_ashr_i32 s1, s0, 6
	.loc	1 40 29                         ; generate_rocjitsu_triton_fixtures.py:40:29 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_lshl_b32 s1, s1, 2
	.loc	1 41 37                         ; generate_rocjitsu_triton_fixtures.py:41:37 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_sub_i32 s8, 16, s1
	.loc	1 41 50 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:41:50 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_min_i32 s9, s8, 4
	.loc	1 43 34 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:43:34 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_abs_i32 s10, s9
	v_cvt_f32_u32_e32 v1, s10
	s_sub_i32 s13, 0, s10
	.loc	1 42 34                         ; generate_rocjitsu_triton_fixtures.py:42:34 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_andn2_b32 s0, s0, 63
	s_sub_i32 s0, s12, s0
	.loc	1 43 34                         ; generate_rocjitsu_triton_fixtures.py:43:34 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s12, s0
	s_xor_b32 s11, s0, s9
	s_ashr_i32 s11, s11, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
.Ltmp2:
	.loc	1 57 44                         ; generate_rocjitsu_triton_fixtures.py:57:44
	v_lshrrev_b32_e32 v2, 6, v0
	v_and_b32_e32 v59, 63, v0
	v_and_b32_e32 v50, 0xc0, v0
.Ltmp3:
	.loc	1 43 34                         ; generate_rocjitsu_triton_fixtures.py:43:34 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s13, s13, s14
	s_mul_hi_u32 s13, s14, s13
	s_add_i32 s14, s14, s13
	s_mul_hi_u32 s13, s12, s14
	s_mul_i32 s14, s13, s10
	s_sub_i32 s12, s12, s14
	s_add_i32 s14, s13, 1
	s_sub_i32 s15, s12, s10
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s13, s14, s13
	s_cselect_b32 s12, s15, s12
	s_add_i32 s14, s13, 1
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s10, s14, s13
	s_xor_b32 s10, s10, s11
	s_sub_i32 s10, s10, s11
	.loc	1 42 48                         ; generate_rocjitsu_triton_fixtures.py:42:48 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_mul_i32 s9, s10, s9
	s_sub_i32 s0, s0, s9
	.loc	1 42 27 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:42:27 @[ generate_rocjitsu_triton_fixtures.py:56:56 ]
	s_add_i32 s0, s0, s1
.Ltmp4:
	.loc	1 57 21 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:57:21
	s_lshl_b32 s9, s0, 6
	.loc	1 58 21                         ; generate_rocjitsu_triton_fixtures.py:58:21
	s_lshl_b32 s10, s10, 6
	.loc	1 57 31                         ; generate_rocjitsu_triton_fixtures.py:57:31
	v_or_b32_e32 v3, s9, v2
	.loc	1 58 31                         ; generate_rocjitsu_triton_fixtures.py:58:31
	v_or_b32_e32 v2, s10, v59
	s_movk_i32 s0, 0x400
	.loc	1 61 31                         ; generate_rocjitsu_triton_fixtures.py:61:31
	v_cmp_gt_i32_e32 vcc, s0, v2
	.loc	1 57 31                         ; generate_rocjitsu_triton_fixtures.py:57:31
	v_lshlrev_b32_e32 v5, 10, v3
	.loc	1 57 44 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:57:44
	v_and_b32_e32 v1, 15, v0
	.loc	1 61 42 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:61:42
	v_cndmask_b32_e32 v2, 0, v2, vcc
	.loc	1 60 31                         ; generate_rocjitsu_triton_fixtures.py:60:31
	v_cmp_gt_i32_e32 vcc, s0, v3
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x1000, v5
	.loc	1 57 44                         ; generate_rocjitsu_triton_fixtures.py:57:44
	v_and_b32_e32 v54, 0x80, v0
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v6, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x2000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v8, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x3000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v10, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x4000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v12, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x5000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v14, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x6000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v16, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x7000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v32, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x8000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v34, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0x9000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v36, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xa000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v38, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xb000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v40, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xc000, v5
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v42, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xd000, v5
	.loc	1 67 29 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:67:29
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 60 42 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v4, 0, v5, vcc
	v_cndmask_b32_e32 v44, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xe000, v5
	.loc	1 67 29 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:67:29
	v_lshl_add_u64 v[22:23], v[8:9], 1, s[2:3]
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshrrev_b32_e32 v9, 1, v0
	.loc	1 60 42 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v46, 0, v3, vcc
	.loc	1 67 47                         ; generate_rocjitsu_triton_fixtures.py:67:47
	v_or_b32_e32 v3, 0xf000, v5
	.loc	1 67 29 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:67:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[28:29], v[14:15], 1, s[2:3]
	v_lshlrev_b32_e32 v8, 3, v1
	v_and_b32_e32 v9, 24, v9
	v_bfe_i32 v15, v0, 6, 1
	v_lshl_add_u64 v[18:19], v[4:5], 1, s[2:3]
	v_lshl_add_u64 v[24:25], v[10:11], 1, s[2:3]
	v_lshlrev_b32_e32 v4, 1, v0
	v_lshrrev_b32_e32 v5, 3, v50
	v_mul_u32_u24_e32 v10, 0x88, v1
	v_xor_b32_e32 v8, v8, v9
	v_and_b32_e32 v15, 0x808, v15
	.loc	1 65 38 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshlrev_b32_e32 v50, 7, v0
	.loc	1 60 42                         ; generate_rocjitsu_triton_fixtures.py:60:42
	v_cndmask_b32_e32 v48, 0, v3, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_xor_b32_e32 v55, v4, v5
	v_xor_b32_e32 v11, v10, v9
	v_xor_b32_e32 v5, v8, v5
	v_xor_b32_e32 v9, v15, v9
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_and_b32_e32 v50, 0x6000, v50
	v_mov_b32_e32 v51, 0
	.loc	1 67 29                         ; generate_rocjitsu_triton_fixtures.py:67:29
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_or_b32 v56, v54, 4, v11
	v_lshl_or_b32 v57, v59, 7, v5
	v_xor_b32_e32 v58, v9, v10
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[50:51]
	.loc	1 67 29                         ; generate_rocjitsu_triton_fixtures.py:67:29
	v_lshl_add_u64 v[20:21], v[6:7], 1, s[2:3]
	v_lshl_add_u64 v[26:27], v[12:13], 1, s[2:3]
	v_lshl_add_u64 v[30:31], v[16:17], 1, s[2:3]
	v_ashrrev_i32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v35, 31, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_ashrrev_i32_e32 v39, 31, v38
	v_ashrrev_i32_e32 v41, 31, v40
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v45, 31, v44
	v_ashrrev_i32_e32 v47, 31, v46
	v_ashrrev_i32_e32 v49, 31, v48
	v_xor_b32_e32 v4, 32, v55
	v_xor_b32_e32 v6, 64, v55
	v_xor_b32_e32 v7, 0x60, v55
	v_xor_b32_e32 v11, 32, v56
	v_xor_b32_e32 v12, 64, v56
	v_xor_b32_e32 v13, 0x60, v56
	v_xor_b32_e32 v5, 32, v57
	v_xor_b32_e32 v8, 64, v57
	v_xor_b32_e32 v14, 0x60, v57
	v_xor_b32_e32 v9, 32, v58
	v_xor_b32_e32 v10, 64, v58
	v_xor_b32_e32 v15, 0x60, v58
	v_xor_b32_e32 v16, 16, v58
	v_xor_b32_e32 v17, 48, v58
	v_xor_b32_e32 v73, 0x50, v58
	v_xor_b32_e32 v74, 0x70, v58
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[4:5]
	s_mov_b64 s[0:1], 0x19800
	s_movk_i32 s8, 0xffc0
	.loc	1 67 29                         ; generate_rocjitsu_triton_fixtures.py:67:29
	v_lshl_add_u64 v[32:33], v[32:33], 1, s[2:3]
	v_lshl_add_u64 v[34:35], v[34:35], 1, s[2:3]
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[2:3]
	v_lshl_add_u64 v[38:39], v[38:39], 1, s[2:3]
	v_lshl_add_u64 v[40:41], v[40:41], 1, s[2:3]
	v_lshl_add_u64 v[42:43], v[42:43], 1, s[2:3]
	v_lshl_add_u64 v[44:45], v[44:45], 1, s[2:3]
	v_lshl_add_u64 v[46:47], v[46:47], 1, s[2:3]
	v_lshl_add_u64 v[48:49], v[48:49], 1, s[2:3]
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[52:53], v[2:3], 0, s[0:1]
	v_lshlrev_b32_e32 v50, 1, v59
	s_mov_b32 s4, 0xfffe7000
	s_mov_b32 s5, 0xfffe8000
	s_mov_b32 s11, 0xfffef000
	s_mov_b32 s12, 0xffff0000
	s_mov_b32 s13, 0xffff7000
	s_movk_i32 s14, 0x8000
	s_movk_i32 s15, 0xf000
	v_add_u32_e32 v59, 0, v4
	v_add_u32_e32 v60, 0, v6
	v_add_u32_e32 v61, 0, v7
	v_add_u32_e32 v62, 0, v11
	v_add_u32_e32 v63, 0, v12
	v_add_u32_e32 v64, 0, v13
	s_mov_b32 s16, 0x5040100
	v_add_u32_e32 v65, 0, v5
	v_add_u32_e32 v66, 0, v8
	v_add_u32_e32 v67, 0, v14
	v_add_u32_e32 v68, 0, v9
	v_add_u32_e32 v69, 0, v10
	v_add_u32_e32 v70, 0, v15
	v_add_u32_e32 v71, 0, v16
	v_add_u32_e32 v72, 0, v17
	v_add_u32_e32 v73, 0, v73
	v_add_u32_e32 v74, 0, v74
	s_mov_b64 s[0:1], 0x20000
	s_mov_b64 s[2:3], 0x80
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[2:3], v[20:21], 0, v[50:51]
	v_lshl_add_u64 v[4:5], v[22:23], 0, v[50:51]
	.loc	1 67 25 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:67:25
	v_lshl_add_u64 v[78:79], v[18:19], 0, v[50:51]
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[6:7], v[24:25], 0, v[50:51]
	v_lshl_add_u64 v[8:9], v[26:27], 0, v[50:51]
	v_lshl_add_u64 v[12:13], v[30:31], 0, v[50:51]
	v_lshl_add_u64 v[14:15], v[32:33], 0, v[50:51]
	v_lshl_add_u64 v[16:17], v[34:35], 0, v[50:51]
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	global_load_ushort v75, v[78:79], off
	global_load_ushort v80, v[2:3], off
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[2:3], v[38:39], 0, v[50:51]
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	global_load_ushort v78, v[4:5], off
	global_load_ushort v79, v[6:7], off
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[4:5], v[40:41], 0, v[50:51]
	v_lshl_add_u64 v[10:11], v[28:29], 0, v[50:51]
	v_lshl_add_u64 v[76:77], v[36:37], 0, v[50:51]
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	global_load_ushort v81, v[8:9], off
	global_load_ushort v82, v[10:11], off
	global_load_ushort v83, v[12:13], off
	global_load_ushort v84, v[14:15], off
	s_nop 0
	global_load_ushort v12, v[16:17], off
	global_load_ushort v13, v[76:77], off
	global_load_ushort v14, v[2:3], off
	global_load_ushort v15, v[4:5], off
	.loc	1 69 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:69:25
	v_add_co_u32_e32 v4, vcc, s4, v52
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[6:7], v[42:43], 0, v[50:51]
	v_lshl_add_u64 v[10:11], v[46:47], 0, v[50:51]
	v_lshl_add_u64 v[2:3], v[48:49], 0, v[50:51]
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	v_addc_co_u32_e32 v5, vcc, -1, v53, vcc
	.loc	1 67 51                         ; generate_rocjitsu_triton_fixtures.py:67:51
	v_lshl_add_u64 v[8:9], v[44:45], 0, v[50:51]
	.loc	1 67 25 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:67:25
	global_load_ushort v16, v[6:7], off
	global_load_ushort v17, v[8:9], off
	s_nop 0
	global_load_ushort v6, v[10:11], off
	global_load_ushort v7, v[2:3], off
	.loc	1 69 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:69:25
	v_add_co_u32_e32 v2, vcc, s5, v52
	v_accvgpr_mov_b32 a17, a11
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v8, v[4:5], off offset:-2048
	global_load_ushort v9, v[2:3], off offset:-4096
	global_load_ushort v10, v[2:3], off offset:-2048
	global_load_ushort v11, v[2:3], off
	v_add_co_u32_e32 v2, vcc, s11, v52
	v_accvgpr_mov_b32 a16, a10
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v85, v[2:3], off offset:-2048
	v_add_co_u32_e32 v2, vcc, s12, v52
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	v_accvgpr_mov_b32 a10, a12
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v86, v[2:3], off offset:-4096
	global_load_ushort v87, v[2:3], off offset:-2048
	global_load_ushort v88, v[2:3], off
	v_add_co_u32_e32 v2, vcc, s13, v52
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	v_accvgpr_mov_b32 a11, a13
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v89, v[2:3], off offset:-2048
	v_add_co_u32_e32 v2, vcc, s15, v52
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	v_accvgpr_mov_b32 a12, a14
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v90, v[2:3], off offset:-2048
	v_add_co_u32_e32 v2, vcc, s14, v52
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	v_accvgpr_mov_b32 a13, a15
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_nop 0
	v_addc_co_u32_e32 v3, vcc, -1, v53, vcc
	global_load_ushort v91, v[2:3], off offset:-4096
	global_load_ushort v92, v[2:3], off offset:-2048
	global_load_ushort v93, v[2:3], off
	global_load_ushort v94, v[52:53], off offset:-2048
	global_load_ushort v95, v[52:53], off
	global_load_ushort v96, v[52:53], off offset:-4096
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	v_add_u32_e32 v2, 0, v55
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_accvgpr_mov_b32 a15, a9
	v_accvgpr_mov_b32 a14, a8
	v_accvgpr_mov_b32 a9, a3
	v_accvgpr_mov_b32 a8, a2
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	v_accvgpr_mov_b32 a2, a4
	v_accvgpr_mov_b32 a3, a5
	v_accvgpr_mov_b32 a4, a6
	v_accvgpr_mov_b32 a5, a7
	v_accvgpr_mov_b32 a7, a1
	v_accvgpr_mov_b32 a6, a0
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	s_add_i32 s8, s8, 64
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[2:3]
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[2:3]
	v_lshl_add_u64 v[48:49], v[48:49], 0, s[2:3]
	v_lshl_add_u64 v[46:47], v[46:47], 0, s[2:3]
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	s_waitcnt vmcnt(31)
	ds_write_b16 v2, v75
	s_waitcnt vmcnt(27)
	ds_write_b16 v2, v81 offset:2048
	s_waitcnt vmcnt(23)
	ds_write_b16 v2, v12 offset:4096
	s_waitcnt vmcnt(19)
	ds_write_b16 v2, v16 offset:6144
	ds_write_b16 v59, v80 offset:512
	ds_write_b16 v59, v82 offset:2560
	ds_write_b16 v59, v13 offset:4608
	s_waitcnt vmcnt(18)
	ds_write_b16 v59, v17 offset:6656
	ds_write_b16 v60, v78 offset:1024
	ds_write_b16 v60, v83 offset:3072
	ds_write_b16 v60, v14 offset:5120
	s_waitcnt vmcnt(17)
	ds_write_b16 v60, v6 offset:7168
	ds_write_b16 v61, v79 offset:1536
	v_add_u32_e32 v2, 0, v56
	ds_write_b16 v61, v84 offset:3584
	ds_write_b16 v61, v15 offset:5632
	s_waitcnt vmcnt(16)
	ds_write_b16 v61, v7 offset:7680
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[14:17], v2 offset1:8
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_waitcnt vmcnt(14)
	v_perm_b32 v76, v9, v8, s16
	s_waitcnt vmcnt(12)
	v_perm_b32 v77, v11, v10, s16
	v_add_u32_e32 v75, 0, v57
	.loc	1 67 25                         ; generate_rocjitsu_triton_fixtures.py:67:25
	ds_read2st64_b64 v[10:13], v62 offset1:8
	ds_read2st64_b64 v[6:9], v63 offset1:8
	ds_read2st64_b64 v[2:5], v64 offset1:8
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b64 v75, v[76:77]
	v_add_u32_e32 v75, 0, v58
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[44:45], v[44:45], 0, s[2:3]
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[2:3]
	v_lshl_add_u64 v[40:41], v[40:41], 0, s[2:3]
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_waitcnt vmcnt(10)
	v_perm_b32 v76, v86, v85, s16
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[2:3]
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_waitcnt vmcnt(8)
	v_perm_b32 v77, v88, v87, s16
	ds_write_b64 v65, v[76:77]
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[2:3]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[2:3]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[2:3]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[2:3]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[2:3]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[2:3]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[2:3]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[2:3]
	s_cmpk_lt_u32 s8, 0x3c0
	v_lshl_add_u64 v[52:53], v[52:53], 0, s[0:1]
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	s_waitcnt vmcnt(5)
	v_perm_b32 v78, v91, v89, s16
	s_waitcnt vmcnt(3)
	v_perm_b32 v79, v93, v92, s16
	ds_write_b64 v66, v[78:79]
	s_waitcnt vmcnt(1)
	v_perm_b32 v81, v95, v94, s16
	s_waitcnt vmcnt(0)
	v_perm_b32 v80, v96, v90, s16
	ds_write_b64 v67, v[80:81]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[76:77], v75
	ds_read_b64 v[78:79], v68
	ds_read_b64 v[80:81], v69
	ds_read_b64 v[82:83], v70
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[10:13], v[76:77], v[14:15], a[10:13]
	v_mfma_f32_16x16x16_f16 a[14:17], v[76:77], v[16:17], a[14:17]
	.loc	1 69 25                         ; generate_rocjitsu_triton_fixtures.py:69:25
	ds_read_b64 v[76:77], v71 offset:4096
	ds_read_b64 v[84:85], v72 offset:4096
	ds_read_b64 v[86:87], v73 offset:4096
	ds_read_b64 v[88:89], v74 offset:4096
	.loc	1 71 30                         ; generate_rocjitsu_triton_fixtures.py:71:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x16_f16 a[2:5], v[76:77], v[14:15], a[2:5]
	v_mfma_f32_16x16x16_f16 a[6:9], v[76:77], v[16:17], a[6:9]
	v_mfma_f32_16x16x16_f16 a[10:13], v[78:79], v[10:11], a[10:13]
	v_mfma_f32_16x16x16_f16 a[14:17], v[78:79], v[12:13], a[14:17]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x16_f16 a[0:3], v[84:85], v[10:11], a[2:5]
	v_mfma_f32_16x16x16_f16 a[4:7], v[84:85], v[12:13], a[6:9]
	v_mfma_f32_16x16x16_f16 a[8:11], v[80:81], v[6:7], a[10:13]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x16_f16 a[0:3], v[86:87], v[6:7], a[0:3]
	v_mfma_f32_16x16x16_f16 a[16:19], v[80:81], v[8:9], a[14:17]
	v_mfma_f32_16x16x16_f16 a[20:23], v[86:87], v[8:9], a[4:7]
	v_mfma_f32_16x16x16_f16 a[12:15], v[82:83], v[2:3], a[8:11]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[4:7], v[88:89], v[2:3], a[0:3]
	v_mfma_f32_16x16x16_f16 a[8:11], v[82:83], v[4:5], a[16:19]
	v_mfma_f32_16x16x16_f16 a[0:3], v[88:89], v[4:5], a[20:23]
	.loc	1 65 38                         ; generate_rocjitsu_triton_fixtures.py:65:38
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	.loc	1 57 44                         ; generate_rocjitsu_triton_fixtures.py:57:44
	v_lshrrev_b32_e32 v2, 3, v54
	v_or_b32_e32 v1, v2, v1
	v_lshrrev_b32_e32 v0, 2, v0
	.loc	1 57 31 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:57:31
	v_or_b32_e32 v2, s9, v1
	.loc	1 58 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:58:31
	v_and_or_b32 v0, v0, 28, s10
	.loc	1 73 43                         ; generate_rocjitsu_triton_fixtures.py:73:43
	v_max_i32_e32 v1, v2, v0
	s_movk_i32 s0, 0x400
	v_cmp_gt_i32_e32 vcc, s0, v1
	.loc	1 72 56                         ; generate_rocjitsu_triton_fixtures.py:72:56
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:                                ; %.critedge
	.loc	1 72 35 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:72:35
	v_mov_b32_e32 v1, 0x8000
	v_lshl_or_b32 v4, v2, 10, v1
	v_lshlrev_b32_e32 v2, 10, v2
	.loc	1 72 17                         ; generate_rocjitsu_triton_fixtures.py:72:17
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 72 39                         ; generate_rocjitsu_triton_fixtures.py:72:39
	v_ashrrev_i32_e32 v1, 31, v0
	.loc	1 72 17                         ; generate_rocjitsu_triton_fixtures.py:72:17
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	.loc	1 72 39                         ; generate_rocjitsu_triton_fixtures.py:72:39
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	.loc	1 72 17                         ; generate_rocjitsu_triton_fixtures.py:72:17
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[6:7]
	.loc	1 72 39                         ; generate_rocjitsu_triton_fixtures.py:72:39
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[0:1]
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	.loc	1 72 56                         ; generate_rocjitsu_triton_fixtures.py:72:56
	global_store_dwordx4 v[0:1], a[12:15], off
	global_store_dwordx4 v[0:1], a[4:7], off offset:128
	global_store_dwordx4 v[4:5], a[8:11], off
	global_store_dwordx4 v[4:5], a[0:3], off offset:128
.LBB0_4:                                ; %.critedge28
	.loc	1 72 4                          ; generate_rocjitsu_triton_fixtures.py:72:4
	s_endpgm
.Ltmp5:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_async_buffer_load_lds
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 10
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 124
		.amdhsa_next_free_sgpr 17
		.amdhsa_accum_offset 100
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	matmul_async_buffer_load_lds, .Lfunc_end0-matmul_async_buffer_load_lds
	.cfi_endproc
                                        ; -- End function
	.set matmul_async_buffer_load_lds.num_vgpr, 97
	.set matmul_async_buffer_load_lds.num_agpr, 24
	.set matmul_async_buffer_load_lds.numbered_sgpr, 17
	.set matmul_async_buffer_load_lds.num_named_barrier, 0
	.set matmul_async_buffer_load_lds.private_seg_size, 0
	.set matmul_async_buffer_load_lds.uses_vcc, 1
	.set matmul_async_buffer_load_lds.uses_flat_scratch, 0
	.set matmul_async_buffer_load_lds.has_dyn_sized_stack, 0
	.set matmul_async_buffer_load_lds.has_recursion, 0
	.set matmul_async_buffer_load_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2760
; TotalNumSgprs: 23
; NumVgprs: 97
; NumAgprs: 24
; TotalNumVgprs: 124
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 23
; NumVGPRsForWavesPerEU: 124
; AccumOffset: 100
; Occupancy: 4
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 24
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x44 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x1e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	56                              ; DW_AT_call_line
	.byte	56                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"generate_rocjitsu_triton_fixtures.py" ; string offset=7
.Linfo_string2:
	.asciz	"/tmp"                          ; string offset=44
.Linfo_string3:
	.asciz	"matmul_async_buffer_load_lds"  ; string offset=49
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     24
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           matmul_async_buffer_load_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     23
    .sgpr_spill_count: 0
    .symbol:         matmul_async_buffer_load_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     124
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
