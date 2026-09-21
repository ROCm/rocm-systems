// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Native CDNA3/gfx942 Triton golden fixture generated from the same source and
// constexprs as the corresponding gfx950 DBT input fixture.
//
// Recorded fixture metadata:
//   kernel: triton_cdna4_matmul_kernel
//   target: gfx942, wavefront_size: 64
//   Triton metadata shared: 4096
//   num_warps: 4
//   num_stages: 2

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	triton_cdna4_matmul_kernel      ; -- Begin function triton_cdna4_matmul_kernel
	.p2align	8
	.type	triton_cdna4_matmul_kernel,@function
triton_cdna4_matmul_kernel:             ; @triton_cdna4_matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.69:
	.file	1 "/tmp" "generate_rocjitsu_triton_fixtures.py"
	.loc	1 14 0 prologue_end             ; generate_rocjitsu_triton_fixtures.py:14:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.70:
.LBB0_0:
.Ltmp1:
	.loc	1 18 21 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:18:21
	s_lshl_b32 s11, s16, 5
	.loc	1 19 21                         ; generate_rocjitsu_triton_fixtures.py:19:21
	s_lshl_b32 s28, s17, 5
	.loc	1 18 44                         ; generate_rocjitsu_triton_fixtures.py:18:44
	v_and_b32_e32 v20, 0xc0, v0
	v_and_b32_e32 v21, 31, v0
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	s_cmp_lt_i32 s10, 1
	.loc	1 18 44                         ; generate_rocjitsu_triton_fixtures.py:18:44
	v_and_b32_e32 v22, 32, v0
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	s_cbranch_scc1 .LBB0_35
; %bb.1:                                ; %.lr.ph
	.loc	1 18 44                         ; generate_rocjitsu_triton_fixtures.py:18:44
	v_lshrrev_b32_e32 v1, 6, v20
	.loc	1 18 31 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:18:31
	v_or_b32_e32 v1, s11, v1
	.loc	1 20 26 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:20:26
	v_and_b32_e32 v27, 0xe0, v0
	v_and_b32_e32 v28, 15, v0
	.loc	1 18 31                         ; generate_rocjitsu_triton_fixtures.py:18:31
	v_or_b32_e32 v2, 4, v1
	.loc	1 20 26                         ; generate_rocjitsu_triton_fixtures.py:20:26
	v_lshrrev_b32_e32 v24, 3, v27
	v_lshlrev_b32_e32 v29, 3, v28
	v_lshrrev_b32_e32 v26, 2, v22
	v_lshrrev_b32_e32 v27, 2, v27
	v_bfe_i32 v31, v0, 4, 1
	.loc	1 18 31                         ; generate_rocjitsu_triton_fixtures.py:18:31
	v_or_b32_e32 v3, 8, v1
	v_or_b32_e32 v4, 12, v1
	v_or_b32_e32 v5, 16, v1
	v_or_b32_e32 v6, 20, v1
	v_or_b32_e32 v7, 24, v1
	v_or_b32_e32 v8, 28, v1
	.loc	1 20 26                         ; generate_rocjitsu_triton_fixtures.py:20:26
	v_and_b32_e32 v23, 63, v0
	.loc	1 25 49                         ; generate_rocjitsu_triton_fixtures.py:25:49
	v_cmp_gt_i32_e32 vcc, s8, v1
	v_cmp_gt_i32_e64 s[0:1], s8, v2
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_mul_lo_u32 v2, s10, v1
	s_lshl_b32 s16, s10, 2
	v_lshlrev_b32_e32 v1, 1, v0
	v_xor_b32_e32 v30, v29, v26
	v_and_b32_e32 v31, 0x808, v31
	v_xor_b32_e32 v27, v29, v27
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_lshrrev_b32_e32 v0, 5, v0
	.loc	1 25 49                         ; generate_rocjitsu_triton_fixtures.py:25:49
	v_cmp_gt_i32_e64 s[20:21], s8, v4
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v4, s16, v2
	v_lshlrev_b32_e32 v28, 7, v28
	v_xor_b32_e32 v27, v27, v31
	v_xor_b32_e32 v29, v30, v31
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v0, s9, v0
	.loc	1 25 49                         ; generate_rocjitsu_triton_fixtures.py:25:49
	v_cmp_gt_i32_e64 s[26:27], s8, v6
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v6, s16, v4
	v_or_b32_e32 v27, v27, v28
	v_or_b32_e32 v28, v29, v28
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_lshlrev_b32_e32 v29, 2, v0
	v_or_b32_e32 v0, 2, v24
	.loc	1 25 49                         ; generate_rocjitsu_triton_fixtures.py:25:49
	v_cmp_gt_i32_e64 s[14:15], s8, v8
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v8, s16, v6
	v_lshl_or_b32 v26, v21, 7, v30
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v30, s9, v0
	v_or_b32_e32 v0, 3, v24
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v10, s16, v8
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v31, s9, v0
	v_or_b32_e32 v0, 32, v24
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v12, s16, v10
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v32, s9, v0
	v_or_b32_e32 v0, 33, v24
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v14, s16, v12
	v_lshrrev_b32_e32 v25, 3, v20
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v33, s9, v0
	v_or_b32_e32 v0, 34, v24
	.loc	1 19 31                         ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v18, s28, v21
	.loc	1 24 47                         ; generate_rocjitsu_triton_fixtures.py:24:47
	v_add_u32_e32 v16, s16, v14
	v_xor_b32_e32 v25, v1, v25
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v34, s9, v0
	v_or_b32_e32 v0, 35, v24
	.loc	1 25 49                         ; generate_rocjitsu_triton_fixtures.py:25:49
	v_cmp_gt_i32_e64 s[22:23], s8, v3
	v_cmp_gt_i32_e64 s[24:25], s8, v5
	v_cmp_gt_i32_e64 s[12:13], s8, v7
	.loc	1 24 29                         ; generate_rocjitsu_triton_fixtures.py:24:29
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v19, 31, v18
	v_xor_b32_e32 v1, 32, v25
	v_xor_b32_e32 v38, 64, v25
	v_xor_b32_e32 v39, 0x60, v25
	v_xor_b32_e32 v40, 16, v26
	v_xor_b32_e32 v41, 32, v26
	v_xor_b32_e32 v42, 48, v26
	v_xor_b32_e32 v43, 64, v26
	v_xor_b32_e32 v44, 0x50, v26
	v_xor_b32_e32 v45, 0x60, v26
	v_xor_b32_e32 v46, 0x70, v26
	v_xor_b32_e32 v47, 64, v27
	v_xor_b32_e32 v48, 16, v28
	v_xor_b32_e32 v49, 32, v28
	v_xor_b32_e32 v50, 48, v28
	v_xor_b32_e32 v51, 64, v28
	v_xor_b32_e32 v52, 0x50, v28
	v_xor_b32_e32 v53, 0x60, v28
	v_xor_b32_e32 v54, 0x70, v28
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_mul_lo_u32 v35, s9, v0
	v_mul_lo_u32 v0, s9, v24
	.loc	1 24 29                         ; generate_rocjitsu_triton_fixtures.py:24:29
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[2:3]
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[2:3]
	.loc	1 27 73                         ; generate_rocjitsu_triton_fixtures.py:27:73
	v_cmp_gt_i32_e64 s[16:17], s9, v18
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[4:5]
	s_lshl_b32 s4, s9, 6
	v_add_u32_e32 v36, s9, v0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	s_mov_b32 s5, 0
	v_add_u32_e32 v37, 0, v1
	v_add_u32_e32 v38, 0, v38
	v_add_u32_e32 v39, 0, v39
	v_add_u32_e32 v40, 0, v40
	v_add_u32_e32 v41, 0, v41
	v_add_u32_e32 v42, 0, v42
	v_add_u32_e32 v43, 0, v43
	v_add_u32_e32 v44, 0, v44
	v_add_u32_e32 v45, 0, v45
	v_add_u32_e32 v46, 0, v46
	s_mov_b32 s29, 0x5040100
	v_add_u32_e32 v47, 0, v47
	v_add_u32_e32 v48, 0, v48
	v_add_u32_e32 v49, 0, v49
	v_add_u32_e32 v50, 0, v50
	v_add_u32_e32 v51, 0, v51
	v_add_u32_e32 v52, 0, v52
	v_add_u32_e32 v53, 0, v53
	v_add_u32_e32 v54, 0, v54
	s_mov_b32 s30, 0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 29 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:29
	s_or_b64 exec, exec, s[2:3]
	.loc	1 24 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:24:25
	v_add_u32_e32 v69, 0, v25
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v69, v61
	ds_write_b16 v69, v65 offset:2048
	ds_write_b16 v37, v63 offset:512
	ds_write_b16 v37, v67 offset:2560
	ds_write_b16 v38, v62 offset:1024
	ds_write_b16 v38, v66 offset:3072
	ds_write_b16 v39, v64 offset:1536
	ds_write_b16 v39, v68 offset:3584
	v_add_u32_e32 v61, 0, v26
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	v_perm_b32 v1, v58, v1, s29
	v_perm_b32 v0, v0, v55, s29
	v_add_u32_e32 v55, 0, v27
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[62:63], v61
	ds_read_b64 v[64:65], v40
	ds_read_b64 v[66:67], v41
	ds_read_b64 v[68:69], v42
	ds_read_b64 v[70:71], v43
	ds_read_b64 v[72:73], v44
	ds_read_b64 v[74:75], v45
	ds_read_b64 v[76:77], v46
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b64 v55, v[0:1]
	v_perm_b32 v0, v60, v57, s29
	v_perm_b32 v1, v56, v59, s29
	ds_write_b64 v47, v[0:1]
	v_add_u32_e32 v0, 0, v28
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[0:1], v0
	ds_read_b64 v[56:57], v48
	ds_read_b64 v[58:59], v49
	ds_read_b64 v[60:61], v50
	.loc	1 28 30                         ; generate_rocjitsu_triton_fixtures.py:28:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[62:63], a[0:15]
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	s_add_i32 s30, s30, 64
	s_add_i32 s5, s5, s4
	s_cmp_lt_i32 s30, s10
	.loc	1 28 30                         ; generate_rocjitsu_triton_fixtures.py:28:30
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[56:57], v[64:65], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[58:59], v[66:67], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[60:61], v[68:69], a[0:15]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	ds_read_b64 v[0:1], v51
	ds_read_b64 v[56:57], v52
	ds_read_b64 v[58:59], v53
	ds_read_b64 v[60:61], v54
	.loc	1 28 30                         ; generate_rocjitsu_triton_fixtures.py:28:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[70:71], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[56:57], v[72:73], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[58:59], v[74:75], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[60:61], v[76:77], a[0:15]
	.loc	1 22 29                         ; generate_rocjitsu_triton_fixtures.py:22:29
	s_cbranch_scc0 .LBB0_36
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 23 22                         ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v0, s30, v23
	.loc	1 25 73                         ; generate_rocjitsu_triton_fixtures.py:25:73
	v_cmp_gt_i32_e64 s[18:19], s10, v0
	.loc	1 25 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], vcc, s[18:19]
	.loc	1 24 51 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_ashrrev_i32_e32 v1, 31, v0
	v_mov_b32_e32 v61, 0
	.loc	1 24 25 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[2:3]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v61, v[56:57], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[0:1], s[18:19]
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[4:5]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v63, v[56:57], off
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[22:23], s[18:19]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[6:7]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v62, v[56:57], off
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[20:21], s[18:19]
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[8:9]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v64, v[56:57], off
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[24:25], s[18:19]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[10:11]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v65, v[56:57], off
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[26:27], s[18:19]
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[12:13]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v67, v[56:57], off
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[34:35], s[12:13], s[18:19]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[56:57], v[0:1], 1, v[14:15]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v66, v[56:57], off
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 55 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:55
	s_and_b64 s[18:19], s[14:15], s[18:19]
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v68, 0
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 24 51 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:24:51
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	.loc	1 24 25                         ; generate_rocjitsu_triton_fixtures.py:24:25
	global_load_ushort v68, v[0:1], off
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v56, s30, v24
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v56
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:25
	v_add_u32_e32 v0, s5, v29
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v55, v[0:1], off
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v0, 1, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v0
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v58, s5, v36
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v59, 31, v58
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[58:59], v[58:59], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v0, v[58:59], off
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v57, 2, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v57
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v58, s5, v30
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v59, 31, v58
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[58:59], v[58:59], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v1, v[58:59], off
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v57, 3, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v57
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v58, s5, v31
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v59, 31, v58
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[58:59], v[58:59], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v58, v[58:59], off
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v59, 32, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v59
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v70, s5, v32
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v71, 31, v70
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v57, v[70:71], off
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v59, 33, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v59
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v70, s5, v33
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v71, 31, v70
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v60, v[70:71], off
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 23 22 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:23:22
	v_add_u32_e32 v69, 34, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v69
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 47 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:47
	v_add_u32_e32 v70, s5, v34
	.loc	1 26 29                         ; generate_rocjitsu_triton_fixtures.py:26:29
	v_ashrrev_i32_e32 v71, 31, v70
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v59, v[70:71], off
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 25 73 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:25:73
	v_add_u32_e32 v56, 35, v56
	.loc	1 27 49                         ; generate_rocjitsu_triton_fixtures.py:27:49
	v_cmp_gt_i32_e64 s[2:3], s10, v56
	.loc	1 27 55 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:27:55
	s_and_b64 s[18:19], s[16:17], s[2:3]
	v_mov_b32_e32 v56, 0
	.loc	1 26 25 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:26:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_2
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	.loc	1 26 29 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:26:29
	v_add_u32_e32 v70, s5, v35
	v_ashrrev_i32_e32 v71, 31, v70
	.loc	1 26 51                         ; generate_rocjitsu_triton_fixtures.py:26:51
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[18:19]
	.loc	1 26 25                         ; generate_rocjitsu_triton_fixtures.py:26:25
	global_load_ushort v56, v[70:71], off
	s_branch .LBB0_2
.LBB0_35:
	.loc	1 0 25                          ; generate_rocjitsu_triton_fixtures.py:0:25
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
.LBB0_36:                               ; %._crit_edge
	.loc	1 18 44 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:18:44
	v_lshrrev_b32_e32 v0, 3, v22
	.loc	1 18 31 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:18:31
	v_or_b32_e32 v1, s11, v21
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v0, s28, v0
	.loc	1 29 35                         ; generate_rocjitsu_triton_fixtures.py:29:35
	v_mul_lo_u32 v2, s9, v1
	.loc	1 30 37                         ; generate_rocjitsu_triton_fixtures.py:30:37
	v_cmp_gt_i32_e32 vcc, s8, v1
	.loc	1 30 61 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[0:1], s9, v0
	.loc	1 29 17 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:17
	v_ashrrev_i32_e32 v3, 31, v2
	.loc	1 30 43                         ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[0:1]
	.loc	1 29 17                         ; generate_rocjitsu_triton_fixtures.py:29:17
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[6:7]
	.loc	1 29 39 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:29:39
	v_ashrrev_i32_e32 v1, 31, v0
	.loc	1 29 56                         ; generate_rocjitsu_triton_fixtures.py:29:56
	v_cmp_eq_u32_e64 s[0:1], 0, v20
	.loc	1 29 39                         ; generate_rocjitsu_triton_fixtures.py:29:39
	v_lshl_add_u64 v[2:3], v[0:1], 2, v[2:3]
	.loc	1 29 56                         ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_38
; %bb.37:
	global_store_dword v[2:3], a0, off
.LBB0_38:
	.loc	1 0 56                          ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 1, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_40
; %bb.39:
	global_store_dword v[2:3], a1, off offset:4
.LBB0_40:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 2, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_42
; %bb.41:
	global_store_dword v[2:3], a2, off offset:8
.LBB0_42:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 3, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_44
; %bb.43:
	global_store_dword v[2:3], a3, off offset:12
.LBB0_44:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 8, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_46
; %bb.45:
	global_store_dword v[2:3], a4, off offset:32
.LBB0_46:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 9, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_48
; %bb.47:
	global_store_dword v[2:3], a5, off offset:36
.LBB0_48:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 10, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_50
; %bb.49:
	global_store_dword v[2:3], a6, off offset:40
.LBB0_50:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 11, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_52
; %bb.51:
	global_store_dword v[2:3], a7, off offset:44
.LBB0_52:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 16, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_54
; %bb.53:
	global_store_dword v[2:3], a8, off offset:64
.LBB0_54:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 17, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_56
; %bb.55:
	global_store_dword v[2:3], a9, off offset:68
.LBB0_56:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 18, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_58
; %bb.57:
	global_store_dword v[2:3], a10, off offset:72
.LBB0_58:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 19, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_60
; %bb.59:
	global_store_dword v[2:3], a11, off offset:76
.LBB0_60:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 24, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_62
; %bb.61:
	global_store_dword v[2:3], a12, off offset:96
.LBB0_62:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 25, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_64
; %bb.63:
	global_store_dword v[2:3], a13, off offset:100
.LBB0_64:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v1, 26, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v1
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_66
; %bb.65:
	global_store_dword v[2:3], a14, off offset:104
.LBB0_66:
	.loc	1 0 56 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 31 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:19:31
	v_or_b32_e32 v0, 27, v0
	.loc	1 30 61                         ; generate_rocjitsu_triton_fixtures.py:30:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 30 43 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:30:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 29 56 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:29:56
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_68
; %bb.67:
	global_store_dword v[2:3], a15, off offset:108
.LBB0_68:
	.loc	1 29 4 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:29:4
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel triton_cdna4_matmul_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 96
		.amdhsa_next_free_sgpr 36
		.amdhsa_accum_offset 80
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
	.size	triton_cdna4_matmul_kernel, .Lfunc_end0-triton_cdna4_matmul_kernel
	.cfi_endproc
                                        ; -- End function
	.set triton_cdna4_matmul_kernel.num_vgpr, 78
	.set triton_cdna4_matmul_kernel.num_agpr, 16
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 36
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3044
; TotalNumSgprs: 42
; NumVgprs: 78
; NumAgprs: 16
; TotalNumVgprs: 96
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 96
; AccumOffset: 80
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 19
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
	.byte	0                               ; DW_CHILDREN_no
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"generate_rocjitsu_triton_fixtures.py" ; string offset=7
.Linfo_string2:
	.asciz	"/tmp"                          ; string offset=44
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     16
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 256
    .name:           triton_cdna4_matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     96
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
