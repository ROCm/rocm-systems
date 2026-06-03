// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Native CDNA3/gfx942 Triton golden fixture generated from the same source and
// constexprs as the corresponding gfx950 DBT input fixture.
//
// Recorded fixture metadata:
//   kernel: flash_attention_fwd_no_async_kernel
//   target: gfx942, wavefront_size: 64
//   Triton metadata shared: 8192
//   num_warps: 4
//   num_stages: 2

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	flash_attention_fwd_no_async_kernel ; -- Begin function flash_attention_fwd_no_async_kernel
	.p2align	8
	.type	flash_attention_fwd_no_async_kernel,@function
flash_attention_fwd_no_async_kernel:    ; @flash_attention_fwd_no_async_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.37:
	.file	1 "/tmp" "generate_rocjitsu_triton_fixtures.py"
	.loc	1 77 0 prologue_end             ; generate_rocjitsu_triton_fixtures.py:77:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.38:
.LBB0_0:
.Ltmp1:
	.loc	1 90 21 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:90:21
	s_lshl_b32 s11, s16, 6
	.loc	1 90 44 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:90:44
	v_lshrrev_b32_e32 v1, 6, v0
	.loc	1 90 31                         ; generate_rocjitsu_triton_fixtures.py:90:31
	v_or_b32_e32 v3, s11, v1
	s_movk_i32 s0, 0x400
	.loc	1 94 57 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:94:57
	v_and_b32_e32 v2, 63, v0
	.loc	1 94 40 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:94:40
	v_cmp_gt_i32_e32 vcc, s0, v3
	.loc	1 93 54 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v5, 0
	v_lshlrev_b32_e32 v16, 1, v2
	v_mov_b32_e32 v6, 0
	.loc	1 93 16 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	v_lshlrev_b32_e32 v6, 6, v3
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[2:3]
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v6, v[6:7], off
.LBB0_2:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x100
	v_lshl_or_b32 v4, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v5, v[4:5], off
.LBB0_4:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x200
	v_lshl_or_b32 v8, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v8, v[8:9], off
.LBB0_6:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x300
	v_lshl_or_b32 v10, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v7, v[10:11], off
.LBB0_8:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x400
	v_lshl_or_b32 v10, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v10, v[10:11], off
.LBB0_10:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x500
	v_lshl_or_b32 v12, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v9, v[12:13], off
.LBB0_12:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x600
	v_lshl_or_b32 v12, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v12, v[12:13], off
.LBB0_14:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x700
	v_lshl_or_b32 v14, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v11, v[14:15], off
.LBB0_16:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x800
	v_lshl_or_b32 v14, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v14, v[14:15], off
.LBB0_18:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0x900
	v_lshl_or_b32 v18, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v13, v[18:19], off
.LBB0_20:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v18, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0xa00
	v_lshl_or_b32 v18, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v18, v[18:19], off
.LBB0_22:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0xb00
	v_lshl_or_b32 v20, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v15, v[20:21], off
.LBB0_24:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_26
; %bb.25:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0xc00
	v_lshl_or_b32 v20, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v20, v[20:21], off
.LBB0_26:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_28
; %bb.27:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0xd00
	v_lshl_or_b32 v22, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v19, v[22:23], off
.LBB0_28:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_30
; %bb.29:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v4, 0xe00
	v_lshl_or_b32 v22, v3, 6, v4
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v22, v[22:23], off
.LBB0_30:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v4, 0xc0, v0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_mov_b32_e32 v17, 0xf00
	v_lshl_or_b32 v24, v3, 6, v17
	.loc	1 93 24                         ; generate_rocjitsu_triton_fixtures.py:93:24
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[2:3]
	.loc	1 93 54                         ; generate_rocjitsu_triton_fixtures.py:93:54
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[16:17]
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	global_load_ushort v21, v[24:25], off
.LBB0_32:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[0:1]
	.loc	1 90 44 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:90:44
	v_and_b32_e32 v3, 64, v0
	v_and_b32_e32 v49, 31, v0
	v_lshrrev_b32_e32 v47, 1, v3
	.loc	1 90 31 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:90:31
	v_or3_b32 v23, s11, v49, v47
	s_movk_i32 s0, 0x400
	.loc	1 94 40 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:94:40
	v_cmp_gt_i32_e32 vcc, s0, v23
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	v_lshlrev_b32_e32 v23, 1, v0
	v_lshrrev_b32_e32 v4, 3, v4
	v_xor_b32_e32 v23, v23, v4
	v_add_u32_e32 v51, 0, v23
	s_waitcnt vmcnt(0)
	ds_write_b16 v51, v6
	ds_write_b16 v51, v10 offset:2048
	ds_write_b16 v51, v14 offset:4096
	ds_write_b16 v51, v20 offset:6144
	v_xor_b32_e32 v6, 32, v23
	v_add_u32_e32 v52, 0, v6
	ds_write_b16 v52, v5 offset:512
	ds_write_b16 v52, v9 offset:2560
	ds_write_b16 v52, v13 offset:4608
	ds_write_b16 v52, v19 offset:6656
	v_xor_b32_e32 v5, 64, v23
	v_add_u32_e32 v53, 0, v5
	v_xor_b32_e32 v5, 0x60, v23
	.loc	1 90 44                         ; generate_rocjitsu_triton_fixtures.py:90:44
	v_and_b32_e32 v48, 32, v0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	v_add_u32_e32 v54, 0, v5
	v_and_b32_e32 v6, 15, v0
	.loc	1 90 44                         ; generate_rocjitsu_triton_fixtures.py:90:44
	v_and_b32_e32 v50, 0x80, v0
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	ds_write_b16 v53, v8 offset:1024
	ds_write_b16 v53, v12 offset:3072
	ds_write_b16 v53, v18 offset:5120
	ds_write_b16 v53, v22 offset:7168
	ds_write_b16 v54, v7 offset:1536
	ds_write_b16 v54, v11 offset:3584
	ds_write_b16 v54, v15 offset:5632
	ds_write_b16 v54, v21 offset:7680
	v_lshlrev_b32_e32 v7, 3, v6
	v_lshrrev_b32_e32 v8, 2, v48
	v_lshlrev_b32_e32 v15, 2, v0
	v_lshl_add_u32 v34, v48, 4, 0
	v_lshlrev_b32_e32 v5, 7, v49
	v_lshlrev_b32_e32 v9, 6, v3
	v_xor_b32_e32 v10, v7, v8
	v_xor_b32_e32 v57, 0x80, v15
	v_lshlrev_b32_e32 v15, 1, v3
	v_add_u32_e32 v39, v34, v3
	v_lshrrev_b32_e32 v3, 1, v50
	v_or3_b32 v9, v10, v9, v5
	v_add_u32_e32 v40, v34, v3
	v_lshrrev_b32_e32 v3, 1, v0
	v_add_u32_e32 v10, 0, v9
	v_xad_u32 v11, v9, 16, 0
	v_xad_u32 v12, v9, 32, 0
	v_and_b32_e32 v3, 24, v3
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_xad_u32 v13, v9, 48, 0
	ds_read_b64 v[18:19], v10
	ds_read_b64 v[20:21], v11
	ds_read_b64 v[22:23], v12
	ds_read_b64 v[24:25], v13
	v_xad_u32 v10, v9, 64, 0
	v_xor_b32_e32 v11, 0x50, v9
	v_xor_b32_e32 v12, 0x60, v9
	v_xor_b32_e32 v9, 0x70, v9
	v_xor_b32_e32 v3, v7, v3
	.loc	1 90 44                         ; generate_rocjitsu_triton_fixtures.py:90:44
	v_bfe_i32 v17, v0, 6, 1
	.loc	1 93 16                         ; generate_rocjitsu_triton_fixtures.py:93:16
	v_add_u32_e32 v11, 0, v11
	v_add_u32_e32 v12, 0, v12
	v_add_u32_e32 v9, 0, v9
	ds_read_b64 v[26:27], v10
	ds_read_b64 v[28:29], v11
	ds_read_b64 v[30:31], v12
	ds_read_b64 v[32:33], v9
	v_xor_b32_e32 v3, v3, v4
	v_bfe_i32 v0, v0, 4, 1
	s_movk_i32 s0, 0x1010
	v_lshl_or_b32 v59, v2, 7, v3
	v_and_b32_e32 v0, 0x808, v0
	v_and_or_b32 v2, v17, s0, v8
	v_xor_b32_e32 v0, v0, v2
	v_or_b32_e32 v5, v5, v7
	v_xor_b32_e32 v0, v0, v7
	.loc	1 98 31                         ; generate_rocjitsu_triton_fixtures.py:98:31
	v_mov_b32_e32 v9, 0x3fb8aa3b
	v_xor_b32_e32 v56, v5, v8
	v_lshl_or_b32 v60, v6, 7, v0
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	v_mov_b32_e32 v17, 0
	.loc	1 98 31                         ; generate_rocjitsu_triton_fixtures.py:98:31
	v_mul_f32_e32 v55, s10, v9
	v_xor_b32_e32 v5, 16, v56
	v_xor_b32_e32 v9, 32, v56
	v_xor_b32_e32 v10, 48, v56
	v_xor_b32_e32 v11, 64, v56
	v_xor_b32_e32 v12, 0x50, v56
	v_xor_b32_e32 v13, 0x60, v56
	v_xor_b32_e32 v14, 0x70, v56
	v_lshl_add_u32 v58, v49, 2, 0
	v_lshlrev_b32_e32 v38, 1, v49
	v_xor_b32_e32 v4, 32, v59
	v_xor_b32_e32 v41, 64, v59
	v_xor_b32_e32 v42, 0x60, v59
	v_xor_b32_e32 v6, 16, v60
	v_xor_b32_e32 v7, 32, v60
	v_xor_b32_e32 v8, 48, v60
	v_xor_b32_e32 v43, 64, v60
	v_xor_b32_e32 v44, 0x50, v60
	v_xor_b32_e32 v45, 0x60, v60
	v_xor_b32_e32 v46, 0x70, v60
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	v_lshlrev_b32_e32 v2, 7, v1
	v_mov_b32_e32 v3, v17
	v_lshlrev_b32_e32 v0, 9, v1
	v_mov_b32_e32 v1, v17
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
	v_lshl_add_u64 v[34:35], s[4:5], 0, v[2:3]
	v_lshl_add_u64 v[36:37], s[6:7], 0, v[0:1]
	v_mov_b32_e32 v61, 0xe0ad78ec
	s_movk_i32 s4, 0xffc0
	s_movk_i32 s5, 0x1000
	v_add_u32_e32 v62, 0, v5
	v_add_u32_e32 v63, 0, v9
	v_add_u32_e32 v64, 0, v10
	v_add_u32_e32 v65, 0, v11
	v_add_u32_e32 v66, 0, v12
	v_add_u32_e32 v67, 0, v13
	v_add_u32_e32 v68, 0, v14
	v_add_u32_e32 v69, v39, v38
	v_add_u32_e32 v70, v40, v38
	s_mov_b32 s6, 0x5040100
	v_add_u32_e32 v71, 0, v4
	v_add_u32_e32 v72, 0, v41
	v_add_u32_e32 v74, 0, v42
	v_add_u32_e32 v75, 0, v6
	v_add_u32_e32 v76, 0, v7
	v_add_u32_e32 v77, 0, v8
	v_add_u32_e32 v78, 0, v43
	v_add_u32_e32 v79, 0, v44
	v_add_u32_e32 v80, 0, v45
	v_add_u32_e32 v81, 0, v46
	s_mov_b64 s[2:3], 0x2000
	v_add_u32_e32 v73, v58, v15
	v_mov_b32_e32 v83, v17
	v_mov_b32_e32 v8, 0xe0ad78ec
.LBB0_33:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 0 48 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:48
	v_mov_b32_e32 v84, v83
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	s_nop 8
	v_accvgpr_read_b32 v1, a15
	v_accvgpr_read_b32 v0, a14
	v_accvgpr_read_b32 v3, a13
	v_accvgpr_read_b32 v2, a12
	v_accvgpr_read_b32 v5, a11
	v_accvgpr_read_b32 v4, a10
	v_accvgpr_read_b32 v7, a9
	v_accvgpr_read_b32 v6, a8
	v_accvgpr_read_b32 v39, a7
	v_accvgpr_read_b32 v38, a6
	v_accvgpr_read_b32 v41, a5
	v_accvgpr_read_b32 v40, a4
	v_accvgpr_read_b32 v43, a3
	v_accvgpr_read_b32 v42, a2
	v_accvgpr_read_b32 v45, a1
	v_accvgpr_read_b32 v44, a0
	; sched_barrier mask(0x00000000)
	.loc	1 102 53 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:102:53
	v_lshl_add_u64 v[10:11], v[34:35], 0, v[16:17]
	; iglp_opt mask(0x00000002)
	; sched_barrier mask(0x00000000)
	.loc	1 102 20 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:102:20
	global_load_ushort v9, v[10:11], off
	global_load_ushort v12, v[10:11], off offset:512
	global_load_ushort v13, v[10:11], off offset:1024
	global_load_ushort v14, v[10:11], off offset:1536
	global_load_ushort v15, v[10:11], off offset:2048
	global_load_ushort v46, v[10:11], off offset:2560
	global_load_ushort v82, v[10:11], off offset:3072
	global_load_ushort v83, v[10:11], off offset:3584
	v_add_co_u32_e64 v10, s[0:1], s5, v10
	.loc	1 99 48 is_stmt 1               ; generate_rocjitsu_triton_fixtures.py:99:48
	s_add_i32 s4, s4, 64
	.loc	1 102 20                        ; generate_rocjitsu_triton_fixtures.py:102:20
	s_nop 0
	v_addc_co_u32_e64 v11, s[0:1], 0, v11, s[0:1]
	global_load_ushort v85, v[10:11], off
	global_load_ushort v86, v[10:11], off offset:512
	global_load_ushort v87, v[10:11], off offset:1024
	global_load_ushort v88, v[10:11], off offset:1536
	global_load_ushort v89, v[10:11], off offset:2048
	global_load_ushort v90, v[10:11], off offset:2560
	global_load_ushort v91, v[10:11], off offset:3072
	global_load_ushort v92, v[10:11], off offset:3584
	s_waitcnt lgkmcnt(0)
	s_barrier
                                        ; kill: killed $vgpr10 killed $vgpr11
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[2:3]
	s_cmpk_lt_u32 s4, 0x3c0
	.loc	1 102 20                        ; generate_rocjitsu_triton_fixtures.py:102:20
	s_waitcnt vmcnt(15)
	ds_write_b16 v51, v9
	v_add_u32_e32 v9, 0, v56
	s_waitcnt vmcnt(14)
	ds_write_b16 v52, v12 offset:512
	s_waitcnt vmcnt(13)
	ds_write_b16 v53, v13 offset:1024
	s_waitcnt vmcnt(11)
	ds_write_b16 v51, v15 offset:2048
	s_waitcnt vmcnt(10)
	ds_write_b16 v52, v46 offset:2560
	s_waitcnt vmcnt(9)
	ds_write_b16 v53, v82 offset:3072
	ds_write_b16 v54, v14 offset:1536
	s_waitcnt vmcnt(8)
	ds_write_b16 v54, v83 offset:3584
	s_waitcnt vmcnt(7)
	ds_write_b16 v51, v85 offset:4096
	s_waitcnt vmcnt(6)
	ds_write_b16 v52, v86 offset:4608
	s_waitcnt vmcnt(5)
	ds_write_b16 v53, v87 offset:5120
	s_waitcnt vmcnt(4)
	ds_write_b16 v54, v88 offset:5632
	s_waitcnt vmcnt(3)
	ds_write_b16 v51, v89 offset:6144
	s_waitcnt vmcnt(2)
	ds_write_b16 v52, v90 offset:6656
	s_waitcnt vmcnt(1)
	ds_write_b16 v53, v91 offset:7168
	s_waitcnt vmcnt(0)
	ds_write_b16 v54, v92 offset:7680
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[10:13], v9 offset1:8
	ds_read2st64_b64 v[86:89], v62 offset1:8
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[10:11], v[18:19], 0
	.loc	1 102 20                        ; generate_rocjitsu_triton_fixtures.py:102:20
	ds_read2st64_b64 v[90:93], v63 offset1:8
	ds_read2st64_b64 v[94:97], v64 offset1:8
	ds_read2st64_b64 v[98:101], v65 offset1:8
	ds_read2st64_b64 v[102:105], v66 offset1:8
	ds_read2st64_b64 v[106:109], v67 offset1:8
	ds_read2st64_b64 v[110:113], v68 offset1:8
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 a[0:15], v[86:87], v[20:21], a[0:15]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 a[0:15], v[90:91], v[22:23], a[0:15]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 a[0:15], v[94:95], v[24:25], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[98:99], v[26:27], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[102:103], v[28:29], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[106:107], v[30:31], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[110:111], v[32:33], a[0:15]
	s_nop 10
	v_accvgpr_read_b32 v9, a0
	v_accvgpr_read_b32 v10, a1
	v_accvgpr_read_b32 v11, a2
	v_accvgpr_read_b32 v14, a3
	v_accvgpr_read_b32 v15, a4
	v_accvgpr_read_b32 v46, a5
	v_accvgpr_read_b32 v82, a6
	v_accvgpr_read_b32 v83, a7
	v_accvgpr_read_b32 v85, a8
	v_accvgpr_read_b32 v86, a9
	v_accvgpr_read_b32 v87, a10
	v_accvgpr_read_b32 v90, a11
	v_accvgpr_read_b32 v91, a12
	v_accvgpr_read_b32 v94, a13
	v_accvgpr_read_b32 v95, a14
	v_accvgpr_read_b32 v98, a15
	v_mfma_f32_32x32x8_f16 a[0:15], v[12:13], v[18:19], 0
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v9, v55, v9
	v_mul_f32_e32 v10, v55, v10
	v_mul_f32_e32 v11, v55, v11
	v_mul_f32_e32 v14, v55, v14
	v_mul_f32_e32 v46, v55, v46
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v9, v61, v9, vcc
	v_cndmask_b32_e32 v10, v61, v10, vcc
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[88:89], v[20:21], a[0:15]
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v15, v55, v15
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v11, v61, v11, vcc
	v_cndmask_b32_e32 v14, v61, v14, vcc
	v_cndmask_b32_e32 v107, v61, v46, vcc
.Ltmp2:
	.file	2 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/.env/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max_f32_e32 v46, v9, v10
.Ltmp3:
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v82, v55, v82
	v_mul_f32_e32 v83, v55, v83
	.loc	1 104 23 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[92:93], v[22:23], a[0:15]
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v15, v61, v15, vcc
.Ltmp4:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v11, v14
.Ltmp5:
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v85, v55, v85
	v_mul_f32_e32 v86, v55, v86
	.loc	1 105 77                        ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v83, v61, v83, vcc
.Ltmp6:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v15, v107
.Ltmp7:
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v87, v55, v87
	.loc	1 104 23 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[96:97], v[24:25], a[0:15]
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v90, v55, v90
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v85, v61, v85, vcc
	v_cndmask_b32_e32 v86, v61, v86, vcc
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v91, v55, v91
	v_mul_f32_e32 v94, v55, v94
	.loc	1 105 77                        ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v87, v61, v87, vcc
	v_cndmask_b32_e32 v90, v61, v90, vcc
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[100:101], v[26:27], a[0:15]
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v95, v55, v95
	v_mul_f32_e32 v98, v55, v98
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v91, v61, v91, vcc
	v_cndmask_b32_e32 v94, v61, v94, vcc
	v_cndmask_b32_e32 v95, v61, v95, vcc
	v_cndmask_b32_e32 v98, v61, v98, vcc
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[104:105], v[28:29], a[0:15]
	v_mfma_f32_32x32x8_f16 a[0:15], v[108:109], v[30:31], a[0:15]
	.loc	1 105 77                        ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v108, v61, v82, vcc
.Ltmp8:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v108, v83
	v_max3_f32 v46, v46, v85, v86
	v_max3_f32 v46, v46, v87, v90
	v_max3_f32 v46, v46, v91, v94
	v_max3_f32 v46, v46, v95, v98
.Ltmp9:
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[112:113], v[32:33], a[0:15]
	s_nop 10
	v_accvgpr_read_b32 v12, a0
	v_accvgpr_read_b32 v13, a1
	v_accvgpr_read_b32 v88, a2
	v_accvgpr_read_b32 v89, a3
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v12, v55, v12
	v_mul_f32_e32 v13, v55, v13
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v92, a4
	v_accvgpr_read_b32 v93, a5
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v88, v55, v88
	v_mul_f32_e32 v89, v55, v89
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v12, v61, v12, vcc
	v_cndmask_b32_e32 v13, v61, v13, vcc
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v96, a6
	v_accvgpr_read_b32 v97, a7
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v92, v55, v92
	v_mul_f32_e32 v93, v55, v93
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v88, v61, v88, vcc
	v_cndmask_b32_e32 v89, v61, v89, vcc
.Ltmp10:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v12, v13
.Ltmp11:
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v99, a8
	v_accvgpr_read_b32 v100, a9
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v96, v55, v96
	v_mul_f32_e32 v97, v55, v97
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v92, v61, v92, vcc
	v_cndmask_b32_e32 v93, v61, v93, vcc
.Ltmp12:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v88, v89
.Ltmp13:
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v101, a10
	v_accvgpr_read_b32 v102, a11
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v99, v55, v99
	v_mul_f32_e32 v100, v55, v100
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v96, v61, v96, vcc
	v_cndmask_b32_e32 v97, v61, v97, vcc
.Ltmp14:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v92, v93
.Ltmp15:
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v103, a12
	v_accvgpr_read_b32 v104, a13
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v101, v55, v101
	v_mul_f32_e32 v102, v55, v102
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v99, v61, v99, vcc
	v_cndmask_b32_e32 v100, v61, v100, vcc
.Ltmp16:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v96, v97
.Ltmp17:
	.loc	1 104 23                        ; generate_rocjitsu_triton_fixtures.py:104:23
	v_accvgpr_read_b32 v105, a14
	v_accvgpr_read_b32 v106, a15
	.loc	1 104 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v103, v55, v103
	v_mul_f32_e32 v104, v55, v104
	.loc	1 105 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v101, v61, v101, vcc
	v_cndmask_b32_e32 v102, v61, v102, vcc
.Ltmp18:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v99, v100
.Ltmp19:
	.loc	1 104 50                        ; generate_rocjitsu_triton_fixtures.py:104:50
	v_mul_f32_e32 v105, v55, v105
	v_mul_f32_e32 v106, v55, v106
	.loc	1 105 77                        ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v103, v61, v103, vcc
	v_cndmask_b32_e32 v104, v61, v104, vcc
.Ltmp20:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v101, v102
.Ltmp21:
	.loc	1 105 77                        ; generate_rocjitsu_triton_fixtures.py:105:77
	v_cndmask_b32_e32 v105, v61, v105, vcc
	v_cndmask_b32_e32 v106, v61, v106, vcc
.Ltmp22:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ] ]
	v_max3_f32 v46, v46, v103, v104
	v_max3_f32 v46, v46, v105, v106
.Ltmp23:
	.loc	2 191 40                        ; standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:106:39 ]
	ds_bpermute_b32 v82, v57, v46
.Ltmp24:
	.loc	1 106 32                        ; generate_rocjitsu_triton_fixtures.py:106:32
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v82, v8, v46, v82
	.loc	1 107 35                        ; generate_rocjitsu_triton_fixtures.py:107:35
	v_sub_f32_e32 v8, v8, v82
	.loc	1 107 29 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:107:29
	v_exp_f32_e32 v46, v8
	.loc	1 108 30 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v8, v9, v82
	v_sub_f32_e32 v9, v10, v82
	v_sub_f32_e32 v10, v11, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v11, v14, v82
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v10, v10
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v14, v15, v82
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v11, v11
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v15, v107, v82
	v_sub_f32_e32 v107, v108, v82
	v_sub_f32_e32 v83, v83, v82
	v_sub_f32_e32 v108, v12, v82
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v12, v14
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v109, v13, v82
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v13, v15
	v_exp_f32_e32 v15, v83
.Ltmp25:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v8, v9
.Ltmp26:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v14, v107
.Ltmp27:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v10, v83
.Ltmp28:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v85, v85, v82
.Ltmp29:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v11, v83
.Ltmp30:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v86, v86, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v85, v85
.Ltmp31:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v12, v83
.Ltmp32:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v87, v87, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v86, v86
.Ltmp33:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v13, v83
.Ltmp34:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v90, v90, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v87, v87
.Ltmp35:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v14, v83
.Ltmp36:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v91, v91, v82
	v_sub_f32_e32 v110, v88, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v88, v90
.Ltmp37:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v15, v83
.Ltmp38:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v94, v94, v82
	v_sub_f32_e32 v111, v89, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v89, v91
.Ltmp39:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v85, v83
.Ltmp40:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v95, v95, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v90, v94
.Ltmp41:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v86, v83
.Ltmp42:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v98, v98, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v91, v95
.Ltmp43:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v87, v83
.Ltmp44:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v112, v92, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v92, v98
.Ltmp45:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v88, v83
.Ltmp46:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v113, v93, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v93, v108
.Ltmp47:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v89, v83
.Ltmp48:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v94, v109
.Ltmp49:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v90, v83
.Ltmp50:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v95, v110
.Ltmp51:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v91, v83
.Ltmp52:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v114, v96, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v96, v111
.Ltmp53:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v92, v83
.Ltmp54:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v115, v97, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v97, v112
.Ltmp55:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v93, v83
.Ltmp56:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v98, v113
.Ltmp57:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v94, v83
.Ltmp58:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v116, v99, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v99, v114
.Ltmp59:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v95, v83
.Ltmp60:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v117, v100, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v100, v115
.Ltmp61:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v96, v83
.Ltmp62:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v118, v101, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v101, v116
.Ltmp63:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v97, v83
.Ltmp64:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v119, v102, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v102, v117
.Ltmp65:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v98, v83
.Ltmp66:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v120, v103, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v103, v118
.Ltmp67:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v99, v83
.Ltmp68:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v121, v104, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v104, v119
.Ltmp69:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v100, v83
.Ltmp70:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v122, v105, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v105, v120
.Ltmp71:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v101, v83
.Ltmp72:
	.loc	1 108 30                        ; generate_rocjitsu_triton_fixtures.py:108:30
	v_sub_f32_e32 v123, v106, v82
	.loc	1 108 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v106, v121
.Ltmp73:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v102, v83
.Ltmp74:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v107, v122
.Ltmp75:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v103, v83
.Ltmp76:
	.loc	1 108 25                        ; generate_rocjitsu_triton_fixtures.py:108:25
	v_exp_f32_e32 v108, v123
.Ltmp77:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	v_add_f32_e32 v83, v104, v83
	v_add_f32_e32 v83, v105, v83
	v_add_f32_e32 v83, v106, v83
	v_add_f32_e32 v83, v107, v83
	v_add_f32_e32 v83, v108, v83
.Ltmp78:
	.loc	2 293 36                        ; standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ]
	ds_bpermute_b32 v109, v57, v83
.Ltmp79:
	.loc	1 110 53                        ; generate_rocjitsu_triton_fixtures.py:110:53
	v_lshl_add_u64 v[114:115], v[36:37], 0, v[16:17]
	.loc	1 110 20 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:110:20
	global_load_ushort v121, v[114:115], off
	global_load_ushort v124, v[114:115], off offset:128
	global_load_ushort v125, v[114:115], off offset:256
	global_load_ushort v126, v[114:115], off offset:384
	global_load_ushort v110, v[114:115], off offset:2176
	global_load_ushort v111, v[114:115], off offset:2304
	global_load_ushort v113, v[114:115], off offset:2432
	v_add_co_u32_e64 v122, s[0:1], s5, v114
.Ltmp80:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:109:37 ] ]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v83, v83, v109
.Ltmp81:
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	global_load_ushort v109, v[114:115], off offset:2048
	v_addc_co_u32_e64 v123, s[0:1], 0, v115, s[0:1]
	global_load_ushort v112, v[122:123], off
	global_load_ushort v114, v[122:123], off offset:128
	global_load_ushort v117, v[122:123], off offset:256
	global_load_ushort v118, v[122:123], off offset:384
	global_load_ushort v115, v[122:123], off offset:2048
	global_load_ushort v116, v[122:123], off offset:2176
	global_load_ushort v119, v[122:123], off offset:2304
	global_load_ushort v120, v[122:123], off offset:2432
	.loc	1 113 26                        ; generate_rocjitsu_triton_fixtures.py:113:26
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v86, v86
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v11, v11
	v_cvt_f16_f32_e32 v12, v12
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v85, v85
	v_cvt_f16_f32_e32 v87, v87
	v_cvt_f16_f32_e32 v88, v88
	v_cvt_f16_f32_e32 v89, v89
	v_cvt_f16_f32_e32 v90, v90
	v_cvt_f16_f32_e32 v91, v91
	v_cvt_f16_f32_e32 v92, v92
	v_cvt_f16_f32_e32 v93, v93
	v_cvt_f16_f32_e32 v94, v94
	v_cvt_f16_f32_e32 v95, v95
	v_cvt_f16_f32_e32 v96, v96
	v_cvt_f16_f32_e32 v97, v97
	v_cvt_f16_f32_e32 v98, v98
	v_cvt_f16_f32_e32 v99, v99
	v_cvt_f16_f32_e32 v100, v100
	v_cvt_f16_f32_e32 v101, v101
	v_cvt_f16_f32_e32 v102, v102
	v_cvt_f16_f32_e32 v103, v103
	v_cvt_f16_f32_e32 v104, v104
	v_cvt_f16_f32_e32 v105, v105
	v_cvt_f16_f32_e32 v106, v106
	v_cvt_f16_f32_e32 v107, v107
	v_cvt_f16_f32_e32 v108, v108
	.loc	1 109 30                        ; generate_rocjitsu_triton_fixtures.py:109:30
	v_fmac_f32_e32 v83, v84, v46
	.loc	1 112 20                        ; generate_rocjitsu_triton_fixtures.py:112:20
	v_add_u32_e32 v84, v58, v50
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v73, v46
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v46, v84
	.loc	1 113 26                        ; generate_rocjitsu_triton_fixtures.py:113:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v69, v8
	ds_write_b16 v69, v9 offset:128
	ds_write_b16 v69, v86 offset:2176
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	v_add_u32_e32 v86, 0, v59
                                        ; kill: killed $vgpr122 killed $vgpr123
	.loc	1 113 26                        ; generate_rocjitsu_triton_fixtures.py:113:26
	ds_write_b16 v69, v10 offset:256
	ds_write_b16 v69, v11 offset:384
	ds_write_b16 v69, v12 offset:1024
	ds_write_b16 v69, v13 offset:1152
	ds_write_b16 v69, v14 offset:1280
	ds_write_b16 v69, v15 offset:1408
	ds_write_b16 v69, v85 offset:2048
	ds_write_b16 v69, v87 offset:2304
	ds_write_b16 v69, v88 offset:2432
	ds_write_b16 v69, v89 offset:3072
	ds_write_b16 v69, v90 offset:3200
	ds_write_b16 v69, v91 offset:3328
	ds_write_b16 v69, v92 offset:3456
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	s_waitcnt vmcnt(14)
	v_perm_b32 v8, v124, v121, s6
	.loc	1 113 26                        ; generate_rocjitsu_triton_fixtures.py:113:26
	ds_write_b16 v69, v93 offset:4096
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	s_waitcnt vmcnt(12)
	v_perm_b32 v9, v126, v125, s6
	.loc	1 113 26                        ; generate_rocjitsu_triton_fixtures.py:113:26
	ds_write_b16 v69, v94 offset:4224
	ds_write_b16 v69, v95 offset:4352
	ds_write_b16 v69, v96 offset:4480
	ds_write_b16 v69, v97 offset:5120
	ds_write_b16 v69, v98 offset:5248
	ds_write_b16 v69, v99 offset:5376
	ds_write_b16 v69, v100 offset:5504
	ds_write_b16 v69, v101 offset:6144
	ds_write_b16 v69, v102 offset:6272
	ds_write_b16 v69, v103 offset:6400
	ds_write_b16 v69, v104 offset:6528
	ds_write_b16 v69, v105 offset:7168
	ds_write_b16 v69, v106 offset:7296
	ds_write_b16 v69, v107 offset:7424
	ds_write_b16 v69, v108 offset:7552
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v10, v70
	ds_read_u16 v11, v70 offset:128
	ds_read_u16 v12, v70 offset:256
	ds_read_u16 v13, v70 offset:384
	ds_read_u16 v14, v70 offset:1024
	ds_read_u16 v15, v70 offset:1152
	ds_read_u16 v85, v70 offset:1280
	ds_read_u16 v104, v70 offset:1408
	ds_read_u16 v106, v70 offset:2048
	ds_read_u16 v108, v70 offset:2176
	ds_read_u16 v107, v70 offset:2304
	ds_read_u16 v122, v70 offset:2432
	ds_read_u16 v123, v70 offset:3072
	ds_read_u16 v127, v70 offset:3200
	ds_read_u16 v128, v70 offset:3328
	ds_read_u16 v129, v70 offset:3456
	ds_read_u16 v130, v70 offset:4096
	ds_read_u16 v131, v70 offset:4224
	ds_read_u16 v132, v70 offset:4352
	ds_read_u16 v133, v70 offset:4480
	ds_read_u16 v134, v70 offset:5120
	ds_read_u16 v135, v70 offset:5248
	ds_read_u16 v136, v70 offset:5376
	ds_read_u16 v137, v70 offset:5504
	ds_read_u16 v138, v70 offset:6144
	ds_read_u16 v139, v70 offset:6272
	ds_read_u16 v140, v70 offset:6400
	ds_read_u16 v141, v70 offset:6528
	ds_read_u16 v142, v70 offset:7168
	ds_read_u16 v143, v70 offset:7296
	ds_read_u16 v144, v70 offset:7424
	ds_read_u16 v145, v70 offset:7552
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b64 v86, v[8:9]
	s_waitcnt vmcnt(9)
	v_perm_b32 v9, v113, v111, s6
	s_waitcnt vmcnt(8)
	v_perm_b32 v8, v110, v109, s6
	ds_write_b64 v71, v[8:9]
	s_waitcnt vmcnt(4)
	v_perm_b32 v9, v118, v117, s6
	v_perm_b32 v8, v114, v112, s6
	ds_write_b64 v72, v[8:9]
	s_waitcnt vmcnt(0)
	v_perm_b32 v9, v120, v119, s6
	v_perm_b32 v8, v116, v115, s6
	ds_write_b64 v74, v[8:9]
	v_add_u32_e32 v8, 0, v60
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[86:87], v8
	ds_read_b64 v[88:89], v75
	ds_read_b64 v[90:91], v76
	ds_read_b64 v[92:93], v77
	.loc	1 113 42                        ; generate_rocjitsu_triton_fixtures.py:113:42
	v_perm_b32 v105, v104, v85, s6
	v_perm_b32 v104, v15, v14, s6
	v_pk_mul_f32 v[14:15], v[0:1], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[44:45], v[46:47] op_sel_hi:[1,0]
	v_perm_b32 v103, v13, v12, s6
	v_perm_b32 v102, v11, v10, s6
	v_pk_mul_f32 v[12:13], v[2:3], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[4:5], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[6:7], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[38:39], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[40:41], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[42:43], v[46:47] op_sel_hi:[1,0]
	v_perm_b32 v107, v122, v107, s6
	v_accvgpr_write_b32 a0, v0
	v_accvgpr_write_b32 a1, v1
	v_accvgpr_write_b32 a2, v2
	v_accvgpr_write_b32 a3, v3
	v_accvgpr_write_b32 a4, v4
	v_accvgpr_write_b32 a5, v5
	v_accvgpr_write_b32 a6, v6
	v_accvgpr_write_b32 a7, v7
	v_accvgpr_write_b32 a8, v8
	v_accvgpr_write_b32 a9, v9
	v_accvgpr_write_b32 a10, v10
	v_accvgpr_write_b32 a11, v11
	v_accvgpr_write_b32 a12, v12
	v_accvgpr_write_b32 a13, v13
	v_accvgpr_write_b32 a14, v14
	v_accvgpr_write_b32 a15, v15
	v_perm_b32 v106, v108, v106, s6
	.loc	1 110 20                        ; generate_rocjitsu_triton_fixtures.py:110:20
	ds_read_b64 v[94:95], v78
	ds_read_b64 v[96:97], v79
	ds_read_b64 v[98:99], v80
	ds_read_b64 v[100:101], v81
	.loc	1 113 42                        ; generate_rocjitsu_triton_fixtures.py:113:42
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x8_f16 a[0:15], v[86:87], v[102:103], a[0:15]
	v_perm_b32 v109, v129, v128, s6
	v_perm_b32 v108, v127, v123, s6
	v_perm_b32 v111, v133, v132, s6
	v_perm_b32 v110, v131, v130, s6
	v_perm_b32 v113, v137, v136, s6
	v_perm_b32 v112, v135, v134, s6
	v_perm_b32 v115, v141, v140, s6
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 a[0:15], v[88:89], v[104:105], a[0:15]
	v_perm_b32 v114, v139, v138, s6
	v_perm_b32 v117, v145, v144, s6
	v_perm_b32 v116, v143, v142, s6
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[2:3]
	v_mov_b32_e32 v8, v82
	.loc	1 113 42                        ; generate_rocjitsu_triton_fixtures.py:113:42
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 a[0:15], v[90:91], v[106:107], a[0:15]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 a[0:15], v[92:93], v[108:109], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[94:95], v[110:111], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[96:97], v[112:113], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[98:99], v[114:115], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[100:101], v[116:117], a[0:15]
	.loc	1 99 48                         ; generate_rocjitsu_triton_fixtures.py:99:48
	s_cbranch_scc1 .LBB0_33
; %bb.34:
	.loc	1 90 44                         ; generate_rocjitsu_triton_fixtures.py:90:44
	v_lshrrev_b32_e32 v16, 2, v50
	.loc	1 113 42                        ; generate_rocjitsu_triton_fixtures.py:113:42
	s_nop 8
	v_accvgpr_read_b32 v0, a0
	.loc	1 90 31                         ; generate_rocjitsu_triton_fixtures.py:90:31
	v_or3_b32 v16, v49, v16, s11
	s_movk_i32 s0, 0x400
	.loc	1 113 42                        ; generate_rocjitsu_triton_fixtures.py:113:42
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	.loc	1 94 40                         ; generate_rocjitsu_triton_fixtures.py:94:40
	v_cmp_gt_i32_e32 vcc, s0, v16
	.loc	1 116 16                        ; generate_rocjitsu_triton_fixtures.py:116:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v73, v83
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 117 80                        ; generate_rocjitsu_triton_fixtures.py:117:80
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_36
; %bb.35:                               ; %.critedge
	.loc	1 116 16                        ; generate_rocjitsu_triton_fixtures.py:116:16
	ds_read_b32 v20, v84
	.loc	1 90 44                         ; generate_rocjitsu_triton_fixtures.py:90:44
	v_lshrrev_b32_e32 v17, 3, v48
	.loc	1 93 42                         ; generate_rocjitsu_triton_fixtures.py:93:42
	v_lshlrev_b32_e32 v16, 6, v16
	.loc	1 94 57                         ; generate_rocjitsu_triton_fixtures.py:94:57
	v_or_b32_e32 v18, v17, v47
	.loc	1 117 21                        ; generate_rocjitsu_triton_fixtures.py:117:21
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 116 16                        ; generate_rocjitsu_triton_fixtures.py:116:16
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v21, s[0:1], v20, v20, v15
	v_rcp_f32_e32 v22, v21
	.loc	1 117 21                        ; generate_rocjitsu_triton_fixtures.py:117:21
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[8:9]
	.loc	1 117 51 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:117:51
	v_lshlrev_b32_e32 v18, 2, v18
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	.loc	1 116 16 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:116:16
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v15, v20, v15
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v14
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v15, v18, v20, v15
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v14, v20, v14
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v13
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v14, v18, v20, v14
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v13, v20, v13
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v12
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v13, v18, v20, v13
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v12, v20, v12
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v11
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v12, v18, v20, v12
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v11, v20, v11
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v10
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v11, v18, v20, v11
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v10, v20, v10
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v9
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v10, v18, v20, v10
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v9, v20, v9
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v8
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v9, v18, v20, v9
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v8, v20, v8
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v7
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v8, v18, v20, v8
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v7, v20, v7
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v6
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v7, v18, v20, v7
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v6, v20, v6
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v5
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v6, v18, v20, v6
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v5, v20, v5
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v4
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v5, v18, v20, v5
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v4, v20, v4
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v3
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v4, v18, v20, v4
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v3, v20, v3
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v2
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v3, v18, v20, v3
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v2, v20, v2
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v0
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v2, v18, v20, v2
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v0, v20, v0
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v1
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v0, v18, v20, v0
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v1, v20, v1
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v1, v18, v20, v1
	.loc	1 117 80                        ; generate_rocjitsu_triton_fixtures.py:117:80
	global_store_dwordx4 v[16:17], v[0:3], off
	global_store_dwordx4 v[16:17], v[4:7], off offset:32
	global_store_dwordx4 v[16:17], v[8:11], off offset:64
	global_store_dwordx4 v[16:17], v[12:15], off offset:96
.LBB0_36:                               ; %.critedge28
	.loc	1 117 4 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:117:4
	s_endpgm
.Ltmp82:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flash_attention_fwd_no_async_kernel
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 164
		.amdhsa_next_free_sgpr 17
		.amdhsa_accum_offset 148
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
	.size	flash_attention_fwd_no_async_kernel, .Lfunc_end0-flash_attention_fwd_no_async_kernel
	.cfi_endproc
                                        ; -- End function
	.set flash_attention_fwd_no_async_kernel.num_vgpr, 146
	.set flash_attention_fwd_no_async_kernel.num_agpr, 16
	.set flash_attention_fwd_no_async_kernel.numbered_sgpr, 17
	.set flash_attention_fwd_no_async_kernel.num_named_barrier, 0
	.set flash_attention_fwd_no_async_kernel.private_seg_size, 0
	.set flash_attention_fwd_no_async_kernel.uses_vcc, 1
	.set flash_attention_fwd_no_async_kernel.uses_flat_scratch, 0
	.set flash_attention_fwd_no_async_kernel.has_dyn_sized_stack, 0
	.set flash_attention_fwd_no_async_kernel.has_recursion, 0
	.set flash_attention_fwd_no_async_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6944
; TotalNumSgprs: 23
; NumVgprs: 146
; NumAgprs: 16
; TotalNumVgprs: 164
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 20
; NumSGPRsForWavesPerEU: 23
; NumVGPRsForWavesPerEU: 164
; AccumOffset: 148
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 36
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
	.byte	1                               ; DW_CHILDREN_yes
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
	.byte	5                               ; Abbreviation Code
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
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
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
	.byte	1                               ; Abbrev [1] 0xb:0x6b DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x45 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x19 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	106                             ; DW_AT_call_line
	.byte	39                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x4d:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	191                             ; DW_AT_call_line
	.byte	40                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	4                               ; Abbrev [4] 0x5a:0x1a DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges2                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	109                             ; DW_AT_call_line
	.byte	37                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x66:0xd DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges3                 ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.short	293                             ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp2-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp22-.Lfunc_begin0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp32-.Lfunc_begin0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp37-.Lfunc_begin0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp42-.Lfunc_begin0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp47-.Lfunc_begin0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Ltmp52-.Lfunc_begin0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.quad	.Ltmp55-.Lfunc_begin0
	.quad	.Ltmp56-.Lfunc_begin0
	.quad	.Ltmp57-.Lfunc_begin0
	.quad	.Ltmp58-.Lfunc_begin0
	.quad	.Ltmp59-.Lfunc_begin0
	.quad	.Ltmp60-.Lfunc_begin0
	.quad	.Ltmp61-.Lfunc_begin0
	.quad	.Ltmp62-.Lfunc_begin0
	.quad	.Ltmp63-.Lfunc_begin0
	.quad	.Ltmp64-.Lfunc_begin0
	.quad	.Ltmp65-.Lfunc_begin0
	.quad	.Ltmp66-.Lfunc_begin0
	.quad	.Ltmp67-.Lfunc_begin0
	.quad	.Ltmp68-.Lfunc_begin0
	.quad	.Ltmp69-.Lfunc_begin0
	.quad	.Ltmp70-.Lfunc_begin0
	.quad	.Ltmp71-.Lfunc_begin0
	.quad	.Ltmp72-.Lfunc_begin0
	.quad	.Ltmp73-.Lfunc_begin0
	.quad	.Ltmp74-.Lfunc_begin0
	.quad	.Ltmp75-.Lfunc_begin0
	.quad	.Ltmp76-.Lfunc_begin0
	.quad	.Ltmp77-.Lfunc_begin0
	.quad	.Ltmp78-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
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
	.asciz	"flash_attention_fwd_no_async_kernel" ; string offset=49
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
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
    .name:           flash_attention_fwd_no_async_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     23
    .sgpr_spill_count: 0
    .symbol:         flash_attention_fwd_no_async_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     164
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
