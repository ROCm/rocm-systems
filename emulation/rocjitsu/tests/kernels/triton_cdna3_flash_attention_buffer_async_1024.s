// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Native CDNA3/gfx942 Triton golden fixture generated from the same source and
// constexprs as the corresponding gfx950 DBT input fixture.
//
// Recorded fixture metadata:
//   kernel: flash_attention_fwd_async_buffer_kernel
//   target: gfx942, wavefront_size: 64
//   Triton metadata shared: 8192
//   num_warps: 4
//   num_stages: 2

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 5
	.text
	.globl	flash_attention_fwd_async_buffer_kernel ; -- Begin function flash_attention_fwd_async_buffer_kernel
	.p2align	8
	.type	flash_attention_fwd_async_buffer_kernel,@function
flash_attention_fwd_async_buffer_kernel: ; @flash_attention_fwd_async_buffer_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.37:
	.file	1 "/tmp" "generate_rocjitsu_triton_fixtures.py"
	.loc	1 122 0 prologue_end            ; generate_rocjitsu_triton_fixtures.py:122:0
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
	.loc	1 135 22 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:135:22
	s_lshl_b32 s11, s16, 6
	.loc	1 135 45 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:135:45
	v_lshrrev_b32_e32 v1, 6, v0
	s_mov_b64 s[0:1], s[6:7]
	.loc	1 135 32                        ; generate_rocjitsu_triton_fixtures.py:135:32
	v_or_b32_e32 v3, s11, v1
	s_movk_i32 s6, 0x400
	.loc	1 138 53 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:53
	v_and_b32_e32 v2, 63, v0
	.loc	1 139 59                        ; generate_rocjitsu_triton_fixtures.py:139:59
	v_cmp_gt_i32_e32 vcc, s6, v3
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	v_lshl_or_b32 v6, v3, 6, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[2:3]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v6, v[6:7], off
.LBB0_2:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_lshlrev_b32_e32 v3, 6, v3
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x100
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v4, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v5, v[4:5], off
.LBB0_4:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x200
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v8, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v8, v[8:9], off
.LBB0_6:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x300
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v10, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v7, v[10:11], off
.LBB0_8:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x400
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v10, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v10, v[10:11], off
.LBB0_10:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x500
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v12, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v9, v[12:13], off
.LBB0_12:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x600
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v12, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v12, v[12:13], off
.LBB0_14:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x700
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v14, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v11, v[14:15], off
.LBB0_16:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x800
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v14, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v14, v[14:15], off
.LBB0_18:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0x900
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v16, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v13, v[16:17], off
.LBB0_20:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xa00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v16, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v16, v[16:17], off
.LBB0_22:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xb00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v18, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v15, v[18:19], off
.LBB0_24:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_26
; %bb.25:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xc00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v18, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v18, v[18:19], off
.LBB0_26:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_28
; %bb.27:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xd00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v20, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v17, v[20:21], off
.LBB0_28:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_30
; %bb.29:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xe00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v20, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v20, v[20:21], off
.LBB0_30:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	v_and_b32_e32 v4, 0xc0, v0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_movk_i32 s12, 0xf00
	.loc	1 138 46 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v22, v3, v2, s12
	.loc	1 139 24                        ; generate_rocjitsu_triton_fixtures.py:139:24
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[2:3]
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	global_load_ushort v19, v[22:23], off
.LBB0_32:
	.loc	1 0 16                          ; generate_rocjitsu_triton_fixtures.py:0:16
	s_or_b64 exec, exec, s[6:7]
	.loc	1 135 45 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:135:45
	v_and_b32_e32 v3, 64, v0
	v_and_b32_e32 v67, 31, v0
	v_lshrrev_b32_e32 v65, 1, v3
	.loc	1 135 32 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:135:32
	v_or3_b32 v21, s11, v67, v65
	s_movk_i32 s2, 0x400
	.loc	1 139 59 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:139:59
	v_cmp_gt_i32_e32 vcc, s2, v21
	.loc	1 139 16 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:139:16
	v_lshlrev_b32_e32 v21, 1, v0
	v_lshrrev_b32_e32 v4, 3, v4
	v_xor_b32_e32 v21, v21, v4
	v_add_u32_e32 v69, 0, v21
	s_waitcnt vmcnt(0)
	ds_write_b16 v69, v6
	ds_write_b16 v69, v10 offset:2048
	ds_write_b16 v69, v14 offset:4096
	ds_write_b16 v69, v18 offset:6144
	v_xor_b32_e32 v6, 32, v21
	v_add_u32_e32 v70, 0, v6
	ds_write_b16 v70, v5 offset:512
	ds_write_b16 v70, v9 offset:2560
	ds_write_b16 v70, v13 offset:4608
	ds_write_b16 v70, v17 offset:6656
	v_xor_b32_e32 v5, 64, v21
	v_add_u32_e32 v71, 0, v5
	v_xor_b32_e32 v5, 0x60, v21
	.loc	1 135 45 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:135:45
	v_and_b32_e32 v66, 32, v0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	v_add_u32_e32 v72, 0, v5
	v_and_b32_e32 v6, 15, v0
	ds_write_b16 v71, v8 offset:1024
	ds_write_b16 v71, v12 offset:3072
	ds_write_b16 v71, v16 offset:5120
	ds_write_b16 v71, v20 offset:7168
	ds_write_b16 v72, v7 offset:1536
	ds_write_b16 v72, v11 offset:3584
	ds_write_b16 v72, v15 offset:5632
	ds_write_b16 v72, v19 offset:7680
	v_lshlrev_b32_e32 v7, 3, v6
	v_lshrrev_b32_e32 v8, 2, v66
	v_lshlrev_b32_e32 v5, 7, v67
	v_lshlrev_b32_e32 v9, 6, v3
	v_xor_b32_e32 v10, v7, v8
	v_or3_b32 v9, v10, v9, v5
	.loc	1 135 45                        ; generate_rocjitsu_triton_fixtures.py:135:45
	v_and_b32_e32 v68, 0x80, v0
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	v_add_u32_e32 v10, 0, v9
	v_xad_u32 v11, v9, 16, 0
	v_xad_u32 v12, v9, 32, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_xad_u32 v13, v9, 48, 0
	ds_read_b64 v[16:17], v10
	ds_read_b64 v[18:19], v11
	ds_read_b64 v[20:21], v12
	ds_read_b64 v[22:23], v13
	v_xad_u32 v10, v9, 64, 0
	v_xor_b32_e32 v11, 0x50, v9
	v_xor_b32_e32 v12, 0x60, v9
	v_xor_b32_e32 v9, 0x70, v9
	v_lshlrev_b32_e32 v15, 2, v0
	v_lshl_add_u32 v34, v66, 4, 0
	v_lshrrev_b32_e32 v35, 1, v68
	.loc	1 135 45                        ; generate_rocjitsu_triton_fixtures.py:135:45
	v_bfe_i32 v32, v0, 6, 1
	.loc	1 139 16                        ; generate_rocjitsu_triton_fixtures.py:139:16
	v_add_u32_e32 v11, 0, v11
	v_add_u32_e32 v12, 0, v12
	v_add_u32_e32 v9, 0, v9
	ds_read_b64 v[24:25], v10
	ds_read_b64 v[26:27], v11
	ds_read_b64 v[28:29], v12
	ds_read_b64 v[30:31], v9
	v_or_b32_e32 v5, v5, v7
	v_xor_b32_e32 v76, 0x80, v15
	v_lshlrev_b32_e32 v15, 1, v3
	v_add_u32_e32 v3, v34, v3
	v_add_u32_e32 v34, v34, v35
	v_lshrrev_b32_e32 v35, 1, v0
	v_bfe_i32 v0, v0, 4, 1
	s_movk_i32 s2, 0x1010
	v_xor_b32_e32 v75, v5, v8
	v_and_b32_e32 v35, 24, v35
	v_and_b32_e32 v0, 0x808, v0
	v_and_or_b32 v8, v32, s2, v8
	v_xor_b32_e32 v35, v7, v35
	v_xor_b32_e32 v0, v0, v8
	v_xor_b32_e32 v4, v35, v4
	v_xor_b32_e32 v0, v0, v7
	.loc	1 144 31                        ; generate_rocjitsu_triton_fixtures.py:144:31
	v_mov_b32_e32 v9, 0x3fb8aa3b
	v_lshl_or_b32 v78, v2, 7, v4
	v_lshl_or_b32 v79, v6, 7, v0
	v_mul_f32_e32 v73, s10, v9
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffe
	v_lshlrev_b32_e32 v74, 1, v2
	v_xor_b32_e32 v5, 16, v75
	v_xor_b32_e32 v9, 32, v75
	v_xor_b32_e32 v10, 48, v75
	v_xor_b32_e32 v11, 64, v75
	v_xor_b32_e32 v12, 0x50, v75
	v_xor_b32_e32 v13, 0x60, v75
	v_xor_b32_e32 v14, 0x70, v75
	v_lshl_add_u32 v77, v67, 2, 0
	v_lshlrev_b32_e32 v33, 1, v67
	v_xor_b32_e32 v2, 32, v78
	v_xor_b32_e32 v4, 64, v78
	v_xor_b32_e32 v35, 0x60, v78
	v_xor_b32_e32 v0, 16, v79
	v_xor_b32_e32 v6, 32, v79
	v_xor_b32_e32 v7, 48, v79
	v_xor_b32_e32 v8, 64, v79
	v_xor_b32_e32 v32, 0x50, v79
	v_xor_b32_e32 v36, 0x60, v79
	v_xor_b32_e32 v37, 0x70, v79
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s1, s1, 0xffff
	.loc	1 145 48                        ; generate_rocjitsu_triton_fixtures.py:145:48
	v_lshlrev_b32_e32 v80, 9, v1
	v_lshlrev_b32_e32 v81, 7, v1
	v_mov_b32_e32 v82, 0xe0ad78ec
	v_mov_b32_e32 v103, 0
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
	s_movk_i32 s10, 0xffc0
	v_add_u32_e32 v83, 0, v5
	v_add_u32_e32 v84, 0, v9
	v_add_u32_e32 v85, 0, v10
	v_add_u32_e32 v86, 0, v11
	v_add_u32_e32 v87, 0, v12
	v_add_u32_e32 v88, 0, v13
	v_add_u32_e32 v89, 0, v14
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	v_add_u32_e32 v90, v3, v33
	v_add_u32_e32 v91, v34, v33
	s_mov_b32 s12, 0x5040100
	v_add_u32_e32 v92, 0, v2
	v_add_u32_e32 v94, 0, v4
	v_add_u32_e32 v95, 0, v35
	v_add_u32_e32 v96, 0, v0
	v_add_u32_e32 v97, 0, v6
	v_add_u32_e32 v98, 0, v7
	v_add_u32_e32 v99, 0, v8
	v_add_u32_e32 v100, 0, v32
	v_add_u32_e32 v101, 0, v36
	v_add_u32_e32 v102, 0, v37
	v_add_u32_e32 v93, v77, v15
	v_mov_b32_e32 v33, 0xe0ad78ec
.LBB0_33:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 0 48 is_stmt 0                ; generate_rocjitsu_triton_fixtures.py:0:48
	v_mov_b32_e32 v32, v103
	.loc	1 145 48                        ; generate_rocjitsu_triton_fixtures.py:145:48
	s_nop 3
	v_accvgpr_read_b32 v15, a15
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v0, a0
	; sched_barrier mask(0x00000000)
	.loc	1 149 20 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:149:20
	v_add_u32_e32 v34, v74, v81
	buffer_load_ushort v35, v34, s[4:7], 0 offen
	buffer_load_ushort v36, v34, s[4:7], 0 offen offset:512
	buffer_load_ushort v37, v34, s[4:7], 0 offen offset:1024
	buffer_load_ushort v38, v34, s[4:7], 0 offen offset:1536
	buffer_load_ushort v39, v34, s[4:7], 0 offen offset:2048
	buffer_load_ushort v40, v34, s[4:7], 0 offen offset:2560
	buffer_load_ushort v41, v34, s[4:7], 0 offen offset:3072
	buffer_load_ushort v42, v34, s[4:7], 0 offen offset:3584
	v_add_u32_e32 v34, 0x1000, v34
	buffer_load_ushort v43, v34, s[4:7], 0 offen
	buffer_load_ushort v44, v34, s[4:7], 0 offen offset:512
	buffer_load_ushort v45, v34, s[4:7], 0 offen offset:1024
	buffer_load_ushort v46, v34, s[4:7], 0 offen offset:1536
	buffer_load_ushort v47, v34, s[4:7], 0 offen offset:2048
	buffer_load_ushort v48, v34, s[4:7], 0 offen offset:2560
	buffer_load_ushort v49, v34, s[4:7], 0 offen offset:3072
	buffer_load_ushort v50, v34, s[4:7], 0 offen offset:3584
                                        ; kill: killed $vgpr34
	v_add_u32_e32 v34, 0, v75
	s_waitcnt lgkmcnt(0)
	s_barrier
	; iglp_opt mask(0x00000002)
	s_waitcnt vmcnt(15)
	ds_write_b16 v69, v35
	s_waitcnt vmcnt(14)
	ds_write_b16 v70, v36 offset:512
	s_waitcnt vmcnt(13)
	ds_write_b16 v71, v37 offset:1024
	s_waitcnt vmcnt(12)
	ds_write_b16 v72, v38 offset:1536
	s_waitcnt vmcnt(11)
	ds_write_b16 v69, v39 offset:2048
	s_waitcnt vmcnt(10)
	ds_write_b16 v70, v40 offset:2560
	s_waitcnt vmcnt(9)
	ds_write_b16 v71, v41 offset:3072
	s_waitcnt vmcnt(8)
	ds_write_b16 v72, v42 offset:3584
	s_waitcnt vmcnt(7)
	ds_write_b16 v69, v43 offset:4096
	s_waitcnt vmcnt(6)
	ds_write_b16 v70, v44 offset:4608
	s_waitcnt vmcnt(5)
	ds_write_b16 v71, v45 offset:5120
	s_waitcnt vmcnt(4)
	ds_write_b16 v72, v46 offset:5632
	s_waitcnt vmcnt(3)
	ds_write_b16 v69, v47 offset:6144
	s_waitcnt vmcnt(2)
	ds_write_b16 v70, v48 offset:6656
	s_waitcnt vmcnt(1)
	ds_write_b16 v71, v49 offset:7168
	s_waitcnt vmcnt(0)
	ds_write_b16 v72, v50 offset:7680
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[34:37], v34 offset1:8
	ds_read2st64_b64 v[38:41], v83 offset1:8
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[34:35], v[16:17], 0
	.loc	1 149 20                        ; generate_rocjitsu_triton_fixtures.py:149:20
	ds_read2st64_b64 v[42:45], v84 offset1:8
	ds_read2st64_b64 v[46:49], v85 offset1:8
	ds_read2st64_b64 v[50:53], v86 offset1:8
	ds_read2st64_b64 v[54:57], v87 offset1:8
	ds_read2st64_b64 v[58:61], v88 offset1:8
	ds_read2st64_b64 v[104:107], v89 offset1:8
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x8_f16 a[0:15], v[38:39], v[18:19], a[0:15]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 a[0:15], v[42:43], v[20:21], a[0:15]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 a[0:15], v[46:47], v[22:23], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[50:51], v[24:25], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[54:55], v[26:27], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[58:59], v[28:29], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[104:105], v[30:31], a[0:15]
	s_nop 10
	v_accvgpr_read_b32 v34, a0
	v_accvgpr_read_b32 v35, a1
	v_accvgpr_read_b32 v38, a2
	v_accvgpr_read_b32 v39, a3
	v_accvgpr_read_b32 v42, a4
	v_accvgpr_read_b32 v43, a5
	v_accvgpr_read_b32 v46, a6
	v_accvgpr_read_b32 v47, a7
	v_accvgpr_read_b32 v50, a8
	v_accvgpr_read_b32 v51, a9
	v_accvgpr_read_b32 v54, a10
	v_accvgpr_read_b32 v55, a11
	v_accvgpr_read_b32 v58, a12
	v_accvgpr_read_b32 v59, a13
	v_accvgpr_read_b32 v62, a14
	v_accvgpr_read_b32 v63, a15
	v_mfma_f32_32x32x8_f16 a[0:15], v[36:37], v[16:17], 0
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v34, v73, v34
	v_mul_f32_e32 v35, v73, v35
	v_mul_f32_e32 v38, v73, v38
	v_mul_f32_e32 v39, v73, v39
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v34, v82, v34, vcc
	v_cndmask_b32_e32 v35, v82, v35, vcc
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v42, v73, v42
	.loc	1 151 23 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[40:41], v[18:19], a[0:15]
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v43, v73, v43
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v38, v82, v38, vcc
	v_cndmask_b32_e32 v39, v82, v39, vcc
.Ltmp2:
	.file	2 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/.env/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max_f32_e32 v104, v34, v35
.Ltmp3:
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v46, v73, v46
	v_mul_f32_e32 v47, v73, v47
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v42, v82, v42, vcc
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[44:45], v[20:21], a[0:15]
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v43, v82, v43, vcc
.Ltmp4:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v38, v39
.Ltmp5:
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v50, v73, v50
	v_mul_f32_e32 v51, v73, v51
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v46, v82, v46, vcc
	v_cndmask_b32_e32 v47, v82, v47, vcc
.Ltmp6:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v42, v43
.Ltmp7:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[48:49], v[22:23], a[0:15]
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v54, v73, v54
	v_mul_f32_e32 v55, v73, v55
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v50, v82, v50, vcc
	v_cndmask_b32_e32 v51, v82, v51, vcc
.Ltmp8:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v46, v47
.Ltmp9:
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v58, v73, v58
	v_mul_f32_e32 v59, v73, v59
	.loc	1 151 23 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[52:53], v[24:25], a[0:15]
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v54, v82, v54, vcc
	v_cndmask_b32_e32 v55, v82, v55, vcc
.Ltmp10:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v50, v51
.Ltmp11:
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v62, v73, v62
	v_mul_f32_e32 v63, v73, v63
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v58, v82, v58, vcc
	v_cndmask_b32_e32 v59, v82, v59, vcc
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[56:57], v[26:27], a[0:15]
.Ltmp12:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v54, v55
.Ltmp13:
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v62, v82, v62, vcc
	v_cndmask_b32_e32 v63, v82, v63, vcc
.Ltmp14:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v58, v59
	v_max3_f32 v104, v104, v62, v63
.Ltmp15:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_mfma_f32_32x32x8_f16 a[0:15], v[60:61], v[28:29], a[0:15]
	v_mfma_f32_32x32x8_f16 a[0:15], v[106:107], v[30:31], a[0:15]
	s_nop 10
	v_accvgpr_read_b32 v36, a0
	v_accvgpr_read_b32 v37, a1
	v_accvgpr_read_b32 v40, a2
	v_accvgpr_read_b32 v41, a3
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v36, v73, v36
	v_mul_f32_e32 v37, v73, v37
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v44, a4
	v_accvgpr_read_b32 v45, a5
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v40, v73, v40
	v_mul_f32_e32 v41, v73, v41
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v36, v82, v36, vcc
	v_cndmask_b32_e32 v37, v82, v37, vcc
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v48, a6
	v_accvgpr_read_b32 v49, a7
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v44, v73, v44
	v_mul_f32_e32 v45, v73, v45
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v40, v82, v40, vcc
	v_cndmask_b32_e32 v41, v82, v41, vcc
.Ltmp16:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v36, v37
.Ltmp17:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v52, a8
	v_accvgpr_read_b32 v53, a9
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v48, v73, v48
	v_mul_f32_e32 v49, v73, v49
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v44, v82, v44, vcc
	v_cndmask_b32_e32 v45, v82, v45, vcc
.Ltmp18:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v40, v41
.Ltmp19:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v56, a10
	v_accvgpr_read_b32 v57, a11
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v52, v73, v52
	v_mul_f32_e32 v53, v73, v53
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v48, v82, v48, vcc
	v_cndmask_b32_e32 v49, v82, v49, vcc
.Ltmp20:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v44, v45
.Ltmp21:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v60, a12
	v_accvgpr_read_b32 v61, a13
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v56, v73, v56
	v_mul_f32_e32 v57, v73, v57
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v52, v82, v52, vcc
	v_cndmask_b32_e32 v53, v82, v53, vcc
.Ltmp22:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v48, v49
.Ltmp23:
	.loc	1 151 23                        ; generate_rocjitsu_triton_fixtures.py:151:23
	v_accvgpr_read_b32 v64, a14
	v_accvgpr_read_b32 v103, a15
	.loc	1 151 50 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v60, v73, v60
	v_mul_f32_e32 v61, v73, v61
	.loc	1 152 77 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v56, v82, v56, vcc
	v_cndmask_b32_e32 v57, v82, v57, vcc
.Ltmp24:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v52, v53
.Ltmp25:
	.loc	1 151 50                        ; generate_rocjitsu_triton_fixtures.py:151:50
	v_mul_f32_e32 v64, v73, v64
	v_mul_f32_e32 v103, v73, v103
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v60, v82, v60, vcc
	v_cndmask_b32_e32 v61, v82, v61, vcc
.Ltmp26:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v56, v57
.Ltmp27:
	.loc	1 152 77                        ; generate_rocjitsu_triton_fixtures.py:152:77
	v_cndmask_b32_e32 v64, v82, v64, vcc
	v_cndmask_b32_e32 v103, v82, v103, vcc
.Ltmp28:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ] ]
	v_max3_f32 v104, v104, v60, v61
	v_max3_f32 v104, v104, v64, v103
.Ltmp29:
	.loc	2 191 40                        ; standard.py:191:40 @[ generate_rocjitsu_triton_fixtures.py:153:39 ]
	ds_bpermute_b32 v105, v76, v104
.Ltmp30:
	.loc	1 153 32                        ; generate_rocjitsu_triton_fixtures.py:153:32
	s_waitcnt lgkmcnt(0)
	v_max3_f32 v104, v33, v104, v105
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v34, v34, v104
	v_sub_f32_e32 v35, v35, v104
	v_sub_f32_e32 v38, v38, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v39, v39, v104
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v38, v38
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v42, v42, v104
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v39, v39
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v43, v43, v104
	v_sub_f32_e32 v64, v64, v104
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v42, v42
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v46, v46, v104
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v43, v43
	v_exp_f32_e32 v106, v64
.Ltmp31:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v34, v35
.Ltmp32:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v47, v47, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v46, v46
.Ltmp33:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v38, v64
.Ltmp34:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v50, v50, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v47, v47
.Ltmp35:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v39, v64
.Ltmp36:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v51, v51, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v50, v50
.Ltmp37:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v42, v64
.Ltmp38:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v54, v54, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v51, v51
.Ltmp39:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v43, v64
.Ltmp40:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v55, v55, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v54, v54
.Ltmp41:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v46, v64
.Ltmp42:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v58, v58, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v55, v55
.Ltmp43:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v47, v64
.Ltmp44:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v59, v59, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v58, v58
.Ltmp45:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v50, v64
.Ltmp46:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v62, v62, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v59, v59
.Ltmp47:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v51, v64
.Ltmp48:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v63, v63, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v62, v62
.Ltmp49:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v54, v64
.Ltmp50:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v36, v36, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v63, v63
.Ltmp51:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v55, v64
.Ltmp52:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v37, v37, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v36, v36
.Ltmp53:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v58, v64
.Ltmp54:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v40, v40, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v37, v37
.Ltmp55:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v59, v64
.Ltmp56:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v41, v41, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v40, v40
.Ltmp57:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v62, v64
.Ltmp58:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v44, v44, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v41, v41
.Ltmp59:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v63, v64
.Ltmp60:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v45, v45, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v44, v44
.Ltmp61:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v36, v64
.Ltmp62:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v48, v48, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v45, v45
.Ltmp63:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v37, v64
.Ltmp64:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v49, v49, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v48, v48
.Ltmp65:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v40, v64
.Ltmp66:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v52, v52, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v49, v49
.Ltmp67:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v41, v64
.Ltmp68:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v53, v53, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v52, v52
.Ltmp69:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v44, v64
.Ltmp70:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v56, v56, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v53, v53
.Ltmp71:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v45, v64
.Ltmp72:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v57, v57, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v56, v56
.Ltmp73:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v48, v64
.Ltmp74:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v60, v60, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v57, v57
.Ltmp75:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v49, v64
.Ltmp76:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v61, v61, v104
	.loc	1 155 25 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v60, v60
.Ltmp77:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v52, v64
.Ltmp78:
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v61, v61
.Ltmp79:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v53, v64
.Ltmp80:
	.loc	1 155 30                        ; generate_rocjitsu_triton_fixtures.py:155:30
	v_sub_f32_e32 v103, v103, v104
.Ltmp81:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v56, v64
.Ltmp82:
	.loc	1 155 25                        ; generate_rocjitsu_triton_fixtures.py:155:25
	v_exp_f32_e32 v107, v103
.Ltmp83:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	v_add_f32_e32 v64, v57, v64
	v_add_f32_e32 v64, v60, v64
	v_add_f32_e32 v64, v61, v64
	v_add_f32_e32 v64, v106, v64
	v_add_f32_e32 v64, v107, v64
.Ltmp84:
	.loc	2 293 36                        ; standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ]
	ds_bpermute_b32 v103, v76, v64
.Ltmp85:
	.loc	1 154 35                        ; generate_rocjitsu_triton_fixtures.py:154:35
	v_sub_f32_e32 v33, v33, v104
	.loc	1 154 29 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:154:29
	v_exp_f32_e32 v33, v33
	.loc	1 161 26 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:161:26
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v37, v37
.Ltmp86:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ generate_rocjitsu_triton_fixtures.py:156:37 ] ]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v103, v64, v103
.Ltmp87:
	.loc	1 156 30                        ; generate_rocjitsu_triton_fixtures.py:156:30
	v_fmac_f32_e32 v103, v32, v33
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	v_add_u32_e32 v32, v74, v80
	buffer_load_ushort v108, v32, s[0:3], 0 offen
	buffer_load_ushort v109, v32, s[0:3], 0 offen offset:128
	buffer_load_ushort v110, v32, s[0:3], 0 offen offset:256
	buffer_load_ushort v111, v32, s[0:3], 0 offen offset:384
	buffer_load_ushort v112, v32, s[0:3], 0 offen offset:2048
	buffer_load_ushort v113, v32, s[0:3], 0 offen offset:2176
	buffer_load_ushort v114, v32, s[0:3], 0 offen offset:2304
	buffer_load_ushort v115, v32, s[0:3], 0 offen offset:2432
	v_add_u32_e32 v32, 0x1000, v32
	buffer_load_ushort v116, v32, s[0:3], 0 offen
	buffer_load_ushort v117, v32, s[0:3], 0 offen offset:128
	buffer_load_ushort v118, v32, s[0:3], 0 offen offset:256
	buffer_load_ushort v119, v32, s[0:3], 0 offen offset:384
	buffer_load_ushort v120, v32, s[0:3], 0 offen offset:2048
	buffer_load_ushort v121, v32, s[0:3], 0 offen offset:2176
	buffer_load_ushort v122, v32, s[0:3], 0 offen offset:2304
	buffer_load_ushort v123, v32, s[0:3], 0 offen offset:2432
                                        ; kill: killed $vgpr32
	.loc	1 160 20                        ; generate_rocjitsu_triton_fixtures.py:160:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v93, v33
	.loc	1 161 26                        ; generate_rocjitsu_triton_fixtures.py:161:26
	v_cvt_f16_f32_e32 v32, v34
	v_cvt_f16_f32_e32 v33, v35
	v_cvt_f16_f32_e32 v34, v38
	v_cvt_f16_f32_e32 v35, v39
	v_cvt_f16_f32_e32 v38, v42
	v_cvt_f16_f32_e32 v39, v43
	v_cvt_f16_f32_e32 v42, v46
	v_cvt_f16_f32_e32 v43, v47
	v_cvt_f16_f32_e32 v46, v50
	v_cvt_f16_f32_e32 v47, v51
	v_cvt_f16_f32_e32 v50, v54
	v_cvt_f16_f32_e32 v51, v55
	v_cvt_f16_f32_e32 v54, v58
	v_cvt_f16_f32_e32 v55, v59
	v_cvt_f16_f32_e32 v58, v62
	v_cvt_f16_f32_e32 v59, v63
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v41, v41
	v_cvt_f16_f32_e32 v44, v44
	v_cvt_f16_f32_e32 v45, v45
	v_cvt_f16_f32_e32 v48, v48
	v_cvt_f16_f32_e32 v49, v49
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v53, v53
	v_cvt_f16_f32_e32 v56, v56
	v_cvt_f16_f32_e32 v57, v57
	v_cvt_f16_f32_e32 v60, v60
	v_cvt_f16_f32_e32 v61, v61
	v_cvt_f16_f32_e32 v62, v106
	v_cvt_f16_f32_e32 v63, v107
	.loc	1 160 20                        ; generate_rocjitsu_triton_fixtures.py:160:20
	v_add_u32_e32 v105, v77, v68
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v64, v105
	.loc	1 161 26                        ; generate_rocjitsu_triton_fixtures.py:161:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v90, v32
	ds_write_b16 v90, v33 offset:128
	ds_write_b16 v90, v34 offset:256
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	v_add_u32_e32 v34, 0, v78
	.loc	1 161 26                        ; generate_rocjitsu_triton_fixtures.py:161:26
	ds_write_b16 v90, v35 offset:384
	ds_write_b16 v90, v38 offset:1024
	ds_write_b16 v90, v39 offset:1152
	ds_write_b16 v90, v42 offset:1280
	ds_write_b16 v90, v43 offset:1408
	ds_write_b16 v90, v46 offset:2048
	ds_write_b16 v90, v47 offset:2176
	ds_write_b16 v90, v50 offset:2304
	ds_write_b16 v90, v51 offset:2432
	ds_write_b16 v90, v54 offset:3072
	ds_write_b16 v90, v55 offset:3200
	ds_write_b16 v90, v58 offset:3328
	ds_write_b16 v90, v59 offset:3456
	ds_write_b16 v90, v36 offset:4096
	ds_write_b16 v90, v37 offset:4224
	ds_write_b16 v90, v40 offset:4352
	ds_write_b16 v90, v41 offset:4480
	ds_write_b16 v90, v44 offset:5120
	ds_write_b16 v90, v45 offset:5248
	ds_write_b16 v90, v48 offset:5376
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	s_waitcnt vmcnt(14)
	v_perm_b32 v32, v109, v108, s12
	.loc	1 161 26                        ; generate_rocjitsu_triton_fixtures.py:161:26
	ds_write_b16 v90, v49 offset:5504
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	s_waitcnt vmcnt(12)
	v_perm_b32 v33, v111, v110, s12
	.loc	1 161 26                        ; generate_rocjitsu_triton_fixtures.py:161:26
	ds_write_b16 v90, v52 offset:6144
	ds_write_b16 v90, v53 offset:6272
	ds_write_b16 v90, v56 offset:6400
	ds_write_b16 v90, v57 offset:6528
	ds_write_b16 v90, v60 offset:7168
	ds_write_b16 v90, v61 offset:7296
	ds_write_b16 v90, v62 offset:7424
	ds_write_b16 v90, v63 offset:7552
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v36, v91
	ds_read_u16 v37, v91 offset:128
	ds_read_u16 v38, v91 offset:256
	ds_read_u16 v39, v91 offset:384
	ds_read_u16 v42, v91 offset:1024
	ds_read_u16 v43, v91 offset:1152
	ds_read_u16 v46, v91 offset:1280
	ds_read_u16 v47, v91 offset:1408
	ds_read_u16 v50, v91 offset:2048
	ds_read_u16 v51, v91 offset:2176
	ds_read_u16 v54, v91 offset:2304
	ds_read_u16 v55, v91 offset:2432
	ds_read_u16 v106, v91 offset:3072
	ds_read_u16 v107, v91 offset:3200
	ds_read_u16 v124, v91 offset:3328
	ds_read_u16 v125, v91 offset:3456
	ds_read_u16 v126, v91 offset:4096
	ds_read_u16 v127, v91 offset:4224
	ds_read_u16 v128, v91 offset:4352
	ds_read_u16 v129, v91 offset:4480
	ds_read_u16 v130, v91 offset:5120
	ds_read_u16 v131, v91 offset:5248
	ds_read_u16 v132, v91 offset:5376
	ds_read_u16 v133, v91 offset:5504
	ds_read_u16 v134, v91 offset:6144
	ds_read_u16 v135, v91 offset:6272
	ds_read_u16 v136, v91 offset:6400
	ds_read_u16 v137, v91 offset:6528
	ds_read_u16 v138, v91 offset:7168
	ds_read_u16 v139, v91 offset:7296
	ds_read_u16 v140, v91 offset:7424
	ds_read_u16 v141, v91 offset:7552
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b64 v34, v[32:33]
	s_waitcnt vmcnt(8)
	v_perm_b32 v33, v115, v114, s12
	v_perm_b32 v32, v113, v112, s12
	ds_write_b64 v92, v[32:33]
	s_waitcnt vmcnt(4)
	v_perm_b32 v33, v119, v118, s12
	v_perm_b32 v32, v117, v116, s12
	ds_write_b64 v94, v[32:33]
	s_waitcnt vmcnt(0)
	v_perm_b32 v33, v123, v122, s12
	v_perm_b32 v32, v121, v120, s12
	ds_write_b64 v95, v[32:33]
	v_add_u32_e32 v32, 0, v79
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[60:61], v32
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_pk_mul_f32 v[0:1], v[0:1], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[64:65] op_sel_hi:[1,0]
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[56:57], v96
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_accvgpr_write_b32 a0, v0
	v_perm_b32 v63, v39, v38, s12
	v_perm_b32 v62, v37, v36, s12
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
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[52:53], v97
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_perm_b32 v59, v47, v46, s12
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[60:61], v[62:63], a[0:15]
	v_perm_b32 v58, v43, v42, s12
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[48:49], v98
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_perm_b32 v55, v55, v54, s12
	v_perm_b32 v54, v51, v50, s12
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[44:45], v99
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_perm_b32 v51, v125, v124, s12
	v_perm_b32 v50, v107, v106, s12
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[56:57], v[58:59], a[0:15]
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[40:41], v100
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_perm_b32 v47, v129, v128, s12
	v_perm_b32 v46, v127, v126, s12
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[34:35], v101
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	v_perm_b32 v43, v133, v132, s12
	v_perm_b32 v42, v131, v130, s12
	.loc	1 158 20                        ; generate_rocjitsu_triton_fixtures.py:158:20
	ds_read_b64 v[32:33], v102
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x8_f16 a[0:15], v[52:53], v[54:55], a[0:15]
	v_perm_b32 v39, v137, v136, s12
	v_perm_b32 v38, v135, v134, s12
	v_perm_b32 v37, v141, v140, s12
	v_perm_b32 v36, v139, v138, s12
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x8_f16 a[0:15], v[48:49], v[50:51], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x8_f16 a[0:15], v[44:45], v[46:47], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8_f16 a[0:15], v[40:41], v[42:43], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_f16 a[0:15], v[34:35], v[38:39], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_f16 a[0:15], v[32:33], v[36:37], a[0:15]
	; sched_barrier mask(0x00000000)
	.loc	1 145 48                        ; generate_rocjitsu_triton_fixtures.py:145:48
	s_add_i32 s10, s10, 64
	v_add_u32_e32 v80, 0x2000, v80
	v_add_u32_e32 v81, 0x2000, v81
	s_cmpk_lt_u32 s10, 0x3c0
	v_mov_b32_e32 v33, v104
	s_cbranch_scc1 .LBB0_33
; %bb.34:
	.loc	1 135 45                        ; generate_rocjitsu_triton_fixtures.py:135:45
	v_lshrrev_b32_e32 v16, 2, v68
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
	s_nop 3
	v_accvgpr_read_b32 v0, a0
	.loc	1 135 32                        ; generate_rocjitsu_triton_fixtures.py:135:32
	v_or3_b32 v16, v67, v16, s11
	s_movk_i32 s0, 0x400
	.loc	1 161 42                        ; generate_rocjitsu_triton_fixtures.py:161:42
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
	.loc	1 139 59                        ; generate_rocjitsu_triton_fixtures.py:139:59
	v_cmp_gt_i32_e32 vcc, s0, v16
	.loc	1 164 16                        ; generate_rocjitsu_triton_fixtures.py:164:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v93, v103
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 166 32                        ; generate_rocjitsu_triton_fixtures.py:166:32
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_36
; %bb.35:                               ; %.critedge
	.loc	1 164 16                        ; generate_rocjitsu_triton_fixtures.py:164:16
	ds_read_b32 v18, v105
	.loc	1 135 45                        ; generate_rocjitsu_triton_fixtures.py:135:45
	v_lshrrev_b32_e32 v17, 3, v66
	.loc	1 138 34                        ; generate_rocjitsu_triton_fixtures.py:138:34
	v_lshlrev_b32_e32 v16, 6, v16
	.loc	1 138 46 is_stmt 0              ; generate_rocjitsu_triton_fixtures.py:138:46
	v_or3_b32 v16, v17, v16, v65
	.loc	1 166 21 is_stmt 1              ; generate_rocjitsu_triton_fixtures.py:166:21
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 164 16                        ; generate_rocjitsu_triton_fixtures.py:164:16
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v19, s[0:1], v18, v18, v15
	v_rcp_f32_e32 v20, v19
	.loc	1 166 21                        ; generate_rocjitsu_triton_fixtures.py:166:21
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[8:9]
	.loc	1 164 16                        ; generate_rocjitsu_triton_fixtures.py:164:16
	v_fma_f32 v21, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v21, v20
	v_div_scale_f32 v21, vcc, v15, v18, v15
	v_mul_f32_e32 v22, v21, v20
	v_fma_f32 v23, -v19, v22, v21
	v_fmac_f32_e32 v22, v23, v20
	v_fma_f32 v19, -v19, v22, v21
	v_div_scale_f32 v21, s[0:1], v18, v18, v14
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v20, v22
	v_div_fixup_f32 v15, v19, v18, v15
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v14, v18, v14
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v13
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v14, v19, v18, v14
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v13, v18, v13
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v12
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v13, v19, v18, v13
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v12, v18, v12
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v11
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v12, v19, v18, v12
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v11, v18, v11
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v10
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v11, v19, v18, v11
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v10, v18, v10
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v9
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v10, v19, v18, v10
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v9, v18, v9
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v8
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v9, v19, v18, v9
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v8, v18, v8
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v7
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v8, v19, v18, v8
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v7, v18, v7
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v6
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v7, v19, v18, v7
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v6, v18, v6
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v5
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v6, v19, v18, v6
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v5, v18, v5
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v4
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v5, v19, v18, v5
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v4, v18, v4
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v3
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v4, v19, v18, v4
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v3, v18, v3
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v2
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v3, v19, v18, v3
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v2, v18, v2
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v0
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v2, v19, v18, v2
	v_fma_f32 v19, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v19, v22
	v_div_scale_f32 v19, vcc, v0, v18, v0
	v_mul_f32_e32 v20, v19, v22
	v_fma_f32 v23, -v21, v20, v19
	v_fmac_f32_e32 v20, v23, v22
	v_fma_f32 v19, -v21, v20, v19
	v_div_scale_f32 v21, s[0:1], v18, v18, v1
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v19, v19, v22, v20
	v_div_fixup_f32 v0, v19, v18, v0
	v_fma_f32 v19, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v19, v23
	v_div_scale_f32 v19, vcc, v1, v18, v1
	v_mul_f32_e32 v20, v19, v23
	v_fma_f32 v22, -v21, v20, v19
	v_fmac_f32_e32 v20, v22, v23
	v_fma_f32 v19, -v21, v20, v19
	v_div_fmas_f32 v19, v19, v23, v20
	v_div_fixup_f32 v1, v19, v18, v1
	.loc	1 166 32                        ; generate_rocjitsu_triton_fixtures.py:166:32
	global_store_dwordx4 v[16:17], v[0:3], off
	global_store_dwordx4 v[16:17], v[4:7], off offset:32
	global_store_dwordx4 v[16:17], v[8:11], off offset:64
	global_store_dwordx4 v[16:17], v[12:15], off offset:96
.LBB0_36:                               ; %.critedge28
	.loc	1 166 4 is_stmt 0               ; generate_rocjitsu_triton_fixtures.py:166:4
	s_endpgm
.Ltmp88:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flash_attention_fwd_async_buffer_kernel
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
		.amdhsa_next_free_vgpr 160
		.amdhsa_next_free_sgpr 17
		.amdhsa_accum_offset 144
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
	.size	flash_attention_fwd_async_buffer_kernel, .Lfunc_end0-flash_attention_fwd_async_buffer_kernel
	.cfi_endproc
                                        ; -- End function
	.set flash_attention_fwd_async_buffer_kernel.num_vgpr, 142
	.set flash_attention_fwd_async_buffer_kernel.num_agpr, 16
	.set flash_attention_fwd_async_buffer_kernel.numbered_sgpr, 17
	.set flash_attention_fwd_async_buffer_kernel.num_named_barrier, 0
	.set flash_attention_fwd_async_buffer_kernel.private_seg_size, 0
	.set flash_attention_fwd_async_buffer_kernel.uses_vcc, 1
	.set flash_attention_fwd_async_buffer_kernel.uses_flat_scratch, 0
	.set flash_attention_fwd_async_buffer_kernel.has_dyn_sized_stack, 0
	.set flash_attention_fwd_async_buffer_kernel.has_recursion, 0
	.set flash_attention_fwd_async_buffer_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6664
; TotalNumSgprs: 23
; NumVgprs: 142
; NumAgprs: 16
; TotalNumVgprs: 160
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 19
; NumSGPRsForWavesPerEU: 23
; NumVGPRsForWavesPerEU: 160
; AccumOffset: 144
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 35
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
	.byte	153                             ; DW_AT_call_line
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
	.byte	156                             ; DW_AT_call_line
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
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
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
	.quad	.Ltmp24-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp27-.Lfunc_begin0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
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
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
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
	.asciz	"flash_attention_fwd_async_buffer_kernel" ; string offset=49
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
    .name:           flash_attention_fwd_async_buffer_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     23
    .sgpr_spill_count: 0
    .symbol:         flash_attention_fwd_async_buffer_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     160
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
