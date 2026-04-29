; Copyright (c) 2026 Advanced Micro Devices, Inc.
; SPDX-License-Identifier: MIT
;
; Human-readable CDNA4 assembly extracted from Triton 3.6.0 for gfx950.
; Do not check in the Python source as a standalone kernel; it is included here
; only as comments so this assembly fixture remains auditable.
;
; Triton source:
; import triton
; import triton.language as tl
;
; @triton.jit
; def triton_cdna4_real_matmul_kernel(
;     A,
;     B,
;     C,
;     M: tl.constexpr,
;     N: tl.constexpr,
;     K: tl.constexpr,
;     stride_am: tl.constexpr,
;     stride_ak: tl.constexpr,
;     stride_bn: tl.constexpr,
;     stride_bk: tl.constexpr,
;     stride_cm: tl.constexpr,
;     stride_cn: tl.constexpr,
;     BLOCK_M: tl.constexpr,
;     BLOCK_N: tl.constexpr,
;     BLOCK_K: tl.constexpr,
; ):
;     pid = tl.program_id(0)
;     num_pid_n = tl.cdiv(N, BLOCK_N)
;
;     pid_m = pid // num_pid_n
;     pid_n = pid - pid_m * num_pid_n
;
;     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
;     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
;     offs_k = tl.arange(0, BLOCK_K)
;
;     acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
;
;     for k0 in tl.range(0, K, BLOCK_K):
;         k_idxs = k0 + offs_k
;
;         a = tl.load(
;             A + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
;             mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
;             other=0.0,
;         )
;         b = tl.load(
;             B + offs_n[:, None] * stride_bn + k_idxs[None, :] * stride_bk,
;             mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
;             other=0.0,
;         )
;
;         acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
;
;     tl.store(
;         C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
;         acc,
;         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
;     )
;
; Compile options: target=hip:gfx950, warp_size=64, num_warps=4, num_stages=2,
; BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, M=N=K=1024.
; A is stored as M x K row-major, B is stored as N x K row-major, C is stored
; as M x N row-major. The kernel launches one CTA per 16x16 C tile.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	triton_cdna4_real_matmul_kernel ; -- Begin function triton_cdna4_real_matmul_kernel
	.p2align	8
	.type	triton_cdna4_real_matmul_kernel,@function
triton_cdna4_real_matmul_kernel:        ; @triton_cdna4_real_matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.23:
	.file	1 "/tmp" "rocjitsu_generate_real_matmul.py"
	.loc	1 11 0 prologue_end             ; rocjitsu_generate_real_matmul.py:11:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.24:
.LBB0_0:
.Ltmp1:
	.loc	1 31 19 is_stmt 1               ; rocjitsu_generate_real_matmul.py:31:19
	s_ashr_i32 s0, s12, 31
	s_lshr_b32 s0, s0, 26
	s_add_i32 s0, s12, s0
	s_ashr_i32 s9, s0, 6
	.loc	1 34 21                         ; rocjitsu_generate_real_matmul.py:34:21
	s_lshl_b32 s10, s9, 4
	.loc	1 34 44 is_stmt 0               ; rocjitsu_generate_real_matmul.py:34:44
	v_lshrrev_b32_e32 v14, 4, v0
	.loc	1 34 31                         ; rocjitsu_generate_real_matmul.py:34:31
	v_or_b32_e32 v1, s10, v14
	.loc	1 44 34 is_stmt 1               ; rocjitsu_generate_real_matmul.py:44:34
	v_lshlrev_b32_e32 v4, 10, v1
	.loc	1 34 44                         ; rocjitsu_generate_real_matmul.py:34:44
	v_and_b32_e32 v2, 15, v0
	s_movk_i32 s8, 0x400
	.loc	1 44 16                         ; rocjitsu_generate_real_matmul.py:44:16
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 45 36                         ; rocjitsu_generate_real_matmul.py:45:36
	v_cmp_gt_i32_e32 vcc, s8, v1
	.loc	1 44 16                         ; rocjitsu_generate_real_matmul.py:44:16
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v8, 1, v2
	v_mov_b32_e32 v6, 0
	.loc	1 44 12 is_stmt 0               ; rocjitsu_generate_real_matmul.py:44:12
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 12                          ; rocjitsu_generate_real_matmul.py:0:12
	v_mov_b32_e32 v9, 0
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[8:9]
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	global_load_ushort v6, v[6:7], off
.LBB0_2:
	.loc	1 0 12                          ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[0:1]
	.loc	1 35 21 is_stmt 1               ; rocjitsu_generate_real_matmul.py:35:21
	s_lshl_b32 s13, s9, 10
	s_lshl_b32 s12, s12, 4
	s_sub_i32 s11, s12, s13
	.loc	1 35 31 is_stmt 0               ; rocjitsu_generate_real_matmul.py:35:31
	v_or_b32_e32 v1, s11, v14
	.loc	1 49 34 is_stmt 1               ; rocjitsu_generate_real_matmul.py:49:34
	v_lshlrev_b32_e32 v10, 10, v1
	s_movk_i32 s0, 0x3ff
	.loc	1 49 16 is_stmt 0               ; rocjitsu_generate_real_matmul.py:49:16
	v_ashrrev_i32_e32 v11, 31, v10
	.loc	1 50 36 is_stmt 1               ; rocjitsu_generate_real_matmul.py:50:36
	v_cmp_lt_i32_e64 s[2:3], s0, v1
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	.loc	1 49 16                         ; rocjitsu_generate_real_matmul.py:49:16
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[4:5]
	.loc	1 49 12 is_stmt 0               ; rocjitsu_generate_real_matmul.py:49:12
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 0 12                          ; rocjitsu_generate_real_matmul.py:0:12
	v_mov_b32_e32 v9, 0
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[8:9]
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	global_load_ushort v3, v[12:13], off
.LBB0_4:                                ; %.peel.begin
	.loc	1 0 12                          ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[8:9]
	.loc	1 44 12 is_stmt 1               ; rocjitsu_generate_real_matmul.py:44:12
	v_lshlrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v1, 0xfe, v1
	v_bfe_i32 v7, v0, 7, 1
	s_movk_i32 s14, 0x108
	v_bitop3_b32 v1, v7, v1, s14 bitop3:0x6c
	v_add_u32_e32 v1, 0, v1
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v6
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	ds_write_b16 v1, v3 offset:512
	v_mov_b32_e32 v6, 0
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	v_mov_b32_e32 v9, 0
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[8:9]
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	global_load_ushort v6, v[6:7], off offset:32
.LBB0_6:
	.loc	1 0 12                          ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[8:9]
	.loc	1 34 44 is_stmt 1               ; rocjitsu_generate_real_matmul.py:34:44
	v_and_b32_e32 v3, 48, v0
	v_lshrrev_b32_e32 v9, 1, v3
	v_bfe_i32 v12, v0, 3, 1
	v_lshlrev_b32_e32 v7, 5, v0
	v_bitop3_b32 v9, v12, v9, s14 bitop3:0x6c
	s_movk_i32 s8, 0xe0
	v_and_or_b32 v7, v7, s8, v9
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	v_add_u32_e32 v16, 0, v7
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[12:13], v16
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_xor_b64 s[2:3], exec, s[8:9]
; %bb.7:                                ; %..peel.next_crit_edge
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr10_vgpr11
; %bb.8:                                ; %Flow
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	s_or_saveexec_b64 s[2:3], s[2:3]
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v9, 0
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_mov_b32_e32 v9, 0
	v_lshl_add_u64 v[8:9], v[10:11], 0, v[8:9]
	.loc	1 49 12 is_stmt 1               ; rocjitsu_generate_real_matmul.py:49:12
	global_load_ushort v9, v[8:9], off offset:32
.LBB0_10:                               ; %.peel.next
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[2:3]
	.loc	1 54 25 is_stmt 1               ; rocjitsu_generate_real_matmul.py:54:25
	ds_read_b64 v[10:11], v16 offset:512
	.loc	1 40 29                         ; rocjitsu_generate_real_matmul.py:40:29
	v_add_u32_e32 v8, s12, v14
	v_subrev_u32_e32 v8, s13, v8
	v_lshlrev_b32_e32 v8, 10, v8
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v6
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	ds_write_b16 v1, v9 offset:512
	.loc	1 40 29                         ; rocjitsu_generate_real_matmul.py:40:29
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshlrev_b32_e32 v6, 1, v2
	.loc	1 54 25                         ; rocjitsu_generate_real_matmul.py:54:25
	v_mfma_f32_16x16x16_f16 a[0:3], v[10:11], v[12:13], 0
	.loc	1 40 29                         ; rocjitsu_generate_real_matmul.py:40:29
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
	s_mov_b32 s4, -16
	s_branch .LBB0_12
.LBB0_11:                               ;   in Loop: Header=BB0_12 Depth=1
	.loc	1 0 29 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:29
	s_or_b64 exec, exec, s[2:3]
	.loc	1 54 25 is_stmt 1               ; rocjitsu_generate_real_matmul.py:54:25
	ds_read_b64 v[12:13], v16 offset:512
	.loc	1 40 29                         ; rocjitsu_generate_real_matmul.py:40:29
	s_add_i32 s4, s4, 32
	v_lshl_add_u64 v[8:9], v[8:9], 0, 64
	s_cmpk_lt_u32 s4, 0x3d0
	v_lshl_add_u64 v[4:5], v[4:5], 0, 64
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v15
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	ds_write_b16 v1, v14 offset:512
	.loc	1 54 25                         ; rocjitsu_generate_real_matmul.py:54:25
	v_mfma_f32_16x16x16_f16 a[0:3], v[12:13], v[10:11], a[0:3]
	.loc	1 40 29                         ; rocjitsu_generate_real_matmul.py:40:29
	s_cbranch_scc0 .LBB0_20
.LBB0_12:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 0 29 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:29
	v_mov_b32_e32 v17, 0
	v_lshl_add_u64 v[10:11], v[4:5], 0, v[6:7]
	v_mov_b32_e32 v18, 0
	.loc	1 44 12 is_stmt 1               ; rocjitsu_generate_real_matmul.py:44:12
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_ushort v18, v[10:11], off offset:64
.LBB0_14:                               ;   in Loop: Header=BB0_12 Depth=1
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[2:3]
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[14:15], v16
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[6:7]
	.loc	1 49 12 is_stmt 1               ; rocjitsu_generate_real_matmul.py:49:12
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_ushort v17, v[12:13], off offset:64
.LBB0_16:                               ;   in Loop: Header=BB0_12 Depth=1
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[2:3]
	.loc	1 54 25 is_stmt 1               ; rocjitsu_generate_real_matmul.py:54:25
	ds_read_b64 v[20:21], v16 offset:512
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v1, v18
	.loc	1 49 12                         ; rocjitsu_generate_real_matmul.py:49:12
	ds_write_b16 v1, v17 offset:512
	.loc	1 54 25                         ; rocjitsu_generate_real_matmul.py:54:25
	v_mfma_f32_16x16x16_f16 a[0:3], v[20:21], v[14:15], a[0:3]
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_ushort v15, v[10:11], off offset:96
.LBB0_18:                               ;   in Loop: Header=BB0_12 Depth=1
	.loc	1 0 12 is_stmt 0                ; rocjitsu_generate_real_matmul.py:0:12
	s_or_b64 exec, exec, s[2:3]
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64 v[10:11], v16
	.loc	1 49 12 is_stmt 1               ; rocjitsu_generate_real_matmul.py:49:12
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_11
; %bb.19:                               ;   in Loop: Header=BB0_12 Depth=1
	global_load_ushort v14, v[12:13], off offset:96
	s_branch .LBB0_11
.LBB0_20:                               ; %.loopexit
	.loc	1 44 12                         ; rocjitsu_generate_real_matmul.py:44:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2st64_b64 v[4:7], v16 offset1:1
	.loc	1 34 44                         ; rocjitsu_generate_real_matmul.py:34:44
	v_lshrrev_b32_e32 v3, 2, v3
	.loc	1 34 31 is_stmt 0               ; rocjitsu_generate_real_matmul.py:34:31
	v_or_b32_e32 v1, s10, v2
	.loc	1 35 31 is_stmt 1               ; rocjitsu_generate_real_matmul.py:35:31
	v_or_b32_e32 v2, s11, v3
	s_movk_i32 s0, 0x400
	.loc	1 59 38                         ; rocjitsu_generate_real_matmul.py:59:38
	v_max_i32_e32 v3, v1, v2
	.loc	1 58 8                          ; rocjitsu_generate_real_matmul.py:58:8
	v_and_b32_e32 v0, 0xc0, v0
	.loc	1 54 25                         ; rocjitsu_generate_real_matmul.py:54:25
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x16_f16 a[0:3], v[6:7], v[4:5], a[0:3]
	.loc	1 59 38                         ; rocjitsu_generate_real_matmul.py:59:38
	v_cmp_gt_i32_e32 vcc, s0, v3
	.loc	1 58 8                          ; rocjitsu_generate_real_matmul.py:58:8
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_22
; %bb.21:                               ; %.critedge
	.loc	1 57 30                         ; rocjitsu_generate_real_matmul.py:57:30
	v_lshlrev_b32_e32 v0, 10, v1
	.loc	1 57 12 is_stmt 0               ; rocjitsu_generate_real_matmul.py:57:12
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
	.loc	1 58 8 is_stmt 1                ; rocjitsu_generate_real_matmul.py:58:8
	global_store_dwordx4 v[0:1], a[0:3], off
.LBB0_22:                               ; %.critedge4
	.loc	1 56 4                          ; rocjitsu_generate_real_matmul.py:56:4
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel triton_cdna4_real_matmul_kernel
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
		.amdhsa_next_free_vgpr 28
		.amdhsa_next_free_sgpr 15
		.amdhsa_accum_offset 24
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
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
	.size	triton_cdna4_real_matmul_kernel, .Lfunc_end0-triton_cdna4_real_matmul_kernel
	.cfi_endproc
                                        ; -- End function
	.set triton_cdna4_real_matmul_kernel.num_vgpr, 22
	.set triton_cdna4_real_matmul_kernel.num_agpr, 4
	.set triton_cdna4_real_matmul_kernel.numbered_sgpr, 15
	.set triton_cdna4_real_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_real_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_real_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_real_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_real_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_real_matmul_kernel.has_recursion, 0
	.set triton_cdna4_real_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1108
; TotalNumSgprs: 21
; NumVgprs: 22
; NumAgprs: 4
; TotalNumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 21
; NumVGPRsForWavesPerEU: 28
; AccumOffset: 24
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 5
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 512, 4, 3212836864
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
	.asciz	"rocjitsu_generate_real_matmul.py" ; string offset=7
.Linfo_string2:
	.asciz	"/tmp"                          ; string offset=40
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     4
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
    .name:           triton_cdna4_real_matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_real_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     28
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
