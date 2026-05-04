// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Generated with Triton 3.6.0 for gfx950/CDNA4 from this source:
//
//   import triton
//   from triton import language as tl
//
//   @triton.jit
//   def triton_cdna4_matmul_kernel(a, b, c,
//                                  M: tl.constexpr,
//                                  N: tl.constexpr,
//                                  K: tl.constexpr,
//                                  BLOCK_M: tl.constexpr,
//                                  BLOCK_N: tl.constexpr,
//                                  BLOCK_K: tl.constexpr):
//       pid_m = tl.program_id(0)
//       pid_n = tl.program_id(1)
//       offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
//       offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
//       offs_k = tl.arange(0, BLOCK_K)
//       acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
//       for k0 in tl.range(0, K, BLOCK_K):
//           k_idxs = k0 + offs_k
//           a_tile = tl.load(a + offs_m[:, None] * K + k_idxs[None, :],
//                            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
//                            other=0.0)
//           b_tile = tl.load(b + k_idxs[:, None] * N + offs_n[None, :],
//                            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
//                            other=0.0)
//           acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
//       tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc,
//                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
//
//   triton.compile(ASTSource(...,
//                            signature={"a": '*fp16', "b": '*fp16', "c": '*fp32'},
//                            constexprs={"M": 31, "N": 31, "K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}),
//                  target=GPUTarget("hip", "gfx950", 64),
//                  options={"num_warps": 4})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	triton_cdna4_matmul_kernel      ; -- Begin function triton_cdna4_matmul_kernel
	.p2align	8
	.type	triton_cdna4_matmul_kernel,@function
triton_cdna4_matmul_kernel:             ; @triton_cdna4_matmul_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.65:
	.file	1 "tests/kernels" "triton_cdna4_matmul_source"
	.loc	1 6 0 prologue_end              ; triton_cdna4_matmul_source:6:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.66:
.LBB0_0:
.Ltmp1:
	.loc	1 15 21 is_stmt 1               ; triton_cdna4_matmul_source:15:21
	s_lshl_b32 s8, s12, 5
	.loc	1 15 44 is_stmt 0               ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v1, 6, v0
	.loc	1 15 31                         ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s8, v1
	.loc	1 22 62 is_stmt 1               ; triton_cdna4_matmul_source:22:62
	v_and_b32_e32 v2, 63, v0
	.loc	1 22 49 is_stmt 0               ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v1
	.loc	1 21 51 is_stmt 1               ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v9, 0
	v_lshlrev_b32_e32 v2, 1, v2
	v_mov_b32_e32 v10, 0
	.loc	1 21 25 is_stmt 0               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_lshlrev_b32_e32 v4, 6, v1
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v10, v[4:5], off
.LBB0_2:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 4, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v9, v[4:5], off
.LBB0_4:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 8, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v12, v[4:5], off
.LBB0_6:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 12, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v11, v[4:5], off
.LBB0_8:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 16, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v14, v[4:5], off
.LBB0_10:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 20, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v13, v[4:5], off
.LBB0_12:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v3, 24, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v3
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v3
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v16, v[4:5], off
.LBB0_14:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 31 is_stmt 1               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, 28, v1
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v1
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v1
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v15, v[2:3], off
.LBB0_16:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 25 38 is_stmt 1               ; triton_cdna4_matmul_source:25:38
	v_lshrrev_b32_e32 v8, 5, v0
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v1, 31, v0
	.loc	1 16 21                         ; triton_cdna4_matmul_source:16:21
	s_lshl_b32 s2, s13, 5
	.loc	1 24 47                         ; triton_cdna4_matmul_source:24:47
	v_mul_u32_u24_e32 v2, 31, v8
	.loc	1 16 31                         ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v4, s2, v1
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_lshlrev_b32_e32 v2, 1, v2
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[2:3]
	.loc	1 24 51 is_stmt 0               ; triton_cdna4_matmul_source:24:51
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 25 73 is_stmt 1               ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e32 vcc, 31, v4
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[6:7], v[4:5], 1, v[6:7]
	v_mov_b32_e32 v2, 0
	.loc	1 24 25 is_stmt 0               ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:
	global_load_ushort v2, v[6:7], off
.LBB0_18:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	global_load_ushort v3, v[6:7], off offset:496
.LBB0_20:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v17, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:
	global_load_ushort v17, v[6:7], off offset:992
.LBB0_22:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 25 38 is_stmt 1               ; triton_cdna4_matmul_source:25:38
	v_or_b32_e32 v18, 24, v8
	.loc	1 24 47                         ; triton_cdna4_matmul_source:24:47
	v_mul_u32_u24_e32 v18, 31, v18
	.loc	1 24 29 is_stmt 0               ; triton_cdna4_matmul_source:24:29
	v_lshlrev_b32_e32 v18, 1, v18
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[18:19], s[4:5], 0, v[18:19]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[18:19], v[4:5], 1, v[18:19]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v18, v[18:19], off
.LBB0_24:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_26
; %bb.25:
	global_load_ushort v20, v[6:7], off offset:1984
.LBB0_26:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_28
; %bb.27:
	global_load_ushort v19, v[6:7], off offset:2480
.LBB0_28:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_30
; %bb.29:
	global_load_ushort v21, v[6:7], off offset:2976
.LBB0_30:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 25 38 is_stmt 1               ; triton_cdna4_matmul_source:25:38
	v_or_b32_e32 v6, 56, v8
	.loc	1 24 47                         ; triton_cdna4_matmul_source:24:47
	v_mul_u32_u24_e32 v6, 31, v6
	.loc	1 24 29 is_stmt 0               ; triton_cdna4_matmul_source:24:29
	v_lshlrev_b32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[4:5], v[4:5], 1, v[6:7]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v22, v[4:5], off
.LBB0_32:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	v_lshlrev_b32_e32 v29, 1, v0
	v_and_b32_e32 v4, 0xfe, v29
	v_bfe_i32 v5, v0, 7, 1
	s_movk_i32 s0, 0x110
	v_bitop3_b32 v4, v5, v4, s0 bitop3:0x6c
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v23, 32, v0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	v_add_u32_e32 v30, 0, v4
	v_xad_u32 v31, v4, 32, 0
	v_xad_u32 v5, v4, 64, 0
	v_xor_b32_e32 v4, 0x60, v4
	s_waitcnt vmcnt(0)
	ds_write_b16 v30, v10
	ds_write_b16 v30, v14 offset:2048
	ds_write_b16 v31, v9 offset:512
	ds_write_b16 v31, v13 offset:2560
	ds_write_b16 v5, v12 offset:1024
	ds_write_b16 v5, v16 offset:3072
	v_add_u32_e32 v4, 0, v4
	v_lshlrev_b32_e32 v16, 3, v0
	s_movk_i32 s1, 0x70
	v_lshrrev_b32_e32 v5, 1, v23
	s_movk_i32 s0, 0x60
	ds_write_b16 v4, v11 offset:1536
	ds_write_b16 v4, v15 offset:3584
	v_lshlrev_b32_e32 v12, 7, v1
	v_and_b32_e32 v4, 0x70, v16
	v_bitop3_b32 v13, v16, v5, s1 bitop3:0x6c
	v_bitop3_b32 v14, v4, v12, v5 bitop3:0xde
	v_bitop3_b32 v12, v13, s0, v12 bitop3:0x36
	v_add_u32_e32 v4, 0, v14
	v_xad_u32 v8, v14, 32, 0
	v_xad_u32 v14, v14, 64, 0
	v_add_u32_e32 v24, 0, v12
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_bfe_i32 v28, v0, 5, 1
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[4:7], v4
	ds_read_b128 v[8:11], v8
	ds_read_b128 v[12:15], v14
	ds_read_b128 v[24:27], v24
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v30, v2
	ds_write_b16 v30, v17 offset:1024
	ds_write_b16 v30, v20 offset:2048
	ds_write_b16 v30, v21 offset:3072
	ds_write_b16 v31, v3 offset:512
	ds_write_b16 v31, v18 offset:1536
	ds_write_b16 v31, v19 offset:2560
	ds_write_b16 v31, v22 offset:3584
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 24, v16
	v_and_b32_e32 v16, 32, v29
	s_movk_i32 s0, 0x220
	v_and_b32_e32 v2, 0xc0, v2
	v_bitop3_b32 v16, v28, v16, s0 bitop3:0x6c
	v_or3_b32 v2, v2, v3, v16
	v_add_u32_e32 v3, 0, v2
	v_xad_u32 v2, v2, 16, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[18:19], v2 offset:256
	ds_read_b64_tr_b16 v[16:17], v3
	ds_read_b64_tr_b16 v[28:29], v3 offset:1024
	ds_read_b64_tr_b16 v[32:33], v3 offset:2048
	ds_read_b64_tr_b16 v[36:37], v3 offset:3072
	ds_read_b64_tr_b16 v[30:31], v2 offset:1280
	ds_read_b64_tr_b16 v[34:35], v2 offset:2304
	ds_read_b64_tr_b16 v[38:39], v2 offset:3328
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[16:19], v[4:7], 0
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v2, 3, v23
	.loc	1 15 31 is_stmt 0               ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s8, v1
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v18, s2, v2
	.loc	1 28 35                         ; triton_cdna4_matmul_source:28:35
	v_mul_lo_u32 v20, v1, 31
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[0:1], 31, v18
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 31, v1
	.loc	1 28 17                         ; triton_cdna4_matmul_source:28:17
	v_ashrrev_i32_e32 v21, 31, v20
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[28:31], v[8:11], a[0:15]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_and_b32_e32 v0, 0xc0, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[0:1]
	.loc	1 28 17                         ; triton_cdna4_matmul_source:28:17
	v_lshl_add_u64 v[20:21], v[20:21], 2, s[6:7]
	.loc	1 28 39 is_stmt 0               ; triton_cdna4_matmul_source:28:39
	v_ashrrev_i32_e32 v19, 31, v18
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	.loc	1 28 39                         ; triton_cdna4_matmul_source:28:39
	v_lshl_add_u64 v[20:21], v[18:19], 2, v[20:21]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	.loc	1 27 30 is_stmt 1               ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[12:15], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[36:39], v[24:27], a[0:15]
	s_nop 11
	v_accvgpr_read_b32 v17, a15
	v_accvgpr_read_b32 v16, a14
	v_accvgpr_read_b32 v15, a13
	v_accvgpr_read_b32 v14, a12
	v_accvgpr_read_b32 v13, a11
	v_accvgpr_read_b32 v12, a10
	v_accvgpr_read_b32 v11, a9
	v_accvgpr_read_b32 v10, a8
	v_accvgpr_read_b32 v9, a7
	v_accvgpr_read_b32 v8, a6
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v4, a2
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v2, a0
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_34
; %bb.33:
	global_store_dword v[20:21], v2, off
.LBB0_34:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 1, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_36
; %bb.35:
	global_store_dword v[20:21], v3, off offset:4
.LBB0_36:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 2, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_38
; %bb.37:
	global_store_dword v[20:21], v4, off offset:8
.LBB0_38:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 3, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_40
; %bb.39:
	global_store_dword v[20:21], v5, off offset:12
.LBB0_40:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 8, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_42
; %bb.41:
	global_store_dword v[20:21], v6, off offset:32
.LBB0_42:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 9, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_44
; %bb.43:
	global_store_dword v[20:21], v7, off offset:36
.LBB0_44:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 10, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_46
; %bb.45:
	global_store_dword v[20:21], v8, off offset:40
.LBB0_46:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 11, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_48
; %bb.47:
	global_store_dword v[20:21], v9, off offset:44
.LBB0_48:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 16, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_50
; %bb.49:
	global_store_dword v[20:21], v10, off offset:64
.LBB0_50:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 17, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_52
; %bb.51:
	global_store_dword v[20:21], v11, off offset:68
.LBB0_52:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 18, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_54
; %bb.53:
	global_store_dword v[20:21], v12, off offset:72
.LBB0_54:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 19, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_56
; %bb.55:
	global_store_dword v[20:21], v13, off offset:76
.LBB0_56:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 24, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_58
; %bb.57:
	global_store_dword v[20:21], v14, off offset:96
.LBB0_58:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 25, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_60
; %bb.59:
	global_store_dword v[20:21], v15, off offset:100
.LBB0_60:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 26, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_62
; %bb.61:
	global_store_dword v[20:21], v16, off offset:104
.LBB0_62:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 16 31 is_stmt 1               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v0, 27, v18
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[2:3], 31, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_64
; %bb.63:
	global_store_dword v[20:21], v17, off offset:108
.LBB0_64:
	.loc	1 28 4 is_stmt 0                ; triton_cdna4_matmul_source:28:4
	s_endpgm
.Ltmp2:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel triton_cdna4_matmul_kernel
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 56
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 40
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
	.set triton_cdna4_matmul_kernel.num_vgpr, 40
	.set triton_cdna4_matmul_kernel.num_agpr, 16
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 14
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2392
; TotalNumSgprs: 20
; NumVgprs: 40
; NumAgprs: 16
; TotalNumVgprs: 56
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 56
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
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
	.asciz	"triton_cdna4_matmul_source"    ; string offset=7
.Linfo_string2:
	.asciz	"tests/kernels"                 ; string offset=34
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
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           triton_cdna4_matmul_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     56
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
