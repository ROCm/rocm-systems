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
//                            constexprs={"M": 512, "N": 512, "K": 512, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}),
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
; %bb.101:
	.file	1 "tests/kernels" "triton_cdna4_matmul_source"
	.loc	1 6 0 prologue_end              ; triton_cdna4_matmul_source:6:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.102:
.LBB0_0:
.Ltmp1:
	.loc	1 15 21 is_stmt 1               ; triton_cdna4_matmul_source:15:21
	s_lshl_b32 s14, s12, 5
	.loc	1 15 44 is_stmt 0               ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v34, 6, v0
	.loc	1 15 31                         ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s14, v34
	.loc	1 21 47 is_stmt 1               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v16, 9, v1
	.loc	1 17 26                         ; triton_cdna4_matmul_source:17:26
	v_and_b32_e32 v40, 63, v0
	s_movk_i32 s0, 0x200
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e64 s[0:1], s0, v1
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[2:3], v[16:17], 1, s[2:3]
	.loc	1 21 51 is_stmt 0               ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v38, 0
	v_lshlrev_b32_e32 v18, 1, v40
	v_mov_b32_e32 v37, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[18:19]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v37, v[4:5], off
.LBB0_2:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0x1000
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[8:9]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[18:19]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v38, v[6:7], off
.LBB0_4:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0x2000
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[6:7], v[2:3], 0, s[8:9]
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v39, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, v18
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[8:9]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v39, v[8:9], off
.LBB0_6:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_or_b32_e32 v8, 0x1800, v16
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[20:21], v[8:9], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[10:11], v[20:21], 0, v[18:19]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_8
; %bb.7:
	global_load_ushort v19, v[10:11], off
.LBB0_8:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0x4000
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[8:9], v[2:3], 0, s[8:9]
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, v18
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[12:13]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v42, v[12:13], off
.LBB0_10:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0x5000
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[12:13], v[2:3], 0, s[8:9]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_12
; %bb.11:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, v18
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v41, v[14:15], off
.LBB0_12:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0x6000
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_lshl_add_u64 v[14:15], v[2:3], 0, s[8:9]
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v43, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_14
; %bb.13:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, v18
	v_lshl_add_u64 v[22:23], v[14:15], 0, v[22:23]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v43, v[22:23], off
.LBB0_14:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_or_b32_e32 v16, 0x3800, v16
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[22:23], v[16:17], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v32, v18
	v_lshl_add_u64 v[16:17], v[22:23], 0, v[32:33]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[8:9], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.15:
	global_load_ushort v33, v[16:17], off
.LBB0_16:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[8:9]
	.loc	1 15 44 is_stmt 1               ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v1, 31, v0
	.loc	1 16 21                         ; triton_cdna4_matmul_source:16:21
	s_lshl_b32 s13, s13, 5
	.loc	1 16 31 is_stmt 0               ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v28, s13, v1
	.loc	1 17 26 is_stmt 1               ; triton_cdna4_matmul_source:17:26
	v_lshrrev_b32_e32 v35, 5, v0
	s_movk_i32 s8, 0x200
	.loc	1 25 73                         ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e64 s[8:9], s8, v28
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_ashrrev_i32_e32 v29, 31, v28
	v_mov_b32_e32 v32, 0
	v_lshlrev_b32_e32 v24, 10, v35
	v_mov_b32_e32 v25, 0
	.loc	1 24 25 is_stmt 0               ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_mov_b32_e32 v25, 0
	v_lshl_add_u64 v[26:27], s[4:5], 0, v[24:25]
	v_lshl_add_u64 v[26:27], v[28:29], 1, v[26:27]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v25, v[26:27], off
.LBB0_18:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v26, 0x2000
	v_lshl_or_b32 v26, v35, 10, v26
	v_mov_b32_e32 v27, 0
	v_lshl_add_u64 v[26:27], s[4:5], 0, v[26:27]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[26:27], v[28:29], 1, v[26:27]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v32, v[26:27], off
.LBB0_20:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_22
; %bb.21:
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v26, 0x4000
	v_lshl_or_b32 v26, v35, 10, v26
	v_mov_b32_e32 v27, 0
	v_lshl_add_u64 v[26:27], s[4:5], 0, v[26:27]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[26:27], v[28:29], 1, v[26:27]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v45, v[26:27], off
.LBB0_22:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 17 26 is_stmt 1               ; triton_cdna4_matmul_source:17:26
	v_or_b32_e32 v26, 24, v35
	v_lshlrev_b32_e32 v26, 10, v26
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 24 29 is_stmt 0               ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v27, 0
	v_lshl_add_u64 v[30:31], s[4:5], 0, v[26:27]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[30:31], v[28:29], 1, v[30:31]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v44, v[30:31], off
.LBB0_24:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_26
; %bb.25:
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v27, 0x8000
	v_lshl_or_b32 v30, v35, 10, v27
	v_mov_b32_e32 v31, 0
	v_lshl_add_u64 v[30:31], s[4:5], 0, v[30:31]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[30:31], v[28:29], 1, v[30:31]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v47, v[30:31], off
.LBB0_26:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_28
; %bb.27:
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v27, 0xa000
	v_lshl_or_b32 v30, v35, 10, v27
	v_mov_b32_e32 v31, 0
	v_lshl_add_u64 v[30:31], s[4:5], 0, v[30:31]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[30:31], v[28:29], 1, v[30:31]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v46, v[30:31], off
.LBB0_28:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v48, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_30
; %bb.29:
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v27, 0xc000
	v_lshl_or_b32 v48, v35, 10, v27
	v_mov_b32_e32 v49, 0
	v_lshl_add_u64 v[48:49], s[4:5], 0, v[48:49]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[48:49], v[28:29], 1, v[48:49]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v48, v[48:49], off
.LBB0_30:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 17 26 is_stmt 1               ; triton_cdna4_matmul_source:17:26
	v_or_b32_e32 v27, 56, v35
	v_lshlrev_b32_e32 v30, 10, v27
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 24 29 is_stmt 0               ; triton_cdna4_matmul_source:24:29
	v_mov_b32_e32 v31, 0
	v_lshl_add_u64 v[50:51], s[4:5], 0, v[30:31]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[50:51], v[28:29], 1, v[50:51]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v31, v[50:51], off
.LBB0_32:                               ; %.peel.begin
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	v_lshlrev_b32_e32 v27, 1, v0
	v_and_b32_e32 v36, 0xfe, v27
	v_bfe_i32 v49, v0, 7, 1
	s_movk_i32 s10, 0x110
	v_bitop3_b32 v49, v49, v36, s10 bitop3:0x6c
	v_add_u32_e32 v36, 0, v49
	s_waitcnt vmcnt(0)
	ds_write_b16 v36, v37
	ds_write_b16 v36, v42 offset:2048
	v_xor_b32_e32 v37, 32, v49
	v_add_u32_e32 v37, 0, v37
	ds_write_b16 v37, v38 offset:512
	ds_write_b16 v37, v41 offset:2560
	v_xor_b32_e32 v38, 64, v49
	v_add_u32_e32 v38, 0, v38
	ds_write_b16 v38, v39 offset:1024
	ds_write_b16 v38, v43 offset:3072
	v_xor_b32_e32 v39, 0x60, v49
	v_add_u32_e32 v39, 0, v39
	ds_write_b16 v39, v19 offset:1536
	ds_write_b16 v39, v33 offset:3584
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	ds_write_b16 v36, v25 offset:4096
	ds_write_b16 v36, v45 offset:5120
	ds_write_b16 v36, v47 offset:6144
	ds_write_b16 v36, v48 offset:7168
	ds_write_b16 v37, v32 offset:4608
	ds_write_b16 v37, v44 offset:5632
	ds_write_b16 v37, v46 offset:6656
	ds_write_b16 v37, v31 offset:7680
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v47, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_34
; %bb.33:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[18:19]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v47, v[2:3], off offset:128
.LBB0_34:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 20 22 is_stmt 1               ; triton_cdna4_matmul_source:20:22
	v_or_b32_e32 v2, 64, v40
	v_lshlrev_b32_e32 v2, 1, v2
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_36
; %bb.35:
	.loc	1 21 51 is_stmt 0               ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v31, v[4:5], off
.LBB0_36:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_38
; %bb.37:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v49, v[4:5], off
.LBB0_38:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_40
; %bb.39:
	global_load_ushort v48, v[10:11], off offset:128
.LBB0_40:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_42
; %bb.41:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v51, v[4:5], off
.LBB0_42:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_44
; %bb.43:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v50, v[4:5], off
.LBB0_44:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_46
; %bb.45:
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[2:3], v[14:15], 0, v[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v53, v[2:3], off
.LBB0_46:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_48
; %bb.47:
	global_load_ushort v52, v[16:17], off offset:128
.LBB0_48:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 15 44 is_stmt 1               ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v40, 32, v0
	v_lshlrev_b32_e32 v19, 3, v0
	s_movk_i32 s10, 0x70
	v_lshrrev_b32_e32 v4, 1, v40
	v_lshlrev_b32_e32 v2, 7, v1
	v_and_b32_e32 v3, 0x70, v19
	v_bitop3_b32 v5, v19, v4, s10 bitop3:0x6c
	v_bitop3_b32 v3, v3, v2, v4 bitop3:0xde
	v_bitop3_b32 v6, v5, 64, v2 bitop3:0x36
	s_movk_i32 s10, 0x60
	v_bitop3_b32 v4, v5, 32, v2 bitop3:0x36
	v_bitop3_b32 v2, v5, s10, v2 bitop3:0x36
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	v_add_u32_e32 v43, 0, v3
	v_add_u32_e32 v41, 0, v6
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v44, 0, v4
	ds_read_b128 v[14:17], v43
	ds_read_b128 v[10:13], v44
	v_add_u32_e32 v42, 0, v2
	ds_read_b128 v[6:9], v41
	ds_read_b128 v[2:5], v42
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[4:5]
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_mov_b32_e32 v25, 0
	v_lshl_add_u64 v[32:33], v[28:29], 0, v[24:25]
	v_mov_b32_e32 v54, 0
	.loc	1 24 25 is_stmt 0               ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_add_co_u32_e32 v54, vcc, 0x10000, v32
	s_nop 1
	v_addc_co_u32_e32 v55, vcc, 0, v33, vcc
	global_load_ushort v54, v[54:55], off
.LBB0_50:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_add_co_u32_e32 v24, vcc, 0x12000, v32
	s_nop 1
	v_addc_co_u32_e32 v25, vcc, 0, v33, vcc
	global_load_ushort v25, v[24:25], off
.LBB0_52:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_add_co_u32_e32 v56, vcc, 0x14000, v32
	s_nop 1
	v_addc_co_u32_e32 v57, vcc, 0, v33, vcc
	global_load_ushort v56, v[56:57], off
.LBB0_54:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 47                         ; triton_cdna4_matmul_source:24:47
	v_lshlrev_b32_e32 v24, 9, v35
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_56
; %bb.55:
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_mov_b32_e32 v35, 0x16000
	v_lshl_or_b32 v58, v24, 1, v35
	v_mov_b32_e32 v59, 0
	v_lshl_add_u64 v[58:59], v[28:29], 0, v[58:59]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v55, v[58:59], off
.LBB0_56:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v57, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_add_co_u32_e32 v58, vcc, 0x18000, v32
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v33, vcc
	global_load_ushort v57, v[58:59], off
.LBB0_58:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_add_co_u32_e32 v58, vcc, 0x1a000, v32
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v33, vcc
	global_load_ushort v35, v[58:59], off
.LBB0_60:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_add_co_u32_e32 v32, vcc, 0x1c000, v32
	s_nop 1
	v_addc_co_u32_e32 v33, vcc, 0, v33, vcc
	global_load_ushort v59, v[32:33], off
.LBB0_62:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_64
; %bb.63:
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_mov_b32_e32 v32, 0x1e000
	v_lshl_or_b32 v32, v24, 1, v32
	v_mov_b32_e32 v33, 0
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[32:33]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v58, v[28:29], off
.LBB0_64:                               ; %.peel.next
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v28, 0x220
	.loc	1 15 44 is_stmt 1               ; triton_cdna4_matmul_source:15:44
	v_cmp_eq_u32_e32 vcc, 0, v40
	v_lshlrev_b32_e32 v24, 4, v0
	v_and_b32_e32 v19, 24, v19
	v_cndmask_b32_e64 v28, v28, 0, vcc
	s_movk_i32 s10, 0xc0
	v_and_or_b32 v19, v24, s10, v19
	v_bitop3_b32 v24, v28, v27, 32 bitop3:0x78
	v_or_b32_e32 v27, v19, v24
	v_bitop3_b32 v19, v19, 16, v24 bitop3:0x36
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	v_add_u32_e32 v46, 0, v19
	v_add_u32_e32 v45, 0, v27
	ds_read_b64_tr_b16 v[62:63], v46 offset:4352
	ds_read_b64_tr_b16 v[60:61], v45 offset:4096
	ds_read_b64_tr_b16 v[64:65], v45 offset:5120
	ds_read_b64_tr_b16 v[68:69], v45 offset:6144
	ds_read_b64_tr_b16 v[72:73], v45 offset:7168
	ds_read_b64_tr_b16 v[66:67], v46 offset:5376
	ds_read_b64_tr_b16 v[70:71], v46 offset:6400
	ds_read_b64_tr_b16 v[74:75], v46 offset:7424
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[60:63], v[14:17], 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v36, v47
	ds_write_b16 v36, v51 offset:2048
	ds_write_b16 v37, v31 offset:512
	ds_write_b16 v37, v50 offset:2560
	ds_write_b16 v38, v49 offset:1024
	ds_write_b16 v38, v53 offset:3072
	ds_write_b16 v39, v48 offset:1536
	ds_write_b16 v39, v52 offset:3584
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	ds_write_b16 v36, v54 offset:4096
	ds_write_b16 v36, v56 offset:5120
	ds_write_b16 v36, v57 offset:6144
	ds_write_b16 v36, v59 offset:7168
	ds_write_b16 v37, v25 offset:4608
	ds_write_b16 v37, v55 offset:5632
	ds_write_b16 v37, v35 offset:6656
	ds_write_b16 v37, v58 offset:7680
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v31, v19
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[64:67], v[10:13], a[0:15]
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	v_mov_b32_e32 v27, v19
	s_mov_b32 s15, 0
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[68:71], v[6:9], a[0:15]
	v_mfma_f32_32x32x16_f16 a[0:15], v[72:75], v[2:5], a[0:15]
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	v_lshlrev_b32_e32 v2, 9, v34
	v_lshl_or_b32 v2, s12, 14, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[24:25], v[2:3], 1, s[2:3]
	v_add_u32_e32 v2, s13, v1
	s_mov_b64 s[2:3], 0x100
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[2:3]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[2:3]
	v_lshlrev_b64 v[28:29], 1, v[2:3]
	v_lshl_add_u64 v[2:3], v[30:31], 0, s[4:5]
	s_mov_b64 s[2:3], 0x20000
	v_lshl_add_u64 v[30:31], v[2:3], 0, s[2:3]
	v_lshlrev_b32_e32 v2, 5, v0
	v_and_b32_e32 v2, 0x1c00, v2
	v_mov_b32_e32 v3, v19
	v_lshl_add_u64 v[32:33], s[4:5], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[26:27], 0, s[4:5]
	v_lshl_add_u64 v[26:27], v[2:3], 0, s[2:3]
	s_mov_b64 s[2:3], 0x80
	s_mov_b64 s[4:5], 0x10000
	s_branch .LBB0_66
.LBB0_65:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 29 is_stmt 0                ; triton_cdna4_matmul_source:0:29
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25 is_stmt 1               ; triton_cdna4_matmul_source:24:25
	ds_read_b64_tr_b16 v[64:65], v46 offset:4352
	ds_read_b64_tr_b16 v[62:63], v45 offset:4096
	ds_read_b64_tr_b16 v[66:67], v45 offset:5120
	ds_read_b64_tr_b16 v[70:71], v45 offset:6144
	ds_read_b64_tr_b16 v[74:75], v45 offset:7168
	ds_read_b64_tr_b16 v[68:69], v46 offset:5376
	ds_read_b64_tr_b16 v[72:73], v46 offset:6400
	ds_read_b64_tr_b16 v[76:77], v46 offset:7424
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[62:65], v[14:17], a[0:15]
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	s_add_i32 s15, s15, 64
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[2:3]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[2:3]
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[2:3]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[4:5]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[4:5]
	s_cmpk_lt_u32 s15, 0x180
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[66:69], v[10:13], a[0:15]
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[4:5]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v36, v47
	ds_write_b16 v36, v51 offset:2048
	ds_write_b16 v37, v49 offset:512
	ds_write_b16 v37, v53 offset:2560
	ds_write_b16 v38, v48 offset:1024
	ds_write_b16 v38, v52 offset:3072
	ds_write_b16 v39, v50 offset:1536
	ds_write_b16 v39, v54 offset:3584
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	ds_write_b16 v36, v55 offset:4096
	ds_write_b16 v36, v57 offset:5120
	ds_write_b16 v36, v59 offset:6144
	ds_write_b16 v36, v61 offset:7168
	ds_write_b16 v37, v56 offset:4608
	ds_write_b16 v37, v58 offset:5632
	ds_write_b16 v37, v60 offset:6656
	ds_write_b16 v37, v34 offset:7680
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[70:73], v[6:9], a[0:15]
	v_mfma_f32_32x32x16_f16 a[0:15], v[74:77], v[2:5], a[0:15]
	.loc	1 19 29                         ; triton_cdna4_matmul_source:19:29
	s_cbranch_scc0 .LBB0_98
.LBB0_66:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[2:3], v[24:25], 0, v[18:19]
	v_mov_b32_e32 v47, 0
	.loc	1 21 25 is_stmt 0               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_66 Depth=1
	global_load_ushort v47, v[2:3], off offset:256
.LBB0_68:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v4, vcc, 0x1000, v2
	s_nop 1
	v_addc_co_u32_e32 v5, vcc, 0, v3, vcc
	global_load_ushort v49, v[4:5], off offset:256
.LBB0_70:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v4, vcc, 0x2000, v2
	s_nop 1
	v_addc_co_u32_e32 v5, vcc, 0, v3, vcc
	global_load_ushort v48, v[4:5], off offset:256
.LBB0_72:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[18:19]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v50, v[4:5], off
.LBB0_74:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v4, vcc, 0x4000, v2
	s_nop 1
	v_addc_co_u32_e32 v5, vcc, 0, v3, vcc
	global_load_ushort v51, v[4:5], off offset:256
.LBB0_76:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v4, vcc, 0x5000, v2
	s_nop 1
	v_addc_co_u32_e32 v5, vcc, 0, v3, vcc
	global_load_ushort v53, v[4:5], off offset:256
.LBB0_78:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v2, vcc, 0x6000, v2
	s_nop 1
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	global_load_ushort v52, v[2:3], off offset:256
.LBB0_80:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[10:11], s[0:1]
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[2:3], v[22:23], 0, v[18:19]
	global_load_ushort v54, v[2:3], off
.LBB0_82:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[14:17], v43
	ds_read_b128 v[10:13], v44
	ds_read_b128 v[6:9], v41
	ds_read_b128 v[2:5], v42
	.loc	1 24 51 is_stmt 1               ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[28:29]
	.loc	1 24 25 is_stmt 0               ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v56, vcc, 0x20000, v34
	s_nop 1
	v_addc_co_u32_e32 v57, vcc, 0, v35, vcc
	global_load_ushort v55, v[56:57], off
.LBB0_84:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v58, vcc, 0x22000, v34
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v35, vcc
	global_load_ushort v56, v[58:59], off
.LBB0_86:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v58, vcc, 0x24000, v34
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v35, vcc
	global_load_ushort v57, v[58:59], off
.LBB0_88:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[60:61], v[26:27], 0, v[28:29]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	global_load_ushort v58, v[60:61], off
.LBB0_90:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v60, vcc, 0x28000, v34
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v35, vcc
	global_load_ushort v59, v[60:61], off
.LBB0_92:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execnz .LBB0_95
; %bb.93:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execnz .LBB0_96
.LBB0_94:                               ;   in Loop: Header=BB0_66 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v34, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_65
	s_branch .LBB0_97
.LBB0_95:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v62, vcc, 0x2a000, v34
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v35, vcc
	global_load_ushort v60, v[62:63], off
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_94
.LBB0_96:                               ;   in Loop: Header=BB0_66 Depth=1
	v_add_co_u32_e32 v34, vcc, 0x2c000, v34
	s_nop 1
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	global_load_ushort v61, v[34:35], off
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v34, 0
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_65
.LBB0_97:                               ;   in Loop: Header=BB0_66 Depth=1
	v_lshl_add_u64 v[34:35], v[30:31], 0, v[28:29]
	global_load_ushort v34, v[34:35], off
	s_branch .LBB0_65
.LBB0_98:                               ; %.loopexit
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	ds_read_b64_tr_b16 v[10:11], v45 offset:4096
	ds_read_b64_tr_b16 v[12:13], v46 offset:4352
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_read_b128 v[2:5], v43
	ds_read_b128 v[6:9], v44
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	ds_read_b64_tr_b16 v[14:15], v45 offset:5120
	ds_read_b64_tr_b16 v[18:19], v45 offset:6144
	ds_read_b64_tr_b16 v[22:23], v45 offset:7168
	ds_read_b64_tr_b16 v[16:17], v46 offset:5376
	ds_read_b64_tr_b16 v[20:21], v46 offset:6400
	ds_read_b64_tr_b16 v[24:25], v46 offset:7424
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 a[16:31], v[10:13], v[2:5], a[0:15]
	.loc	1 15 31                         ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s14, v1
	s_movk_i32 s0, 0x200
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_and_b32_e32 v0, 0xc0, v0
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[16:31], v[14:17], v[6:9], a[16:31]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_read_b128 v[2:5], v41
	ds_read_b128 v[6:9], v42
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[16:31], v[18:21], v[2:5], a[16:31]
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v2, 3, v40
	.loc	1 16 31                         ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v2, s13, v2
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	v_max_i32_e32 v3, v1, v2
	v_cmp_gt_i32_e32 vcc, s0, v3
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	s_and_b64 s[0:1], s[0:1], vcc
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[16:31], v[22:25], v[6:9], a[16:31]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_100
; %bb.99:                               ; %.critedge
	.loc	1 28 35 is_stmt 0               ; triton_cdna4_matmul_source:28:35
	v_lshlrev_b32_e32 v0, 9, v1
	.loc	1 28 17                         ; triton_cdna4_matmul_source:28:17
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	.loc	1 28 39                         ; triton_cdna4_matmul_source:28:39
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_nop 4
	global_store_dwordx4 v[0:1], a[16:19], off
	global_store_dwordx4 v[0:1], a[20:23], off offset:32
	global_store_dwordx4 v[0:1], a[24:27], off offset:64
	global_store_dwordx4 v[0:1], a[28:31], off offset:96
.LBB0_100:                              ; %.critedge28
	.loc	1 28 4                          ; triton_cdna4_matmul_source:28:4
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
		.amdhsa_next_free_vgpr 112
		.amdhsa_next_free_sgpr 16
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
	.set triton_cdna4_matmul_kernel.num_agpr, 32
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 16
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3900
; TotalNumSgprs: 22
; NumVgprs: 78
; NumAgprs: 32
; TotalNumVgprs: 112
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 112
; AccumOffset: 80
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
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
	.asciz	"triton_cdna4_matmul_source"    ; string offset=7
.Linfo_string2:
	.asciz	"tests/kernels"                 ; string offset=34
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     32
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
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     112
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
