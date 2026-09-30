// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Generated with Triton 3.6.0 for gfx950/CDNA4 from this source:
//
//   import triton
//   from triton import language as tl
//
//   @triton.jit
//   def triton_cdna4_matmul_kernel(a, b, c, M, N, K,
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
//                            signature={"a": '*fp16', "b": '*fp16', "c": '*fp32', "M": 'i32', "N": 'i32', "K": 'i32'},
//                            constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}),
//                  target=GPUTarget("hip", "gfx950", 64),
//                  options={"num_warps": 4})
//
// Recorded fixture metadata:
//   kernel: triton_cdna4_matmul_kernel
//   target: gfx950, wavefront_size: 64
//   launch sharedMemBytes used by cdna4_to_cdna3_dispatch_test: 8192
//   amdhsa_group_segment_fixed_size: 0
//   max_flat_workgroup_size: 256
//   kernarg_segment_size: 56
//   registers: vgpr_count=112, agpr_count=16, sgpr_count=44

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
; %bb.140:
	.file	1 "tests/kernels" "triton_cdna4_matmul_source"
	.loc	1 6 0 prologue_end              ; triton_cdna4_matmul_source:6:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.141:
.LBB0_0:
.Ltmp1:
	.loc	1 12 21 is_stmt 1               ; triton_cdna4_matmul_source:12:21
	s_lshl_b32 s11, s16, 5
	.loc	1 12 44 is_stmt 0               ; triton_cdna4_matmul_source:12:44
	v_lshrrev_b32_e32 v1, 6, v0
	.loc	1 12 31                         ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v1, s11, v1
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_and_b32_e32 v53, 63, v0
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_mul_lo_u32 v4, s10, v1
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 19 73                         ; triton_cdna4_matmul_source:19:73
	v_cmp_gt_i32_e32 vcc, s10, v53
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[18:19], v[4:5], 1, s[2:3]
	.loc	1 19 55                         ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[14:15], s[0:1], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_mov_b32_e32 v42, 0
	v_lshlrev_b32_e32 v2, 1, v53
	v_mov_b32_e32 v43, 0
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[6:7], v[18:19], 0, v[2:3]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v43, v[6:7], off
.LBB0_2:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[12:13]
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	s_lshl_b32 s16, s10, 2
	v_add_u32_e32 v6, s16, v4
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v3, 4, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[24:25], s8, v3
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[20:21], v[6:7], 1, s[2:3]
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_mov_b32_e32 v3, v42
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[14:15], s[24:25], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[2:3]
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_4
; %bb.3:
	global_load_ushort v42, v[4:5], off
.LBB0_4:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[12:13]
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_add_u32_e32 v10, s16, v6
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v3, 8, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v11, 31, v10
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[20:21], s8, v3
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[22:23], v[10:11], 1, s[2:3]
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_mov_b32_e32 v3, 0
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[14:15], s[20:21], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[6:7], v[22:23], 0, v[2:3]
	v_mov_b32_e32 v44, v3
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_6
; %bb.5:
	global_load_ushort v44, v[6:7], off
.LBB0_6:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[12:13]
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v8, 12, v1
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[26:27], s8, v8
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_mul_lo_u32 v8, s10, v8
	.loc	1 18 29 is_stmt 0               ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[24:25], v[8:9], 1, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[14:15], s[26:27], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[8:9], v[24:25], 0, v[2:3]
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_8
; %bb.7:
	global_load_ushort v3, v[8:9], off
.LBB0_8:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[12:13]
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_lshl_add_u32 v12, s10, 3, v10
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v11, 16, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v13, 31, v12
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[28:29], s8, v11
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[26:27], v[12:13], 1, s[2:3]
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, v2
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[14:15], s[28:29], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[10:11], v[26:27], 0, v[36:37]
	v_mov_b32_e32 v45, v37
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB0_10
; %bb.9:
	global_load_ushort v45, v[10:11], off
.LBB0_10:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[12:13]
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_add_u32_e32 v14, s16, v12
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v13, 20, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v15, 31, v14
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[12:13], s8, v13
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[28:29], v[14:15], 1, s[2:3]
	.loc	1 19 55                         ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[18:19], s[12:13], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[12:13], v[28:29], 0, v[36:37]
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[14:15], s[18:19]
	s_cbranch_execz .LBB0_12
; %bb.11:
	global_load_ushort v37, v[12:13], off
.LBB0_12:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[14:15]
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v15, 24, v1
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_add_u32_e32 v14, s16, v14
	.loc	1 19 49                         ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[14:15], s8, v15
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[30:31], v[14:15], 1, s[2:3]
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, v2
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[22:23], s[14:15], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[14:15], v[30:31], 0, v[38:39]
	v_mov_b32_e32 v36, v39
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[18:19], s[22:23]
	s_cbranch_execz .LBB0_14
; %bb.13:
	global_load_ushort v36, v[14:15], off
.LBB0_14:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[18:19]
	.loc	1 12 31 is_stmt 1               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v1, 28, v1
	.loc	1 18 47                         ; triton_cdna4_matmul_source:18:47
	v_mul_lo_u32 v16, s10, v1
	.loc	1 18 29 is_stmt 0               ; triton_cdna4_matmul_source:18:29
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 19 49 is_stmt 1               ; triton_cdna4_matmul_source:19:49
	v_cmp_gt_i32_e64 s[30:31], s8, v1
	.loc	1 18 29                         ; triton_cdna4_matmul_source:18:29
	v_lshl_add_u64 v[32:33], v[16:17], 1, s[2:3]
	.loc	1 19 55                         ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[18:19], s[30:31], vcc
	.loc	1 18 51                         ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[38:39]
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_16
; %bb.15:
	global_load_ushort v39, v[34:35], off
.LBB0_16:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 12 44 is_stmt 1               ; triton_cdna4_matmul_source:12:44
	v_and_b32_e32 v1, 31, v0
	.loc	1 13 21                         ; triton_cdna4_matmul_source:13:21
	s_lshl_b32 s33, s17, 5
	.loc	1 13 31 is_stmt 0               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v16, s33, v1
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_lshrrev_b32_e32 v47, 5, v0
	.loc	1 22 73                         ; triton_cdna4_matmul_source:22:73
	v_cmp_gt_i32_e64 s[18:19], s9, v16
	.loc	1 22 49 is_stmt 0               ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v47
	.loc	1 22 55                         ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 21 51 is_stmt 1               ; triton_cdna4_matmul_source:21:51
	v_ashrrev_i32_e32 v17, 31, v16
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v48, 0
	.loc	1 21 25 is_stmt 0               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_mul_lo_u32 v40, s9, v47
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[40:41], v[40:41], 1, s[4:5]
	v_lshl_add_u64 v[40:41], v[16:17], 1, v[40:41]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v48, v[40:41], off
.LBB0_18:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v40, 8, v47
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cmp_gt_i32 s10, 0
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v40
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cselect_b64 s[2:3], -1, 0
	.loc	1 22 55                         ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v40, s9, v40
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v38, v[50:51], off
.LBB0_20:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v41, 16, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v41
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 21 47 is_stmt 1               ; triton_cdna4_matmul_source:21:47
	s_lshl_b32 s34, s9, 3
	v_add_u32_e32 v40, s34, v40
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_22
; %bb.21:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v55, v[50:51], off
.LBB0_22:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v58, 24, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v58
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 16 29 is_stmt 1               ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v50, s9, v58
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v54, v[50:51], off
.LBB0_24:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v41, 32, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v41
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 21 47 is_stmt 1               ; triton_cdna4_matmul_source:21:47
	s_lshl_b32 s35, s9, 4
	v_add_u32_e32 v40, s35, v40
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_26
; %bb.25:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v57, v[50:51], off
.LBB0_26:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v41, 40, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v41
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 21 47 is_stmt 1               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v40, s34, v40
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_28
; %bb.27:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[50:51], v[40:41], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v56, v[50:51], off
.LBB0_28:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v41, 48, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v41
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 16 29 is_stmt 1               ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[22:23], s[2:3], s[16:17]
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v61, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[16:17], s[22:23]
	s_cbranch_execz .LBB0_30
; %bb.29:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v50, s34, v40
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[50:51], v[16:17], 1, v[50:51]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v61, v[50:51], off
.LBB0_30:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[16:17]
	.loc	1 14 26 is_stmt 1               ; triton_cdna4_matmul_source:14:26
	v_or_b32_e32 v59, 56, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v59
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[16:17], s[18:19], vcc
	.loc	1 16 29 is_stmt 1               ; triton_cdna4_matmul_source:16:29
	s_and_b64 s[16:17], s[2:3], s[16:17]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[16:17]
	s_cbranch_execz .LBB0_32
; %bb.31:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v40, s9, v59
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[40:41], v[40:41], 1, s[4:5]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[40:41], v[16:17], 1, v[40:41]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v41, v[40:41], off
.LBB0_32:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 18 25 is_stmt 1               ; triton_cdna4_matmul_source:18:25
	v_lshlrev_b32_e32 v60, 1, v0
	v_and_b32_e32 v40, 0xfe, v60
	v_bfe_i32 v49, v0, 7, 1
	s_movk_i32 s2, 0x110
	v_bitop3_b32 v40, v49, v40, s2 bitop3:0x6c
	v_add_u32_e32 v49, 0, v40
	s_waitcnt vmcnt(0)
	ds_write_b16 v49, v43
	ds_write_b16 v49, v45 offset:2048
	v_xor_b32_e32 v43, 32, v40
	v_add_u32_e32 v50, 0, v43
	ds_write_b16 v50, v42 offset:512
	ds_write_b16 v50, v37 offset:2560
	v_xor_b32_e32 v37, 64, v40
	v_add_u32_e32 v51, 0, v37
	ds_write_b16 v51, v44 offset:1024
	ds_write_b16 v51, v36 offset:3072
	v_xor_b32_e32 v36, 0x60, v40
	.loc	1 12 44                         ; triton_cdna4_matmul_source:12:44
	v_and_b32_e32 v46, 32, v0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_add_u32_e32 v52, 0, v36
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_sub_i32 s16, s10, 64
	.loc	1 12 44                         ; triton_cdna4_matmul_source:12:44
	v_cmp_eq_u32_e64 s[22:23], 0, v46
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	ds_write_b16 v52, v3 offset:1536
	ds_write_b16 v52, v39 offset:3584
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_write_b16 v49, v48 offset:4096
	ds_write_b16 v49, v55 offset:5120
	ds_write_b16 v49, v57 offset:6144
	ds_write_b16 v49, v61 offset:7168
	ds_write_b16 v50, v38 offset:4608
	ds_write_b16 v50, v54 offset:5632
	ds_write_b16 v50, v56 offset:6656
	ds_write_b16 v50, v41 offset:7680
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cmp_gt_i32 s16, 0
	v_lshlrev_b32_e32 v61, 4, v0
	v_lshlrev_b32_e32 v48, 3, v0
	s_cbranch_scc1 .LBB0_34
; %bb.33:                               ; %.._crit_edge_crit_edge
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	v_lshlrev_b32_e32 v3, 3, v0
	v_mov_b32_e32 v37, 0x220
	v_and_b32_e32 v36, 24, v3
	v_cndmask_b32_e64 v37, v37, 0, s[22:23]
	s_movk_i32 s3, 0xc0
	v_and_or_b32 v38, v61, s3, v36
	v_bitop3_b32 v37, v37, v60, 32 bitop3:0x78
	v_or_b32_e32 v36, v38, v37
	v_bitop3_b32 v37, v38, s2, v37 bitop3:0x36
	s_mov_b64 s[2:3], 0
	s_branch .LBB0_35
.LBB0_34:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_mov_b64 s[2:3], -1
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr36
                                        ; implicit-def: $vgpr37
.LBB0_35:                               ; %Flow126
	v_mov_b32_e32 v62, 0
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
	s_andn2_b64 vcc, exec, s[2:3]
	v_accvgpr_write_b32 a15, 0
	s_cbranch_vccnz .LBB0_105
; %bb.36:                               ; %.lr.ph
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v3, 64, v53
	.loc	1 19 73                         ; triton_cdna4_matmul_source:19:73
	v_cmp_gt_i32_e32 vcc, s10, v3
	.loc	1 19 55 is_stmt 0               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[0:1], vcc
	v_mov_b32_e32 v63, 0
	.loc	1 18 25 is_stmt 1               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_38
; %bb.37:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v63, v[2:3], off offset:128
.LBB0_38:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[24:25], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_40
; %bb.39:
	global_load_ushort v62, v[4:5], off offset:128
.LBB0_40:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[20:21], vcc
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_42
; %bb.41:
	global_load_ushort v65, v[6:7], off offset:128
.LBB0_42:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[26:27], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_44
; %bb.43:
	global_load_ushort v64, v[8:9], off offset:128
.LBB0_44:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[28:29], vcc
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_46
; %bb.45:
	global_load_ushort v67, v[10:11], off offset:128
.LBB0_46:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[12:13], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_48
; %bb.47:
	global_load_ushort v66, v[12:13], off offset:128
.LBB0_48:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[14:15], vcc
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_50
; %bb.49:
	global_load_ushort v69, v[14:15], off offset:128
.LBB0_50:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[36:37], s[30:31], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[36:37]
	s_cbranch_execz .LBB0_52
; %bb.51:
	global_load_ushort v68, v[34:35], off offset:128
.LBB0_52:
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	s_movk_i32 s2, 0x70
	v_lshrrev_b32_e32 v4, 1, v46
	v_lshlrev_b32_e32 v2, 7, v1
	v_and_b32_e32 v3, 0x70, v48
	v_bitop3_b32 v5, v48, v4, s2 bitop3:0x6c
	v_bitop3_b32 v3, v3, v2, v4 bitop3:0xde
	v_bitop3_b32 v6, v5, 64, v2 bitop3:0x36
	s_movk_i32 s2, 0x60
	v_bitop3_b32 v4, v5, 32, v2 bitop3:0x36
	v_bitop3_b32 v2, v5, s2, v2 bitop3:0x36
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x48, v47
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_add_u32_e32 v54, 0, v3
	v_add_u32_e32 v56, 0, v6
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	v_lshl_add_u64 v[34:35], v[16:17], 1, s[4:5]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_add_u32_e32 v55, 0, v4
	ds_read_b128 v[14:17], v54
	ds_read_b128 v[10:13], v55
	v_add_u32_e32 v57, 0, v2
	ds_read_b128 v[6:9], v56
	ds_read_b128 v[2:5], v57
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v44, v37, s9
	v_add_u32_e32 v42, s34, v44
	.loc	1 17 22                         ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v36, 64, v47
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v40, s35, v42
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v36
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v38, s34, v40
	.loc	1 22 55                         ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v36, s34, v38
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v70, 0
	.loc	1 21 25 is_stmt 0               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_54
; %bb.53:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_mul_i32 s4, s9, 0xffffffd0
	v_add_u32_e32 v70, s4, v36
	v_ashrrev_i32_e32 v71, 31, v70
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v70, v[70:71], off
.LBB0_54:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 22 49 is_stmt 1               ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_56
; %bb.55:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v45, 31, v44
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[44:45], v[44:45], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v45, v[44:45], off
.LBB0_56:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x50, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_58
; %bb.57:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v43, 31, v42
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[42:43], v[42:43], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v43, v[42:43], off
.LBB0_58:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x58, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_60
; %bb.59:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v72, v37, s9
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v73, 31, v72
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[72:73], v[72:73], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v44, v[72:73], off
.LBB0_60:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x60, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_62
; %bb.61:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v41, 31, v40
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[40:41], v[40:41], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v41, v[40:41], off
.LBB0_62:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x68, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_64
; %bb.63:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v39, 31, v38
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v42, v[38:39], off
.LBB0_64:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v37, 0x70, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v37
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_66
; %bb.65:
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v37, 31, v36
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v72, v[36:37], off
.LBB0_66:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_or_b32_e32 v36, 0x78, v47
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v36
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[4:5], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_68
; %bb.67:
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_mul_lo_u32 v36, v36, s9
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v37, 31, v36
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[36:37], v[36:37], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v71, v[36:37], off
.LBB0_68:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v37, 0x220
	v_and_b32_e32 v36, 24, v48
	v_cndmask_b32_e64 v38, v37, 0, s[22:23]
	s_movk_i32 s2, 0xc0
	v_and_or_b32 v37, v61, s2, v36
	v_bitop3_b32 v38, v38, v60, 32 bitop3:0x78
	v_bitop3_b32 v40, v37, 16, v38 bitop3:0x36
	v_or_b32_e32 v36, v37, v38
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	v_add_u32_e32 v40, 0, v40
	v_add_u32_e32 v39, 0, v36
	ds_read_b64_tr_b16 v[76:77], v40 offset:4352
	ds_read_b64_tr_b16 v[74:75], v39 offset:4096
	ds_read_b64_tr_b16 v[78:79], v39 offset:5120
	ds_read_b64_tr_b16 v[82:83], v39 offset:6144
	ds_read_b64_tr_b16 v[86:87], v39 offset:7168
	ds_read_b64_tr_b16 v[80:81], v40 offset:5376
	ds_read_b64_tr_b16 v[84:85], v40 offset:6400
	ds_read_b64_tr_b16 v[88:89], v40 offset:7424
	.loc	1 24 30 is_stmt 1               ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[74:77], v[14:17], 0
	s_movk_i32 s4, 0x110
	s_mov_b32 s5, 0
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cmpk_lt_u32 s16, 0x41
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v49, v63
	ds_write_b16 v49, v67 offset:2048
	ds_write_b16 v50, v62 offset:512
	ds_write_b16 v50, v66 offset:2560
	ds_write_b16 v51, v65 offset:1024
	ds_write_b16 v51, v69 offset:3072
	ds_write_b16 v52, v64 offset:1536
	ds_write_b16 v52, v68 offset:3584
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_write_b16 v49, v70 offset:4096
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[78:81], v[10:13], a[0:15]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_write_b16 v49, v43 offset:5120
	ds_write_b16 v49, v41 offset:6144
	ds_write_b16 v49, v72 offset:7168
	ds_write_b16 v50, v45 offset:4608
	ds_write_b16 v50, v44 offset:5632
	ds_write_b16 v50, v42 offset:6656
	ds_write_b16 v50, v71 offset:7680
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[82:85], v[6:9], a[0:15]
	v_mfma_f32_32x32x16_f16 a[0:15], v[86:89], v[2:5], a[0:15]
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cbranch_scc1 .LBB0_104
; %bb.69:                               ; %.peel.next
	v_or_b32_e32 v2, 0xb0, v47
	v_mul_lo_u32 v43, s9, v2
	v_or_b32_e32 v2, 0xa8, v47
	v_mul_lo_u32 v44, s9, v2
	v_or_b32_e32 v2, 0xa0, v47
	v_mul_lo_u32 v45, s9, v2
	v_or_b32_e32 v2, 0x90, v47
	v_mul_lo_u32 v60, s9, v2
	v_or_b32_e32 v2, 0x88, v47
	v_or_b32_e32 v41, 0x80, v59
	v_or_b32_e32 v58, 0x80, v58
	v_mul_lo_u32 v61, s9, v2
	v_or_b32_e32 v2, 0x80, v47
	v_mul_lo_u32 v42, s9, v41
	s_lshl_b32 s17, s9, 6
	v_mul_lo_u32 v59, s9, v58
	v_mul_lo_u32 v62, s9, v2
	v_or_b32_e32 v53, 0x80, v53
	s_mov_b32 s22, 0
.LBB0_70:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 17 22                         ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v2, s22, v53
	.loc	1 19 73                         ; triton_cdna4_matmul_source:19:73
	v_cmp_gt_i32_e32 vcc, s10, v2
	.loc	1 19 55 is_stmt 0               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[0:1], vcc
	.loc	1 18 51 is_stmt 1               ; triton_cdna4_matmul_source:18:51
	v_ashrrev_i32_e32 v3, 31, v2
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	.loc	1 18 25 is_stmt 0               ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[18:19]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v65, v[4:5], off
.LBB0_72:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[24:25], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[20:21]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v64, v[4:5], off
.LBB0_74:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[20:21], vcc
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[22:23]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v67, v[4:5], off
.LBB0_76:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[26:27], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[24:25]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v66, v[4:5], off
.LBB0_78:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[28:29], vcc
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[26:27]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v69, v[4:5], off
.LBB0_80:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[12:13], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[28:29]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v68, v[4:5], off
.LBB0_82:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[14:15], vcc
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[4:5], v[2:3], 1, v[30:31]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v71, v[4:5], off
.LBB0_84:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 55 is_stmt 1               ; triton_cdna4_matmul_source:19:55
	s_and_b64 s[34:35], s[30:31], vcc
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_86
; %bb.85:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 18 51 is_stmt 0               ; triton_cdna4_matmul_source:18:51
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[32:33]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	global_load_ushort v70, v[2:3], off
.LBB0_86:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[14:17], v54
	ds_read_b128 v[10:13], v55
	ds_read_b128 v[6:9], v56
	ds_read_b128 v[2:5], v57
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v63, s22, v47
	v_add_u32_e32 v72, 0x80, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v72
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25 is_stmt 0                ; triton_cdna4_matmul_source:0:25
	v_add_u32_e32 v74, s5, v62
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshl_add_u64 v[74:75], v[74:75], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v73, v[74:75], off
.LBB0_88:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v74, 0x88, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v74
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v74, s5, v61
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v75, 31, v74
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[74:75], v[74:75], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v72, v[74:75], off
.LBB0_90:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v74, 0x90, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v74
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_92
; %bb.91:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v76, s5, v60
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v77, 31, v76
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[76:77], v[76:77], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v75, v[76:77], off
.LBB0_92:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v76, s22, v58
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v76
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v76, s5, v59
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v77, 31, v76
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[76:77], v[76:77], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v74, v[76:77], off
.LBB0_94:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v76, 0xa0, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v76
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_96
; %bb.95:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v78, s5, v45
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v79, 31, v78
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[78:79], v[78:79], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v77, v[78:79], off
.LBB0_96:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v78, 0xa8, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v78
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v78, s5, v44
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v79, 31, v78
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[78:79], v[78:79], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v76, v[78:79], off
.LBB0_98:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 17 22 is_stmt 1               ; triton_cdna4_matmul_source:17:22
	v_add_u32_e32 v63, 0xb0, v63
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v63
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v78, 0
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 47 is_stmt 0               ; triton_cdna4_matmul_source:21:47
	v_add_u32_e32 v78, s5, v43
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v79, 31, v78
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[78:79], v[78:79], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v78, v[78:79], off
.LBB0_100:                              ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 19 73 is_stmt 1               ; triton_cdna4_matmul_source:19:73
	v_add_u32_e32 v79, s22, v41
	.loc	1 22 49                         ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, s10, v79
	.loc	1 22 55 is_stmt 0               ; triton_cdna4_matmul_source:22:55
	s_and_b64 s[34:35], s[18:19], vcc
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[2:3], s[34:35]
	s_cbranch_execz .LBB0_102
; %bb.101:                              ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 21 29 is_stmt 0               ; triton_cdna4_matmul_source:21:29
	v_add_u32_e32 v80, s5, v42
	v_ashrrev_i32_e32 v81, 31, v80
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[80:81], v[80:81], 1, v[34:35]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v63, v[80:81], off
.LBB0_102:                              ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[2:3]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_read_b64_tr_b16 v[82:83], v40 offset:4352
	ds_read_b64_tr_b16 v[80:81], v39 offset:4096
	ds_read_b64_tr_b16 v[84:85], v39 offset:5120
	ds_read_b64_tr_b16 v[88:89], v39 offset:6144
	ds_read_b64_tr_b16 v[92:93], v39 offset:7168
	ds_read_b64_tr_b16 v[86:87], v40 offset:5376
	ds_read_b64_tr_b16 v[90:91], v40 offset:6400
	ds_read_b64_tr_b16 v[94:95], v40 offset:7424
	.loc	1 24 30 is_stmt 1               ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[80:83], v[14:17], a[0:15]
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_add_i32 s2, s22, 64
	s_add_i32 s3, s22, 0x80
	s_add_i32 s5, s5, s17
	s_cmp_lt_i32 s3, s16
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b16 v49, v65
	ds_write_b16 v49, v69 offset:2048
	ds_write_b16 v50, v64 offset:512
	ds_write_b16 v50, v68 offset:2560
	ds_write_b16 v51, v67 offset:1024
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[84:87], v[10:13], a[0:15]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	ds_write_b16 v51, v71 offset:3072
	ds_write_b16 v52, v66 offset:1536
	ds_write_b16 v52, v70 offset:3584
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_write_b16 v49, v73 offset:4096
	ds_write_b16 v49, v75 offset:5120
	ds_write_b16 v49, v77 offset:6144
	ds_write_b16 v49, v78 offset:7168
	ds_write_b16 v50, v72 offset:4608
	ds_write_b16 v50, v74 offset:5632
	ds_write_b16 v50, v76 offset:6656
	ds_write_b16 v50, v63 offset:7680
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	v_mfma_f32_32x32x16_f16 a[0:15], v[88:91], v[6:9], a[0:15]
	v_mfma_f32_32x32x16_f16 a[0:15], v[92:95], v[2:5], a[0:15]
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_cbranch_scc0 .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_70 Depth=1
	.loc	1 0 29 is_stmt 0                ; triton_cdna4_matmul_source:0:29
	s_mov_b32 s22, s2
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_branch .LBB0_70
.LBB0_104:                              ; %Flow125
	.loc	1 0 29                          ; triton_cdna4_matmul_source:0:29
	v_bitop3_b32 v37, v37, s4, v38 bitop3:0x36
	v_mov_b32_e32 v3, v48
.LBB0_105:                              ; %._crit_edge
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	v_add_u32_e32 v2, 0, v36
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_read_b64_tr_b16 v[32:33], v2 offset:4096
	ds_read_b64_tr_b16 v[28:29], v2 offset:5120
	ds_read_b64_tr_b16 v[24:25], v2 offset:6144
	ds_read_b64_tr_b16 v[20:21], v2 offset:7168
	v_add_u32_e32 v2, 0, v37
	ds_read_b64_tr_b16 v[34:35], v2 offset:4096
	ds_read_b64_tr_b16 v[30:31], v2 offset:5120
	ds_read_b64_tr_b16 v[26:27], v2 offset:6144
	ds_read_b64_tr_b16 v[22:23], v2 offset:7168
	.loc	1 16 29                         ; triton_cdna4_matmul_source:16:29
	s_add_i32 s10, s10, 63
	s_cmp_lt_i32 s10, 64
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	s_cbranch_scc1 .LBB0_107
; %bb.106:
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_lshlrev_b32_e32 v2, 7, v1
	v_and_b32_e32 v36, 0x70, v3
	v_lshrrev_b32_e32 v40, 1, v46
	v_bitop3_b32 v41, v36, v2, v40 bitop3:0xde
	v_add_u32_e32 v36, 0, v41
	ds_read_b128 v[36:39], v36
	v_accvgpr_read_b32 v4, a0
	v_accvgpr_read_b32 v5, a1
	v_accvgpr_read_b32 v6, a2
	v_accvgpr_read_b32 v7, a3
	v_accvgpr_read_b32 v8, a4
	v_accvgpr_read_b32 v9, a5
	v_accvgpr_read_b32 v10, a6
	v_accvgpr_read_b32 v11, a7
	v_accvgpr_read_b32 v12, a8
	v_accvgpr_read_b32 v13, a9
	v_accvgpr_read_b32 v14, a10
	v_accvgpr_read_b32 v15, a11
	v_accvgpr_read_b32 v16, a12
	v_accvgpr_read_b32 v17, a13
	v_accvgpr_read_b32 v18, a14
	v_accvgpr_read_b32 v19, a15
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	v_accvgpr_write_b32 a0, v4
	v_accvgpr_write_b32 a1, v5
	v_accvgpr_write_b32 a2, v6
	v_accvgpr_write_b32 a3, v7
	v_accvgpr_write_b32 a4, v8
	v_accvgpr_write_b32 a5, v9
	v_accvgpr_write_b32 a6, v10
	v_accvgpr_write_b32 a7, v11
	v_accvgpr_write_b32 a8, v12
	v_accvgpr_write_b32 a9, v13
	v_accvgpr_write_b32 a10, v14
	v_accvgpr_write_b32 a11, v15
	v_accvgpr_write_b32 a12, v16
	v_accvgpr_write_b32 a13, v17
	v_accvgpr_write_b32 a14, v18
	v_accvgpr_write_b32 a15, v19
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_xad_u32 v4, v41, 32, 0
	ds_read_b128 v[4:7], v4
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[36:39], a[0:15]
	s_movk_i32 s0, 0x70
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_bitop3_b32 v3, v3, v40, s0 bitop3:0x6c
	s_movk_i32 s0, 0x60
	v_bitop3_b32 v2, v3, s0, v2 bitop3:0x36
	v_add_u32_e32 v2, 0, v2
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[28:31], v[4:7], a[0:15]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	v_xad_u32 v4, v41, 64, 0
	ds_read_b128 v[4:7], v4
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[24:27], v[4:7], a[0:15]
	.loc	1 18 25                         ; triton_cdna4_matmul_source:18:25
	ds_read_b128 v[2:5], v2
	.loc	1 24 30                         ; triton_cdna4_matmul_source:24:30
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[20:23], v[2:5], a[0:15]
.LBB0_107:                              ; %._crit_edge._crit_edge
	.loc	1 12 44                         ; triton_cdna4_matmul_source:12:44
	v_lshrrev_b32_e32 v2, 3, v46
	.loc	1 12 31 is_stmt 0               ; triton_cdna4_matmul_source:12:31
	v_or_b32_e32 v1, s11, v1
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v2, s33, v2
	.loc	1 25 35                         ; triton_cdna4_matmul_source:25:35
	v_mul_lo_u32 v4, s9, v1
	.loc	1 26 37                         ; triton_cdna4_matmul_source:26:37
	v_cmp_gt_i32_e32 vcc, s8, v1
	.loc	1 26 61 is_stmt 0               ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[0:1], s9, v2
	.loc	1 25 17 is_stmt 1               ; triton_cdna4_matmul_source:25:17
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	1 25 56 is_stmt 0               ; triton_cdna4_matmul_source:25:56
	v_and_b32_e32 v0, 0xc0, v0
	.loc	1 26 43 is_stmt 1               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[0:1]
	.loc	1 25 17                         ; triton_cdna4_matmul_source:25:17
	v_lshl_add_u64 v[4:5], v[4:5], 2, s[6:7]
	.loc	1 25 39 is_stmt 0               ; triton_cdna4_matmul_source:25:39
	v_ashrrev_i32_e32 v3, 31, v2
	.loc	1 25 56                         ; triton_cdna4_matmul_source:25:56
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	.loc	1 25 39                         ; triton_cdna4_matmul_source:25:39
	v_lshl_add_u64 v[4:5], v[2:3], 2, v[4:5]
	.loc	1 25 56                         ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_109
; %bb.108:
	global_store_dword v[4:5], a0, off
.LBB0_109:
	.loc	1 0 56                          ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 1, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_111
; %bb.110:
	global_store_dword v[4:5], a1, off offset:4
.LBB0_111:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 2, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_113
; %bb.112:
	global_store_dword v[4:5], a2, off offset:8
.LBB0_113:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 3, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_115
; %bb.114:
	global_store_dword v[4:5], a3, off offset:12
.LBB0_115:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 8, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_117
; %bb.116:
	global_store_dword v[4:5], a4, off offset:32
.LBB0_117:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 9, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_119
; %bb.118:
	global_store_dword v[4:5], a5, off offset:36
.LBB0_119:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 10, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_121
; %bb.120:
	global_store_dword v[4:5], a6, off offset:40
.LBB0_121:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 11, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_123
; %bb.122:
	global_store_dword v[4:5], a7, off offset:44
.LBB0_123:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 16, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_125
; %bb.124:
	global_store_dword v[4:5], a8, off offset:64
.LBB0_125:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 17, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_127
; %bb.126:
	global_store_dword v[4:5], a9, off offset:68
.LBB0_127:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 18, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_129
; %bb.128:
	global_store_dword v[4:5], a10, off offset:72
.LBB0_129:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 19, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_131
; %bb.130:
	global_store_dword v[4:5], a11, off offset:76
.LBB0_131:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 24, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_133
; %bb.132:
	global_store_dword v[4:5], a12, off offset:96
.LBB0_133:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 25, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_135
; %bb.134:
	global_store_dword v[4:5], a13, off offset:100
.LBB0_135:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 26, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[4:5], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_137
; %bb.136:
	global_store_dword v[4:5], a14, off offset:104
.LBB0_137:
	.loc	1 0 56 is_stmt 0                ; triton_cdna4_matmul_source:0:56
	s_or_b64 exec, exec, s[2:3]
	.loc	1 13 31 is_stmt 1               ; triton_cdna4_matmul_source:13:31
	v_or_b32_e32 v0, 27, v2
	.loc	1 26 61                         ; triton_cdna4_matmul_source:26:61
	v_cmp_gt_i32_e64 s[2:3], s9, v0
	.loc	1 26 43 is_stmt 0               ; triton_cdna4_matmul_source:26:43
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 25 56 is_stmt 1               ; triton_cdna4_matmul_source:25:56
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_139
; %bb.138:
	global_store_dword v[4:5], a15, off offset:108
.LBB0_139:
	.loc	1 25 4 is_stmt 0                ; triton_cdna4_matmul_source:25:4
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
		.amdhsa_next_free_vgpr 112
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 96
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
	.set triton_cdna4_matmul_kernel.num_vgpr, 96
	.set triton_cdna4_matmul_kernel.num_agpr, 16
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 38
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5384
; TotalNumSgprs: 44
; NumVgprs: 96
; NumAgprs: 16
; TotalNumVgprs: 112
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 44
; NumVGPRsForWavesPerEU: 112
; AccumOffset: 96
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 23
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
    .sgpr_count:     44
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
