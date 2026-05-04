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
//                            constexprs={"M": 32, "N": 32, "K": 64, "BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}),
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
; %bb.35:
	.file	1 "tests/kernels" "triton_cdna4_matmul_source"
	.loc	1 6 0 prologue_end              ; triton_cdna4_matmul_source:6:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.36:
.LBB0_0:
.Ltmp1:
	.loc	1 15 21 is_stmt 1               ; triton_cdna4_matmul_source:15:21
	s_lshl_b32 s8, s12, 5
	.loc	1 15 44 is_stmt 0               ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v1, 6, v0
	.loc	1 15 31                         ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s8, v1
	.loc	1 21 47 is_stmt 1               ; triton_cdna4_matmul_source:21:47
	v_lshlrev_b32_e32 v4, 6, v1
	.loc	1 22 62                         ; triton_cdna4_matmul_source:22:62
	v_and_b32_e32 v2, 63, v0
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], v[4:5], 1, s[2:3]
	.loc	1 21 51 is_stmt 0               ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 1, v2
	.loc	1 22 49 is_stmt 1               ; triton_cdna4_matmul_source:22:49
	v_cmp_gt_i32_e32 vcc, 32, v1
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[2:3]
	v_mov_b32_e32 v8, 0
	.loc	1 21 25 is_stmt 0               ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	global_load_ushort v8, v[6:7], off
.LBB0_2:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	global_load_ushort v3, v[6:7], off offset:512
.LBB0_4:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	global_load_ushort v9, v[6:7], off offset:1024
.LBB0_6:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_or_b32_e32 v10, 0x300, v4
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, v2
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v10, v[10:11], off
.LBB0_8:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	global_load_ushort v12, v[6:7], off offset:2048
.LBB0_10:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	global_load_ushort v11, v[6:7], off offset:2560
.LBB0_12:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	global_load_ushort v13, v[6:7], off offset:3072
.LBB0_14:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	.loc	1 21 47                         ; triton_cdna4_matmul_source:21:47
	v_or_b32_e32 v4, 0x700, v4
	.loc	1 21 29                         ; triton_cdna4_matmul_source:21:29
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 21 51                         ; triton_cdna4_matmul_source:21:51
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, v2
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	global_load_ushort v14, v[4:5], off
.LBB0_16:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 15 44 is_stmt 1               ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v1, 31, v0
	.loc	1 16 21                         ; triton_cdna4_matmul_source:16:21
	s_lshl_b32 s2, s13, 5
	.loc	1 25 38                         ; triton_cdna4_matmul_source:25:38
	v_and_b32_e32 v2, 0xe0, v0
	.loc	1 16 31                         ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v6, s2, v1
	.loc	1 24 29                         ; triton_cdna4_matmul_source:24:29
	v_lshlrev_b32_e32 v4, 1, v2
	v_mov_b32_e32 v5, 0
	v_lshl_add_u64 v[16:17], s[4:5], 0, v[4:5]
	.loc	1 24 51 is_stmt 0               ; triton_cdna4_matmul_source:24:51
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	1 25 73 is_stmt 1               ; triton_cdna4_matmul_source:25:73
	v_cmp_gt_i32_e32 vcc, 32, v6
	.loc	1 24 51                         ; triton_cdna4_matmul_source:24:51
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[16:17]
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
	global_load_ushort v5, v[6:7], off offset:512
.LBB0_20:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v15, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:
	global_load_ushort v15, v[6:7], off offset:1024
.LBB0_22:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:
	global_load_ushort v4, v[6:7], off offset:1536
.LBB0_24:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_26
; %bb.25:
	global_load_ushort v17, v[6:7], off offset:2048
.LBB0_26:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_28
; %bb.27:
	global_load_ushort v16, v[6:7], off offset:2560
.LBB0_28:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_30
; %bb.29:
	global_load_ushort v19, v[6:7], off offset:3072
.LBB0_30:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_32
; %bb.31:
	global_load_ushort v18, v[6:7], off offset:3584
.LBB0_32:
	.loc	1 0 25                          ; triton_cdna4_matmul_source:0:25
	s_or_b64 exec, exec, s[0:1]
	.loc	1 21 25 is_stmt 1               ; triton_cdna4_matmul_source:21:25
	v_lshlrev_b32_e32 v30, 1, v0
	v_and_b32_e32 v6, 0xfe, v30
	v_bfe_i32 v7, v0, 7, 1
	s_movk_i32 s0, 0x110
	v_bitop3_b32 v6, v7, v6, s0 bitop3:0x6c
	v_add_u32_e32 v31, 0, v6
	v_xad_u32 v32, v6, 32, 0
	s_waitcnt vmcnt(0)
	ds_write_b16 v31, v8
	ds_write_b16 v31, v12 offset:2048
	ds_write_b16 v32, v3 offset:512
	ds_write_b16 v32, v11 offset:2560
	v_xad_u32 v3, v6, 64, 0
	ds_write_b16 v3, v9 offset:1024
	ds_write_b16 v3, v13 offset:3072
	v_xor_b32_e32 v3, 0x60, v6
	v_add_u32_e32 v3, 0, v3
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_and_b32_e32 v28, 32, v0
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	ds_write_b16 v3, v10 offset:1536
	ds_write_b16 v3, v14 offset:3584
	v_lshlrev_b32_e32 v14, 3, v0
	v_lshlrev_b32_e32 v3, 7, v1
	s_movk_i32 s1, 0x70
	v_and_b32_e32 v6, 0x70, v14
	v_lshrrev_b32_e32 v7, 1, v28
	s_movk_i32 s0, 0x60
	v_bitop3_b32 v20, v14, v7, s1 bitop3:0x6c
	v_bitop3_b32 v21, v6, v3, v7 bitop3:0xde
	v_add_u32_e32 v6, 0, v21
	v_xad_u32 v10, v21, 32, 0
	v_xad_u32 v21, v21, 64, 0
	v_bitop3_b32 v3, v20, s0, v3 bitop3:0x36
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_bfe_i32 v29, v0, 5, 1
	.loc	1 21 25                         ; triton_cdna4_matmul_source:21:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[6:9], v6
	ds_read_b128 v[10:13], v10
	v_add_u32_e32 v3, 0, v3
	ds_read_b128 v[20:23], v21
	ds_read_b128 v[24:27], v3
	.loc	1 24 25                         ; triton_cdna4_matmul_source:24:25
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v31, v2
	ds_write_b16 v31, v15 offset:1024
	ds_write_b16 v31, v17 offset:2048
	ds_write_b16 v31, v19 offset:3072
	ds_write_b16 v32, v5 offset:512
	ds_write_b16 v32, v4 offset:1536
	ds_write_b16 v32, v16 offset:2560
	ds_write_b16 v32, v18 offset:3584
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v4, 32, v30
	s_movk_i32 s0, 0x220
	v_and_b32_e32 v2, 0xc0, v2
	v_and_b32_e32 v3, 24, v14
	v_bitop3_b32 v4, v29, v4, s0 bitop3:0x6c
	v_or3_b32 v2, v2, v3, v4
	v_add_u32_e32 v3, 0, v2
	v_xad_u32 v2, v2, 16, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[16:17], v2 offset:256
	ds_read_b64_tr_b16 v[14:15], v3
	ds_read_b64_tr_b16 v[30:31], v3 offset:1024
	ds_read_b64_tr_b16 v[34:35], v3 offset:2048
	ds_read_b64_tr_b16 v[38:39], v3 offset:3072
	ds_read_b64_tr_b16 v[32:33], v2 offset:1280
	ds_read_b64_tr_b16 v[36:37], v2 offset:2304
	ds_read_b64_tr_b16 v[40:41], v2 offset:3328
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[14:17], v[6:9], 0
	.loc	1 15 44                         ; triton_cdna4_matmul_source:15:44
	v_lshrrev_b32_e32 v2, 3, v28
	.loc	1 16 31                         ; triton_cdna4_matmul_source:16:31
	v_or_b32_e32 v2, s2, v2
	.loc	1 15 31                         ; triton_cdna4_matmul_source:15:31
	v_or_b32_e32 v1, s8, v1
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	v_max_i32_e32 v3, v1, v2
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_and_b32_e32 v0, 0xc0, v0
	.loc	1 29 43                         ; triton_cdna4_matmul_source:29:43
	v_cmp_gt_i32_e32 vcc, 32, v3
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	v_cmp_eq_u32_e64 s[0:1], 0, v0
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[30:33], v[10:13], a[0:15]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_b64 s[0:1], s[0:1], vcc
	.loc	1 27 30                         ; triton_cdna4_matmul_source:27:30
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[34:37], v[20:23], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[38:41], v[24:27], a[0:15]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_34
; %bb.33:                               ; %.critedge
	.loc	1 28 35 is_stmt 0               ; triton_cdna4_matmul_source:28:35
	v_lshlrev_b32_e32 v0, 5, v1
	.loc	1 28 17                         ; triton_cdna4_matmul_source:28:17
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	.loc	1 28 39                         ; triton_cdna4_matmul_source:28:39
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 2, v[0:1]
	.loc	1 28 56                         ; triton_cdna4_matmul_source:28:56
	s_nop 4
	global_store_dwordx4 v[0:1], a[0:3], off
	global_store_dwordx4 v[0:1], a[4:7], off offset:32
	global_store_dwordx4 v[0:1], a[8:11], off offset:64
	global_store_dwordx4 v[0:1], a[12:15], off offset:96
.LBB0_34:                               ; %.critedge28
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
		.amdhsa_next_free_vgpr 60
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 44
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
	.set triton_cdna4_matmul_kernel.num_vgpr, 42
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
; codeLenInByte = 1428
; TotalNumSgprs: 20
; NumVgprs: 42
; NumAgprs: 16
; TotalNumVgprs: 60
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 60
; AccumOffset: 44
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 10
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
    .vgpr_count:     60
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
