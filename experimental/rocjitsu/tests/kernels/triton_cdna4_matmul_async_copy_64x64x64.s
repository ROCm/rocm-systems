// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Generated with Triton 3.6.0 for gfx950/CDNA4 from plain Triton, not Gluon.
// Generation used TRITON_HIP_USE_ASYNC_COPY=1. The pointer divisibility hints
// are required for Triton's AMD pipeliner to choose global_load_lds async-copy
// lowering instead of register loads plus local_store.
//
// Full Python source used to generate this assembly:
//
//   from __future__ import annotations
//
//   import argparse
//   import pathlib
//
//   import triton
//   import triton.language as tl
//   from triton import knobs
//   from triton.backends.compiler import GPUTarget
//   from triton.compiler import ASTSource
//
//
//   @triton.jit
//   def triton_cdna4_matmul_kernel(
//       a_ptr,
//       b_ptr,
//       c_ptr,
//       M,
//       N,
//       K,
//       BLOCK_M: tl.constexpr,
//       BLOCK_N: tl.constexpr,
//       BLOCK_K: tl.constexpr,
//       GROUP_M: tl.constexpr,
//       NUM_STAGES: tl.constexpr,
//   ):
//       """Compute C[M, N] = A[M, K] x B[K, N] for dense row-major matrices.
//
//       This source is deliberately minimal and assumes dimensions are
//       multiples of the tile sizes. That keeps the generated code focused on
//       the DBT features we want to study: multi-workgroup dispatch, pipelined
//       async global-to-LDS copies, and MFMA math. Use a masked variant if
//       ragged matrix sizes are needed.
//       """
//
//       pid = tl.program_id(0)
//
//       num_pid_m = tl.cdiv(M, BLOCK_M)
//       num_pid_n = tl.cdiv(N, BLOCK_N)
//       group_size = GROUP_M * num_pid_n
//       group_id = pid // group_size
//       first_pid_m = group_id * GROUP_M
//       group_m = min(num_pid_m - first_pid_m, GROUP_M)
//       pid_in_group = pid - group_id * group_size
//       pid_m = first_pid_m + (pid_in_group % group_m)
//       pid_n = pid_in_group // group_m
//
//       offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
//       offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
//       offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
//       offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
//       offs_k = tl.arange(0, BLOCK_K)
//
//       a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
//       b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
//       accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
//
//       for _ in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
//           a = tl.load(tl.multiple_of(a_ptrs, [16, 16]))
//           b = tl.load(tl.multiple_of(b_ptrs, [16, 16]))
//           accumulator = tl.dot(a, b, acc=accumulator)
//           a_ptrs += BLOCK_K
//           b_ptrs += BLOCK_K * N
//
//       c = accumulator.to(tl.float16)
//       c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
//       tl.store(c_ptrs, c)
//
//
//   def launch_grid(M: int, N: int, block_m: int = 64, block_n: int = 64) -> tuple[int]:
//       """Return a one-dimensional grid with one program per output tile."""
//
//       return (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
//
//
//   def matmul_gfx950_async_copy(
//       a,
//       b,
//       c,
//       *,
//       block_m: int = 64,
//       block_n: int = 64,
//       block_k: int = 64,
//       group_m: int = 4,
//       num_stages: int = 4,
//       num_warps: int = 4,
//   ):
//       """Launch the gfx950 async-copy matmul for dense row-major FP16 tensors."""
//
//       M = a.shape[0]
//       K = a.shape[1]
//       N = b.shape[1]
//       grid = launch_grid(M, N, block_m, block_n)
//       with knobs.amd.scope():
//           knobs.amd.use_async_copy = True
//           triton_cdna4_matmul_kernel[grid](
//               a,
//               b,
//               c,
//               M,
//               N,
//               K,
//               BLOCK_M=block_m,
//               BLOCK_N=block_n,
//               BLOCK_K=block_k,
//               GROUP_M=group_m,
//               NUM_STAGES=num_stages,
//               num_warps=num_warps,
//               num_stages=num_stages,
//           )
//       return c
//
//
//   def compile_gfx950(
//       *,
//       block_m: int = 64,
//       block_n: int = 64,
//       block_k: int = 64,
//       group_m: int = 4,
//       num_stages: int = 4,
//       num_warps: int = 4,
//   ):
//       """Compile the row-major gfx950 specialization without requiring a GPU."""
//
//       source = ASTSource(
//           fn=triton_cdna4_matmul_kernel,
//           signature={
//               "a_ptr": "*fp16",
//               "b_ptr": "*fp16",
//               "c_ptr": "*fp16",
//               "M": "i32",
//               "N": "i32",
//               "K": "i32",
//               "BLOCK_M": "constexpr",
//               "BLOCK_N": "constexpr",
//               "BLOCK_K": "constexpr",
//               "GROUP_M": "constexpr",
//               "NUM_STAGES": "constexpr",
//           },
//           constexprs={
//               "BLOCK_M": block_m,
//               "BLOCK_N": block_n,
//               "BLOCK_K": block_k,
//               "GROUP_M": group_m,
//               "NUM_STAGES": num_stages,
//           },
//       )
//       with knobs.amd.scope(), knobs.compilation.scope():
//           knobs.amd.use_async_copy = True
//           # The async-copy knob is not part of older Triton cache keys, so
//           # force a fresh offline compile for this inspection helper.
//           knobs.compilation.always_compile = True
//           return triton.compile(
//               source,
//               target=GPUTarget("hip", "gfx950", 64),
//               options={"num_warps": num_warps, "num_stages": num_stages},
//           )
//
//
//   def main() -> None:
//       parser = argparse.ArgumentParser()
//       parser.add_argument("--emit", choices=("ttgir", "amdgcn", "hsaco"), default="amdgcn")
//       parser.add_argument("--output", type=pathlib.Path)
//       args = parser.parse_args()
//
//       compiled = compile_gfx950()
//       blob = compiled.asm[args.emit]
//       if args.output is None:
//           if isinstance(blob, bytes):
//               raise SystemExit("--output is required when emitting hsaco bytes")
//           print(blob)
//           return
//
//       if isinstance(blob, bytes):
//           args.output.write_bytes(blob)
//       else:
//           args.output.write_text(str(blob))
//
//
//   if __name__ == "__main__":
//       main()
//
// Generation command:
//
//   TRITON_CACHE_DIR=/tmp/rocjitsu-triton-cache \
//     .env/bin/python tests/kernels/triton_gfx950_async_matmul.py \
//     --emit amdgcn \
//     --output tests/kernels/triton_cdna4_matmul_async_copy_64x64x64.s

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
; %bb.48:
	.file	1 "tests/kernels" "triton_cdna4_matmul_async_copy_source"
	.loc	1 26 0 prologue_end             ; triton_cdna4_matmul_async_copy_source:26:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.49:
.LBB0_0:
.Ltmp1:
	.file	2 "triton/language" "standard.py"
	.loc	2 43 17 is_stmt 1               ; standard.py:43:17 @[ triton_cdna4_matmul_async_copy_source:51:27 ]
	s_add_i32 s0, s9, 63
	.loc	2 43 30 is_stmt 0               ; standard.py:43:30 @[ triton_cdna4_matmul_async_copy_source:51:27 ]
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 26
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 6
.Ltmp2:
	.loc	1 52 27 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:52:27
	s_lshl_b32 s0, s0, 2
	.loc	1 53 22                         ; triton_cdna4_matmul_async_copy_source:53:22
	s_abs_i32 s1, s0
	v_cvt_f32_u32_e32 v1, s1
	s_sub_i32 s13, 0, s1
.Ltmp3:
	.loc	2 43 17                         ; standard.py:43:17 @[ triton_cdna4_matmul_async_copy_source:50:27 ]
	s_add_i32 s8, s8, 63
.Ltmp4:
	.loc	1 53 22                         ; triton_cdna4_matmul_async_copy_source:53:22
	s_abs_i32 s12, s16
	v_rcp_iflag_f32_e32 v1, v1
.Ltmp5:
	.loc	2 43 30                         ; standard.py:43:30 @[ triton_cdna4_matmul_async_copy_source:50:27 ]
	s_ashr_i32 s11, s8, 31
	s_lshr_b32 s11, s11, 26
	s_add_i32 s8, s8, s11
.Ltmp6:
	.loc	1 53 22                         ; triton_cdna4_matmul_async_copy_source:53:22
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s11, s16, s0
.Ltmp7:
	.loc	2 43 30                         ; standard.py:43:30 @[ triton_cdna4_matmul_async_copy_source:50:27 ]
	s_ashr_i32 s8, s8, 6
.Ltmp8:
	.loc	1 53 22                         ; triton_cdna4_matmul_async_copy_source:53:22
	s_ashr_i32 s11, s11, 31
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s13, s13, s14
	s_mul_hi_u32 s13, s14, s13
	s_add_i32 s14, s14, s13
	s_mul_hi_u32 s13, s12, s14
	s_mul_i32 s14, s13, s1
	s_sub_i32 s12, s12, s14
	s_add_i32 s14, s13, 1
	s_sub_i32 s15, s12, s1
	s_cmp_ge_u32 s12, s1
	s_cselect_b32 s13, s14, s13
	s_cselect_b32 s12, s15, s12
	s_add_i32 s14, s13, 1
	s_cmp_ge_u32 s12, s1
	s_cselect_b32 s1, s14, s13
	s_xor_b32 s1, s1, s11
	s_sub_i32 s1, s1, s11
	.loc	1 54 29                         ; triton_cdna4_matmul_async_copy_source:54:29
	s_lshl_b32 s13, s1, 2
	.loc	1 55 30                         ; triton_cdna4_matmul_async_copy_source:55:30
	s_sub_i32 s8, s8, s13
	.loc	1 55 43 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:55:43
	s_min_i32 s8, s8, 4
	.loc	1 58 28 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:58:28
	s_abs_i32 s12, s8
	v_cvt_f32_u32_e32 v1, s12
	.loc	1 56 36                         ; triton_cdna4_matmul_async_copy_source:56:36
	s_mul_i32 s1, s1, s0
	.loc	1 58 28                         ; triton_cdna4_matmul_async_copy_source:58:28
	s_sub_i32 s14, 0, s12
	.loc	1 56 25                         ; triton_cdna4_matmul_async_copy_source:56:25
	s_sub_i32 s0, s16, s1
	.loc	1 58 28                         ; triton_cdna4_matmul_async_copy_source:58:28
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s1, s0, s8
	s_ashr_i32 s11, s1, 31
	s_abs_i32 s1, s0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	.loc	1 60 44                         ; triton_cdna4_matmul_async_copy_source:60:44
	v_lshlrev_b32_e32 v16, 3, v0
	v_and_b32_e32 v22, 56, v16
	v_and_b32_e32 v8, 63, v0
	.loc	1 58 28                         ; triton_cdna4_matmul_async_copy_source:58:28
	v_readfirstlane_b32 s15, v1
	s_mul_i32 s14, s14, s15
	s_mul_hi_u32 s14, s15, s14
	s_add_i32 s15, s15, s14
	s_mul_hi_u32 s14, s1, s15
	s_mul_i32 s15, s14, s12
	s_sub_i32 s1, s1, s15
	s_add_i32 s15, s14, 1
	s_sub_i32 s16, s1, s12
	s_cmp_ge_u32 s1, s12
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s1, s16, s1
	s_add_i32 s15, s14, 1
	s_cmp_ge_u32 s1, s12
	s_cselect_b32 s1, s15, s14
	s_xor_b32 s24, s1, s11
	s_sub_i32 s12, s24, s11
	.loc	1 57 42                         ; triton_cdna4_matmul_async_copy_source:57:42
	s_mul_i32 s1, s12, s8
	s_sub_i32 s0, s0, s1
	.loc	1 57 27 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:57:27
	s_add_i32 s13, s13, s0
	.loc	1 60 21 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:60:21
	s_lshl_b32 s8, s13, 6
	.loc	1 60 44 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:60:44
	v_lshrrev_b32_e32 v1, 3, v0
	.loc	1 60 31                         ; triton_cdna4_matmul_async_copy_source:60:31
	v_or_b32_e32 v2, s8, v1
	.loc	1 66 39 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:66:39
	v_mul_lo_u32 v2, v2, s10
	.loc	1 66 21 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:66:21
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[2:3]
	.loc	1 66 43                         ; triton_cdna4_matmul_async_copy_source:66:43
	v_mov_b32_e32 v3, 0
	v_lshlrev_b32_e32 v2, 1, v22
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[2:3]
	.loc	1 71 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshrrev_b32_e32 v4, 1, v0
	v_bitop3_b32 v4, v16, v4, 56 bitop3:0x78
	v_sub_u32_e32 v4, v4, v16
	v_ashrrev_i32_e32 v4, 3, v4
.Ltmp9:
	.loc	2 43 17                         ; standard.py:43:17 @[ triton_cdna4_matmul_async_copy_source:70:36 ]
	s_add_i32 s25, s10, 63
.Ltmp10:
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v47, v4, v8
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	v_readfirstlane_b32 s17, v0
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cmp_gt_i32 s25, 63
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshlrev_b32_e32 v48, 2, v47
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cselect_b64 s[14:15], -1, 0
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_lshl_b32 s0, s17, 4
	ds_bpermute_b32 v4, v48, v6
	ds_bpermute_b32 v5, v48, v7
	s_and_b32 s23, s0, 0xc00
	v_cmp_gt_i32_e64 s[0:1], s25, 63
	s_nop 1
	v_lshrrev_b64 v[8:9], v47, s[0:1]
	v_and_b32_e32 v8, 1, v8
	v_cmp_eq_u32_e32 vcc, 1, v8
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 m0, s23, 0
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[4:5], off
.LBB0_2:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	.loc	1 60 31 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:60:31
	s_waitcnt lgkmcnt(0)
	v_or3_b32 v4, v1, s8, 32
	.loc	1 66 39                         ; triton_cdna4_matmul_async_copy_source:66:39
	v_mul_lo_u32 v4, v4, s10
	.loc	1 66 21 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:66:21
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 66 43                         ; triton_cdna4_matmul_async_copy_source:66:43
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[2:3]
	.loc	1 71 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v2, v48, v8
	ds_bpermute_b32 v3, v48, v9
	v_cndmask_b32_e64 v17, 0, 1, s[14:15]
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[4:5], v47, vcc
	v_and_b32_e32 v4, 1, v4
	v_cmp_eq_u32_e32 vcc, 1, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s13, s23, 0
	s_add_i32 m0, s13, 0x1000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[2:3], off
.LBB0_4:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	.loc	1 61 21 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:61:21
	s_lshl_b32 s22, s12, 6
	.loc	1 67 39                         ; triton_cdna4_matmul_async_copy_source:67:39
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v2, s9, v1
	.loc	1 61 31                         ; triton_cdna4_matmul_async_copy_source:61:31
	v_or_b32_e32 v12, s22, v22
	.loc	1 67 21                         ; triton_cdna4_matmul_async_copy_source:67:21
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[4:5]
	.loc	1 67 43 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:67:43
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[10:11], v[12:13], 1, v[4:5]
	.loc	1 72 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:72:20
	ds_bpermute_b32 v4, v48, v10
	ds_bpermute_b32 v5, v48, v11
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[14:15], v47, vcc
	v_and_b32_e32 v14, 1, v14
	v_cmp_eq_u32_e32 vcc, 1, v14
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s12, s23, 0
	s_add_i32 m0, s12, 0x8000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[4:5], off
.LBB0_6:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	.loc	1 67 39 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:67:39
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, s9, 5, v2
	.loc	1 67 21 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:67:21
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[14:15], v[4:5], 1, s[4:5]
	.loc	1 67 43                         ; triton_cdna4_matmul_async_copy_source:67:43
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[14:15]
	.loc	1 72 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:72:20
	ds_bpermute_b32 v14, v48, v12
	ds_bpermute_b32 v15, v48, v13
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[18:19], v47, vcc
	v_and_b32_e32 v17, 1, v18
	v_cmp_eq_u32_e32 vcc, 1, v17
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s12, s23, 0
	s_add_i32 m0, s12, 0x9000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[14:15], off
.LBB0_8:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0x80
	.loc	1 74 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:74:18
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[14:15], v[6:7], 0, s[0:1]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v14, v48, v14
	ds_bpermute_b32 v15, v48, v15
	v_mov_b32_e32 v17, 0x7f
	v_cmp_gt_i32_e32 vcc, s25, v17
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cmpk_gt_i32 s25, 0x7f
	s_cselect_b64 s[12:13], -1, 0
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshrrev_b64 v[18:19], v47, vcc
	v_and_b32_e32 v17, 1, v18
	v_cmp_eq_u32_e32 vcc, 1, v17
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s18, s23, 0
	s_add_i32 m0, s18, 0x2000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_nop 0
	global_load_lds_dwordx4 v[14:15], off
.LBB0_10:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[16:17]
	.loc	1 74 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:74:18
	v_lshl_add_u64 v[14:15], v[8:9], 0, s[0:1]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v14, v48, v14
	ds_bpermute_b32 v15, v48, v15
	v_cndmask_b32_e64 v17, 0, 1, s[12:13]
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[18:19], v47, vcc
	v_and_b32_e32 v18, 1, v18
	v_cmp_eq_u32_e32 vcc, 1, v18
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s16, s23, 0
	s_add_i32 m0, s16, 0x3000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[14:15], off
.LBB0_12:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	.loc	1 75 28 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:75:28
	s_lshl_b32 s20, s9, 6
	.loc	1 75 18 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:75:18
	s_ashr_i32 s21, s20, 31
	v_lshl_add_u64 v[10:11], s[20:21], 1, v[10:11]
	.loc	1 72 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v14, v48, v10
	ds_bpermute_b32 v15, v48, v11
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[18:19], v47, vcc
	v_and_b32_e32 v18, 1, v18
	v_cmp_eq_u32_e32 vcc, 1, v18
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s16, s23, 0
	s_add_i32 m0, s16, 0xa000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[14:15], off
.LBB0_14:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	.loc	1 75 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:75:18
	v_lshl_add_u64 v[12:13], s[20:21], 1, v[12:13]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v14, v48, v12
	ds_bpermute_b32 v15, v48, v13
	v_cmp_ne_u32_e32 vcc, 0, v17
	s_nop 1
	v_lshrrev_b64 v[18:19], v47, vcc
	v_and_b32_e32 v17, 1, v18
	v_cmp_eq_u32_e32 vcc, 1, v17
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s16, s23, 0
	s_add_i32 m0, s16, 0xb000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[14:15], off
.LBB0_16:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[16:17], 0x100
	.loc	1 74 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:74:18
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[16:17]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v6, v48, v6
	ds_bpermute_b32 v7, v48, v7
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v14, 0xbf
	v_cmp_gt_i32_e32 vcc, s25, v14
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cmpk_gt_i32 s25, 0xbf
	s_cselect_b64 s[0:1], -1, 0
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshrrev_b64 v[14:15], v47, vcc
	v_and_b32_e32 v14, 1, v14
	v_cmp_eq_u32_e32 vcc, 1, v14
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s26, s23, 0
	s_add_i32 m0, s26, 0x4000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_nop 0
	global_load_lds_dwordx4 v[6:7], off
.LBB0_18:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[18:19]
	.loc	1 74 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:74:18
	v_lshl_add_u64 v[6:7], v[8:9], 0, s[16:17]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v6, v48, v6
	ds_bpermute_b32 v7, v48, v7
	v_cndmask_b32_e64 v8, 0, 1, s[0:1]
	v_cmp_ne_u32_e32 vcc, 0, v8
	s_nop 1
	v_lshrrev_b64 v[14:15], v47, vcc
	v_and_b32_e32 v9, 1, v14
	v_cmp_eq_u32_e32 vcc, 1, v9
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s18, s23, 0
	s_add_i32 m0, s18, 0x5000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[6:7], off
.LBB0_20:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[16:17]
	.loc	1 75 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:75:18
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[6:7], s[20:21], 1, v[10:11]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_bpermute_b32 v6, v48, v6
	ds_bpermute_b32 v7, v48, v7
	v_cmp_ne_u32_e32 vcc, 0, v8
	s_nop 1
	v_lshrrev_b64 v[10:11], v47, vcc
	v_and_b32_e32 v9, 1, v10
	v_cmp_eq_u32_e32 vcc, 1, v9
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s18, s23, 0
	s_add_i32 m0, s18, 0xc000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[6:7], off
.LBB0_22:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[16:17]
	.loc	1 75 18 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:75:18
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[6:7], s[20:21], 1, v[12:13]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_bpermute_b32 v6, v48, v6
	ds_bpermute_b32 v7, v48, v7
	v_cmp_ne_u32_e32 vcc, 0, v8
	s_nop 1
	v_lshrrev_b64 v[8:9], v47, vcc
	v_and_b32_e32 v8, 1, v8
	v_cmp_eq_u32_e32 vcc, 1, v8
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s18, s23, 0
	s_add_i32 m0, s18, 0xd000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[6:7], off
.LBB0_24:
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[16:17]
	.loc	1 60 44 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:60:44
	v_and_b32_e32 v40, 32, v0
	s_waitcnt lgkmcnt(0)
	v_lshrrev_b32_e32 v6, 1, v40
	s_movk_i32 s16, 0x70
	v_bitop3_b32 v8, v16, v6, s16 bitop3:0x6c
	v_lshlrev_b32_e32 v6, 5, v0
	v_and_b32_e32 v9, 24, v16
	s_movk_i32 s17, 0x580
	v_lshlrev_b32_e32 v10, 1, v0
	v_and_or_b32 v6, v6, s17, v9
	v_and_b32_e32 v39, 0x80, v0
	v_and_b32_e32 v41, 64, v0
	v_bitop3_b32 v6, v10, v6, s16 bitop3:0x6c
	v_and_b32_e32 v38, 31, v0
	v_lshlrev_b32_e32 v7, 5, v39
	v_xor_b32_e32 v46, v6, v41
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cmpk_gt_i32 s25, 0xff
	v_lshl_or_b32 v7, v38, 7, v7
	s_movk_i32 s18, 0x60
	v_xor_b32_e32 v49, 32, v46
	s_cbranch_scc1 .LBB0_26
; %bb.25:                               ; %.._crit_edge_crit_edge
	.loc	1 0 46 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:46
	v_xor_b32_e32 v6, 32, v46
	s_mov_b64 s[16:17], 0
	s_branch .LBB0_27
.LBB0_26:
	s_mov_b64 s[16:17], -1
                                        ; implicit-def: $vgpr6
.LBB0_27:                               ; %Flow54
	v_or_b32_e32 v45, v7, v8
	v_bitop3_b32 v44, v7, 32, v8 bitop3:0x36
	v_bitop3_b32 v42, v7, 64, v8 bitop3:0x36
	s_andn2_b64 vcc, exec, s[16:17]
	v_bitop3_b32 v43, v7, s18, v8 bitop3:0x36
	s_cbranch_vccnz .LBB0_39
; %bb.28:                               ; %.lr.ph
.Ltmp11:
	.loc	2 43 30 is_stmt 1               ; standard.py:43:30 @[ triton_cdna4_matmul_async_copy_source:70:36 ]
	s_ashr_i32 s16, s25, 31
	s_lshr_b32 s16, s16, 26
	s_add_i32 s25, s25, s16
.Ltmp12:
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_and_b32_e32 v0, 7, v0
.Ltmp13:
	.loc	2 43 30                         ; standard.py:43:30 @[ triton_cdna4_matmul_async_copy_source:70:36 ]
	s_ashr_i32 s16, s25, 6
.Ltmp14:
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_lshlrev_b32_e32 v16, 4, v0
	v_or_b32_e32 v0, 32, v1
	s_max_i32 s25, s16, 4
	s_mul_i32 s18, s9, 0x180
	v_mul_lo_u32 v0, s9, v0
	s_lshl_b32 s26, s24, 7
	s_lshl_b32 s27, s11, 7
	s_lshl_b32 s16, s9, 7
	v_lshl_add_u32 v0, v0, 1, s26
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_add_u32 s18, s4, s18
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_subrev_u32_e32 v18, s27, v0
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_addc_u32 s19, s5, 0
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_lshl_or_b32 v0, s24, 6, v22
	s_lshl_b32 s11, s11, 6
	v_subrev_u32_e32 v6, s11, v0
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	v_mov_b32_e32 v17, 0
	v_lshlrev_b64 v[4:5], 1, v[4:5]
	v_lshl_add_u32 v0, v2, 1, s26
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	v_lshl_add_u64 v[20:21], s[18:19], 0, v[16:17]
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_mad_i64_i32 v[22:23], s[18:19], s20, 6, v[4:5]
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	s_lshl_b64 s[18:19], s[20:21], 1
	v_mad_i64_i32 v[26:27], s[20:21], s20, 6, v[2:3]
	v_add_u32_e32 v2, s8, v1
	v_subrev_u32_e32 v24, s27, v0
	v_add_u32_e32 v0, 32, v2
	v_mul_lo_u32 v0, s10, v0
	v_lshl_add_u32 v1, v0, 1, s2
	v_add_u32_e32 v28, 0x180, v1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	s_mov_b64 s[20:21], 0x180
	v_lshl_add_u64 v[30:31], v[0:1], 0, s[20:21]
	v_mul_lo_u32 v0, s10, v2
	v_lshl_add_u32 v1, v0, 1, s2
	v_add_u32_e32 v32, 0x180, v1
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s17, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
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
	v_mov_b32_e32 v19, v17
	v_mov_b32_e32 v25, v17
	v_mov_b32_e32 v29, v17
	v_mov_b32_e32 v33, v17
	v_lshl_add_u64 v[34:35], v[0:1], 0, s[20:21]
	s_add_i32 s21, 0, 0x2000
	s_add_i32 s20, 0, 0x4000
	s_add_i32 s29, 0, 0x8000
	s_add_i32 s24, 0, 0xa000
	s_add_i32 s10, 0, 0xc000
	s_add_i32 s27, s25, -3
	s_mov_b32 s28, 2
	s_mov_b64 s[2:3], 0x80
	s_mov_b32 s30, s17
.LBB0_29:                               ; =>This Inner Loop Header: Depth=1
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshl_add_u64 v[0:1], v[16:17], 0, v[34:35]
	v_add_u32_e32 v0, v16, v32
	s_mov_b32 s26, s24
	s_mov_b32 s24, s10
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_add_i32 s10, s28, 1
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_bpermute_b32 v0, v48, v0
	ds_bpermute_b32 v1, v48, v1
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cmp_lt_i32 s10, 4
	s_cselect_b32 s28, s10, 0
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshrrev_b64 v[2:3], v47, exec
	s_lshl_b32 s10, s28, 13
	v_and_b32_e32 v2, 1, v2
	s_mov_b32 s25, s21
	s_mov_b32 s21, s20
	s_add_i32 s20, s10, 0
	v_cmp_eq_u32_e32 vcc, 1, v2
	s_waitcnt vmcnt(8) lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 m0, s20, s23
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_nop 0
	global_load_lds_dwordx4 v[0:1], off
.LBB0_31:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_lshl_add_u64 v[0:1], v[16:17], 0, v[30:31]
	v_add_u32_e32 v0, v16, v28
	ds_bpermute_b32 v0, v48, v0
	ds_bpermute_b32 v1, v48, v1
	v_lshrrev_b64 v[2:3], v47, exec
	v_and_b32_e32 v2, 1, v2
	v_cmp_eq_u32_e32 vcc, 1, v2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s31, s20, s23
	s_add_i32 m0, s31, 0x1000
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[0:1], off
.LBB0_33:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v0, s30, v45
	v_add_u32_e32 v1, s30, v44
	ds_read_b128 v[8:11], v0
	ds_read_b128 v[4:7], v1
	v_add_u32_e32 v0, s30, v42
	v_add_u32_e32 v1, s30, v43
	.loc	1 72 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:72:20
	v_lshl_add_u64 v[36:37], s[4:5], 0, v[26:27]
	v_add_u32_e32 v21, v20, v24
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_read_b128 v[12:15], v0
	ds_read_b128 v[0:3], v1
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_bpermute_b32 v36, v48, v21
	ds_bpermute_b32 v37, v48, v37
	v_lshrrev_b64 v[50:51], v47, exec
	v_and_b32_e32 v21, 1, v50
	v_cmp_eq_u32_e32 vcc, 1, v21
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s30, s20, s23
	s_add_i32 m0, s30, 0x8000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[36:37], off
.LBB0_35:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[36:37], s[4:5], 0, v[22:23]
	v_add_u32_e32 v21, v20, v18
	ds_bpermute_b32 v36, v48, v21
	ds_bpermute_b32 v37, v48, v37
	v_lshrrev_b64 v[50:51], v47, exec
	v_and_b32_e32 v21, 1, v50
	v_cmp_eq_u32_e32 vcc, 1, v21
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_add_i32 s30, s20, s23
	s_add_i32 m0, s30, 0x9000
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	global_load_lds_dwordx4 v[36:37], off
.LBB0_37:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 20                          ; triton_cdna4_matmul_async_copy_source:0:20
	s_or_b64 exec, exec, s[10:11]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v36, s29, v49
	v_add_u32_e32 v21, s29, v46
	ds_read_b64_tr_b16 v[52:53], v36 offset:512
	ds_read_b64_tr_b16 v[50:51], v21
	ds_read_b64_tr_b16 v[54:55], v21 offset:2048
	ds_read_b64_tr_b16 v[58:59], v21 offset:4096
	ds_read_b64_tr_b16 v[62:63], v21 offset:6144
	ds_read_b64_tr_b16 v[56:57], v36 offset:2560
	ds_read_b64_tr_b16 v[60:61], v36 offset:4608
	ds_read_b64_tr_b16 v[64:65], v36 offset:6656
	.loc	1 73 32 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[50:53], v[8:11], a[0:15]
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_add_i32 s10, s20, 0x8000
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_add_i32 s27, s27, -1
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[16:17]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[18:19]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[16:17]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[18:19]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[2:3]
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	v_mfma_f32_32x32x16_f16 a[0:15], v[54:57], v[4:7], a[0:15]
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[2:3]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[2:3]
	s_cmp_lg_u32 s27, 0
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[2:3]
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	v_mfma_f32_32x32x16_f16 a[0:15], v[58:61], v[12:15], a[0:15]
	v_mfma_f32_32x32x16_f16 a[0:15], v[62:65], v[0:3], a[0:15]
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_cbranch_scc0 .LBB0_40
; %bb.38:                               ;   in Loop: Header=BB0_29 Depth=1
	.loc	1 0 46 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:0:46
	s_mov_b32 s29, s26
	s_mov_b32 s30, s25
	.loc	1 70 46                         ; triton_cdna4_matmul_async_copy_source:70:46
	s_branch .LBB0_29
.LBB0_39:
	.loc	1 0 46                          ; triton_cdna4_matmul_async_copy_source:0:46
	s_mov_b32 s25, 0
	s_add_i32 s21, 0, 0x2000
	s_add_i32 s20, 0, 0x4000
	s_add_i32 s26, 0, 0x8000
	s_add_i32 s24, 0, 0xa000
	s_add_i32 s10, 0, 0xc000
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
	s_branch .LBB0_41
.LBB0_40:                               ; %Flow
	v_mov_b32_e32 v6, v49
.LBB0_41:                               ; %._crit_edge
	.loc	1 72 20 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:72:20
	v_add_u32_e32 v0, s26, v46
	v_add_u32_e32 v2, s26, v6
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_read_b64_tr_b16 v[32:33], v0
	ds_read_b64_tr_b16 v[28:29], v0 offset:2048
	ds_read_b64_tr_b16 v[24:25], v0 offset:4096
	ds_read_b64_tr_b16 v[0:1], v0 offset:6144
	ds_read_b64_tr_b16 v[34:35], v2 offset:512
	ds_read_b64_tr_b16 v[30:31], v2 offset:2560
	ds_read_b64_tr_b16 v[26:27], v2 offset:4608
	ds_read_b64_tr_b16 v[2:3], v2 offset:6656
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_andn2_b64 vcc, exec, s[14:15]
	s_cbranch_vccnz .LBB0_43
; %bb.42:
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s25, v45
	ds_read_b128 v[48:51], v4
	v_accvgpr_read_b32 v8, a0
	v_accvgpr_read_b32 v9, a1
	v_accvgpr_read_b32 v10, a2
	v_accvgpr_read_b32 v11, a3
	v_accvgpr_read_b32 v12, a4
	v_accvgpr_read_b32 v13, a5
	v_accvgpr_read_b32 v14, a6
	v_accvgpr_read_b32 v15, a7
	v_accvgpr_read_b32 v16, a8
	v_accvgpr_read_b32 v17, a9
	v_accvgpr_read_b32 v18, a10
	v_accvgpr_read_b32 v19, a11
	v_accvgpr_read_b32 v20, a12
	v_accvgpr_read_b32 v21, a13
	v_accvgpr_read_b32 v22, a14
	v_accvgpr_read_b32 v23, a15
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	v_accvgpr_write_b32 a0, v8
	v_accvgpr_write_b32 a1, v9
	v_accvgpr_write_b32 a2, v10
	v_accvgpr_write_b32 a3, v11
	v_accvgpr_write_b32 a4, v12
	v_accvgpr_write_b32 a5, v13
	v_accvgpr_write_b32 a6, v14
	v_accvgpr_write_b32 a7, v15
	v_accvgpr_write_b32 a8, v16
	v_accvgpr_write_b32 a9, v17
	v_accvgpr_write_b32 a10, v18
	v_accvgpr_write_b32 a11, v19
	v_accvgpr_write_b32 a12, v20
	v_accvgpr_write_b32 a13, v21
	v_accvgpr_write_b32 a14, v22
	v_accvgpr_write_b32 a15, v23
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s25, v44
	ds_read_b128 v[8:11], v4
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[48:51], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s25, v42
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[28:31], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_read_b128 v[8:11], v4
	v_add_u32_e32 v4, s25, v43
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[24:27], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_read_b128 v[8:11], v4
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[0:3], v[8:11], a[0:15]
.LBB0_43:                               ; %._crit_edge._crit_edge
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(4)
	v_add_u32_e32 v0, s24, v46
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v2, s24, v6
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_read_b64_tr_b16 v[32:33], v0
	ds_read_b64_tr_b16 v[28:29], v0 offset:2048
	ds_read_b64_tr_b16 v[24:25], v0 offset:4096
	ds_read_b64_tr_b16 v[0:1], v0 offset:6144
	ds_read_b64_tr_b16 v[34:35], v2 offset:512
	ds_read_b64_tr_b16 v[30:31], v2 offset:2560
	ds_read_b64_tr_b16 v[26:27], v2 offset:4608
	ds_read_b64_tr_b16 v[2:3], v2 offset:6656
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_45
; %bb.44:
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s21, v45
	ds_read_b128 v[48:51], v4
	.loc	1 0 0 is_stmt 0                 ; triton_cdna4_matmul_async_copy_source:0
	v_accvgpr_read_b32 v8, a0
	v_accvgpr_read_b32 v9, a1
	v_accvgpr_read_b32 v10, a2
	v_accvgpr_read_b32 v11, a3
	v_accvgpr_read_b32 v12, a4
	v_accvgpr_read_b32 v13, a5
	v_accvgpr_read_b32 v14, a6
	v_accvgpr_read_b32 v15, a7
	v_accvgpr_read_b32 v16, a8
	v_accvgpr_read_b32 v17, a9
	v_accvgpr_read_b32 v18, a10
	v_accvgpr_read_b32 v19, a11
	v_accvgpr_read_b32 v20, a12
	v_accvgpr_read_b32 v21, a13
	v_accvgpr_read_b32 v22, a14
	v_accvgpr_read_b32 v23, a15
	.loc	1 73 32 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:73:32
	v_accvgpr_write_b32 a0, v8
	v_accvgpr_write_b32 a1, v9
	v_accvgpr_write_b32 a2, v10
	v_accvgpr_write_b32 a3, v11
	v_accvgpr_write_b32 a4, v12
	v_accvgpr_write_b32 a5, v13
	v_accvgpr_write_b32 a6, v14
	v_accvgpr_write_b32 a7, v15
	v_accvgpr_write_b32 a8, v16
	v_accvgpr_write_b32 a9, v17
	v_accvgpr_write_b32 a10, v18
	v_accvgpr_write_b32 a11, v19
	v_accvgpr_write_b32 a12, v20
	v_accvgpr_write_b32 a13, v21
	v_accvgpr_write_b32 a14, v22
	v_accvgpr_write_b32 a15, v23
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s21, v44
	ds_read_b128 v[8:11], v4
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[48:51], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s21, v42
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[28:31], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_read_b128 v[8:11], v4
	v_add_u32_e32 v4, s21, v43
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[24:27], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	ds_read_b128 v[8:11], v4
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[0:3], v[8:11], a[0:15]
.LBB0_45:
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	s_waitcnt lgkmcnt(4)
	v_add_u32_e32 v0, s10, v46
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v2, s10, v6
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 72 20                         ; triton_cdna4_matmul_async_copy_source:72:20
	ds_read_b64_tr_b16 v[28:29], v0
	ds_read_b64_tr_b16 v[24:25], v0 offset:2048
	ds_read_b64_tr_b16 v[4:5], v0 offset:4096
	ds_read_b64_tr_b16 v[0:1], v0 offset:6144
	ds_read_b64_tr_b16 v[30:31], v2 offset:512
	ds_read_b64_tr_b16 v[26:27], v2 offset:2560
	ds_read_b64_tr_b16 v[6:7], v2 offset:4608
	ds_read_b64_tr_b16 v[2:3], v2 offset:6656
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_47
; %bb.46:
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v32, s20, v45
	ds_read_b128 v[32:35], v32
	v_accvgpr_read_b32 v8, a0
	v_accvgpr_read_b32 v9, a1
	v_accvgpr_read_b32 v10, a2
	v_accvgpr_read_b32 v11, a3
	v_accvgpr_read_b32 v12, a4
	v_accvgpr_read_b32 v13, a5
	v_accvgpr_read_b32 v14, a6
	v_accvgpr_read_b32 v15, a7
	v_accvgpr_read_b32 v16, a8
	v_accvgpr_read_b32 v17, a9
	v_accvgpr_read_b32 v18, a10
	v_accvgpr_read_b32 v19, a11
	v_accvgpr_read_b32 v20, a12
	v_accvgpr_read_b32 v21, a13
	v_accvgpr_read_b32 v22, a14
	v_accvgpr_read_b32 v23, a15
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	v_accvgpr_write_b32 a0, v8
	v_accvgpr_write_b32 a1, v9
	v_accvgpr_write_b32 a2, v10
	v_accvgpr_write_b32 a3, v11
	v_accvgpr_write_b32 a4, v12
	v_accvgpr_write_b32 a5, v13
	v_accvgpr_write_b32 a6, v14
	v_accvgpr_write_b32 a7, v15
	v_accvgpr_write_b32 a8, v16
	v_accvgpr_write_b32 a9, v17
	v_accvgpr_write_b32 a10, v18
	v_accvgpr_write_b32 a11, v19
	v_accvgpr_write_b32 a12, v20
	v_accvgpr_write_b32 a13, v21
	v_accvgpr_write_b32 a14, v22
	v_accvgpr_write_b32 a15, v23
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v8, s20, v44
	ds_read_b128 v[8:11], v8
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[28:31], v[32:35], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[24:27], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v8, s20, v42
	ds_read_b128 v[8:11], v8
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[4:7], v[8:11], a[0:15]
	.loc	1 71 20                         ; triton_cdna4_matmul_async_copy_source:71:20
	v_add_u32_e32 v4, s20, v43
	ds_read_b128 v[4:7], v4
	.loc	1 73 32                         ; triton_cdna4_matmul_async_copy_source:73:32
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[0:3], v[4:7], a[0:15]
.LBB0_47:
	.loc	1 60 44                         ; triton_cdna4_matmul_async_copy_source:60:44
	s_waitcnt lgkmcnt(4)
	v_lshrrev_b32_e32 v1, 2, v40
	s_waitcnt lgkmcnt(0)
	v_lshrrev_b32_e32 v2, 1, v41
	.loc	1 61 31                         ; triton_cdna4_matmul_async_copy_source:61:31
	v_or3_b32 v8, v1, v2, s22
	.loc	1 60 44                         ; triton_cdna4_matmul_async_copy_source:60:44
	v_lshrrev_b32_e32 v1, 2, v39
	s_nop 5
	v_accvgpr_read_b32 v0, a0
	.loc	1 60 31 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:60:31
	v_or3_b32 v9, v38, v1, s8
	.loc	1 77 23 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:77:23
	v_accvgpr_read_b32 v1, a1
	v_cvt_pk_f16_f32 v0, v0, v1
	v_accvgpr_read_b32 v1, a3
	v_accvgpr_read_b32 v2, a2
	v_cvt_pk_f16_f32 v1, v2, v1
	v_accvgpr_read_b32 v2, a5
	v_accvgpr_read_b32 v3, a4
	v_cvt_pk_f16_f32 v2, v3, v2
	v_accvgpr_read_b32 v3, a7
	v_accvgpr_read_b32 v4, a6
	v_cvt_pk_f16_f32 v3, v4, v3
	v_accvgpr_read_b32 v4, a9
	v_accvgpr_read_b32 v5, a8
	v_cvt_pk_f16_f32 v4, v5, v4
	v_accvgpr_read_b32 v5, a11
	v_accvgpr_read_b32 v6, a10
	v_cvt_pk_f16_f32 v5, v6, v5
	v_accvgpr_read_b32 v6, a13
	v_accvgpr_read_b32 v7, a12
	v_cvt_pk_f16_f32 v6, v7, v6
	v_accvgpr_read_b32 v7, a15
	v_accvgpr_read_b32 v10, a14
	v_cvt_pk_f16_f32 v7, v10, v7
	.loc	1 78 39                         ; triton_cdna4_matmul_async_copy_source:78:39
	v_mul_lo_u32 v10, v9, s9
	.loc	1 78 21 is_stmt 0               ; triton_cdna4_matmul_async_copy_source:78:21
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
	.loc	1 78 43                         ; triton_cdna4_matmul_async_copy_source:78:43
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 1, v[10:11]
	.loc	1 77 23 is_stmt 1               ; triton_cdna4_matmul_async_copy_source:77:23
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	v_permlane32_swap_b32_e32 v4, v6
	v_permlane32_swap_b32_e32 v5, v7
	.loc	1 79 21                         ; triton_cdna4_matmul_async_copy_source:79:21
	global_store_dwordx4 v[8:9], v[0:3], off
	global_store_dwordx4 v[8:9], v[4:7], off offset:32
	.loc	1 79 4 is_stmt 0                ; triton_cdna4_matmul_async_copy_source:79:4
	s_endpgm
.Ltmp15:
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 84
		.amdhsa_next_free_sgpr 32
		.amdhsa_accum_offset 68
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
	.set triton_cdna4_matmul_kernel.num_vgpr, 66
	.set triton_cdna4_matmul_kernel.num_agpr, 16
	.set triton_cdna4_matmul_kernel.numbered_sgpr, 32
	.set triton_cdna4_matmul_kernel.num_named_barrier, 0
	.set triton_cdna4_matmul_kernel.private_seg_size, 0
	.set triton_cdna4_matmul_kernel.uses_vcc, 1
	.set triton_cdna4_matmul_kernel.uses_flat_scratch, 0
	.set triton_cdna4_matmul_kernel.has_dyn_sized_stack, 0
	.set triton_cdna4_matmul_kernel.has_recursion, 0
	.set triton_cdna4_matmul_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4944
; TotalNumSgprs: 38
; NumVgprs: 66
; NumAgprs: 16
; TotalNumVgprs: 84
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 84
; AccumOffset: 68
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 16
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
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x64 DW_TAG_compile_unit
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
	.byte	3                               ; Abbrev [3] 0x30:0x3e DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	51                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x55:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	50                              ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x61:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	70                              ; DW_AT_call_line
	.byte	36                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"triton_cdna4_matmul_async_copy_source" ; string offset=7
.Linfo_string2:
	.asciz	"tests/kernels" ; string offset=37
.Linfo_string3:
	.asciz	"triton_cdna4_matmul_kernel"    ; string offset=131
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
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         triton_cdna4_matmul_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     84
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
