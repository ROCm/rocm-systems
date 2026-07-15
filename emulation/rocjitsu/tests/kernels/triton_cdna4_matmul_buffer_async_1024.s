// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Reproduction information for this checked-in assembly fixture.
// The original source used to generate this assembly is embedded here
// as comments so normal builds do not need Python, Torch, or Triton.
//
// --- begin tests/kernels/triton/matmul_async_dma.py ---
// # Copyright (c) 2026 Advanced Micro Devices, Inc.
// # SPDX-License-Identifier: MIT
//
// """Triton matmul sources for DBT buffer-load-to-LDS stress fixtures.
//
// This file intentionally contains source kernels only.  The tests that consume
// these kernels can choose whether to JIT them at runtime or compile them offline
// to assembly/hsaco fixtures.
//
// The kernel is an ordinary fp16-input matmul shaped to give the AMD Triton
// pipeline enough regular LDS traffic to choose a buffer-load-to-LDS lowering.
// The instruction spelling is still a compiler outcome, not a source-level
// guarantee; use compile_fixtures.py to verify what a particular Triton build
// emits.
//
// Compile for gfx950/CDNA4 with `TRITON_HIP_USE_ASYNC_COPY=1` and
// `AMDGCN_USE_BUFFER_OPS=1`. Buffer lowering also needs the base pointers to be
// eligible for 32-bit buffer offsets; runtime JIT normally supplies that through
// `tt.pointer_range=32` for tensors smaller than 2 GiB, while offline compilation
// may need equivalent pointer attributes or small static/range-analysis bounds.
// That is the backend configuration intended to select
// `buffer_load_lds_dwordx4`.
// """
//
// # Recorded fixture metadata:
// #   kernel: matmul_async_buffer_load_lds
// #   target: gfx950, wavefront_size: 64
// #   launch sharedMemBytes used by cdna4_to_cdna3_dispatch_test: 65536
// #   amdhsa_group_segment_fixed_size: 0
// #   max_flat_workgroup_size: 256
// #   kernarg_segment_size: 40
// #   registers: vgpr_count=68, agpr_count=16, sgpr_count=32
//
// import triton
// import triton.language as tl
//
//
// # Environment knobs the fixture compiler should set around `triton.compile`.
// TARGET_ARCH = "gfx950"
// DEFAULT_M = 1024
// DEFAULT_N = 1024
// DEFAULT_K = 1024
// COMMON_HIP_OPTIONS = {
//     "num_warps": 4,
//     "num_stages": 3,
//     # Keep this explicit for fixture generation.  It is a backend compile
//     # option, not a Triton kernel meta-parameter.
//     "matrix_instr_nonkdim": 16,
//     "waves_per_eu": 0,
// }
// BUFFER_LOAD_LDS_ENV = {
//     "TRITON_HIP_USE_ASYNC_COPY": "1",
//     "AMDGCN_USE_BUFFER_OPS": "1",
// }
//
//
// # Suggested compile-time meta/options.  BLOCK_M, BLOCK_N, and BLOCK_K are kept
// # at 64 so the generated dynamic LDS footprint fits gfx942's 64 KiB per-workgroup
// # limit while still giving the AMD backend enough contiguous data to select wide
// # buffer-load-to-LDS traffic.  NUM_STAGES must be at least 2 because the AMD
// # scheduler only pipelines loops with multiple stages.
// BUFFER_LOAD_LDS_CONFIG = triton.Config(
//     {
//         "BLOCK_M": 64,
//         "BLOCK_N": 64,
//         "BLOCK_K": 64,
//         "GROUP_M": 4,
//         "NUM_STAGES": 3,
//     },
//     num_warps=4,
//     num_stages=3,
// )
//
//
// @triton.jit
// def _program_ids(
//     M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr
// ):
//     """Group M programs together so neighboring CTAs reuse the same B tile."""
//     pid = tl.program_id(0)
//     num_pid_m = tl.cdiv(M, BLOCK_M)
//     num_pid_n = tl.cdiv(N, BLOCK_N)
//     group_size = GROUP_M * num_pid_n
//     group_id = pid // group_size
//     first_pid_m = group_id * GROUP_M
//     group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
//     pid_m = first_pid_m + ((pid % group_size) % group_m)
//     pid_n = (pid % group_size) // group_m
//     return pid_m, pid_n
//
//
// @triton.jit
// def matmul_async_buffer_load_lds(
//     a,
//     b,
//     c,
//     M: tl.constexpr,
//     N: tl.constexpr,
//     K: tl.constexpr,
//     BLOCK_M: tl.constexpr,
//     BLOCK_N: tl.constexpr,
//     BLOCK_K: tl.constexpr,
//     GROUP_M: tl.constexpr,
//     NUM_STAGES: tl.constexpr,
// ):
//     """C[M, N] = A[M, K] x B[K, N], compiled with AMD buffer-op lowering."""
//     tl.static_assert(BLOCK_M >= 64, "BLOCK_M must be at least 64")
//     tl.static_assert(BLOCK_N >= 64, "BLOCK_N must be at least 64")
//     tl.static_assert(BLOCK_K >= 64, "BLOCK_K must be at least 64")
//     tl.static_assert(NUM_STAGES >= 2, "async load-to-LDS needs a pipelined loop")
//
//     # The source uses contiguous row-major A, B, and C.  Contiguous fp16 runs
//     # along K for A and along N for B give the AMD backend the vector shape it
//     # needs to coalesce eligible async copies into buffer-load-to-LDS ops.
//     pid_m, pid_n = _program_ids(M, N, BLOCK_M, BLOCK_N, GROUP_M)
//     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
//     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
//     offs_k = tl.arange(0, BLOCK_K)
//     load_m = tl.where(offs_m < M, offs_m, 0)
//     load_n = tl.where(offs_n < N, offs_n, 0)
//     load_m = tl.max_contiguous(tl.multiple_of(load_m, BLOCK_M), BLOCK_M)
//     load_n = tl.max_contiguous(tl.multiple_of(load_n, BLOCK_N), BLOCK_N)
//
//     acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
//     for k0 in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
//         k_idxs = k0 + offs_k
//         a_tile = tl.load(
//             a + load_m[:, None] * K + k_idxs[None, :],
//             mask=k_idxs[None, :] < K,
//             other=0.0,
//         )
//         b_tile = tl.load(
//             b + k_idxs[:, None] * N + load_n[None, :],
//             mask=k_idxs[:, None] < K,
//             other=0.0,
//         )
//         acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
//
//     tl.store(
//         c + offs_m[:, None] * N + offs_n[None, :],
//         acc,
//         mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
//     )
//
//
// __all__ = [
//     "BUFFER_LOAD_LDS_ENV",
//     "BUFFER_LOAD_LDS_CONFIG",
//     "COMMON_HIP_OPTIONS",
//     "DEFAULT_K",
//     "DEFAULT_M",
//     "DEFAULT_N",
//     "TARGET_ARCH",
//     "matmul_async_buffer_load_lds",
// ]
// --- end tests/kernels/triton/matmul_async_dma.py ---

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_async_buffer_load_lds    ; -- Begin function matmul_async_buffer_load_lds
	.p2align	8
	.type	matmul_async_buffer_load_lds,@function
matmul_async_buffer_load_lds:           ; @matmul_async_buffer_load_lds
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.1:
	.file	1 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/tests/kernels/triton" "matmul_async_dma.py"
	.loc	1 84 0 prologue_end             ; matmul_async_dma.py:84:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.2:
.LBB0_0:
	s_mov_b64 s[8:9], s[2:3]
.Ltmp1:
	.loc	1 75 22 is_stmt 1               ; matmul_async_dma.py:75:22 @[ matmul_async_dma.py:106:56 ]
	s_ashr_i32 s2, s12, 31
	s_lshr_b32 s2, s2, 26
	s_add_i32 s2, s12, s2
	s_ashr_i32 s3, s2, 6
	.loc	1 76 29                         ; matmul_async_dma.py:76:29 @[ matmul_async_dma.py:106:56 ]
	s_lshl_b32 s3, s3, 2
	s_mov_b64 s[0:1], s[6:7]
	.loc	1 77 37                         ; matmul_async_dma.py:77:37 @[ matmul_async_dma.py:106:56 ]
	s_sub_i32 s6, 16, s3
	.loc	1 77 50 is_stmt 0               ; matmul_async_dma.py:77:50 @[ matmul_async_dma.py:106:56 ]
	s_min_i32 s6, s6, 4
	.loc	1 79 34 is_stmt 1               ; matmul_async_dma.py:79:34 @[ matmul_async_dma.py:106:56 ]
	s_abs_i32 s7, s6
	v_cvt_f32_u32_e32 v1, s7
	.loc	1 78 34                         ; matmul_async_dma.py:78:34 @[ matmul_async_dma.py:106:56 ]
	s_andn2_b32 s2, s2, 63
	s_sub_i32 s2, s12, s2
	.loc	1 79 34                         ; matmul_async_dma.py:79:34 @[ matmul_async_dma.py:106:56 ]
	s_sub_i32 s12, 0, s7
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s11, s2
	s_xor_b32 s10, s2, s6
	s_ashr_i32 s10, s10, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
.Ltmp2:
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_lshlrev_b32_e32 v9, 3, v0
	v_lshrrev_b32_e32 v5, 3, v0
	v_and_b32_e32 v3, 56, v9
.Ltmp3:
	.loc	1 79 34                         ; matmul_async_dma.py:79:34 @[ matmul_async_dma.py:106:56 ]
	v_readfirstlane_b32 s13, v1
	s_mul_i32 s12, s12, s13
	s_mul_hi_u32 s12, s13, s12
	s_add_i32 s13, s13, s12
	s_mul_hi_u32 s12, s11, s13
	s_mul_i32 s13, s12, s7
	s_sub_i32 s11, s11, s13
	s_add_i32 s13, s12, 1
	s_sub_i32 s15, s11, s7
	s_cmp_ge_u32 s11, s7
	s_cselect_b32 s12, s13, s12
	s_cselect_b32 s11, s15, s11
	s_add_i32 s13, s12, 1
	s_cmp_ge_u32 s11, s7
	s_cselect_b32 s7, s13, s12
	s_xor_b32 s7, s7, s10
	s_sub_i32 s7, s7, s10
	.loc	1 78 48                         ; matmul_async_dma.py:78:48 @[ matmul_async_dma.py:106:56 ]
	s_mul_i32 s6, s7, s6
	s_sub_i32 s2, s2, s6
	.loc	1 78 27 is_stmt 0               ; matmul_async_dma.py:78:27 @[ matmul_async_dma.py:106:56 ]
	s_add_i32 s2, s2, s3
.Ltmp4:
	.loc	1 108 21 is_stmt 1              ; matmul_async_dma.py:108:21
	s_lshl_b32 s13, s7, 6
	.loc	1 107 21                        ; matmul_async_dma.py:107:21
	s_lshl_b32 s3, s2, 6
	.loc	1 107 44 is_stmt 0              ; matmul_async_dma.py:107:44
	v_or_b32_e32 v7, 32, v5
	.loc	1 108 31 is_stmt 1              ; matmul_async_dma.py:108:31
	v_or_b32_e32 v8, s13, v3
	s_movk_i32 s12, 0x400
	.loc	1 107 31                        ; matmul_async_dma.py:107:31
	v_or_b32_e32 v4, s3, v5
	v_or_b32_e32 v6, s3, v7
	.loc	1 111 31                        ; matmul_async_dma.py:111:31
	v_cmp_gt_i32_e32 vcc, s12, v8
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_lshlrev_b32_e32 v6, 10, v6
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_and_b32_e32 v1, 0x70, v0
	.loc	1 111 42                        ; matmul_async_dma.py:111:42
	v_cndmask_b32_e32 v10, 0, v8, vcc
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_lshlrev_b32_e32 v8, 10, v4
	.loc	1 110 31                        ; matmul_async_dma.py:110:31
	v_cmp_gt_i32_e32 vcc, s12, v4
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_and_b32_e32 v2, 63, v0
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	v_readfirstlane_b32 s14, v0
	.loc	1 110 42                        ; matmul_async_dma.py:110:42
	v_cndmask_b32_e32 v4, 0, v8, vcc
	v_cndmask_b32_e32 v6, 0, v6, vcc
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v11, v4, v3
	v_or_b32_e32 v8, v6, v3
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	v_lshrrev_b32_e32 v3, 1, v1
	v_xor_b32_e32 v3, v9, v3
	v_sub_u32_e32 v3, v3, v9
	v_ashrrev_i32_e32 v3, 3, v3
	v_add_u32_e32 v2, v3, v2
	v_lshlrev_b32_e32 v4, 2, v2
	ds_bpermute_b32 v6, v4, v11
	ds_bpermute_b32 v12, v4, v8
	s_lshl_b32 s6, s14, 4
	v_lshrrev_b64 v[2:3], v2, exec
	s_and_b32 s6, s6, 0xc00
	v_and_b32_e32 v3, 1, v2
	s_add_i32 s17, s6, 0
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v6, 1, v6
	v_bfrev_b32_e32 v2, 1
	v_cmp_eq_u32_e32 vcc, 1, v3
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	v_cndmask_b32_e32 v3, v2, v6, vcc
	s_mov_b32 m0, s17
	s_add_i32 s14, s17, 0x1000
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v12
	v_cndmask_b32_e32 v3, v2, v3, vcc
	s_mov_b32 m0, s14
	.loc	1 124 38 is_stmt 1              ; matmul_async_dma.py:124:38
	v_lshl_add_u32 v6, v5, 10, v10
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v3, v4, v6
	.loc	1 124 38 is_stmt 0              ; matmul_async_dma.py:124:38
	v_lshl_add_u32 v5, v7, 10, v10
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v7, v4, v5
	s_add_i32 s15, s17, 0x6000
	s_and_b32 s5, s5, 0xffff
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v3, 1, v3
	s_mov_b32 s6, s10
	s_mov_b32 s7, s11
	v_cndmask_b32_e32 v3, v2, v3, vcc
	s_mov_b32 m0, s15
	s_add_i32 s16, s17, 0x7000
	buffer_load_dwordx4 v3, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v7
	v_cndmask_b32_e32 v3, v2, v3, vcc
	s_mov_b32 m0, s16
	.loc	1 119 38 is_stmt 1              ; matmul_async_dma.py:119:38
	v_or_b32_e32 v7, 64, v8
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v3, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v3, 64, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v3, v4, v3
	ds_bpermute_b32 v7, v4, v7
	s_add_i32 s19, s17, 0x2000
	s_mov_b32 m0, s19
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v3
	v_cndmask_b32_e32 v3, v2, v3, vcc
	s_barrier
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	s_add_i32 s20, s17, 0x3000
	v_lshlrev_b32_e32 v3, 1, v7
	v_cndmask_b32_e32 v3, v2, v3, vcc
	.loc	1 124 38 is_stmt 1              ; matmul_async_dma.py:124:38
	v_add_u32_e32 v7, 0x10000, v6
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s20
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v7, v4, v7
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v3, 0x10000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v3, v4, v3
	s_add_i32 s21, s17, 0x8000
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v7, 1, v7
	v_cndmask_b32_e32 v7, v2, v7, vcc
	s_mov_b32 m0, s21
	s_add_i32 s22, s17, 0x9000
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v3
	buffer_load_dwordx4 v7, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v3, v2, v3, vcc
	s_mov_b32 m0, s22
	v_lshrrev_b32_e32 v7, 1, v0
	buffer_load_dwordx4 v3, s[4:7], 0 offen lds
	v_lshlrev_b32_e32 v3, 6, v0
	s_movk_i32 s18, 0x820
	.loc	1 119 38 is_stmt 1              ; matmul_async_dma.py:119:38
	v_or_b32_e32 v13, 0x80, v11
	v_bitop3_b32 v3, v3, s18, v7 bitop3:0xc8
	v_lshlrev_b32_e32 v7, 5, v0
	v_bfe_i32 v10, v0, 4, 1
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v13, v4, v13
	v_and_b32_e32 v7, 0x80, v7
	v_and_b32_e32 v10, 0x440, v10
	v_or3_b32 v12, v3, v7, v10
	v_bfe_i32 v3, v0, 3, 1
	v_and_b32_e32 v7, 24, v9
	s_movk_i32 s18, 0x110
	v_bitop3_b32 v15, v3, v7, s18 bitop3:0x6c
	.loc	1 107 44 is_stmt 1              ; matmul_async_dma.py:107:44
	v_and_b32_e32 v3, 15, v0
	v_and_b32_e32 v7, 0x70, v9
	v_and_b32_e32 v9, 48, v0
	v_and_b32_e32 v0, 0x80, v0
	v_lshl_or_b32 v7, v3, 7, v7
	v_lshlrev_b32_e32 v14, 4, v0
	v_or_b32_e32 v10, v12, v15
	v_bitop3_b32 v7, v7, v14, v9 bitop3:0xde
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v9, 0x80, v8
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v13, 1, v13
	s_movk_i32 s23, 0x60
	v_cndmask_b32_e32 v32, v2, v13, vcc
	ds_bpermute_b32 v33, v4, v9
	v_add_u32_e32 v9, 0, v7
	v_xad_u32 v7, v7, 64, 0
	.loc	1 124 12 is_stmt 1              ; matmul_async_dma.py:124:12
	v_add_u32_e32 v13, 0, v10
	v_xad_u32 v14, v10, 32, 0
	v_xad_u32 v10, v10, 64, 0
	v_bitop3_b32 v12, v12, s23, v15 bitop3:0x36
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[16:19], v9
	ds_read_b128 v[20:23], v9 offset:4096
	ds_read_b128 v[24:27], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:24576
	ds_read_b64_tr_b16 v[36:37], v14 offset:25088
	ds_read_b64_tr_b16 v[38:39], v13 offset:28672
	ds_read_b64_tr_b16 v[40:41], v14 offset:29184
	v_add_u32_e32 v12, 0, v12
	ds_read_b64_tr_b16 v[42:43], v10 offset:24576
	ds_read_b64_tr_b16 v[44:45], v12 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_add_i32 s18, s17, 0x4000
	s_mov_b32 m0, s18
	s_add_i32 s23, s17, 0x5000
	v_lshlrev_b32_e32 v15, 1, v33
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], 0
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:28672
	ds_read_b64_tr_b16 v[48:49], v12 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v32, s[8:11], 0 offen lds
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], 0
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x20000, v6
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s23
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x20000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_add_i32 s24, s17, 0xa000
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	s_mov_b32 m0, s24
	s_add_i32 s25, s17, 0xb000
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s25
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], 0
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0xc0, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0xc0, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:32768
	ds_read_b64_tr_b16 v[36:37], v14 offset:33280
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s17
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_lshrrev_b32_e32 v0, 3, v0
	v_lshrrev_b32_e32 v1, 2, v1
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], 0
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:36864
	ds_read_b64_tr_b16 v[40:41], v14 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:8192
	ds_read_b128 v[20:23], v9 offset:12288
	ds_read_b128 v[24:27], v7 offset:8192
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:32768
	ds_read_b64_tr_b16 v[44:45], v12 offset:33280
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_or_b32_e32 v0, v0, v3
	.loc	1 107 31 is_stmt 0              ; matmul_async_dma.py:107:31
	v_or_b32_e32 v3, s3, v0
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:12288
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:36864
	ds_read_b64_tr_b16 v[48:49], v12 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s14
	.loc	1 131 12                        ; matmul_async_dma.py:131:12
	s_lshl_b32 s2, s2, 16
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x30000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x30000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s15
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x100, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x100, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:40960
	ds_read_b64_tr_b16 v[36:37], v14 offset:41472
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s19
	.loc	1 131 34                        ; matmul_async_dma.py:131:34
	v_lshlrev_b32_e32 v0, 10, v0
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	s_and_b32 s1, s1, 0xffff
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:45056
	ds_read_b64_tr_b16 v[40:41], v14 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:16384
	ds_read_b128 v[20:23], v9 offset:20480
	ds_read_b128 v[24:27], v7 offset:16384
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:40960
	ds_read_b64_tr_b16 v[44:45], v12 offset:41472
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	s_mov_b32 s3, s11
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:20480
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:45056
	ds_read_b64_tr_b16 v[48:49], v12 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s20
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x40000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x40000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s21
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s22
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x140, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x140, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:24576
	ds_read_b64_tr_b16 v[36:37], v14 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s18
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:28672
	ds_read_b64_tr_b16 v[40:41], v14 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9
	ds_read_b128 v[20:23], v9 offset:4096
	ds_read_b128 v[24:27], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:24576
	ds_read_b64_tr_b16 v[44:45], v12 offset:25088
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:28672
	ds_read_b64_tr_b16 v[48:49], v12 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s23
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x50000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x50000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s24
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s25
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x180, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x180, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:32768
	ds_read_b64_tr_b16 v[36:37], v14 offset:33280
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s17
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:36864
	ds_read_b64_tr_b16 v[40:41], v14 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:8192
	ds_read_b128 v[20:23], v9 offset:12288
	ds_read_b128 v[24:27], v7 offset:8192
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:32768
	ds_read_b64_tr_b16 v[44:45], v12 offset:33280
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:12288
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:36864
	ds_read_b64_tr_b16 v[48:49], v12 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s14
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x60000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x60000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s15
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x1c0, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x1c0, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:40960
	ds_read_b64_tr_b16 v[36:37], v14 offset:41472
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s19
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:45056
	ds_read_b64_tr_b16 v[40:41], v14 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:16384
	ds_read_b128 v[20:23], v9 offset:20480
	ds_read_b128 v[24:27], v7 offset:16384
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:40960
	ds_read_b64_tr_b16 v[44:45], v12 offset:41472
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:20480
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:45056
	ds_read_b64_tr_b16 v[48:49], v12 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s20
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x70000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x70000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s21
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s22
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x200, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x200, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:24576
	ds_read_b64_tr_b16 v[36:37], v14 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s18
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:28672
	ds_read_b64_tr_b16 v[40:41], v14 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9
	ds_read_b128 v[20:23], v9 offset:4096
	ds_read_b128 v[24:27], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:24576
	ds_read_b64_tr_b16 v[44:45], v12 offset:25088
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:28672
	ds_read_b64_tr_b16 v[48:49], v12 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s23
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x80000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x80000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s24
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s25
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x240, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x240, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:32768
	ds_read_b64_tr_b16 v[36:37], v14 offset:33280
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s17
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:36864
	ds_read_b64_tr_b16 v[40:41], v14 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:8192
	ds_read_b128 v[20:23], v9 offset:12288
	ds_read_b128 v[24:27], v7 offset:8192
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:32768
	ds_read_b64_tr_b16 v[44:45], v12 offset:33280
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:12288
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:36864
	ds_read_b64_tr_b16 v[48:49], v12 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s14
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0x90000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0x90000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s15
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x280, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x280, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:40960
	ds_read_b64_tr_b16 v[36:37], v14 offset:41472
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s19
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:45056
	ds_read_b64_tr_b16 v[40:41], v14 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:16384
	ds_read_b128 v[20:23], v9 offset:20480
	ds_read_b128 v[24:27], v7 offset:16384
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:40960
	ds_read_b64_tr_b16 v[44:45], v12 offset:41472
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:20480
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:45056
	ds_read_b64_tr_b16 v[48:49], v12 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s20
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0xa0000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0xa0000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s21
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s22
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x2c0, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x2c0, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:24576
	ds_read_b64_tr_b16 v[36:37], v14 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s18
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:28672
	ds_read_b64_tr_b16 v[40:41], v14 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9
	ds_read_b128 v[20:23], v9 offset:4096
	ds_read_b128 v[24:27], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:24576
	ds_read_b64_tr_b16 v[44:45], v12 offset:25088
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:28672
	ds_read_b64_tr_b16 v[48:49], v12 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s23
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0xb0000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0xb0000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s24
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s25
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x300, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x300, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:32768
	ds_read_b64_tr_b16 v[36:37], v14 offset:33280
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s17
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:36864
	ds_read_b64_tr_b16 v[40:41], v14 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:8192
	ds_read_b128 v[20:23], v9 offset:12288
	ds_read_b128 v[24:27], v7 offset:8192
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:32768
	ds_read_b64_tr_b16 v[44:45], v12 offset:33280
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:12288
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:36864
	ds_read_b64_tr_b16 v[48:49], v12 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s14
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0xc0000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0xc0000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s15
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x340, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x340, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:40960
	ds_read_b64_tr_b16 v[36:37], v14 offset:41472
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s19
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:45056
	ds_read_b64_tr_b16 v[40:41], v14 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:16384
	ds_read_b128 v[20:23], v9 offset:20480
	ds_read_b128 v[24:27], v7 offset:16384
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:40960
	ds_read_b64_tr_b16 v[44:45], v12 offset:41472
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:20480
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:45056
	ds_read_b64_tr_b16 v[48:49], v12 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s20
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0xd0000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0xd0000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	s_mov_b32 m0, s21
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v16, 1, v16
	v_cndmask_b32_e32 v16, v2, v16, vcc
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s22
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v15, 0x380, v11
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v16, 0x380, v8
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v32, v4, v16
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[34:35], v13 offset:24576
	ds_read_b64_tr_b16 v[36:37], v14 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_lshlrev_b32_e32 v15, 1, v15
	v_cndmask_b32_e32 v15, v2, v15, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s18
	.loc	1 119 38 is_stmt 0              ; matmul_async_dma.py:119:38
	v_or_b32_e32 v11, 0x3c0, v11
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v11, v4, v11
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[38:39], v13 offset:28672
	ds_read_b64_tr_b16 v[40:41], v14 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9
	ds_read_b128 v[20:23], v9 offset:4096
	ds_read_b128 v[24:27], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[42:43], v10 offset:24576
	ds_read_b64_tr_b16 v[44:45], v12 offset:25088
	.loc	1 119 38                        ; matmul_async_dma.py:119:38
	v_or_b32_e32 v8, 0x3c0, v8
	.loc	1 119 12 is_stmt 0              ; matmul_async_dma.py:119:12
	ds_bpermute_b32 v8, v4, v8
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[46:47], v10 offset:28672
	ds_read_b64_tr_b16 v[48:49], v12 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	v_lshlrev_b32_e32 v15, 1, v32
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_f16 a[0:3], v[34:37], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s23
	v_lshlrev_b32_e32 v11, 1, v11
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_f16 a[4:7], v[42:45], v[16:19], a[4:7]
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v16, 0xe0000, v6
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v16, v4, v16
	.loc	1 119 12 is_stmt 1              ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v15, s[8:11], 0 offen lds
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v15, 0xe0000, v5
	.loc	1 124 12 is_stmt 0              ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v15, v4, v15
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v6, 0xf0000, v6
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_bpermute_b32 v6, v4, v6
	.loc	1 124 38                        ; matmul_async_dma.py:124:38
	v_add_u32_e32 v5, 0xf0000, v5
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v16, 1, v16
	ds_bpermute_b32 v4, v4, v5
	.loc	1 128 30 is_stmt 1              ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[34:37], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	v_cndmask_b32_e32 v16, v2, v16, vcc
	s_mov_b32 m0, s24
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b32_e32 v15, 1, v15
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[42:45], v[20:23], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v16, s[4:7], 0 offen lds
	v_cndmask_b32_e32 v15, v2, v15, vcc
	s_mov_b32 m0, s25
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	v_cndmask_b32_e32 v11, v2, v11, vcc
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v15, s[4:7], 0 offen lds
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_mov_b32 m0, s17
	v_lshlrev_b32_e32 v8, 1, v8
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v11, s[8:11], 0 offen lds
	v_cndmask_b32_e32 v8, v2, v8, vcc
	s_mov_b32 m0, s14
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	v_lshlrev_b32_e32 v6, 1, v6
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[38:41], v[24:27], a[0:3]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[32:33], v13 offset:32768
	ds_read_b64_tr_b16 v[34:35], v14 offset:33280
	v_cndmask_b32_e32 v5, v2, v6, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[46:49], v[24:27], a[4:7]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	v_cndmask_b32_e32 v4, v2, v4, vcc
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[38:41], v[28:31], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[36:37], v13 offset:36864
	ds_read_b64_tr_b16 v[38:39], v14 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:8192
	ds_read_b128 v[20:23], v9 offset:12288
	ds_read_b128 v[24:27], v7 offset:8192
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[40:41], v10 offset:32768
	ds_read_b64_tr_b16 v[42:43], v12 offset:33280
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[46:49], v[28:31], a[12:15]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:12288
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[44:45], v10 offset:36864
	ds_read_b64_tr_b16 v[46:47], v12 offset:37376
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	buffer_load_dwordx4 v8, s[8:11], 0 offen lds
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	s_mov_b32 m0, s15
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[32:35], v[16:19], a[0:3]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v5, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	.loc	1 108 31                        ; matmul_async_dma.py:108:31
	v_or_b32_e32 v8, s13, v1
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	buffer_load_dwordx4 v4, s[4:7], 0 offen lds
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[40:43], v[16:19], a[4:7]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[8:11], v[32:35], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[32:33], v13 offset:40960
	ds_read_b64_tr_b16 v[34:35], v14 offset:41472
	.loc	1 133 38                        ; matmul_async_dma.py:133:38
	v_max_i32_e32 v3, v3, v8
	.loc	1 131 34                        ; matmul_async_dma.py:131:34
	s_add_i32 s4, s2, s13
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[40:43], v[20:23], a[12:15]
	.loc	1 133 38                        ; matmul_async_dma.py:133:38
	v_cmp_gt_i32_e32 vcc, s12, v3
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	s_mov_b32 s2, s10
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[36:39], v[24:27], a[0:3]
	v_mfma_f32_16x16x32_f16 a[4:7], v[44:47], v[24:27], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], v[36:39], v[28:31], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[36:37], v13 offset:45056
	ds_read_b64_tr_b16 v[38:39], v14 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[16:19], v9 offset:16384
	ds_read_b128 v[20:23], v9 offset:20480
	ds_read_b128 v[24:27], v7 offset:16384
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[44:47], v[28:31], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[40:41], v10 offset:40960
	ds_read_b64_tr_b16 v[42:43], v12 offset:41472
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[28:31], v7 offset:20480
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[44:45], v10 offset:45056
	ds_read_b64_tr_b16 v[46:47], v12 offset:45568
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt vmcnt(0) lgkmcnt(7)
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[0:3], v[32:35], v[16:19], a[0:3]
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[4:7], v[40:43], v[16:19], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], v[32:35], v[20:23], a[8:11]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[32:33], v13 offset:24576
	ds_read_b64_tr_b16 v[34:35], v14 offset:25088
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	v_mfma_f32_16x16x32_f16 a[12:15], v[40:43], v[20:23], a[12:15]
	v_mfma_f32_16x16x32_f16 a[0:3], v[36:39], v[24:27], a[0:3]
	v_mfma_f32_16x16x32_f16 a[4:7], v[44:47], v[24:27], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], v[36:39], v[28:31], a[8:11]
	v_mfma_f32_16x16x32_f16 a[12:15], v[44:47], v[28:31], a[12:15]
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[26:27], v13 offset:28672
	ds_read_b64_tr_b16 v[28:29], v14 offset:29184
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[14:17], v9
	ds_read_b128 v[18:21], v9 offset:4096
	ds_read_b128 v[22:25], v7
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[36:37], v10 offset:24576
	ds_read_b64_tr_b16 v[38:39], v12 offset:25088
	.loc	1 119 12                        ; matmul_async_dma.py:119:12
	ds_read_b128 v[4:7], v7 offset:4096
	.loc	1 124 12                        ; matmul_async_dma.py:124:12
	ds_read_b64_tr_b16 v[10:11], v10 offset:28672
	ds_read_b64_tr_b16 v[12:13], v12 offset:29184
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_f16 a[0:3], v[32:35], v[14:17], a[0:3]
	.loc	1 107 44                        ; matmul_async_dma.py:107:44
	v_or_b32_e32 v9, 32, v1
	.loc	1 128 30                        ; matmul_async_dma.py:128:30
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_f16 a[4:7], v[36:39], v[14:17], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], v[32:35], v[18:21], a[8:11]
	v_mfma_f32_16x16x32_f16 a[12:15], v[36:39], v[18:21], a[12:15]
	v_mfma_f32_16x16x32_f16 a[0:3], v[26:29], v[22:25], a[0:3]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_f16 a[4:7], v[10:13], v[22:25], a[4:7]
	v_mfma_f32_16x16x32_f16 a[8:11], v[26:29], v[4:7], a[8:11]
	v_mfma_f32_16x16x32_f16 a[12:15], v[10:13], v[4:7], a[12:15]
	.loc	1 131 34                        ; matmul_async_dma.py:131:34
	v_or_b32_e32 v4, 0x8000, v0
	v_or_b32_e32 v5, v0, v1
	v_or_b32_e32 v0, v0, v9
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	v_add_lshl_u32 v5, v5, s4, 2
	v_add_lshl_u32 v0, v0, s4, 2
	.loc	1 131 34                        ; matmul_async_dma.py:131:34
	v_or_b32_e32 v1, v4, v1
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	v_cndmask_b32_e32 v3, v2, v5, vcc
	v_cndmask_b32_e32 v0, v2, v0, vcc
	buffer_store_dwordx4 a[0:3], v3, s[0:3], 0 offen
	buffer_store_dwordx4 a[4:7], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v1, s4, 2
	.loc	1 131 34                        ; matmul_async_dma.py:131:34
	v_or_b32_e32 v4, v4, v9
	.loc	1 132 8                         ; matmul_async_dma.py:132:8
	v_cndmask_b32_e32 v0, v2, v0, vcc
	buffer_store_dwordx4 a[8:11], v0, s[0:3], 0 offen
	v_add_lshl_u32 v0, v4, s4, 2
	v_cndmask_b32_e32 v0, v2, v0, vcc
	buffer_store_dwordx4 a[12:15], v0, s[0:3], 0 offen
	.loc	1 130 4                         ; matmul_async_dma.py:130:4
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
		.amdhsa_next_free_vgpr 68
		.amdhsa_next_free_sgpr 26
		.amdhsa_accum_offset 52
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
	.set matmul_async_buffer_load_lds.num_vgpr, 50
	.set matmul_async_buffer_load_lds.num_agpr, 16
	.set matmul_async_buffer_load_lds.numbered_sgpr, 26
	.set matmul_async_buffer_load_lds.num_named_barrier, 0
	.set matmul_async_buffer_load_lds.private_seg_size, 0
	.set matmul_async_buffer_load_lds.uses_vcc, 1
	.set matmul_async_buffer_load_lds.uses_flat_scratch, 0
	.set matmul_async_buffer_load_lds.has_dyn_sized_stack, 0
	.set matmul_async_buffer_load_lds.has_recursion, 0
	.set matmul_async_buffer_load_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6280
; TotalNumSgprs: 32
; NumVgprs: 50
; NumAgprs: 16
; TotalNumVgprs: 68
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 68
; AccumOffset: 52
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 12
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 12
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
	.byte	106                             ; DW_AT_call_line
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
	.asciz	"matmul_async_dma.py"           ; string offset=7
.Linfo_string2:
	.asciz	"/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/tests/kernels/triton" ; string offset=27
.Linfo_string3:
	.asciz	"matmul_async_buffer_load_lds"  ; string offset=143
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
    .name:           matmul_async_buffer_load_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         matmul_async_buffer_load_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     68
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
