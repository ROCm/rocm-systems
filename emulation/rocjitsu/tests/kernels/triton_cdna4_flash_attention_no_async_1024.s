// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Reproduction information for this checked-in assembly fixture.
// The original source used to generate this assembly is embedded here
// as comments so normal builds do not need Python, Torch, or Triton.
//
// --- begin tests/kernels/triton/flash_attention.py ---
// """Triton flash-attention kernels for DBT compile stress tests.
//
// This file is intentionally source-only: it does not register tests or change
// the kernel CMake flow.  The kernels below are small fp16 forward kernels for a
// single Q/K/V/O matrix pair with explicit row/feature strides.  They are meant to
// exercise instruction decoding/translation paths when compiled with AMD Triton,
// not to be a production attention implementation.
//
// The async/buffer variant uses AMD Triton backend options that are intended to
// select the desired lowering.  The final instruction spelling is compiler-build
// dependent, so verify generated assembly with compile_fixtures.py:
//
//   AMDGCN_USE_BUFFER_OPS=1
//       Run pointer canonicalization and conversion to AMDGPU buffer ops.
//
//   TRITON_HIP_USE_ASYNC_COPY=1
//       Prefer the async global-to-LDS copy pipeline.
//
// For best buffer-load vectorization, pass compact test tensors with feature
// strides of one for Q/K/V/O and 16-byte-aligned base pointers.
// """
//
// # Recorded no-async fixture metadata:
// #   kernel: flash_attention_fwd_no_async_kernel
// #   target: gfx950, wavefront_size: 64
// #   launch sharedMemBytes used by cdna4_to_cdna3_dispatch_test: 65536
// #   amdhsa_group_segment_fixed_size: 0
// #   max_flat_workgroup_size: 256
// #   kernarg_segment_size: 56
// #   registers: vgpr_count=132, agpr_count=16, sgpr_count=23
//
// import triton
// import triton.language as tl
//
//
// TARGET_ARCH = "gfx950"
// DEFAULT_Q_LEN = 1024
// DEFAULT_KV_LEN = 1024
// DEFAULT_HEAD_DIM = 64
// DEFAULT_BLOCK_M = 64
// DEFAULT_BLOCK_N = 64
// DEFAULT_BLOCK_D = 64
// DEFAULT_STRIDE_QM = DEFAULT_HEAD_DIM
// DEFAULT_STRIDE_QD = 1
// DEFAULT_STRIDE_KN = DEFAULT_HEAD_DIM
// DEFAULT_STRIDE_KD = 1
// DEFAULT_STRIDE_VN = DEFAULT_HEAD_DIM
// DEFAULT_STRIDE_VD = 1
// DEFAULT_STRIDE_OM = DEFAULT_HEAD_DIM
// DEFAULT_STRIDE_OD = 1
//
// NO_ASYNC_ENV = {
//     "TRITON_HIP_USE_ASYNC_COPY": "0",
//     "AMDGCN_USE_BUFFER_OPS": "0",
// }
// ASYNC_BUFFER_ENV = {
//     "TRITON_HIP_USE_ASYNC_COPY": "1",
//     "AMDGCN_USE_BUFFER_OPS": "1",
// }
// COMMON_HIP_OPTIONS = {
//     "num_warps": 4,
//     "num_stages": 2,
//     # The attention scheduler hint is local to the AMD backend and is useful
//     # for avoiding spills in the generated stress fixture.
//     "schedule_hint": "attention",
// }
//
//
// @triton.jit
// def flash_attention_fwd_no_async_kernel(
//     q_ptr,
//     k_ptr,
//     v_ptr,
//     o_ptr,
//     stride_qm: tl.constexpr,
//     stride_qd: tl.constexpr,
//     stride_kn: tl.constexpr,
//     stride_kd: tl.constexpr,
//     stride_vn: tl.constexpr,
//     stride_vd: tl.constexpr,
//     stride_om: tl.constexpr,
//     stride_od: tl.constexpr,
//     softmax_scale,
//     Q_LEN: tl.constexpr,
//     KV_LEN: tl.constexpr,
//     HEAD_DIM: tl.constexpr,
//     BLOCK_M: tl.constexpr,
//     BLOCK_N: tl.constexpr,
//     BLOCK_D: tl.constexpr,
// ):
//     """Plain blockwise flash attention.
//
//     Compile this variant with TRITON_HIP_USE_ASYNC_COPY=0 when the test should
//     avoid AMD's async global-to-LDS lowering.  The loop also uses one stage so
//     the source itself does not request Triton loop pipelining.
//     """
//
//     tl.static_assert(BLOCK_M >= 64)
//     tl.static_assert(BLOCK_N >= 64)
//     tl.static_assert(BLOCK_D >= 64)
//     tl.static_assert(BLOCK_D >= HEAD_DIM)
//
//     pid_m = tl.program_id(0)
//     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
//     offs_n = tl.arange(0, BLOCK_N)
//     offs_d = tl.arange(0, BLOCK_D)
//
//     q_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM)
//     q = tl.load(
//         q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
//         mask=q_mask,
//         other=0.0,
//     )
//
//     # The online softmax state follows the FlashAttention recurrence.  A large
//     # finite negative sentinel avoids inf - inf when the last program block has
//     # rows outside Q_LEN; those rows are masked on store.
//     m_i = tl.full((BLOCK_M,), -1.0e20, tl.float32)
//     l_i = tl.zeros((BLOCK_M,), tl.float32)
//     acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
//
//     # exp2 is the path used by Triton attention examples; convert the caller's
//     # natural-exp scale once so the recurrence remains mathematically standard.
//     qk_scale = softmax_scale * 1.4426950408889634
//
//     for start_n in tl.range(0, KV_LEN, BLOCK_N, num_stages=1):
//         start_n = tl.multiple_of(start_n, BLOCK_N)
//         n = start_n + offs_n
//
//         k_mask = (n[None, :] < KV_LEN) & (offs_d[:, None] < HEAD_DIM)
//         k = tl.load(
//             k_ptr + n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
//             mask=k_mask,
//             other=0.0,
//         )
//
//         qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale
//         qk = tl.where((offs_m[:, None] < Q_LEN) & (n[None, :] < KV_LEN), qk, -1.0e20)
//
//         m_new = tl.maximum(m_i, tl.max(qk, axis=1))
//         alpha = tl.math.exp2(m_i - m_new)
//         p = tl.math.exp2(qk - m_new[:, None])
//         l_new = l_i * alpha + tl.sum(p, axis=1)
//
//         v_mask = (n[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM)
//         v = tl.load(
//             v_ptr + n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
//             mask=v_mask,
//             other=0.0,
//         )
//
//         acc = acc * alpha[:, None]
//         acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
//         m_i = m_new
//         l_i = l_new
//
//     out = acc / l_i[:, None]
//     o_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM)
//     tl.store(
//         o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
//         out,
//         mask=o_mask,
//     )
//
//
// @triton.jit
// def flash_attention_fwd_async_buffer_kernel(
//     q_ptr,
//     k_ptr,
//     v_ptr,
//     o_ptr,
//     stride_qm: tl.constexpr,
//     stride_qd: tl.constexpr,
//     stride_kn: tl.constexpr,
//     stride_kd: tl.constexpr,
//     stride_vn: tl.constexpr,
//     stride_vd: tl.constexpr,
//     stride_om: tl.constexpr,
//     stride_od: tl.constexpr,
//     softmax_scale,
//     Q_LEN: tl.constexpr,
//     KV_LEN: tl.constexpr,
//     HEAD_DIM: tl.constexpr,
//     BLOCK_M: tl.constexpr,
//     BLOCK_N: tl.constexpr,
//     BLOCK_D: tl.constexpr,
// ):
//     """Blockwise flash attention shaped for AMD buffer ops plus async DMA.
//
//     The math matches flash_attention_fwd_no_async_kernel.  Strides are
//     constexpr so the fixture compiler can model compact contiguous tensors and
//     keep address expressions simple enough for AMD buffer-op lowering.
//     """
//
//     tl.static_assert(BLOCK_M >= 64)
//     tl.static_assert(BLOCK_N >= 64)
//     tl.static_assert(BLOCK_D >= 64)
//     tl.static_assert(BLOCK_D >= HEAD_DIM)
//
//     pid_m = tl.program_id(0)
//     offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
//     offs_n = tl.arange(0, BLOCK_N).to(tl.int32)
//     offs_d = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D).to(tl.int32)
//
//     q_offsets = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
//     q_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM)
//     q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
//
//     m_i = tl.full((BLOCK_M,), -1.0e20, tl.float32)
//     l_i = tl.zeros((BLOCK_M,), tl.float32)
//     acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
//     qk_scale = softmax_scale * 1.4426950408889634
//
//     # num_stages=2 marks this as a pipelined loop in the source.  When launched
//     # with num_stages=2 and AMD async-copy knobs enabled, the backend has the
//     # information it needs to choose buffer-load-to-LDS for eligible K/V loads.
//     for start_n in tl.range(0, KV_LEN, BLOCK_N, num_stages=2):
//         start_n = tl.multiple_of(start_n, BLOCK_N)
//         n = (start_n + offs_n).to(tl.int32)
//
//         k_offsets = n[None, :] * stride_kn + offs_d[:, None] * stride_kd
//         k_mask = (n[None, :] < KV_LEN) & (offs_d[:, None] < HEAD_DIM)
//         k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
//
//         qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale
//         qk = tl.where((offs_m[:, None] < Q_LEN) & (n[None, :] < KV_LEN), qk, -1.0e20)
//
//         m_new = tl.maximum(m_i, tl.max(qk, axis=1))
//         alpha = tl.math.exp2(m_i - m_new)
//         p = tl.math.exp2(qk - m_new[:, None])
//         l_new = l_i * alpha + tl.sum(p, axis=1)
//
//         v_offsets = n[:, None] * stride_vn + offs_d[None, :] * stride_vd
//         v_mask = (n[:, None] < KV_LEN) & (offs_d[None, :] < HEAD_DIM)
//         v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)
//
//         acc = acc * alpha[:, None]
//         acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
//         m_i = m_new
//         l_i = l_new
//
//     o_offsets = offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
//     out = acc / l_i[:, None]
//     o_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < HEAD_DIM)
//     tl.store(o_ptr + o_offsets, out, mask=o_mask)
//
//
// def flash_attention_grid(q_len=DEFAULT_Q_LEN, block_m=DEFAULT_BLOCK_M):
//     return (triton.cdiv(q_len, block_m),)
//
//
// __all__ = [
//     "ASYNC_BUFFER_ENV",
//     "COMMON_HIP_OPTIONS",
//     "DEFAULT_BLOCK_D",
//     "DEFAULT_BLOCK_M",
//     "DEFAULT_BLOCK_N",
//     "DEFAULT_HEAD_DIM",
//     "DEFAULT_KV_LEN",
//     "DEFAULT_Q_LEN",
//     "DEFAULT_STRIDE_KD",
//     "DEFAULT_STRIDE_KN",
//     "DEFAULT_STRIDE_OD",
//     "DEFAULT_STRIDE_OM",
//     "DEFAULT_STRIDE_QD",
//     "DEFAULT_STRIDE_QM",
//     "DEFAULT_STRIDE_VD",
//     "DEFAULT_STRIDE_VN",
//     "NO_ASYNC_ENV",
//     "TARGET_ARCH",
//     "flash_attention_fwd_async_buffer_kernel",
//     "flash_attention_fwd_no_async_kernel",
//     "flash_attention_grid",
// ]
// --- end tests/kernels/triton/flash_attention.py ---

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	flash_attention_fwd_no_async_kernel ; -- Begin function flash_attention_fwd_no_async_kernel
	.p2align	8
	.type	flash_attention_fwd_no_async_kernel,@function
flash_attention_fwd_no_async_kernel:    ; @flash_attention_fwd_no_async_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.9:
	.file	1 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/tests/kernels/triton" "flash_attention.py"
	.loc	1 61 0 prologue_end             ; flash_attention.py:61:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.10:
.LBB0_0:
.Ltmp1:
	.loc	1 95 21 is_stmt 1               ; flash_attention.py:95:21
	s_lshl_b32 s11, s16, 6
	.loc	1 95 44 is_stmt 0               ; flash_attention.py:95:44
	v_lshrrev_b32_e32 v1, 3, v0
	.loc	1 95 31                         ; flash_attention.py:95:31
	v_or_b32_e32 v4, s11, v1
	.loc	1 99 49 is_stmt 1               ; flash_attention.py:99:49
	v_lshlrev_b32_e32 v1, 3, v0
	s_movk_i32 s0, 0x400
	v_and_b32_e32 v2, 56, v1
	.loc	1 99 32 is_stmt 0               ; flash_attention.py:99:32
	v_cmp_gt_i32_e32 vcc, s0, v4
	.loc	1 101 46 is_stmt 1              ; flash_attention.py:101:46
	v_mov_b32_e32 v6, 0
	v_lshlrev_b32_e32 v2, 1, v2
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	.loc	1 101 8 is_stmt 0               ; flash_attention.py:101:8
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	.loc	1 0 8                           ; flash_attention.py:0:8
	v_lshlrev_b32_e32 v8, 6, v4
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	v_mov_b32_e32 v3, v6
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 101 8                         ; flash_attention.py:101:8
	global_load_dwordx4 v[10:13], v[8:9], off
.LBB0_2:
	.loc	1 0 8                           ; flash_attention.py:0:8
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	.loc	1 101 8                         ; flash_attention.py:101:8
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_4
; %bb.3:
	.loc	1 101 34                        ; flash_attention.py:101:34
	v_mov_b32_e32 v3, 0x800
	v_lshl_or_b32 v4, v4, 6, v3
	.loc	1 101 16                        ; flash_attention.py:101:16
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	.loc	1 101 46                        ; flash_attention.py:101:46
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	.loc	1 101 8                         ; flash_attention.py:101:8
	global_load_dwordx4 v[6:9], v[2:3], off
.LBB0_4:
	.loc	1 0 8                           ; flash_attention.py:0:8
	s_or_b64 exec, exec, s[0:1]
	.loc	1 95 44 is_stmt 1               ; flash_attention.py:95:44
	v_and_b32_e32 v2, 64, v0
	v_and_b32_e32 v52, 31, v0
	v_lshrrev_b32_e32 v50, 1, v2
	.loc	1 95 31 is_stmt 0               ; flash_attention.py:95:31
	v_or3_b32 v4, s11, v52, v50
	s_movk_i32 s0, 0x400
	.loc	1 99 32 is_stmt 1               ; flash_attention.py:99:32
	v_cmp_gt_i32_e32 vcc, s0, v4
	.loc	1 101 8                         ; flash_attention.py:101:8
	v_lshlrev_b32_e32 v4, 4, v0
	s_movk_i32 s0, 0x70
	v_bitop3_b32 v14, v4, v0, s0 bitop3:0x78
	.loc	1 95 44                         ; flash_attention.py:95:44
	v_and_b32_e32 v51, 32, v0
	.loc	1 101 8                         ; flash_attention.py:101:8
	v_add_u32_e32 v54, 0, v14
	s_waitcnt vmcnt(0)
	ds_write_b128 v54, v[10:13]
	ds_write_b128 v54, v[6:9] offset:4096
	v_and_b32_e32 v7, 0x70, v1
	v_lshrrev_b32_e32 v8, 1, v51
	v_lshlrev_b32_e32 v9, 6, v2
	v_lshlrev_b32_e32 v6, 7, v52
	v_bitop3_b32 v9, v7, v9, v8 bitop3:0xde
	v_or_b32_e32 v10, v9, v6
	s_movk_i32 s0, 0x60
	v_add_u32_e32 v11, 0, v10
	v_xad_u32 v12, v10, 32, 0
	v_xad_u32 v10, v10, 64, 0
	v_bitop3_b32 v9, v9, s0, v6 bitop3:0x36
	.loc	1 95 44                         ; flash_attention.py:95:44
	v_bfe_i32 v3, v0, 5, 1
	.loc	1 101 8                         ; flash_attention.py:101:8
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[16:19], v11
	ds_read_b128 v[20:23], v12
	v_add_u32_e32 v9, 0, v9
	ds_read_b128 v[24:27], v10
	ds_read_b128 v[28:31], v9
	v_lshlrev_b32_e32 v10, 1, v52
	s_movk_i32 s0, 0x210
	.loc	1 115 31                        ; flash_attention.py:115:31
	v_mov_b32_e32 v9, 0x3fb8aa3b
	v_and_b32_e32 v11, 0x210, v3
	v_bitop3_b32 v3, v3, v10, s0 bitop3:0x6c
	v_and_b32_e32 v1, 24, v1
	v_bfe_i32 v15, v0, 3, 1
	s_movk_i32 s0, 0x108
	.loc	1 95 44                         ; flash_attention.py:95:44
	v_and_b32_e32 v53, 0x80, v0
	.loc	1 115 31                        ; flash_attention.py:115:31
	v_mul_f32_e32 v55, s10, v9
	v_lshlrev_b32_e32 v9, 1, v2
	v_or_b32_e32 v10, v3, v2
	v_bitop3_b32 v12, v3, 8, v2 bitop3:0x36
	v_bitop3_b32 v13, v3, 32, v2 bitop3:0x36
	v_bitop3_b32 v2, v3, 40, v2 bitop3:0x36
	v_lshlrev_b32_e32 v3, 5, v0
	v_bitop3_b32 v15, v15, v1, s0 bitop3:0x6c
	.loc	1 101 8                         ; flash_attention.py:101:8
	v_and_b32_e32 v5, 0x70, v0
	v_and_b32_e32 v14, 0x80, v3
	v_lshrrev_b32_e32 v34, 1, v53
	v_xor_b32_e32 v15, v15, v11
	s_movk_i32 s0, 0x180
	v_lshlrev_b32_e32 v32, 1, v0
	v_or3_b32 v14, v14, v34, v15
	v_lshrrev_b32_e32 v5, 1, v5
	v_and_b32_e32 v34, 0x48, v0
	v_and_or_b32 v1, v3, s0, v1
	.loc	1 117 48                        ; flash_attention.py:117:48
	v_and_b32_e32 v0, 7, v0
	v_bitop3_b32 v56, v6, v8, v7 bitop3:0x36
	v_and_b32_e32 v33, 32, v32
	v_xor_b32_e32 v58, v4, v5
	v_bitop3_b32 v1, v11, v1, v34 bitop3:0x36
	v_and_b32_e32 v4, 0xf80, v4
	v_lshlrev_b32_e32 v0, 4, v0
	s_movk_i32 s0, 0x1000
	v_xor_b32_e32 v6, 32, v56
	v_xor_b32_e32 v7, 64, v56
	v_xor_b32_e32 v8, 0x60, v56
	v_lshl_add_u32 v57, v52, 2, 0
	v_or_b32_e32 v15, v14, v33
	v_bitop3_b32 v14, v14, 32, v32 bitop3:0x34
	v_xor_b32_e32 v5, 8, v58
	v_or_b32_e32 v3, v1, v33
	v_bitop3_b32 v1, v1, 32, v32 bitop3:0x34
	v_or3_b32 v36, v4, v0, s0
	v_mov_b32_e32 v37, 0
	.loc	1 119 22                        ; flash_attention.py:119:22
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
	.loc	1 117 48                        ; flash_attention.py:117:48
	v_lshl_add_u64 v[32:33], s[6:7], 0, v[36:37]
	v_lshl_add_u64 v[34:35], s[4:5], 0, v[36:37]
	v_mov_b32_e32 v36, 0xe0ad78ec
	s_movk_i32 s2, 0xffc0
	v_add_u32_e32 v59, 0, v6
	v_add_u32_e32 v60, 0, v7
	v_add_u32_e32 v61, 0, v8
	v_add_u32_e32 v62, 0, v10
	v_add_u32_e32 v64, 0, v12
	v_add_u32_e32 v65, 0, v13
	v_add_u32_e32 v66, 0, v2
	v_add_u32_e32 v67, 0, v15
	v_add_u32_e32 v68, 0, v14
	v_add_u32_e32 v69, 0, v5
	v_add_u32_e32 v70, 0, v3
	v_add_u32_e32 v71, 0, v1
	s_mov_b64 s[0:1], 0x2000
	v_add_u32_e32 v63, v57, v9
	v_mov_b32_e32 v0, 0xe0ad78ec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 123 12                        ; flash_attention.py:123:12
	global_load_dwordx4 v[2:5], v[34:35], off offset:-4096
	global_load_dwordx4 v[12:15], v[34:35], off
	v_add_u32_e32 v6, 0, v56
	.loc	1 119 22                        ; flash_attention.py:119:22
	s_nop 7
	v_accvgpr_read_b32 v9, a15
	v_accvgpr_read_b32 v8, a14
	v_accvgpr_read_b32 v11, a13
	v_accvgpr_read_b32 v10, a12
	v_accvgpr_read_b32 v39, a11
	v_accvgpr_read_b32 v38, a10
	v_accvgpr_read_b32 v41, a9
	v_accvgpr_read_b32 v40, a8
	v_accvgpr_read_b32 v43, a7
	v_accvgpr_read_b32 v42, a6
	v_accvgpr_read_b32 v45, a5
	v_accvgpr_read_b32 v44, a4
	v_accvgpr_read_b32 v47, a3
	v_accvgpr_read_b32 v46, a2
	v_accvgpr_read_b32 v49, a1
	v_accvgpr_read_b32 v48, a0
	.loc	1 123 12                        ; flash_attention.py:123:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v1, v37
	.loc	1 117 48                        ; flash_attention.py:117:48
	s_add_i32 s2, s2, 64
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[0:1]
	s_cmpk_lt_u32 s2, 0x3c0
	.loc	1 123 12                        ; flash_attention.py:123:12
	s_waitcnt vmcnt(1)
	ds_write_b128 v54, v[2:5]
	s_waitcnt vmcnt(0)
	ds_write_b128 v54, v[12:15] offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[2:5], v6
	ds_read_b128 v[12:15], v6 offset:4096
	ds_read_b128 v[72:75], v59
	ds_read_b128 v[76:79], v59 offset:4096
	ds_read_b128 v[80:83], v60
	ds_read_b128 v[84:87], v60 offset:4096
	ds_read_b128 v[88:91], v61
	ds_read_b128 v[92:95], v61 offset:4096
	.loc	1 128 23                        ; flash_attention.py:128:23
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_32x32x16_f16 a[0:15], v[2:5], v[16:19], 0
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_32x32x16_f16 a[0:15], v[72:75], v[20:23], a[0:15]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 a[0:15], v[80:83], v[24:27], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[88:91], v[28:31], a[0:15]
	s_nop 11
	v_accvgpr_read_b32 v2, a0
	v_accvgpr_read_b32 v3, a1
	v_accvgpr_read_b32 v4, a2
	v_accvgpr_read_b32 v5, a3
	v_accvgpr_read_b32 v6, a4
	v_accvgpr_read_b32 v7, a5
	v_accvgpr_read_b32 v37, a6
	v_accvgpr_read_b32 v72, a7
	v_accvgpr_read_b32 v73, a8
	v_accvgpr_read_b32 v74, a9
	v_accvgpr_read_b32 v75, a10
	v_accvgpr_read_b32 v80, a11
	v_accvgpr_read_b32 v81, a12
	v_accvgpr_read_b32 v82, a13
	v_accvgpr_read_b32 v83, a14
	v_accvgpr_read_b32 v88, a15
	v_mfma_f32_32x32x16_f16 a[0:15], v[12:15], v[16:19], 0
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v2, v55, v2
	v_mul_f32_e32 v3, v55, v3
	v_mul_f32_e32 v4, v55, v4
	v_mul_f32_e32 v5, v55, v5
	v_mul_f32_e32 v72, v55, v72
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v2, v36, v2, vcc
	v_cndmask_b32_e32 v3, v36, v3, vcc
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_mfma_f32_32x32x16_f16 a[0:15], v[76:79], v[20:23], a[0:15]
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v6, v55, v6
	v_mul_f32_e32 v7, v55, v7
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v4, v36, v4, vcc
	v_cndmask_b32_e32 v5, v36, v5, vcc
	.loc	1 128 50                        ; flash_attention.py:128:50
	v_mul_f32_e32 v37, v55, v37
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v6, v36, v6, vcc
	v_cndmask_b32_e32 v7, v36, v7, vcc
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_mfma_f32_32x32x16_f16 a[0:15], v[84:87], v[24:27], a[0:15]
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v73, v55, v73
	v_mul_f32_e32 v74, v55, v74
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v37, v36, v37, vcc
	.loc	1 128 50                        ; flash_attention.py:128:50
	v_mul_f32_e32 v75, v55, v75
	v_mul_f32_e32 v80, v55, v80
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v73, v36, v73, vcc
	v_cndmask_b32_e32 v74, v36, v74, vcc
	.loc	1 128 23                        ; flash_attention.py:128:23
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[92:95], v[28:31], a[0:15]
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v93, v36, v72, vcc
.Ltmp2:
	.file	2 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/.env/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max_f32_e32 v72, v2, v3
	v_max3_f32 v72, v72, v4, v5
	v_max3_f32 v72, v72, v6, v7
	v_max3_f32 v72, v72, v37, v93
.Ltmp3:
	.loc	1 128 50                        ; flash_attention.py:128:50
	v_mul_f32_e32 v81, v55, v81
	v_mul_f32_e32 v82, v55, v82
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v75, v36, v75, vcc
	v_cndmask_b32_e32 v80, v36, v80, vcc
.Ltmp4:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v73, v74
.Ltmp5:
	.loc	1 128 50                        ; flash_attention.py:128:50
	v_mul_f32_e32 v83, v55, v83
	v_mul_f32_e32 v88, v55, v88
	.loc	1 128 23 is_stmt 0              ; flash_attention.py:128:23
	v_accvgpr_read_b32 v12, a0
	v_accvgpr_read_b32 v13, a1
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v81, v36, v81, vcc
	v_cndmask_b32_e32 v82, v36, v82, vcc
.Ltmp6:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v75, v80
.Ltmp7:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v14, a2
	v_accvgpr_read_b32 v15, a3
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v12, v55, v12
	v_mul_f32_e32 v13, v55, v13
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v83, v36, v83, vcc
	v_cndmask_b32_e32 v88, v36, v88, vcc
.Ltmp8:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v81, v82
.Ltmp9:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v76, a4
	v_accvgpr_read_b32 v77, a5
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v14, v55, v14
	v_mul_f32_e32 v15, v55, v15
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v12, v36, v12, vcc
	v_cndmask_b32_e32 v13, v36, v13, vcc
.Ltmp10:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v83, v88
.Ltmp11:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v78, a6
	v_accvgpr_read_b32 v79, a7
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v76, v55, v76
	v_mul_f32_e32 v77, v55, v77
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v14, v36, v14, vcc
	v_cndmask_b32_e32 v15, v36, v15, vcc
.Ltmp12:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v12, v13
.Ltmp13:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v84, a8
	v_accvgpr_read_b32 v85, a9
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v78, v55, v78
	v_mul_f32_e32 v79, v55, v79
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v76, v36, v76, vcc
	v_cndmask_b32_e32 v77, v36, v77, vcc
.Ltmp14:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v14, v15
.Ltmp15:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v86, a10
	v_accvgpr_read_b32 v87, a11
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v84, v55, v84
	v_mul_f32_e32 v85, v55, v85
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v78, v36, v78, vcc
	v_cndmask_b32_e32 v79, v36, v79, vcc
.Ltmp16:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v76, v77
.Ltmp17:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v89, a12
	v_accvgpr_read_b32 v90, a13
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v86, v55, v86
	v_mul_f32_e32 v87, v55, v87
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v84, v36, v84, vcc
	v_cndmask_b32_e32 v85, v36, v85, vcc
.Ltmp18:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v78, v79
.Ltmp19:
	.loc	1 128 23                        ; flash_attention.py:128:23
	v_accvgpr_read_b32 v91, a14
	v_accvgpr_read_b32 v92, a15
	.loc	1 128 50 is_stmt 0              ; flash_attention.py:128:50
	v_mul_f32_e32 v89, v55, v89
	v_mul_f32_e32 v90, v55, v90
	.loc	1 130 67 is_stmt 1              ; flash_attention.py:130:67
	v_cndmask_b32_e32 v86, v36, v86, vcc
	v_cndmask_b32_e32 v87, v36, v87, vcc
.Ltmp20:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v84, v85
.Ltmp21:
	.loc	1 128 50                        ; flash_attention.py:128:50
	v_mul_f32_e32 v91, v55, v91
	v_mul_f32_e32 v92, v55, v92
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v89, v36, v89, vcc
	v_cndmask_b32_e32 v90, v36, v90, vcc
.Ltmp22:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v86, v87
.Ltmp23:
	.loc	1 130 67                        ; flash_attention.py:130:67
	v_cndmask_b32_e32 v91, v36, v91, vcc
	v_cndmask_b32_e32 v92, v36, v92, vcc
.Ltmp24:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:133:39 ] ]
	v_max3_f32 v72, v72, v89, v90
	v_max3_f32 v72, v72, v91, v92
.Ltmp25:
	.loc	2 191 40                        ; standard.py:191:40 @[ flash_attention.py:133:39 ]
	v_mov_b32_e32 v94, v72
	s_nop 1
	v_permlane32_swap_b32_e32 v72, v94
.Ltmp26:
	.loc	1 133 32                        ; flash_attention.py:133:32
	v_max3_f32 v72, v0, v72, v94
	.loc	1 134 35                        ; flash_attention.py:134:35
	v_sub_f32_e32 v0, v0, v72
	.loc	1 134 29 is_stmt 0              ; flash_attention.py:134:29
	v_exp_f32_e32 v102, v0
	.loc	1 135 30 is_stmt 1              ; flash_attention.py:135:30
	v_sub_f32_e32 v0, v2, v72
	v_sub_f32_e32 v2, v3, v72
	v_sub_f32_e32 v3, v4, v72
	v_sub_f32_e32 v99, v12, v72
	v_sub_f32_e32 v100, v13, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v12, v0
	v_exp_f32_e32 v13, v2
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v4, v5, v72
	v_sub_f32_e32 v101, v14, v72
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v14, v3
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v5, v6, v72
	v_sub_f32_e32 v103, v15, v72
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v15, v4
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v6, v7, v72
	v_sub_f32_e32 v7, v37, v72
	v_sub_f32_e32 v37, v93, v72
	v_sub_f32_e32 v93, v74, v72
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v74, v5
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v94, v75, v72
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v75, v6
.Ltmp27:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v12, v13
.Ltmp28:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v104, v76, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v76, v7
.Ltmp29:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v14, v0
.Ltmp30:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v73, v73, v72
	v_sub_f32_e32 v105, v77, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v77, v37
.Ltmp31:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v15, v0
.Ltmp32:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v106, v78, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v78, v73
.Ltmp33:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v74, v0
.Ltmp34:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v107, v79, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v79, v93
.Ltmp35:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v75, v0
.Ltmp36:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v95, v80, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v80, v94
.Ltmp37:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v76, v0
.Ltmp38:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v96, v81, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v81, v95
.Ltmp39:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v77, v0
.Ltmp40:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v97, v82, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v82, v96
.Ltmp41:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v78, v0
.Ltmp42:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v98, v83, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v83, v97
.Ltmp43:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v79, v0
.Ltmp44:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v88, v88, v72
	v_sub_f32_e32 v108, v84, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v84, v98
.Ltmp45:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v80, v0
.Ltmp46:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v109, v85, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v85, v88
.Ltmp47:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v81, v0
.Ltmp48:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v110, v86, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v86, v99
.Ltmp49:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v82, v0
.Ltmp50:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v111, v87, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v87, v100
.Ltmp51:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v83, v0
.Ltmp52:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v88, v101
.Ltmp53:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v84, v0
.Ltmp54:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v112, v89, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v89, v103
.Ltmp55:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v85, v0
.Ltmp56:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v113, v90, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v90, v104
.Ltmp57:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v86, v0
.Ltmp58:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v114, v91, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v91, v105
.Ltmp59:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v87, v0
.Ltmp60:
	.loc	1 135 30                        ; flash_attention.py:135:30
	v_sub_f32_e32 v115, v92, v72
	.loc	1 135 25 is_stmt 0              ; flash_attention.py:135:25
	v_exp_f32_e32 v92, v106
.Ltmp61:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v88, v0
.Ltmp62:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v93, v107
.Ltmp63:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v89, v0
.Ltmp64:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v94, v108
.Ltmp65:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v90, v0
.Ltmp66:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v95, v109
.Ltmp67:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v91, v0
.Ltmp68:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v96, v110
.Ltmp69:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v92, v0
.Ltmp70:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v97, v111
.Ltmp71:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v93, v0
.Ltmp72:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v98, v112
.Ltmp73:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v94, v0
.Ltmp74:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v99, v113
.Ltmp75:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v95, v0
.Ltmp76:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v100, v114
.Ltmp77:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v96, v0
.Ltmp78:
	.loc	1 135 25                        ; flash_attention.py:135:25
	v_exp_f32_e32 v101, v115
.Ltmp79:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v0, v97, v0
	v_add_f32_e32 v0, v98, v0
	v_add_f32_e32 v0, v99, v0
	v_add_f32_e32 v0, v100, v0
	v_add_f32_e32 v0, v101, v0
.Ltmp80:
	.loc	2 293 36                        ; standard.py:293:36 @[ flash_attention.py:136:37 ]
	v_mov_b32_e32 v2, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v2
.Ltmp81:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:136:37 ] ]
	v_add_f32_e32 v37, v0, v2
.Ltmp82:
	.loc	1 136 30                        ; flash_attention.py:136:30
	v_fmac_f32_e32 v37, v1, v102
	.loc	1 140 12                        ; flash_attention.py:140:12
	global_load_dwordx4 v[0:3], v[32:33], off offset:-4096
	global_load_dwordx4 v[4:7], v[32:33], off
	.loc	1 145 20                        ; flash_attention.py:145:20
	v_add_u32_e32 v73, v57, v53
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v63, v102
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v102, v73
	.loc	1 146 26                        ; flash_attention.py:146:26
	v_cvt_pk_f16_f32 v12, v12, v13
	v_cvt_pk_f16_f32 v13, v14, v15
	v_cvt_pk_f16_f32 v14, v74, v75
	v_cvt_pk_f16_f32 v15, v76, v77
	v_cvt_pk_f16_f32 v74, v78, v79
	v_cvt_pk_f16_f32 v75, v80, v81
	v_cvt_pk_f16_f32 v76, v82, v83
	v_cvt_pk_f16_f32 v77, v84, v85
	v_cvt_pk_f16_f32 v78, v86, v87
	v_cvt_pk_f16_f32 v79, v88, v89
	v_cvt_pk_f16_f32 v80, v90, v91
	v_cvt_pk_f16_f32 v81, v92, v93
	v_cvt_pk_f16_f32 v82, v94, v95
	v_cvt_pk_f16_f32 v83, v96, v97
	v_cvt_pk_f16_f32 v84, v98, v99
	v_cvt_pk_f16_f32 v85, v100, v101
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v62, v12
	ds_write_b16_d16_hi v62, v12 offset:128
	ds_write_b16 v62, v74 offset:2048
	ds_write_b16_d16_hi v62, v74 offset:2176
	ds_write_b16 v62, v78 offset:4096
	ds_write_b16_d16_hi v62, v78 offset:4224
	ds_write_b16 v62, v82 offset:6144
	ds_write_b16_d16_hi v62, v82 offset:6272
	ds_write_b16 v64, v13 offset:256
	ds_write_b16_d16_hi v64, v13 offset:384
	ds_write_b16 v64, v75 offset:2304
	ds_write_b16_d16_hi v64, v75 offset:2432
	ds_write_b16 v64, v79 offset:4352
	ds_write_b16_d16_hi v64, v79 offset:4480
	ds_write_b16 v64, v83 offset:6400
	ds_write_b16_d16_hi v64, v83 offset:6528
	ds_write_b16 v65, v14 offset:1024
	ds_write_b16_d16_hi v65, v14 offset:1152
	ds_write_b16 v65, v76 offset:3072
	ds_write_b16_d16_hi v65, v76 offset:3200
	ds_write_b16 v65, v80 offset:5120
	ds_write_b16_d16_hi v65, v80 offset:5248
	ds_write_b16 v65, v84 offset:7168
	ds_write_b16_d16_hi v65, v84 offset:7296
	ds_write_b16 v66, v15 offset:1280
	ds_write_b16_d16_hi v66, v15 offset:1408
	ds_write_b16 v66, v77 offset:3328
	ds_write_b16_d16_hi v66, v77 offset:3456
	ds_write_b16 v66, v81 offset:5376
	ds_write_b16_d16_hi v66, v81 offset:5504
	ds_write_b16 v66, v85 offset:7424
	ds_write_b16_d16_hi v66, v85 offset:7552
	.loc	1 140 12                        ; flash_attention.py:140:12
	v_add_u32_e32 v12, 0, v58
	.loc	1 146 26                        ; flash_attention.py:146:26
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[74:75], v67
	ds_read_b64_tr_b16 v[78:79], v67 offset:2048
	ds_read_b64_tr_b16 v[82:83], v67 offset:4096
	ds_read_b64_tr_b16 v[86:87], v67 offset:6144
	ds_read_b64_tr_b16 v[76:77], v68 offset:1024
	ds_read_b64_tr_b16 v[80:81], v68 offset:3072
	ds_read_b64_tr_b16 v[84:85], v68 offset:5120
	ds_read_b64_tr_b16 v[88:89], v68 offset:7168
	.loc	1 140 12                        ; flash_attention.py:140:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write2st64_b64 v12, v[0:1], v[4:5] offset1:8
	ds_write2st64_b64 v69, v[2:3], v[6:7] offset1:8
	.loc	1 146 42                        ; flash_attention.py:146:42
	v_pk_mul_f32 v[0:1], v[48:49], v[102:103] op_sel_hi:[1,0]
	.loc	1 140 12                        ; flash_attention.py:140:12
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[90:91], v70
	ds_read_b64_tr_b16 v[94:95], v70 offset:2048
	ds_read_b64_tr_b16 v[98:99], v70 offset:4096
	ds_read_b64_tr_b16 v[104:105], v70 offset:6144
	ds_read_b64_tr_b16 v[92:93], v71 offset:1024
	ds_read_b64_tr_b16 v[96:97], v71 offset:3072
	ds_read_b64_tr_b16 v[100:101], v71 offset:5120
	ds_read_b64_tr_b16 v[106:107], v71 offset:7168
	.loc	1 146 42                        ; flash_attention.py:146:42
	v_pk_mul_f32 v[14:15], v[8:9], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[10:11], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[38:39], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[40:41], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[42:43], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[44:45], v[102:103] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[46:47], v[102:103] op_sel_hi:[1,0]
	.loc	1 117 48                        ; flash_attention.py:117:48
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	.loc	1 146 42                        ; flash_attention.py:146:42
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
	v_mov_b32_e32 v0, v72
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 a[0:15], v[90:93], v[74:77], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[94:97], v[78:81], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[98:101], v[82:85], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[104:107], v[86:89], a[0:15]
	.loc	1 117 48                        ; flash_attention.py:117:48
	s_cbranch_scc1 .LBB0_5
; %bb.6:
	.loc	1 95 44                         ; flash_attention.py:95:44
	v_lshrrev_b32_e32 v16, 2, v53
	.loc	1 146 42                        ; flash_attention.py:146:42
	s_nop 9
	v_accvgpr_read_b32 v0, a0
	.loc	1 95 31                         ; flash_attention.py:95:31
	v_or3_b32 v16, v52, v16, s11
	s_movk_i32 s0, 0x400
	.loc	1 146 42                        ; flash_attention.py:146:42
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
	.loc	1 99 32                         ; flash_attention.py:99:32
	v_cmp_gt_i32_e32 vcc, s0, v16
	.loc	1 150 16                        ; flash_attention.py:150:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v63, v37
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 154 8                         ; flash_attention.py:154:8
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:                                ; %.critedge
	.loc	1 150 16                        ; flash_attention.py:150:16
	ds_read_b32 v20, v73
	.loc	1 95 44                         ; flash_attention.py:95:44
	v_lshrrev_b32_e32 v17, 3, v51
	.loc	1 101 34                        ; flash_attention.py:101:34
	v_lshlrev_b32_e32 v16, 6, v16
	.loc	1 99 49                         ; flash_attention.py:99:49
	v_or_b32_e32 v18, v17, v50
	.loc	1 153 16                        ; flash_attention.py:153:16
	v_ashrrev_i32_e32 v17, 31, v16
	.loc	1 150 16                        ; flash_attention.py:150:16
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v21, s[0:1], v20, v20, v15
	v_rcp_f32_e32 v22, v21
	.loc	1 153 16                        ; flash_attention.py:153:16
	v_lshl_add_u64 v[16:17], v[16:17], 2, s[8:9]
	.loc	1 153 46 is_stmt 0              ; flash_attention.py:153:46
	v_lshlrev_b32_e32 v18, 2, v18
	v_mov_b32_e32 v19, 0
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	.loc	1 150 16 is_stmt 1              ; flash_attention.py:150:16
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
	v_div_scale_f32 v21, s[0:1], v20, v20, v0
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v8, v18, v20, v8
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
	v_div_scale_f32 v21, s[0:1], v20, v20, v2
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v1, v18, v20, v1
	v_fma_f32 v18, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v18, v22
	v_div_scale_f32 v18, vcc, v2, v20, v2
	v_mul_f32_e32 v19, v18, v22
	v_fma_f32 v23, -v21, v19, v18
	v_fmac_f32_e32 v19, v23, v22
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v3
	v_rcp_f32_e32 v23, v21
	v_div_fmas_f32 v18, v18, v22, v19
	v_div_fixup_f32 v2, v18, v20, v2
	v_fma_f32 v18, -v21, v23, 1.0
	v_fmac_f32_e32 v23, v18, v23
	v_div_scale_f32 v18, vcc, v3, v20, v3
	v_mul_f32_e32 v19, v18, v23
	v_fma_f32 v22, -v21, v19, v18
	v_fmac_f32_e32 v19, v22, v23
	v_fma_f32 v18, -v21, v19, v18
	v_div_scale_f32 v21, s[0:1], v20, v20, v7
	v_rcp_f32_e32 v22, v21
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v3, v18, v20, v3
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
	v_div_fmas_f32 v18, v18, v23, v19
	v_div_fixup_f32 v4, v18, v20, v4
	.loc	1 154 8                         ; flash_attention.py:154:8
	global_store_dwordx4 v[16:17], v[0:3], off
	global_store_dwordx4 v[16:17], v[4:7], off offset:32
	global_store_dwordx4 v[16:17], v[8:11], off offset:64
	global_store_dwordx4 v[16:17], v[12:15], off offset:96
.LBB0_8:                                ; %.critedge4
	.loc	1 152 4                         ; flash_attention.py:152:4
	s_endpgm
.Ltmp83:
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
		.amdhsa_next_free_vgpr 132
		.amdhsa_next_free_sgpr 17
		.amdhsa_accum_offset 116
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
	.set flash_attention_fwd_no_async_kernel.num_vgpr, 116
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
; codeLenInByte = 4840
; TotalNumSgprs: 23
; NumVgprs: 116
; NumAgprs: 16
; TotalNumVgprs: 132
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 16
; NumSGPRsForWavesPerEU: 23
; NumVGPRsForWavesPerEU: 132
; AccumOffset: 116
; Occupancy: 3
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 28
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
	.byte	133                             ; DW_AT_call_line
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
	.byte	136                             ; DW_AT_call_line
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
	.quad	.Ltmp26-.Lfunc_begin0
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
	.quad	0
	.quad	0
.Ldebug_ranges2:
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
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
	.quad	.Ltmp79-.Lfunc_begin0
	.quad	.Ltmp80-.Lfunc_begin0
	.quad	.Ltmp81-.Lfunc_begin0
	.quad	.Ltmp82-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0
.Linfo_string1:
	.asciz	"flash_attention.py"            ; string offset=7
.Linfo_string2:
	.asciz	"/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/tests/kernels/triton" ; string offset=26
.Linfo_string3:
	.asciz	"flash_attention_fwd_no_async_kernel" ; string offset=142
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
    .vgpr_count:     132
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
