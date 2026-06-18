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
// # Recorded async-buffer fixture metadata:
// #   kernel: flash_attention_fwd_async_buffer_kernel
// #   target: gfx950, wavefront_size: 64
// #   launch sharedMemBytes used by cdna4_to_cdna3_dispatch_test: 65536
// #   amdhsa_group_segment_fixed_size: 0
// #   max_flat_workgroup_size: 256
// #   kernarg_segment_size: 56
// #   registers: vgpr_count=156, agpr_count=32, sgpr_count=31
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
	.globl	flash_attention_fwd_async_buffer_kernel ; -- Begin function flash_attention_fwd_async_buffer_kernel
	.p2align	8
	.type	flash_attention_fwd_async_buffer_kernel,@function
flash_attention_fwd_async_buffer_kernel: ; @flash_attention_fwd_async_buffer_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
	.file	1 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/tests/kernels/triton" "flash_attention.py"
	.loc	1 160 0 prologue_end            ; flash_attention.py:160:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.4:
.LBB0_0:
.Ltmp1:
	.loc	1 198 53 is_stmt 1              ; flash_attention.py:198:53
	v_lshlrev_b32_e32 v11, 3, v0
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_lshrrev_b32_e32 v1, 3, v0
	.loc	1 198 53                        ; flash_attention.py:198:53
	v_and_b32_e32 v3, 56, v11
	.loc	1 194 22                        ; flash_attention.py:194:22
	s_lshl_b32 s17, s16, 6
	.loc	1 200 24                        ; flash_attention.py:200:24
	s_lshl_b32 s11, s16, 12
	v_lshl_or_b32 v41, v1, 6, v3
	s_mov_b64 s[12:13], s[6:7]
	.loc	1 194 32                        ; flash_attention.py:194:32
	v_or_b32_e32 v2, s17, v1
	s_movk_i32 s6, 0x400
	.loc	1 200 24                        ; flash_attention.py:200:24
	v_or_b32_e32 v1, 0x800, v41
	v_or_b32_e32 v3, s11, v41
	s_mov_b64 s[0:1], s[2:3]
	v_or_b32_e32 v4, s11, v1
	.loc	1 200 16 is_stmt 0              ; flash_attention.py:200:16
	v_lshlrev_b32_e32 v3, 1, v3
	v_bfrev_b32_e32 v42, 1
	.loc	1 199 32 is_stmt 1              ; flash_attention.py:199:32
	v_cmp_gt_i32_e32 vcc, s6, v2
	.loc	1 200 16                        ; flash_attention.py:200:16
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffe
	v_cndmask_b32_e32 v10, v42, v3, vcc
	v_lshlrev_b32_e32 v2, 1, v4
	v_cndmask_b32_e32 v12, v42, v2, vcc
	buffer_load_dwordx4 v[2:5], v10, s[0:3], 0 offen
	buffer_load_dwordx4 v[6:9], v12, s[0:3], 0 offen
	v_lshlrev_b32_e32 v10, 4, v0
	v_and_b32_e32 v12, 0x70, v0
	v_xad_u32 v15, v10, v12, 0
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_lshlrev_b32_e32 v33, 1, v0
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_and_b32_e32 v32, 64, v0
	v_and_b32_e32 v46, 32, v0
	v_and_b32_e32 v45, 31, v0
	.loc	1 200 16                        ; flash_attention.py:200:16
	v_and_b32_e32 v36, 0x70, v11
	v_lshrrev_b32_e32 v37, 1, v46
	v_lshlrev_b32_e32 v10, 6, v32
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_lshrrev_b32_e32 v12, 1, v12
	s_movk_i32 s0, 0x60
	.loc	1 200 16                        ; flash_attention.py:200:16
	v_lshlrev_b32_e32 v35, 7, v45
	v_bitop3_b32 v10, v36, v10, v37 bitop3:0xde
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_xor_b32_e32 v12, v11, v12
	.loc	1 200 16                        ; flash_attention.py:200:16
	v_or_b32_e32 v13, v10, v35
	v_bitop3_b32 v10, v10, s0, v35 bitop3:0x36
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_sub_u32_e32 v12, v12, v11
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_and_b32_e32 v14, 63, v0
	.loc	1 200 16                        ; flash_attention.py:200:16
	v_add_u32_e32 v19, 0, v10
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_ashrrev_i32_e32 v10, 3, v12
	v_add_u32_e32 v10, v10, v14
	.loc	1 200 16                        ; flash_attention.py:200:16
	v_add_u32_e32 v16, 0, v13
	v_xad_u32 v17, v13, 32, 0
	v_xad_u32 v18, v13, 64, 0
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_lshlrev_b32_e32 v51, 2, v10
	v_lshrrev_b64 v[12:13], v10, exec
	ds_bpermute_b32 v13, v51, v41
	ds_bpermute_b32 v1, v51, v1
	s_movk_i32 s1, 0x1c0
	.loc	1 230 20                        ; flash_attention.py:230:20
	v_readfirstlane_b32 s7, v0
	.loc	1 198 53                        ; flash_attention.py:198:53
	v_lshlrev_b32_e32 v34, 1, v45
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_bfe_u32 s19, s7, 0x20006
	.loc	1 230 28 is_stmt 0              ; flash_attention.py:230:28
	v_and_or_b32 v50, v33, s1, v34
	.loc	1 216 20 is_stmt 1              ; flash_attention.py:216:20
	v_and_b32_e32 v12, 1, v12
	s_lshl_b32 s16, s19, 10
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v13, 1, v13
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_eq_u32_e32 vcc, 1, v12
	s_and_b32 s5, s5, 0xffff
	s_add_i32 s7, s16, 0
	v_cndmask_b32_e32 v12, v42, v13, vcc
	v_cndmask_b32_e32 v1, v42, v1, vcc
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_or_b32_e32 v38, 0x200, v50
	.loc	1 216 20                        ; flash_attention.py:216:20
	s_mov_b32 s0, s4
	s_mov_b32 s1, s5
	s_mov_b32 m0, s7
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s14, s2
	s_mov_b32 s15, s3
	.loc	1 230 28 is_stmt 0              ; flash_attention.py:230:28
	v_or_b32_e32 v13, 0xa00, v50
	v_or_b32_e32 v39, 0xe00, v50
	.loc	1 194 45 is_stmt 1              ; flash_attention.py:194:45
	v_lshrrev_b32_e32 v47, 1, v32
	v_bitop3_b32 v62, v35, v37, v36 bitop3:0x36
	s_mov_b32 s18, 0
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_mov_b32_e32 v53, 0
	.loc	1 210 48                        ; flash_attention.py:210:48
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	.loc	1 200 16                        ; flash_attention.py:200:16
	s_waitcnt vmcnt(1)
	ds_write_b128 v15, v[2:5]
	s_waitcnt vmcnt(0)
	ds_write_b128 v15, v[6:9] offset:4096
	.loc	1 230 20                        ; flash_attention.py:230:20
	v_lshrrev_b32_e32 v2, 4, v0
	v_bitop3_b32 v2, v33, v2, 12 bitop3:0x78
	v_xor_b32_e32 v5, 0x210, v2
	v_sub_u32_e32 v5, v5, v33
	v_add_u32_e32 v5, 0xfffffe00, v5
	v_ashrrev_i32_e32 v6, 1, v5
	v_xor_b32_e32 v5, 0x610, v2
	v_sub_u32_e32 v5, v5, v33
	v_add_u32_e32 v5, 0xfffffa00, v5
	v_sub_u32_e32 v4, v2, v33
	v_ashrrev_i32_e32 v8, 1, v5
	v_xor_b32_e32 v5, 0xa10, v2
	v_xor_b32_e32 v2, 0xe10, v2
	v_sub_u32_e32 v2, v2, v33
	v_ashrrev_i32_e32 v4, 1, v4
	v_sub_u32_e32 v5, v5, v33
	v_add_u32_e32 v2, 0xfffff200, v2
	v_add_u32_e32 v5, 0xfffff600, v5
	v_ashrrev_i32_e32 v43, 1, v2
	v_add_u32_e32 v2, v4, v14
	v_ashrrev_i32_e32 v40, 1, v5
	v_lshlrev_b32_e32 v54, 2, v2
	v_lshrrev_b64 v[4:5], v2, exec
	ds_bpermute_b32 v7, v54, v50
	v_and_b32_e32 v4, 1, v4
	v_cmp_eq_u32_e32 vcc, 1, v4
	v_add_u32_e32 v4, v6, v14
	v_lshlrev_b32_e32 v55, 2, v4
	.loc	1 200 16                        ; flash_attention.py:200:16
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[28:31], v16
	ds_read_b128 v[24:27], v17
	ds_read_b128 v[20:23], v18
	ds_read_b128 v[16:19], v19
	.loc	1 216 20                        ; flash_attention.py:216:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v12, s[0:3], 0 offen lds
	s_add_i32 m0, s7, 0x1000
	.loc	1 230 20                        ; flash_attention.py:230:20
	ds_bpermute_b32 v9, v55, v38
	.loc	1 216 20                        ; flash_attention.py:216:20
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_mul_i32 s0, s19, 0xfffffd00
	s_add_i32 s7, s7, s0
	v_lshlrev_b32_e32 v5, 1, v7
	s_add_i32 m0, s7, 0x4000
	v_cndmask_b32_e32 v5, v42, v5, vcc
	v_lshrrev_b64 v[6:7], v4, exec
	.loc	1 230 28 is_stmt 0              ; flash_attention.py:230:28
	v_or_b32_e32 v1, 0x400, v50
	.loc	1 230 20                        ; flash_attention.py:230:20
	buffer_load_dword v5, s[12:15], 0 offen lds
	v_and_b32_e32 v5, 1, v6
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v6, 1, v9
	ds_bpermute_b32 v1, v54, v1
	v_cmp_eq_u32_e64 s[0:1], 1, v5
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_or_b32_e32 v3, 0x600, v50
	v_or_b32_e32 v12, 0x800, v50
	.loc	1 230 20                        ; flash_attention.py:230:20
	v_cndmask_b32_e64 v5, v42, v6, s[0:1]
	v_add_u32_e32 v6, v8, v14
	v_lshlrev_b32_e32 v56, 2, v6
	ds_bpermute_b32 v3, v56, v3
	s_add_i32 m0, s7, 0x4400
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_load_dword v5, s[12:15], 0 offen lds
	s_add_i32 m0, s7, 0x4800
	v_cndmask_b32_e32 v1, v42, v1, vcc
	v_lshrrev_b64 v[8:9], v6, exec
	ds_bpermute_b32 v5, v54, v12
	buffer_load_dword v1, s[12:15], 0 offen lds
	v_and_b32_e32 v1, 1, v8
	v_add_u32_e32 v8, v40, v14
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v3, 1, v3
	v_cmp_eq_u32_e64 s[0:1], 1, v1
	v_lshlrev_b32_e32 v57, 2, v8
	s_add_i32 m0, s7, 0x4c00
	v_cndmask_b32_e64 v1, v42, v3, s[0:1]
	ds_bpermute_b32 v3, v57, v13
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_or_b32_e32 v15, 0xc00, v50
	.loc	1 230 20                        ; flash_attention.py:230:20
	buffer_load_dword v1, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v5
	s_add_i32 m0, s7, 0x5000
	v_cndmask_b32_e32 v1, v42, v1, vcc
	v_lshrrev_b64 v[12:13], v8, exec
	ds_bpermute_b32 v5, v54, v15
	buffer_load_dword v1, s[12:15], 0 offen lds
	v_and_b32_e32 v1, 1, v12
	v_add_u32_e32 v12, v43, v14
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v3, 1, v3
	v_cmp_eq_u32_e64 s[0:1], 1, v1
	v_lshlrev_b32_e32 v58, 2, v12
	s_add_i32 m0, s7, 0x5400
	v_cndmask_b32_e64 v1, v42, v3, s[0:1]
	ds_bpermute_b32 v3, v58, v39
	buffer_load_dword v1, s[12:15], 0 offen lds
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v1, 1, v5
	s_add_i32 m0, s7, 0x5800
	v_cndmask_b32_e32 v1, v42, v1, vcc
	v_lshrrev_b64 v[14:15], v12, exec
	buffer_load_dword v1, s[12:15], 0 offen lds
	v_and_b32_e32 v1, 1, v14
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v3, 1, v3
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_add_i32 m0, s7, 0x5c00
	s_movk_i32 s0, 0x210
	v_cndmask_b32_e32 v1, v42, v3, vcc
	buffer_load_dword v1, s[12:15], 0 offen lds
	.loc	1 194 32 is_stmt 1              ; flash_attention.py:194:32
	v_or3_b32 v3, s17, v45, v47
	.loc	1 194 45 is_stmt 0              ; flash_attention.py:194:45
	v_bfe_i32 v1, v0, 5, 1
	.loc	1 199 32 is_stmt 1              ; flash_attention.py:199:32
	v_cmp_gt_i32_e32 vcc, s6, v3
	.loc	1 205 31                        ; flash_attention.py:205:31
	v_mov_b32_e32 v3, 0x3fb8aa3b
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_and_b32_e32 v40, 0x80, v0
	.loc	1 205 31                        ; flash_attention.py:205:31
	v_mul_f32_e32 v52, s10, v3
	v_lshlrev_b32_e32 v3, 5, v0
	v_and_b32_e32 v5, 24, v11
	v_and_b32_e32 v7, 0x48, v0
	v_and_b32_e32 v11, 0x210, v1
	v_bitop3_b32 v1, v1, v34, s0 bitop3:0x6c
	v_bfe_i32 v0, v0, 3, 1
	s_movk_i32 s0, 0x108
	s_movk_i32 s1, 0x180
	v_bitop3_b32 v0, v0, v5, s0 bitop3:0x6c
	v_and_or_b32 v13, v3, s1, v5
	v_and_b32_e32 v3, 0x80, v3
	v_xor_b32_e32 v0, v0, v11
	v_lshrrev_b32_e32 v5, 1, v40
	v_and_b32_e32 v9, 32, v33
	v_or3_b32 v0, v3, v5, v0
	v_or_b32_e32 v64, v1, v32
	v_bitop3_b32 v65, v1, 8, v32 bitop3:0x36
	v_bitop3_b32 v66, v1, 32, v32 bitop3:0x36
	v_bitop3_b32 v67, v1, 40, v32 bitop3:0x36
	v_or_b32_e32 v68, v0, v9
	v_bitop3_b32 v69, v0, 32, v33 bitop3:0x34
	v_lshrrev_b64 v[0:1], v10, exec
	v_and_b32_e32 v72, 1, v0
	v_lshrrev_b64 v[0:1], v2, exec
	v_and_b32_e32 v74, 1, v0
	v_lshrrev_b64 v[0:1], v4, exec
	v_and_b32_e32 v75, 1, v0
	v_lshrrev_b64 v[0:1], v6, exec
	v_and_b32_e32 v76, 1, v0
	v_lshrrev_b64 v[0:1], v8, exec
	v_bitop3_b32 v7, v11, v13, v7 bitop3:0x36
	v_lshlrev_b32_e32 v43, 2, v45
	v_and_b32_e32 v77, 1, v0
	v_lshrrev_b64 v[0:1], v12, exec
	.loc	1 216 20                        ; flash_attention.py:216:20
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	.loc	1 210 48                        ; flash_attention.py:210:48
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_lshl_b32 s19, s19, 8
	v_xor_b32_e32 v61, 32, v62
	v_xor_b32_e32 v60, 64, v62
	v_xor_b32_e32 v59, 0x60, v62
	v_or_b32_e32 v48, v7, v9
	v_lshlrev_b32_e32 v44, 1, v32
	v_add_u32_e32 v63, 0, v43
	v_bitop3_b32 v49, v7, 32, v33 bitop3:0x34
	.loc	1 210 48                        ; flash_attention.py:210:48
	v_or_b32_e32 v70, 0x1e00, v50
	v_or_b32_e32 v71, 0x1600, v50
	s_add_i32 s22, 0, 0x4000
	v_mov_b32_e32 v73, 0xe0ad78ec
	s_movk_i32 s20, 0xffc0
	v_and_b32_e32 v78, 1, v0
	s_mov_b32 s10, 0
	s_mov_b32 s21, 0
	v_mov_b32_e32 v79, 0xe0ad78ec
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 0 48 is_stmt 0                ; flash_attention.py:0:48
	v_mov_b32_e32 v104, v79
	v_mov_b32_e32 v105, v53
	s_mov_b32 s2, s10
	.loc	1 210 48                        ; flash_attention.py:210:48
	s_nop 7
	v_accvgpr_read_b32 v1, a15
	v_accvgpr_read_b32 v0, a14
	v_accvgpr_read_b32 v3, a13
	v_accvgpr_read_b32 v2, a12
	v_accvgpr_read_b32 v5, a11
	v_accvgpr_read_b32 v4, a10
	v_accvgpr_read_b32 v7, a9
	v_accvgpr_read_b32 v6, a8
	v_accvgpr_read_b32 v33, a7
	v_accvgpr_read_b32 v32, a6
	v_accvgpr_read_b32 v35, a5
	v_accvgpr_read_b32 v34, a4
	v_accvgpr_read_b32 v37, a3
	v_accvgpr_read_b32 v36, a2
	v_accvgpr_read_b32 v39, a1
	v_accvgpr_read_b32 v38, a0
	; sched_barrier mask(0x00000000)
	s_add_i32 s3, s21, 1
	s_cmp_lt_i32 s3, 2
	s_cselect_b32 s21, s3, 0
	.loc	1 216 28 is_stmt 1              ; flash_attention.py:216:28
	v_add_u32_e32 v8, s18, v41
	.loc	1 216 20 is_stmt 0              ; flash_attention.py:216:20
	s_lshl_b32 s3, s21, 13
	.loc	1 216 28                        ; flash_attention.py:216:28
	v_add_u32_e32 v9, 0x1000, v8
	.loc	1 216 20                        ; flash_attention.py:216:20
	s_add_i32 s10, s3, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_cmp_eq_u32_e64 s[0:1], 1, v72
	; iglp_opt mask(0x00000002)
	v_add_u32_e32 v8, 0x1800, v8
	ds_bpermute_b32 v9, v51, v9
	s_add_i32 s3, s10, s16
	; sched_barrier mask(0x00000000)
	ds_bpermute_b32 v8, v51, v8
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v9, 1, v9
	v_cndmask_b32_e64 v9, v42, v9, s[0:1]
	s_mov_b32 m0, s3
	v_add_u32_e32 v12, s2, v62
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_load_dwordx4 v9, s[4:7], 0 offen lds
	s_add_i32 m0, s3, 0x1000
	v_cndmask_b32_e64 v8, v42, v8, s[0:1]
	buffer_load_dwordx4 v8, s[4:7], 0 offen lds
	ds_read_b128 v[8:11], v12
	v_add_u32_e32 v53, s2, v61
	ds_read_b128 v[80:83], v53
	ds_read_b128 v[84:87], v53 offset:4096
	v_add_u32_e32 v53, s2, v60
	ds_read_b128 v[12:15], v12 offset:4096
	ds_read_b128 v[88:91], v53
	.loc	1 218 23 is_stmt 1              ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x16_f16 a[0:15], v[8:11], v[28:31], 0
	.loc	1 216 20                        ; flash_attention.py:216:20
	ds_read_b128 v[92:95], v53 offset:4096
	v_add_u32_e32 v53, s2, v59
	ds_read_b128 v[96:99], v53
	ds_read_b128 v[100:103], v53 offset:4096
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_add_i32 s24, s10, s19
	v_cmp_eq_u32_e64 s[0:1], 1, v74
	s_add_i32 m0, s24, 0x4000
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_32x32x16_f16 a[0:15], v[80:83], v[24:27], a[0:15]
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	v_cmp_eq_u32_e64 s[2:3], 1, v75
	s_add_i32 s23, s10, 0x4000
	.loc	1 210 48                        ; flash_attention.py:210:48
	s_add_i32 s20, s20, 64
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 a[0:15], v[88:91], v[20:23], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[96:99], v[16:19], a[0:15]
	s_nop 11
	v_accvgpr_read_b32 v8, a0
	v_accvgpr_read_b32 v9, a1
	v_accvgpr_read_b32 v10, a2
	v_accvgpr_read_b32 v11, a3
	v_accvgpr_read_b32 v53, a4
	v_accvgpr_read_b32 v79, a5
	v_accvgpr_read_b32 v80, a6
	v_accvgpr_read_b32 v81, a7
	v_accvgpr_read_b32 v82, a8
	v_accvgpr_read_b32 v83, a9
	v_accvgpr_read_b32 v88, a10
	v_accvgpr_read_b32 v89, a11
	v_accvgpr_read_b32 v90, a12
	v_accvgpr_read_b32 v91, a13
	v_accvgpr_read_b32 v96, a14
	v_accvgpr_read_b32 v97, a15
	v_mfma_f32_32x32x16_f16 a[0:15], v[12:15], v[28:31], 0
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v8, v52, v8
	v_mul_f32_e32 v9, v52, v9
	v_mul_f32_e32 v10, v52, v10
	v_mul_f32_e32 v11, v52, v11
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v9, v73, v9, vcc
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v53, v52, v53
	v_mul_f32_e32 v79, v52, v79
	.loc	1 218 23 is_stmt 0              ; flash_attention.py:218:23
	v_mfma_f32_32x32x16_f16 a[0:15], v[84:87], v[24:27], a[0:15]
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v10, v73, v10, vcc
	v_cndmask_b32_e32 v11, v73, v11, vcc
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v80, v52, v80
	v_mul_f32_e32 v81, v52, v81
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v53, v73, v53, vcc
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v82, v52, v82
	v_mul_f32_e32 v83, v52, v83
	.loc	1 218 23 is_stmt 0              ; flash_attention.py:218:23
	v_mfma_f32_32x32x16_f16 a[0:15], v[92:95], v[20:23], a[0:15]
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v80, v73, v80, vcc
	v_cndmask_b32_e32 v81, v73, v81, vcc
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v88, v52, v88
	v_mul_f32_e32 v89, v52, v89
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v82, v73, v82, vcc
	v_cndmask_b32_e32 v83, v73, v83, vcc
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v90, v52, v90
	.loc	1 218 23 is_stmt 0              ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[100:103], v[16:19], a[0:15]
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v102, v73, v8, vcc
.Ltmp2:
	.file	2 "/home/kunwar/Work/runtime-evolution/rocm-systems-rocjitsu-register-feedback/emulation/rocjitsu/.env/lib/python3.12/site-packages/triton/language" "standard.py"
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max_f32_e32 v8, v102, v9
.Ltmp3:
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v103, v73, v79, vcc
.Ltmp4:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v10, v11
	v_max3_f32 v8, v8, v53, v103
	v_max3_f32 v8, v8, v80, v81
.Ltmp5:
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v91, v52, v91
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v88, v73, v88, vcc
	v_cndmask_b32_e32 v89, v73, v89, vcc
.Ltmp6:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v82, v83
.Ltmp7:
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v96, v52, v96
	v_mul_f32_e32 v97, v52, v97
	.loc	1 218 23 is_stmt 0              ; flash_attention.py:218:23
	v_accvgpr_read_b32 v12, a0
	v_accvgpr_read_b32 v13, a1
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v90, v73, v90, vcc
	v_cndmask_b32_e32 v91, v73, v91, vcc
.Ltmp8:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v88, v89
.Ltmp9:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v14, a2
	v_accvgpr_read_b32 v15, a3
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v12, v52, v12
	v_mul_f32_e32 v13, v52, v13
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v96, v73, v96, vcc
	v_cndmask_b32_e32 v97, v73, v97, vcc
.Ltmp10:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v90, v91
.Ltmp11:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v84, a4
	v_accvgpr_read_b32 v85, a5
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v14, v52, v14
	v_mul_f32_e32 v15, v52, v15
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v12, v73, v12, vcc
	v_cndmask_b32_e32 v13, v73, v13, vcc
.Ltmp12:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v96, v97
.Ltmp13:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v86, a6
	v_accvgpr_read_b32 v87, a7
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v84, v52, v84
	v_mul_f32_e32 v85, v52, v85
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v14, v73, v14, vcc
	v_cndmask_b32_e32 v15, v73, v15, vcc
.Ltmp14:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v12, v13
.Ltmp15:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v92, a8
	v_accvgpr_read_b32 v93, a9
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v86, v52, v86
	v_mul_f32_e32 v87, v52, v87
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v84, v73, v84, vcc
	v_cndmask_b32_e32 v85, v73, v85, vcc
.Ltmp16:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v14, v15
.Ltmp17:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v94, a10
	v_accvgpr_read_b32 v95, a11
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v92, v52, v92
	v_mul_f32_e32 v93, v52, v93
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v86, v73, v86, vcc
	v_cndmask_b32_e32 v87, v73, v87, vcc
.Ltmp18:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v84, v85
.Ltmp19:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v98, a12
	v_accvgpr_read_b32 v99, a13
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v94, v52, v94
	v_mul_f32_e32 v95, v52, v95
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v92, v73, v92, vcc
	v_cndmask_b32_e32 v93, v73, v93, vcc
.Ltmp20:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v86, v87
.Ltmp21:
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v100, a14
	v_accvgpr_read_b32 v101, a15
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v98, v52, v98
	v_mul_f32_e32 v99, v52, v99
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_cndmask_b32_e32 v94, v73, v94, vcc
	v_cndmask_b32_e32 v95, v73, v95, vcc
.Ltmp22:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v92, v93
.Ltmp23:
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v100, v52, v100
	v_mul_f32_e32 v101, v52, v101
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v98, v73, v98, vcc
	v_cndmask_b32_e32 v99, v73, v99, vcc
.Ltmp24:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v94, v95
.Ltmp25:
	.loc	1 220 67                        ; flash_attention.py:220:67
	v_cndmask_b32_e32 v100, v73, v100, vcc
	v_cndmask_b32_e32 v101, v73, v101, vcc
.Ltmp26:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max3_f32 v8, v8, v98, v99
	v_max3_f32 v8, v8, v100, v101
.Ltmp27:
	.loc	2 191 40                        ; standard.py:191:40 @[ flash_attention.py:223:39 ]
	v_mov_b32_e32 v79, v8
	s_nop 1
	v_permlane32_swap_b32_e32 v8, v79
.Ltmp28:
	.loc	1 223 32                        ; flash_attention.py:223:32
	v_max3_f32 v79, v104, v8, v79
	.loc	1 224 35                        ; flash_attention.py:224:35
	v_sub_f32_e32 v8, v104, v79
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v102, v102, v79
	v_sub_f32_e32 v104, v9, v79
	v_sub_f32_e32 v106, v10, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v9, v102
	v_exp_f32_e32 v10, v104
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v107, v11, v79
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v11, v106
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v53, v53, v79
	v_sub_f32_e32 v108, v12, v79
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v12, v107
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v103, v103, v79
	v_sub_f32_e32 v109, v13, v79
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v13, v53
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v80, v80, v79
	v_sub_f32_e32 v110, v14, v79
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v14, v103
.Ltmp29:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v9, v10
.Ltmp30:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v81, v81, v79
	v_sub_f32_e32 v111, v15, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v15, v80
.Ltmp31:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v11, v53
.Ltmp32:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v82, v82, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v80, v81
.Ltmp33:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v12, v53
.Ltmp34:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v83, v83, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v81, v82
.Ltmp35:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v13, v53
.Ltmp36:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v88, v88, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v82, v83
.Ltmp37:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v14, v53
.Ltmp38:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v89, v89, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v83, v88
.Ltmp39:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v15, v53
.Ltmp40:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v90, v90, v79
	v_sub_f32_e32 v112, v84, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v84, v89
.Ltmp41:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v80, v53
.Ltmp42:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v91, v91, v79
	v_sub_f32_e32 v113, v85, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v85, v90
.Ltmp43:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v81, v53
.Ltmp44:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v96, v96, v79
	v_sub_f32_e32 v114, v86, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v86, v91
.Ltmp45:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v82, v53
.Ltmp46:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v97, v97, v79
	v_sub_f32_e32 v115, v87, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v87, v96
.Ltmp47:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v83, v53
.Ltmp48:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v88, v97
.Ltmp49:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v84, v53
.Ltmp50:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v89, v108
.Ltmp51:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v85, v53
.Ltmp52:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v90, v109
.Ltmp53:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v86, v53
.Ltmp54:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v91, v110
.Ltmp55:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v87, v53
.Ltmp56:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v116, v92, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v92, v111
.Ltmp57:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v88, v53
.Ltmp58:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v117, v93, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v93, v112
.Ltmp59:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v89, v53
.Ltmp60:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v118, v94, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v94, v113
.Ltmp61:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v90, v53
.Ltmp62:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v119, v95, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v95, v114
.Ltmp63:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v91, v53
.Ltmp64:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v96, v115
.Ltmp65:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v92, v53
.Ltmp66:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v97, v116
.Ltmp67:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v93, v53
.Ltmp68:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v120, v98, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v98, v117
.Ltmp69:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v94, v53
.Ltmp70:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v121, v99, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v99, v118
.Ltmp71:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v95, v53
.Ltmp72:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v122, v100, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v100, v119
.Ltmp73:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v96, v53
.Ltmp74:
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v123, v101, v79
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v101, v120
.Ltmp75:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v97, v53
.Ltmp76:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v102, v121
.Ltmp77:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v98, v53
.Ltmp78:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v103, v122
.Ltmp79:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v99, v53
.Ltmp80:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v104, v123
.Ltmp81:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v100, v53
	v_add_f32_e32 v53, v101, v53
	v_add_f32_e32 v53, v102, v53
	v_add_f32_e32 v53, v103, v53
.Ltmp82:
	.loc	1 224 29                        ; flash_attention.py:224:29
	v_exp_f32_e32 v8, v8
.Ltmp83:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v104, v53
.Ltmp84:
	.loc	2 293 36                        ; standard.py:293:36 @[ flash_attention.py:226:37 ]
	v_mov_b32_e32 v106, v53
	s_nop 1
	v_permlane32_swap_b32_e32 v53, v106
.Ltmp85:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v53, v53, v106
.Ltmp86:
	.loc	1 226 30                        ; flash_attention.py:226:30
	v_fmac_f32_e32 v53, v105, v8
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_add_u32_e32 v105, s18, v50
	v_add_u32_e32 v106, 0x1000, v105
	.loc	1 230 20 is_stmt 0              ; flash_attention.py:230:20
	ds_bpermute_b32 v106, v54, v106
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_add_u32_e32 v107, 0x1200, v105
	v_add_u32_e32 v108, 0x1400, v105
	v_add_u32_e32 v109, s18, v71
	v_add_u32_e32 v110, 0x1800, v105
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[0:1]
	buffer_load_dword v106, s[12:15], 0 offen lds
	ds_bpermute_b32 v106, v55, v107
	s_add_i32 m0, s24, 0x4400
	.loc	1 230 28                        ; flash_attention.py:230:28
	v_add_u32_e32 v111, 0x1a00, v105
	v_add_u32_e32 v105, 0x1c00, v105
	.loc	1 230 20                        ; flash_attention.py:230:20
	ds_bpermute_b32 v105, v54, v105
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[2:3]
	buffer_load_dword v106, s[12:15], 0 offen lds
	ds_bpermute_b32 v106, v54, v108
	s_add_i32 m0, s24, 0x4800
	v_cmp_eq_u32_e64 s[2:3], 1, v76
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b32_e32 v105, 1, v105
	v_add_u32_e32 v112, s18, v70
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[0:1]
	buffer_load_dword v106, s[12:15], 0 offen lds
	ds_bpermute_b32 v106, v56, v109
	s_add_i32 m0, s24, 0x4c00
	v_cndmask_b32_e64 v105, v42, v105, s[0:1]
	.loc	1 210 48 is_stmt 1              ; flash_attention.py:210:48
	s_addk_i32 s18, 0x1000
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[2:3]
	buffer_load_dword v106, s[12:15], 0 offen lds
	ds_bpermute_b32 v106, v54, v110
	s_add_i32 m0, s24, 0x5000
	v_cmp_eq_u32_e64 s[2:3], 1, v77
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[0:1]
	buffer_load_dword v106, s[12:15], 0 offen lds
	ds_bpermute_b32 v106, v57, v111
	s_add_i32 m0, s24, 0x5400
	v_cmp_eq_u32_e64 s[0:1], 1, v78
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v106, 1, v106
	v_cndmask_b32_e64 v106, v42, v106, s[2:3]
	buffer_load_dword v106, s[12:15], 0 offen lds
	s_add_i32 m0, s24, 0x5800
	s_nop 0
	buffer_load_dword v105, s[12:15], 0 offen lds
	ds_bpermute_b32 v105, v58, v112
	s_add_i32 m0, s24, 0x5c00
	.loc	1 210 48                        ; flash_attention.py:210:48
	s_cmpk_lt_u32 s20, 0x380
	.loc	1 230 20                        ; flash_attention.py:230:20
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v105, 1, v105
	v_cndmask_b32_e64 v105, v42, v105, s[0:1]
	buffer_load_dword v105, s[12:15], 0 offen lds
	v_add_u32_e32 v105, s22, v48
	ds_read_b64_tr_b16 v[108:109], v105
	ds_read_b64_tr_b16 v[112:113], v105 offset:2048
	ds_read_b64_tr_b16 v[116:117], v105 offset:4096
	ds_read_b64_tr_b16 v[120:121], v105 offset:6144
	v_add_u32_e32 v105, s22, v49
	ds_read_b64_tr_b16 v[110:111], v105 offset:1024
	ds_read_b64_tr_b16 v[114:115], v105 offset:3072
	ds_read_b64_tr_b16 v[118:119], v105 offset:5120
	ds_read_b64_tr_b16 v[122:123], v105 offset:7168
	.loc	1 232 20                        ; flash_attention.py:232:20
	v_add_u32_e32 v105, v63, v44
	s_waitcnt vmcnt(0)
	ds_write_b32 v105, v8 offset:32768
	v_add_u32_e32 v8, v63, v40
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v106, v8 offset:32768
	.loc	1 233 26                        ; flash_attention.py:233:26
	v_cvt_pk_f16_f32 v8, v9, v10
	v_cvt_pk_f16_f32 v9, v11, v12
	v_cvt_pk_f16_f32 v11, v15, v80
	v_cvt_pk_f16_f32 v12, v81, v82
	v_cvt_pk_f16_f32 v81, v89, v90
	v_add_u32_e32 v80, 0, v64
	v_cvt_pk_f16_f32 v82, v91, v92
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v80, v81 offset:36864
	ds_write_b16_d16_hi v80, v81 offset:36992
	v_add_u32_e32 v81, 0, v65
	v_cvt_pk_f16_f32 v10, v13, v14
	v_cvt_pk_f16_f32 v13, v83, v84
	v_cvt_pk_f16_f32 v83, v93, v94
	ds_write_b16 v81, v82 offset:37120
	ds_write_b16_d16_hi v81, v82 offset:37248
	v_add_u32_e32 v82, 0, v66
	v_cvt_pk_f16_f32 v14, v85, v86
	v_cvt_pk_f16_f32 v84, v95, v96
	v_cvt_pk_f16_f32 v85, v97, v98
	ds_write_b16 v82, v83 offset:37888
	ds_write_b16_d16_hi v82, v83 offset:38016
	v_add_u32_e32 v83, 0, v67
	v_cvt_pk_f16_f32 v15, v87, v88
	v_cvt_pk_f16_f32 v86, v99, v100
	v_cvt_pk_f16_f32 v87, v101, v102
	v_cvt_pk_f16_f32 v88, v103, v104
	ds_write_b16 v80, v85 offset:38912
	ds_write_b16_d16_hi v80, v85 offset:39040
	ds_write_b16 v83, v84 offset:38144
	ds_write_b16_d16_hi v83, v84 offset:38272
	v_add_u32_e32 v84, 0, v68
	v_add_u32_e32 v85, 0, v69
	ds_write_b16 v80, v8 offset:32768
	ds_write_b16_d16_hi v80, v8 offset:32896
	ds_write_b16 v80, v12 offset:34816
	ds_write_b16_d16_hi v80, v12 offset:34944
	ds_write_b16 v81, v9 offset:33024
	ds_write_b16_d16_hi v81, v9 offset:33152
	ds_write_b16 v81, v13 offset:35072
	ds_write_b16_d16_hi v81, v13 offset:35200
	ds_write_b16 v81, v86 offset:39168
	ds_write_b16_d16_hi v81, v86 offset:39296
	ds_write_b16 v82, v10 offset:33792
	ds_write_b16_d16_hi v82, v10 offset:33920
	ds_write_b16 v82, v14 offset:35840
	ds_write_b16_d16_hi v82, v14 offset:35968
	ds_write_b16 v82, v87 offset:39936
	ds_write_b16_d16_hi v82, v87 offset:40064
	ds_write_b16 v83, v11 offset:34048
	ds_write_b16_d16_hi v83, v11 offset:34176
	ds_write_b16 v83, v15 offset:36096
	ds_write_b16_d16_hi v83, v15 offset:36224
	ds_write_b16 v83, v88 offset:40192
	ds_write_b16_d16_hi v83, v88 offset:40320
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[86:87], v84 offset:32768
	ds_read_b64_tr_b16 v[90:91], v84 offset:34816
	ds_read_b64_tr_b16 v[94:95], v84 offset:36864
	ds_read_b64_tr_b16 v[98:99], v84 offset:38912
	ds_read_b64_tr_b16 v[88:89], v85 offset:33792
	ds_read_b64_tr_b16 v[92:93], v85 offset:35840
	ds_read_b64_tr_b16 v[96:97], v85 offset:37888
	ds_read_b64_tr_b16 v[100:101], v85 offset:39936
	.loc	1 233 42 is_stmt 0              ; flash_attention.py:233:42
	v_pk_mul_f32 v[14:15], v[0:1], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[38:39], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[2:3], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[4:5], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[6:7], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[32:33], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[34:35], v[106:107] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[36:37], v[106:107] op_sel_hi:[1,0]
	s_mov_b32 s22, s23
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
	s_waitcnt lgkmcnt(3)
	s_nop 0
	v_mfma_f32_32x32x16_f16 a[0:15], v[108:111], v[86:89], a[0:15]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[112:115], v[90:93], a[0:15]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[116:119], v[94:97], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[120:123], v[98:101], a[0:15]
	.loc	1 210 48 is_stmt 1              ; flash_attention.py:210:48
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	.loc	1 216 20                        ; flash_attention.py:216:20
	v_add_u32_e32 v4, s10, v62
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[0:3], v4
	v_add_u32_e32 v5, s10, v61
	ds_read_b128 v[32:35], v4 offset:4096
	v_add_u32_e32 v4, s10, v60
	ds_read_b128 v[36:39], v5 offset:4096
	ds_read_b128 v[54:57], v4 offset:4096
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 a[16:31], v[0:3], v[28:31], 0
	.loc	1 216 20                        ; flash_attention.py:216:20
	ds_read_b128 v[0:3], v5
	v_add_u32_e32 v5, s10, v59
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_lshrrev_b32_e32 v41, 2, v40
	s_movk_i32 s0, 0x400
	.loc	1 240 32                        ; flash_attention.py:240:32
	s_and_b32 s9, s9, 0xffff
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[16:31], v[0:3], v[24:27], a[16:31]
	.loc	1 216 20                        ; flash_attention.py:216:20
	ds_read_b128 v[0:3], v4
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[16:31], v[0:3], v[20:23], a[16:31]
	.loc	1 216 20                        ; flash_attention.py:216:20
	ds_read_b128 v[0:3], v5
	ds_read_b128 v[58:61], v5 offset:4096
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[16:31], v[0:3], v[16:19], a[16:31]
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v0, a0
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
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_mfma_f32_32x32x16_f16 a[0:15], v[32:35], v[28:31], 0
	.loc	1 194 45                        ; flash_attention.py:194:45
	v_lshrrev_b32_e32 v28, 3, v46
	v_or_b32_e32 v30, v41, v45
	.loc	1 198 53                        ; flash_attention.py:198:53
	v_or_b32_e32 v29, v28, v47
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v31, a16
	v_accvgpr_read_b32 v32, a17
	v_accvgpr_read_b32 v33, a22
	v_accvgpr_read_b32 v34, a23
	v_mfma_f32_32x32x16_f16 a[0:15], v[36:39], v[24:27], a[0:15]
	v_accvgpr_read_b32 v24, a18
	v_accvgpr_read_b32 v25, a19
	v_accvgpr_read_b32 v26, a20
	v_accvgpr_read_b32 v27, a21
	v_accvgpr_read_b32 v35, a24
	v_accvgpr_read_b32 v36, a29
	v_accvgpr_read_b32 v37, a30
	v_mfma_f32_32x32x16_f16 a[0:15], v[54:57], v[20:23], a[0:15]
	v_accvgpr_read_b32 v20, a25
	v_accvgpr_read_b32 v21, a26
	v_accvgpr_read_b32 v22, a27
	v_accvgpr_read_b32 v23, a28
	v_accvgpr_read_b32 v38, a31
	.loc	1 218 50 is_stmt 0              ; flash_attention.py:218:50
	v_mul_f32_e32 v31, v52, v31
	v_mul_f32_e32 v32, v52, v32
	.loc	1 218 23                        ; flash_attention.py:218:23
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x16_f16 a[0:15], v[58:61], v[16:19], a[0:15]
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v24, v52, v24
	v_mul_f32_e32 v25, v52, v25
	v_mul_f32_e32 v26, v52, v26
	v_mul_f32_e32 v27, v52, v27
	v_mul_f32_e32 v33, v52, v33
	v_mul_f32_e32 v34, v52, v34
	v_mul_f32_e32 v35, v52, v35
	v_mul_f32_e32 v20, v52, v20
	v_mul_f32_e32 v21, v52, v21
	v_mul_f32_e32 v22, v52, v22
	v_mul_f32_e32 v23, v52, v23
	v_mul_f32_e32 v36, v52, v36
	.loc	1 218 23                        ; flash_attention.py:218:23
	v_accvgpr_read_b32 v16, a0
	v_accvgpr_read_b32 v17, a1
	v_accvgpr_read_b32 v18, a2
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v39, a4
	v_accvgpr_read_b32 v41, a5
	v_accvgpr_read_b32 v42, a6
	v_accvgpr_read_b32 v45, a7
	v_accvgpr_read_b32 v46, a8
	v_accvgpr_read_b32 v47, a9
	v_accvgpr_read_b32 v50, a10
	v_accvgpr_read_b32 v51, a11
	v_accvgpr_read_b32 v54, a12
	v_accvgpr_read_b32 v55, a13
	v_accvgpr_read_b32 v56, a14
	v_accvgpr_read_b32 v57, a15
	.loc	1 218 50                        ; flash_attention.py:218:50
	v_mul_f32_e32 v37, v52, v37
	v_mul_f32_e32 v38, v52, v38
	v_mul_f32_e32 v16, v52, v16
	v_mul_f32_e32 v17, v52, v17
	v_mul_f32_e32 v18, v52, v18
	v_mul_f32_e32 v19, v52, v19
	v_mul_f32_e32 v39, v52, v39
	v_mul_f32_e32 v41, v52, v41
	v_mul_f32_e32 v42, v52, v42
	v_mul_f32_e32 v45, v52, v45
	v_mul_f32_e32 v46, v52, v46
	v_mul_f32_e32 v47, v52, v47
	v_mul_f32_e32 v50, v52, v50
	v_mul_f32_e32 v51, v52, v51
	v_mul_f32_e32 v54, v52, v54
	v_mul_f32_e32 v55, v52, v55
	v_mul_f32_e32 v56, v52, v56
	v_mul_f32_e32 v52, v52, v57
	.loc	1 220 67 is_stmt 1              ; flash_attention.py:220:67
	v_mov_b32_e32 v57, 0xe0ad78ec
	v_cndmask_b32_e32 v31, v57, v31, vcc
	v_cndmask_b32_e32 v32, v57, v32, vcc
	v_cndmask_b32_e32 v24, v57, v24, vcc
	v_cndmask_b32_e32 v25, v57, v25, vcc
	v_cndmask_b32_e32 v26, v57, v26, vcc
	v_cndmask_b32_e32 v27, v57, v27, vcc
	v_cndmask_b32_e32 v33, v57, v33, vcc
	v_cndmask_b32_e32 v34, v57, v34, vcc
	v_cndmask_b32_e32 v35, v57, v35, vcc
	v_cndmask_b32_e32 v20, v57, v20, vcc
	v_cndmask_b32_e32 v21, v57, v21, vcc
	v_cndmask_b32_e32 v22, v57, v22, vcc
	v_cndmask_b32_e32 v23, v57, v23, vcc
	v_cndmask_b32_e32 v36, v57, v36, vcc
	v_cndmask_b32_e32 v37, v57, v37, vcc
	v_cndmask_b32_e32 v38, v57, v38, vcc
	v_cndmask_b32_e32 v16, v57, v16, vcc
	v_cndmask_b32_e32 v17, v57, v17, vcc
	v_cndmask_b32_e32 v18, v57, v18, vcc
	v_cndmask_b32_e32 v19, v57, v19, vcc
	v_cndmask_b32_e32 v39, v57, v39, vcc
	v_cndmask_b32_e32 v41, v57, v41, vcc
	v_cndmask_b32_e32 v42, v57, v42, vcc
	v_cndmask_b32_e32 v45, v57, v45, vcc
	v_cndmask_b32_e32 v46, v57, v46, vcc
	v_cndmask_b32_e32 v47, v57, v47, vcc
	v_cndmask_b32_e32 v50, v57, v50, vcc
	v_cndmask_b32_e32 v51, v57, v51, vcc
	v_cndmask_b32_e32 v54, v57, v54, vcc
	v_cndmask_b32_e32 v55, v57, v55, vcc
	v_cndmask_b32_e32 v56, v57, v56, vcc
	v_cndmask_b32_e32 v52, v57, v52, vcc
.Ltmp87:
	.loc	2 170 27                        ; standard.py:170:27 @[ standard.py:191:40 @[ flash_attention.py:223:39 ] ]
	v_max_f32_e32 v57, v31, v32
	v_max3_f32 v57, v57, v24, v25
	v_max3_f32 v57, v57, v26, v27
	v_max3_f32 v57, v57, v33, v34
	v_max3_f32 v57, v57, v35, v20
	v_max3_f32 v57, v57, v21, v22
	v_max3_f32 v57, v57, v23, v36
	v_max3_f32 v57, v57, v37, v38
	v_max3_f32 v57, v57, v16, v17
	v_max3_f32 v57, v57, v18, v19
	v_max3_f32 v57, v57, v39, v41
	v_max3_f32 v57, v57, v42, v45
	v_max3_f32 v57, v57, v46, v47
	v_max3_f32 v57, v57, v50, v51
	v_max3_f32 v57, v57, v54, v55
	v_max3_f32 v57, v57, v56, v52
.Ltmp88:
	.loc	2 191 40                        ; standard.py:191:40 @[ flash_attention.py:223:39 ]
	v_mov_b32_e32 v58, v57
	s_nop 1
	v_permlane32_swap_b32_e32 v57, v58
.Ltmp89:
	.loc	1 223 32                        ; flash_attention.py:223:32
	v_max3_f32 v57, v79, v57, v58
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v31, v31, v57
	v_sub_f32_e32 v32, v32, v57
	v_sub_f32_e32 v24, v24, v57
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v25, v25, v57
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v24, v24
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v26, v26, v57
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v25, v25
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v27, v27, v57
	v_sub_f32_e32 v16, v16, v57
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v26, v26
	.loc	1 224 35 is_stmt 1              ; flash_attention.py:224:35
	v_sub_f32_e32 v58, v79, v57
	.loc	1 225 30                        ; flash_attention.py:225:30
	v_sub_f32_e32 v33, v33, v57
	v_sub_f32_e32 v34, v34, v57
	v_sub_f32_e32 v35, v35, v57
	v_sub_f32_e32 v20, v20, v57
	v_sub_f32_e32 v21, v21, v57
	v_sub_f32_e32 v22, v22, v57
	v_sub_f32_e32 v23, v23, v57
	v_sub_f32_e32 v36, v36, v57
	v_sub_f32_e32 v37, v37, v57
	v_sub_f32_e32 v38, v38, v57
	v_sub_f32_e32 v17, v17, v57
	v_sub_f32_e32 v18, v18, v57
	v_sub_f32_e32 v19, v19, v57
	v_sub_f32_e32 v39, v39, v57
	v_sub_f32_e32 v41, v41, v57
	v_sub_f32_e32 v42, v42, v57
	v_sub_f32_e32 v45, v45, v57
	v_sub_f32_e32 v46, v46, v57
	v_sub_f32_e32 v47, v47, v57
	v_sub_f32_e32 v50, v50, v57
	v_sub_f32_e32 v51, v51, v57
	v_sub_f32_e32 v54, v54, v57
	v_sub_f32_e32 v55, v55, v57
	v_sub_f32_e32 v56, v56, v57
	v_sub_f32_e32 v52, v52, v57
	.loc	1 225 25 is_stmt 0              ; flash_attention.py:225:25
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v57, v16
.Ltmp90:
	.loc	2 263 15 is_stmt 1              ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v31, v32
.Ltmp91:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v33, v33
.Ltmp92:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v24, v16
.Ltmp93:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v34, v34
.Ltmp94:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v25, v16
.Ltmp95:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v35, v35
.Ltmp96:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v26, v16
.Ltmp97:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v20, v20
.Ltmp98:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v27, v16
.Ltmp99:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v21, v21
.Ltmp100:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v33, v16
.Ltmp101:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v22, v22
.Ltmp102:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v34, v16
.Ltmp103:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v23, v23
.Ltmp104:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v35, v16
.Ltmp105:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v36, v36
.Ltmp106:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v20, v16
.Ltmp107:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v37, v37
.Ltmp108:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v21, v16
.Ltmp109:
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v38, v38
.Ltmp110:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v16, v22, v16
	v_add_f32_e32 v16, v23, v16
	v_add_f32_e32 v16, v36, v16
	v_add_f32_e32 v16, v37, v16
	v_add_f32_e32 v59, v38, v16
.Ltmp111:
	.loc	1 230 20                        ; flash_attention.py:230:20
	v_add_u32_e32 v16, s10, v48
	.loc	1 224 29                        ; flash_attention.py:224:29
	v_exp_f32_e32 v58, v58
	.loc	1 230 20                        ; flash_attention.py:230:20
	ds_read_b64_tr_b16 v[60:61], v16 offset:16384
	ds_read_b64_tr_b16 v[64:65], v16 offset:18432
	ds_read_b64_tr_b16 v[68:69], v16 offset:20480
	ds_read_b64_tr_b16 v[72:73], v16 offset:22528
	v_add_u32_e32 v16, s10, v49
	ds_read_b64_tr_b16 v[62:63], v16 offset:17408
	ds_read_b64_tr_b16 v[66:67], v16 offset:19456
	ds_read_b64_tr_b16 v[70:71], v16 offset:21504
	ds_read_b64_tr_b16 v[74:75], v16 offset:23552
	.loc	1 232 20                        ; flash_attention.py:232:20
	v_or_b32_e32 v16, v43, v44
	v_add_u32_e32 v44, 0, v16
	v_or_b32_e32 v16, v43, v40
	v_add_u32_e32 v40, 0, v16
	ds_write_b32 v44, v58 offset:32768
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v16, v40 offset:32768
	.loc	1 225 25                        ; flash_attention.py:225:25
	v_exp_f32_e32 v17, v17
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v39, v39
	v_exp_f32_e32 v41, v41
	v_exp_f32_e32 v42, v42
	v_exp_f32_e32 v45, v45
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v50, v50
	v_exp_f32_e32 v51, v51
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v52, v52
	.loc	1 233 26                        ; flash_attention.py:233:26
	v_cvt_pk_f16_f32 v31, v31, v32
	v_cvt_pk_f16_f32 v24, v24, v25
	v_cvt_pk_f16_f32 v25, v26, v27
	v_cvt_pk_f16_f32 v26, v33, v34
	v_cvt_pk_f16_f32 v20, v35, v20
	v_cvt_pk_f16_f32 v21, v21, v22
	v_cvt_pk_f16_f32 v22, v23, v36
	v_cvt_pk_f16_f32 v23, v37, v38
	v_cvt_pk_f16_f32 v27, v57, v17
	v_cvt_pk_f16_f32 v32, v18, v19
	v_cvt_pk_f16_f32 v33, v39, v41
	v_cvt_pk_f16_f32 v34, v42, v45
	v_cvt_pk_f16_f32 v35, v46, v47
	.loc	1 233 42 is_stmt 0              ; flash_attention.py:233:42
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[0:1], v[0:1], v[16:17] op_sel_hi:[1,0]
	.loc	1 233 26                        ; flash_attention.py:233:26
	v_cvt_pk_f16_f32 v36, v50, v51
	v_cvt_pk_f16_f32 v37, v54, v55
	v_cvt_pk_f16_f32 v38, v56, v52
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v80, v31 offset:32768
	ds_write_b16_d16_hi v80, v31 offset:32896
	ds_write_b16 v80, v20 offset:34816
	ds_write_b16_d16_hi v80, v20 offset:34944
	ds_write_b16 v80, v27 offset:36864
	ds_write_b16_d16_hi v80, v27 offset:36992
	ds_write_b16 v80, v35 offset:38912
	ds_write_b16_d16_hi v80, v35 offset:39040
	ds_write_b16 v81, v24 offset:33024
	ds_write_b16_d16_hi v81, v24 offset:33152
	ds_write_b16 v81, v21 offset:35072
	ds_write_b16_d16_hi v81, v21 offset:35200
	ds_write_b16 v81, v32 offset:37120
	ds_write_b16_d16_hi v81, v32 offset:37248
	ds_write_b16 v81, v36 offset:39168
	ds_write_b16_d16_hi v81, v36 offset:39296
	ds_write_b16 v82, v25 offset:33792
	ds_write_b16_d16_hi v82, v25 offset:33920
	ds_write_b16 v82, v22 offset:35840
	ds_write_b16_d16_hi v82, v22 offset:35968
	ds_write_b16 v82, v33 offset:37888
	ds_write_b16_d16_hi v82, v33 offset:38016
	ds_write_b16 v82, v37 offset:39936
	ds_write_b16_d16_hi v82, v37 offset:40064
	ds_write_b16 v83, v26 offset:34048
	ds_write_b16_d16_hi v83, v26 offset:34176
	ds_write_b16 v83, v23 offset:36096
	ds_write_b16_d16_hi v83, v23 offset:36224
	ds_write_b16 v83, v34 offset:38144
	ds_write_b16_d16_hi v83, v34 offset:38272
	ds_write_b16 v83, v38 offset:40192
	ds_write_b16_d16_hi v83, v38 offset:40320
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b16 v[20:21], v84 offset:32768
	ds_read_b64_tr_b16 v[24:25], v84 offset:34816
	ds_read_b64_tr_b16 v[32:33], v84 offset:36864
	ds_read_b64_tr_b16 v[76:77], v84 offset:38912
	ds_read_b64_tr_b16 v[22:23], v85 offset:33792
	ds_read_b64_tr_b16 v[26:27], v85 offset:35840
	ds_read_b64_tr_b16 v[34:35], v85 offset:37888
	ds_read_b64_tr_b16 v[78:79], v85 offset:39936
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_pk_mul_f32 v[14:15], v[14:15], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[16:17] op_sel_hi:[1,0]
	.loc	1 194 32 is_stmt 1              ; flash_attention.py:194:32
	v_or_b32_e32 v28, s17, v30
	.loc	1 233 42                        ; flash_attention.py:233:42
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
.Ltmp112:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v0, v57, v59
	v_add_f32_e32 v0, v17, v0
.Ltmp113:
	.loc	1 233 42                        ; flash_attention.py:233:42
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_32x32x16_f16 a[0:15], v[60:63], v[20:23], a[0:15]
.Ltmp114:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v0, v18, v0
	v_add_f32_e32 v0, v19, v0
	v_add_f32_e32 v0, v39, v0
	v_add_f32_e32 v0, v41, v0
	v_add_f32_e32 v0, v42, v0
	v_add_f32_e32 v0, v45, v0
	v_add_f32_e32 v0, v46, v0
.Ltmp115:
	.loc	1 233 42                        ; flash_attention.py:233:42
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x16_f16 a[0:15], v[64:67], v[24:27], a[0:15]
.Ltmp116:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v0, v47, v0
	v_add_f32_e32 v0, v50, v0
	v_add_f32_e32 v0, v51, v0
	v_add_f32_e32 v0, v54, v0
	v_add_f32_e32 v0, v55, v0
	v_add_f32_e32 v0, v56, v0
	v_add_f32_e32 v0, v52, v0
.Ltmp117:
	.loc	1 233 42                        ; flash_attention.py:233:42
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x16_f16 a[0:15], v[68:71], v[32:35], a[0:15]
.Ltmp118:
	.loc	2 293 36                        ; standard.py:293:36 @[ flash_attention.py:226:37 ]
	v_mov_b32_e32 v1, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v0, v1
.Ltmp119:
	.loc	2 263 15                        ; standard.py:263:15 @[ standard.py:293:36 @[ flash_attention.py:226:37 ] ]
	v_add_f32_e32 v0, v0, v1
.Ltmp120:
	.loc	1 226 30                        ; flash_attention.py:226:30
	v_fmac_f32_e32 v0, v53, v58
	.loc	1 238 16                        ; flash_attention.py:238:16
	ds_write_b32 v44, v0
	s_waitcnt lgkmcnt(0)
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_mfma_f32_32x32x16_f16 a[0:15], v[72:75], v[76:79], a[0:15]
	.loc	1 238 16                        ; flash_attention.py:238:16
	s_barrier
	ds_read_b32 v15, v40
	s_mov_b32 s10, 0x7ffffffe
	.loc	1 233 42                        ; flash_attention.py:233:42
	s_nop 8
	v_accvgpr_read_b32 v1, a0
	.loc	1 238 16                        ; flash_attention.py:238:16
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v0, s[2:3], v15, v15, v1
	v_rcp_f32_e32 v14, v0
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v2, a1
	v_accvgpr_read_b32 v3, a2
	v_accvgpr_read_b32 v4, a3
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v19, -v0, v14, 1.0
	v_fmac_f32_e32 v14, v19, v14
	v_div_scale_f32 v19, vcc, v1, v15, v1
	v_mul_f32_e32 v20, v19, v14
	v_fma_f32 v21, -v0, v20, v19
	v_fmac_f32_e32 v20, v21, v14
	v_fma_f32 v0, -v0, v20, v19
	v_div_scale_f32 v19, s[2:3], v15, v15, v2
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v0, v0, v14, v20
	v_div_fixup_f32 v0, v0, v15, v1
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v5, a4
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v1, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v1, v21
	v_div_scale_f32 v1, vcc, v2, v15, v2
	v_mul_f32_e32 v14, v1, v21
	v_fma_f32 v20, -v19, v14, v1
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v1, -v19, v14, v1
	v_div_scale_f32 v19, s[2:3], v15, v15, v3
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v1, v1, v21, v14
	v_div_fixup_f32 v1, v1, v15, v2
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v6, a5
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v2, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v2, v20
	v_div_scale_f32 v2, vcc, v3, v15, v3
	v_mul_f32_e32 v14, v2, v20
	v_fma_f32 v21, -v19, v14, v2
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v2, -v19, v14, v2
	v_div_scale_f32 v19, s[2:3], v15, v15, v4
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v2, v2, v20, v14
	v_div_fixup_f32 v2, v2, v15, v3
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v7, a6
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v3, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v3, v21
	v_div_scale_f32 v3, vcc, v4, v15, v4
	v_mul_f32_e32 v14, v3, v21
	v_fma_f32 v20, -v19, v14, v3
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v3, -v19, v14, v3
	v_div_scale_f32 v19, s[2:3], v15, v15, v5
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v3, v3, v21, v14
	v_div_fixup_f32 v3, v3, v15, v4
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v8, a7
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v4, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v4, v20
	v_div_scale_f32 v4, vcc, v5, v15, v5
	v_mul_f32_e32 v14, v4, v20
	v_fma_f32 v21, -v19, v14, v4
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v4, -v19, v14, v4
	v_div_scale_f32 v19, s[2:3], v15, v15, v6
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v4, v4, v20, v14
	v_div_fixup_f32 v4, v4, v15, v5
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v9, a8
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v5, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v5, v21
	v_div_scale_f32 v5, vcc, v6, v15, v6
	v_mul_f32_e32 v14, v5, v21
	v_fma_f32 v20, -v19, v14, v5
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v5, -v19, v14, v5
	v_div_scale_f32 v19, s[2:3], v15, v15, v7
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v5, v5, v21, v14
	v_div_fixup_f32 v5, v5, v15, v6
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v10, a9
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v6, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v6, v20
	v_div_scale_f32 v6, vcc, v7, v15, v7
	v_mul_f32_e32 v14, v6, v20
	v_fma_f32 v21, -v19, v14, v6
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v6, -v19, v14, v6
	v_div_scale_f32 v19, s[2:3], v15, v15, v8
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v6, v6, v20, v14
	v_div_fixup_f32 v6, v6, v15, v7
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v11, a10
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v7, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v7, v21
	v_div_scale_f32 v7, vcc, v8, v15, v8
	v_mul_f32_e32 v14, v7, v21
	v_fma_f32 v20, -v19, v14, v7
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v7, -v19, v14, v7
	v_div_scale_f32 v19, s[2:3], v15, v15, v9
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v7, v7, v21, v14
	v_div_fixup_f32 v7, v7, v15, v8
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v12, a11
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v8, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v8, v20
	v_div_scale_f32 v8, vcc, v9, v15, v9
	v_mul_f32_e32 v14, v8, v20
	v_fma_f32 v21, -v19, v14, v8
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v8, -v19, v14, v8
	v_div_scale_f32 v19, s[2:3], v15, v15, v10
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v8, v8, v20, v14
	v_div_fixup_f32 v8, v8, v15, v9
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v13, a12
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v9, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v9, v21
	v_div_scale_f32 v9, vcc, v10, v15, v10
	v_mul_f32_e32 v14, v9, v21
	v_fma_f32 v20, -v19, v14, v9
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v9, -v19, v14, v9
	v_div_scale_f32 v19, s[2:3], v15, v15, v11
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v9, v9, v21, v14
	v_div_fixup_f32 v9, v9, v15, v10
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v16, a13
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v10, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v10, v20
	v_div_scale_f32 v10, vcc, v11, v15, v11
	v_mul_f32_e32 v14, v10, v20
	v_fma_f32 v21, -v19, v14, v10
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v10, -v19, v14, v10
	v_div_scale_f32 v19, s[2:3], v15, v15, v12
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v10, v10, v20, v14
	v_div_fixup_f32 v10, v10, v15, v11
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v17, a14
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v11, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v11, v21
	v_div_scale_f32 v11, vcc, v12, v15, v12
	v_mul_f32_e32 v14, v11, v21
	v_fma_f32 v20, -v19, v14, v11
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v11, -v19, v14, v11
	v_div_scale_f32 v19, s[2:3], v15, v15, v13
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v11, v11, v21, v14
	v_div_fixup_f32 v11, v11, v15, v12
	.loc	1 233 42                        ; flash_attention.py:233:42
	v_accvgpr_read_b32 v18, a15
	.loc	1 238 16                        ; flash_attention.py:238:16
	v_fma_f32 v12, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v12, v20
	v_div_scale_f32 v12, vcc, v13, v15, v13
	v_mul_f32_e32 v14, v12, v20
	v_fma_f32 v21, -v19, v14, v12
	v_fmac_f32_e32 v14, v21, v20
	v_fma_f32 v12, -v19, v14, v12
	v_div_scale_f32 v19, s[2:3], v15, v15, v16
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v12, v12, v20, v14
	v_div_fixup_f32 v12, v12, v15, v13
	v_fma_f32 v13, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v13, v21
	v_div_scale_f32 v13, vcc, v16, v15, v16
	v_mul_f32_e32 v14, v13, v21
	v_fma_f32 v20, -v19, v14, v13
	v_fmac_f32_e32 v14, v20, v21
	v_fma_f32 v13, -v19, v14, v13
	v_div_scale_f32 v19, s[2:3], v15, v15, v17
	v_rcp_f32_e32 v20, v19
	v_div_fmas_f32 v13, v13, v21, v14
	v_div_fixup_f32 v13, v13, v15, v16
	v_fma_f32 v14, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v14, v20
	v_div_scale_f32 v14, vcc, v17, v15, v17
	v_mul_f32_e32 v16, v14, v20
	v_fma_f32 v21, -v19, v16, v14
	v_fmac_f32_e32 v16, v21, v20
	v_fma_f32 v14, -v19, v16, v14
	v_div_scale_f32 v19, s[2:3], v15, v15, v18
	v_rcp_f32_e32 v21, v19
	v_div_fmas_f32 v14, v14, v20, v16
	v_div_fixup_f32 v14, v14, v15, v17
	v_fma_f32 v16, -v19, v21, 1.0
	v_fmac_f32_e32 v21, v16, v21
	v_div_scale_f32 v16, vcc, v18, v15, v18
	v_mul_f32_e32 v17, v16, v21
	v_fma_f32 v20, -v19, v17, v16
	v_fmac_f32_e32 v17, v20, v21
	v_fma_f32 v16, -v19, v17, v16
	v_div_fmas_f32 v16, v16, v21, v17
	v_div_fixup_f32 v15, v16, v15, v18
	.loc	1 240 21                        ; flash_attention.py:240:21
	v_lshlrev_b32_e32 v16, 6, v30
	v_or3_b32 v17, v29, v16, s11
	.loc	1 240 32 is_stmt 0              ; flash_attention.py:240:32
	v_lshlrev_b32_e32 v17, 2, v17
	v_bfrev_b32_e32 v18, 1
	.loc	1 199 32 is_stmt 1              ; flash_attention.py:199:32
	v_cmp_gt_i32_e32 vcc, s0, v28
	.loc	1 240 21                        ; flash_attention.py:240:21
	v_or3_b32 v16, s11, v29, v16
	s_mov_b32 s11, 0x27000
	.loc	1 240 32 is_stmt 0              ; flash_attention.py:240:32
	v_cndmask_b32_e32 v17, v18, v17, vcc
	buffer_store_dwordx4 v[0:3], v17, s[8:11], 0 offen
	s_nop 1
	v_lshlrev_b32_e32 v0, 2, v16
	v_or_b32_e32 v1, 32, v0
	v_cndmask_b32_e32 v1, v18, v1, vcc
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_or_b32_e32 v1, 64, v0
	v_or_b32_e32 v0, 0x60, v0
	v_cndmask_b32_e32 v1, v18, v1, vcc
	v_cndmask_b32_e32 v0, v18, v0, vcc
	buffer_store_dwordx4 v[8:11], v1, s[8:11], 0 offen
	buffer_store_dwordx4 v[12:15], v0, s[8:11], 0 offen
	.loc	1 240 4                         ; flash_attention.py:240:4
	s_endpgm
.Ltmp121:
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
		.amdhsa_next_free_vgpr 156
		.amdhsa_next_free_sgpr 25
		.amdhsa_accum_offset 124
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
	.set flash_attention_fwd_async_buffer_kernel.num_vgpr, 124
	.set flash_attention_fwd_async_buffer_kernel.num_agpr, 32
	.set flash_attention_fwd_async_buffer_kernel.numbered_sgpr, 25
	.set flash_attention_fwd_async_buffer_kernel.num_named_barrier, 0
	.set flash_attention_fwd_async_buffer_kernel.private_seg_size, 0
	.set flash_attention_fwd_async_buffer_kernel.uses_vcc, 1
	.set flash_attention_fwd_async_buffer_kernel.uses_flat_scratch, 0
	.set flash_attention_fwd_async_buffer_kernel.has_dyn_sized_stack, 0
	.set flash_attention_fwd_async_buffer_kernel.has_recursion, 0
	.set flash_attention_fwd_async_buffer_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8328
; TotalNumSgprs: 31
; NumVgprs: 124
; NumAgprs: 32
; TotalNumVgprs: 156
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 19
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 156
; AccumOffset: 124
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 30
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
	.byte	223                             ; DW_AT_call_line
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
	.byte	226                             ; DW_AT_call_line
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
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp89-.Lfunc_begin0
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
	.quad	.Ltmp87-.Lfunc_begin0
	.quad	.Ltmp88-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges2:
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
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp118-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges3:
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
	.quad	.Ltmp83-.Lfunc_begin0
	.quad	.Ltmp84-.Lfunc_begin0
	.quad	.Ltmp85-.Lfunc_begin0
	.quad	.Ltmp86-.Lfunc_begin0
	.quad	.Ltmp90-.Lfunc_begin0
	.quad	.Ltmp91-.Lfunc_begin0
	.quad	.Ltmp92-.Lfunc_begin0
	.quad	.Ltmp93-.Lfunc_begin0
	.quad	.Ltmp94-.Lfunc_begin0
	.quad	.Ltmp95-.Lfunc_begin0
	.quad	.Ltmp96-.Lfunc_begin0
	.quad	.Ltmp97-.Lfunc_begin0
	.quad	.Ltmp98-.Lfunc_begin0
	.quad	.Ltmp99-.Lfunc_begin0
	.quad	.Ltmp100-.Lfunc_begin0
	.quad	.Ltmp101-.Lfunc_begin0
	.quad	.Ltmp102-.Lfunc_begin0
	.quad	.Ltmp103-.Lfunc_begin0
	.quad	.Ltmp104-.Lfunc_begin0
	.quad	.Ltmp105-.Lfunc_begin0
	.quad	.Ltmp106-.Lfunc_begin0
	.quad	.Ltmp107-.Lfunc_begin0
	.quad	.Ltmp108-.Lfunc_begin0
	.quad	.Ltmp109-.Lfunc_begin0
	.quad	.Ltmp110-.Lfunc_begin0
	.quad	.Ltmp111-.Lfunc_begin0
	.quad	.Ltmp112-.Lfunc_begin0
	.quad	.Ltmp113-.Lfunc_begin0
	.quad	.Ltmp114-.Lfunc_begin0
	.quad	.Ltmp115-.Lfunc_begin0
	.quad	.Ltmp116-.Lfunc_begin0
	.quad	.Ltmp117-.Lfunc_begin0
	.quad	.Ltmp119-.Lfunc_begin0
	.quad	.Ltmp120-.Lfunc_begin0
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
	.asciz	"flash_attention_fwd_async_buffer_kernel" ; string offset=142
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
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         flash_attention_fwd_async_buffer_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     156
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
