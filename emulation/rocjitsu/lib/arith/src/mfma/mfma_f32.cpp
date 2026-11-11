// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 and XF32 MFMA retain the legacy full-FP32 approximation, without new exact claims.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/reduction.h"
#include "rj_arith/rj_arith.h"

namespace {
// Validate the approximate model and evaluate a full matrix; errors do not touch operands.
template <int M, int K, int Batch>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_f32_x86_64_v4<M, K, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_f32_x86_64_v3<M, K, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; ++i) {
    left[i] = as_float(a[i / 64][i % 64]);
    right[i] = as_float(b[i / 64][i % 64]);
  }
  mfma_reduce_scalar<M, K, Batch, 1>(controls, d, left, right, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 16x16x8 XF32 MFMA with the inherited approximate arithmetic.
int rj_arith_v_mfma_f32_16x16x8_xf32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<16, 8, 1>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x4 XF32 MFMA with the inherited approximate arithmetic.
int rj_arith_v_mfma_f32_32x32x4_xf32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<32, 4, 1>(flags, controls, d, a, b, c);
}

// Evaluate 2 independent 32x32x1 FP32 matrices.
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return evaluate<32, 1, 2>(flags, controls, d, a, b, c);
}

// Evaluate 4 independent 16x16x1 FP32 matrices.
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return evaluate<16, 1, 4>(flags, controls, d, a, b, c);
}

// Evaluate 16 independent 4x4x1 FP32 matrices.
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  return evaluate<4, 1, 16>(flags, controls, d, a, b, c);
}

// Evaluate 1 independent 32x32x2 FP32 matrices.
int rj_arith_v_mfma_f32_32x32x2_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<32, 2, 1>(flags, controls, d, a, b, c);
}

// Evaluate 1 independent 16x16x4 FP32 matrices.
int rj_arith_v_mfma_f32_16x16x4_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 4, 1>(flags, controls, d, a, b, c);
}
