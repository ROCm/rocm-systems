// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx942/gfx950 FP64 MFMA validation, baseline evaluation and SIMD dispatch.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/f64.h"
#include "rj_arith/rj_arith.h"
namespace {
// Validate the inherited approximate model before accessing operands.
template <int M, int Batch>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error =
          validate(flags, controls & ~7ULL, kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_f64_x86_64_v4<M, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_f64_x86_64_v3<M, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  mfma_f64_kernel<M, Batch>(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 1 independent 16x16 FP64 output matrices.
int rj_arith_v_mfma_f64_16x16x4_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 1>(flags, controls, d, a, b, c);
}

// Evaluate 4 independent 4x4 FP64 output matrices.
int rj_arith_v_mfma_f64_4x4x4_4b_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<4, 4>(flags, controls, d, a, b, c);
}
