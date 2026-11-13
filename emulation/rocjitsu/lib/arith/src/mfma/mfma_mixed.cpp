// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed FP8/FP6/FP4 MFMA validation and CPU dispatch for gfx950.
#include "mfma/backends.h"
#include "mfma/mixed.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
namespace {
// Validate independent formats and semantics before reading operands.
template <int M, int K, bool Scaled = false>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c, const uint32_t *scale_a = nullptr,
             const uint32_t *scale_b = nullptr) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~(Scaled ? 0xfffULL : 0x3fULL),
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((controls & 7) > 4 || (controls >> 3 & 7) > 4) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const bool clamp_overflow = Scaled && (flags & RJ_ARITH_FP16_OVFL);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    if constexpr (Scaled) {
      mfma_scaled_x86_64_v4<M, K>(controls, d, a, b, c, scale_a, scale_b, clamp_overflow);
    } else {
      mfma_mixed_x86_64_v4<M, K>(controls, d, a, b, c);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    if constexpr (Scaled) {
      mfma_scaled_x86_64_v3<M, K>(controls, d, a, b, c, scale_a, scale_b, clamp_overflow);
    } else {
      mfma_mixed_x86_64_v3<M, K>(controls, d, a, b, c);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  mfma_mixed_kernel<M, K, Scaled>(controls, d, a, b, c, scale_a, scale_b, clamp_overflow);
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Evaluate the full wave with independently selected packed input formats.
int rj_arith_v_mfma_f32_16x16x128_f8f6f4_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16, 128>(flags, controls, d, a, b, c);
}
// Evaluate the full wave with independently selected packed input formats.
int rj_arith_v_mfma_f32_32x32x64_f8f6f4_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *c) {
  return evaluate<32, 64>(flags, controls, d, a, b, c);
}

// Apply independent block scale exponents after each ordered fused dot.
int rj_arith_v_mfma_scale_f32_16x16x128_f8f6f4_w64(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c,
                                                   const uint32_t *scale_a,
                                                   const uint32_t *scale_b) {
  return evaluate<16, 128, true>(flags, controls, d, a, b, c, scale_a, scale_b);
}

// Apply independent block scale exponents after each ordered fused dot.
int rj_arith_v_mfma_scale_f32_32x32x64_f8f6f4_w64(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c, const uint32_t *scale_a,
                                                  const uint32_t *scale_b) {
  return evaluate<32, 64, true>(flags, controls, d, a, b, c, scale_a, scale_b);
}
