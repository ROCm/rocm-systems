// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx942/gfx950 FP8 MFMA dispatch, retaining FNUZ/OCP decoding and approximate reduction.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/fp8_formats.h"
#include "mfma/gfx942_exact.h"
#include "mfma/gfx950_exact.h"
#include "mfma/reduction.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"

namespace {
// Keep the scalar stack buffer out of the SIMD dispatch prologue.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
[[gnu::noinline]] void evaluate_scalar(uint64_t controls, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  float left[M * K], right[M * K];
  for (int i = 0; i < M * K; ++i) {
    left[i] = mfma_decode_fp8<Bf8A, Fnuz>(uint8_t(a[i / 256][i / 4 % 64] >> (8 * (i % 4))));
    right[i] = mfma_decode_fp8<Bf8B, Fnuz>(uint8_t(b[i / 256][i / 4 % 64] >> (8 * (i % 4))));
  }
  mfma_reduce_scalar<M, K, 1, 4>(controls, d, left, right, c);
}

// Validate the approximate policy before accessing any source or destination.
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  // Default source controls are qualified on MI300X (FNUZ) and MI350X (OCP) across both
  // FP16_OVFL states. Compatibility retains the fast approximate implementation.
  if constexpr (Fnuz) {
    if (is_exact(flags) && !controls) {
      constexpr MfmaGfx942Format format =
          Bf8A ? (Bf8B ? MfmaGfx942Format::Bf8Bf8 : MfmaGfx942Format::Bf8Fp8)
               : (Bf8B ? MfmaGfx942Format::Fp8Bf8 : MfmaGfx942Format::Fp8Fp8);
      const bool no_inf_nan = (flags & RJ_ARITH_FP16_OVFL) != 0;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
        mfma_gfx942_exact_x86_64_v4<format, M, K, 1>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
        mfma_gfx942_exact_x86_64_v3<format, M, K, 1>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
      mfma_gfx942_exact(format, M, K, 1, no_inf_nan, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
  } else if (is_exact(flags) && !controls) {
    mfma_gfx950_exact(Bf8A ? (Bf8B ? MfmaGfx950Format::Bf8Bf8 : MfmaGfx950Format::Bf8Fp8)
                           : (Bf8B ? MfmaGfx950Format::Fp8Bf8 : MfmaGfx950Format::Fp8Fp8),
                      M, K, 1, (flags & RJ_ARITH_FP16_OVFL) != 0, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4_BF16)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_ZEN4) &&
      mfma_fp8_x86_64_v4_bf16<M, K, Bf8A, Bf8B, Fnuz>(controls, d, a, b, c)) {
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_fp8_x86_64_v4<M, K, Bf8A, Bf8B, Fnuz>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_fp8_x86_64_v3<M, K, Bf8A, Bf8B, Fnuz>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  evaluate_scalar<M, K, Bf8A, Bf8B, Fnuz>(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 16x16x32 MFMA with FNUZ FP8/FP8 factors.
int rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16, 32, false, false, true>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with OCP FP8/FP8 factors.
int rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<16, 32, false, false, false>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with FNUZ FP8/BF8 factors.
int rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16, 32, false, true, true>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with OCP FP8/BF8 factors.
int rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<16, 32, false, true, false>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with FNUZ BF8/FP8 factors.
int rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16, 32, true, false, true>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with OCP BF8/FP8 factors.
int rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<16, 32, true, false, false>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with FNUZ BF8/BF8 factors.
int rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<16, 32, true, true, true>(flags, controls, d, a, b, c);
}
// Evaluate 16x16x32 MFMA with OCP BF8/BF8 factors.
int rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<16, 32, true, true, false>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with FNUZ FP8/FP8 factors.
int rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<32, 16, false, false, true>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with OCP FP8/FP8 factors.
int rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<32, 16, false, false, false>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with FNUZ FP8/BF8 factors.
int rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<32, 16, false, true, true>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with OCP FP8/BF8 factors.
int rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<32, 16, false, true, false>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with FNUZ BF8/FP8 factors.
int rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<32, 16, true, false, true>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with OCP BF8/FP8 factors.
int rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<32, 16, true, false, false>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with FNUZ BF8/BF8 factors.
int rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  return evaluate<32, 16, true, true, true>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x16 MFMA with OCP BF8/BF8 factors.
int rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c) {
  return evaluate<32, 16, true, true, false>(flags, controls, d, a, b, c);
}
