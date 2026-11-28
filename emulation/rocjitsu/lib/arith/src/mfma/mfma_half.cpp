// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16/BF16 MFMA validation, conversion, and target-specific accumulation policy.
#include "common/float_formats.h"
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/gfx942_exact.h"
#include "mfma/reduction.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"

namespace {

// Keep the scalar stack buffer out of the SIMD dispatch prologue.
template <int M, int K, int Batch, bool Bf16, rocjitsu::arith::MfmaF32Model Model>
[[gnu::noinline]] void mfma_half_scalar(uint64_t instruction_flags, uint64_t flags,
                                        bool clamp_overflow, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  // Convert each source once, even when multiple output rows reuse it.
  float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; ++i) {
    uint16_t x = static_cast<uint16_t>(a[i / 128][i / 2 % 64] >> (16 * (i % 2)));
    uint16_t y = static_cast<uint16_t>(b[i / 128][i / 2 % 64] >> (16 * (i % 2)));
    if constexpr (Model == MfmaF32Model::Gfx90a) {
      constexpr uint16_t Exponent = Bf16 ? 0x7f80U : 0x7c00U;
      x = (x & Exponent) ? x : x & 0x8000U;
      y = (y & Exponent) ? y : y & 0x8000U;
    }
    left[i] = Bf16 ? bf16_to_float(x) : f16_to_float(x);
    right[i] = Bf16 ? bf16_to_float(y) : f16_to_float(y);
  }
  mfma_reduce_scalar<M, K, Batch, 2, Bf16, Model>(instruction_flags, d, left, right, c,
                                                  clamp_overflow, flags);
}

// Evaluate a full MFMA wave after validating controls; errors do not access
// operands.
template <int M, int K, int Batch, bool Bf16, bool Gfx942 = false,
          rocjitsu::arith::MfmaF32Model Model = rocjitsu::arith::MfmaF32Model::Default>
int mfma(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
         const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  [[maybe_unused]] const uint64_t implementation_flags = flags;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, instruction_flags & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  // Captured gfx942 forms with default source controls support exact modes for
  // both FP16_OVFL=0 and 1. Compatibility retains the fast approximate reduction.
  if constexpr (Gfx942) {
    if (is_exact(flags) && !instruction_flags) {
      constexpr MfmaGfx942Format format = Bf16 ? MfmaGfx942Format::Bf16 : MfmaGfx942Format::F16;
      const bool no_inf_nan = (flags & RJ_ARITH_FP16_OVFL) != 0;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
        mfma_gfx942_exact_x86_64_v4<format, M, K, Batch>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
        mfma_gfx942_exact_x86_64_v3<format, M, K, Batch>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
      mfma_gfx942_exact(format, M, K, Batch, no_inf_nan, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // Exact gfx942 modes clamp each K-step above; gfx942 approximate arithmetic
  // ignores FP16_OVFL in every backend, including SIMD implementations.
  if constexpr (Gfx942) {
    flags &= ~RJ_ARITH_FP16_OVFL;
  }
  const bool clamp_overflow =
      Model == MfmaF32Model::Default && !Gfx942 && Bf16 && (flags & RJ_ARITH_FP16_OVFL);
  // CDNA2 half-input MFMA fixes both denormal policies, independently of MODE.
  if constexpr (Model == MfmaF32Model::Gfx90a) {
    flags |= RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4_BF16)
  if constexpr (Bf16 && (Batch == 1 || M == 4)) {
    constexpr int ABlockLanes = Model == MfmaF32Model::Gfx90a ? M : 0;
    const bool native_policy =
        Model == MfmaF32Model::Default ||
        (implementation_flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY;
    if (native_policy && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_ZEN4) &&
        mfma_half_x86_64_v4_bf16<M, K, Batch, ABlockLanes>(
            instruction_flags,
            implementation_flags &
                ~(Gfx942 || Model != MfmaF32Model::Default ? RJ_ARITH_FP16_OVFL : 0ULL),
            d, a, b, c)) {
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_half_x86_64_v4<M, K, Batch, Bf16, Model>(instruction_flags, flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_half_x86_64_v3<M, K, Batch, Bf16, Model>(instruction_flags, flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  mfma_half_scalar<M, K, Batch, Bf16, Model>(instruction_flags, flags, clamp_overflow, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 32x32x4 FP16 MFMA with 2 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 4, 2, false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x4 FP16 MFMA with 4 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 4, 4, false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 4x4x4 FP16 MFMA with 16 independent matrix block(s).
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<4, 4, 16, false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x8 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x8_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                        uint32_t *const *d, const uint32_t *const *a,
                                        const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x16 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 16, 1, false, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x32 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x32_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 32, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x16 FP16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x16_f16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 16, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x4 BF16 MFMA with 2 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 4, 2, true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x4 BF16 MFMA with 4 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                            uint32_t *const *d, const uint32_t *const *a,
                                            const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 4, 4, true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 4x4x4 BF16 MFMA with 16 independent matrix block(s).
int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                           uint32_t *const *d, const uint32_t *const *a,
                                           const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<4, 4, 16, true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x8 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x8_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                         uint32_t *const *d, const uint32_t *const *a,
                                         const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x16 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 16, 1, true, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 16x16x32 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<16, 32, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 32x32x16 BF16 MFMA with 1 independent matrix block(s).
int rj_arith_v_mfma_f32_32x32x16_bf16_w64(uint64_t flags, uint64_t instruction_flags,
                                          uint32_t *const *d, const uint32_t *const *a,
                                          const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 16, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<32, 4, 2, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<16, 4, 4, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return mfma<4, 4, 16, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<16, 16, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<32, 4, 2, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<16, 4, 4, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<4, 4, 16, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<32, 8, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return mfma<16, 16, 1, true>(flags, instruction_flags, d, a, b, c);
}

// Evaluate 2 32x32x2 BF16 matrix block(s) with the inherited gfx908 policy.
int rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<32, 2, 2, true, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 4 16x16x2 BF16 matrix block(s) with the inherited gfx908 policy.
int rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<16, 2, 4, true, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 16 4x4x2 BF16 matrix block(s) with the inherited gfx908 policy.
int rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<4, 2, 16, true, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 32x32x4 BF16 matrix block(s) with the inherited gfx908 policy.
int rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<32, 4, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 16x16x8 BF16 matrix block(s) with the inherited gfx908 policy.
int rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<16, 8, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 2 32x32x4 F16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<32, 4, 2, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                             b, c);
}

// Evaluate 4 16x16x4 F16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<16, 4, 4, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                             b, c);
}

// Evaluate 16 4x4x4 F16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return mfma<4, 4, 16, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                             b, c);
}

// Evaluate 1 32x32x8 F16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                             b, c);
}

// Evaluate 1 16x16x16 F16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<16, 16, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                              b, c);
}

// Evaluate 2 32x32x2 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<32, 2, 2, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 4 16x16x2 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<16, 2, 4, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 16 4x4x2 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<4, 2, 16, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 32x32x4 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<32, 4, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 16x16x8 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<16, 8, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 2 32x32x4 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<32, 4, 2, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 4 16x16x4 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c) {
  return mfma<16, 4, 4, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 16 4x4x4 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<4, 4, 16, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 32x32x8 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<32, 8, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                            b, c);
}

// Evaluate 1 16x16x16 BF16 matrix block(s) with the inherited gfx90a policy.
int rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return mfma<16, 16, 1, true, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d, a,
                                                                             b, c);
}

// Evaluate gfx908 FP16 with its inherited approximate arithmetic policy.
int rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<32, 4, 2, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx908 FP16 with its inherited approximate arithmetic policy.
int rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return mfma<16, 4, 4, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx908 FP16 with its inherited approximate arithmetic policy.
int rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return mfma<4, 4, 16, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx908 FP16 with its inherited approximate arithmetic policy.
int rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return mfma<32, 8, 1, false>(flags, instruction_flags, d, a, b, c);
}

// Evaluate gfx908 FP16 with its inherited approximate arithmetic policy.
int rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx908(uint64_t flags, uint64_t instruction_flags,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return mfma<16, 16, 1, false>(flags, instruction_flags, d, a, b, c);
}
