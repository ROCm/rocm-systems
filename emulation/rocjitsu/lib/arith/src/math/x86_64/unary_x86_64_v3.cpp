// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose FP32 unary math and exact integral-rounding SIMD helpers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/x86_64/unary_x86_64_v3.h"
#include "common/sdwa_output.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "conversion/x86_64/lanes.h"
#include "math/unary_impl.h"
#include "math/x86_64/fract_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Evaluate unary math for the selected lanes. Requires validated controls; preserve inactive
// destination lanes.
template <Unary Op>
void run(uint32_t exec_mask, uint32_t modifiers, uint32_t *d, const uint32_t *a, uint64_t flags) {
  const __m256i keep = _mm256_set1_epi32((modifiers & RJ_ARITH_ALU_ABS_A) ? 0x7fffffff : -1);
  const __m256i flip = _mm256_set1_epi32((modifiers & RJ_ARITH_ALU_NEG_A) ? INT32_MIN : 0);
  const auto sdwa_output = sdwa_output_policy(flags, modifiers);
  const float scales[] = {1, 2, 4, 0.5f};
  const __m256 scale = _mm256_set1_ps(scales[(modifiers >> 6) & 3]);
  for (int lane = 0; lane < 32; lane += 8) {
    __m256 value = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), keep),
        flip));
    if constexpr (unary_flushes_f32<Op>) {
      value = flush_denorm_f32(value);
    } else {
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        value = flush_denorm_f32(value);
      }
    }
    const auto input = value;
    if constexpr (Op == Unary::Fract) {
      value = fract_value_x86_64_v3(value, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24),
                                    flags & RJ_ARITH_FP_IEEE_MODE);
    } else {
      value = unary_value<Op>(value);
      if constexpr (Op == Unary::FrexpMant) {
        if (flags & RJ_ARITH_FP_IEEE_MODE) {
          auto bits = _mm256_castps_si256(value);
          auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)),
                                        _mm256_set1_epi32(0x7f800000));
          value = _mm256_castsi256_ps(
              _mm256_or_si256(bits, _mm256_and_si256(nan, _mm256_set1_epi32(0x00400000))));
        }
      }
    }
    if constexpr (unary_flushes_f32<Op>) {
      value = flush_denorm_f32(value);
    }
    if (modifiers & RJ_ARITH_ALU_SDWA) {
      // The core flushes its result before SDWA scales and rounds that encoding.
      if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
        value = flush_denorm_f32(value);
      }
      value = _mm256_castsi256_ps(
          sdwa_output_f32(conversion_simd::Lanes<32>(_mm256_castps_si256(value)), sdwa_output)
              .bits);
    } else {
      if (modifiers & RJ_ARITH_ALU_OMOD_MASK) {
        value = prepare_omod_f32(value, modifiers);
        const auto unscaled = value;
        value = _mm256_mul_ps(value, scale);
        if constexpr (unary_rounds_f32<Op>) {
          const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
          if (rounding) {
            // Scaling an integral result is exact except for finite overflow.
            // Correct nearest-even infinity when guest rounding selects maximum finite.
            const auto infinity = _mm256_set1_epi32(0x7f800000);
            const auto original = _mm256_castps_si256(unscaled);
            const auto sign = _mm256_and_si256(original, _mm256_set1_epi32(INT32_MIN));
            auto overflow = _mm256_and_si256(
                _mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX)),
                    infinity),
                _mm256_cmpgt_epi32(infinity,
                                   _mm256_and_si256(original, _mm256_set1_epi32(INT32_MAX))));
            const auto away = rounding == 1   ? _mm256_cmpeq_epi32(sign, _mm256_setzero_si256())
                              : rounding == 2 ? _mm256_srai_epi32(sign, 31)
                                              : _mm256_setzero_si256();
            overflow = _mm256_andnot_si256(away, overflow);
            const auto finite = _mm256_or_si256(sign, _mm256_set1_epi32(0x7f7fffff));
            value =
                _mm256_blendv_ps(value, _mm256_castsi256_ps(finite), _mm256_castsi256_ps(overflow));
          }
        }
      } else if constexpr (Op == Unary::Fract) {
        // OMOD already enforces output flushing, independently of guest policy.
        if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
          value = flush_denorm_f32(value);
        }
      }
      if (modifiers & RJ_ARITH_ALU_CLAMP) {
        value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
      }
    }
    if constexpr (unary_rounds_f32<Op> || Op == Unary::Fract) {
      if (!(flags & RJ_ARITH_FP_IEEE_MODE) &&
          (!(modifiers & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF))) {
        auto nan = _mm256_cmpgt_epi32(
            _mm256_and_si256(_mm256_castps_si256(input), _mm256_set1_epi32(INT32_MAX)),
            _mm256_set1_epi32(0x7f800000));
        value = _mm256_blendv_ps(value, input, _mm256_castsi256_ps(nan));
      }
    }
    __m256i lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                               _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace

// Evaluate unary math. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void unary_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                     const uint32_t *a, uint64_t flags) {
  switch (op) {
  case Unary::Fract:
    return run<Unary::Fract>(exec_mask, modifiers, d, a, flags);
  case Unary::FrexpMant:
    return run<Unary::FrexpMant>(exec_mask, modifiers, d, a, flags);
  case Unary::Trunc:
    return run<Unary::Trunc>(exec_mask, modifiers, d, a, flags);
  case Unary::Ceil:
    return run<Unary::Ceil>(exec_mask, modifiers, d, a, flags);
  case Unary::Rndne:
    return run<Unary::Rndne>(exec_mask, modifiers, d, a, flags);
  case Unary::Floor:
    return run<Unary::Floor>(exec_mask, modifiers, d, a, flags);
  case Unary::Sqrt:
    return run<Unary::Sqrt>(exec_mask, modifiers, d, a, flags);
  case Unary::Rcp:
    return run<Unary::Rcp>(exec_mask, modifiers, d, a, flags);
  case Unary::Rsq:
    return run<Unary::Rsq>(exec_mask, modifiers, d, a, flags);
  case Unary::Exp:
    return run<Unary::Exp>(exec_mask, modifiers, d, a, flags);
  case Unary::Log:
    return run<Unary::Log>(exec_mask, modifiers, d, a, flags);
  }
}

// Fixed-rounding operations preserve infinities and quiet NaNs with their
// sign/payload, matching round_integral_bits. Modifiers share the loose kernel.
void round_exact_x86_64_v3(Unary op, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                           const uint32_t *a, uint64_t flags) {
  unary_x86_64_v3(op, exec_mask, mode, d, a, flags);
}

} // namespace rocjitsu::arith
