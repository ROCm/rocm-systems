// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local decoding and finite-format rounding for scaled narrowing.
#pragma once

#include "conversion/scaled_narrow.h"
#include "conversion/x86_64/fp8_narrow_x86_64_v3.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v3 {
// Decode the low source element without modifying host FP controls. Any NaN
// payload quieting is immaterial to the inherited sign-only FP8 NaN policy.
inline __m256 scaled_narrow_decode(__m256i raw, ScaledNarrowSource source, bool flush_input) {
  if (source == ScaledNarrowSource::F16) {
    raw = _mm256_and_si256(raw, _mm256_set1_epi32(0xffff));
    auto halves = _mm_packus_epi32(_mm256_castsi256_si128(raw), _mm256_extracti128_si256(raw, 1));
    return _mm256_cvtph_ps(halves);
  }
  if (source == ScaledNarrowSource::Bf16) {
    raw = _mm256_slli_epi32(raw, 16);
  }
  if (source == ScaledNarrowSource::F32 && flush_input) {
    auto tiny = _mm256_cmpeq_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7f800000)),
                                   _mm256_setzero_si256());
    raw = _mm256_blendv_epi8(raw, _mm256_and_si256(raw, _mm256_set1_epi32(int(0x80000000U))), tiny);
  }
  return _mm256_castsi256_ps(raw);
}
// Round scaled FP32 to the inherited finite-only FP4/FP6/BF6 model. Integer operations keep
// NaNs, saturation and the stochastic tiny-value cutoff on the vector path.
template <int ExponentBits, int FractionBits, bool Stochastic>
inline __m256i scaled_narrow_finite(__m256i raw, __m256i seed) {
  constexpr int format_bias = (1 << (ExponentBits - 1)) - 1;
  constexpr uint32_t sign_bit = 1U << (ExponentBits + FractionBits);
  constexpr uint32_t max_code = sign_bit - 1;
  constexpr uint32_t max_value = ((127 + (1 << ExponentBits) - 1 - format_bias) << 23) |
                                 (((1U << FractionBits) - 1) << (23 - FractionBits));
  constexpr uint32_t cutoff = (127 - format_bias - FractionBits) << 23;
  auto zero = _mm256_setzero_si256();
  auto one = _mm256_set1_epi32(1);
  auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  auto sign = _mm256_and_si256(_mm256_srli_epi32(raw, 31 - ExponentBits - FractionBits),
                               _mm256_set1_epi32(sign_bit));
  auto exponent =
      _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127 - format_bias));
  auto shift = _mm256_max_epi32(_mm256_set1_epi32(23 - FractionBits),
                                _mm256_sub_epi32(_mm256_set1_epi32(24 - FractionBits), exponent));
  auto significand = _mm256_or_si256(_mm256_and_si256(raw, _mm256_set1_epi32(0x7fffff)),
                                     _mm256_set1_epi32(0x800000));
  __m256i increment;
  if constexpr (Stochastic) {
    increment = _mm256_srlv_epi32(seed, _mm256_sub_epi32(_mm256_set1_epi32(32), shift));
  } else {
    auto half = _mm256_sllv_epi32(one, _mm256_sub_epi32(shift, one));
    auto odd = _mm256_and_si256(_mm256_srlv_epi32(significand, shift), one);
    increment = _mm256_add_epi32(_mm256_sub_epi32(half, one), odd);
  }
  auto result = _mm256_srlv_epi32(_mm256_add_epi32(significand, increment), shift);
  auto bias =
      _mm256_slli_epi32(_mm256_max_epi32(_mm256_sub_epi32(exponent, one), zero), FractionBits);
  result = _mm256_min_epu32(_mm256_add_epi32(result, bias), _mm256_set1_epi32(max_code));
  result = _mm256_or_si256(result, sign);
  // Oversized variable shifts are defined by x86. Override out-of-range
  // lanes after rounding rather than falling back or evaluating C++ shifts.
  result =
      _mm256_blendv_epi8(result, sign, _mm256_cmpgt_epi32(_mm256_set1_epi32(cutoff), magnitude));
  result = _mm256_blendv_epi8(result, _mm256_or_si256(sign, _mm256_set1_epi32(max_code)),
                              _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(max_value)));
  result = _mm256_blendv_epi8(result, zero,
                              _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000)));
  return result;
}

// Select only the rounding policy; source decoding, scale and packing are shared.
template <ScaledNarrowDestination Dst, bool Stochastic>
inline __m256i scaled_narrow_encode(__m256i raw, __m256i seed, bool saturate) {
  if constexpr (Dst == ScaledNarrowDestination::Fp4) {
    return scaled_narrow_finite<2, 1, Stochastic>(raw, seed);
  } else if constexpr (Dst == ScaledNarrowDestination::Fp6 || Dst == ScaledNarrowDestination::Bf6) {
    constexpr bool bf6 = Dst == ScaledNarrowDestination::Bf6;
    return scaled_narrow_finite<(bf6 ? 3 : 2), (bf6 ? 2 : 3), Stochastic>(raw, seed);
  } else {
    return fp8_narrow_detail::narrow<Dst == ScaledNarrowDestination::Bf8, Stochastic>(
        raw, seed, saturate, Fp8NarrowPolicy::Ocp);
  }
}
} // namespace rocjitsu::arith::x86_64_v3
