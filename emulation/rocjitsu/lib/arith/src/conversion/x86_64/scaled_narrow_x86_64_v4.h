// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local decoding and finite-format rounding for scaled narrowing.
#pragma once

#include "conversion/scaled_narrow.h"
#include "conversion/x86_64/fp8_narrow_x86_64_v4.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v4 {
// Decode the low source element without modifying host FP controls. Any NaN
// payload quieting is immaterial to the inherited sign-only FP8 NaN policy.
inline __m512 scaled_narrow_decode(__m512i raw, ScaledNarrowSource source, bool flush_input) {
  if (source == ScaledNarrowSource::F16) {
    return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(raw));
  }
  if (source == ScaledNarrowSource::Bf16) {
    raw = _mm512_slli_epi32(raw, 16);
  }
  if (source == ScaledNarrowSource::F32 && flush_input) {
    auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(raw, _mm512_set1_epi32(0x7f800000)),
                                        _mm512_setzero_si512());
    raw = _mm512_mask_mov_epi32(raw, tiny,
                                _mm512_and_si512(raw, _mm512_set1_epi32(int(0x80000000U))));
  }
  return _mm512_castsi512_ps(raw);
}
// Round scaled FP32 to the inherited finite-only FP4/FP6/BF6 model. Integer operations keep
// NaNs, saturation and the stochastic tiny-value cutoff on the vector path.
template <int ExponentBits, int FractionBits, bool Stochastic>
inline __m512i scaled_narrow_finite(__m512i raw, __m512i seed) {
  constexpr int format_bias = (1 << (ExponentBits - 1)) - 1;
  constexpr uint32_t sign_bit = 1U << (ExponentBits + FractionBits);
  constexpr uint32_t max_code = sign_bit - 1;
  constexpr uint32_t max_value = ((127 + (1 << ExponentBits) - 1 - format_bias) << 23) |
                                 (((1U << FractionBits) - 1) << (23 - FractionBits));
  constexpr uint32_t cutoff = (127 - format_bias - FractionBits) << 23;
  auto zero = _mm512_setzero_si512();
  auto one = _mm512_set1_epi32(1);
  auto magnitude = _mm512_and_si512(raw, _mm512_set1_epi32(0x7fffffff));
  auto sign = _mm512_and_si512(_mm512_srli_epi32(raw, 31 - ExponentBits - FractionBits),
                               _mm512_set1_epi32(sign_bit));
  auto exponent =
      _mm512_sub_epi32(_mm512_srli_epi32(magnitude, 23), _mm512_set1_epi32(127 - format_bias));
  auto shift = _mm512_max_epi32(_mm512_set1_epi32(23 - FractionBits),
                                _mm512_sub_epi32(_mm512_set1_epi32(24 - FractionBits), exponent));
  auto significand = _mm512_or_si512(_mm512_and_si512(raw, _mm512_set1_epi32(0x7fffff)),
                                     _mm512_set1_epi32(0x800000));
  __m512i increment;
  if constexpr (Stochastic) {
    increment = _mm512_srlv_epi32(seed, _mm512_sub_epi32(_mm512_set1_epi32(32), shift));
  } else {
    auto half = _mm512_sllv_epi32(one, _mm512_sub_epi32(shift, one));
    auto odd = _mm512_and_si512(_mm512_srlv_epi32(significand, shift), one);
    increment = _mm512_add_epi32(_mm512_sub_epi32(half, one), odd);
  }
  auto result = _mm512_srlv_epi32(_mm512_add_epi32(significand, increment), shift);
  auto bias =
      _mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(exponent, one), zero), FractionBits);
  result = _mm512_min_epu32(_mm512_add_epi32(result, bias), _mm512_set1_epi32(max_code));
  result = _mm512_or_si512(result, sign);
  // Oversized variable shifts are defined by x86. Override out-of-range
  // lanes after rounding rather than falling back or evaluating C++ shifts.
  result = _mm512_mask_blend_epi32(_mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(cutoff)),
                                   result, sign);
  result = _mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(max_value)),
                                   result, _mm512_or_si512(sign, _mm512_set1_epi32(max_code)));
  result = _mm512_mask_blend_epi32(
      _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000)), result, zero);
  return result;
}

// Select only the rounding policy; source decoding, scale and packing are shared.
template <ScaledNarrowDestination Dst, bool Stochastic>
inline __m512i scaled_narrow_encode(__m512i raw, __m512i seed, bool saturate) {
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
} // namespace rocjitsu::arith::x86_64_v4
