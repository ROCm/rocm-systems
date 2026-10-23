// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 vector widening of FP8/BF8 encodings, including subnormals and special values.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <immintrin.h>

namespace rocjitsu::arith {

// Widen low bytes from OCP E4M3FN/E5M2 to FP32. NegativeNan selects the
// conversion instruction canonical NaN; otherwise NaNs preserve their sign.
template <bool Bf8, bool NegativeNan = false> inline __m512 widen_fp8(__m512i bytes) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  auto magnitude = _mm512_and_si512(bytes, _mm512_set1_epi32(127));
  auto fraction = _mm512_and_si512(magnitude, _mm512_set1_epi32((1 << fraction_bits) - 1));
  auto normal = _mm512_add_epi32(_mm512_slli_epi32(magnitude, 23 - fraction_bits),
                                 _mm512_set1_epi32((127 - bias) << 23));
  auto subnormal =
      _mm512_mul_ps(_mm512_cvtepi32_ps(fraction), _mm512_set1_ps(Bf8 ? 0x1p-16f : 0x1p-9f));
  auto small =
      _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(1 << fraction_bits), _MM_CMPINT_LT);
  auto result = _mm512_mask_mov_epi32(normal, small, _mm512_castps_si512(subnormal));
  if constexpr (Bf8) {
    auto special = _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(123), _MM_CMPINT_GT);
    auto nan = _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(124), _MM_CMPINT_GT);
    auto exceptional =
        _mm512_mask_mov_epi32(_mm512_set1_epi32(0x7f800000), nan,
                              _mm512_set1_epi32(NegativeNan ? int(0xffc00000U) : 0x7fc00000));
    result = _mm512_mask_mov_epi32(result, special, exceptional);
  } else {
    auto nan = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(127));
    result = _mm512_mask_mov_epi32(result, nan,
                                   _mm512_set1_epi32(NegativeNan ? int(0xffc00000U) : 0x7fc00000));
  }
  auto sign = _mm512_slli_epi32(_mm512_and_si512(bytes, _mm512_set1_epi32(128)), 24);
  return _mm512_castsi512_ps(_mm512_or_si512(result, sign));
}

} // namespace rocjitsu::arith
