// SPDX-License-Identifier: MIT

// x86 SIMD vector widening of FP8/BF8 encodings, including subnormals and special values.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <immintrin.h>

namespace rocjitsu::arith {

// Widen the low byte of each 32-bit lane from OCP E4M3FN/E5M2 to FP32.
// Finite inputs widen exactly. NegativeNan selects the conversion instruction
// canonical NaN; otherwise quiet NaNs preserve their input sign.
template <bool Bf8, bool NegativeNan = false> inline __m256 widen_fp8(__m256i bytes) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  auto magnitude = _mm256_and_si256(bytes, _mm256_set1_epi32(127));
  auto fraction = _mm256_and_si256(magnitude, _mm256_set1_epi32((1 << fraction_bits) - 1));
  auto normal = _mm256_add_epi32(_mm256_slli_epi32(magnitude, 23 - fraction_bits),
                                 _mm256_set1_epi32((127 - bias) << 23));
  auto subnormal =
      _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(Bf8 ? 0x1p-16f : 0x1p-9f));
  auto small = _mm256_cmpgt_epi32(_mm256_set1_epi32(1 << fraction_bits), magnitude);
  auto result = _mm256_blendv_epi8(normal, _mm256_castps_si256(subnormal), small);
  if constexpr (Bf8) {
    auto special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(123));
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(124));
    auto exceptional = _mm256_or_si256(
        _mm256_set1_epi32(0x7f800000),
        _mm256_and_si256(nan, _mm256_set1_epi32(NegativeNan ? int(0x80400000U) : 0x00400000)));
    result = _mm256_blendv_epi8(result, exceptional, special);
  } else {
    auto nan = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(127));
    result = _mm256_blendv_epi8(
        result, _mm256_set1_epi32(NegativeNan ? int(0xffc00000U) : 0x7fc00000), nan);
  }
  auto sign = _mm256_slli_epi32(_mm256_and_si256(bytes, _mm256_set1_epi32(128)), 24);
  return _mm256_castsi256_ps(_mm256_or_si256(result, sign));
}

#if defined(__AVX512F__)
// Widen low-byte FP8/BF8 encodings into FP32 lanes; NegativeNan selects the NaN sign
// convention.
template <bool Bf8, bool NegativeNan = false> inline __m512 widen_fp8(__m512i bytes) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  constexpr int bias = Bf8 ? 15 : 7;
  auto magnitude = _mm512_and_si512(bytes, _mm512_set1_epi32(127));
  auto fraction = _mm512_and_si512(magnitude, _mm512_set1_epi32((1 << fraction_bits) - 1));
  auto normal = _mm512_add_epi32(_mm512_slli_epi32(magnitude, 23 - fraction_bits),
                                 _mm512_set1_epi32((127 - bias) << 23));
  auto subnormal =
      _mm512_mul_ps(_mm512_cvtepi32_ps(fraction), _mm512_set1_ps(Bf8 ? 0x1p-16f : 0x1p-9f));
  auto small = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1 << fraction_bits), magnitude);
  auto result = _mm512_mask_blend_epi32(small, normal, _mm512_castps_si512(subnormal));
  if constexpr (Bf8) {
    auto special = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(123));
    auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(124));
    auto exceptional =
        _mm512_or_si512(_mm512_set1_epi32(0x7f800000),
                        _mm512_maskz_set1_epi32(nan, NegativeNan ? int(0x80400000U) : 0x00400000));
    result = _mm512_mask_blend_epi32(special, result, exceptional);
  } else {
    auto nan = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(127));
    result = _mm512_mask_set1_epi32(result, nan, NegativeNan ? int(0xffc00000U) : 0x7fc00000);
  }
  auto sign = _mm512_slli_epi32(_mm512_and_si512(bytes, _mm512_set1_epi32(128)), 24);
  return _mm512_castsi512_ps(_mm512_or_si512(result, sign));
}

#endif

} // namespace rocjitsu::arith
