// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 compensated finite FMA and nearest-even narrowing into FP16.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Round four finite FP64 values to FP16 encodings in four 32-bit elements.
inline __m128i mixed_half_narrow4(__m256d value, bool saturate) {
  // Nearest-even variant of rocjitsu's round_finite_f16_simd.
  auto bits = _mm256_castpd_si256(value);
  auto exponent = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(2047));
  auto significand = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffff)),
                                     _mm256_set1_epi64x(INT64_C(1) << 52));
  auto normal = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1008));
  auto not_tiny = _mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(997));
  auto shift = _mm256_blendv_epi8(_mm256_sub_epi64(_mm256_set1_epi64x(1051), exponent),
                                  _mm256_set1_epi64x(42), normal);
  shift = _mm256_blendv_epi8(_mm256_set1_epi64x(53), shift, not_tiny);
  auto rounded = _mm256_srlv_epi64(significand, shift);
  rounded = _mm256_add_epi64(
      rounded,
      _mm256_and_si256(
          normal, _mm256_slli_epi64(_mm256_sub_epi64(exponent, _mm256_set1_epi64x(1009)), 10)));

  auto unit = _mm256_sllv_epi64(_mm256_set1_epi64x(1), shift);
  auto remainder = _mm256_and_si256(significand, _mm256_sub_epi64(unit, _mm256_set1_epi64x(1)));

  auto halfway = _mm256_srli_epi64(unit, 1);
  auto increment = _mm256_or_si256(
      _mm256_cmpgt_epi64(remainder, halfway),
      _mm256_and_si256(_mm256_cmpeq_epi64(remainder, halfway),
                       _mm256_sub_epi64(_mm256_setzero_si256(),
                                        _mm256_and_si256(rounded, _mm256_set1_epi64x(1)))));
  rounded = _mm256_sub_epi64(rounded, _mm256_and_si256(increment, not_tiny));

  auto overflow = _mm256_or_si256(_mm256_cmpgt_epi64(exponent, _mm256_set1_epi64x(1038)),
                                  _mm256_cmpgt_epi64(rounded, _mm256_set1_epi64x(0x7bff)));
  rounded = _mm256_blendv_epi8(rounded, _mm256_set1_epi64x(saturate ? 0x7bff : 0x7c00), overflow);
  auto sign = _mm256_and_si256(_mm256_srli_epi64(bits, 48), _mm256_set1_epi64x(0x8000));
  auto packed = _mm256_shuffle_epi32(_mm256_or_si256(sign, rounded), _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

// Return four nearest-even FP16 FMA results from finite FP32 inputs.
// The caller handles exceptional inputs; saturation and output flushing are explicit.
inline __m128i mixed_half_value4(__m128 a, __m128 b, __m128 c, bool saturate,
                                 bool flush_output = false) {
  auto x = _mm256_cvtps_pd(a), y = _mm256_cvtps_pd(b), z = _mm256_cvtps_pd(c);
  auto product = _mm256_mul_pd(x, y);
  auto value = _mm256_add_pd(product, z);

  // Recover the addition residual so the intermediate can retain a sticky bit before FP16
  // rounding.
  auto virtual_c = _mm256_sub_pd(value, product);
  auto error = _mm256_add_pd(_mm256_sub_pd(product, _mm256_sub_pd(value, virtual_c)),
                             _mm256_sub_pd(z, virtual_c));
  auto bits = _mm256_castpd_si256(value);
  auto even =
      _mm256_cmpeq_epi64(_mm256_and_si256(bits, _mm256_set1_epi64x(1)), _mm256_setzero_si256());
  auto adjust = _mm256_and_si256(
      even, _mm256_castpd_si256(_mm256_cmp_pd(error, _mm256_setzero_pd(), _CMP_NEQ_OQ)));
  auto same_sign =
      _mm256_cmpeq_epi64(_mm256_srli_epi64(_mm256_xor_si256(bits, _mm256_castpd_si256(error)), 63),
                         _mm256_setzero_si256());
  auto step = _mm256_blendv_epi8(_mm256_set1_epi64x(-1), _mm256_set1_epi64x(1), same_sign);
  bits = _mm256_add_epi64(bits, _mm256_and_si256(adjust, step));
  auto result = mixed_half_narrow4(_mm256_castsi256_pd(bits), saturate);

  if (flush_output) {
    auto magnitude = _mm256_and_pd(_mm256_castsi256_pd(bits),
                                   _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX)));
    auto tiny = _mm256_castpd_si256(
        _mm256_cmp_pd(magnitude, _mm256_set1_pd(0x1p-14 - 0x1p-26), _CMP_LT_OQ));
    auto packed = _mm256_shuffle_epi32(tiny, _MM_SHUFFLE(2, 0, 2, 0));
    auto mask =
        _mm_unpacklo_epi64(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
    result = _mm_blendv_epi8(result, _mm_and_si128(result, _mm_set1_epi32(0x8000)), mask);
  }
  return result;
}

} // namespace rocjitsu::arith
