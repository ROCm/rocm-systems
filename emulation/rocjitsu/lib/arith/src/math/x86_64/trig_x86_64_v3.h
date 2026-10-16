// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 loose vector sine/cosine approximation and input/output modifiers.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#ifndef RJ_ARITH_TRIG_X86_64_V3_H_
#define RJ_ARITH_TRIG_X86_64_V3_H_

#include "math/trig_coefficients.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Return loose vector sine/cosine results from raw FP32 input encodings.
template <bool Cosine> __m256 trig_value(__m256i bits) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i one = _mm256_set1_epi32(1);
  const __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  const __m256i mantissa = _mm256_or_si256(_mm256_and_si256(magnitude, _mm256_set1_epi32(0x7fffff)),
                                           _mm256_set1_epi32(0x800000));

  // Reduce in turns. Variable shifts with counts >=32 yield zero, including
  // negative counts represented as unsigned, so neither large nor tiny inputs
  // require a float-to-integer conversion or a separate reduction path.
  const __m256i shift = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(119));
  const __m256i phase =
      _mm256_and_si256(_mm256_or_si256(_mm256_sllv_epi32(mantissa, shift),
                                       _mm256_srlv_epi32(mantissa, _mm256_sub_epi32(zero, shift))),
                       _mm256_set1_epi32(0x7fffffff));
  const __m256i quadrant = _mm256_srli_epi32(phase, 29);
  const __m256i quarter = _mm256_and_si256(phase, _mm256_set1_epi32(0x1fffffff));
  const __m256i reflected = _mm256_cmpgt_epi32(quarter, _mm256_set1_epi32(0x0fffffff));
  const __m256i reduced = _mm256_blendv_epi8(
      quarter, _mm256_sub_epi32(_mm256_set1_epi32(0x20000000), quarter), reflected);
  const __m256i table = _mm256_and_si256(
      _mm256_xor_si256(_mm256_xor_si256(quadrant, reflected), _mm256_set1_epi32(Cosine)), one);
  const __m256i index = _mm256_srli_epi32(_mm256_add_epi32(reduced, reflected), 24);
  const __m256i fraction = _mm256_sub_epi32(reduced, _mm256_slli_epi32(index, 24));
  const __m256i column = _mm256_add_epi32(index, _mm256_slli_epi32(table, 4));
  const __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(fraction), _mm256_set1_ps(0x1p-24f));
  const __m256 c0 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[0], column, 4));
  const __m256 c1 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[1], column, 4));
  const __m256 c2 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[2], column, 4));
  const __m256 c3 = _mm256_cvtepi32_ps(_mm256_i32gather_epi32(trig::coefficients[3], column, 4));
  __m256 value =
      _mm256_mul_ps(_mm256_fmadd_ps(t, _mm256_fmadd_ps(t, _mm256_fmadd_ps(t, c3, c2), c1), c0),
                    _mm256_set1_ps(0x1p-26f));
  __m256i sign = _mm256_srli_epi32(quadrant, 1);
  if constexpr (Cosine) {
    sign = _mm256_xor_si256(sign, quadrant);
  } else {
    sign = _mm256_xor_si256(sign, _mm256_srli_epi32(bits, 31));
  }

  // Captured phase zeros are +0; the original small-sine region preserves -0.
  sign = _mm256_andnot_si256(
      _mm256_castps_si256(_mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_EQ_OQ)),
      _mm256_slli_epi32(sign, 31));
  value = _mm256_xor_ps(value, _mm256_castsi256_ps(sign));
  if constexpr (!Cosine) {
    const __m256i tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x39c00000), magnitude);

    // Limit this multiplication to tiny inputs to avoid spurious overflow for
    // huge finite values and invalid exceptions for nonfinite unused lanes.
    __m256 small = _mm256_castsi256_ps(_mm256_and_si256(bits, tiny));
    small = _mm256_mul_ps(small, _mm256_castsi256_ps(_mm256_set1_epi32(0x40c90fd5)));
    value = _mm256_blendv_ps(value, small, _mm256_castsi256_ps(tiny));
  }
  const __m256i nonfinite = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff));
  const __m256i infinity = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
  const __m256i nan = _mm256_blendv_epi8(_mm256_or_si256(bits, _mm256_set1_epi32(0x400000)),
                                         _mm256_set1_epi32(int32_t(0xffc00000)), infinity);
  return _mm256_blendv_ps(value, _mm256_castsi256_ps(nan), _mm256_castsi256_ps(nonfinite));
}

} // namespace rocjitsu::arith

#endif
