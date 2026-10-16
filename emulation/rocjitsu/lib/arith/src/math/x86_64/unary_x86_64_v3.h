// SPDX-License-Identifier: MIT

// AVX2 loose FP32 unary math and exact integral-rounding SIMD helpers.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "math/unary.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Approximate exp2 with a reduced polynomial. Half inputs need only [-64, 64];
// FP32 outputs outside [-126, 128) flush to zero or overflow to infinity.
template <bool Half> inline __m256 exp2_loose(__m256 input) {
  auto x = _mm256_min_ps(_mm256_max_ps(input, _mm256_set1_ps(Half ? -64 : -126)),
                         _mm256_set1_ps(Half ? 64 : 128));
  auto exponent = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  auto r = _mm256_sub_ps(x, exponent);

  // Taylor coefficients (ln(2)^n / n!) on the reduced interval [-0.5, 0.5].
  auto p = _mm256_set1_ps(1.525273380405984e-5f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0001540353039338161f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.001333355814642844f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.009618129107628477f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.05550410866482158f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265069591007f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.6931471805599453f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1));

  // Keep the scale normal when rounding the exponent produces 128. Apply the
  // remaining factor to p before scaling so finite values below overflow survive.
  if constexpr (!Half) {
    auto high = _mm256_cmp_ps(exponent, _mm256_set1_ps(127), _CMP_GT_OQ);
    exponent = _mm256_min_ps(exponent, _mm256_set1_ps(127));
    p = _mm256_blendv_ps(p, _mm256_add_ps(p, p), high);
  }
  auto scale = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(exponent), _mm256_set1_epi32(127)), 23));
  auto result = _mm256_mul_ps(p, scale);
  if constexpr (!Half) {
    result = _mm256_blendv_ps(result, _mm256_setzero_ps(),
                              _mm256_cmp_ps(input, _mm256_set1_ps(-126), _CMP_LT_OQ));
    result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000)),
                              _mm256_cmp_ps(input, _mm256_set1_ps(128), _CMP_GE_OQ));
  }
  auto infinity = _mm256_castsi256_ps(_mm256_set1_epi32(0x7f800000));
  auto negative_infinity = _mm256_castsi256_ps(_mm256_set1_epi32(int(0xff800000u)));
  result = _mm256_blendv_ps(result, _mm256_setzero_ps(),
                            _mm256_cmp_ps(input, negative_infinity, _CMP_EQ_OQ));
  auto positive_inf_or_nan = _mm256_or_ps(_mm256_cmp_ps(input, infinity, _CMP_EQ_OQ),
                                          _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
  return _mm256_blendv_ps(result, input, positive_inf_or_nan);
}

// Approximate log2 for normal FP32 inputs, including widened FP16 subnormals.
// The FP32 caller flushes input subnormals first. Zero returns -infinity.
inline __m256 log2_loose(__m256 input) {
  auto bits = _mm256_castps_si256(input);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127));
  auto m = _mm256_castsi256_ps(_mm256_or_si256(
      _mm256_and_si256(magnitude, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f800000)));
  auto upper = _mm256_cmp_ps(m, _mm256_set1_ps(1.4142135623730951f), _CMP_GT_OQ);
  m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), upper);
  exponent = _mm256_sub_epi32(exponent, _mm256_castps_si256(upper));

  // log(m) = 2 * (z + z^3/3 + z^5/5 + ...), |z| <= 0.171573.
  auto z = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1)), _mm256_add_ps(m, _mm256_set1_ps(1)));
  auto z2 = _mm256_mul_ps(z, z);
  auto p = _mm256_set1_ps(1.0f / 9);
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 7));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 5));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1.0f / 3));
  p = _mm256_fmadd_ps(p, z2, _mm256_set1_ps(1));
  auto result = _mm256_fmadd_ps(_mm256_mul_ps(z, _mm256_set1_ps(2.885390081777927f)), p,
                                _mm256_cvtepi32_ps(exponent));
  auto infinity = _mm256_set1_epi32(0x7f800000);
  result =
      _mm256_blendv_ps(result, input, _mm256_castsi256_ps(_mm256_cmpeq_epi32(magnitude, infinity)));
  auto invalid =
      _mm256_or_si256(_mm256_srai_epi32(bits, 31), _mm256_cmpgt_epi32(magnitude, infinity));
  result = _mm256_blendv_ps(result, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000)),
                            _mm256_castsi256_ps(invalid));
  return _mm256_blendv_ps(
      result, _mm256_castsi256_ps(_mm256_set1_epi32(int(0xff800000u))),
      _mm256_castsi256_ps(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256())));
}

// Half inputs have already widened exactly to normal FP32 values (or zero).
template <Unary Op, bool Half = false> __m256 unary_value(__m256 value) {
  if constexpr (Op == Unary::Exp) {
    return exp2_loose<Half>(value);
  }
  if constexpr (Op == Unary::Log) {
    return log2_loose(value);
  }
  if constexpr (Op == Unary::Fract) {
    value = _mm256_sub_ps(value, _mm256_floor_ps(value));
    auto limit = _mm256_castsi256_ps(_mm256_set1_epi32(Half ? 0x3f7fe000 : 0x3f7fffff));
    value = _mm256_blendv_ps(value, limit, _mm256_cmp_ps(value, limit, _CMP_GT_OQ));
  }
  if constexpr (Op == Unary::FrexpMant) {
    auto original = _mm256_castps_si256(value);
    auto magnitude = _mm256_and_si256(original, _mm256_set1_epi32(INT32_MAX));
    auto normalized = value;
    if constexpr (!Half) {
      auto subnormal = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);

      // Subnormal magnitudes are 23-bit integers and convert exactly under
      // every host FP mode. Conversions of normal encodings are discarded.
      // Never consume a subnormal float that host DAZ can erase.
      normalized =
          _mm256_blendv_ps(value, _mm256_cvtepi32_ps(_mm256_sign_epi32(magnitude, original)),
                           _mm256_castsi256_ps(subnormal));
    }
    auto mantissa = _mm256_or_si256(
        _mm256_and_si256(_mm256_castps_si256(normalized), _mm256_set1_epi32(int(0x807fffff))),
        _mm256_set1_epi32(0x3f000000));
    auto special =
        _mm256_or_si256(_mm256_cmpeq_epi32(magnitude, _mm256_setzero_si256()),
                        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - 1)));
    value = _mm256_castsi256_ps(_mm256_blendv_epi8(mantissa, original, special));
  }
  if constexpr (Op == Unary::Trunc) {
    value = _mm256_round_ps(value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  }
  if constexpr (Op == Unary::Ceil) {
    value = _mm256_round_ps(value, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  }
  if constexpr (Op == Unary::Floor) {
    value = _mm256_round_ps(value, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  }
  if constexpr (Op == Unary::Rndne) {
    value = _mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
  if constexpr (Op == Unary::Sqrt || Op == Unary::Rsq) {
    value = _mm256_sqrt_ps(value);
  }
  if constexpr (Op == Unary::Rcp || Op == Unary::Rsq) {
    value = _mm256_div_ps(_mm256_set1_ps(1), value);
  }
  return value;
}

} // namespace rocjitsu::arith
