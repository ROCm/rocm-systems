// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 native reciprocal/roots and shared Loose EXP2/LOG2 polynomials.
// LOG inputs are normal FP32 numbers, widened halves, zeros or special values.
#pragma once

#include "common/isa.h"
#include "math/unary.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Approximate EXP2, retaining infinities/NaNs and the destination's useful exponent range.
template <bool Half> inline __m512 exp2_loose(__m512 input) {
  auto x = _mm512_min_ps(_mm512_max_ps(input, _mm512_set1_ps(Half ? -64 : -126)),
                         _mm512_set1_ps(Half ? 64 : 128));
  auto exponent = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  auto r = _mm512_sub_ps(x, exponent);
  // Same reduced Taylor polynomial and evaluation order as the AVX2 path.
  auto p = _mm512_set1_ps(1.525273380405984e-5f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.0001540353039338161f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.001333355814642844f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.009618129107628477f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.05550410866482158f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.2402265069591007f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.6931471805599453f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1));
  if constexpr (!Half) {
    // Keep the power-of-two scale finite at exponent 128, as in the AVX2 polynomial.
    const auto high = _mm512_cmp_ps_mask(exponent, _mm512_set1_ps(127), _CMP_GT_OQ);
    exponent = _mm512_min_ps(exponent, _mm512_set1_ps(127));
    p = _mm512_mask_add_ps(p, high, p, p);
  }
  auto scale = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_add_epi32(_mm512_cvttps_epi32(exponent), _mm512_set1_epi32(127)), 23));
  auto result = _mm512_mul_ps(p, scale);
  if constexpr (!Half) {
    result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(input, _mm512_set1_ps(-126), _CMP_LT_OQ),
                                _mm512_setzero_ps());
    result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(input, _mm512_set1_ps(128), _CMP_GE_OQ),
                                _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000)));
  }
  auto infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0x7f800000));
  auto negative_infinity = _mm512_castsi512_ps(_mm512_set1_epi32(0xff800000U));
  result = _mm512_mask_mov_ps(result, _mm512_cmp_ps_mask(input, negative_infinity, _CMP_EQ_OQ),
                              _mm512_setzero_ps());
  auto special = _mm512_cmp_ps_mask(input, infinity, _CMP_EQ_OQ) |
                 _mm512_cmp_ps_mask(input, input, _CMP_UNORD_Q);
  return _mm512_mask_mov_ps(result, special, input);
}

// Approximate LOG2 for normal FP32 inputs; zero gives -Inf and negative inputs NaN.
inline __m512 log2_loose(__m512 input) {
  auto bits = _mm512_castps_si512(input);
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  auto exponent = _mm512_sub_epi32(_mm512_srli_epi32(magnitude, 23), _mm512_set1_epi32(127));
  auto m = _mm512_castsi512_ps(_mm512_or_si512(
      _mm512_and_si512(magnitude, _mm512_set1_epi32(0x007fffff)), _mm512_set1_epi32(0x3f800000)));
  auto upper = _mm512_cmp_ps_mask(m, _mm512_set1_ps(1.4142135623730951f), _CMP_GT_OQ);
  m = _mm512_mask_mul_ps(m, upper, m, _mm512_set1_ps(0.5f));
  exponent = _mm512_mask_add_epi32(exponent, upper, exponent, _mm512_set1_epi32(1));
  // Same odd polynomial on the reduced interval |z| <= 0.171573 as AVX2.
  auto z = _mm512_div_ps(_mm512_sub_ps(m, _mm512_set1_ps(1)), _mm512_add_ps(m, _mm512_set1_ps(1)));
  auto z2 = _mm512_mul_ps(z, z);
  auto p = _mm512_set1_ps(1.0f / 9);
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 7));
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 5));
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1.0f / 3));
  p = _mm512_fmadd_ps(p, z2, _mm512_set1_ps(1));
  auto result = _mm512_fmadd_ps(_mm512_mul_ps(z, _mm512_set1_ps(2.885390081777927f)), p,
                                _mm512_cvtepi32_ps(exponent));
  auto infinity = _mm512_set1_epi32(0x7f800000);
  result = _mm512_mask_mov_ps(result, _mm512_cmpeq_epi32_mask(magnitude, infinity), input);
  auto invalid = _mm512_cmplt_epi32_mask(bits, _mm512_setzero_si512()) |
                 _mm512_cmpgt_epi32_mask(magnitude, infinity);
  result = _mm512_mask_mov_ps(result, invalid, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
  return _mm512_mask_mov_ps(result, _mm512_cmpeq_epi32_mask(magnitude, _mm512_setzero_si512()),
                            _mm512_castsi512_ps(_mm512_set1_epi32(0xff800000U)));
}

// Evaluate native FP32 math; the caller handles fixed denormal and destination policies.
template <Unary Op> inline __m512 unary_loose_value(__m512 input) {
  if constexpr (Op == Unary::Rcp) {
    return _mm512_div_ps(_mm512_set1_ps(1), input);
  } else if constexpr (Op == Unary::Sqrt) {
    return _mm512_sqrt_ps(input);
  } else if constexpr (Op == Unary::Rsq) {
    return _mm512_div_ps(_mm512_set1_ps(1), _mm512_sqrt_ps(input));
  } else if constexpr (Op == Unary::Exp) {
    return exp2_loose<false>(input);
  } else {
    static_assert(Op == Unary::Log);
    return log2_loose(input);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
