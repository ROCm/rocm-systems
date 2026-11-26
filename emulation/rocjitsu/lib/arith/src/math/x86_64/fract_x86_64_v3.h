// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FRACT with guest-directed subtraction under the default host FP environment.
#pragma once

#include "common/isa.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return the capped fractional part with selected NaN quieting; source flushing is external.
inline __m256 fract_value_x86_64_v3(__m256 x, unsigned rounding, bool quiet_snan) {
  auto value = _mm256_sub_ps(x, _mm256_floor_ps(x));
  if (rounding) {
    // FastTwoSum: for -0.5<x<0, x-(value-1) is the exact rounding residual.
    // Other finite fractions are exact. Positive residual means the nearest result was low;
    // toward zero uses the downward correction because fractional parts are nonnegative.
    auto small = _mm256_and_ps(_mm256_cmp_ps(x, _mm256_set1_ps(-0.5f), _CMP_GT_OQ),
                               _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ));
    auto residual = _mm256_sub_ps(x, _mm256_sub_ps(value, _mm256_set1_ps(1)));
    auto adjust = _mm256_and_ps(
        small, (rounding == 1 ? _mm256_cmp_ps(residual, _mm256_setzero_ps(), _CMP_GT_OQ)
                              : _mm256_cmp_ps(residual, _mm256_setzero_ps(), _CMP_LT_OQ)));
    auto step = _mm256_and_si256(_mm256_castps_si256(adjust), _mm256_set1_epi32(1));
    auto bits = _mm256_castps_si256(value);
    value = _mm256_castsi256_ps(rounding == 1 ? _mm256_add_epi32(bits, step)
                                              : _mm256_sub_epi32(bits, step));
  }
  value = _mm256_blendv_ps(value, _mm256_setzero_ps(),
                           _mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_EQ_OQ));
  auto limit = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7fffff));
  value = _mm256_blendv_ps(value, limit, _mm256_cmp_ps(value, limit, _CMP_GT_OQ));
  auto bits = _mm256_castps_si256(x);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  auto infinity = _mm256_set1_epi32(0x7f800000);
  auto special =
      _mm256_blendv_epi8(_mm256_set1_epi32(int(0xffc00000U)),
                         _mm256_or_si256(bits, _mm256_set1_epi32(quiet_snan ? 0x00400000 : 0)),
                         _mm256_cmpgt_epi32(magnitude, infinity));
  auto is_special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff));
  return _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(value), special, is_special));
}

// Return the same fractional-part model in four FP64 lanes, including directed rounding.
inline __m256d fract_value_x86_64_v3(__m256d x, unsigned rounding, bool quiet_snan) {
  auto value = _mm256_sub_pd(x, _mm256_floor_pd(x));
  if (rounding) {
    auto small = _mm256_and_pd(_mm256_cmp_pd(x, _mm256_set1_pd(-0.5), _CMP_GT_OQ),
                               _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LT_OQ));
    auto residual = _mm256_sub_pd(x, _mm256_sub_pd(value, _mm256_set1_pd(1)));
    auto adjust = _mm256_and_pd(
        small, (rounding == 1 ? _mm256_cmp_pd(residual, _mm256_setzero_pd(), _CMP_GT_OQ)
                              : _mm256_cmp_pd(residual, _mm256_setzero_pd(), _CMP_LT_OQ)));
    auto step = _mm256_and_si256(_mm256_castpd_si256(adjust), _mm256_set1_epi64x(1));
    auto bits = _mm256_castpd_si256(value);
    value = _mm256_castsi256_pd(rounding == 1 ? _mm256_add_epi64(bits, step)
                                              : _mm256_sub_epi64(bits, step));
  }
  value = _mm256_blendv_pd(value, _mm256_setzero_pd(),
                           _mm256_cmp_pd(value, _mm256_setzero_pd(), _CMP_EQ_OQ));
  auto limit = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3fefffffffffffffLL));
  value = _mm256_blendv_pd(value, limit, _mm256_cmp_pd(value, limit, _CMP_GT_OQ));
  auto bits = _mm256_castpd_si256(x);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi64x(INT64_MAX));
  auto infinity = _mm256_set1_epi64x(0x7ff0000000000000LL);
  auto special = _mm256_blendv_epi8(
      _mm256_set1_epi64x(int64_t(0xfff8000000000000ULL)),
      _mm256_or_si256(bits, _mm256_set1_epi64x(quiet_snan ? 0x0008000000000000LL : 0)),
      _mm256_cmpgt_epi64(magnitude, infinity));
  auto is_special = _mm256_cmpgt_epi64(magnitude, _mm256_set1_epi64x(0x7fefffffffffffffLL));
  return _mm256_castsi256_pd(_mm256_blendv_epi8(_mm256_castpd_si256(value), special, is_special));
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
