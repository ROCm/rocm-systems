// SPDX-License-Identifier: MIT

// AVX2 vector floating-point selection with GPU NaN and signed-zero policies.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool Propagate> __m256 minmax(__m256 x, __m256 y) {
  auto a = _mm256_castps_si256(x), b = _mm256_castps_si256(y);
  auto magnitude = _mm256_set1_epi32(INT32_MAX);
  auto infinity = _mm256_set1_epi32(0x7f800000);
  auto quiet = _mm256_set1_epi32(0x00400000);
  auto an = _mm256_cmpgt_epi32(_mm256_and_si256(a, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi32(_mm256_and_si256(b, magnitude), infinity);
  auto value = Maximum ? _mm256_max_ps(x, y) : _mm256_min_ps(x, y);
  auto equal = _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
  auto tie = Maximum ? _mm256_and_si256(a, b) : _mm256_or_si256(a, b);
  value = _mm256_blendv_ps(value, _mm256_castsi256_ps(tie), equal);
  if constexpr (Propagate) {
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b, quiet), quiet), bn);
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(a, quiet), quiet), an);
    auto selected = _mm256_blendv_epi8(b, a, an);
    selected = _mm256_blendv_epi8(selected, b, snb);
    selected = _mm256_blendv_epi8(selected, a, sna);
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(selected, quiet)),
                             _mm256_castsi256_ps(_mm256_or_si256(an, bn)));
  } else {
    value = _mm256_blendv_ps(value, x, _mm256_castsi256_ps(bn));
    value = _mm256_blendv_ps(value, y, _mm256_castsi256_ps(an));
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(a, quiet)),
                             _mm256_castsi256_ps(_mm256_and_si256(an, bn)));
  }
  return value;
}

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool Propagate> __m256d minmax(__m256d x, __m256d y) {
  auto a = _mm256_castpd_si256(x), b = _mm256_castpd_si256(y);
  auto magnitude = _mm256_set1_epi64x(INT64_MAX);
  auto infinity = _mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
  auto quiet = _mm256_set1_epi64x(INT64_C(0x0008000000000000));
  auto an = _mm256_cmpgt_epi64(_mm256_and_si256(a, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi64(_mm256_and_si256(b, magnitude), infinity);
  auto value = Maximum ? _mm256_max_pd(x, y) : _mm256_min_pd(x, y);
  auto equal = _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
  auto tie = Maximum ? _mm256_and_si256(a, b) : _mm256_or_si256(a, b);
  value = _mm256_blendv_pd(value, _mm256_castsi256_pd(tie), equal);
  if constexpr (Propagate) {
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(b, quiet), quiet), bn);
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(a, quiet), quiet), an);
    auto selected = _mm256_blendv_epi8(b, a, an);
    selected = _mm256_blendv_epi8(selected, b, snb);
    selected = _mm256_blendv_epi8(selected, a, sna);
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(selected, quiet)),
                             _mm256_castsi256_pd(_mm256_or_si256(an, bn)));
  } else {
    value = _mm256_blendv_pd(value, x, _mm256_castsi256_pd(bn));
    value = _mm256_blendv_pd(value, y, _mm256_castsi256_pd(an));
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(a, quiet)),
                             _mm256_castsi256_pd(_mm256_and_si256(an, bn)));
  }
  return value;
}

// RDNA3 selection adds IEEE-mode signaling-NaN priority to number preference.
template <bool Maximum> __m256 minmax_rdna3(__m256 x, __m256 y, bool ieee) {
  auto value = minmax<Maximum, false>(x, y);
  auto a_bits = _mm256_castps_si256(x), b_bits = _mm256_castps_si256(y);
  auto magnitude = _mm256_set1_epi32(INT32_MAX);
  auto infinity = _mm256_set1_epi32(0x7f800000);
  auto quiet = _mm256_set1_epi32(0x00400000);
  auto an = _mm256_cmpgt_epi32(_mm256_and_si256(a_bits, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi32(_mm256_and_si256(b_bits, magnitude), infinity);
  if (ieee) {
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(a_bits, quiet), quiet), an);
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(b_bits, quiet), quiet), bn);
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(b_bits, quiet)),
                             _mm256_castsi256_ps(snb));
    value = _mm256_blendv_ps(value, _mm256_castsi256_ps(_mm256_or_si256(a_bits, quiet)),
                             _mm256_castsi256_ps(sna));
  } else {
    value = _mm256_blendv_ps(value, x, _mm256_castsi256_ps(_mm256_and_si256(an, bn)));
  }
  return value;
}

// RDNA3 selection adds IEEE-mode signaling-NaN priority to number preference.
template <bool Maximum> __m256d minmax_rdna3(__m256d x, __m256d y, bool ieee) {
  auto value = minmax<Maximum, false>(x, y);
  auto a_bits = _mm256_castpd_si256(x), b_bits = _mm256_castpd_si256(y);
  auto magnitude = _mm256_set1_epi64x(INT64_MAX);
  auto infinity = _mm256_set1_epi64x(0x7ff0000000000000LL);
  auto quiet = _mm256_set1_epi64x(0x0008000000000000LL);
  auto an = _mm256_cmpgt_epi64(_mm256_and_si256(a_bits, magnitude), infinity);
  auto bn = _mm256_cmpgt_epi64(_mm256_and_si256(b_bits, magnitude), infinity);
  if (ieee) {
    auto sna = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(a_bits, quiet), quiet), an);
    auto snb = _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(b_bits, quiet), quiet), bn);
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(b_bits, quiet)),
                             _mm256_castsi256_pd(snb));
    value = _mm256_blendv_pd(value, _mm256_castsi256_pd(_mm256_or_si256(a_bits, quiet)),
                             _mm256_castsi256_pd(sna));
  } else {
    value = _mm256_blendv_pd(value, x, _mm256_castsi256_pd(_mm256_and_si256(an, bn)));
  }
  return value;
}

// Select A/B first and then C, including the median's NaN and signed-zero rules.
// Return the selected three-input minimum, maximum, median, or combined selection with GPU NaN
// rules.
template <bool FirstMaximum, bool SecondMaximum, bool Propagate, bool Median = false,
          bool OrderedMedian = false>
__m256 minmax3_value(__m256 x, __m256 y, __m256 z) {
  __m256 value;
  if constexpr (Median) {
    if constexpr (OrderedMedian) {
      value = minmax<true, false>(minmax<false, false>(x, y),
                                  minmax<false, false>(minmax<true, false>(x, y), z));
    } else {
      auto ab = minmax<true, false>(x, y);
      auto maximum = minmax<true, false>(ab, z);
      value =
          _mm256_blendv_ps(ab, minmax<true, false>(x, z), _mm256_cmp_ps(maximum, y, _CMP_EQ_OQ));
      value =
          _mm256_blendv_ps(value, minmax<true, false>(y, z), _mm256_cmp_ps(maximum, x, _CMP_EQ_OQ));
    }
    auto has_nan =
        _mm256_or_ps(_mm256_cmp_ps(x, y, _CMP_UNORD_Q), _mm256_cmp_ps(z, z, _CMP_UNORD_Q));
    auto minimum = minmax<false, false>(minmax<false, false>(x, y), z);
    value = _mm256_blendv_ps(value, minimum, has_nan);
  } else {
    auto ab = minmax<FirstMaximum, Propagate>(x, y);
    value = minmax<SecondMaximum, Propagate>(ab, z);
  }
  return value;
}

// Apply RDNA3's binary selection policy at each step of the ternary operation.
template <bool FirstMaximum, bool SecondMaximum, bool Median>
__m256 minmax3_rdna3(__m256 x, __m256 y, __m256 z, bool ieee) {
  if constexpr (Median) {
    auto ab = minmax_rdna3<true>(x, y, ieee);
    auto maximum = minmax_rdna3<true>(ab, z, ieee);
    auto value =
        _mm256_blendv_ps(ab, minmax_rdna3<true>(x, z, ieee), _mm256_cmp_ps(maximum, y, _CMP_EQ_OQ));
    value = _mm256_blendv_ps(value, minmax_rdna3<true>(y, z, ieee),
                             _mm256_cmp_ps(maximum, x, _CMP_EQ_OQ));
    auto has_nan =
        _mm256_or_ps(_mm256_cmp_ps(x, y, _CMP_UNORD_Q), _mm256_cmp_ps(z, z, _CMP_UNORD_Q));
    auto minimum = minmax_rdna3<false>(minmax_rdna3<false>(x, y, ieee), z, ieee);
    return _mm256_blendv_ps(value, minimum, has_nan);
  }
  return minmax_rdna3<SecondMaximum>(minmax_rdna3<FirstMaximum>(x, y, ieee), z, ieee);
}

} // namespace rocjitsu::arith
