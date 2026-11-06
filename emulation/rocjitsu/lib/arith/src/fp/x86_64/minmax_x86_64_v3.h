// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 selection implementing the inherited NaN and signed-zero choices.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "rj_arith/rj_arith.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Repair OMOD overflow for guest directed rounding, preserving infinities/NaNs.
// FP16 selections are represented exactly in FP32 before and after scaling.
template <int Width>
inline __m256 selection_omod_round(__m256 before, __m256 scaled, uint64_t flags) {
  static_assert(Width == 16 || Width == 32);
  // OMOD must preserve the selected NaN encoding, including signaling NaNs.
  scaled = _mm256_blendv_ps(scaled, before, _mm256_cmp_ps(before, before, _CMP_UNORD_Q));
  unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if (!rounding) {
    return scaled;
  }
  auto magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), scaled);
  auto limit = _mm256_set1_ps(Width == 16 ? 0x1.ffcp15f : 0x1.fffffep127f);
  auto finite = _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), before),
                              _mm256_set1_ps(__builtin_inff()), _CMP_LT_OQ);
  auto repair = _mm256_and_ps(finite, _mm256_cmp_ps(magnitude, limit, _CMP_GT_OQ));
  if (rounding != 3) {
    auto negative = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(before), 31));
    repair = rounding == 1 ? _mm256_and_ps(repair, negative) : _mm256_andnot_ps(negative, repair);
  }
  auto saturated = _mm256_or_ps(limit, _mm256_and_ps(before, _mm256_set1_ps(-0.0f)));
  return _mm256_blendv_ps(scaled, saturated, repair);
}

// FP64 counterpart: scaling a normal by a power of two only rounds on overflow.
inline __m256d selection_omod_round(__m256d before, __m256d scaled, uint64_t flags) {
  // OMOD must preserve the selected NaN encoding, including signaling NaNs.
  scaled = _mm256_blendv_pd(scaled, before, _mm256_cmp_pd(before, before, _CMP_UNORD_Q));
  unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  if (!rounding) {
    return scaled;
  }
  auto magnitude = _mm256_andnot_pd(_mm256_set1_pd(-0.0), scaled);
  auto limit = _mm256_set1_pd(0x1.fffffffffffffp1023);
  auto finite = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), before),
                              _mm256_set1_pd(__builtin_inf()), _CMP_LT_OQ);
  auto repair = _mm256_and_pd(finite, _mm256_cmp_pd(magnitude, limit, _CMP_GT_OQ));
  if (rounding != 3) {
    auto negative = _mm256_cmp_pd(before, _mm256_setzero_pd(), _CMP_LT_OQ);
    repair = rounding == 1 ? _mm256_and_pd(repair, negative) : _mm256_andnot_pd(negative, repair);
  }
  auto saturated = _mm256_or_pd(limit, _mm256_and_pd(before, _mm256_set1_pd(-0.0)));
  return _mm256_blendv_pd(scaled, saturated, repair);
}

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
// Requires the default CPU FP environment. Bitwise NaN selection/quieting keeps
// the Compatibility payload contract independent of compiler FP optimizations.
template <bool Maximum, bool PropagateNaN> __m256 minmax(__m256 x, __m256 y) {
  auto x_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
  auto quiet_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x00400000));
  if constexpr (PropagateNaN) {
    // Select the first NaN before setting its quiet bit; preserve sign/payload.
    auto nan = _mm256_or_ps(_mm256_blendv_ps(y, x, x_nan), quiet_bit);
    auto either_nan = _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
    // Opposite operand orders expose both zero signs; AND selects +0, OR -0.
    auto value = Maximum ? _mm256_and_ps(_mm256_max_ps(x, y), _mm256_max_ps(y, x))
                         : _mm256_or_ps(_mm256_min_ps(x, y), _mm256_min_ps(y, x));
    return _mm256_blendv_ps(value, nan, either_nan);
  } else {
    auto x_quiet = _mm256_or_ps(x, _mm256_and_ps(x_nan, quiet_bit));
    auto equal = _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
    auto y_nan = _mm256_cmp_ps(y, y, _CMP_UNORD_Q);
    auto value = Maximum ? _mm256_max_ps(x, y) : _mm256_min_ps(x, y);
    auto tie = Maximum ? _mm256_and_ps(x, y) : _mm256_or_ps(x, y);
    value = _mm256_blendv_ps(value, tie, equal);
    return _mm256_blendv_ps(value, x_quiet, y_nan);
  }
}

// Correct the x86 selection rules for NaNs and opposite-signed zeros.
template <bool Maximum, bool PropagateNaN> __m256d minmax(__m256d x, __m256d y) {
  auto x_nan = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
  auto quiet_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(0x0008000000000000LL));
  if constexpr (PropagateNaN) {
    // Select the first NaN before setting its quiet bit; preserve sign/payload.
    auto nan = _mm256_or_pd(_mm256_blendv_pd(y, x, x_nan), quiet_bit);
    auto either_nan = _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
    // Opposite operand orders expose both zero signs; AND selects +0, OR -0.
    auto value = Maximum ? _mm256_and_pd(_mm256_max_pd(x, y), _mm256_max_pd(y, x))
                         : _mm256_or_pd(_mm256_min_pd(x, y), _mm256_min_pd(y, x));
    return _mm256_blendv_pd(value, nan, either_nan);
  } else {
    auto x_quiet = _mm256_or_pd(x, _mm256_and_pd(x_nan, quiet_bit));
    auto equal = _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
    auto y_nan = _mm256_cmp_pd(y, y, _CMP_UNORD_Q);
    auto value = Maximum ? _mm256_max_pd(x, y) : _mm256_min_pd(x, y);
    auto tie = Maximum ? _mm256_and_pd(x, y) : _mm256_or_pd(x, y);
    value = _mm256_blendv_pd(value, tie, equal);
    return _mm256_blendv_pd(value, x_quiet, y_nan);
  }
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

// Select A/B then C. Median orders -0 below +0 and returns the chained
// minimumNumber if any input is NaN. These encoding choices exceed the loose API guarantees.
template <bool FirstMaximum, bool SecondMaximum, bool PropagateNaN, bool Median = false>
__m256 minmax3_value(__m256 x, __m256 y, __m256 z) {
  __m256 value;
  if constexpr (Median) {
    value = minmax<true, false>(minmax<false, false>(x, y),
                                minmax<false, false>(minmax<true, false>(x, y), z));
    auto has_nan =
        _mm256_or_ps(_mm256_cmp_ps(x, y, _CMP_UNORD_Q), _mm256_cmp_ps(z, z, _CMP_UNORD_Q));
    auto minimum = minmax<false, false>(minmax<false, false>(x, y), z);
    value = _mm256_blendv_ps(value, minimum, has_nan);
  } else {
    auto ab = minmax<FirstMaximum, PropagateNaN>(x, y);
    value = minmax<SecondMaximum, PropagateNaN>(ab, z);
  }
  return value;
}

// Apply RDNA3's binary selection policy at each step of the ternary operation.
template <bool FirstMaximum, bool SecondMaximum, bool Median>
__m256 minmax3_rdna3(__m256 x, __m256 y, __m256 z, bool ieee) {
  if constexpr (Median) {
    // Match the ordered-zero median used by the scalar RDNA3 model.
    auto value =
        minmax_rdna3<true>(minmax_rdna3<false>(x, y, ieee),
                           minmax_rdna3<false>(minmax_rdna3<true>(x, y, ieee), z, ieee), ieee);
    auto has_nan =
        _mm256_or_ps(_mm256_cmp_ps(x, y, _CMP_UNORD_Q), _mm256_cmp_ps(z, z, _CMP_UNORD_Q));
    auto minimum = minmax_rdna3<false>(minmax_rdna3<false>(x, y, ieee), z, ieee);
    return _mm256_blendv_ps(value, minimum, has_nan);
  }
  return minmax_rdna3<SecondMaximum>(minmax_rdna3<FirstMaximum>(x, y, ieee), z, ieee);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
