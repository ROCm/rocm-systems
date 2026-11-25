// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local v3 FP8/BF8 rounding, shared by scaled and unscaled conversions.
#pragma once
#include "conversion/fp8_narrow.h"
#include <immintrin.h>

namespace rocjitsu::arith::x86_64_v3::fp8_narrow_detail {

using rocjitsu::arith::Fp8NarrowPolicy;

// Select yes where signed a > b, and no otherwise.
inline __m256i select_greater(__m256i a, __m256i b, __m256i yes, __m256i no) {
  return _mm256_blendv_epi8(no, yes, _mm256_cmpgt_epi32(a, b));
}

// Narrow FP32 bits to FP8/BF8 using nearest or seeded stochastic rounding and optional
// saturation.
template <bool Bf8, bool Stochastic>
inline __m256i narrow(__m256i raw, __m256i seed, bool saturate, Fp8NarrowPolicy policy) {
  constexpr int fraction_bits = Bf8 ? 2 : 3;
  bool fnuz = policy == Fp8NarrowPolicy::Fnuz;
  bool unsigned_format = policy == Fp8NarrowPolicy::E5M3;
  bool rdna = policy == Fp8NarrowPolicy::Gfx12;
  int terminal = fnuz ? 128 : unsigned_format ? 255 : Bf8 ? 124 : 127;
  int bias = unsigned_format ? 15 : (Bf8 ? 15 : 7) + int(fnuz);
  auto one = _mm256_set1_epi32(1), zero = _mm256_setzero_si256();
  auto magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(0x7fffffff));
  auto exponent = _mm256_sub_epi32(_mm256_srli_epi32(magnitude, 23), _mm256_set1_epi32(127 - bias));
  auto significand = _mm256_or_si256(_mm256_and_si256(magnitude, _mm256_set1_epi32(0x7fffff)),
                                     _mm256_set1_epi32(0x800000));
  auto alignment = _mm256_max_epi32(_mm256_sub_epi32(one, exponent), zero);
  __m256i result;
  if constexpr (Stochastic) {
    if (rdna) {
      significand = _mm256_srlv_epi32(significand, alignment);
      result = _mm256_srli_epi32(
          _mm256_add_epi32(significand, _mm256_srli_epi32(seed, 9 + fraction_bits)),
          23 - fraction_bits);
    } else {
      auto shift = _mm256_add_epi32(alignment, _mm256_set1_epi32(23 - fraction_bits));
      auto random = _mm256_srlv_epi32(seed, _mm256_sub_epi32(_mm256_set1_epi32(32), shift));
      result = _mm256_srlv_epi32(_mm256_add_epi32(significand, random), shift);
      result = select_greater(shift, _mm256_set1_epi32(24), zero, result);
    }
  } else {
    auto shift = _mm256_min_epi32(
        _mm256_add_epi32(alignment, _mm256_set1_epi32(23 - fraction_bits)), _mm256_set1_epi32(31));
    auto half = _mm256_sllv_epi32(one, _mm256_sub_epi32(shift, one));
    auto odd = _mm256_and_si256(_mm256_srlv_epi32(significand, shift), one);
    auto rounded =
        _mm256_add_epi32(significand, _mm256_add_epi32(_mm256_sub_epi32(half, one), odd));
    result = _mm256_srlv_epi32(rounded, shift);
  }
  auto normal_exponent =
      _mm256_slli_epi32(_mm256_max_epi32(_mm256_sub_epi32(exponent, one), zero), fraction_bits);
  result = _mm256_add_epi32(result, normal_exponent);
  auto limit = _mm256_set1_epi32(terminal);
  if (saturate) {
    auto finite_limit = _mm256_set1_epi32(terminal - 1);
    limit = rdna ? select_greater(_mm256_set1_epi32(0x7f800000), magnitude, finite_limit, limit)
                 : finite_limit;
  }
  result = _mm256_min_epu32(result, limit);
  auto sign =
      unsigned_format ? zero : _mm256_and_si256(_mm256_srli_epi32(raw, 24), _mm256_set1_epi32(128));
  if (fnuz) {
    sign = select_greater(result, zero, sign, zero);
    auto signed_result = _mm256_or_si256(result, sign);
    result = select_greater(result, _mm256_set1_epi32(127), _mm256_set1_epi32(128), signed_result);
  } else {
    result = _mm256_or_si256(result, sign);
  }
  auto nan = fnuz              ? _mm256_set1_epi32(128)
             : unsigned_format ? _mm256_set1_epi32(255)
             : rdna            ? _mm256_set1_epi32(Bf8 ? 0xfe : 0xff)
                               : _mm256_or_si256(sign, _mm256_set1_epi32(127));
  return select_greater(magnitude, _mm256_set1_epi32(0x7f800000), nan, result);
}

} // namespace rocjitsu::arith::x86_64_v3::fp8_narrow_detail
