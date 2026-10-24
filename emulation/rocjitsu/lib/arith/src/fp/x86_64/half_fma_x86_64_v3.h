// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 FMA with guest rounding and optional FP status.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/x86_64/half_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Return eight FP16 encodings in 32-bit elements after fused rounding and output
// modifiers. Inputs are widened halves with source modifiers already applied.
inline __m256i half_fma_value(__m256 x, __m256 y, __m256 z, bool saturate, uint32_t mode,
                              bool flush_output = false, bool preserve_nan = false) {
  uint32_t omod = (mode >> 6) & 3;
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[omod]);

  // Half products are exact in FP32. TwoSum's residual gives the direction
  // needed for round-to-odd, avoiding double rounding on half-way cases.
  auto product = _mm256_mul_ps(x, y);
  auto value = _mm256_add_ps(product, z);
  auto virtual_c = _mm256_sub_ps(value, product);
  auto error = _mm256_add_ps(_mm256_sub_ps(product, _mm256_sub_ps(value, virtual_c)),
                             _mm256_sub_ps(z, virtual_c));
  auto bits = _mm256_castps_si256(value);
  auto even =
      _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(1)), _mm256_setzero_si256());
  auto adjust = _mm256_and_si256(
      even, _mm256_castps_si256(_mm256_cmp_ps(error, _mm256_setzero_ps(), _CMP_NEQ_OQ)));
  auto opposite = _mm256_srai_epi32(_mm256_xor_si256(bits, _mm256_castps_si256(error)), 31);
  auto step = _mm256_or_si256(opposite, _mm256_set1_epi32(1));
  bits = _mm256_add_epi32(bits, _mm256_and_si256(adjust, step));
  value = _mm256_castsi256_ps(bits);
  auto result = half_narrow(value, saturate);
  if (omod || flush_output) {
    auto magnitude = _mm256_and_si256(result, _mm256_set1_epi32(0x7fff));
    auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x0400), magnitude);
    auto boundary = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x0400));
    auto abs_value = _mm256_and_ps(value, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    auto below = _mm256_castps_si256(
        _mm256_cmp_ps(abs_value, _mm256_set1_ps(0x1p-14f - 0x1p-26f), _CMP_LT_OQ));
    tiny = _mm256_or_si256(tiny, _mm256_and_si256(boundary, below));
    if (omod) {
      auto scaled = half_narrow(_mm256_mul_ps(half_input(result, 0, 0), scale), saturate);
      if (omod == 3) {
        auto newly_tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x0800), magnitude);
        scaled = _mm256_blendv_epi8(scaled, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                                    newly_tiny);
      }
      result = _mm256_andnot_si256(tiny, scaled);
    } else {
      result =
          _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)), tiny);
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    auto negative = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7fff));
    auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                  _mm256_set1_epi32(0x7c00));
    auto clamped = _mm256_andnot_si256(_mm256_or_si256(negative, nan),
                                       _mm256_min_epi32(result, _mm256_set1_epi32(0x3c00)));
    result = preserve_nan ? _mm256_blendv_epi8(clamped, result, nan) : clamped;
  }
  return result;
}

} // namespace rocjitsu::arith
