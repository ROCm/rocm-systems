// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 FMA with guest rounding and optional FP status.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Return eight FP16 encodings in 32-bit elements after fused rounding and output
// modifiers. Inputs are widened halves with source modifiers already applied.
inline __m256i half_fma_value(__m256 x, __m256 y, __m256 z, bool saturate,
                              uint32_t instruction_flags, bool flush_output = false,
                              bool preserve_nan = false) {
  uint32_t omod = (instruction_flags >> 6) & 3;
  const float scales[] = {1, 2, 4, 0.5f};
  auto scale = _mm256_set1_ps(scales[omod]);

  // This helper serves only Loose dispatch: a native FMA may round before FP16 narrowing.
  auto value = _mm256_fmadd_ps(x, y, z);
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
  if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
    auto negative = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7fff));
    auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                  _mm256_set1_epi32(0x7c00));
    auto clamped = _mm256_andnot_si256(_mm256_or_si256(negative, nan),
                                       _mm256_min_epi32(result, _mm256_set1_epi32(0x3c00)));
    result = preserve_nan ? _mm256_blendv_epi8(clamped, result, nan) : clamped;
  }
  return result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
