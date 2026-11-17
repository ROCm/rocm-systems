// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local output rounding shared by small and wide scaled conversions.
#pragma once

#include "conversion/scaled_widen.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v3::scaled_widen_detail {
// Narrow FP32 to F16/BF16 with RNE and optional finite-overflow saturation.
// OCP decoding supplies quiet NaNs; native F16 conversion retains their bits.
inline __m256i encode(__m256 value, ScaledDestination destination, bool saturate) {
  auto bits = _mm256_castps_si256(value);
  if (destination == ScaledDestination::F32) {
    return bits;
  }
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  __m256i result;
  int infinity;
  if (destination == ScaledDestination::F16) {
    result = _mm256_cvtepu16_epi32(
        _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    infinity = 0x7c00;
  } else {
    // Integer BF16 rounding preserves subnormals, unlike VCVTNEPS2BF16.
    auto tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    auto rounded = _mm256_add_epi32(bits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), tie));
    auto special = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff));
    result = _mm256_srli_epi32(_mm256_blendv_epi8(rounded, bits, special), 16);
    infinity = 0x7f80;
  }
  if (saturate) {
    auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
    auto overflow = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(infinity));
    auto saturated = _mm256_sub_epi32(result, _mm256_set1_epi32(1));
    result = _mm256_blendv_epi8(result, saturated, _mm256_and_si256(finite, overflow));
  }
  return result;
}
} // namespace rocjitsu::arith::x86_64_v3::scaled_widen_detail
