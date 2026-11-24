// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-local output rounding shared by small and wide scaled conversions.
#pragma once

#include "conversion/scaled_widen.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::x86_64_v4::scaled_widen_detail {
// Narrow FP32 to F16/BF16 with RNE and optional finite-overflow saturation.
// OCP decoding supplies quiet NaNs; native F16 conversion retains their bits.
inline __m512i encode(__m512 value, ScaledDestination destination, bool saturate) {
  auto bits = _mm512_castps_si512(value);
  if (destination == ScaledDestination::F32) {
    return bits;
  }
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
  __m512i result;
  int infinity;
  if (destination == ScaledDestination::F16) {
    result = _mm512_cvtepu16_epi32(
        _mm512_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    infinity = 0x7c00;
  } else {
    // Integer BF16 rounding preserves subnormals, unlike VCVTNEPS2BF16.
    auto tie = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    auto rounded = _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), tie));
    auto special = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f7fffff));
    result = _mm512_srli_epi32(_mm512_mask_blend_epi32(special, rounded, bits), 16);
    infinity = 0x7f80;
  }
  if (saturate) {
    auto finite = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(0x7f800000), magnitude);
    auto overflow = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(infinity));
    result = _mm512_mask_sub_epi32(result, finite & overflow, result, _mm512_set1_epi32(1));
  }
  return result;
}
} // namespace rocjitsu::arith::x86_64_v4::scaled_widen_detail
