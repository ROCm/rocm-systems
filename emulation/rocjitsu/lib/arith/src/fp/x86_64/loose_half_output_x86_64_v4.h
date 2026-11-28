// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose FP16 output rounding shared by ordinary, packed and mixed FMA.
#pragma once

#include "common/isa.h"
#include "common/x86_64/half_x86_64_v4.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Narrow a Loose FP32 result, retaining the existing FP16 output controls.
// Requires nearest guest rounding. Scaling consumes a rounded half result.
// Host controls are unchanged.
inline __m512i loose_half_output(__m512 value, uint64_t flags, uint32_t mode) {
  const bool saturate = flags & RJ_ARITH_FP16_OVFL;
  const unsigned omod = (mode >> 6) & 3;
  auto result = half_narrow(value, saturate);
  if (omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    const auto magnitude = _mm512_and_si512(result, _mm512_set1_epi32(0x7fff));
    auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
    // Reaching minimum normal only on the subnormal rounding grid is still tiny.
    const auto boundary = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
    const auto abs_value = _mm512_and_ps(value, _mm512_castsi512_ps(_mm512_set1_epi32(INT32_MAX)));
    tiny |=
        boundary & _mm512_cmp_ps_mask(abs_value, _mm512_set1_ps(0x1p-14f - 0x1p-26f), _CMP_LT_OQ);
    if (omod) {
      const float scales[] = {1, 2, 4, 0.5f};
      auto scaled = half_narrow(
          _mm512_mul_ps(half_input(result, 0, 0), _mm512_set1_ps(scales[omod])), saturate);
      // Dividing the rounded half by two can create another tiny result.
      if (omod == 3) {
        const auto newly_tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x800));
        scaled = _mm512_mask_and_epi32(scaled, newly_tiny, result, _mm512_set1_epi32(0x8000));
      }
      result = _mm512_mask_mov_epi32(scaled, tiny, _mm512_setzero_si512());
    } else {
      result = _mm512_mask_and_epi32(result, tiny, result, _mm512_set1_epi32(0x8000));
    }
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    const auto nan = _mm512_cmpgt_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                             _mm512_set1_epi32(0x7c00));
    const auto negative = _mm512_test_epi32_mask(result, _mm512_set1_epi32(0x8000));
    auto clamped = _mm512_mask_mov_epi32(result, negative | nan, _mm512_setzero_si512());
    clamped = _mm512_min_epu32(clamped, _mm512_set1_epi32(0x3c00));
    result =
        flags & RJ_ARITH_FP_DX10_CLAMP_OFF ? _mm512_mask_mov_epi32(clamped, nan, result) : clamped;
  }
  return result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
