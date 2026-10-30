// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 vector saturation, denormal flushing, and output-modifier helpers.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/isa.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Clamp to [0,1], optionally preserving NaNs; zero results become positive zero.
inline __m512 clamp_f32(__m512 value, bool preserve_nan) {
  if (preserve_nan) {
    // Keep value as the second MIN/MAX operand so NaNs survive.
    value = _mm512_min_ps(_mm512_set1_ps(1), _mm512_max_ps(_mm512_setzero_ps(), value));
    auto zero = _mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_EQ_OQ);
    return _mm512_mask_mov_ps(value, zero, _mm512_setzero_ps());
  }
  return _mm512_min_ps(_mm512_max_ps(value, _mm512_setzero_ps()), _mm512_set1_ps(1));
}

// Replace FP32 subnormals with zero of the same sign.
inline __m512 flush_denorm_f32(__m512 value) {
  auto bits = _mm512_castps_si512(value);
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x00800000));
  auto sign = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MIN));
  return _mm512_mask_mov_ps(value, tiny, _mm512_castsi512_ps(sign));
}

// Prepare FP32 results for a nonzero OMOD: tiny values become +0, while
// halving normal values below twice minimum normal produces signed zero.
inline __m512 prepare_omod_f32(__m512 value, uint32_t instruction_flags) {
  auto magnitude = _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MAX));
  auto tiny = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x00800000));
  value = _mm512_mask_mov_ps(value, tiny, _mm512_setzero_ps());
  if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF) {
    auto underflow = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x01000000));
    auto sign = _mm512_castsi512_ps(
        _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MIN)));
    value = _mm512_mask_mov_ps(value, underflow, sign);
  }
  return value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
