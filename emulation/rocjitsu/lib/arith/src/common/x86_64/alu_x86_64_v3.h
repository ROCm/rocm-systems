// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 vector saturation, denormal flushing, and output-modifier helpers.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/isa.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Clamp FP32 lanes to [0,1], mapping either sign of zero to +0. Optionally
// preserve arithmetic NaNs, including sign/payload, instead of mapping to zero.
inline __m256 clamp_f32(__m256 value, bool preserve_nan) {
  if (preserve_nan) {
    // MIN/MAX select their second operand for unordered inputs.
    value = _mm256_min_ps(_mm256_set1_ps(1), _mm256_max_ps(_mm256_setzero_ps(), value));
    auto zero = _mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_EQ_OQ);
    return _mm256_andnot_ps(zero, value);
  }
  return _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
}

// Replace FP32 subnormals with zero of the same sign.
inline __m256 flush_denorm_f32(__m256 value) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
  auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN));
  return _mm256_blendv_ps(value, _mm256_castsi256_ps(sign), _mm256_castsi256_ps(tiny));
}

// Apply the FP32 input/output tininess rules associated with active OMOD scaling.
inline __m256 prepare_omod_f32(__m256 value, uint32_t instruction_flags) {
  auto magnitude = _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x00800000), magnitude);
  value = _mm256_andnot_ps(_mm256_castsi256_ps(tiny), value);
  if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF) {
    auto underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x01000000), magnitude);
    auto sign = _mm256_and_ps(value, _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MIN)));
    value = _mm256_blendv_ps(value, sign, _mm256_castsi256_ps(underflow));
  }
  return value;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
