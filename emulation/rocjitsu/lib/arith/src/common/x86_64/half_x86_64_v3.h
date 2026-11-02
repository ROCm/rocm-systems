// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16 vector conversion and operand/output-modifier handling.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/isa.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Flush widened FP16 subnormals to signed zero. 0x38800000 is the FP16
// minimum normal (2^-14) represented in FP32, not the FP32 normal threshold.
inline __m256 flush_denorm_f16(__m256 input) {
  auto bits = _mm256_castps_si256(input);
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000),
                                 _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)));
  return _mm256_castsi256_ps(
      _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)), tiny));
}

// Applies FP16 OMOD tiny-result handling before output scaling. OMOD must be
// enabled; values are FP16 arithmetic results represented as FP32.
inline __m256 prepare_omod_f16(__m256 value, uint32_t instruction_flags) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude);
  if ((instruction_flags & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF) {
    auto underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x39000000), magnitude);
    auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(int(0x80000000u)));
    bits = _mm256_blendv_epi8(bits, sign, underflow);
  }
  value = _mm256_castsi256_ps(_mm256_andnot_si256(tiny, bits));
  return value;
}

// Widen the selected FP16 halves and apply ABS then NEG.
inline __m256 half_input(__m256i words, int shift, uint32_t modifiers) {
  auto halves =
      _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                       _mm256_set1_epi32(modifiers & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm256_xor_si256(halves, _mm256_set1_epi32(modifiers & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));

  auto value = _mm256_cvtph_ps(
      _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));

  return value;
}

// Narrow to FP16 with optional saturation of finite overflow.
inline __m256i half_narrow(__m256 value, bool saturate) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));

  auto result =
      _mm256_cvtepu16_epi32(_mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  if (saturate) {
    auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
    auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(0x7c00));
    result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
  }

  return result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
