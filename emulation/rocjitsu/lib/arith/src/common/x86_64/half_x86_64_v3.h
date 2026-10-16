// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP16/BF16 vector conversion and operand/output-modifier handling.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Applies FP16 OMOD tiny-result handling before output scaling. OMOD must be
// enabled; values are FP16 arithmetic results represented as FP32.
inline __m256 prepare_omod_f16(__m256 value, uint32_t mode) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000), magnitude);
  if ((mode & RJ_ARITH_ALU_OMOD_HALF) == RJ_ARITH_ALU_OMOD_HALF) {
    auto underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x39000000), magnitude);
    auto sign = _mm256_and_si256(bits, _mm256_set1_epi32(int(0x80000000u)));
    bits = _mm256_blendv_epi8(bits, sign, underflow);
  }
  value = _mm256_castsi256_ps(_mm256_andnot_si256(tiny, bits));
  return value;
}

// Widen the selected halves with ABS then NEG. PreserveNan retains signaling
// bits through x86 conversion for selection operations.
template <bool Bf16 = false, bool PreserveNan = false>
__m256 half_input(__m256i words, int shift, uint32_t mode) {
  auto halves = _mm256_and_si256(_mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                 _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm256_xor_si256(halves, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
  if constexpr (Bf16) {
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(halves, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    halves =
        _mm256_blendv_epi8(halves, _mm256_and_si256(halves, _mm256_set1_epi32(0x8000)), zero_exp);
    return _mm256_castsi256_ps(_mm256_slli_epi32(halves, 16));
  } else {
    auto value = _mm256_cvtph_ps(
        _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
    if constexpr (PreserveNan) {
      // All exponent bits set, quiet bit clear: sNaN or infinity. F16C only
      // changes the former's quiet bit; clearing that bit is harmless for infinity.
      auto signaling = _mm256_cmpeq_epi32(_mm256_and_si256(halves, _mm256_set1_epi32(0x7e00)),
                                          _mm256_set1_epi32(0x7c00));
      auto clear_quiet = _mm256_and_si256(signaling, _mm256_set1_epi32(0x00400000));
      value = _mm256_andnot_ps(_mm256_castsi256_ps(clear_quiet), value);
    }
    return value;
  }
}

// Narrow to FP16/BF16 with optional finite-overflow saturation. PreserveNan leaves
// the selection operation in control of NaN quieting.
template <bool Bf16 = false, bool PreserveNan = false>
__m256i half_narrow(__m256 value, bool saturate) {
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  if constexpr (Bf16) {
    auto tie = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    auto result = _mm256_srli_epi32(
        _mm256_add_epi32(bits, _mm256_add_epi32(tie, _mm256_set1_epi32(0x7fff))), 16);
    auto nan = _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000));
    result = _mm256_blendv_epi8(
        result, _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x40)), nan);
    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    return _mm256_blendv_epi8(result, _mm256_and_si256(result, _mm256_set1_epi32(0x8000)),
                              zero_exp);
  } else {
    auto result = _mm256_cvtepu16_epi32(
        _mm256_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    if (saturate) {
      auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
      auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                         _mm256_set1_epi32(0x7c00));
      result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
    }
    if constexpr (PreserveNan) {
      // These NaNs originate in FP16, so narrowing retains their payload.
      auto signaling = _mm256_cmpeq_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x7fc00000)),
                                          _mm256_set1_epi32(0x7f800000));
      auto clear_quiet = _mm256_and_si256(signaling, _mm256_set1_epi32(0x0200));
      result = _mm256_andnot_si256(clear_quiet, result);
    }
    return result;
  }
}

} // namespace rocjitsu::arith
