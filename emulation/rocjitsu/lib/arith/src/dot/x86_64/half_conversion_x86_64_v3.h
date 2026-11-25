// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DOT-specific BF16 conversion: flush subnormals, quiet output NaNs, and ignore
// finite-overflow saturation. FP16 delegates to the ordinary arithmetic helpers.
#pragma once

#include "common/x86_64/half_x86_64_v3.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen packed DOT inputs with ABS/NEG and the format's fixed denormal policy.
template <bool Bf16> __m256 dot_half_input(__m256i words, int shift, uint32_t instruction_flags) {
  if constexpr (Bf16) {
    auto halves = _mm256_and_si256(
        _mm256_srl_epi32(words, _mm_cvtsi32_si128(shift)),
        _mm256_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    halves = _mm256_xor_si256(
        halves, _mm256_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));

    auto zero_exp = _mm256_cmpeq_epi32(_mm256_and_si256(halves, _mm256_set1_epi32(0x7f80)),
                                       _mm256_setzero_si256());
    halves =
        _mm256_blendv_epi8(halves, _mm256_and_si256(halves, _mm256_set1_epi32(0x8000)), zero_exp);
    return _mm256_castsi256_ps(_mm256_slli_epi32(halves, 16));

  } else {
    return half_input<>(words, shift, instruction_flags);
  }
}

// Narrow DOT output; saturation applies only to FP16. BF16 NaNs are quieted.
template <bool Bf16> __m256i dot_half_narrow(__m256 value, bool saturate) {
  if constexpr (Bf16) {
    auto bits = _mm256_castps_si256(value);
    auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));

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
    return half_narrow<>(value, saturate);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
