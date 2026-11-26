// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DOT-specific BF16 conversion: flush subnormals, quiet output NaNs, and ignore
// finite-overflow saturation. FP16 delegates to the ordinary arithmetic helpers.
#pragma once

#include "common/x86_64/half_x86_64_v4.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen packed DOT inputs with ABS/NEG and the format's fixed denormal policy.
template <bool Bf16>
inline __m512 dot_half_input(__m512i words, int shift, uint32_t instruction_flags) {
  if constexpr (Bf16) {
    auto halves = _mm512_and_si512(
        _mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
        _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
    halves = _mm512_xor_si512(
        halves, _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));

    auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(halves, _mm512_set1_epi32(0x7f80)),
                                        _mm512_setzero_si512());
    halves = _mm512_mask_and_epi32(halves, tiny, halves, _mm512_set1_epi32(0x8000));
    return _mm512_castsi512_ps(_mm512_slli_epi32(halves, 16));

  } else {
    return half_input(words, shift, instruction_flags);
  }
}

// Narrow DOT output; saturation applies only to FP16. BF16 NaNs are quieted.
template <bool Bf16> inline __m512i dot_half_narrow(__m512 value, bool saturate) {
  if constexpr (Bf16) {
    auto bits = _mm512_castps_si512(value);
    auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));

    auto tie = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    auto result = _mm512_srli_epi32(
        _mm512_add_epi32(bits, _mm512_add_epi32(tie, _mm512_set1_epi32(0x7fff))), 16);
    auto nan = _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    result = _mm512_mask_mov_epi32(
        result, nan, _mm512_or_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x40)));
    auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7f80)),
                                        _mm512_setzero_si512());
    return _mm512_mask_and_epi32(result, tiny, result, _mm512_set1_epi32(0x8000));

  } else {
    return half_narrow(value, saturate);
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
