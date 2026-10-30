// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP16/BF16 vector conversion and operand/output-modifier handling.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen one selected FP16/BF16 half per lane and apply source sign modifiers.
template <bool Bf16> inline __m512 half_input(__m512i words, int shift, uint32_t mode) {
  auto halves = _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                                 _mm512_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm512_xor_si512(halves, _mm512_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));
  if constexpr (Bf16) {
    auto tiny = _mm512_cmpeq_epi32_mask(_mm512_and_si512(halves, _mm512_set1_epi32(0x7f80)),
                                        _mm512_setzero_si512());
    halves = _mm512_mask_and_epi32(halves, tiny, halves, _mm512_set1_epi32(0x8000));
    return _mm512_castsi512_ps(_mm512_slli_epi32(halves, 16));
  } else {
    return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(halves));
  }
}

// Nearest-even narrowing with the same finite-saturation/NaN policy as the
// existing AVX2 arithmetic helper. BF16 subnormal results are flushed.
template <bool Bf16> inline __m512i half_narrow(__m512 value, bool saturate) {
  auto bits = _mm512_castps_si512(value);
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  if constexpr (Bf16) {
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
    auto result = _mm512_cvtepu16_epi32(
        _mm512_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    if (saturate) {
      auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
      auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                              _mm512_set1_epi32(0x7c00));
      result = _mm512_mask_sub_epi32(result, finite & infinity, result, _mm512_set1_epi32(1));
    }
    return result;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
