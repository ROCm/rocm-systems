// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FP16 vector conversion and operand/output-modifier handling.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "common/bits.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Widen one selected FP16 half per lane and apply source sign instruction_flags.
inline __m512 half_input(__m512i words, int shift, uint32_t instruction_flags) {
  auto halves =
      _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)),
                       _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_A ? 0x7fff : 0xffff));
  halves = _mm512_xor_si512(halves,
                            _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_A ? 0x8000 : 0));

  return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(halves));
}

// Nearest-even narrowing with the same finite-saturation/NaN policy as the
// existing AVX2 arithmetic helper.
inline __m512i half_narrow(__m512 value, bool saturate) {
  auto bits = _mm512_castps_si512(value);
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));

  auto result =
      _mm512_cvtepu16_epi32(_mm512_cvtps_ph(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  if (saturate) {
    auto finite = _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
    auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(0x7c00));
    result = _mm512_mask_sub_epi32(result, finite & infinity, result, _mm512_set1_epi32(1));
  }
  return result;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
