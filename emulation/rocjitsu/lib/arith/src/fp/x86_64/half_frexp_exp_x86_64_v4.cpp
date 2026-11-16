// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 half exponent extraction uses integer classification and leading-zero
// counts, including subnormals. No host FP arithmetic or environment controls.

#include "fp/half_exponent.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Extract exponents for active lanes, preserving other halves and inactive storage.
// Requires x86-64-v4 and validated flags; A and D may alias.
void half_frexp_exp_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a) {
  const int a_shift = instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  const int d_shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  const auto half_mask = _mm512_set1_epi32(0xffff);
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(a + lane);
    auto magnitude = _mm512_and_si512(_mm512_srl_epi32(raw, _mm_cvtsi32_si128(a_shift)),
                                      _mm512_set1_epi32(0x7fff));
    auto field = _mm512_srli_epi32(magnitude, 10);
    auto fraction = _mm512_and_si512(magnitude, _mm512_set1_epi32(0x3ff));
    const auto subnormal = _mm512_cmpeq_epi32_mask(field, _mm512_setzero_si512());
    // The highest fraction bit p encodes 2^(p-24), whose FREXP exponent is p-23.
    auto subnormal_exponent = _mm512_sub_epi32(_mm512_set1_epi32(8), _mm512_lzcnt_epi32(fraction));
    auto result = _mm512_mask_blend_epi32(subnormal, _mm512_sub_epi32(field, _mm512_set1_epi32(14)),
                                          subnormal_exponent);
    auto special = _mm512_cmpeq_epi32_mask(magnitude, _mm512_setzero_si512()) |
                   _mm512_cmpeq_epi32_mask(field, _mm512_set1_epi32(31));
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      special |= subnormal;
    }
    result = _mm512_maskz_mov_epi32(__mmask16(~special), _mm512_and_si512(result, half_mask));
    const auto shifted_mask = _mm512_sll_epi32(half_mask, _mm_cvtsi32_si128(d_shift));
    result = _mm512_or_si512(_mm512_sll_epi32(result, _mm_cvtsi32_si128(d_shift)),
                             _mm512_andnot_si512(shifted_mask, _mm512_loadu_si512(d + lane)));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
