// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native BF16 DOT2 for Loose FP32 and packed BF16 results.
// Pair-dot rounding and flushing are accepted without scalar qualification or repair.

#include "common/bits.h"
#include "common/simd.h"
#include "dot/operand_halves.h"
#include "integer/x86_64/packed_integer_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
// Accumulate two BF16 products per lane, preserving EXEC and the unselected output half.
// Packed results use ALU modifiers; FP32 results use DOT half/sign selection.
template <int Lanes, bool Packed>
void dot2_bf16_x86_64_v4_bf16(uint64_t exec_mask, uint32_t controls, uint32_t *d, const uint32_t *a,
                              const uint32_t *b, const uint32_t *c) {
  __m512i select_a{}, select_b{};
  if constexpr (!Packed) {
    if (controls & kDotHalfSelection) {
      select_a = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_A_HIGH,
                                                                 controls & RJ_ARITH_DOT_HI_A_LOW));
      select_b = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_B_HIGH,
                                                                 controls & RJ_ARITH_DOT_HI_B_LOW));
    }
  }
  const auto keep_a =
      _mm512_set1_epi32(Packed && (controls & RJ_ARITH_ALU_ABS_A) ? 0x7fff7fff : -1);
  const auto keep_b =
      _mm512_set1_epi32(Packed && (controls & RJ_ARITH_ALU_ABS_B) ? 0x7fff7fff : -1);
  const uint32_t flip_a = Packed ? (controls & RJ_ARITH_ALU_NEG_A ? 0x80008000U : 0U)
                                 : ((controls & RJ_ARITH_DOT_NEG_LO_A ? 0x8000U : 0U) |
                                    (controls & RJ_ARITH_DOT_NEG_HI_A ? 0x80000000U : 0U));
  const uint32_t flip_b = Packed ? (controls & RJ_ARITH_ALU_NEG_B ? 0x80008000U : 0U)
                                 : ((controls & RJ_ARITH_DOT_NEG_LO_B ? 0x8000U : 0U) |
                                    (controls & RJ_ARITH_DOT_NEG_HI_B ? 0x80000000U : 0U));
  const auto keep_c = _mm512_set1_epi32(controls & RJ_ARITH_DOT_ABS_C ? INT32_MAX : -1);
  const auto flip_c = _mm512_set1_epi32(controls & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0);
  const int c_shift = controls & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  const int d_shift = controls & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto av = _mm512_loadu_si512(a + lane), bv = _mm512_loadu_si512(b + lane);
    auto cv = _mm512_loadu_si512(c + lane);
    if constexpr (Packed) {
      cv = _mm512_slli_epi32(_mm512_srl_epi32(cv, _mm_cvtsi32_si128(c_shift)), 16);
    } else {
      if (controls & kDotHalfSelection) {
        av = _mm512_shuffle_epi8(av, select_a);
        bv = _mm512_shuffle_epi8(bv, select_b);
      }
    }
    av = _mm512_xor_si512(_mm512_and_si512(av, keep_a), _mm512_set1_epi32(int(flip_a)));
    bv = _mm512_xor_si512(_mm512_and_si512(bv, keep_b), _mm512_set1_epi32(int(flip_b)));
    cv = _mm512_xor_si512(_mm512_and_si512(cv, keep_c), flip_c);
    const auto value =
        _mm512_dpbf16_ps(_mm512_castsi512_ps(cv), bit_cast<__m512bh>(av), bit_cast<__m512bh>(bv));
    auto result = _mm512_castps_si512(value);
    if constexpr (Packed) {
      // Native narrowing shares the pair-dot's fixed denormal policy.
      result = _mm512_sll_epi32(_mm512_cvtepu16_epi32(bit_cast<__m256i>(_mm512_cvtneps_pbh(value))),
                                _mm_cvtsi32_si128(d_shift));
      result =
          _mm512_or_si512(result, _mm512_and_si512(_mm512_loadu_si512(d + lane),
                                                   _mm512_set1_epi32(d_shift ? 0xffff : -65536)));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void dot2_bf16_x86_64_v4_bf16<32, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void dot2_bf16_x86_64_v4_bf16<32, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void dot2_bf16_x86_64_v4_bf16<64, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void dot2_bf16_x86_64_v4_bf16<64, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
