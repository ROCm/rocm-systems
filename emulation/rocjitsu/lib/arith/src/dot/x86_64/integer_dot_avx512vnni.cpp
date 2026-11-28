// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Full-range INT4/INT8 vector dots through non-saturating AVX-512 VNNI.
// Bias corrections retain signedness; saturation applies only after adding C.

#include "common/compiler.h"
#include "common/simd.h"
#include "dot/operand_halves.h"
#include "integer/x86_64/packed_integer_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Accumulate a complete validated wave, preserving inactive lanes and input aliases.
// The packed dot fits int32; only C addition wraps or saturates.
template <int Lanes, int Bits, bool Unsigned>
void integer_dot_avx512vnni(uint64_t exec_mask, uint32_t controls, uint32_t *d, const uint32_t *a,
                            const uint32_t *b, const uint32_t *c) {
  static_assert(Bits == 4 || Bits == 8);
  const bool signed_a = !Unsigned && (controls & RJ_ARITH_DOT_SIGNED_A);
  const bool signed_b = !Unsigned && (controls & RJ_ARITH_DOT_SIGNED_B);
  __m512i select_a{}, select_b{};
  if (controls & kDotHalfSelection) {
    select_a = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_A_HIGH,
                                                               controls & RJ_ARITH_DOT_HI_A_LOW));
    select_b = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_B_HIGH,
                                                               controls & RJ_ARITH_DOT_HI_B_LOW));
  }
  const auto zero = _mm512_setzero_si512();
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto av = _mm512_loadu_si512(a + lane), bv = _mm512_loadu_si512(b + lane);
    if (controls & kDotHalfSelection) {
      av = _mm512_shuffle_epi8(av, select_a);
      bv = _mm512_shuffle_epi8(bv, select_b);
    }
    __m512i sum;
    if constexpr (Bits == 8) {
      const auto bias_a = _mm512_set1_epi32(signed_a ? int(0x80808080U) : 0);
      const auto bias_b = _mm512_set1_epi32(signed_b ? 0 : int(0x80808080U));
      const auto au = _mm512_xor_si512(av, bias_a);
      const auto bs = _mm512_xor_si512(bv, bias_b);
      sum = _mm512_dpbusd_epi32(zero, au, bs);
      if (signed_a) {
        sum = _mm512_sub_epi32(sum, _mm512_dpbusd_epi32(zero, bias_a, bs));
      }
      if (!signed_b) {
        // Undo unsigned B's offset. Signed A also contributes a four-product cross term.
        sum = _mm512_add_epi32(
            sum,
            _mm512_slli_epi32(_mm512_dpbusd_epi32(zero, au, _mm512_set1_epi32(0x01010101)), 7));
        if (signed_a) {
          sum = _mm512_sub_epi32(sum, _mm512_set1_epi32(4 * 128 * 128));
        }
      }
    } else {
      const auto bias_a = _mm512_set1_epi32(signed_a ? 0x08080808 : 0);
      const auto bias_b = _mm512_set1_epi32(signed_b ? 0x08080808 : 0);
      sum = zero;
      RJ_ARITH_UNROLL(2)
      for (int shift = 0; shift < 8; shift += 4) {
        auto au = _mm512_xor_si512(_mm512_and_si512(_mm512_srl_epi32(av, _mm_cvtsi32_si128(shift)),
                                                    _mm512_set1_epi32(0x0f0f0f0f)),
                                   bias_a);
        auto bs = _mm512_sub_epi8(
            _mm512_xor_si512(_mm512_and_si512(_mm512_srl_epi32(bv, _mm_cvtsi32_si128(shift)),
                                              _mm512_set1_epi32(0x0f0f0f0f)),
                             bias_b),
            bias_b);
        sum = _mm512_dpbusd_epi32(sum, au, bs);
        if (signed_a) {
          sum = _mm512_sub_epi32(sum, _mm512_dpbusd_epi32(zero, bias_a, bs));
        }
      }
    }
    const auto acc = _mm512_loadu_si512(c + lane);
    auto result = _mm512_add_epi32(acc, sum);
    if (controls & RJ_ARITH_DOT_CLAMP) {
      if constexpr (Unsigned) {
        result = _mm512_mask_mov_epi32(result, _mm512_cmpgt_epu32_mask(acc, result),
                                       _mm512_set1_epi32(-1));
      } else {
        const auto overflow =
            _mm512_and_si512(_mm512_xor_si512(acc, result), _mm512_xor_si512(sum, result));
        const auto limit =
            _mm512_xor_si512(_mm512_srai_epi32(acc, 31), _mm512_set1_epi32(INT32_MAX));
        result = _mm512_mask_mov_epi32(result, _mm512_movepi32_mask(overflow), limit);
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void integer_dot_avx512vnni<32, 4, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                   const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<32, 4, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<32, 8, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                   const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<32, 8, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<64, 4, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                   const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<64, 4, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<64, 8, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                   const uint32_t *, const uint32_t *);
template void integer_dot_avx512vnni<64, 8, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
