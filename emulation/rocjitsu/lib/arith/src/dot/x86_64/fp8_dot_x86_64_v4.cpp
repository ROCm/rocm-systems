// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Loose FP8/BF8 DOT4 with AVX-512 table decoding and FP32 FMA accumulation.
// Whole-wave execution handles half selectors and masked stores without staging operands.

#include "common/compiler.h"
#include "common/simd.h"
#include "common/x86_64/fp8_bf16_x86_64_v4.h"
#include "dot/operand_halves.h"
#include "integer/x86_64/packed_integer_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Accumulate a validated OCP FP8/BF8 wave; preserve inactive lanes and whole-VGPR aliases.
template <int Lanes, bool Bf8A, bool Bf8B>
void fp8_dot_x86_64_v4(uint64_t exec_mask, uint32_t controls, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, const uint32_t *c) {
  const Fp8Bf16Lut<Bf8A> lut_a;
  const Fp8Bf16Lut<Bf8B> lut_b;
  const auto keep = _mm512_set1_epi32(controls & RJ_ARITH_DOT_ABS_C ? INT32_MAX : -1);
  const auto flip = _mm512_set1_epi32(controls & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0);
  __m512i select_a{}, select_b{};
  if (controls & kDotHalfSelection) {
    select_a = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_A_HIGH,
                                                               controls & RJ_ARITH_DOT_HI_A_LOW));
    select_b = _mm512_broadcast_i64x4(packed_integer_selection(controls & RJ_ARITH_DOT_LO_B_HIGH,
                                                               controls & RJ_ARITH_DOT_HI_B_LOW));
  }
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto av = _mm512_loadu_si512(a + lane), bv = _mm512_loadu_si512(b + lane);
    auto value = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(c + lane), keep), flip));
    if (controls & kDotHalfSelection) {
      av = _mm512_shuffle_epi8(av, select_a);
      bv = _mm512_shuffle_epi8(bv, select_b);
    }
    RJ_ARITH_UNROLL(2)
    for (int half = 0; half < 2; ++half) {
      const auto ap = _mm512_cvtepi32_epi16(half ? _mm512_srli_epi32(av, 16) : av);
      const auto bp = _mm512_cvtepi32_epi16(half ? _mm512_srli_epi32(bv, 16) : bv);
      const auto aw = lut_a.decode_bf16(ap), bw = lut_b.decode_bf16(bp);
      // Place each BF16 encoding in the exponent/fraction positions of FP32.
      value = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(aw, 16)),
                              _mm512_castsi512_ps(_mm512_slli_epi32(bw, 16)), value);
      value = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_and_si512(aw, _mm512_set1_epi32(-65536))),
                              _mm512_castsi512_ps(_mm512_and_si512(bw, _mm512_set1_epi32(-65536))),
                              value);
    }
    _mm512_mask_storeu_ps(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

template void fp8_dot_x86_64_v4<32, false, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<32, false, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<32, true, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<32, true, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<64, false, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                  const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<64, false, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<64, true, false>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                 const uint32_t *, const uint32_t *);
template void fp8_dot_x86_64_v4<64, true, true>(uint64_t, uint32_t, uint32_t *, const uint32_t *,
                                                const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
