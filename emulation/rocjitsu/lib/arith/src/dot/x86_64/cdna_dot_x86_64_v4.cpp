// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 CDNA FP16/BF16 DOT2 using widened FP32 arithmetic.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen selected FP16/BF16 halfwords and apply the supplied sign flip.
template <bool Bf16> __m512 widen(__m512i words, int shift, uint32_t negate) {
  auto halves =
      _mm512_and_si512(_mm512_srl_epi32(words, _mm_cvtsi32_si128(shift)), _mm512_set1_epi32(65535));
  halves = _mm512_xor_si512(halves, _mm512_set1_epi32(negate ? 0x8000 : 0));
  if constexpr (Bf16) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(halves, 16));
  } else {
    return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(halves));
  }
}

// Accumulate loose FP16/BF16 DOT2 products for the selected lanes. Requires validated
// controls; preserve inactive destination lanes.
template <bool Bf16>
void dot(uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  const int a0 = instruction_flags & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
  const int b0 = instruction_flags & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
  const int a1 = instruction_flags & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
  const int b1 = instruction_flags & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
  const auto neg_c = _mm512_set1_epi32(instruction_flags & RJ_ARITH_DOT_NEG_C ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    auto acc = _mm512_castsi512_ps(_mm512_xor_si512(
        _mm512_and_si512(
            _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane)),
            _mm512_set1_epi32(instruction_flags & RJ_ARITH_DOT_ABS_C ? 0x7fffffff : -1)),
        neg_c));
    auto low = _mm512_mul_ps(widen<Bf16>(va, a0, instruction_flags & RJ_ARITH_DOT_NEG_LO_A),
                             widen<Bf16>(vb, b0, instruction_flags & RJ_ARITH_DOT_NEG_LO_B));
    auto high = _mm512_mul_ps(widen<Bf16>(va, a1, instruction_flags & RJ_ARITH_DOT_NEG_HI_A),
                              widen<Bf16>(vb, b1, instruction_flags & RJ_ARITH_DOT_NEG_HI_B));
    acc = _mm512_add_ps(_mm512_add_ps(low, high), acc);
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(acc));
  }
}

} // namespace

// Compute CDNA pair sums at sixteen-lane width without modifying host FP controls.
void cdna_dot2_x86_64_v4(bool bf16, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf16) {
    dot<true>(exec_mask, instruction_flags, d, a, b, c);
  } else {
    dot<false>(exec_mask, instruction_flags, d, a, b, c);
  }
}

} // namespace rocjitsu::arith
