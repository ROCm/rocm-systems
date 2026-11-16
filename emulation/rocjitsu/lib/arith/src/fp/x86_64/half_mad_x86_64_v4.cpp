// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX512 half MAD evaluates sixteen products and sums, sharing vector output policy with AVX2.

#include "fp/half_mad.h"
#include "fp/x86_64/half_mad_output_x86_64_v3.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a wave in sixteen-lane chunks without scalar exceptional-lane fallback.
void half_mad_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto load = [&](const uint32_t *p, uint32_t m) {
      auto bits = _mm512_loadu_si512(p + lane);
      bits = _mm512_srl_epi32(bits, _mm_cvtsi32_si128(m & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
      if (m & RJ_ARITH_ALU_ABS_A) {
        bits = _mm512_and_si512(bits, _mm512_set1_epi32(0x7fff));
      }
      if (m & RJ_ARITH_ALU_NEG_A) {
        bits = _mm512_xor_si512(bits, _mm512_set1_epi32(0x8000));
      }
      return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(bits));
    };
    auto x = load(a, mode), y = load(b, mode >> 1), z = load(c, mode >> 2);
    auto value = _mm512_fmadd_ps(x, y, z);
    auto low = finish_half_mad(_mm512_castps512_ps256(value), mode, flags);
    auto high = finish_half_mad(_mm512_extractf32x8_ps(value, 1), mode, flags);
    auto result = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
    result = _mm512_sll_epi32(result, _mm_cvtsi32_si128(shift));
    auto previous = _mm512_loadu_si512(d + lane);
    result = _mm512_or_si512(result,
                             _mm512_and_si512(previous, _mm512_set1_epi32(shift ? 65535 : -65536)));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace rocjitsu::arith
