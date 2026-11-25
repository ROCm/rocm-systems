// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 byte clamps and nibble packing with masked destination stores.
#include "conversion/nibble_pack.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Clamp every byte, compact the low nibbles, and replace only the selected half.
void nibble_pack_x86_64_v4(bool signed_values, int lanes, uint64_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto half_mask = _mm512_set1_epi32(0xffffU << shift);
  for (int lane = 0; lane < lanes; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    raw = signed_values
              ? _mm512_min_epi8(_mm512_max_epi8(raw, _mm512_set1_epi8(-8)), _mm512_set1_epi8(7))
              : _mm512_min_epu8(raw, _mm512_set1_epi8(15));
    auto value = _mm512_and_si512(raw, _mm512_set1_epi32(0xf));
    value = _mm512_or_si512(value,
                            _mm512_and_si512(_mm512_srli_epi32(raw, 4), _mm512_set1_epi32(0xf0)));
    value = _mm512_or_si512(value,
                            _mm512_and_si512(_mm512_srli_epi32(raw, 8), _mm512_set1_epi32(0xf00)));
    value = _mm512_or_si512(
        value, _mm512_and_si512(_mm512_srli_epi32(raw, 12), _mm512_set1_epi32(0xf000)));
    value = _mm512_sll_epi32(value, _mm_cvtsi32_si128(shift));
    auto old = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    value = _mm512_or_si512(value, _mm512_andnot_si512(half_mask, old));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}
} // namespace rocjitsu::arith
