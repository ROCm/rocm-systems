// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 signed shifts and saturated byte packing with halfword preservation.

#include "integer/byte_shift.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>

namespace rocjitsu::arith {

// Pack two shifted signed words into bytes, preserving inactive lanes and the other half.
template <bool Signed>
void byte_shift_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, const uint32_t *c) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    auto y = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(b + lane));
    auto count = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(c + lane));
    auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
    count = _mm512_and_si512(count, _mm512_set1_epi32(31));
    x = _mm512_srav_epi32(x, count);
    y = _mm512_srav_epi32(y, count);
    auto low = _mm512_set1_epi32(Signed ? -128 : 0), high = _mm512_set1_epi32(Signed ? 127 : 255);
    x = _mm512_min_epi32(_mm512_max_epi32(x, low), high);
    y = _mm512_min_epi32(_mm512_max_epi32(y, low), high);
    auto bytes = _mm512_set1_epi32(255);
    auto packed = _mm512_or_si512(_mm512_and_si512(x, bytes),
                                  _mm512_slli_epi32(_mm512_and_si512(y, bytes), 8));
    auto output_mask = _mm512_set1_epi32(int(0xffffU << shift));
    auto result = _mm512_or_si512(_mm512_sllv_epi32(packed, _mm512_set1_epi32(shift)),
                                  _mm512_andnot_si512(output_mask, original));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

template void byte_shift_x86_64_v4<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                          const uint32_t *, const uint32_t *);
template void byte_shift_x86_64_v4<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                         const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
