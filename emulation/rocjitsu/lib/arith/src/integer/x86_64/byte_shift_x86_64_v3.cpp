// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 signed shifts and saturated byte packing with halfword preservation.

#include "integer/byte_shift.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>

namespace rocjitsu::arith {

// Pack two shifted signed words into bytes, preserving inactive lanes and the other half.
template <bool Signed>
void byte_shift_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, const uint32_t *c) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto count = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    count = _mm256_and_si256(count, _mm256_set1_epi32(31));
    x = _mm256_srav_epi32(x, count);
    y = _mm256_srav_epi32(y, count);
    auto low = _mm256_set1_epi32(Signed ? -128 : 0), high = _mm256_set1_epi32(Signed ? 127 : 255);
    x = _mm256_min_epi32(_mm256_max_epi32(x, low), high);
    y = _mm256_min_epi32(_mm256_max_epi32(y, low), high);
    auto bytes = _mm256_set1_epi32(255);
    auto packed = _mm256_or_si256(_mm256_and_si256(x, bytes),
                                  _mm256_slli_epi32(_mm256_and_si256(y, bytes), 8));
    auto output_mask = _mm256_set1_epi32(int(0xffffU << shift));
    auto result = _mm256_or_si256(_mm256_sllv_epi32(packed, _mm256_set1_epi32(shift)),
                                  _mm256_andnot_si256(output_mask, original));
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask, result);
  }
}

template void byte_shift_x86_64_v3<false>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                          const uint32_t *, const uint32_t *);
template void byte_shift_x86_64_v3<true>(uint32_t, uint32_t, uint32_t *, const uint32_t *,
                                         const uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
