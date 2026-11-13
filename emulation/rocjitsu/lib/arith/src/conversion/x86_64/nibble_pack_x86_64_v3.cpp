// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 byte clamps and nibble packing with masked destination stores.
#include "conversion/nibble_pack.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Clamp every byte, compact the low nibbles, and replace only the selected half.
void nibble_pack_x86_64_v3(bool signed_values, int lanes, uint64_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a) {
  const int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto half_mask = _mm256_set1_epi32(0xffffU << shift);
  for (int lane = 0; lane < lanes; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    raw = signed_values
              ? _mm256_min_epi8(_mm256_max_epi8(raw, _mm256_set1_epi8(-8)), _mm256_set1_epi8(7))
              : _mm256_min_epu8(raw, _mm256_set1_epi8(15));
    auto value = _mm256_and_si256(raw, _mm256_set1_epi32(0xf));
    value = _mm256_or_si256(value,
                            _mm256_and_si256(_mm256_srli_epi32(raw, 4), _mm256_set1_epi32(0xf0)));
    value = _mm256_or_si256(value,
                            _mm256_and_si256(_mm256_srli_epi32(raw, 8), _mm256_set1_epi32(0xf00)));
    value = _mm256_or_si256(
        value, _mm256_and_si256(_mm256_srli_epi32(raw, 12), _mm256_set1_epi32(0xf000)));
    value = _mm256_sll_epi32(value, _mm_cvtsi32_si128(shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    value = _mm256_or_si256(value, _mm256_andnot_si256(half_mask, old));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}
} // namespace rocjitsu::arith
