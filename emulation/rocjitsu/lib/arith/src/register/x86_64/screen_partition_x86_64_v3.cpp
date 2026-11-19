// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 gathered lookup of compact screen-partition coverage masks.

#include "register/screen_partition.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Preserve masked destinations and source aliases across the entire Wave64.
void screen_partition_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a) {
  __m256i result[64 / 8];
  for (int lane = 0; lane < 64; lane += 8) {
    auto selector = _mm256_and_si256(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), _mm256_set1_epi32(255));
    auto index = _mm256_srli_epi32(selector, 3);
    auto shift = _mm256_slli_epi32(_mm256_and_si256(selector, _mm256_set1_epi32(7)), 2);
    auto packed =
        _mm256_i32gather_epi32(reinterpret_cast<const int *>(screen_partition_table), index, 4);
    result[lane / 8] = _mm256_and_si256(_mm256_srlv_epi32(packed, shift), _mm256_set1_epi32(15));
  }
  for (int lane = 0; lane < 64; lane += 8) {
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result[lane / 8]);
  }
}

} // namespace rocjitsu::arith
