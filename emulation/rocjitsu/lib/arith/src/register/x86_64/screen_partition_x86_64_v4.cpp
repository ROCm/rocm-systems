// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 register-table lookup of compact screen-partition coverage masks.

#include "register/screen_partition.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Preserve masked destinations and source aliases across the entire Wave64.
void screen_partition_x86_64_v4(uint64_t exec_mask, uint32_t *d, const uint32_t *a) {
  __m512i result[64 / 16];
  auto table0 = _mm512_loadu_si512(screen_partition_table);
  auto table1 = _mm512_loadu_si512(screen_partition_table + 16);
  for (int lane = 0; lane < 64; lane += 16) {
    auto selector = _mm512_and_si512(_mm512_loadu_si512(a + lane), _mm512_set1_epi32(255));
    auto index = _mm512_srli_epi32(selector, 3);
    auto shift = _mm512_slli_epi32(_mm512_and_si512(selector, _mm512_set1_epi32(7)), 2);
    auto packed = _mm512_permutex2var_epi32(table0, index, table1);
    result[lane / 16] = _mm512_and_si512(_mm512_srlv_epi32(packed, shift), _mm512_set1_epi32(15));
  }
  for (int lane = 0; lane < 64; lane += 16) {
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result[lane / 16]);
  }
}

} // namespace rocjitsu::arith
