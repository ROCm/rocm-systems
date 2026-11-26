// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 masked copies of register pairs, preserving whole-register aliases.
#include "register/pair_move.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Read both source words before either store, including when D overlaps A or B.
void pair_move_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *low,
                         const uint32_t *high) {
  for (int lane = 0; lane < lanes; lane += 8) {
    auto lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(low + lane));
    auto hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(high + lane));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), active, lo);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d[1] + lane), active, hi);
  }
}
} // namespace rocjitsu::arith
