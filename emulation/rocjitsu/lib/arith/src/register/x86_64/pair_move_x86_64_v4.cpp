// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 masked copies of register pairs, preserving whole-register aliases.
#include "register/pair_move.h"
#include <immintrin.h>
namespace rocjitsu::arith {
// Read both source words before either store, including when D overlaps A or B.
void pair_move_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *low,
                         const uint32_t *high) {
  for (int lane = 0; lane < lanes; lane += 16) {
    auto lo = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(low + lane));
    auto hi = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(high + lane));
    auto active = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(d[0] + lane, active, lo);
    _mm512_mask_storeu_epi32(d[1] + lane, active, hi);
  }
}
} // namespace rocjitsu::arith
