// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 masked copies of register pairs, preserving whole-register aliases.
#include "register/pair_move.h"
#include <immintrin.h>
namespace rocjitsu::arith {
namespace {
// Snapshot both input registers before storing to either destination. The fixed wave
// size keeps the source chunks in registers across all masked stores.
template <int Lanes>
void move_pair(uint64_t exec_mask, uint32_t *lo_d, uint32_t *hi_d, const uint32_t *low,
               const uint32_t *high) {
  __m512i lo[Lanes / 16], hi[Lanes / 16];
  for (int lane = 0; lane < Lanes; lane += 16) {
    lo[lane / 16] = _mm512_loadu_si512(low + lane);
    hi[lane / 16] = _mm512_loadu_si512(high + lane);
  }
  for (int lane = 0; lane < Lanes; lane += 16) {
    auto active = __mmask16(exec_mask >> lane);
    _mm512_mask_storeu_epi32(lo_d + lane, active, lo[lane / 16]);
    _mm512_mask_storeu_epi32(hi_d + lane, active, hi[lane / 16]);
  }
}
} // namespace

// Load the register pointers once, before any possibly aliased stores.
void pair_move_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *low,
                         const uint32_t *high) {
  if (lanes == 32) {
    move_pair<32>(exec_mask, d[0], d[1], low, high);
  } else {
    move_pair<64>(exec_mask, d[0], d[1], low, high);
  }
}
} // namespace rocjitsu::arith
