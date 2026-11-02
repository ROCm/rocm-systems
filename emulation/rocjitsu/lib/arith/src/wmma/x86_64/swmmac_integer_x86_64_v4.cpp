// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 integer SWMMAC accumulation over decoded sparse operands, processing
// sixteen output columns at a time. Uses the shared compressed-K stage mapping
// and applies CLAMP at stage boundaries before storing the output tile.

#include "common/x86_64/alu_x86_64_v4.h"
#include "wmma/swmmac_integer_impl.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Accumulate decoded sparse INT8/INT4 operands and write eight 32-lane D
// sections. k is the dense reduction size (32 or 64); clamp saturates each
// stage, otherwise results wrap. Input snapshots must not overlap D.
void swmmac_integer_x86_64_v4(int k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input) {
  uint32_t result[8][32];
  for (int row = 0; row < 16; ++row) {
    for (int col = 0; col < 16; col += 16) {
      auto acc = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input.acc[row] + col));
      for (int stage = 0; stage < 2; ++stage) {
        auto sum = _mm512_setzero_si512();
        for (int local = 0; local < k / 4; ++local) {
          int ck = swmmac_integer_ck(stage, local);
          auto b = _mm512_loadu_si512(
              reinterpret_cast<const __m512i *>(input.b[input.selected[row][ck]] + col));
          sum = _mm512_add_epi32(sum, _mm512_mullo_epi32(_mm512_set1_epi32(input.a[row][ck]), b));
        }
        // The stage sum fits int32; only its addition to C can overflow.
        auto next = _mm512_add_epi32(acc, sum);
        if (clamp) {
          next = saturate_signed_sum(acc, sum, next);
        }
        acc = next;
      }
      _mm512_storeu_si512(reinterpret_cast<__m512i *>(result[row % 8] + col + 16 * (row / 8)), acc);
    }
  }
  for (int reg = 0; reg < 8; ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result[reg] + lane));
      _mm512_storeu_si512(d[reg] + lane, value);
    }
  }
}

} // namespace rocjitsu::arith
