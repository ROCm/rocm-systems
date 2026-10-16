// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Population counts, bit reversal, and significant-bit searches.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/bit_count_reference.h.
// Computed references are exact for these integer ISA operations.
enum class BitCount { Clz, Ctz, Cls, Bcnt, MbcntLo, MbcntHi };

// Compute the selected bit-count/search operation, including its lane-dependent input when
// required.
inline uint32_t bit_count_reference(BitCount operation, uint32_t a, uint32_t b, int lane) {
  int op = int(operation);
  if (op < 3) {
    uint32_t count = 0;
    bool sign = op == 2 && (a >> 31);
    for (int bit = 0; bit < 32; ++bit) {
      int position = op == 1 ? bit : 31 - bit;
      if (bool((a >> position) & 1) != sign) {
        return count;
      }
      ++count;
    }
    return UINT32_MAX;
  }
  uint32_t count = 0;
  for (int bit = 0; bit < 32; ++bit) {
    bool selected = op == 3 || ((op == 4 || op == 6) ? bit < lane : bit + 32 < lane);
    if (selected && ((a >> bit) & 1)) {
      ++count;
    }
  }
  return b + count;
}

} // namespace rocjitsu::arith::test
