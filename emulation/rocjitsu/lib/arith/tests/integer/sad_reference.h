// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sums of absolute differences, sliding windows, and packed accumulators.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/sad_reference.h.
// Computed references are exact for these integer ISA operations.
enum class Sad { U8, HiU8, U16, U32, MaskedU8, QuadPacked, MaskedQuadPacked, MaskedQuad };

// Return SAD accumulator words, including quad windows and packed-result forms.
inline std::array<uint32_t, 4> sad_reference(Sad operation, uint32_t a_low, uint32_t a_high,
                                             uint32_t b, const uint32_t *c, bool clamp) {
  int op = int(operation);
  const bool packed = op == 5 || op == 6;
  const bool masked = op == 4 || op == 6 || op == 7;
  const int width = op == 3 ? 32 : op == 2 ? 16 : 8;
  const uint64_t base = 1ULL << width;
  const uint64_t a = a_low + (uint64_t(a_high) << 32);
  std::array<uint32_t, 4> result{};
  for (int window = 0; window < (op >= 5 ? 4 : 1); ++window) {
    uint64_t x = a >> (window * 8), y = b, total = 0;
    for (int field = 0; field < 32 / width; ++field) {
      int64_t left = int64_t(x % base), right = int64_t(y % base);
      if (!masked || right != 0) {
        total += uint64_t(std::abs(left - right));
      }
      x /= base;
      y /= base;
    }
    if (op == 1) {
      total *= 65536;
    }
    uint64_t limit = packed ? 65535 : UINT32_MAX;
    total += packed ? (c[window / 2] >> (16 * (window % 2))) & 65535 : c[window];
    if (clamp) {
      total = std::min(total, limit);
    } else {
      total %= limit + 1;
    }
    if (packed) {
      result[window / 2] |= uint32_t(total) << (16 * (window % 2));
    } else {
      result[window] = uint32_t(total);
    }
  }
  return result;
}

} // namespace rocjitsu::arith::test
