// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Bitfield extraction/insertion and word alignment.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/bitfield_reference.h.
// Computed references are exact for these integer ISA operations.
enum class Bitfield { BfeU32, BfeI32, Bfi, Bfm, Bfrev, Alignbit, Alignbyte, Perm };

// Compute extraction, insertion, or alignment with the instruction's masked counts.
inline uint32_t bitfield_reference(Bitfield operation, uint32_t a, uint32_t b, uint32_t c) {
  int op = int(operation);
  uint32_t result = 0;
  int offset = b % 32, width = c % 32;
  for (int bit = 0; bit < 32; ++bit) {
    unsigned value = 0;
    if (op < 2 && width) {
      int source = offset + (bit < width ? bit : width - 1);
      if (bit < width || op == 1) {
        value = source < 32 ? (a >> source) & 1 : (op == 1 ? a >> 31 : 0);
      }
    }
    if (op == 2) {
      value = ((a >> bit) & 1) ? (b >> bit) & 1 : (c >> bit) & 1;
    }
    if (op == 3) {
      value = bit >= offset && bit - offset < int(a % 32);
    }
    if (op == 4) {
      value = (a >> (31 - bit)) & 1;
    }
    if (op == 5 || op == 6) {
      int source = bit + (op == 5 ? c % 32 : (c % 4) * 8);
      value = source < 32 ? (b >> source) & 1 : (a >> (source - 32)) & 1;
    }
    if (op == 7) {
      unsigned selector = (c >> (8 * (bit / 8))) & 255;
      int source = selector < 8    ? 8 * selector + bit % 8
                   : selector < 12 ? 16 * (selector - 8) + 15
                                   : 64;
      value = source < 32   ? (b >> source) & 1
              : source < 64 ? (a >> (source - 32)) & 1
                            : selector > 12;
    }
    result |= uint32_t(value) << bit;
  }
  return result;
}

} // namespace rocjitsu::arith::test
