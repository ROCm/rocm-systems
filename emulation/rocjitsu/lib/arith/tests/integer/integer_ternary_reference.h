// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Combined shift/boolean and three-input integer arithmetic.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/integer_ternary_reference.h.
// Computed references are exact for these integer ISA operations.
enum class Ternary { LshlAdd, AddLshl, LshlOr, AndOr, Or3, Xor3, Xad, Lerp };

// Compute the selected combined shift/boolean or three-input integer operation.
inline uint32_t integer_ternary_reference(Ternary operation, uint32_t a, uint32_t b, uint32_t c) {
  int op = int(operation);
  if (op == 0) {
    return uint32_t(uint64_t(a) * (1ULL << (b % 32)) + c);
  }
  if (op == 1) {
    return uint32_t((uint64_t(a) + b) * (1ULL << (c % 32)));
  }
  if (op == 7) {
    uint32_t result = 0;
    for (int byte = 0; byte < 4; ++byte) {
      unsigned numerator = (a % 256) + (b % 256) + (c % 2);
      result += (numerator / 2) * (1U << (8 * byte));
      a /= 256;
      b /= 256;
      c /= 256;
    }
    return result;
  }
  uint32_t shifted = op == 2 ? uint32_t(uint64_t(a) * (1ULL << (b % 32))) : a;
  uint32_t result = 0;
  for (int bit = 0; bit < 32; ++bit) {
    unsigned x = (shifted >> bit) & 1, y = (b >> bit) & 1, z = (c >> bit) & 1;
    bool value = op == 2   ? x + z != 0
                 : op == 3 ? x + y == 2 || z != 0
                 : op == 4 ? x + y + z != 0
                 : op == 5 ? (x + y + z) % 2 != 0
                           : x != y;
    if (value) {
      result |= 1U << bit;
    }
  }
  return op == 6 ? uint32_t(uint64_t(result) + c) : result;
}

} // namespace rocjitsu::arith::test
