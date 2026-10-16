// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Two- and three-input bitwise boolean operations.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/boolean_reference.h.
// Computed references are exact for these integer ISA operations.
enum class Boolean { And32, Or32, Xor32, Not32, And16, Or16, Xor16, Not16, Xnor };

// Compute the selected boolean operation, including destination input or truth-table controls.
inline uint32_t boolean_reference(Boolean operation, uint32_t a, uint32_t b, uint32_t d,
                                  uint32_t mode) {
  int op = int(operation);
  uint32_t result = (op < 4 || op == 8) ? 0 : d;
  unsigned sa = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  unsigned sb = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  unsigned sd = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int bit = 0; bit < ((op < 4 || op == 8) ? 32 : 16); ++bit) {
    bool x = (a >> (sa + bit)) & 1, y = (b >> (sb + bit)) & 1;
    bool value = false;
    switch (op % 4) {
    case 0:
      value = x && y;
      break;
    case 1:
      value = x || y;
      break;
    case 2:
      value = x != y;
      break;
    case 3:
      value = !x;
      break;
    }
    if (op == 8) {
      value = x == y;
    }
    uint32_t mask = uint32_t(1) << (sd + bit);
    result = (result & ~mask) | (value ? mask : 0);
  }
  return result;
}

} // namespace rocjitsu::arith::test
