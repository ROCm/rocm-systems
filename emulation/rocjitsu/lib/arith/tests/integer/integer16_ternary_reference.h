// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 16-bit integer multiply-add and three-input selection.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class HalfTernary { MadU16, MadI16, Min3U16, Min3I16, Max3U16, Max3I16, Med3U16, Med3I16 };

// Compute the selected 16-bit ternary operation and retain the other destination half.
inline uint32_t integer16_ternary_reference(HalfTernary operation, uint32_t a, uint32_t b,
                                            uint32_t c, uint32_t mode, uint32_t d = 0xfacecafe) {
  int op = int(operation);
  const uint32_t words[] = {a, b, c};
  int64_t inputs[3];
  for (int i = 0; i < 3; ++i) {
    inputs[i] = (words[i] >> (mode & (RJ_ARITH_ALU_HIGH_A << i) ? 16 : 0)) & 65535;
    if ((op & 1) && inputs[i] > 32767) {
      inputs[i] -= 65536;
    }
  }
  int64_t value = inputs[0];
  if (op < 2) {
    value = inputs[0] * inputs[1] + inputs[2];
    if (mode & RJ_ARITH_ALU_CLAMP) {
      int64_t low = op ? -32768 : 0, high = op ? 32767 : 65535;
      if (value < low) {
        value = low;
      }
      if (value > high) {
        value = high;
      }
    }
  } else if (op >= 6) {
    // Sort the three scalar values, independently of the SIMD comparison network.
    for (int i = 0; i < 3; ++i) {
      for (int j = i + 1; j < 3; ++j) {
        if (inputs[j] < inputs[i]) {
          int64_t temporary = inputs[i];
          inputs[i] = inputs[j];
          inputs[j] = temporary;
        }
      }
    }
    value = inputs[1];
  } else {
    for (int i = 1; i < 3; ++i) {
      if (op < 4 ? inputs[i] < value : inputs[i] > value) {
        value = inputs[i];
      }
    }
  }
  uint32_t bits = uint32_t(value) & 65535;
  return mode & RJ_ARITH_ALU_HIGH_D ? (bits << 16) | (d & 65535) : (d & 0xffff0000u) | bits;
}

} // namespace rocjitsu::arith::test
