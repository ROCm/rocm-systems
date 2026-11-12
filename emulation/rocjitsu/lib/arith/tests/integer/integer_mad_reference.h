// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Narrow-factor integer multiply-add.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class Mad { U16, I16, U24, I24 };

// Compute narrow-factor integer multiply-add with the selected modifiers.
inline uint32_t integer_mad_reference(Mad operation, uint32_t a, uint32_t b, uint32_t c,
                                      uint32_t mode) {
  int op = int(operation);
  const int bits = op < 2 ? 16 : 24;
  const int64_t range = 1LL << bits;
  int64_t x = (op < 2 && (mode & RJ_ARITH_ALU_HIGH_A) ? a >> 16 : a) % range;
  int64_t y = (op < 2 && (mode & RJ_ARITH_ALU_HIGH_B) ? b >> 16 : b) % range;
  int64_t z = c;
  if (op & 1) {
    if (x >= range / 2) {
      x -= range;
    }
    if (y >= range / 2) {
      y -= range;
    }
    if (z > INT32_MAX) {
      z -= 4294967296LL;
    }
  }
  int64_t result = x * y + z;
  if (mode & RJ_ARITH_ALU_CLAMP) {
    int64_t low = op & 1 ? INT32_MIN : 0, high = op & 1 ? INT32_MAX : int64_t(UINT32_MAX);
    if (result < low) {
      result = low;
    }
    if (result > high) {
      result = high;
    }
  }
  return uint32_t(result);
}

} // namespace rocjitsu::arith::test
