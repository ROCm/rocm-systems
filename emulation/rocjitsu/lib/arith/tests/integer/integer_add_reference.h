// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 32-bit integer add/subtract with saturation.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class Add { AddU32, SubU32, SubrevU32, AddI32, SubI32, Add3U32 };

// Compute wrapping or saturating integer add/subtract from raw source words.
inline uint32_t integer_add_reference(Add operation, uint32_t a, uint32_t b, uint32_t c,
                                      bool clamp) {
  int op = int(operation);
  int64_t x = a, y = b;
  bool signed_op = op == 3 || op == 4;
  if (signed_op) {
    if (x > INT32_MAX) {
      x -= 4294967296LL;
    }
    if (y > INT32_MAX) {
      y -= 4294967296LL;
    }
  }
  int64_t result = op == 1 || op == 4 ? x - y : op == 2 ? y - x : x + y;
  if (op == 5) {
    result += c;
  }
  if (clamp && op != 5) {
    int64_t low = signed_op ? INT32_MIN : 0, high = signed_op ? INT32_MAX : int64_t(UINT32_MAX);
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
