// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer minimum, maximum, median, and saturating add-then-select.

#pragma once

#include "rj_arith/detail/rj_arith_common.h"

#include <algorithm>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class Select {
  MinI32,
  MaxI32,
  Min3I32,
  Max3I32,
  MinmaxI32,
  MaxminI32,
  Med3I32,
  MinU32,
  MaxU32,
  Min3U32,
  Max3U32,
  MinmaxU32,
  MaxminU32,
  Med3U32,
  AddMinI32,
  AddMaxI32,
  AddMinU32,
  AddMaxU32
};

// Compute integer selections, including saturating sums, using the operation's signedness.
inline uint32_t integer_minmax_reference(Select operation, uint32_t a, uint32_t b, uint32_t c,
                                         uint32_t mode) {
  if (operation >= Select::AddMinI32) {
    bool signed_order = operation == Select::AddMinI32 || operation == Select::AddMaxI32;
    bool maximum = operation == Select::AddMaxI32 || operation == Select::AddMaxU32;
    int64_t values[] = {a, b, c};
    if (signed_order) {
      for (auto &value : values) {
        if (value > INT32_MAX) {
          value -= 4294967296LL;
        }
      }
    }
    int64_t sum = std::clamp<int64_t>(values[0] + values[1], signed_order ? INT32_MIN : 0,
                                      signed_order ? INT32_MAX : int64_t(UINT32_MAX));
    int64_t value = maximum ? std::max(sum, values[2]) : std::min(sum, values[2]);
    return uint32_t(signed_order && (mode & RJ_ARITH_ALU_CLAMP) ? std::max(value, int64_t(0))
                                                                : value);
  }
  int instruction = int(operation);
  int64_t values[] = {a, b, c};
  if (instruction < 7) {
    for (auto &value : values) {
      if (value > INT32_MAX) {
        value -= 4294967296LL;
      }
    }
  }
  int op = instruction % 7;
  std::sort(values, values + 2);
  if (op == 0 || op == 1) {
    return uint32_t(values[op]);
  }
  if (op == 4) {
    return uint32_t(values[0] > values[2] ? values[0] : values[2]);
  }
  if (op == 5) {
    return uint32_t(values[1] < values[2] ? values[1] : values[2]);
  }
  std::sort(values, values + 3);
  return uint32_t(values[op == 2 ? 0 : op == 3 ? 2 : 1]);
}

} // namespace rocjitsu::arith::test
