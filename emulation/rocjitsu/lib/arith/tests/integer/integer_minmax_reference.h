// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer minimum, maximum, and median.

#pragma once

#include "rj_arith/rj_arith.h"

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
  Med3U32
};

// Compute integer min/max/median using the operation's signedness.
inline uint32_t integer_minmax_reference(Select operation, uint32_t a, uint32_t b, uint32_t c) {
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
