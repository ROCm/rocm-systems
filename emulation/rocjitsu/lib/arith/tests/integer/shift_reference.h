// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// 32/64-bit shifts and count masking.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Computed references are exact for these integer ISA operations.
enum class Shift { Left, LogicalRight, ArithmeticRight };

// Shift a 32/64-bit word using the operation's signedness and masked count.
inline uint64_t shift_reference(int bits, Shift operation, uint32_t count, uint64_t input) {
  const int distance = int(count % unsigned(bits));
  uint64_t result = 0;
  for (int bit = 0; bit < bits; ++bit) {
    int source = operation == Shift::Left ? bit - distance : bit + distance;
    bool value = source >= 0 && source < bits                            ? (input >> source) & 1
                 : source >= bits && operation == Shift::ArithmeticRight ? (input >> (bits - 1)) & 1
                                                                         : false;
    if (value) {
      result |= 1ULL << bit;
    }
  }
  return result;
}

} // namespace rocjitsu::arith::test
