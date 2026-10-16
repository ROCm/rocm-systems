// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer arithmetic with carry/borrow masks.
// Test-side reference calculations; retain their documented provenance and supported domain.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Independent scalar reference inherited from tests/integer/carry_reference.h.
// Computed references are exact for these integer ISA operations.
enum class Carry { Add, Sub, Subrev, AddCi, SubCi, SubrevCi };

struct CarryResult {
  uint32_t value;
  bool carry;
};

// Return the integer result and carry/borrow bit, with optional saturation.
inline CarryResult carry_reference(Carry operation, uint32_t a, uint32_t b, bool input_carry,
                                   bool clamp) {
  int op = int(operation);
  uint64_t first = op % 3 == 2 ? b : a, second = op % 3 == 2 ? a : b;
  second += op >= 3 && input_carry;
  bool carry;
  uint64_t value;
  if (op % 3 == 0) {
    value = first + second;
    carry = value > UINT32_MAX;
  } else {
    value = first - second;
    carry = first < second;
  }
  if (clamp && carry) {
    value = op % 3 == 0 ? UINT32_MAX : 0;
  }
  return {uint32_t(value), carry};
}

} // namespace rocjitsu::arith::test
