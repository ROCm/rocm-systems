// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Combined shift, boolean, and three-input integer operations.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "integer/integer_ternary_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check combined integer operations with routing and aliases for the supplied entry point.
template <int Lanes, int Operands, typename Call>
void check_combined(Call call, Ternary operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return integer_ternary_reference(operation, a, b, c); },
      RJ_ARITH_ALU_CLAMP, false);
}

} // namespace rocjitsu::arith::test
