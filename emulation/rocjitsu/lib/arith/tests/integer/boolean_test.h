// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Two- and three-input bitwise boolean operations.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "integer/boolean_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check boolean operations and supported truth-table controls for the supplied entry point.
template <int Lanes, int Operands, typename Call> void check_boolean(Call call, Boolean operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return boolean_reference(operation, a, b, d, mode); },
      RJ_ARITH_ALU_CLAMP | (operation >= Boolean::And16 && operation <= Boolean::Not16
                                ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D |
                                      (operation == Boolean::Not16 ? 0 : RJ_ARITH_ALU_HIGH_B)
                                : 0),
      false);
}

} // namespace rocjitsu::arith::test
