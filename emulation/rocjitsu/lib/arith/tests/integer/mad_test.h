// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Narrow-factor integer multiply-add.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/integer_mad_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check narrow-factor multiply-add with routing and aliases for the supplied entry point.
template <int Lanes, int Operands, typename Call> void check_mad(Call call, Mad operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return integer_mad_reference(operation, a, b, c, mode); },
      RJ_ARITH_ALU_CLAMP | (operation == Mad::U16 || operation == Mad::I16
                                ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B
                                : 0),
      false);
}

} // namespace rocjitsu::arith::test
