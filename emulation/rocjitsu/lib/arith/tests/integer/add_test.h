// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer addition/subtraction and saturation.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/integer_add_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check integer addition/subtraction with saturation, routing, and aliases for the supplied
// entry point.
template <int Lanes, int Operands, typename Call> void check_add(Call call, Add operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode, [[maybe_unused]] int lane) {
        return integer_add_reference(operation, a, b, c, mode & RJ_ARITH_ALU_CLAMP);
      },
      RJ_ARITH_ALU_CLAMP, operation == Add::SubrevU32);
}

} // namespace rocjitsu::arith::test
