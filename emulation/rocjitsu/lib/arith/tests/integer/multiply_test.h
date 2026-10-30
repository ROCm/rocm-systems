// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Integer multiplication and high-half products.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/integer_mul_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check low/high integer products with saturation and aliases for the supplied entry point.
template <int Lanes, int Operands, bool SupportsDpp = true, typename Call>
void check_multiply(Call call, Multiply operation) {
  check_integer_word<Lanes, Operands, SupportsDpp>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode, [[maybe_unused]] int lane) {
        return integer_mul_reference(operation, a, b, mode & RJ_ARITH_ALU_CLAMP);
      },
      operation == Multiply::I24 || operation == Multiply::U24 ? RJ_ARITH_ALU_CLAMP : 0, false);
}

} // namespace rocjitsu::arith::test
