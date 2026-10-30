// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Single-halfword integer arithmetic and selection.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/integer16_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check single-halfword integer arithmetic and destination preservation for the supplied entry
// point.
template <int Lanes, int Operands, typename Call>
void check_half_binary(Call call, HalfBinary operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return integer16_reference(operation, a, b, d, mode); },
      RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D, false);
}

} // namespace rocjitsu::arith::test
