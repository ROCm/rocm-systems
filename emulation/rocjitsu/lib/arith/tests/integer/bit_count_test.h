// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Population counts, bit reversal, and significant-bit searches.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "integer/bit_count_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check integer counts/searches with routing and aliases for the supplied entry point.
template <int Lanes, int Operands, class Call> void check_bit_count(Call call, BitCount operation) {
  check_integer_word<Lanes, Operands, true>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return bit_count_reference(operation, a, b, lane); },
      RJ_ARITH_ALU_CLAMP, false);
}

} // namespace rocjitsu::arith::test
