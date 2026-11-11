// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed 16-bit integer arithmetic and selection.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "common/word_test.h"
#include "integer/packed_integer_reference.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check packed integer arithmetic and independent halfword selectors for the supplied entry
// point.
template <int Lanes, int Operands, typename Call>
void check_packed_binary(Call call, HalfBinary operation) {
  check_integer_word<Lanes, Operands, false>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return packed_integer_reference(operation, a, b, mode); },
      RJ_ARITH_PK_CLAMP | RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_HI_A_LOW |
          RJ_ARITH_PK_HI_B_LOW,
      false);
}

} // namespace rocjitsu::arith::test
