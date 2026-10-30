// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed 16-bit multiply-add and saturation.
// Shared family test runner; instruction-specific cases supply operations and expectations.

#pragma once

#include "integer/packed_mad_reference.h"
#include "integer/word_test.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace rocjitsu::arith::test {

// Check packed integer multiply-add and saturation for the supplied entry point.
template <int Lanes, int Operands, typename Call> void check_packed_mad(Call call, bool operation) {
  check_integer_word<Lanes, Operands, false>(
      call,
      [=]([[maybe_unused]] uint32_t a, [[maybe_unused]] uint32_t b, [[maybe_unused]] uint32_t c,
          [[maybe_unused]] uint32_t d, [[maybe_unused]] uint32_t mode,
          [[maybe_unused]] int lane) { return packed_mad_reference(operation, a, b, c, mode); },
      RJ_ARITH_PK_CLAMP | RJ_ARITH_PK_LO_A_HIGH | RJ_ARITH_PK_LO_B_HIGH | RJ_ARITH_PK_LO_C_HIGH |
          RJ_ARITH_PK_HI_A_LOW | RJ_ARITH_PK_HI_B_LOW | RJ_ARITH_PK_HI_C_LOW,
      false);
}

} // namespace rocjitsu::arith::test
