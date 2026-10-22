// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared rejection checks for reserved semantics and unsupported instruction flags.

#pragma once

#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace rocjitsu::arith::test {

// Call the instruction with reserved semantics, an invalid global flag and
// each unsupported modifier bit. The caller binds operands and checks that
// any supplied storage is preserved; null operands can test early rejection.
template <typename Call> void expect_invalid_flags(Call call, uint64_t supported_modifiers) {
  for (int semantics = 4; semantics <= 7; ++semantics) {
    EXPECT_EQ(call(uint64_t(semantics) << 16, 0), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  EXPECT_EQ(call(1ULL << 63, 0), RJ_ARITH_ERROR_INVALID_FLAGS);
  for (int bit = 0; bit < 64; ++bit) {
    if (!(supported_modifiers & (1ULL << bit))) {
      SCOPED_TRACE(bit);
      EXPECT_EQ(call(0, 1ULL << bit), RJ_ARITH_ERROR_INVALID_FLAGS);
    }
  }
}

} // namespace rocjitsu::arith::test
