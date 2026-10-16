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
  for (int semantics = 4; semantics <= 6; ++semantics) {
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

// The arithmetic test supplies low modifier bits; DPP is sampled separately.
// Treat descriptor fields as a unit, then explicitly check malformed encodings.
template <bool SupportsDpp, class Call>
void expect_alu_invalid_flags(Call call, uint64_t low_modifiers) {
  uint64_t descriptor = !SupportsDpp ? 0
                                     : RJ_ARITH_DPP8 | RJ_ARITH_DPP16 | RJ_ARITH_DPP_FI |
                                           RJ_ARITH_DPP_BOUND_CTRL | RJ_ARITH_DPP_CTRL_MASK |
                                           RJ_ARITH_DPP_ROW_MASK | RJ_ARITH_DPP_BANK_MASK |
                                           RJ_ARITH_DPP8_SELECT_MASK;
  expect_invalid_flags(call, low_modifiers | descriptor);
  const uint64_t invalid_modes[] = {RJ_ARITH_DPP8 | RJ_ARITH_DPP16,
                                    RJ_ARITH_DPP16 | (0x131ULL << RJ_ARITH_DPP_CTRL_SHIFT),
                                    1ULL << RJ_ARITH_DPP8_SELECT_SHIFT};
  for (uint64_t invalid : invalid_modes) {
    EXPECT_EQ(call(0, invalid), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}

} // namespace rocjitsu::arith::test
