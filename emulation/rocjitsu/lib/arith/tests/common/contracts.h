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

// Reject unsupported guest controls before accessing operands bound by the caller.
template <typename Call> void expect_unsupported_guest_controls(Call call) {
  for (uint64_t flag :
       {RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
        RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_FP_ROUND_UP,
        RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16,
        RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    SCOPED_TRACE(flag);
    EXPECT_EQ(call(flag), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}

} // namespace rocjitsu::arith::test
