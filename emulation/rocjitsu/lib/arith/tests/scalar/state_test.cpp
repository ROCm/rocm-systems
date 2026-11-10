// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for guest rounding and denormal MODE-field updates.

#include "rj_arith/rj_arith.h"

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/integer.h"
#include "common/semantics.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

// Check updates to one guest MODE field with preservation of the other bits for the supplied
// entry point.
template <typename Call> void check_mode(Call call, int shift) {
  for (auto cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      std::minstd_rand random;
      for (int sample = 0; sample < 128; ++sample) {
        uint32_t mode = rocjitsu::arith::test::random_word(random), before = mode;
        uint16_t immediate = uint16_t(rocjitsu::arith::test::random_word(random));
        ASSERT_EQ(call(cpu | semantics, 0, immediate, &mode), RJ_ARITH_SUCCESS);
        EXPECT_EQ(mode, (before & ~(15U << shift)) | ((immediate & 15U) << shift));
        ASSERT_EQ(call(cpu | semantics, 0, immediate, nullptr), RJ_ARITH_SUCCESS);
      }
    }
  }
  uint32_t mode = 123;
  rocjitsu::arith::test::expect_invalid_flags(
      [&](auto f, auto m) {
        int error = call(f, m, 0, &mode);
        EXPECT_EQ(mode, 123U);
        return error;
      },
      0);
}

} // namespace

// Check updates to one guest MODE field with preservation of the other bits.
TEST(s_round_mode, Arithmetic) { check_mode(rj_arith_s_round_mode, 0); }

// Check updates to one guest MODE field with preservation of the other bits.
TEST(s_denorm_mode, Arithmetic) { check_mode(rj_arith_s_denorm_mode, 4); }
