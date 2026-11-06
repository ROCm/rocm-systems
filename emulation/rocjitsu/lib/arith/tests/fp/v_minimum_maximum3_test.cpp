// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for three-input NaN-propagating minimum/maximum.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimum3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_minimum3_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, true),
                                                z, false, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimum3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_minimum3_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, true),
                                                z, false, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimum3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_minimum3_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, true),
                                                z, false, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimum3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_minimum3_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, true),
                                                z, false, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximum3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_maximum3_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, true),
                                                z, true, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximum3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_maximum3_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, true),
                                                z, true, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximum3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_maximum3_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, true),
                                                z, true, true);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximum3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_maximum3_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, true),
                                                z, true, true);
      });
}

// Propagating ternary selection retains A's NaN payload through both selections.
TEST(FpMinimumMaximum3, CompatibilityFirstNaN) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_minimum3_f32_w32, rj_arith_v_maximum3_f32_w32}) {
      Registers<4, 32> r;
      r.v[0][0] = 0xff800004U;
      r.v[1][0] = 0x7fc00001U;
      r.v[2][0] = 0x7f800002U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2,
                     nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0xffc00004U);
    }
  }
}
