// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for combined legacy min/max selections.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_minmax_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_minmax_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_minmax_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_minmax_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_maxmin_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_maxmin_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_maxmin_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_maxmin_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}
