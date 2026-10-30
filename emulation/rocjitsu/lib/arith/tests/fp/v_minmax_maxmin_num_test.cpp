// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for combined numeric-preferred min/max selections.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_minmax_num_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_minmax_num_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_minmax_num_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinmaxNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_minmax_num_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_maxmin_num_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_maxmin_num_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_maxmin_num_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxminNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_maxmin_num_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, false, false);
      });
}
