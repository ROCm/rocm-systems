// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy three-input FP min/max/median.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(rj_arith_v_min3_f16_w32, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(rj_arith_v_min3_f16_w64, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(rj_arith_v_min3_f32_w32, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(rj_arith_v_min3_f32_w64, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_max3_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_max3_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_max3_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_max3_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_med3_f16_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_med3_f16_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_med3_f32_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_med3_f32_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}
