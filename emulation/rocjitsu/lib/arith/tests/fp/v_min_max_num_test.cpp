// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for numeric-preferred FP min/max.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_min_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_min_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_min_num_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_min_num_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_min_num_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_min_num_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_max_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_max_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_max_num_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_max_num_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_max_num_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_max_num_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}
