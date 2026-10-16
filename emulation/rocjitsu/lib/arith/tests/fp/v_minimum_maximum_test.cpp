// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for naN-propagating minimum/maximum.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_minimum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_minimum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_minimum_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_minimum_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_minimum_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_minimum_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_maximum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_maximum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_maximum_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_maximum_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_maximum_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_maximum_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}
