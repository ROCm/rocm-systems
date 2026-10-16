// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 selection operations.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}
