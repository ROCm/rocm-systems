// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy FP min/max.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_min_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_min_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_min_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_min_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_min_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_min_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(rj_arith_v_max_f16_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(rj_arith_v_max_f16_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(rj_arith_v_max_f32_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(rj_arith_v_max_f32_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(rj_arith_v_max_f64_w32, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(rj_arith_v_max_f64_w64, [](double x, double y) {
    return rocjitsu::arith::test::select_fp(x, y, true, false);
  });
}
