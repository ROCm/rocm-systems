// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for ordered floating-point comparisons.

#include "compare/float_test.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_lt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_lt_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_le_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_le_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_gt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_gt_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_ge_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_ge_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_lt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_lt_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_le_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_le_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_gt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_gt_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_ge_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_ge_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_lt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_lt_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_le_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_le_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_gt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_gt_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_ge_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_ge_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_lt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_lt_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_le_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_le_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_gt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_gt_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_ge_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_ge_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_lt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_lt_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_le_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_le_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_gt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_gt_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_ge_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_ge_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_lt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_lt_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_le_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_le_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_gt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_gt_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_ge_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_ge_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_lt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_lt_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_le_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_le_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_gt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_gt_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_ge_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_ge_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_lt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_lt_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_le_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_le_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_gt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_gt_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_ge_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_ge_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_lt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_lt_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_le_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_le_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_gt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_gt_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_ge_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_ge_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_lt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_lt_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_le_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_le_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_gt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_gt_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_ge_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_ge_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_lt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_lt_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_le_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_le_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_gt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_gt_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_ge_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpGeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_ge_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_lt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_lt_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a < b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_le_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_le_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a <= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_gt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_gt_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a > b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_ge_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a >= b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxGeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_ge_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a >= b; });
}
