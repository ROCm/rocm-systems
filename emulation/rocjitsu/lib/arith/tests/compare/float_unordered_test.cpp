// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for unordered floating-point comparisons and negated predicates.

#include "compare/float_test.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_o_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_o_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_u_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_u_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nge_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nge_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_ngt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_ngt_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nle_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nle_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nlt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_nlt_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_o_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_o_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_u_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_u_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nge_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nge_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_ngt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_ngt_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nle_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nle_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nlt_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_nlt_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_o_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_o_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_u_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_u_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nge_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nge_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_ngt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_ngt_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nle_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nle_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nlt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_nlt_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_o_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_o_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_u_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_u_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nge_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nge_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_ngt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_ngt_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nle_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nle_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nlt_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_nlt_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_o_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_o_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_u_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_u_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nge_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nge_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_ngt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_ngt_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nle_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nle_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nlt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_nlt_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_o_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_o_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_u_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_u_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nge_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nge_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_ngt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_ngt_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nle_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nle_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nlt_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_nlt_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_o_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_o_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_u_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_u_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nge_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nge_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_ngt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_ngt_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nle_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nle_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nlt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_nlt_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_o_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_o_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_u_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_u_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nge_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nge_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_ngt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_ngt_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nle_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nle_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nlt_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_nlt_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_o_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_o_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_u_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_u_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nge_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nge_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_ngt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_ngt_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nle_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nle_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nlt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_nlt_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_o_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_o_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_u_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_u_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nge_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nge_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_ngt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_ngt_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nle_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nle_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nlt_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_nlt_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_o_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpOF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_o_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_u_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpUF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_u_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nge_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nge_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_ngt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNgtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_ngt_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nle_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNleF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nle_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nlt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNltF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_nlt_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_o_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxOF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_o_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_u_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxUF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_u_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nge_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgeF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nge_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a >= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_ngt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNgtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_ngt_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a > b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nle_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNleF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nle_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a <= b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nlt_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a < b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNltF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_nlt_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a < b); });
}
