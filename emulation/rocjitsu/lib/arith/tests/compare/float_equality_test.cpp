// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point equality and inequality, including unordered variants.

#include "compare/float_test.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_eq_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_eq_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_lg_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_lg_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_nlg_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmp_nlg_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_neq_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmp_neq_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_eq_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_eq_f16_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_lg_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_lg_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_nlg_f16_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(
      rj_arith_v_cmpx_nlg_f16_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF16W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_neq_f16_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 32>(rj_arith_v_cmpx_neq_f16_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_eq_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_eq_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_lg_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_lg_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_nlg_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmp_nlg_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_neq_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmp_neq_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_eq_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_eq_f32_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_lg_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_lg_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_nlg_f32_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(
      rj_arith_v_cmpx_nlg_f32_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF32W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_neq_f32_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 32>(rj_arith_v_cmpx_neq_f32_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_eq_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_eq_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_lg_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_lg_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_nlg_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmp_nlg_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_neq_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmp_neq_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_eq_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_eq_f64_w32_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_lg_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_lg_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_nlg_f64_w32, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(
      rj_arith_v_cmpx_nlg_f64_w32_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF64W32, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_neq_f64_w32, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 32>(rj_arith_v_cmpx_neq_f64_w32_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_eq_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_eq_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_lg_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_lg_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_nlg_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmp_nlg_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_neq_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmp_neq_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_eq_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_eq_f16_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_lg_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_lg_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_nlg_f16_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(
      rj_arith_v_cmpx_nlg_f16_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF16W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_neq_f16_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<16, 64>(rj_arith_v_cmpx_neq_f16_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_eq_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_eq_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_lg_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_lg_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_nlg_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmp_nlg_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_neq_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmp_neq_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_eq_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_eq_f32_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_lg_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_lg_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_nlg_f32_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(
      rj_arith_v_cmpx_nlg_f32_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF32W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_neq_f32_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<32, 64>(rj_arith_v_cmpx_neq_f32_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_eq_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpEqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_eq_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_lg_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpLgF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_lg_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_nlg_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNlgF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmp_nlg_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_neq_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpNeqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmp_neq_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_eq_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxEqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_eq_f64_w64_gfx12, /*gfx12=*/true, [](double a, double b) { return a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_lg_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxLgF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_lg_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return !std::isnan(a) && !std::isnan(b) && a != b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_nlg_f64_w64, /*gfx12=*/false,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNlgF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(
      rj_arith_v_cmpx_nlg_f64_w64_gfx12, /*gfx12=*/true,
      [](double a, double b) { return std::isnan(a) || std::isnan(b) || a == b; });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF64W64, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_neq_f64_w64, /*gfx12=*/false,
                                                     [](double a, double b) { return !(a == b); });
}

// Check FP predicate masks, modifiers, aliases, and FP-status policy.
TEST(CompareCmpxNeqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_float_compare<64, 64>(rj_arith_v_cmpx_neq_f64_w64_gfx12,
                                                     /*gfx12=*/true,
                                                     [](double a, double b) { return !(a == b); });
}
