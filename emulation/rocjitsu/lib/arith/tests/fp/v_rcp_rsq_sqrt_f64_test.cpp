// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 reciprocal, reciprocal square root, and square root.

#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rcp_f64_w32_gfx12,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W32, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rcp_f64_w32,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rcp_f64_w64_gfx12,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W64, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rcp_f64_w64,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rsq_f64_w32_gfx12,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W32, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rsq_f64_w32,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rsq_f64_w64_gfx12,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W64, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rsq_f64_w64,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_sqrt_f64_w32_gfx12, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W32, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_sqrt_f64_w32, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_sqrt_f64_w64_gfx12, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W64, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_sqrt_f64_w64, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}
