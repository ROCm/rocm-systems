// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for reciprocal, reciprocal-square-root, and square-root entry points.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_sqrt_f32_w32, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rcp_f32_w32, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rsq_f32_w32, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_sqrt_f32_w32_gfx12, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rcp_f32_w32_gfx12, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rsq_f32_w32_gfx12, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_sqrt_f32_w64, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rcp_f32_w64, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rsq_f32_w64, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_sqrt_f32_w64_gfx12, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rcp_f32_w64_gfx12, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rsq_f32_w64_gfx12, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_sqrt_f16_w32, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_rcp_f16_w32, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_rsq_f16_w32, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_sqrt_f16_w32_gfx12, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_rcp_f16_w32_gfx12, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_rsq_f16_w32_gfx12, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_sqrt_f16_w64, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_rcp_f16_w64, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_rsq_f16_w64, 6);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathSqrtF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_sqrt_f16_w64_gfx12, 4);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRcpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_rcp_f16_w64_gfx12, 5);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRsqF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_rsq_f16_w64_gfx12, 6);
}
