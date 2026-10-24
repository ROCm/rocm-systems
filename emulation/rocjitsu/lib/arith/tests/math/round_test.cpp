// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for rounding to integral values and rounding-related FP status.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_trunc_f32_w32, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_ceil_f32_w32, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rndne_f32_w32, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_floor_f32_w32, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_trunc_f32_w32_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_ceil_f32_w32_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rndne_f32_w32_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_floor_f32_w32_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_trunc_f32_w64, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_ceil_f32_w64, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rndne_f32_w64, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_floor_f32_w64, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_trunc_f32_w64_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_ceil_f32_w64_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rndne_f32_w64_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_floor_f32_w64_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_trunc_f16_w32, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_ceil_f16_w32, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_rndne_f16_w32, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_floor_f16_w32, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_trunc_f16_w32_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_ceil_f16_w32_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_rndne_f16_w32_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_floor_f16_w32_gfx12, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_trunc_f16_w64, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_ceil_f16_w64, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_rndne_f16_w64, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_floor_f16_w64, 3);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathTruncF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_trunc_f16_w64_gfx12, 0);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathCeilF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_ceil_f16_w64_gfx12, 1);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathRndneF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_rndne_f16_w64_gfx12, 2);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFloorF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_floor_f16_w64_gfx12, 3);
}
