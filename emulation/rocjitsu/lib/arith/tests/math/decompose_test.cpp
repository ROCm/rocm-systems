// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fractional-part and mantissa decomposition.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_frexp_mant_f32_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_fract_f32_w32_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_frexp_mant_f32_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_frexp_mant_f32_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_fract_f32_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_frexp_mant_f32_w64_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_frexp_mant_f16_w32, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_fract_f16_w32_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_frexp_mant_f16_w32_gfx12, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_frexp_mant_f16_w64, 10);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFractF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_fract_f16_w64_gfx12, 9);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathFrexpMantF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_frexp_mant_f16_w64_gfx12, 10);
}
