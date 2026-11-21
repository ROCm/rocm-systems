// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fractional-part and mantissa decomposition.

#include "math/older_half_unary_test.h"
#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_math.h"

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
  rocjitsu::arith::test::check_half_fract_rounding<32>(rj_arith_v_fract_f16_w32_gfx12);
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

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, true>(rj_arith_v_frexp_mant_f16_w32_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, true>(rj_arith_v_frexp_mant_f16_w64_gfx942,
                                                             10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<32, false>(rj_arith_v_frexp_mant_f32_w32_gfx942,
                                                              10);
}

// Share the arithmetic contract while checking the CDNA NaN-policy difference.
TEST(MathFrexpMantF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_cdna_integral_unary<64, false>(rj_arith_v_frexp_mant_f32_w64_gfx942,
                                                              10);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_fract_f16_w32_gfx942, 9);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF16W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_fract_f16_w64_gfx942, 9);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W32Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_fract_f32_w32_gfx942, 9);
}

// Check inherited fractional arithmetic and unsupported strict semantics.
TEST(MathFractF32W64Gfx942, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_fract_f32_w64_gfx942, 9);
}

// Inherited mantissa mapping retains payload quieting, overlap and unsupported strict modes.
TEST(MathOlderHalfFrexp, Arithmetic) {
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_frexp_mant_f16_w32_gfx908, 4);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_frexp_mant_f16_w64_gfx908, 4);
}

// Fractional decomposition keeps the inherited special-value and aliasing contract.
TEST(MathOlderHalfFract, Arithmetic) {
  rocjitsu::arith::test::check_older_half_unary<32>(rj_arith_v_fract_f16_w32_gfx908, 5);
  rocjitsu::arith::test::check_half_fract_rounding<32>(rj_arith_v_fract_f16_w32_gfx908);
  rocjitsu::arith::test::check_older_half_unary<64>(rj_arith_v_fract_f16_w64_gfx908, 5);
  rocjitsu::arith::test::check_half_fract_rounding<64>(rj_arith_v_fract_f16_w64_gfx908);
}
