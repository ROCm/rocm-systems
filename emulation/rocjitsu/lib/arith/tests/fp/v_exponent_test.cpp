// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for exponent extraction and power-of-two scaling entry points.

#include "fp/frexp_exp_test.h"
#include "fp/ldexp_test.h"
#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64);
}

// Check exponent extraction and power-of-two scaling entry points.
TEST(FpFrexpMantF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_frexp_mant_f64_w32_gfx12,
                                                   [](double x) {
                                                     int exponent;
                                                     double value = x;
                                                     if (std::isfinite(x) && x != 0) {
                                                       value = std::frexp(x, &exponent);
                                                     }
                                                     return value;
                                                   });
}

// Check exponent extraction and power-of-two scaling entry points.
TEST(FpFrexpMantF64W32, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_frexp_mant_f64_w32, [](double x) {
    int exponent;
    double value = x;
    if (std::isfinite(x) && x != 0) {
      value = std::frexp(x, &exponent);
    }
    return value;
  });
}

// Check exponent extraction and power-of-two scaling entry points.
TEST(FpFrexpMantF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_frexp_mant_f64_w64_gfx12,
                                                   [](double x) {
                                                     int exponent;
                                                     double value = x;
                                                     if (std::isfinite(x) && x != 0) {
                                                       value = std::frexp(x, &exponent);
                                                     }
                                                     return value;
                                                   });
}

// Check exponent extraction and power-of-two scaling entry points.
TEST(FpFrexpMantF64W64, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_frexp_mant_f64_w64, [](double x) {
    int exponent;
    double value = x;
    if (std::isfinite(x) && x != 0) {
      value = std::frexp(x, &exponent);
    }
    return value;
  });
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 32>(rj_arith_v_ldexp_f16_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W32, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 32>(rj_arith_v_ldexp_f16_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64);
}
