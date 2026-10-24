// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for exponent extraction and power-of-two scaling entry points.

#include "fp/frexp_exp_test.h"
#include "fp/ldexp_test.h"
#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 32>(rj_arith_v_frexp_exp_i16_f16_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI16F16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<16, 64>(rj_arith_v_frexp_exp_i16_f16_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 32>(rj_arith_v_frexp_exp_i32_f32_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<32, 64>(rj_arith_v_frexp_exp_i32_f32_w64);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 32>(rj_arith_v_frexp_exp_i32_f64_w32);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64_gfx12);
}

// Check integer exponent results, operand selection, and inactive-lane preservation.
TEST(FpFrexpExpI32F64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_frexp_exp<64, 64>(rj_arith_v_frexp_exp_i32_f64_w64);
}

// Check the floating-point mantissa and operand contract.
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

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_frexp_mant_f64_w32, [](double x) {
    int exponent;
    double value = x;
    if (std::isfinite(x) && x != 0) {
      value = std::frexp(x, &exponent);
    }
    return value;
  });
}

// Check the floating-point mantissa and operand contract.
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

// Check the floating-point mantissa and operand contract.
TEST(FpFrexpMantF64W64Gfx11, Arithmetic) {
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
TEST(FpLdexpF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 32>(rj_arith_v_ldexp_f16_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<16, 64>(rj_arith_v_ldexp_f16_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 32>(rj_arith_v_ldexp_f32_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<32, 64>(rj_arith_v_ldexp_f32_w64);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 32>(rj_arith_v_ldexp_f64_w32);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64_gfx12);
}

// Check power-of-two scaling, guest controls, and source/destination aliases.
TEST(FpLdexpF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_ldexp<64, 64>(rj_arith_v_ldexp_f64_w64);
}

// Selected gfx1201/model boundaries distinguish tininess before rounding from
// classification of the rounded result. OMOD cannot rescue a tiny or overflowing intermediate.
TEST(FpLdexp, RoundingAndOutputStages) {
  using namespace rocjitsu::arith::test;
  struct Case {
    int width;
    uint64_t input;
    int exponent;
    uint32_t mode;
    uint64_t fp;
    uint64_t output;
  };
  const Case cases[] = {
      {16, 0x0400, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
      {16, 0x7bff, 1, RJ_ARITH_ALU_OMOD_HALF, 0, 0x7c00},
      {16, 0x6bff, -26, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {32, 0x00ffffff, -1, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {32, 0x00ffffff, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
      {32, 0x00ffffff, -1, 0, RJ_ARITH_FP_ROUND_UP | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {64, 0x001fffffffffffffULL, -1, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0},
      {64, 0x001fffffffffffffULL, -1, RJ_ARITH_ALU_OMOD_2, 0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &test : cases) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << test.width << '/'
                                          << test.mode << '/' << test.fp);
        Registers<5, 32> r;
        write_fp(r.v, 0, test.width, test.input);
        r.v[2][0] = uint32_t(test.exponent);
        auto call = test.width == 16   ? rj_arith_v_ldexp_f16_w32_gfx12
                    : test.width == 32 ? rj_arith_v_ldexp_f32_w32_gfx12
                                       : rj_arith_v_ldexp_f64_w32_gfx12;
        ASSERT_EQ(call(cpu | semantics | test.fp, 1, test.mode, r.v + 3, r.v, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 3, 0, test.width), test.output);
      }
    }
  }
}

// gfx1201 FREXP components describe the same input after guest flushing.
TEST(FpFrexpF64, PairedSubnormal) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t flush : {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS}) {
      Registers<5, 32> r;
      write_fp(r.v, 0, 64, 1);
      uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | flush;
      ASSERT_EQ(rj_arith_v_frexp_mant_f64_w32_gfx12(flags, 1, 0, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      ASSERT_EQ(rj_arith_v_frexp_exp_i32_f64_w32_gfx12(flags, 1, 0, r.v + 4, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), flush ? 0ULL : 0x3fe0000000000000ULL);
      EXPECT_EQ(r.v[4][0], flush ? 0U : uint32_t(-1073));
    }
  }
}

// RDNA3 output scaling is enabled only with flushed outputs and IEEE mode clear.
TEST(FpLdexpF32, Rdna3OmodGate) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x3f800000U;
        r.v[1][0] = 0;
        ASSERT_EQ(rj_arith_v_ldexp_f32_w32(cpu | semantics | fp, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2,
                                           r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], fp == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 0x40000000U : 0x3f800000U);
      }
    }
  }
}
