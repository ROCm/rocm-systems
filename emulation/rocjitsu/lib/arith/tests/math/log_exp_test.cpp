// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point logarithm and exponential entry points.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_exp_f32_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_log_f32_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_exp_f32_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_log_f32_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_exp_f32_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_log_f32_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_exp_f32_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_log_f32_w64_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_exp_f16_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_log_f16_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_exp_f16_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_log_f16_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_exp_f16_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_log_f16_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathExpF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_exp_f16_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(MathLogF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_log_f16_w64_gfx12, 8);
}

// gfx1201 TRANS OMOD overflow ignores ROUND=RTZ (MODE=0xff).
TEST(MathTrans, OutputOverflowIgnoresGuestRounding) {
  using namespace rocjitsu::arith::test;
  struct Case {
    decltype(&rj_arith_v_exp_f32_w32_gfx12) call;
    uint32_t input, expected;
  };
  const Case cases[] = {
      {rj_arith_v_exp_f32_w32_gfx12, 0x42fe0000U, 0x7f800000U}, // exp2(127)
      {rj_arith_v_exp_f16_w32_gfx12, 0x4b80U, 0x7c00U},         // exp2(15)
      {rj_arith_v_rcp_f32_w32_gfx12, 0x00800000U, 0x7f800000U},
      {rj_arith_v_rcp_f16_w32_gfx12, 0x0400U, 0x7c00U},
  };
  for (auto cpu : supported_cpu_levels()) {
    for (auto semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (const auto &test : cases) {
        Registers<2, 32> r;
        r.v[0][0] = test.input;
        r.v[1][0] = 0; // FP16 writes preserve the other half.
        uint32_t status = 0;
        ASSERT_EQ(test.call(cpu | semantics | RJ_ARITH_FP_ROUND_ZERO, 1, RJ_ARITH_ALU_OMOD_4,
                            r.v + 1, r.v, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[1][0], test.expected);
      }
    }
  }
}
