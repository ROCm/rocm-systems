// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for reciprocal, reciprocal-square-root, and square-root entry points.

#include "math/unary_test.h"
#include "rj_arith/detail/rj_arith_math.h"

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

// gfx1201 hardware-verified status for v_sqrt_f32: sNaN and negative normal/inf raise INV;
// negative subnormal flushes without INV; positive non-exact normal raises INX unless OMOD/CLAMP.
TEST(MathSqrtF32Gfx12, StatusRules) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint32_t input;
    uint32_t mode;
    uint32_t causes;
  } cases[] = {
      {0x40800000U, 0, 0},
      {0x40000000U, 0, RJ_ARITH_EXCEPTION_INEXACT},
      {0x40000000U, RJ_ARITH_ALU_OMOD_2, 0},
      {0x40000000U, RJ_ARITH_ALU_CLAMP, 0},
      {0x80000001U, 0, 0},
      {0xc0800000U, 0, RJ_ARITH_EXCEPTION_INVALID},
      {0xc0800000U, RJ_ARITH_ALU_CLAMP, 0},
      {0x7f800001U, 0, RJ_ARITH_EXCEPTION_INVALID},
      {0x7fc00000U, 0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
      for (const auto &sample : cases) {
        Registers<2, 32> r;
        r.v[0][0] = sample.input;
        uint32_t status = 0x80U;
        ASSERT_EQ(
            rj_arith_v_sqrt_f32_w32_gfx12(cpu | semantics, 1U, sample.mode, r.v + 1, r.v, &status),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 0x80U | sample.causes);
      }
    }
  }
}
