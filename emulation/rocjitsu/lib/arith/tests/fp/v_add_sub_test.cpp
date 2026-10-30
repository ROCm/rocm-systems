// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP add/subtract and reversed-subtract entry points.

#include "fp/binary_arithmetic_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_add_f32_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_add_f32_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 32>(
      rj_arith_v_add_f64_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpAddF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<64, 64>(
      rj_arith_v_add_f64_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_sub_f32_w32, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64_gfx12, [](double x, double y) { return x - y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_sub_f32_w64, [](double x, double y) { return x - y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 32>(
      rj_arith_v_subrev_f32_w32, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64_gfx12, [](double x, double y) { return y - x; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpSubrevF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<32, 64>(
      rj_arith_v_subrev_f32_w64, [](double x, double y) { return y - x; }, /*gfx12=*/false);
}

// gfx1100/gfx1201 subtraction captures select NaNs in original source order.
TEST(FpSubtractF32, NaNSignAndPriority) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_sub_f32_w32, rj_arith_v_sub_f32_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3f800000U;
      r.v[1][0] = 0x7fc00001U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0xffc00001U);
    }
    for (auto call : {rj_arith_v_subrev_f32_w32, rj_arith_v_subrev_f32_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x7fc00001U;
      r.v[1][0] = 0xffc00002U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0xffc00001U);
    }
    // RDNA3 FMA's inherited IEEE-clear model preserves the selected signaling NaN.
    for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x3f800000U;
      r.v[1][0] = 0x7f800001U;
      ASSERT_EQ(
          rj_arith_v_sub_f32_w32(cpu | ieee | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                                 1, 0, r.v + 2, r.v, r.v + 1, nullptr),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], ieee ? 0xffc00001U : 0xff800001U);
    }
  }
}

// DX10_CLAMP_OFF is a shared guest control, including for ordinary RDNA3 binary arithmetic.
TEST(FpBinaryF32, NaNClampControl) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (auto call : {rj_arith_v_add_f32_w32, rj_arith_v_sub_f32_w32, rj_arith_v_subrev_f32_w32,
                        rj_arith_v_mul_f32_w32, rj_arith_v_mul_dx9_zero_f32_w32}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x7fc00001U;
        r.v[1][0] = 0x3f800000U;
        ASSERT_EQ(call(cpu | semantics | RJ_ARITH_FP_DX10_CLAMP_OFF, 1, RJ_ARITH_ALU_CLAMP, r.v + 2,
                       r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_GT(r.v[2][0] & 0x7fffffffU, 0x7f800000U);
      }
    }
  }
}

// Subtraction negates the subtrahend, including NaNs, without swapping source priority.
TEST(FpSubF16, CompatibilityNanSigns) {
  using namespace rocjitsu::arith::test;
  constexpr uint16_t inputs[][4] = {
      {0x8000, 0x7e00, 0xfe00, 0x7e00},
      {0x0000, 0x7c01, 0xfe01, 0x7e01},
      {0x7e00, 0x7e00, 0x7e00, 0xfe00},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool reverse : {false, true}) {
      auto call = reverse ? rj_arith_v_subrev_f16_w32_gfx12 : rj_arith_v_sub_f16_w32_gfx12;
      for (const auto &sample : inputs) {
        Registers<3, 32> r;
        r.v[0][0] = sample[0];
        r.v[1][0] = sample[1];
        uint32_t status = 0;
        ASSERT_EQ(
            call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, r.v + 2, r.v, r.v + 1, &status),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0] & 0xffffU, sample[reverse ? 3 : 2]);
      }
    }
  }
}

// Compatibility retains Rocjitsu's quieting policy, including gfx11 with IEEE clear.
TEST(FpBinaryF16, CompatibilitySignalingNan) {
  using namespace rocjitsu::arith::test;
  const auto calls = {rj_arith_v_add_f16_w32, rj_arith_v_sub_f16_w32, rj_arith_v_mul_f16_w32};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : calls) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_UP}) {
          for (uint64_t omod : {uint64_t(0), uint64_t(RJ_ARITH_ALU_OMOD_2)}) {
            Registers<3, 32> r;
            r.v[0][0] = 0x3c00U;
            r.v[1][0] = 0x7c01U;
            // gfx11 OMOD is enabled by output flushing and IEEE clear.
            uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee | rounding |
                             (omod ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
            ASSERT_EQ(call(flags, 1, omod, r.v + 2, r.v, r.v + 1, nullptr), RJ_ARITH_SUCCESS);
            EXPECT_EQ(r.v[2][0] & 0xffffU, call == rj_arith_v_sub_f16_w32 ? 0xfe01U : 0x7e01U);
          }
        }
      }
    }
  }
}
