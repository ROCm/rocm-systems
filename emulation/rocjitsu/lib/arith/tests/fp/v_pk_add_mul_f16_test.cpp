// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 addition and multiplication.

#include "fp/binary_arithmetic_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w32_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// The packed guest path applies flushing and finite-overflow saturation to both halves.
TEST(FpPackedBinaryF16, GuestControls) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_pk_mul_f16_w32, rj_arith_v_pk_mul_f16_w32_gfx12}) {
      Registers<3, 32> r;
      r.v[0][0] = 0x00010001U;
      r.v[1][0] = 0x3c003c00U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 1, 0,
                     r.v + 2, r.v, r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0U);
      r.v[0][0] = 0x7bff7bffU;
      r.v[1][0] = 0x40004000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP16_OVFL, 1, 0, r.v + 2, r.v,
                     r.v + 1, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[2][0], 0x7bff7bffU);
    }
  }
}

// Packed products judge tininess after rounding like V_MUL_F16: the low half's
// (1 - 2^-11) * 2^-14 flushes although subnormal rounding gives the smallest normal. Expected
// results are a gfx1201 capture in nearest, up, down and zero rounding.
TEST(FpPackedBinaryF16, TinyAfterRounding) {
  using namespace rocjitsu::arith::test;
  const uint32_t flushed[] = {0x43ff0000U, 0x43ff0000U, 0x43ff0000U, 0x43ff0000U};
  const uint32_t kept[] = {0x43ff0400U, 0x43ff0400U, 0x43ff03ffU, 0x43ff03ffU};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        for (bool flush : {true, false}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x84003bffU;
          r.v[1][0] = 0xfbff0400U;
          ASSERT_EQ(
              rj_arith_v_pk_mul_f16_w32_gfx12(cpu | semantics | (uint64_t(rounding) << 24) |
                                                  (flush ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0),
                                              1, 0, r.v + 2, r.v, r.v + 1, nullptr),
              RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[2][0], flush ? flushed[rounding] : kept[rounding])
              << "rounding " << rounding << " flush " << flush;
        }
      }
    }
  }
}

// gfx11 packed and ordinary FP16 arithmetic retain input sNaNs unless IEEE requests quieting.
TEST(FpPackedBinaryF16, CompatibilityNaNs) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding : {uint64_t{0}, RJ_ARITH_FP_ROUND_ZERO}) {
      for (uint64_t ieee : {uint64_t{0}, RJ_ARITH_FP_IEEE_MODE}) {
        for (auto call : {rj_arith_v_pk_add_f16_w32, rj_arith_v_pk_mul_f16_w32}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x3c003c00U;
          r.v[1][0] = 0x7d017c01U;
          for (uint32_t instruction_flags : {0U, RJ_ARITH_PK_CLAMP}) {
            ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding | ieee |
                               RJ_ARITH_FP_DX10_CLAMP_OFF,
                           1, instruction_flags, r.v + 2, r.v, r.v + 1, nullptr),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(r.v[2][0], ieee ? 0x7f017e01U : 0x7d017c01U);
          }
        }
        for (auto call : {rj_arith_v_add_f16_w32, rj_arith_v_mul_f16_w32}) {
          Registers<3, 32> r;
          r.v[0][0] = 0x3c00U;
          r.v[1][0] = 0x7c01U;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | rounding | ieee, 1, 0, r.v + 2,
                         r.v, r.v + 1, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[2][0] & 0xffffU, ieee ? 0x7e01U : 0x7c01U);
        }
      }
    }
  }
}
