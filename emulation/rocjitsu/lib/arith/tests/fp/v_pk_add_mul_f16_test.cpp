// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 addition and multiplication.

#include "fp/binary_arithmetic_test.h"
#include "rj_arith/rj_arith.h"

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
