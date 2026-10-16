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
TEST(FpPkAddF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w32, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64_gfx12, [](double x, double y) { return x + y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkAddF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_add_f16_w64, [](double x, double y) { return x + y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W32, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w32, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64_gfx12, [](double x, double y) { return x * y; }, /*gfx12=*/true);
}

// Check arithmetic results, modifiers, masks, aliases, and supported FP status.
TEST(FpPkMulF16W64, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64, /*Packed=*/true>(
      rj_arith_v_pk_mul_f16_w64, [](double x, double y) { return x * y; }, /*gfx12=*/false);
}
