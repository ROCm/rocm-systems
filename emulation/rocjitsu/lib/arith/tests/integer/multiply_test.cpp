// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for integer multiplication and high-half products.

#include "integer/multiply_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulLoU32W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_lo_u32_w32, rocjitsu::arith::test::Multiply::LoU32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulLoU32W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_lo_u32_w64, rocjitsu::arith::test::Multiply::LoU32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiU32W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_hi_u32_w32, rocjitsu::arith::test::Multiply::HiU32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiU32W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_hi_u32_w64, rocjitsu::arith::test::Multiply::HiU32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiI32W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_hi_i32_w32, rocjitsu::arith::test::Multiply::HiI32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiI32W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/false>(
      rj_arith_v_mul_hi_i32_w64, rocjitsu::arith::test::Multiply::HiI32);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulI32I24W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_i32_i24_w32, rocjitsu::arith::test::Multiply::I24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulI32I24W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_i32_i24_w64, rocjitsu::arith::test::Multiply::I24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiI32I24W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_hi_i32_i24_w32, rocjitsu::arith::test::Multiply::HiI24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiI32I24W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_hi_i32_i24_w64, rocjitsu::arith::test::Multiply::HiI24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulU32U24W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_u32_u24_w32, rocjitsu::arith::test::Multiply::U24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulU32U24W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_u32_u24_w64, rocjitsu::arith::test::Multiply::U24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiU32U24W32, Arithmetic) {
  rocjitsu::arith::test::check_multiply<32, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_hi_u32_u24_w32, rocjitsu::arith::test::Multiply::HiU24);
}

// Check low/high integer products with saturation and aliases.
TEST(IntegerMulHiU32U24W64, Arithmetic) {
  rocjitsu::arith::test::check_multiply<64, 2, /*SupportsDpp=*/true>(
      rj_arith_v_mul_hi_u32_u24_w64, rocjitsu::arith::test::Multiply::HiU24);
}
