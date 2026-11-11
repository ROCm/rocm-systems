// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for integer arithmetic with carry/borrow masks.

#include "integer/carry_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerAddCoU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/false>(
      rj_arith_v_add_co_u32_w32, rocjitsu::arith::test::Carry::Add);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerAddCoCiU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/true>(
      rj_arith_v_add_co_ci_u32_w32, rocjitsu::arith::test::Carry::AddCi);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubCoU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/false>(
      rj_arith_v_sub_co_u32_w32, rocjitsu::arith::test::Carry::Sub);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubCoCiU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/true>(
      rj_arith_v_sub_co_ci_u32_w32, rocjitsu::arith::test::Carry::SubCi);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubrevCoU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/false>(
      rj_arith_v_subrev_co_u32_w32, rocjitsu::arith::test::Carry::Subrev);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubrevCoCiU32W32, Arithmetic) {
  rocjitsu::arith::test::check_carry<32, /*WithInputCarry=*/true>(
      rj_arith_v_subrev_co_ci_u32_w32, rocjitsu::arith::test::Carry::SubrevCi);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerAddCoU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/false>(
      rj_arith_v_add_co_u32_w64, rocjitsu::arith::test::Carry::Add);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerAddCoCiU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/true>(
      rj_arith_v_add_co_ci_u32_w64, rocjitsu::arith::test::Carry::AddCi);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubCoU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/false>(
      rj_arith_v_sub_co_u32_w64, rocjitsu::arith::test::Carry::Sub);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubCoCiU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/true>(
      rj_arith_v_sub_co_ci_u32_w64, rocjitsu::arith::test::Carry::SubCi);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubrevCoU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/false>(
      rj_arith_v_subrev_co_u32_w64, rocjitsu::arith::test::Carry::Subrev);
}

// Check carry/borrow masks alongside arithmetic, EXEC, and aliases.
TEST(IntegerSubrevCoCiU32W64, Arithmetic) {
  rocjitsu::arith::test::check_carry<64, /*WithInputCarry=*/true>(
      rj_arith_v_subrev_co_ci_u32_w64, rocjitsu::arith::test::Carry::SubrevCi);
}
