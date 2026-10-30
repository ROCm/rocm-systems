// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for 32/64-bit shifts and count masking.

#include "integer/shift_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshlrevB32W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 32>(rj_arith_v_lshlrev_b32_w32,
                                             rocjitsu::arith::test::Shift::Left);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshrrevB32W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 32>(rj_arith_v_lshrrev_b32_w32,
                                             rocjitsu::arith::test::Shift::LogicalRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerAshrrevI32W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 32>(rj_arith_v_ashrrev_i32_w32,
                                             rocjitsu::arith::test::Shift::ArithmeticRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshlrevB64W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 32>(rj_arith_v_lshlrev_b64_w32,
                                             rocjitsu::arith::test::Shift::Left);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshrrevB64W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 32>(rj_arith_v_lshrrev_b64_w32,
                                             rocjitsu::arith::test::Shift::LogicalRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerAshrrevI64W32, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 32>(rj_arith_v_ashrrev_i64_w32,
                                             rocjitsu::arith::test::Shift::ArithmeticRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshlrevB32W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 64>(rj_arith_v_lshlrev_b32_w64,
                                             rocjitsu::arith::test::Shift::Left);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshrrevB32W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 64>(rj_arith_v_lshrrev_b32_w64,
                                             rocjitsu::arith::test::Shift::LogicalRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerAshrrevI32W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<32, 64>(rj_arith_v_ashrrev_i32_w64,
                                             rocjitsu::arith::test::Shift::ArithmeticRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshlrevB64W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 64>(rj_arith_v_lshlrev_b64_w64,
                                             rocjitsu::arith::test::Shift::Left);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerLshrrevB64W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 64>(rj_arith_v_lshrrev_b64_w64,
                                             rocjitsu::arith::test::Shift::LogicalRight);
}

// Check wide/narrow shifts, count masking, and aliases.
TEST(IntegerAshrrevI64W64, Arithmetic) {
  rocjitsu::arith::test::check_shift<64, 64>(rj_arith_v_ashrrev_i64_w64,
                                             rocjitsu::arith::test::Shift::ArithmeticRight);
}
