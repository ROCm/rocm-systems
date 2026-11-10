// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for combined shift, boolean, and three-input integer operations.

#include "integer/combined_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check combined integer operations with routing and aliases.
TEST(IntegerLshlAddU32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_lshl_add_u32_w32,
                                               rocjitsu::arith::test::Ternary::LshlAdd);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerLshlAddU32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_lshl_add_u32_w64,
                                               rocjitsu::arith::test::Ternary::LshlAdd);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerAddLshlU32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_add_lshl_u32_w32,
                                               rocjitsu::arith::test::Ternary::AddLshl);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerAddLshlU32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_add_lshl_u32_w64,
                                               rocjitsu::arith::test::Ternary::AddLshl);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerLshlOrB32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_lshl_or_b32_w32,
                                               rocjitsu::arith::test::Ternary::LshlOr);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerLshlOrB32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_lshl_or_b32_w64,
                                               rocjitsu::arith::test::Ternary::LshlOr);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerAndOrB32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_and_or_b32_w32,
                                               rocjitsu::arith::test::Ternary::AndOr);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerAndOrB32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_and_or_b32_w64,
                                               rocjitsu::arith::test::Ternary::AndOr);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerOr3B32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_or3_b32_w32,
                                               rocjitsu::arith::test::Ternary::Or3);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerOr3B32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_or3_b32_w64,
                                               rocjitsu::arith::test::Ternary::Or3);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerXor3B32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_xor3_b32_w32,
                                               rocjitsu::arith::test::Ternary::Xor3);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerXor3B32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_xor3_b32_w64,
                                               rocjitsu::arith::test::Ternary::Xor3);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerXadU32W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_xad_u32_w32,
                                               rocjitsu::arith::test::Ternary::Xad);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerXadU32W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_xad_u32_w64,
                                               rocjitsu::arith::test::Ternary::Xad);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerLerpU8W32, Arithmetic) {
  rocjitsu::arith::test::check_combined<32, 3>(rj_arith_v_lerp_u8_w32,
                                               rocjitsu::arith::test::Ternary::Lerp);
}

// Check combined integer operations with routing and aliases.
TEST(IntegerLerpU8W64, Arithmetic) {
  rocjitsu::arith::test::check_combined<64, 3>(rj_arith_v_lerp_u8_w64,
                                               rocjitsu::arith::test::Ternary::Lerp);
}
