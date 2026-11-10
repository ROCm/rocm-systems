// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for narrow-factor integer multiply-add.

#include "integer/mad_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadU32U16W32, Arithmetic) {
  rocjitsu::arith::test::check_mad<32, 3>(rj_arith_v_mad_u32_u16_w32,
                                          rocjitsu::arith::test::Mad::U16);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadU32U16W64, Arithmetic) {
  rocjitsu::arith::test::check_mad<64, 3>(rj_arith_v_mad_u32_u16_w64,
                                          rocjitsu::arith::test::Mad::U16);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadI32I16W32, Arithmetic) {
  rocjitsu::arith::test::check_mad<32, 3>(rj_arith_v_mad_i32_i16_w32,
                                          rocjitsu::arith::test::Mad::I16);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadI32I16W64, Arithmetic) {
  rocjitsu::arith::test::check_mad<64, 3>(rj_arith_v_mad_i32_i16_w64,
                                          rocjitsu::arith::test::Mad::I16);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadU32U24W32, Arithmetic) {
  rocjitsu::arith::test::check_mad<32, 3>(rj_arith_v_mad_u32_u24_w32,
                                          rocjitsu::arith::test::Mad::U24);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadU32U24W64, Arithmetic) {
  rocjitsu::arith::test::check_mad<64, 3>(rj_arith_v_mad_u32_u24_w64,
                                          rocjitsu::arith::test::Mad::U24);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadI32I24W32, Arithmetic) {
  rocjitsu::arith::test::check_mad<32, 3>(rj_arith_v_mad_i32_i24_w32,
                                          rocjitsu::arith::test::Mad::I24);
}

// Check narrow-factor multiply-add with routing and aliases.
TEST(IntegerMadI32I24W64, Arithmetic) {
  rocjitsu::arith::test::check_mad<64, 3>(rj_arith_v_mad_i32_i24_w64,
                                          rocjitsu::arith::test::Mad::I24);
}
