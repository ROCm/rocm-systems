// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for integer addition/subtraction and saturation.

#include "integer/add_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAddNcU32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 2>(rj_arith_v_add_nc_u32_w32,
                                          rocjitsu::arith::test::Add::AddU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAddNcU32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 2>(rj_arith_v_add_nc_u32_w64,
                                          rocjitsu::arith::test::Add::AddU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubNcU32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 2>(rj_arith_v_sub_nc_u32_w32,
                                          rocjitsu::arith::test::Add::SubU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubNcU32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 2>(rj_arith_v_sub_nc_u32_w64,
                                          rocjitsu::arith::test::Add::SubU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubrevNcU32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 2>(rj_arith_v_subrev_nc_u32_w32,
                                          rocjitsu::arith::test::Add::SubrevU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubrevNcU32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 2>(rj_arith_v_subrev_nc_u32_w64,
                                          rocjitsu::arith::test::Add::SubrevU32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAddNcI32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 2>(rj_arith_v_add_nc_i32_w32,
                                          rocjitsu::arith::test::Add::AddI32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAddNcI32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 2>(rj_arith_v_add_nc_i32_w64,
                                          rocjitsu::arith::test::Add::AddI32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubNcI32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 2>(rj_arith_v_sub_nc_i32_w32,
                                          rocjitsu::arith::test::Add::SubI32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerSubNcI32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 2>(rj_arith_v_sub_nc_i32_w64,
                                          rocjitsu::arith::test::Add::SubI32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAdd3U32W32, Arithmetic) {
  rocjitsu::arith::test::check_add<32, 3>(rj_arith_v_add3_u32_w32,
                                          rocjitsu::arith::test::Add::Add3U32);
}

// Check integer addition/subtraction with saturation, routing, and aliases.
TEST(IntegerAdd3U32W64, Arithmetic) {
  rocjitsu::arith::test::check_add<64, 3>(rj_arith_v_add3_u32_w64,
                                          rocjitsu::arith::test::Add::Add3U32);
}
