// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for integer minimum, maximum, and median.

#include "integer/select_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinI32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 2>(rj_arith_v_min_i32_w32,
                                             rocjitsu::arith::test::Select::MinI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinI32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 2>(rj_arith_v_min_i32_w64,
                                             rocjitsu::arith::test::Select::MinI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxI32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 2>(rj_arith_v_max_i32_w32,
                                             rocjitsu::arith::test::Select::MaxI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxI32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 2>(rj_arith_v_max_i32_w64,
                                             rocjitsu::arith::test::Select::MaxI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMin3I32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_min3_i32_w32,
                                             rocjitsu::arith::test::Select::Min3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMin3I32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_min3_i32_w64,
                                             rocjitsu::arith::test::Select::Min3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMax3I32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_max3_i32_w32,
                                             rocjitsu::arith::test::Select::Max3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMax3I32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_max3_i32_w64,
                                             rocjitsu::arith::test::Select::Max3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinmaxI32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_minmax_i32_w32,
                                             rocjitsu::arith::test::Select::MinmaxI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinmaxI32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_minmax_i32_w64,
                                             rocjitsu::arith::test::Select::MinmaxI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxminI32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_maxmin_i32_w32,
                                             rocjitsu::arith::test::Select::MaxminI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxminI32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_maxmin_i32_w64,
                                             rocjitsu::arith::test::Select::MaxminI32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMed3I32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_med3_i32_w32,
                                             rocjitsu::arith::test::Select::Med3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMed3I32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_med3_i32_w64,
                                             rocjitsu::arith::test::Select::Med3I32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinU32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 2>(rj_arith_v_min_u32_w32,
                                             rocjitsu::arith::test::Select::MinU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinU32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 2>(rj_arith_v_min_u32_w64,
                                             rocjitsu::arith::test::Select::MinU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxU32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 2>(rj_arith_v_max_u32_w32,
                                             rocjitsu::arith::test::Select::MaxU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxU32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 2>(rj_arith_v_max_u32_w64,
                                             rocjitsu::arith::test::Select::MaxU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMin3U32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_min3_u32_w32,
                                             rocjitsu::arith::test::Select::Min3U32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMin3U32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_min3_u32_w64,
                                             rocjitsu::arith::test::Select::Min3U32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMax3U32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_max3_u32_w32,
                                             rocjitsu::arith::test::Select::Max3U32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMax3U32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_max3_u32_w64,
                                             rocjitsu::arith::test::Select::Max3U32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinmaxU32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_minmax_u32_w32,
                                             rocjitsu::arith::test::Select::MinmaxU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMinmaxU32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_minmax_u32_w64,
                                             rocjitsu::arith::test::Select::MinmaxU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxminU32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_maxmin_u32_w32,
                                             rocjitsu::arith::test::Select::MaxminU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMaxminU32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_maxmin_u32_w64,
                                             rocjitsu::arith::test::Select::MaxminU32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMed3U32W32, Arithmetic) {
  rocjitsu::arith::test::check_select<32, 3>(rj_arith_v_med3_u32_w32,
                                             rocjitsu::arith::test::Select::Med3U32);
}

// Check integer min/max/median with routing and aliases.
TEST(IntegerMed3U32W64, Arithmetic) {
  rocjitsu::arith::test::check_select<64, 3>(rj_arith_v_med3_u32_w64,
                                             rocjitsu::arith::test::Select::Med3U32);
}
