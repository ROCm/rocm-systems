// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for single-halfword integer arithmetic and selection.

#include "integer/half_binary_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAddNcI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_add_nc_i16_w32,
                                                  rocjitsu::arith::test::HalfBinary::AddI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAddNcI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_add_nc_i16_w64,
                                                  rocjitsu::arith::test::HalfBinary::AddI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerSubNcI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_sub_nc_i16_w32,
                                                  rocjitsu::arith::test::HalfBinary::SubI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerSubNcI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_sub_nc_i16_w64,
                                                  rocjitsu::arith::test::HalfBinary::SubI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAddNcU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_add_nc_u16_w32,
                                                  rocjitsu::arith::test::HalfBinary::AddU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAddNcU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_add_nc_u16_w64,
                                                  rocjitsu::arith::test::HalfBinary::AddU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerSubNcU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_sub_nc_u16_w32,
                                                  rocjitsu::arith::test::HalfBinary::SubU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerSubNcU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_sub_nc_u16_w64,
                                                  rocjitsu::arith::test::HalfBinary::SubU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMinI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_min_i16_w32,
                                                  rocjitsu::arith::test::HalfBinary::MinI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMinI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_min_i16_w64,
                                                  rocjitsu::arith::test::HalfBinary::MinI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMaxI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_max_i16_w32,
                                                  rocjitsu::arith::test::HalfBinary::MaxI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMaxI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_max_i16_w64,
                                                  rocjitsu::arith::test::HalfBinary::MaxI16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMinU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_min_u16_w32,
                                                  rocjitsu::arith::test::HalfBinary::MinU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMinU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_min_u16_w64,
                                                  rocjitsu::arith::test::HalfBinary::MinU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMaxU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_max_u16_w32,
                                                  rocjitsu::arith::test::HalfBinary::MaxU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMaxU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_max_u16_w64,
                                                  rocjitsu::arith::test::HalfBinary::MaxU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMulLoU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_mul_lo_u16_w32,
                                                  rocjitsu::arith::test::HalfBinary::MulLoU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerMulLoU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_mul_lo_u16_w64,
                                                  rocjitsu::arith::test::HalfBinary::MulLoU16);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerLshlrevB16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_lshlrev_b16_w32,
                                                  rocjitsu::arith::test::HalfBinary::Lshl);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerLshlrevB16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_lshlrev_b16_w64,
                                                  rocjitsu::arith::test::HalfBinary::Lshl);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerLshrrevB16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_lshrrev_b16_w32,
                                                  rocjitsu::arith::test::HalfBinary::Lshr);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerLshrrevB16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_lshrrev_b16_w64,
                                                  rocjitsu::arith::test::HalfBinary::Lshr);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAshrrevI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<32, 2>(rj_arith_v_ashrrev_i16_w32,
                                                  rocjitsu::arith::test::HalfBinary::Ashr);
}

// Check single-halfword integer arithmetic and destination preservation.
TEST(IntegerAshrrevI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_binary<64, 2>(rj_arith_v_ashrrev_i16_w64,
                                                  rocjitsu::arith::test::HalfBinary::Ashr);
}
