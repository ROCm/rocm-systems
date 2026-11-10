// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for single-halfword integer multiply-add and three-input selection.

#include "integer/half_ternary_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMadU16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_mad_u16_w32,
                                                   rocjitsu::arith::test::HalfTernary::MadU16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMadU16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_mad_u16_w64,
                                                   rocjitsu::arith::test::HalfTernary::MadU16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMadI16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_mad_i16_w32,
                                                   rocjitsu::arith::test::HalfTernary::MadI16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMadI16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_mad_i16_w64,
                                                   rocjitsu::arith::test::HalfTernary::MadI16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMin3U16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_min3_u16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Min3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMin3U16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_min3_u16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Min3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMin3I16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_min3_i16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Min3I16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMin3I16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_min3_i16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Min3I16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMax3U16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_max3_u16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Max3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMax3U16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_max3_u16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Max3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMax3I16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_max3_i16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Max3I16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMax3I16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_max3_i16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Max3I16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMed3U16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_med3_u16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Med3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMed3U16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_med3_u16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Med3U16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMed3I16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<32, 3>(rj_arith_v_med3_i16_w32,
                                                   rocjitsu::arith::test::HalfTernary::Med3I16);
}

// Check 16-bit ternary arithmetic and destination preservation.
TEST(IntegerMed3I16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_ternary<64, 3>(rj_arith_v_med3_i16_w64,
                                                   rocjitsu::arith::test::HalfTernary::Med3I16);
}
