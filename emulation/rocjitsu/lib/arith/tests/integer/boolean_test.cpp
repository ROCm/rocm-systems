// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for two- and three-input bitwise boolean operations.

#include "integer/boolean_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_and_b32_w32,
                                              rocjitsu::arith::test::Boolean::And32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_and_b32_w64,
                                              rocjitsu::arith::test::Boolean::And32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_or_b32_w32,
                                              rocjitsu::arith::test::Boolean::Or32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_or_b32_w64,
                                              rocjitsu::arith::test::Boolean::Or32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xor_b32_w32,
                                              rocjitsu::arith::test::Boolean::Xor32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xor_b32_w64,
                                              rocjitsu::arith::test::Boolean::Xor32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 1>(rj_arith_v_not_b32_w32,
                                              rocjitsu::arith::test::Boolean::Not32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 1>(rj_arith_v_not_b32_w64,
                                              rocjitsu::arith::test::Boolean::Not32);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_and_b16_w32,
                                              rocjitsu::arith::test::Boolean::And16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerAndB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_and_b16_w64,
                                              rocjitsu::arith::test::Boolean::And16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_or_b16_w32,
                                              rocjitsu::arith::test::Boolean::Or16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerOrB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_or_b16_w64,
                                              rocjitsu::arith::test::Boolean::Or16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xor_b16_w32,
                                              rocjitsu::arith::test::Boolean::Xor16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXorB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xor_b16_w64,
                                              rocjitsu::arith::test::Boolean::Xor16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB16W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 1>(rj_arith_v_not_b16_w32,
                                              rocjitsu::arith::test::Boolean::Not16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerNotB16W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 1>(rj_arith_v_not_b16_w64,
                                              rocjitsu::arith::test::Boolean::Not16);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXnorB32W32, Arithmetic) {
  rocjitsu::arith::test::check_boolean<32, 2>(rj_arith_v_xnor_b32_w32,
                                              rocjitsu::arith::test::Boolean::Xnor);
}

// Check boolean operations and supported truth-table controls.
TEST(IntegerXnorB32W64, Arithmetic) {
  rocjitsu::arith::test::check_boolean<64, 2>(rj_arith_v_xnor_b32_w64,
                                              rocjitsu::arith::test::Boolean::Xnor);
}
