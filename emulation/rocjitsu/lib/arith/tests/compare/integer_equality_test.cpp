// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for signed and unsigned integer equality predicates.

#include "compare/integer_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i16_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i16_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i16_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i16_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u16_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u16_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u16_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU16W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u16_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i32_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i32_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i32_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i32_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u32_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u32_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u32_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU32W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u32_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i64_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i64_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i64_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i64_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u64_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u64_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u64_w32, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU64W32, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 32, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u64_w32, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i16_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i16_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i16_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i16_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u16_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u16_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u16_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU16W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<16, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u16_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i32_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i32_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i32_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i32_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u32_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u32_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u32_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU32W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<32, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u32_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_eq_i64_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmp_ne_i64_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_eq_i64_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeI64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/true>(
      rj_arith_v_cmpx_ne_i64_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpEqU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_eq_u64_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpNeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmp_ne_u64_w64, [](uint64_t a, uint64_t b) { return a != b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxEqU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_eq_u64_w64, [](uint64_t a, uint64_t b) { return a == b; });
}

// Check signed and unsigned integer equality predicates.
TEST(CompareCmpxNeU64W64, Arithmetic) {
  rocjitsu::arith::test::check_integer_compare<64, 64, /*Signed=*/false>(
      rj_arith_v_cmpx_ne_u64_w64, [](uint64_t a, uint64_t b) { return a != b; });
}
