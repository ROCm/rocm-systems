// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point class predicates and their lane-mask results.

#include "compare/class_test.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <gtest/gtest.h>

// Check class predicates and result masks.
TEST(CompareCmpClassF16W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<16, 32>(rj_arith_v_cmp_class_f16_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpClassF32W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<32, 32>(rj_arith_v_cmp_class_f32_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpClassF64W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<64, 32>(rj_arith_v_cmp_class_f64_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF16W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<16, 32>(rj_arith_v_cmpx_class_f16_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF32W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<32, 32>(rj_arith_v_cmpx_class_f32_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF64W32, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<64, 32>(rj_arith_v_cmpx_class_f64_w32);
}

// Check class predicates and result masks.
TEST(CompareCmpClassF16W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<16, 64>(rj_arith_v_cmp_class_f16_w64);
}

// Check class predicates and result masks.
TEST(CompareCmpClassF32W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<32, 64>(rj_arith_v_cmp_class_f32_w64);
}

// Check class predicates and result masks.
TEST(CompareCmpClassF64W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<64, 64>(rj_arith_v_cmp_class_f64_w64);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF16W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<16, 64>(rj_arith_v_cmpx_class_f16_w64);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF32W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<32, 64>(rj_arith_v_cmpx_class_f32_w64);
}

// Check class predicates and result masks.
TEST(CompareCmpxClassF64W64, Arithmetic) {
  rocjitsu::arith::test::check_class_compare<64, 64>(rj_arith_v_cmpx_class_f64_w64);
}
