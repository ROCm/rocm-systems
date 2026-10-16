// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for floating-point logarithm and exponential entry points.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_exp_f32_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_log_f32_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_exp_f32_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_log_f32_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_exp_f32_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_log_f32_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_exp_f32_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_log_f32_w64_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f16_w32, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_exp_f16_w32, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f16_w32, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, false>(rj_arith_v_log_f16_w32, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f16_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_exp_f16_w32_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f16_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, true, true>(rj_arith_v_log_f16_w32_gfx12, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f16_w64, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_exp_f16_w64, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f16_w64, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, false>(rj_arith_v_log_f16_w64, 8);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_exp_f16_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_exp_f16_w64_gfx12, 7);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_log_f16_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, true, true>(rj_arith_v_log_f16_w64_gfx12, 8);
}
