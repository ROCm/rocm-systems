// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for reciprocal-with-flags numeric results and FP-status behavior.

#include "math/unary_test.h"

// Check unary math results and semantics/status support across CPU levels.
TEST(v_rcp_iflag_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, false>(rj_arith_v_rcp_iflag_f32_w32, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_rcp_iflag_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<32, false, true>(rj_arith_v_rcp_iflag_f32_w32_gfx12, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_rcp_iflag_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, false>(rj_arith_v_rcp_iflag_f32_w64, 5, true);
}

// Check unary math results and semantics/status support across CPU levels.
TEST(v_rcp_iflag_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_unary<64, false, true>(rj_arith_v_rcp_iflag_f32_w64_gfx12, 5, true);
}
