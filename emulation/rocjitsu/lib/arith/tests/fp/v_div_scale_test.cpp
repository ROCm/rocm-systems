// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division operand rescaling and the resulting condition mask.

#include "fp/div_scale_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W32, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 32>(rj_arith_v_div_scale_f32_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF32W64, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<32, 64>(rj_arith_v_div_scale_f32_w64);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W32, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 32>(rj_arith_v_div_scale_f64_w32);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64_gfx12);
}

// Check scaled operands and the associated condition-mask output.
TEST(FpDivScaleF64W64, Arithmetic) {
  rocjitsu::arith::test::check_div_scale<64, 64>(rj_arith_v_div_scale_f64_w64);
}
