// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP16/FP32 interpolation stages and quad-lane operand routing.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_f16_f32_w32, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_f16_f32_w32, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p10_rtz_f16_f32_w32, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_interp_p2_rtz_f16_f32_w32, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp32, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_f16_f32_w64, 0,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2F16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_f16_f32_w64, 1,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP10RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p10_rtz_f16_f32_w64, 2,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsInterpP2RtzF16F32W64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_interp_p2_rtz_f16_f32_w64, 3,
                                            rocjitsu::arith::test::GraphicsFamily::Interp16, false);
}
