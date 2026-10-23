// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for lit-multiply special values and gfx11/gfx12 modifier behavior.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_mullit_f32_w32, 0, 0, true, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_mullit_f32_w32_gfx12, 0, 0, true, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_mullit_f32_w64, 0, 0, true, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsMullitF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_mullit_f32_w64_gfx12, 0, 0, true, true);
}
