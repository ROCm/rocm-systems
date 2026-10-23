// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for cube-map coordinates, face selection, and major-axis magnitude.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubeidF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubeid_f32_w32, 0, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubeidF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubeid_f32_w32_gfx12, 0, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubescF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubesc_f32_w32, 1, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubescF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubesc_f32_w32_gfx12, 1, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubetcF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubetc_f32_w32, 2, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubetcF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubetc_f32_w32_gfx12, 2, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubemaF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubema_f32_w32, 3, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubemaF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(rj_arith_v_cubema_f32_w32_gfx12, 3, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubeidF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubeid_f32_w64, 0, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubeidF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubeid_f32_w64_gfx12, 0, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubescF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubesc_f32_w64, 1, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubescF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubesc_f32_w64_gfx12, 1, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubetcF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubetc_f32_w64, 2, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubetcF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubetc_f32_w64_gfx12, 2, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubemaF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubema_f32_w64, 3, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(GraphicsCubemaF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(rj_arith_v_cubema_f32_w64_gfx12, 3, 0, false, true);
}
