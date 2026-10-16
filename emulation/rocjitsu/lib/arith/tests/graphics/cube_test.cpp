// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for cube-map coordinates, face selection, and major-axis magnitude.

#include "graphics/helpers.h"

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubeid_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubeid_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      0, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubeid_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubeid_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      0, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubesc_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubesc_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      1, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubesc_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubesc_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      1, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubetc_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubetc_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      2, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubetc_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubetc_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      2, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubema_f32_w32, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubema_f32_w32(flags, exec_mask, mode, d, a, b, c);
      },
      3, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubema_f32_w32_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<32>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubema_f32_w32_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      3, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubeid_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubeid_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      0, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubeid_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubeid_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      0, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubesc_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubesc_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      1, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubesc_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubesc_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      1, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubetc_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubetc_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      2, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubetc_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubetc_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      2, 0, false, true);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubema_f32_w64, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubema_f32_w64(flags, exec_mask, mode, d, a, b, c);
      },
      3, 0, false, false);
}

// Check graphics results, modifiers, EXEC masks, and source/destination aliases.
TEST(v_cubema_f32_w64_gfx12, Arithmetic) {
  rocjitsu::arith::test::check_graphics<64>(
      [](auto flags, auto exec_mask, auto mode, auto d, auto a, auto b, auto c, auto status) {
        (void)status;
        return rj_arith_v_cubema_f32_w64_gfx12(flags, exec_mask, mode, d, a, b, c);
      },
      3, 0, false, true);
}
