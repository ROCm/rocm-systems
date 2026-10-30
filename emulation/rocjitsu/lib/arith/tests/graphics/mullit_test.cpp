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

// Selected MODE witnesses from rocjitsu's MULLIT regressions (#13669).
TEST(GraphicsMullit, CompatibilityGuestMode) {
  using namespace rocjitsu::arith::test;
  constexpr struct {
    uint32_t a, b, c, expected;
    uint64_t fp;
  } cases[] = {
      {0x2d1d4000, 0x49032148, 0x44630000, 0x36a11862, RJ_ARITH_FP_ROUND_ZERO},
      {0x00800000, 0x3f7fffff, 0x3f800000, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {0x3f800000, 0x3f800000, 1, 0xff7fffff, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS},
      {0x7f80002a, 0, 0x3f800000, 0, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_mullit_f32_w32, rj_arith_v_mullit_f32_w32_gfx12}) {
      for (const auto &sample : cases) {
        Registers<4, 32> r;
        r.v[0][0] = sample.a;
        r.v[1][0] = sample.b;
        r.v[2][0] = sample.c;
        uint32_t status = 0x80000000U;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | sample.fp, 1, 0, r.v + 3, r.v,
                       r.v + 1, r.v + 2, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0], sample.expected);
        EXPECT_EQ(status, 0x80000000U);
      }
    }
  }
}
