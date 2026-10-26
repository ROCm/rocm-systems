// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for naN-propagating minimum/maximum.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_minimum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_minimum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_minimum_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_minimum_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_minimum_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinimumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_minimum_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_maximum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_maximum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_maximum_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_maximum_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_maximum_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaximumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_maximum_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Propagating selection quiets the first NaN in source order, as gfx1201 does.
// A quiet A beats a signaling B, which the ISA pseudocode would prefer instead.
TEST(FpPropagatingSelection, FirstNanWinsOverSignaling) {
  const struct {
    decltype(&rj_arith_v_minimum_f32_w32) call;
    uint32_t quiet_nan, signaling_nan;
  } cases[] = {{rj_arith_v_minimum_f16_w32, 0x7e00U, 0x7c01U},
               {rj_arith_v_maximum_f16_w32, 0x7e00U, 0x7c01U},
               {rj_arith_v_minimum_f32_w32, 0x7fc00000U, 0x7f800001U},
               {rj_arith_v_maximum_f32_w32, 0x7fc00000U, 0x7f800001U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &sample : cases) {
      rocjitsu::arith::test::Registers<3, 32> r;
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = sample.quiet_nan;
        r.v[1][lane] = sample.signaling_nan;
        r.v[2][lane] = 0;
      }
      ASSERT_EQ(sample.call(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0xffffffffU, 0, r.v + 2, r.v, r.v + 1,
                            nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(r.v[2][lane], sample.quiet_nan);
      }
    }
  }
}
