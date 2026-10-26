// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for three-input numeric-preferred FP min/max/median.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3NumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_min3_num_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3NumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_min3_num_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3NumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_min3_num_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3NumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_min3_num_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(
            rocjitsu::arith::test::select_fp(x, y, false, false), z, false, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3NumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_max3_num_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3NumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_max3_num_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3NumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_max3_num_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3NumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_max3_num_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3NumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_med3_num_f16_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3NumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_med3_num_f16_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3NumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_med3_num_f32_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3NumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_med3_num_f32_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// MED3_NUM(-0, -1, +0) is -0 when the median orders -0 below +0, as gfx1201 does.
// FP16 takes that ordered path; FP32 uses floating equality and returns +0.
TEST(FpMedianNum, SignedZeroOrdering) {
  const struct {
    decltype(&rj_arith_v_med3_num_f32_w32) call;
    uint32_t negative_zero, negative_one;
  } cases[] = {{rj_arith_v_med3_num_f16_w32, 0x8000U, 0xbc00U},
               {rj_arith_v_med3_num_f32_w32, 0x80000000U, 0xbf800000U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &sample : cases) {
      rocjitsu::arith::test::Registers<4, 32> r;
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = sample.negative_zero;
        r.v[1][lane] = sample.negative_one;
        r.v[2][lane] = 0;
        r.v[3][lane] = 0;
      }
      ASSERT_EQ(sample.call(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0xffffffffU, 0, r.v + 3, r.v, r.v + 1,
                            r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(r.v[3][lane], sample.negative_zero);
      }
    }
  }
}
