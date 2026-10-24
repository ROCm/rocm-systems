// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for legacy three-input FP min/max/median.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(rj_arith_v_min3_f16_w32, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(rj_arith_v_min3_f16_w64, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(rj_arith_v_min3_f32_w32, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMin3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(rj_arith_v_min3_f32_w64, [](double x, double y,
                                                                                double z) {
    return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, false, false), z,
                                            false, false);
  });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_max3_f16_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_max3_f16_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_max3_f32_w32, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMax3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_max3_f32_w64, [](double x, double y, double z) {
        return rocjitsu::arith::test::select_fp(rocjitsu::arith::test::select_fp(x, y, true, false),
                                                z, true, false);
      });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 3>(
      rj_arith_v_med3_f16_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 3>(
      rj_arith_v_med3_f16_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 3>(
      rj_arith_v_med3_f32_w32,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMed3F32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 3>(
      rj_arith_v_med3_f32_w64,
      [](double x, double y, double z) { return rocjitsu::arith::test::median_fp(x, y, z); });
}

// Inherited rocjitsu RDNA3 policy distinguishes -0 from +0 when selecting a median.
TEST(FpLegacyMedian, SignedZeroOrdering) {
  const struct {
    decltype(&rj_arith_v_med3_f32_w32) call;
    uint32_t negative_zero, negative_one;
  } cases[] = {{rj_arith_v_med3_f16_w32, 0x8000U, 0xbc00U},
               {rj_arith_v_med3_f32_w32, 0x80000000U, 0xbf800000U}};
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

// The inherited legacy FP16 model rounds OMOD overflow using the guest mode.
TEST(FpTernaryF16, GuestOmodRounding) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding : {uint64_t(0), RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      Registers<4, 32> r;
      r.v[0][0] = r.v[1][0] = r.v[2][0] = 0x7bff;
      ASSERT_EQ(rj_arith_v_min3_f16_w32(
                    cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | rounding,
                    1, RJ_ARITH_ALU_OMOD_2, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 3, 0, 16), rounding ? 0x7bffU : 0x7c00U);
    }
  }
}
