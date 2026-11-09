// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for NaN-propagating minimum/maximum.

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

// Compatibility retains the first NaN through FP16 conversion and FP64 SIMD.
// Loose shares this implementation; these raw bits are not a Loose API guarantee.
TEST(FpMinimumMaximum, FirstNaN) {
  using namespace rocjitsu::arith::test;
  const struct {
    decltype(&rj_arith_v_minimum_f32_w32) call;
    int width;
    uint64_t a, b, expected;
  } cases[] = {{rj_arith_v_minimum_f16_w32, 16, 0xfc04, 0x7c01, 0xfe04},
               {rj_arith_v_maximum_f16_w32, 16, 0xfc04, 0x7c01, 0xfe04},
               {rj_arith_v_minimum_f32_w32, 32, 0xffc00000, 0x7f800001, 0xffc00000},
               {rj_arith_v_maximum_f32_w32, 32, 0xffc00000, 0x7f800001, 0xffc00000},
               {rj_arith_v_minimum_f64_w32, 64, 0xfff0000000000004ULL, 0x7ff0000000000001ULL,
                0xfff8000000000004ULL},
               {rj_arith_v_maximum_f64_w32, 64, 0xfff0000000000004ULL, 0x7ff0000000000001ULL,
                0xfff8000000000004ULL}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &sample : cases) {
        Registers<6, 32> r;
        write_fp(r.v, 0, sample.width, sample.a);
        write_fp(r.v + 2, 0, sample.width, sample.b);
        ASSERT_EQ(sample.call(cpu | semantics, 1, 0, r.v + 4, r.v, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 4, 0, sample.width), sample.expected);
      }
    }
  }
}

// Selection preserves its chosen subnormal when only output flushing is enabled.
TEST(FpMinimumF64, OutputFlushing) {
  rocjitsu::arith::test::Registers<6, 32> input;
  for (int lane = 0; lane < 32; ++lane) {
    input.v[0][lane] = 1;
    input.v[1][lane] = 0;
    input.v[2][lane] = 0;
    input.v[3][lane] = 0x3ff00000U;
  }
  const uint32_t *a[] = {input.v[0], input.v[1]};
  const uint32_t *b[] = {input.v[2], input.v[3]};
  uint32_t *d[] = {input.v[4], input.v[5]};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (auto call : {rj_arith_v_min_num_f64_w32, rj_arith_v_minimum_f64_w32}) {
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, ~0U, 0, d,
                     a, b, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(d[0][lane], 1U);
        EXPECT_EQ(d[1][lane], 0U);
      }
    }
  }
}
