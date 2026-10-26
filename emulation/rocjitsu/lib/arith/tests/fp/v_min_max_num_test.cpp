// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for numeric-preferred FP min/max.

#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_min_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_min_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_min_num_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_min_num_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_min_num_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMinNumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_min_num_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2>(
      rj_arith_v_max_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2>(
      rj_arith_v_max_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF32W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 32, 2>(
      rj_arith_v_max_num_f32_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF32W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<32, 64, 2>(
      rj_arith_v_max_num_f32_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF64W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 32, 2>(
      rj_arith_v_max_num_f64_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpMaxNumF64W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<64, 64, 2>(
      rj_arith_v_max_num_f64_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Selection returns a source unchanged, so guest output flushing must not erase
// a selected subnormal. Input flushing stays off so the source reaches selection.
TEST(FpSelectionF64, OutputFlushPreservesSelectedSubnormal) {
  using namespace rocjitsu::arith::test;
  constexpr uint64_t subnormal = 1;
  const struct {
    decltype(&rj_arith_v_min_num_f64_w32) call;
    uint64_t other;
  } cases[] = {{rj_arith_v_min_num_f64_w32, 0x3ff0000000000000ULL},
               {rj_arith_v_max_num_f64_w32, 0xbff0000000000000ULL},
               {rj_arith_v_minimum_f64_w32, 0x3ff0000000000000ULL},
               {rj_arith_v_maximum_f64_w32, 0xbff0000000000000ULL}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      Registers<6, 32> r;
      uint32_t status = 0x80000000U;
      for (int lane = 0; lane < 32; ++lane) {
        write_fp(r.v, lane, 64, subnormal);
        write_fp(r.v + 2, lane, 64, sample.other);
        write_fp(r.v + 4, lane, 64, 0);
      }
      ASSERT_EQ(sample.call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                            0xffffffffU, 0, r.v + 4, r.v, r.v + 2, &status),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(read_fp(r.v + 4, lane, 64), subnormal);
      }
    }
  }
}

// The same contract for FP32 selection, which gates output flushing separately.
TEST(FpSelectionF32, OutputFlushPreservesSelectedSubnormal) {
  using namespace rocjitsu::arith::test;
  constexpr uint32_t subnormal = 1;
  const struct {
    decltype(&rj_arith_v_min_num_f32_w32) call;
    uint32_t other;
  } cases[] = {{rj_arith_v_min_num_f32_w32, 0x3f800000U},
               {rj_arith_v_max_num_f32_w32, 0xbf800000U},
               {rj_arith_v_minimum_f32_w32, 0x3f800000U},
               {rj_arith_v_maximum_f32_w32, 0xbf800000U}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      Registers<3, 32> r;
      uint32_t status = 0x80000000U;
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = subnormal;
        r.v[1][lane] = sample.other;
        r.v[2][lane] = 0;
      }
      ASSERT_EQ(sample.call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                            0xffffffffU, 0, r.v + 2, r.v, r.v + 1, &status),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(r.v[2][lane], subnormal);
      }
    }
  }
}
