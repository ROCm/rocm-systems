// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 selection operations.

#include "fp/reference.h"
#include "fp/selection_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_min_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaxNumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_max_num_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, false); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMinimumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_minimum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, false, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W32, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 32, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w32,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Check FP selection results with NaNs, modifiers, EXEC masks, and aliases.
TEST(FpPkMaximumF16W64, Arithmetic) {
  rocjitsu::arith::test::check_selection<16, 64, 2, /*Packed=*/true>(
      rj_arith_v_pk_maximum_f16_w64,
      [](double x, double y) { return rocjitsu::arith::test::select_fp(x, y, true, true); });
}

// Selection preserves subnormal result bits even when arithmetic output flushing is enabled.
TEST(FpPackedSelection, PreservesDenormalResults) {
  const struct {
    decltype(&rj_arith_v_pk_min_f16_w32) call;
    uint32_t b;
  } cases[] = {
      {rj_arith_v_pk_min_f16_w32, 0x3c003c00U},     {rj_arith_v_pk_max_f16_w32, 0xbc00bc00U},
      {rj_arith_v_pk_min_num_f16_w32, 0x3c003c00U}, {rj_arith_v_pk_max_num_f16_w32, 0xbc00bc00U},
      {rj_arith_v_pk_minimum_f16_w32, 0x3c003c00U}, {rj_arith_v_pk_maximum_f16_w32, 0xbc00bc00U}};
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &sample : cases) {
        rocjitsu::arith::test::Registers<3, 32> r;
        for (int lane = 0; lane < 32; ++lane) {
          r.v[0][lane] = 0x80010001U;
          r.v[1][lane] = sample.b;
        }
        ASSERT_EQ(sample.call(cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 0xffffffffU, 0,
                              r.v + 2, r.v, r.v + 1, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          EXPECT_EQ(r.v[2][lane], 0x80010001U);
        }
      }
    }
  }
}

// gfx1201 observations support first-NaN selection and preserved positive
// subnormals under output flushing. Compatibility intentionally keeps this
// hardware-fidelity improvement over the interpreter's canonicalize/flush policy.
TEST(FpPackedSelection, FirstNaN) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (auto call : {rj_arith_v_pk_minimum_f16_w32, rj_arith_v_pk_maximum_f16_w32}) {
        Registers<3, 32> r;
        r.v[0][0] = 0x7c04fe01U;
        r.v[1][0] = 0xfe017c04U;
        ASSERT_EQ(call(cpu | semantics, 1, 0, r.v + 2, r.v, r.v + 1, nullptr), RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[2][0], 0x7e04fe01U);
      }
    }
  }
}
