// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 reciprocal, reciprocal square root, and square root.

#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rcp_f64_w32_gfx12,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rcp_f64_w32,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rcp_f64_w64_gfx12,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRcpF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rcp_f64_w64,
                                                   [](double x) { return 1.0 / x; });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rsq_f64_w32_gfx12,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_rsq_f64_w32,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rsq_f64_w64_gfx12,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpRsqF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_rsq_f64_w64,
                                                   [](double x) { return 1.0 / std::sqrt(x); });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_sqrt_f64_w32_gfx12, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(rj_arith_v_sqrt_f64_w32, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_sqrt_f64_w64_gfx12, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Check FP64 reciprocal, reciprocal square root, and square root.
TEST(FpSqrtF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(rj_arith_v_sqrt_f64_w64, [](double x) {
    double value = std::sqrt(x);
    return value;
  });
}

// Guest denormal controls affect unary arithmetic on both architectures.
TEST(FpUnaryF64, DenormalControls) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_ceil_f64_w32, rj_arith_v_ceil_f64_w32_gfx12}) {
      Registers<4, 32> r;
      write_fp(r.v, 0, 64, 1);
      uint32_t status = 0x80000000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 1, 0,
                     r.v + 2, r.v, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), 0U);
      EXPECT_EQ(status, 0x80000000U);
    }
    for (auto call : {rj_arith_v_rcp_f64_w32, rj_arith_v_rcp_f64_w32_gfx12}) {
      Registers<4, 32> r;
      write_fp(r.v, 0, 64, 0x7fe8000000000000ULL);
      uint32_t status = 0x80000000U;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1, 0,
                     r.v + 2, r.v, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), 0U);
      EXPECT_EQ(status, 0x80000000U);
    }
  }
}
