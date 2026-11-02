// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 rounding to integral values and FRACT.

#include "fp/loose_unary_f64.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpCeilF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_ceil_f64_w32_gfx12,
      [](double x) {
        double value = std::ceil(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpCeilF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_ceil_f64_w32,
      [](double x) {
        double value = std::ceil(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpCeilF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_ceil_f64_w64_gfx12,
      [](double x) {
        double value = std::ceil(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpCeilF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_ceil_f64_w64,
      [](double x) {
        double value = std::ceil(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpFloorF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_floor_f64_w32_gfx12,
      [](double x) {
        double value = std::floor(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpFloorF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_floor_f64_w32,
      [](double x) {
        double value = std::floor(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpFloorF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_floor_f64_w64_gfx12,
      [](double x) {
        double value = std::floor(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpFloorF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_floor_f64_w64,
      [](double x) {
        double value = std::floor(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpTruncF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_trunc_f64_w32_gfx12,
      [](double x) {
        double value = std::trunc(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpTruncF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_trunc_f64_w32,
      [](double x) {
        double value = std::trunc(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpTruncF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_trunc_f64_w64_gfx12,
      [](double x) {
        double value = std::trunc(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpTruncF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_trunc_f64_w64,
      [](double x) {
        double value = std::trunc(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpRndneF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_rndne_f64_w32_gfx12,
      [](double x) {
        double value = std::nearbyint(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpRndneF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_rndne_f64_w32,
      [](double x) {
        double value = std::nearbyint(x);
        return value;
      },
      false);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpRndneF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_rndne_f64_w64_gfx12,
      [](double x) {
        double value = std::nearbyint(x);
        return value;
      },
      true);
}

// Check this FP64 rounding operation across CPU paths and output modifiers.
TEST(FpRndneF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_rndne_f64_w64,
      [](double x) {
        double value = std::nearbyint(x);
        return value;
      },
      false);
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_fract_f64_w32_gfx12,
      [](double x) {
        double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
        return value;
      },
      true);
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_fract_f64_w32,
      [](double x) {
        double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
        return value;
      },
      false);
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_fract_f64_w64_gfx12,
      [](double x) {
        double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
        return value;
      },
      true);
}

// Check the fractional part, including the upper bound below one.
TEST(FpFractF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_fract_f64_w64,
      [](double x) {
        double value = std::min(x - std::floor(x), 0x1.fffffffffffffp-1);
        return value;
      },
      false);
}

// Guest input flushing turns a positive subnormal into zero before CEIL.
TEST(FpCeilF64, InputDenormalControls) {
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
  }
}
