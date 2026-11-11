// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP64 rounding to integral values and FRACT.

#include "fp/captured_gfx11_test.h"
#include "fp/loose_unary_f64.h"
#include "rj_arith/detail/rj_arith_fp.h"

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

// Captured gfx1201 FRACT witnesses distinguish subtraction rounding; overflow
// checks isolate the output scaling shared by integral operations.
TEST(FpRoundF64, DirectedRounding) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      Registers<4, 32> r;
      const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | rounding;
      write_fp(r.v, 0, 64, 0xbce254ae32fc7a4fULL);
      ASSERT_EQ(rj_arith_v_fract_f64_w32_gfx12(flags, 1, 0, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64),
                rounding == RJ_ARITH_FP_ROUND_DOWN || rounding == RJ_ARITH_FP_ROUND_ZERO
                    ? 0x3fefffffffffffedULL
                    : 0x3fefffffffffffeeULL);
      write_fp(r.v, 0, 64, 0xbff0000000000000ULL);
      ASSERT_EQ(rj_arith_v_fract_f64_w32_gfx12(flags, 1, 0, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), 0ULL);
      write_fp(r.v, 0, 64, 0x7fefffffffffffffULL);
      ASSERT_EQ(rj_arith_v_ceil_f64_w32_gfx12(flags, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64),
                rounding == RJ_ARITH_FP_ROUND_DOWN || rounding == RJ_ARITH_FP_ROUND_ZERO
                    ? 0x7fefffffffffffffULL
                    : 0x7ff0000000000000ULL);
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee): each operation returns a
// signaling NaN source unquieted, with ABS/NEG applied, also under directed rounding and OMOD.
const CapturedGfx11 ceil_f64_lanes[] = {
    {"E32Mode00", 0, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x5a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};
const CapturedGfx11 floor_f64_lanes[] = {
    {"E32Mode00", 0, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x5a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegAPayload", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff29526f4863633, 0, 0, 0xfff29526f4863633},
};
const CapturedGfx11 fract_f64_lanes[] = {
    {"E32Mode00", 0, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x5a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"AbsAPayload", RJ_ARITH_ALU_ABS_A, 0xf0, 0xfff563bc5abbaeb9, 0, 0, 0x7ff563bc5abbaeb9},
};
const CapturedGfx11 rndne_f64_lanes[] = {
    {"E32Mode00", 0, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x5a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};
const CapturedGfx11 trunc_f64_lanes[] = {
    {"E32Mode00", 0, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x5a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0xf0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x00, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};

} // namespace

// Compatibility reproduces captured gfx1100 FP64 rounding and FRACT lanes.
TEST(FpRoundF64Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11;
  check_captured_gfx11<32, 64>(rj_arith_v_ceil_f64_w32, ceil_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_ceil_f64_w64, ceil_f64_lanes);
  check_captured_gfx11<32, 64>(rj_arith_v_floor_f64_w32, floor_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_floor_f64_w64, floor_f64_lanes);
  check_captured_gfx11<32, 64>(rj_arith_v_fract_f64_w32, fract_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_fract_f64_w64, fract_f64_lanes);
  check_captured_gfx11<32, 64>(rj_arith_v_rndne_f64_w32, rndne_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_rndne_f64_w64, rndne_f64_lanes);
  check_captured_gfx11<32, 64>(rj_arith_v_trunc_f64_w32, trunc_f64_lanes);
  check_captured_gfx11<64, 64>(rj_arith_v_trunc_f64_w64, trunc_f64_lanes);
}
