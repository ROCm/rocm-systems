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
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t rounding :
           {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
        SCOPED_TRACE(::testing::Message() << cpu << '/' << semantics << '/' << rounding);
        Registers<4, 32> r;
        const uint64_t flags = cpu | semantics | rounding;
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
        ASSERT_EQ(
            rj_arith_v_ceil_f64_w32_gfx12(flags, 1, RJ_ARITH_ALU_OMOD_2, r.v + 2, r.v, nullptr),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 2, 0, 64),
                  rounding == RJ_ARITH_FP_ROUND_DOWN || rounding == RJ_ARITH_FP_ROUND_ZERO
                      ? 0x7fefffffffffffffULL
                      : 0x7ff0000000000000ULL);
      }
    }
  }
}

namespace {

using UnaryF64 = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                         uint32_t *);

// gfx1201 captures of the RDNA4 FP64 unary forms with denormals kept unless a case flushes
// inputs: NaNs are quieted with their payload, CLAMP maps them to +0, FRACT of a tiny negative
// input caps below one before OMOD, and input flushing applies before the operation.
struct UnaryF64Case {
  UnaryF64 call;
  uint64_t flags;
  uint32_t modifiers;
  uint64_t input, expected;
};

const UnaryF64Case unary_f64_cases[] = {
    {rj_arith_v_trunc_f64_w32_gfx12, 0, 0, 0x7ff0000000000001ULL, 0x7ff8000000000001ULL},
    {rj_arith_v_rndne_f64_w32_gfx12, 0, 0, 0x7ff4000012345678ULL, 0x7ffc000012345678ULL},
    {rj_arith_v_rndne_f64_w32_gfx12, 0, 0, 0xbfe0000000000000ULL, 0x8000000000000000ULL},
    {rj_arith_v_trunc_f64_w32_gfx12, 0, RJ_ARITH_ALU_CLAMP, 0x7ff8000000000000ULL, 0},
    {rj_arith_v_fract_f64_w32_gfx12, 0, 0, 0x7ff0000000000001ULL, 0x7ff8000000000001ULL},
    {rj_arith_v_fract_f64_w32_gfx12, 0, 0, 0x800fffffffffffffULL, 0x3fefffffffffffffULL},
    {rj_arith_v_fract_f64_w32_gfx12, RJ_ARITH_FP_ROUND_ZERO, 0, 0x800fffffffffffffULL,
     0x3fefffffffffffffULL},
    {rj_arith_v_fract_f64_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0, 0x800fffffffffffffULL,
     0},
    {rj_arith_v_fract_f64_w32_gfx12, 0, RJ_ARITH_ALU_OMOD_HALF, 0x8000000000000001ULL,
     0x3fdfffffffffffffULL},
    {rj_arith_v_frexp_mant_f64_w32_gfx12, 0, 0, 0x7ff4000012345678ULL, 0x7ffc000012345678ULL},
    {rj_arith_v_frexp_mant_f64_w32_gfx12, 0, 0, 0x800fffffffffffffULL, 0xbfeffffffffffffeULL},
    {rj_arith_v_frexp_mant_f64_w32_gfx12, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, 0,
     0x800fffffffffffffULL, 0x8000000000000000ULL},
};

} // namespace

// Pin the captured results in Compatibility and strict exact mode, so that a faster loose path
// cannot change what default-mode RDNA4 execution returns.
TEST(FpRoundF64Gfx12, ExactMatchesHardware) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : hardware_semantics()) {
      for (const UnaryF64Case &test : unary_f64_cases) {
        SCOPED_TRACE(::testing::Message()
                     << cpu << '/' << semantics << '/' << (&test - unary_f64_cases));
        Registers<4, 32> r;
        write_fp(r.v, 0, 64, test.input);
        ASSERT_EQ(test.call(cpu | semantics | RJ_ARITH_FP_IEEE_MODE | test.flags, 1, test.modifiers,
                            r.v + 2, r.v, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 2, 0, 64), test.expected);
      }
    }
  }
}

namespace {

using rocjitsu::arith::test::CapturedGfx11;

// gfx1100 lanes from the MODE.IEEE-clear corpus (hw1100-noieee): each operation returns a
// signaling NaN source unquieted, with ABS/NEG applied, also under directed rounding and OMOD.
const CapturedGfx11 ceil_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};
const CapturedGfx11 floor_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegAPayload", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff29526f4863633, 0, 0, 0xfff29526f4863633},
};
const CapturedGfx11 fract_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"AbsAPayload", RJ_ARITH_ALU_ABS_A, 0x1f0, 0xfff563bc5abbaeb9, 0, 0, 0x7ff563bc5abbaeb9},
};
const CapturedGfx11 rndne_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};
const CapturedGfx11 trunc_f64_lanes[] = {
    {"E32Mode00", 0, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"E32RoundDown", 0, 0x15a, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
    {"NegA", RJ_ARITH_ALU_NEG_A, 0x1f0, 0x7ff0000000000001, 0, 0, 0xfff0000000000001},
    {"Mul4", RJ_ARITH_ALU_OMOD_4, 0x100, 0x7ff0000000000001, 0, 0, 0x7ff0000000000001},
};

} // namespace

// Compatibility and strict exact reproduce captured gfx1100 FP64 rounding and FRACT lanes.
TEST(FpRoundF64Gfx11, CapturedSignalingNans) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_ceil_f64_w32, ceil_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_ceil_f64_w64, ceil_f64_lanes);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_floor_f64_w32, floor_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_floor_f64_w64, floor_f64_lanes);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_fract_f64_w32, fract_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_fract_f64_w64, fract_f64_lanes);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_rndne_f64_w32, rndne_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_rndne_f64_w64, rndne_f64_lanes);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_trunc_f64_w32, trunc_f64_lanes);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_trunc_f64_w64, trunc_f64_lanes);
}

namespace {

// gfx1100 (W7900, mkm2) lanes with MODE.IEEE clear and DX10_CLAMP set: hw1100-noieee
// (2026-09-25) for MODE 0x1f0/0x100 and the Track C hw1100-wide corpus (2026-11-18) for the
// directed-rounding words 0x1f5 (toward +inf) and 0x1fa (toward -inf). FRACT caps a tiny
// negative input below one under every rounding, so a kernel that rounds x - floor(x) to 1.0
// fails the directed rows; OMOD applies only when output denormals are flushed, so a kernel
// that scales with denormals kept fails the Mul2 rows; CLAMP maps NaNs to +0 with DX10_CLAMP.
const CapturedGfx11 fract_f64_rules[] = {
    {"TinyNegativeNearest", 0, 0x1f0, 0x8010000000000000, 0, 0, 0x3fefffffffffffff},
    {"TinyNegativeRoundUp", 0, 0x1f5, 0x8010000000000000, 0, 0, 0x3fefffffffffffff},
    {"TinyNegativeRoundDown", 0, 0x1fa, 0x8010000000000000, 0, 0, 0x3fefffffffffffff},
    {"MinusHalf", 0, 0x1f0, 0xbfe0000000000000, 0, 0, 0x3fe0000000000000},
    {"Mul2FlushedDenormal", RJ_ARITH_ALU_OMOD_2, 0x100, 0x0000000000000001, 0, 0, 0},
    {"Mul2FlushedNegativeDenormal", RJ_ARITH_ALU_OMOD_2, 0x100, 0x800fffffffffffff, 0, 0, 0},
};
const CapturedGfx11 rndne_f64_rules[] = {
    {"ClampQuietNan", RJ_ARITH_ALU_CLAMP, 0x1f0, 0x7ff8000000000000, 0, 0, 0},
    {"ClampSignalingNan", RJ_ARITH_ALU_CLAMP, 0x1f0, 0x7ff0000000000001, 0, 0, 0},
    {"ClampQuietNanFlush", RJ_ARITH_ALU_CLAMP, 0x100, 0x7ff8000000000000, 0, 0, 0},
};
const CapturedGfx11 trunc_f64_rules[] = {
    {"SignalingPayloadKept", 0, 0x1f0, 0x7ff4000012345678, 0, 0, 0x7ff4000012345678},
};
// The gfx1100 IEEE-set corpus (hw2-gfx1100, 2026-10-06) quiets the same signaling NaN.
const CapturedGfx11 trunc_f64_ieee_rules[] = {
    {"SignalingQuietedIeee", 0, 0x3f0, 0x7ff0000000000001, 0, 0, 0x7ff8000000000001},
};

} // namespace

// Compatibility and strict exact reproduce the captured RDNA3 rounding, OMOD and CLAMP rules.
TEST(FpRoundF64Gfx11, CapturedRoundingOmodAndClampRules) {
  using rocjitsu::arith::test::check_captured_gfx11_exact;
  check_captured_gfx11_exact<32, 64>(rj_arith_v_fract_f64_w32, fract_f64_rules);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_fract_f64_w64, fract_f64_rules);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_rndne_f64_w32, rndne_f64_rules);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_rndne_f64_w64, rndne_f64_rules);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_trunc_f64_w32, trunc_f64_rules);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_trunc_f64_w64, trunc_f64_rules);
  check_captured_gfx11_exact<32, 64>(rj_arith_v_trunc_f64_w32, trunc_f64_ieee_rules);
  check_captured_gfx11_exact<64, 64>(rj_arith_v_trunc_f64_w64, trunc_f64_ieee_rules);
}

// Check inherited legacy arithmetic, aliasing, EXEC and unsupported strict modes.
TEST(FpIntegralF64Legacy, Arithmetic) {
  const auto reference = [](int op, double x) {
    switch (op) {
    case 0:
      return std::trunc(x);
    case 1:
      return std::ceil(x);
    case 2:
      return std::nearbyint(x);
    default:
      return std::floor(x);
    }
  };
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_trunc_f64_w32_gfx908, [&](double x) { return reference(0, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_ceil_f64_w32_gfx908, [&](double x) { return reference(1, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_rndne_f64_w32_gfx908, [&](double x) { return reference(2, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<32>(
      rj_arith_v_floor_f64_w32_gfx908, [&](double x) { return reference(3, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_trunc_f64_w64_gfx908, [&](double x) { return reference(0, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_ceil_f64_w64_gfx908, [&](double x) { return reference(1, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_rndne_f64_w64_gfx908, [&](double x) { return reference(2, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
  rocjitsu::arith::test::check_loose_unary_f64<64>(
      rj_arith_v_floor_f64_w64_gfx908, [&](double x) { return reference(3, x); }, false,
      /*check_omod=*/true, /*accepts_exact=*/false);
}

// Legacy NaN quieting is unconditional, unlike gfx11 with guest IEEE clear.
TEST(FpIntegralF64Legacy, SignalingNan) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_trunc_f64_w64_gfx908, rj_arith_v_ceil_f64_w64_gfx908,
                      rj_arith_v_rndne_f64_w64_gfx908, rj_arith_v_floor_f64_w64_gfx908}) {
      Registers<4, 64> r;
      write_fp(r.v, 0, 64, 0xfff0123456789abcULL);
      uint32_t status = 0x100;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, r.v + 2, r.v, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(read_fp(r.v + 2, 0, 64), 0xfff8123456789abcULL);
      EXPECT_EQ(status, 0x100U);
    }
  }
}
