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
