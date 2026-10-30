// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fused arithmetic across FP widths, aliases, FP-status reporting, and guest controls.

#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/elementwise.h"
#include "fp/fma_dpp_test.h"
#include "fp/fma_reference.h"
#include "fp/fma_simd_test.h"
#include "fp/fma_test.h"
#include "fp/half_fma_test.h"
#include "fp/reference.h"
#include "rj_arith/rj_arith.h"
#include <bit>
#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// FP exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
using rocjitsu::arith::test::StatusCase;

const StatusCase status_cases_f16[] = {
    {0x3c00U, 0x3c00U, 0x0U, 0},
    {0x0U, 0x7c00U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7bffU, 0x4000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3c01U, 0x3c01U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1U, 0x3800U, 0x0U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
};

} // namespace

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w32_gfx12, /*gfx12=*/true, status_cases_f16);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w32, /*gfx12=*/false, status_cases_f16);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w64_gfx12, /*gfx12=*/true, status_cases_f16);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w64, /*gfx12=*/false, status_cases_f16);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 32>(rj_arith_v_fma_f32_w32_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 32>(rj_arith_v_fma_f32_w32, /*gfx12=*/false);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 64>(rj_arith_v_fma_f32_w64_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 64>(rj_arith_v_fma_f32_w64, /*gfx12=*/false);
}

TEST(FpFmaF16W32Gfx12, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<32, true>(rj_arith_v_fma_f16_w32_gfx12);
}
TEST(FpFmaF32W32Gfx11, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<32>(rj_arith_v_fma_f32_w32);
}
TEST(FpFmaF16W32Gfx11, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<32, true>(rj_arith_v_fma_f16_w32);
}
TEST(FpFmaF32W32Gfx12, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<32>(rj_arith_v_fma_f32_w32_gfx12);
}
TEST(FpFmaF32W64Gfx12, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<64>(rj_arith_v_fma_f32_w64_gfx12);
}
TEST(FpFmaF16W64Gfx12, Dpp16) {
  rocjitsu::arith::test::check_fma_dpp16<64, true>(rj_arith_v_fma_f16_w64_gfx12);
}

// Halfway additions distinguish guest rounding, and a tiny product distinguishes
// both guest flushing controls. These witnesses use exactly representable inputs.
template <bool Half, typename Call> static void check_guest_controls(Call call) {
  using namespace rocjitsu::arith::test;
  constexpr uint32_t one = Half ? 0x3c00U : 0x3f800000U;
  constexpr uint32_t half_ulp = Half ? 0x1000U : 0x33800000U;
  for (uint64_t cpu : supported_cpu_levels()) {
    if constexpr (!Half) {
      // Existing rocjitsu FmaF32FlushesTinySignificandBeforePacking witness.
      Registers<4, 32> r;
      r.v[0][0] = r.v[1][0] = 1;
      r.v[2][0] = 0x007fffff;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                         RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                     1, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0U);
    }

    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      if constexpr (!Half) {
        // Tiny negative product promotes to minimum normal only after significand rounding.
        Registers<4, 32> boundary;
        boundary.v[0][0] = 0x80000001;
        boundary.v[1][0] = 1;
        boundary.v[2][0] = 0x00800000;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding |
                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                       1, 0, boundary.v + 3, boundary.v, boundary.v + 1, boundary.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(boundary.v[3][0],
                  rounding == 0 || rounding == RJ_ARITH_FP_ROUND_UP ? 0x00800000U : 0U);
      }
      Registers<4, 32> r;
      r.v[0][0] = r.v[1][0] = one;
      r.v[2][0] = half_ulp;
      r.v[3][0] = 0;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding, 1, 0,
                     r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], one + (rounding == RJ_ARITH_FP_ROUND_UP));
    }
    for (uint64_t flushing :
         {uint64_t(0), RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
      for (uint64_t semantics :
           {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
        Registers<4, 32> r;
        r.v[0][0] = 1;
        r.v[1][0] = one;
        r.v[2][0] = r.v[3][0] = 0;
        ASSERT_EQ(call(cpu | semantics | flushing, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0], flushing ? 0U : 1U);
      }
    }
  }
}
TEST(FpFmaF16W32Gfx12, GuestControls) { check_guest_controls<true>(rj_arith_v_fma_f16_w32_gfx12); }
TEST(FpFmaF32W32Gfx12, GuestControls) { check_guest_controls<false>(rj_arith_v_fma_f32_w32_gfx12); }
// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 32>(rj_arith_v_fma_f64_w32_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 32>(rj_arith_v_fma_f64_w32, /*gfx12=*/false);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 64>(rj_arith_v_fma_f64_w64_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 64>(rj_arith_v_fma_f64_w64, /*gfx12=*/false);
}

// gfx1201 WAVE_EXCP_FLAG_USER observations under MODE=0xf0 (RNE, preserve denormals).
// FP16_OVFL rows additionally set MODE bit 23 (MODE=0x008000f0).
// These cases distinguish tininess, NaN suppression, and finite saturation.
template <int Lanes, typename Call> static void check_fma16_status(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr Mask high_lane = Mask(1) << (Lanes - 1);
  const struct {
    uint16_t a, b, c, expected;
    uint32_t causes;
    uint64_t flags;
    uint32_t instruction_flags = 0;
  } cases[] = {
      {0x3c00, 0x3c00, 0, 0x3c00, 0, 0},
      {0x3c01, 0x3c01, 0, 0x3c02, RJ_ARITH_EXCEPTION_INEXACT, 0},
      {1, 0x3800, 0, 0,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       0},
      // gfx1201 VOP3 lane 31, EXEC=0x80000000, MODE=0x010000f0.
      {0x3c00, 0x3c00, 0x7c01, 0x7e01, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x3c00, 0x7c00, 0xfc00, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0, 0x7c00, 1, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},

      // Also observed by the reviewer on gfx1201 (R9700), with and without FP16_OVFL.
      {1, 0xb800, 0x0400, 0x0400,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       0},
      {1, 0xb400, 0x0400, 0x0400, RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_INEXACT, 0},
      {0x0400, 0x3bff, 0, 0x0400, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT, 0},
      {1, 0x3c00, 0, 1, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {1, 0x3c00, 0x7e00, 0x7e00, 0, 0},
      {1, 0x7c00, 0, 0x7c00, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {0x7c01, 0x3c00, 0, 0x7e01, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7e02, 0x7c01, 0, 0x7e02, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7c00, 0x3c00, 0xfc00, 0xfe00, RJ_ARITH_EXCEPTION_INVALID, 0},
      {0x7c00, 0x3c00, 0x7c00, 0x7c00, 0, 0},
      {1, 0x7c00, 0x7c00, 0x7c00, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0},
      {0x7bff, 0x4000, 0, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT,
       RJ_ARITH_FP16_OVFL},
      // gfx1201 Wave32, RNE, preserved denormals: OMOD changes the reported causes.
      {0x7800, 0x3c00, 0, 0x7c00, RJ_ARITH_EXCEPTION_OVERFLOW, 0, RJ_ARITH_ALU_OMOD_2},
      {0x7400, 0x3c00, 0, 0x7c00, RJ_ARITH_EXCEPTION_OVERFLOW, 0, RJ_ARITH_ALU_OMOD_4},
      {0x3c01, 0x3c01, 0, 0x4002, 0, 0, RJ_ARITH_ALU_OMOD_2},
      {0x0400, 0x3bff, 0, 0, 0, 0, RJ_ARITH_ALU_OMOD_2},
      {1, 0x3c00, 0x3c00, 0x4000, RJ_ARITH_EXCEPTION_INPUT_DENORM, 0, RJ_ARITH_ALU_OMOD_2},
  };
  constexpr uint32_t preserved = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      SCOPED_TRACE(testing::Message() << "cpu=" << cpu << " a=" << sample.a << " b=" << sample.b
                                      << " c=" << sample.c << " flags=" << sample.flags);
      Registers<4, Lanes> r;
      r.v[0][Lanes - 1] = sample.a;
      r.v[1][Lanes - 1] = sample.b;
      r.v[2][Lanes - 1] = sample.c;
      uint32_t status = preserved;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | sample.flags, high_lane,
                     sample.instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(uint16_t(r.v[3][Lanes - 1]), sample.expected);
      EXPECT_EQ(status, preserved | sample.causes);
    }
    for (uint32_t instruction_flags : {0U, RJ_ARITH_ALU_OMOD_2}) {
      // Each exceptional lane contributes independently, and masked lanes contribute nothing.
      for (Mask exec_mask : {Mask(0), Mask(1), high_lane, high_lane | Mask(1)}) {
        SCOPED_TRACE(testing::Message() << "cpu=" << cpu << " exec_mask=" << exec_mask);
        Registers<4, Lanes> r;
        r.v[0][0] = 0;
        r.v[1][0] = 0x7c00;
        r.v[2][0] = 0;
        r.v[0][Lanes - 1] = 0x7bff;
        r.v[1][Lanes - 1] = 0x4000;
        r.v[2][Lanes - 1] = 0;
        uint32_t status = preserved;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, exec_mask, instruction_flags,
                       r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                  RJ_ARITH_SUCCESS);
        uint32_t expected = preserved;
        if (exec_mask & 1) {
          expected |= RJ_ARITH_EXCEPTION_INVALID;
        }
        if (exec_mask & high_lane) {
          expected |=
              RJ_ARITH_EXCEPTION_OVERFLOW | (instruction_flags ? 0 : RJ_ARITH_EXCEPTION_INEXACT);
        }
        EXPECT_EQ(status, expected);
      }
    }
  }
}

// The architectural causes accumulate identically across both EXEC widths.
TEST(FpFmaF16W32Gfx12, StatusBoundariesAndExec) {
  check_fma16_status<32>(rj_arith_v_fma_f16_w32_gfx12);
}
TEST(FpFmaF16W64Gfx12, StatusBoundariesAndExec) {
  check_fma16_status<64>(rj_arith_v_fma_f16_w64_gfx12);
}

// Directed numeric regression cases shared across wave widths.
template <int Lanes, typename Call> void check_fma16_rounding_boundaries(Call call) {
  using namespace rocjitsu::arith::test;
  struct Case {
    uint16_t a, b, c, expected, saturated;
    uint32_t expected_status;
  };
  const Case cases[] = {
      // Overflow is classified after rounding, not by comparison with 65504.
      {0x4300, 0x7492, 0, 0x7c00, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x88e4, 0x0a8b, 0x81ff, 0x81ff, 0x81ff,
       RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x32f9, 0xb0b7, 0x3780, 0x36fd, 0x36fd, RJ_ARITH_EXCEPTION_INEXACT},
      {0x4300, 0x7492, 0x9400, 0x7bff, 0x7bff, RJ_ARITH_EXCEPTION_INEXACT},
      {0xc300, 0x7492, 0x1400, 0xfbff, 0xfbff, RJ_ARITH_EXCEPTION_INEXACT},
      {0x7bff, 0x4000, 0, 0x7c00, 0x7bff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0xfbff, 0x4000, 0, 0xfc00, 0xfbff, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
      {0x7c00, 0x3c00, 0, 0x7c00, 0x7c00, 0},
      {0xfc00, 0x3c00, 0, 0xfc00, 0xfc00, 0},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (const auto &sample : cases) {
          SCOPED_TRACE(testing::Message()
                       << "cpu=" << cpu << " semantics=" << semantics << " fp_flags=" << fp_flags
                       << " a=" << sample.a << " b=" << sample.b << " c=" << sample.c);
          Registers<4, Lanes> r;
          r.v[0][0] = sample.a;
          r.v[1][0] = sample.b;
          r.v[2][0] = sample.c;
          uint32_t status = 0;
          ASSERT_EQ(call(cpu | semantics | fp_flags, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(uint16_t(r.v[3][0]), fp_flags ? sample.saturated : sample.expected);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            EXPECT_EQ(status, sample.expected_status);
          }
        }
      }
    }
  }
}
TEST(FpFmaF16W32Gfx12, RoundingBoundaries) {
  check_fma16_rounding_boundaries<32>(rj_arith_v_fma_f16_w32_gfx12);
}
TEST(FpFmaF16W64Gfx12, RoundingBoundaries) {
  check_fma16_rounding_boundaries<64>(rj_arith_v_fma_f16_w64_gfx12);
}

// Directed numeric regression cases shared across wave widths.
template <int Lanes, bool Rdna3 = false, typename Call>
void check_fma16_exact_encodings(Call call) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint16_t a, b, c, expected;
    uint64_t flags;
  } cases[] = {
      // MODE.IEEE=0: gfx11 preserves the selected signaling NaN; gfx12 quiets it.
      {0x3c00, 0x7c05, 0x7c07, Rdna3 ? 0x7c05U : 0x7e05U, 0}, // B precedes C.
      {0x3c00, 0x3c00, 0x7c07, Rdna3 ? 0x7c07U : 0x7e07U, 0}, // C selected.

      {0x7e01, 0x7c03, 0, 0x7e01, 0},      // Earlier qNaN beats later sNaN.
      {0x3c00, 0x7e05, 0x7d07, 0x7e05, 0}, // B NaN beats C sNaN.
      {0x7c00, 0x3c00, 0xfc00, 0xfe00, 0}, // Opposite infinities.
      {0x8000, 0x3c00, 0x8000, 0x8000, 0}, // Matching negative zeros.
      {0x8001, 0x3800, 0, 0x8000, 0},      // Negative halfway underflow.
      {0x0001, 0x3800, 0, 0x0000, 0},      // Tie rounds to even zero.
      {0x0003, 0x3800, 0, 0x0002, 0},      // Tie rounds up to even subnormal.

      {0x4782, 0xda00, 0x03e1, 0xe5a1, 0},               // Rounding witness.
      {0, 0x7c00, 0x7e01, 0xfe00, 0},                    // Invalid product precedes C NaN.
      {0x7c03, 0x7e01, 0, Rdna3 ? 0x7c03U : 0x7e03U, 0}, // Select the first source NaN.
      {0x7bff, 0x7bff, 0, 0x7c00, 0},                    // Finite overflow.
      {0x7bff, 0x7bff, 0, 0x7bff, RJ_ARITH_FP16_OVFL},   // Saturate finite overflow.
      {0x3c00, 0x3c00, 0xbc00, 0, 0},                    // Exact cancellation.
      {1, 0x3c00, 0, 1, 0},                              // Exact subnormal result.
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      if (Rdna3 && semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
        continue;
      }
      for (const auto &sample : cases) {
        SCOPED_TRACE(testing::Message()
                     << "cpu=" << cpu << " semantics=" << semantics << " a=" << sample.a
                     << " b=" << sample.b << " c=" << sample.c << " flags=" << sample.flags);
        Registers<4, Lanes> r;
        r.v[0][0] = sample.a;
        r.v[1][0] = sample.b;
        r.v[2][0] = sample.c;
        uint32_t status = 0;
        ASSERT_EQ(
            call(cpu | semantics | sample.flags, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(uint16_t(r.v[3][0]), sample.expected);
      }
    }
  }
}
TEST(FpFmaF16W32Gfx12, ExactEncodings) {
  check_fma16_exact_encodings<32>(rj_arith_v_fma_f16_w32_gfx12);
}
TEST(FpFmaF16W64Gfx12, ExactEncodings) {
  check_fma16_exact_encodings<64>(rj_arith_v_fma_f16_w64_gfx12);
}
TEST(FpFmaF16W32Gfx11, ExactEncodings) {
  check_fma16_exact_encodings<32, true>(rj_arith_v_fma_f16_w32);
}
TEST(FpFmaF16W64Gfx11, ExactEncodings) {
  check_fma16_exact_encodings<64, true>(rj_arith_v_fma_f16_w64);
}

// Directed numeric regression cases shared across wave widths.
template <int Lanes, bool Rdna3 = false, typename Call>
void check_fma32_exact_encodings(Call call) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint32_t a, b, c, expected;
  } cases[] = {
      // Boundary encodings confirmed by the gfx1201 probes in PR #13939's review.
      {0x3f7fffff, 0x3f800000, 0x33000000, 0x3f800000}, // Carry into next binade.
      {1, 0x3f400000, 0, 1},                            // Above underflow midpoint.
      {1, 0x3f000000, 0, 0},                            // Tie to even zero.
      {0x7f800000, 0x3f800000, 0xff800000, 0xffc00000}, // Opposite infinities.
      {0x3f800000, 0x7f800001, 0x7fc00002, Rdna3 ? 0x7f800001U : 0x7fc00001U}, // B NaN precedes C.
      {0x7f7fffff, 0x40000000, 0, 0x7f800000},                                 // Finite overflow.
      {0x007fffff, 0x3f800001, 0, 0x00800000},                                 // Promote to normal.
      {0x3f800001, 0x3f7ffffe, 0xbf800000, 0xa8800000}, // Fused cancellation.
      {0, 0x7f800000, 0x7fc12345, 0xffc00000},          // Invalid product beats C NaN.
      {0x7f800003, 0x7fc00005, 0,
       Rdna3 ? 0x7f800003U : 0x7fc00003U},              // Select the first source NaN.
      {0x80000000, 0x3f800000, 0x80000000, 0x80000000}, // Matching negative zeros.
      {3, 0x3f000000, 0, 2},                            // Subnormal halfway rounds to even.
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                               RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t fp_flags : {uint64_t{0}, RJ_ARITH_FP16_OVFL}) {
        for (const auto &sample : cases) {
          SCOPED_TRACE(testing::Message()
                       << "cpu=" << cpu << " semantics=" << semantics << " fp_flags=" << fp_flags
                       << " a=" << sample.a << " b=" << sample.b << " c=" << sample.c);
          Registers<4, Lanes> r;
          r.v[0][0] = sample.a;
          r.v[1][0] = sample.b;
          r.v[2][0] = sample.c;
          ASSERT_EQ(call(cpu | semantics | fp_flags, 1, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(r.v[3][0], sample.expected);
        }
      }
    }
  }
}
TEST(FpFmaF32W32Gfx12, ExactEncodings) {
  check_fma32_exact_encodings<32>(rj_arith_v_fma_f32_w32_gfx12);
}
TEST(FpFmaF32W64Gfx12, ExactEncodings) {
  check_fma32_exact_encodings<64>(rj_arith_v_fma_f32_w64_gfx12);
}
TEST(FpFmaF32W32Gfx11, ExactEncodings) {
  check_fma32_exact_encodings<32, true>(rj_arith_v_fma_f32_w32);
}
TEST(FpFmaF32W64Gfx11, ExactEncodings) {
  check_fma32_exact_encodings<64, true>(rj_arith_v_fma_f32_w64);
}

// Literal OMOD boundaries: directed overflow, signed zero, and FP16 saturation.
// gfx11 shares this model when output flushing enables OMOD and IEEE is clear.
namespace {

struct OmodCase {
  uint32_t a, b, c, expected;
  uint32_t instruction_flags;
  uint64_t guest;
};

// Check raw encodings, including the sign of zero, in both wave sizes and CPU paths.
template <int Lanes, bool Half, typename Call, size_t N>
void check_omod_rounding(Call call, bool gfx12, const OmodCase (&cases)[N]) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  for (uint64_t cpu : supported_cpu_levels()) {
    constexpr uint32_t one = Half ? 0x3c00U : 0x3f800000U;
    constexpr uint32_t sign = Half ? 0x8000U : 0x80000000U;
    // Cancellation takes the guest rounding sign; matching zero terms keep their sign.
    for (uint64_t rounding :
         {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_ROUND_ZERO}) {
      const uint32_t zero_cases[][4] = {
          {one, one, one | sign, rounding == RJ_ARITH_FP_ROUND_DOWN ? sign : 0U},
          {0, one, 0, 0},
          {sign, one, sign, sign}};
      for (const auto &sample : zero_cases) {
        Registers<4, Lanes> regs;
        regs.v[0][Lanes - 1] = sample[0];
        regs.v[1][Lanes - 1] = sample[1];
        regs.v[2][Lanes - 1] = sample[2];
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | rounding,
                       Mask(1) << (Lanes - 1), 0, regs.v + 3, regs.v, regs.v + 1, regs.v + 2,
                       nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(regs.v[3][Lanes - 1] & (Half ? 0xffffU : 0xffffffffU), sample[3]);
      }
    }
    // Unlike gfx11, gfx12 retains OMOD under IEEE mode, with either output-denormal setting.
    if (gfx12) {
      for (uint64_t flushing : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        Registers<4, Lanes> regs;
        regs.v[0][0] = regs.v[1][0] = one;
        regs.v[2][0] = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS |
                           RJ_ARITH_FP_IEEE_MODE | flushing,
                       1, RJ_ARITH_ALU_OMOD_2, regs.v + 3, regs.v, regs.v + 1, regs.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(regs.v[3][0] & (Half ? 0xffffU : 0xffffffffU), Half ? 0x4000U : 0x40000000U);
      }
    }
    for (const auto &sample : cases) {
      Registers<4, Lanes> regs;
      regs.v[0][Lanes - 1] = sample.a;
      regs.v[1][Lanes - 1] = sample.b;
      regs.v[2][Lanes - 1] = sample.c;
      uint64_t guest = sample.guest | (gfx12 ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
      uint64_t flags = cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | guest;
      ASSERT_EQ(call(flags, Mask(1) << (Lanes - 1), sample.instruction_flags, regs.v + 3, regs.v,
                     regs.v + 1, regs.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(regs.v[3][Lanes - 1] & (Half ? 0xffffU : 0xffffffffU), sample.expected);
      if constexpr (Half) {
        EXPECT_EQ(half_fma_reference::evaluate(sample.a, sample.b, sample.c,
                                               sample.instruction_flags, guest & RJ_ARITH_FP16_OVFL,
                                               guest, !gfx12),
                  sample.expected);
      }
    }
  }
}

} // namespace

TEST(FpFmaF16, GuestOmodRounding) {
  const OmodCase cases[] = {
      {0x7800, 0x3c00, 0, 0x7bff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_DOWN},
      {0x7800, 0x3c00, 0, 0x7bff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_ZERO},
      {0xf800, 0x3c00, 0, 0xfbff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_UP},
      {0xf800, 0x3c00, 0, 0xfbff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_ZERO},
      {0x7400, 0x3c00, 0, 0x7bff, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_FP_ROUND_ZERO},
      {0x0001, 0x3900, 0x03ff, 0x0800, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_UP},
      {0x07ff, 0x3c00, 0, 0, RJ_ARITH_ALU_OMOD_HALF, 0},
      {0x87ff, 0x3c00, 0, 0x8000, RJ_ARITH_ALU_OMOD_HALF, 0},
      {0x8001, 0x3c00, 0, 0, RJ_ARITH_ALU_OMOD_2, 0},
      {0x7800, 0x3c00, 0, 0x7bff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP16_OVFL},
      {0x7bff, 0x4000, 0, 0x77ff, RJ_ARITH_ALU_OMOD_HALF, RJ_ARITH_FP16_OVFL},
      {0x7bff, 0x4000, 0, 0x7c00, RJ_ARITH_ALU_OMOD_HALF, 0},
  };
  check_omod_rounding<32, true>(rj_arith_v_fma_f16_w32_gfx12, true, cases);
  check_omod_rounding<64, true>(rj_arith_v_fma_f16_w64_gfx12, true, cases);
  check_omod_rounding<32, true>(rj_arith_v_fma_f16_w32, false, cases);
  check_omod_rounding<64, true>(rj_arith_v_fma_f16_w64, false, cases);
}

TEST(FpFmaF32, GuestOmodRounding) {
  const OmodCase cases[] = {
      {0x7f000000, 0x3f800000, 0, 0x7f7fffff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_ZERO},
      {0xff000000, 0x3f800000, 0, 0xff7fffff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_ZERO},
      {0x7f000000, 0x3f800000, 0, 0x7f7fffff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_DOWN},
      {0xff000000, 0x3f800000, 0, 0xff7fffff, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_UP},
      {0x7f000000, 0x3f800000, 0, 0x7f800000, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_UP},
      {0xff000000, 0x3f800000, 0, 0xff800000, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_FP_ROUND_DOWN},
      {0x00ffffff, 0x3f800000, 0, 0, RJ_ARITH_ALU_OMOD_HALF, 0},
      {0x80ffffff, 0x3f800000, 0, 0x80000000, RJ_ARITH_ALU_OMOD_HALF, 0},
      {0x80000001, 0x3f800000, 0, 0, RJ_ARITH_ALU_OMOD_2, 0},
  };
  check_omod_rounding<32, false>(rj_arith_v_fma_f32_w32_gfx12, true, cases);
  check_omod_rounding<64, false>(rj_arith_v_fma_f32_w64_gfx12, true, cases);
  check_omod_rounding<32, false>(rj_arith_v_fma_f32_w32, false, cases);
  check_omod_rounding<64, false>(rj_arith_v_fma_f32_w64, false, cases);
}

// Invalid routing and selectors reject before operand access, even with empty EXEC.
template <bool Half, typename Call> static void check_fma_invalid_controls(Call call) {
  using namespace rocjitsu::arith::test;
  uint64_t invalid[] = {RJ_ARITH_DPP16 | (0x100ULL << RJ_ARITH_DPP_CTRL_SHIFT),
                        RJ_ARITH_DPP16 | (0x110ULL << RJ_ARITH_DPP_CTRL_SHIFT),
                        RJ_ARITH_DPP16 | (0x130ULL << RJ_ARITH_DPP_CTRL_SHIFT),
                        RJ_ARITH_DPP16 | (0x142ULL << RJ_ARITH_DPP_CTRL_SHIFT),
                        RJ_ARITH_DPP8 | RJ_ARITH_DPP16,
                        RJ_ARITH_DPP8 | RJ_ARITH_DPP_BOUND_CTRL,
                        Half ? (1ULL << 31) : uint64_t(RJ_ARITH_ALU_HIGH_A),
                        Half ? (1ULL << 31) : uint64_t(RJ_ARITH_ALU_HIGH_D)};
  for (uint64_t instruction_flags : invalid) {
    for (uint32_t exec_mask : {0U, ~0U}) {
      Registers<4, 32> input;
      auto r = input;
      uint32_t status = 0x80000000U;
      uint64_t flags = RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
      EXPECT_EQ(call(flags, exec_mask, instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                RJ_ARITH_ERROR_INVALID_FLAGS);
      r.expect_preserved(input, 0, 0);
      EXPECT_EQ(
          call(flags, exec_mask, instruction_flags, nullptr, nullptr, nullptr, nullptr, &status),
          RJ_ARITH_ERROR_INVALID_FLAGS);
      EXPECT_EQ(status, 0x80000000U);
    }
  }
  for (uint64_t invalid_flags : {uint64_t(1) << 63, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16,
                                 RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16}) {
    EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | invalid_flags, 0, 0, nullptr, nullptr,
                   nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  EXPECT_EQ(call(RJ_ARITH_FP_DX10_CLAMP_OFF, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(FpFmaF16W32Gfx12, InvalidControls) {
  check_fma_invalid_controls<true>(rj_arith_v_fma_f16_w32_gfx12);
}
TEST(FpFmaF32W32Gfx12, InvalidControls) {
  check_fma_invalid_controls<false>(rj_arith_v_fma_f32_w32_gfx12);
}

// gfx11 IEEE gates OMOD and NaN quieting; DX10 controls NaN clamping.
// Literal expectations cover each encoding without relying on the arithmetic oracle.
template <bool Half, int Lanes, typename Call> static void check_gfx11_controls(Call call) {
  using namespace rocjitsu::arith::test;
  constexpr uint32_t one = Half ? 0x3c00U : 0x3f800000U;
  constexpr uint32_t snan = Half ? 0x7c01U : 0x7f800001U;
  constexpr uint32_t quiet = Half ? 0x0200U : 0x00400000U;
  struct Case {
    uint64_t flags;
    uint32_t instruction_flags, a, expected;
  };
  const Case cases[] = {
      {RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_2, one, Half ? 0x4000U : 0x40000000U},
      {RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_4, one, Half ? 0x4400U : 0x40800000U},
      {RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, RJ_ARITH_ALU_OMOD_HALF, one,
       Half ? 0x3800U : 0x3f000000U},
      {RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_ALU_OMOD_2, one, one},
      {0, 0, snan, snan},
      {RJ_ARITH_FP_IEEE_MODE, 0, snan, snan | quiet},
      {0, RJ_ARITH_ALU_CLAMP, snan, 0},
      {RJ_ARITH_FP_DX10_CLAMP_OFF, RJ_ARITH_ALU_CLAMP, snan, snan}};
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &test : cases) {
      SCOPED_TRACE(testing::Message() << cpu << '/' << test.flags << '/' << test.instruction_flags);
      Registers<4, Lanes> r;
      r.v[0][Lanes - 1] = test.a;
      r.v[1][Lanes - 1] = one;
      r.v[2][Lanes - 1] = r.v[3][Lanes - 1] = 0;
      using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | test.flags,
                     Mask(1) << (Lanes - 1), test.instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2,
                     nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][Lanes - 1], test.expected);
    }
  }
}
TEST(FpFmaF16W64Gfx11, GuestControls) { check_gfx11_controls<true, 64>(rj_arith_v_fma_f16_w64); }
TEST(FpFmaF32W32Gfx11, GuestControls) { check_gfx11_controls<false, 32>(rj_arith_v_fma_f32_w32); }

// Loose and exact share DX10's NaN-preserving clamp policy.
TEST(FpFmaF32W32Gfx11, LooseNanClamp) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t guest : {uint64_t(0), RJ_ARITH_FP_DX10_CLAMP_OFF}) {
      Registers<4, 32> r;
      r.v[0][0] = 0x7fc12345U;
      r.v[1][0] = 0x3f800000U;
      r.v[2][0] = 0;
      ASSERT_EQ(rj_arith_v_fma_f32_w32(cpu | RJ_ARITH_SEMANTICS_LOOSE | guest, 1,
                                       RJ_ARITH_ALU_CLAMP, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      if (guest) {
        EXPECT_GT(r.v[3][0] & 0x7fffffffU, 0x7f800000U);
      } else {
        EXPECT_EQ(r.v[3][0], 0U);
      }
    }
  }
}

// Tiny loose results need raw-bit checks: fuzzy tolerances also accept minimum normal.
TEST(FpFmaF16, LooseTinyResults) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint32_t a, b, instruction_flags, expected;
  } cases[] = {{0x07ff, 0x3c00, RJ_ARITH_ALU_OMOD_HALF, 0},
               {0x87ff, 0x3c00, RJ_ARITH_ALU_OMOD_HALF, 0x8000},
               {0x0400, 0x3bff, RJ_ARITH_ALU_OMOD_2, 0},
               {0x0400, 0x3bff, 0, 0},
               {0x8400, 0x3bff, 0, 0x8000}};
  for (auto call : {rj_arith_v_fma_f16_w32, rj_arith_v_fma_f16_w32_gfx12}) {
    for (uint64_t cpu : supported_cpu_levels()) {
      for (const auto &sample : cases) {
        Registers<4, 32> r;
        r.v[0][0] = sample.a;
        r.v[1][0] = sample.b;
        r.v[2][0] = 0;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS, 1,
                       sample.instruction_flags, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0] & 0xffffU, sample.expected);
      }
    }
  }
}

// Exact literal witnesses for guest rounding, independent of host rounding controls.
TEST(FpFmaF64, GuestRounding) {
  using namespace rocjitsu::arith::test;
  const struct {
    double a, b, c;
    uint64_t expected[4];
  } cases[] = {
      {1,
       0x1p-53,
       1,
       {0x3ff0000000000000ULL, 0x3ff0000000000001ULL, 0x3ff0000000000000ULL,
        0x3ff0000000000000ULL}},
      {-1,
       0x1p-53,
       -1,
       {0xbff0000000000000ULL, 0xbff0000000000000ULL, 0xbff0000000000001ULL,
        0xbff0000000000000ULL}},
      {1, 1, -1, {0, 0, 0x8000000000000000ULL, 0}},
      {0x1p-1074, 0.5, 0, {0, 1, 0, 0}},
      {0x1.fffffffffffffp1023,
       2,
       0,
       {0x7ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7fefffffffffffffULL,
        0x7fefffffffffffffULL}},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      for (const auto &sample : cases) {
        Registers<8, 32> r;
        const double inputs[] = {sample.a, sample.b, sample.c};
        for (int operand = 0; operand < 3; ++operand) {
          const uint64_t bits = std::bit_cast<uint64_t>(inputs[operand]);
          r.v[2 * operand][31] = uint32_t(bits);
          r.v[2 * operand + 1][31] = uint32_t(bits >> 32);
        }
        ASSERT_EQ(rj_arith_v_fma_f64_w32_gfx12(
                      cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24),
                      0x80000000U, 0, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(uint64_t(r.v[6][31]) | (uint64_t(r.v[7][31]) << 32), sample.expected[rounding]);
      }
    }
  }
}
