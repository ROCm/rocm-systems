// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division fixup, exceptional operands, and FP-status expectations.

#include "fp/div_fixup_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Literal exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

const StatusCase status_cases_f16[] = {
    {0x3c00U, 0x3c00U, 0x3c00U, 0},
    {0x3c00U, 0x0U, 0x3c00U, RJ_ARITH_EXCEPTION_FLOAT_DIV0},
    {0x3c00U, 0x0U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x3c00U, 0x7c00U, 0x7c00U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7c00U, 0x3c00U, 0x3c00U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3c00U, 0x1U, 0x3c00U, RJ_ARITH_EXCEPTION_INPUT_DENORM},
};

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x3f800000U, 0x3f800000U, 0},
    {0x3f800000U, 0x0U, 0x3f800000U, RJ_ARITH_EXCEPTION_FLOAT_DIV0},
    {0x3f800000U, 0x0U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x3f800000U, 0x7f800000U, 0x7f800000U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7f800000U, 0x3f800000U, 0x3f800000U,
     RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800000U, 0x1U, 0x3f800000U, RJ_ARITH_EXCEPTION_INPUT_DENORM},
};

const StatusCase status_cases_f64[] = {
    {0x3ff0000000000000ULL, 0x3ff0000000000000ULL, 0x3ff0000000000000ULL, 0},
    {0x3ff0000000000000ULL, 0x0ULL, 0x3ff0000000000000ULL, RJ_ARITH_EXCEPTION_FLOAT_DIV0},
    {0x3ff0000000000000ULL, 0x0ULL, 0x0ULL, RJ_ARITH_EXCEPTION_INVALID},
    {0x3ff0000000000000ULL, 0x7ff0000000000000ULL, 0x7ff0000000000000ULL,
     RJ_ARITH_EXCEPTION_INVALID},
    {0x7ff0000000000000ULL, 0x3ff0000000000000ULL, 0x3ff0000000000000ULL,
     RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3ff0000000000000ULL, 0x1ULL, 0x3ff0000000000000ULL, RJ_ARITH_EXCEPTION_INPUT_DENORM},
};

} // namespace

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 32>(rj_arith_v_div_fixup_f16_w32_gfx12, /*gfx12=*/true,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 32>(rj_arith_v_div_fixup_f16_w32, /*gfx12=*/false,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 64>(rj_arith_v_div_fixup_f16_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 64>(rj_arith_v_div_fixup_f16_w64, /*gfx12=*/false,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 32>(rj_arith_v_div_fixup_f32_w32_gfx12, /*gfx12=*/true,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 32>(rj_arith_v_div_fixup_f32_w32, /*gfx12=*/false,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 64>(rj_arith_v_div_fixup_f32_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 64>(rj_arith_v_div_fixup_f32_w64, /*gfx12=*/false,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 32>(rj_arith_v_div_fixup_f64_w32_gfx12, /*gfx12=*/true,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 32>(rj_arith_v_div_fixup_f64_w32, /*gfx12=*/false,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 64>(rj_arith_v_div_fixup_f64_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 64>(rj_arith_v_div_fixup_f64_w64, /*gfx12=*/false,
                                                 status_cases_f64);
}

// gfx1100 review observations distinguish IEEE-gated propagation from quieting.
// The inherited compatibility classifier still reports INVALID in both modes.
template <int Width, int Lanes, typename Call> static void check_rdna3_ieee_nan(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t one = Width == 16   ? 0x3c00ULL
                           : Width == 32 ? 0x3f800000ULL
                                         : 0x3ff0000000000000ULL;
  constexpr uint64_t snan = Width == 16   ? 0x7c01ULL
                            : Width == 32 ? 0x7f800001ULL
                                          : 0x7ff0000000000001ULL;
  constexpr uint64_t quiet = Width == 16   ? 0x0200ULL
                             : Width == 32 ? 0x00400000ULL
                                           : 0x0008000000000000ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int operand : {1, 2}) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        Registers<8, Lanes> r;
        for (int i = 0; i < 3; ++i) {
          write_fp(r.v + 2 * i, Lanes - 1, Width, i == operand ? snan : one);
        }
        uint32_t status = 0x80000000U;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | ieee | RJ_ARITH_FP_DX10_CLAMP_OFF,
                       Mask(1) << (Lanes - 1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan | (ieee ? quiet : 0));
        EXPECT_EQ(status, 0x80000000U | (Width == 16 ? 0U : RJ_ARITH_EXCEPTION_INVALID));
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS | ieee,
                       Mask(1) << (Lanes - 1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan | (ieee ? quiet : 0));
        auto before = r;
        ASSERT_EQ(
            call(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | ieee | RJ_ARITH_FP_DX10_CLAMP_OFF,
                 Mask(1) << (Lanes - 1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, &status),
            RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        r.expect_preserved(before, 0, 0);
        EXPECT_EQ(status, 0x80000000U | (Width == 16 ? 0U : RJ_ARITH_EXCEPTION_INVALID));
      }
    }
  }
}

TEST(FpDivFixupGfx11, IeeeNanAndStatusQualification) {
  check_rdna3_ieee_nan<16, 32>(rj_arith_v_div_fixup_f16_w32);
  check_rdna3_ieee_nan<16, 64>(rj_arith_v_div_fixup_f16_w64);
  check_rdna3_ieee_nan<32, 32>(rj_arith_v_div_fixup_f32_w32);
  check_rdna3_ieee_nan<32, 64>(rj_arith_v_div_fixup_f32_w64);
  check_rdna3_ieee_nan<64, 32>(rj_arith_v_div_fixup_f64_w32);
  check_rdna3_ieee_nan<64, 64>(rj_arith_v_div_fixup_f64_w64);
}

// Disputed gfx1100 CLAMP/NaN/OMOD status rules cannot be promised by strict mode.
TEST(FpDivFixupGfx11, UnsupportedStatusBeforeOperandAccess) {
  for (auto call : {rj_arith_v_div_fixup_f32_w32, rj_arith_v_div_fixup_f64_w32}) {
    for (uint64_t fp : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
      for (uint64_t mode :
           {uint64_t(0), uint64_t(RJ_ARITH_ALU_CLAMP), uint64_t(RJ_ARITH_ALU_OMOD_2)}) {
        uint32_t status = 0x80000000U;
        EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP_DX10_CLAMP_OFF | fp, 1,
                       mode, nullptr, nullptr, nullptr, nullptr, &status),
                  RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        EXPECT_EQ(status, 0x80000000U);
      }
    }
  }
}

// Active RDNA3 FP16 OMOD resolves Compatibility to loose; strict rejection is transactional,
// including empty EXEC. IEEE clear keeps the selected sNaN under OMOD.
TEST(FpDivFixupF16, Rdna3OmodQualification) {
  using namespace rocjitsu::arith::test;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      Registers<4, 64> r;
      for (int lane = 0; lane < 64; ++lane) {
        r.v[0][lane] = r.v[2][lane] = 0x3c00;
        r.v[1][lane] = 0x7c01;
        r.v[3][lane] = 0x1234;
      }
      const uint64_t flags = cpu | semantics | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
      const int expected = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                               ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                           : semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                               ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS
                               : RJ_ARITH_SUCCESS;
      uint32_t status = 0x40;
      for (bool wide : {false, true}) {
        for (uint64_t exec_mask : {0ULL, 1ULL}) {
          r.v[3][0] = 0x1234;
          const int result =
              wide ? rj_arith_v_div_fixup_f16_w64(flags, exec_mask, RJ_ARITH_ALU_OMOD_2, r.v + 3,
                                                  r.v, r.v + 1, r.v + 2, &status)
                   : rj_arith_v_div_fixup_f16_w32(flags, uint32_t(exec_mask), RJ_ARITH_ALU_OMOD_2,
                                                  r.v + 3, r.v, r.v + 1, r.v + 2, &status);
          EXPECT_EQ(result, expected);
          const uint64_t invalid_flags[] = {RJ_ARITH_ALU_OMOD_2 | (1ULL << 30),
                                            RJ_ARITH_ALU_OMOD_2 | RJ_ARITH_DPP8 | RJ_ARITH_DPP16};
          for (uint64_t invalid : invalid_flags) {
            EXPECT_EQ(wide ? rj_arith_v_div_fixup_f16_w64(flags, exec_mask, invalid, r.v + 3, r.v,
                                                          r.v + 1, r.v + 2, &status)
                           : rj_arith_v_div_fixup_f16_w32(flags, uint32_t(exec_mask), invalid,
                                                          r.v + 3, r.v, r.v + 1, r.v + 2, &status),
                      RJ_ARITH_ERROR_INVALID_FLAGS);
          }
          EXPECT_EQ(r.v[3][0], expected || !exec_mask ? 0x1234u : 0x7c01u);
          EXPECT_EQ(status, 0x40u);
        }
      }
    }
  }
}

// gfx1201 captures: an OMOD overflow of a finite F16 quotient follows the guest rounding mode
// unless FP16_OVFL saturates it to the largest finite value. Directed modes give the largest
// finite value only when rounding toward zero for the result's sign.
TEST(FpDivFixupF16Gfx12, OmodOverflowRounding) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint32_t p, b, c;
    uint64_t flags;
    uint32_t expected;
  } cases[] = {
      {0x7bff, 0x0001, 0x0001, 0, 0x7c00},
      {0x7bff, 0x83ff, 0x0001, 0, 0xfc00},
      {0x7bff, 0x0001, 0x0001, RJ_ARITH_FP_ROUND_ZERO, 0x7bff},
      {0x7bff, 0x83ff, 0x0001, RJ_ARITH_FP_ROUND_ZERO, 0xfbff},
      {0x7bff, 0x3c00, 0x0001, RJ_ARITH_FP_ROUND_ZERO, 0x7bff},
      {0x7bff, 0xbc00, 0x3c00, RJ_ARITH_FP_ROUND_UP, 0xfbff},
      {0x7bff, 0xbc00, 0x3c00, RJ_ARITH_FP_ROUND_DOWN, 0xfc00},
      {0x7bff, 0x3c00, 0x3c00, RJ_ARITH_FP_ROUND_DOWN, 0x7bff},
      {0x7bff, 0x3c00, 0x3c00, RJ_ARITH_FP_ROUND_UP, 0x7c00},
      {0x7bff, 0x0001, 0x0001, RJ_ARITH_FP16_OVFL, 0x7bff},
      {0x7bff, 0x83ff, 0x0001, RJ_ARITH_FP16_OVFL, 0xfbff},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      for (uint64_t omod : {uint64_t(RJ_ARITH_ALU_OMOD_2), uint64_t(RJ_ARITH_ALU_OMOD_4)}) {
        SCOPED_TRACE(::testing::Message()
                     << std::hex << cpu << '/' << sample.b << '/' << sample.flags << '/' << omod);
        Registers<4, 64> r;
        for (int lane = 0; lane < 64; ++lane) {
          r.v[0][lane] = sample.p;
          r.v[1][lane] = sample.b;
          r.v[2][lane] = sample.c;
        }
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | sample.flags;
        // Check each wave size on its own destination writes.
        for (bool wide : {false, true}) {
          for (int lane = 0; lane < 64; ++lane) {
            r.v[3][lane] = 0x1234;
          }
          ASSERT_EQ(wide ? rj_arith_v_div_fixup_f16_w64_gfx12(flags, ~0ULL, omod, r.v + 3, r.v,
                                                              r.v + 1, r.v + 2, nullptr)
                         : rj_arith_v_div_fixup_f16_w32_gfx12(flags, ~0U, omod, r.v + 3, r.v,
                                                              r.v + 1, r.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          for (int lane = 0; lane < (wide ? 64 : 32); ++lane) {
            EXPECT_EQ(r.v[3][lane] & 0xffffU, sample.expected) << wide << '/' << lane;
          }
        }
      }
    }
  }
}

// Compare inherited CDNA policies across CPUs, with literal NaN/status witnesses.
template <int Width, int Lanes, bool Older, typename Call> static void check_cdna_fixup(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  const auto input = make_fp_inputs<Width, Lanes>();
  for (uint64_t cpu : supported_cpu_levels()) {
    for (int sample = 0; sample < 64; ++sample) {
      constexpr uint64_t semantics[] = {RJ_ARITH_SEMANTICS_LOOSE,
                                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                                        RJ_ARITH_SEMANTICS_COMPATIBILITY};
      uint64_t flags = semantics[sample % 3] | (uint64_t(sample & 3) << 24) |
                       (sample & 4 ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0) |
                       (sample & 8 ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0) |
                       (sample & 16 ? RJ_ARITH_FP_IEEE_MODE : 0);
      if (Older && (sample & 32)) {
        flags |= RJ_ARITH_FP_DX10_CLAMP_OFF;
      }
      uint64_t mode = uint64_t((sample >> 2) & 3) * RJ_ARITH_ALU_OMOD_2 |
                      (sample & 32 ? RJ_ARITH_ALU_CLAMP : 0) |
                      (sample & 16 ? RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_ABS_C : 0);
      if constexpr (Width == 16) {
        flags |= sample & 16 ? RJ_ARITH_FP16_OVFL : 0;
        mode |= sample & 1 ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D : 0;
      }
      const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL) | 1;
      const int destination = sample & 1 ? 0 : 6;
      auto expected = input, actual = input;
      uint32_t expected_status = 0x40, status = 0x40;
      ASSERT_EQ(call(flags, exec_mask, mode, expected.v + destination, expected.v, expected.v + 2,
                     expected.v + 4, &expected_status),
                0);
      ASSERT_EQ(call(flags | cpu, exec_mask, mode, actual.v + destination, actual.v, actual.v + 2,
                     actual.v + 4, &status),
                0);
      EXPECT_EQ(status, expected_status);
      actual.expect_preserved(input, destination, Width == 64 ? 2 : 1);
      for (int reg = 0; reg < (Width == 64 ? 2 : 1); ++reg) {
        for (int lane = 0; lane < Lanes; ++lane) {
          EXPECT_EQ(actual.v[destination + reg][lane], expected.v[destination + reg][lane]);
        }
      }
    }
    constexpr uint64_t one = Width == 16   ? 0x3c00ULL
                             : Width == 32 ? 0x3f800000ULL
                                           : 0x3ff0000000000000ULL;
    constexpr uint64_t inf = Width == 16   ? 0x7c00ULL
                             : Width == 32 ? 0x7f800000ULL
                                           : 0x7ff0000000000000ULL;
    constexpr uint64_t quiet = Width == 16   ? 0x200ULL
                               : Width == 32 ? 0x400000ULL
                                             : 0x8000000000000ULL;
    Registers<8, Lanes> r;
    write_fp(r.v, 0, Width, one);
    write_fp(r.v + 2, 0, Width, one);
    write_fp(r.v + 4, 0, Width, inf | 1);
    uint32_t status = 0x40;
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(1), 0, r.v + 6, r.v, r.v + 2,
                   r.v + 4, &status),
              0);
    EXPECT_EQ(read_fp(r.v + 6, 0, Width), inf | quiet | 1);
    EXPECT_EQ(status, 0x40U | (Width == 16 ? 0 : RJ_ARITH_EXCEPTION_INVALID));
    // Legacy CDNA retains a denormal cause alongside sNaN, including under CLAMP.
    // Input flushing removes only the denormal cause before classification.
    write_fp(r.v + 2, 0, Width, 1);
    for (bool flush : {false, true}) {
      status = 0x40;
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                         (flush ? RJ_ARITH_FP_FLUSH_INPUT_DENORMALS : 0),
                     Mask(1), RJ_ARITH_ALU_CLAMP, r.v + 6, r.v, r.v + 2, r.v + 4, &status),
                0);
      EXPECT_EQ(read_fp(r.v + 6, 0, Width), 0U);
      EXPECT_EQ(status, 0x40U | (Width == 16 ? 0
                                             : RJ_ARITH_EXCEPTION_INVALID |
                                                   (flush ? 0 : RJ_ARITH_EXCEPTION_INPUT_DENORM)));
    }
    write_fp(r.v + 2, 0, Width, 0);
    write_fp(r.v + 4, 0, Width, one);
    status = 0x40;
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(1), RJ_ARITH_ALU_CLAMP, r.v + 6,
                   r.v, r.v + 2, r.v + 4, &status),
              0);
    EXPECT_EQ(read_fp(r.v + 6, 0, Width), one);
    EXPECT_EQ(status, 0x40U | (Width == 16 ? 0 : RJ_ARITH_EXCEPTION_FLOAT_DIV0));
    // Directed overflow recovery chooses max-finite for positive RTZ/RTN.
    for (unsigned rounding = 0; rounding < 4; ++rounding) {
      write_fp(r.v, 0, Width, inf);
      write_fp(r.v + 2, 0, Width, one);
      write_fp(r.v + 4, 0, Width, one);
      const auto flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | (uint64_t(rounding) << 24);
      ASSERT_EQ(call(flags, Mask(1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr), 0);
      EXPECT_EQ(read_fp(r.v + 6, 0, Width), rounding < 2 ? inf : inf - 1);
      if constexpr (Width == 16) {
        // Directed rounding uses scalar packing, including its input-flush policy.
        write_fp(r.v, 0, Width, one);
        write_fp(r.v + 2, 0, Width, 1);
        ASSERT_EQ(call(flags | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, Mask(1), 0, r.v + 6, r.v, r.v + 2,
                       r.v + 4, nullptr),
                  0);
        EXPECT_EQ(read_fp(r.v + 6, 0, Width), inf);
      }
      if constexpr (Width != 16) {
        // The exponent-gap recovery is below half the smallest subnormal.
        write_fp(r.v, 0, Width, one);
        write_fp(r.v + 2, 0, Width, inf - 1);
        write_fp(r.v + 4, 0, Width, 1);
        ASSERT_EQ(call(flags, Mask(1), 0, r.v + 6, r.v, r.v + 2, r.v + 4, nullptr), 0);
        EXPECT_EQ(read_fp(r.v + 6, 0, Width), rounding == 1 ? 1U : 0U);
      }
    }
  }
  EXPECT_EQ(call(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, Mask(0), 0, nullptr, nullptr, nullptr,
                 nullptr, nullptr),
            RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
}

// gfx1100 IEEE=0 capture (hw1100-noieee) lanes at their MODE. FP16_OVFL saturates the
// recovered overflow before OMOD; OMOD scales the flushed result and then flushes to signed
// zero; IEEE clear keeps a selected sNaN under OMOD. The MODE 0x0c and 0x04 rows are separate
// gfx1100 (W7900) lanes: an OMOD overflow rounds toward zero to max-finite but up to infinity.
TEST(FpDivFixupF16Gfx11, CapturedOverflowOmodAndNan) {
  using namespace rocjitsu::arith::test;
  const struct {
    uint32_t guest_mode, p, b, c;
    uint64_t omod;
    uint32_t expected;
  } cases[] = {
      {0x800030, 0x7c00, 0x0400, 0x0400, 0, 0x7bff},
      {0x8000f0, 0x7c00, 0x0001, 0x0001, 0, 0x7bff},
      {0x80003a, 0x7c00, 0xbc00, 0x0400, 0, 0xfbff},
      {0x800000, 0x7c00, 0x0400, 0x0400, RJ_ARITH_ALU_OMOD_2, 0x7bff},
      {0x800000, 0x7c00, 0x0400, 0x0400, RJ_ARITH_ALU_OMOD_HALF, 0x77ff},
      {0x000000, 0x83ff, 0x0400, 0x0400, RJ_ARITH_ALU_OMOD_2, 0x0000},
      {0x000000, 0x83ff, 0x0400, 0x0400, RJ_ARITH_ALU_OMOD_4, 0x0000},
      {0x000000, 0x0400, 0xbc00, 0x0400, RJ_ARITH_ALU_OMOD_HALF, 0x8000},
      {0x000050, 0x0001, 0x83ff, 0x0001, RJ_ARITH_ALU_OMOD_2, 0x0000},
      {0x000050, 0x0400, 0x83ff, 0x0001, RJ_ARITH_ALU_OMOD_HALF, 0x8000},
      {0x000000, 0x0000, 0x7c01, 0x0000, RJ_ARITH_ALU_OMOD_2, 0x7c01},
      {0x000000, 0x0000, 0x7c01, 0x0000, RJ_ARITH_ALU_OMOD_4, 0x7c01},
      {0x000000, 0x0000, 0x7c01, 0x0000, RJ_ARITH_ALU_OMOD_HALF, 0x7c01},
      {0x00000c, 0x7bff, 0x3c00, 0x3c00, RJ_ARITH_ALU_OMOD_2, 0x7bff},
      {0x000004, 0x7bff, 0x3c00, 0x3c00, RJ_ARITH_ALU_OMOD_2, 0x7c00},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (const auto &sample : cases) {
      for (bool wide : {false, true}) {
        SCOPED_TRACE(::testing::Message() << std::hex << cpu << '/' << sample.guest_mode << '/'
                                          << sample.p << '/' << sample.omod << '/' << wide);
        Registers<4, 64> r;
        for (int lane = 0; lane < 64; ++lane) {
          r.v[0][lane] = 0xdc3a0000 | sample.p;
          r.v[1][lane] = 0x6a9e0000 | sample.b;
          r.v[2][lane] = 0x96e90000 | sample.c;
          r.v[3][lane] = 0xa5a5a5a5;
        }
        // Translate the F16 MODE fields; the capture leaves DX10_CLAMP and IEEE clear.
        const uint32_t m = sample.guest_mode;
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_DX10_CLAMP_OFF |
                               (uint64_t((m >> 2) & 3) << 24) |
                               (m & 0x40 ? 0 : RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) |
                               (m & 0x80 ? 0 : RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) |
                               (m & 0x800000 ? RJ_ARITH_FP16_OVFL : 0);
        ASSERT_EQ(wide ? rj_arith_v_div_fixup_f16_w64(flags, ~0ULL, sample.omod, r.v + 3, r.v,
                                                      r.v + 1, r.v + 2, nullptr)
                       : rj_arith_v_div_fixup_f16_w32(flags, ~0U, sample.omod, r.v + 3, r.v,
                                                      r.v + 1, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 64; ++lane) {
          EXPECT_EQ(r.v[3][lane], wide || lane < 32 ? 0xa5a50000 | sample.expected : 0xa5a5a5a5)
              << lane;
        }
      }
    }
  }
}

TEST(CdnaDivFixup, Arithmetic) {
  check_cdna_fixup<16, 32, true>(rj_arith_v_div_fixup_f16_w32_gfx942);
  check_cdna_fixup<16, 64, true>(rj_arith_v_div_fixup_f16_w64_gfx942);
  check_cdna_fixup<32, 32, true>(rj_arith_v_div_fixup_f32_w32_gfx942);
  check_cdna_fixup<32, 64, true>(rj_arith_v_div_fixup_f32_w64_gfx942);
  check_cdna_fixup<64, 32, true>(rj_arith_v_div_fixup_f64_w32_gfx942);
  check_cdna_fixup<64, 64, true>(rj_arith_v_div_fixup_f64_w64_gfx942);
  check_cdna_fixup<16, 32, false>(rj_arith_v_div_fixup_f16_w32_gfx1250);
  check_cdna_fixup<32, 32, false>(rj_arith_v_div_fixup_f32_w32_gfx1250);
  check_cdna_fixup<64, 32, false>(rj_arith_v_div_fixup_f64_w32_gfx1250);
}

// Legacy names share the inherited CDNA numerical and API contract.
TEST(CdnaDivFixup, LegacyHalf) {
  check_cdna_fixup<16, 32, true>(rj_arith_v_div_fixup_legacy_f16_w32);
  check_cdna_fixup<16, 64, true>(rj_arith_v_div_fixup_legacy_f16_w64);
}

// The inherited CDNA output order differs from RDNA's signed-underflow stage.
TEST(CdnaDivFixup, HalfOutputOrder) {
  using namespace rocjitsu::arith::test;
  struct Case {
    uint32_t p, c, omod, expected;
    uint64_t flags;
  };
  const Case cases[] = {
      {0x8400, 0xbc00, RJ_ARITH_ALU_OMOD_HALF, 0, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS},
      {0x7bff, 0x3c00, RJ_ARITH_ALU_OMOD_2, 0x7c00,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_ZERO},
      {0x7c00, 0x3c00, 0, 0x7c00, RJ_ARITH_FP16_OVFL},
      {0x7bff, 0x3c00, RJ_ARITH_ALU_OMOD_2, 0x7bff,
       RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP16_OVFL}};
  for (auto cpu : supported_cpu_levels()) {
    for (auto call : {rj_arith_v_div_fixup_f16_w32_gfx942, rj_arith_v_div_fixup_f16_w32_gfx1250,
                      rj_arith_v_div_fixup_legacy_f16_w32}) {
      for (auto test : cases) {
        Registers<4, 32> r;
        r.v[0][0] = test.p;
        r.v[1][0] = 0x3c00;
        r.v[2][0] = test.c;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | test.flags, 1, test.omod, r.v + 3,
                       r.v, r.v + 1, r.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(r.v[3][0] & 0xffffU, test.expected) << cpu << '/' << test.omod;
      }
    }
  }
}
