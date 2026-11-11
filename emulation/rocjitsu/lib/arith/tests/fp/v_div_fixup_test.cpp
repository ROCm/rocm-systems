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

// Active RDNA3 FP16 OMOD has no empirical qualification. Rejection is transactional,
// including empty EXEC; Compatibility and fallback-allowed semantics keep the loose result.
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
          EXPECT_EQ(r.v[3][0], expected || !exec_mask ? 0x1234u : 0x7e01u);
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
