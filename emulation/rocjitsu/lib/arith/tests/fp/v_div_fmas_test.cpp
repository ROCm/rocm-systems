// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for conditionally scaled division FMA and FP-status expectations.

#include "fp/div_fmas_test.h"
#include "rj_arith/rj_arith.h"

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

const StatusCase status_cases_f32[] = {
    {0x3f800000U, 0x3f800000U, 0x0U, 0},
    {0x0U, 0x7f800000U, 0x0U, 0},
    {0x7f7fffffU, 0x40000000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3f800001U, 0x3f800001U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1U, 0x3f000000U, 0x0U, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    // The inherited classifier tests tininess at normal precision, before
    // limiting the exponent. Rounding up to minimum normal only at subnormal
    // precision still reports UNDERFLOW; the closer product below does not.
    {0x00800000U, 0x3f7fffffU, 0, RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800001U, 0x3f7ffffeU, 0, RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800000U, 0x3f7fffffU, 0x80000001U,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x00800000U, 0x3f7ffffeU, 0, 0},
};

const StatusCase status_cases_f64[] = {
    {0x3ff0000000000000ULL, 0x3ff0000000000000ULL, 0x0ULL, 0},
    {0x0ULL, 0x7ff0000000000000ULL, 0x0ULL, 0},
    {0x7fefffffffffffffULL, 0x4000000000000000ULL, 0x0ULL,
     RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3ff0000000000001ULL, 0x3ff0000000000001ULL, 0x0ULL, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1ULL, 0x3fe0000000000000ULL, 0x0ULL,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3fefffffffffffffULL, 0,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000001ULL, 0x3feffffffffffffeULL, 0, RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3fefffffffffffffULL, 0x8000000000000001ULL,
     RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x0010000000000000ULL, 0x3feffffffffffffeULL, 0, 0},
};

} // namespace

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 32>(rj_arith_v_div_fmas_f32_w32_gfx12, /*gfx12=*/true,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 32>(rj_arith_v_div_fmas_f32_w32, /*gfx12=*/false,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 64>(rj_arith_v_div_fmas_f32_w64_gfx12, /*gfx12=*/true,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF32W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<32, 64>(rj_arith_v_div_fmas_f32_w64, /*gfx12=*/false,
                                                status_cases_f32);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 32>(rj_arith_v_div_fmas_f64_w32_gfx12, /*gfx12=*/true,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 32>(rj_arith_v_div_fmas_f64_w32, /*gfx12=*/false,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 64>(rj_arith_v_div_fmas_f64_w64_gfx12, /*gfx12=*/true,
                                                status_cases_f64);
}

// Check division FMA results, conditional scaling, and supported FP status.
TEST(FpDivFmasF64W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_div_fmas<64, 64>(rj_arith_v_div_fmas_f64_w64, /*gfx12=*/false,
                                                status_cases_f64);
}

// gfx1100/W7900 captures supplied in PR #13952: selected sNaNs quiet only
// with IEEE set (VCC=0).
template <int Width, int Lanes, typename Call> static void check_rdna3_nan(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  constexpr uint64_t one = Width == 32 ? 0x3f800000ULL : 0x3ff0000000000000ULL;
  constexpr uint64_t snan = Width == 32 ? 0x7f800001ULL : 0x7ff0000000000001ULL;
  constexpr uint64_t quiet = Width == 32 ? 0x00400000ULL : 0x0008000000000000ULL;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS}) {
      for (uint64_t ieee : {uint64_t(0), RJ_ARITH_FP_IEEE_MODE}) {
        for (int operand : {0, 1, 2}) {
          Registers<8, Lanes> r;
          for (int i = 0; i < 3; ++i) {
            write_fp(r.v + 2 * i, Lanes - 1, Width, i == operand ? snan : one);
          }
          uint32_t status = 0x80000000U;
          ASSERT_EQ(call(cpu | semantics | ieee, Mask(1) << (Lanes - 1), 0, r.v + 6, r.v, r.v + 2,
                         r.v + 4, Mask(0), &status),
                    RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x80000000U);
          EXPECT_EQ(read_fp(r.v + 6, Lanes - 1, Width), snan | (ieee ? quiet : 0));
        }
      }
    }
  }
}

TEST(FpDivFmasGfx11, SignalingNan) {
  check_rdna3_nan<32, 32>(rj_arith_v_div_fmas_f32_w32);
  check_rdna3_nan<32, 64>(rj_arith_v_div_fmas_f32_w64);
  check_rdna3_nan<64, 32>(rj_arith_v_div_fmas_f64_w32);
  check_rdna3_nan<64, 64>(rj_arith_v_div_fmas_f64_w64);
}
