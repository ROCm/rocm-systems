// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Gfx10 FP16 numeric requests and inherited Compatibility status, including both
// wave sizes and CPU levels, aliases, DPP, partial EXEC and strict-mode rejection.

#include "common/cpu.h"
#include "common/registers.h"
#include "fp/binary_arithmetic_test.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <type_traits>

namespace {

// Pin a signaling NaN and arithmetic overflow without reusing the implementation classifier.
// Only Compatibility accumulates these inherited causes; other semantics leave status alone.
template <int Lanes, bool Reporting, typename Call>
void check_status(Call call, uint16_t overflow_a, uint16_t overflow_b) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  struct Case {
    uint16_t a, b;
    uint32_t causes;
  };
  const Case cases[] = {
      {0x7c01, 0x3c00, RJ_ARITH_EXCEPTION_INVALID},
      {overflow_a, overflow_b, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (const auto &c : cases) {
        Registers<3, Lanes> input;
        input.v[0][Lanes - 1] = c.a;
        input.v[1][Lanes - 1] = c.b;
        uint32_t prior = 0x80000000U | RJ_ARITH_EXCEPTION_FLOAT_DIV0;
        uint32_t status = prior;
        ASSERT_EQ(call(cpu | semantics, Mask(1) << (Lanes - 1), 0, input.v + 2, input.v,
                       input.v + 1, &status),
                  RJ_ARITH_SUCCESS);
        EXPECT_EQ(status,
                  prior |
                      (Reporting && semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY ? c.causes : 0));
      }
    }
  }
}

} // namespace

// Check the numeric contract and the selected legacy status policy.
TEST(FpAddF16W32Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx1010, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<32, true>(rj_arith_v_add_f16_w32_gfx1010, 0x7bff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpAddF16W64Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx1010, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<64, true>(rj_arith_v_add_f16_w64_gfx1010, 0x7bff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubF16W32Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx1010, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<32, false>(rj_arith_v_sub_f16_w32_gfx1010, 0x7bff, 0xfbff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubF16W64Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx1010, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<64, false>(rj_arith_v_sub_f16_w64_gfx1010, 0x7bff, 0xfbff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubrevF16W32Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx1010, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<32, false>(rj_arith_v_subrev_f16_w32_gfx1010, 0xfbff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubrevF16W64Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx1010, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<64, false>(rj_arith_v_subrev_f16_w64_gfx1010, 0xfbff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpMulF16W32Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32_gfx1010, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<32, false>(rj_arith_v_mul_f16_w32_gfx1010, 0x7bff, 0x4000);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpMulF16W64Gfx1010, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx1010, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = false, .half_exact = false});
  check_status<64, false>(rj_arith_v_mul_f16_w64_gfx1010, 0x7bff, 0x4000);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpAddF16W32Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_add_f16_w32_gfx1030, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<32, true>(rj_arith_v_add_f16_w32_gfx1030, 0x7bff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpAddF16W64Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_add_f16_w64_gfx1030, [](double x, double y) { return x + y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<64, true>(rj_arith_v_add_f16_w64_gfx1030, 0x7bff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubF16W32Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_sub_f16_w32_gfx1030, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<32, true>(rj_arith_v_sub_f16_w32_gfx1030, 0x7bff, 0xfbff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubF16W64Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_sub_f16_w64_gfx1030, [](double x, double y) { return x - y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<64, true>(rj_arith_v_sub_f16_w64_gfx1030, 0x7bff, 0xfbff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubrevF16W32Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_subrev_f16_w32_gfx1030, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<32, true>(rj_arith_v_subrev_f16_w32_gfx1030, 0xfbff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpSubrevF16W64Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_subrev_f16_w64_gfx1030, [](double x, double y) { return y - x; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<64, true>(rj_arith_v_subrev_f16_w64_gfx1030, 0xfbff, 0x7bff);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpMulF16W32Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 32>(
      rj_arith_v_mul_f16_w32_gfx1030, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<32, true>(rj_arith_v_mul_f16_w32_gfx1030, 0x7bff, 0x4000);
}

// Check the numeric contract and the selected legacy status policy.
TEST(FpMulF16W64Gfx1030, Arithmetic) {
  rocjitsu::arith::test::check_binary_arithmetic<16, 64>(
      rj_arith_v_mul_f16_w64_gfx1030, [](double x, double y) { return x * y; },
      /*gfx12=*/false, /*fp_status_supported=*/false,
      {.half_selectors = false, .compatibility_reports_status = true, .half_exact = false});
  check_status<64, true>(rj_arith_v_mul_f16_w64_gfx1030, 0x7bff, 0x4000);
}
