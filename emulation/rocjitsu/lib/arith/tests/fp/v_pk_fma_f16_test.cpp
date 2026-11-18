// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 fused multiply-add.

#include "fp/fma_reference.h"
#include "fp/half_fma_test.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// FP exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
using rocjitsu::arith::test::StatusCase;

const StatusCase status_cases[] = {
    {0x3c00U, 0x3c00U, 0x0U, 0},
    {0x0U, 0x7c00U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7bffU, 0x4000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3c01U, 0x3c01U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
    {0x1U, 0x3800U, 0x0U,
     RJ_ARITH_EXCEPTION_INPUT_DENORM | RJ_ARITH_EXCEPTION_UNDERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
};

} // namespace

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmaF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/true, /*Accumulate=*/false>(
      rj_arith_v_pk_fma_f16_w32_gfx12,
      /*gfx12=*/true, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmaF16W32Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/true, /*Accumulate=*/false>(
      rj_arith_v_pk_fma_f16_w32, /*gfx12=*/false, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmaF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/true, /*Accumulate=*/false>(
      rj_arith_v_pk_fma_f16_w64_gfx12,
      /*gfx12=*/true, status_cases);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpPkFmaF16W64Gfx11, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/true, /*Accumulate=*/false>(
      rj_arith_v_pk_fma_f16_w64, /*gfx12=*/false, status_cases);
}

// Directed narrowing covers cancellation, tiny results and overflow without host-control writes.
TEST(FpPackedFma, GuestRounding) {
  struct Case {
    uint16_t a, b, c, expected;
    unsigned rounding;
    bool flush_output;
  };
  constexpr Case cases[] = {
      {0x3c00, 0x3c00, 0xbc00, 0x8000, 2, false}, // Cancellation under downward rounding.
      {0x8000, 0x3c00, 0x8000, 0x8000, 2, false}, // Matching negative zeros.
      {0x0001, 0x3800, 0, 0x0001, 1, false},      // Upward rounding reaches the minimum subnormal.
      {0x8001, 0x3800, 0, 0x8001, 2, false},
      {0x7bff, 0x4000, 0, 0x7bff, 2, false}, // Inward overflow remains finite.
      {0xfbff, 0x4000, 0, 0xfc00, 2, false},
      {0x0001, 0x3400, 0x8400, 0x8400, 2, true}, // Test tininess after guest significand rounding.
      {0x0001, 0x3400, 0x8400, 0x8000, 3, true},
  };
  for (uint64_t cpu : rocjitsu::arith::test::supported_cpu_levels()) {
    for (const auto &test : cases) {
      rocjitsu::arith::test::Registers<4, 32> r;
      for (int lane = 0; lane < 32; ++lane) {
        r.v[0][lane] = uint32_t(test.a) * 0x10001U;
        r.v[1][lane] = uint32_t(test.b) * 0x10001U;
        r.v[2][lane] = uint32_t(test.c) * 0x10001U;
      }
      const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                             (uint64_t(test.rounding) << 24) |
                             (test.flush_output ? RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS : 0);
      ASSERT_EQ(rj_arith_v_pk_fma_f16_w32(flags, ~0U, 0, r.v + 3, r.v, r.v + 1, r.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        EXPECT_EQ(r.v[3][lane], uint32_t(test.expected) * 0x10001U) << cpu << '/' << test.rounding;
      }
    }
  }
}
