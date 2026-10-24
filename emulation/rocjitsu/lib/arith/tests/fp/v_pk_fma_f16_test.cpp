// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for packed FP16 fused multiply-add.

#include "fp/half_fma_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// FP exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

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
