// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fMA with a literal addend.

#include "fp/immediate_fma_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

// Literal exception contributions from the inherited rocjitsu model and the
// old suite's gfx1201-validated cases. Test each cause alone, then accumulated;
// an inactive exceptional lane must not contribute. No host flags are read.
struct StatusCase {
  uint64_t a, b, c;
  uint32_t causes;
};

const StatusCase status_cases_f16[] = {
    {0x0U, 0x3c00U, 0x0U, 0},
    {0x0U, 0x7c00U, 0x0U, RJ_ARITH_EXCEPTION_INVALID},
    {0x7bffU, 0x4000U, 0x0U, RJ_ARITH_EXCEPTION_OVERFLOW | RJ_ARITH_EXCEPTION_INEXACT},
    {0x3c01U, 0x3c01U, 0x0U, RJ_ARITH_EXCEPTION_INEXACT},
};

} // namespace

// Check fMA with a literal addend.
TEST(FpFmaakF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f16<32, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f16_w32_gfx12,
      /*gfx12=*/true, status_cases_f16);
}

// Check fMA with a literal addend.
TEST(FpFmaakF16W32, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f16<32, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f16_w32,
      /*gfx12=*/false, status_cases_f16);
}

// Check fMA with a literal addend.
TEST(FpFmaakF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f16<64, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f16_w64_gfx12,
      /*gfx12=*/true, status_cases_f16);
}

// Check fMA with a literal addend.
TEST(FpFmaakF16W64, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f16<64, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f16_w64,
      /*gfx12=*/false, status_cases_f16);
}

// Check fMA with a literal addend.
TEST(FpFmaakF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f32<32, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f32_w32_gfx12);
}

// Check fMA with a literal addend.
TEST(FpFmaakF32W32, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f32<32, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f32_w32);
}

// Check fMA with a literal addend.
TEST(FpFmaakF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f32<64, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f32_w64_gfx12);
}

// Check fMA with a literal addend.
TEST(FpFmaakF32W64, Arithmetic) {
  rocjitsu::arith::test::check_immediate_fma_f32<64, /*MultiplyConstant=*/false>(
      rj_arith_v_fmaak_f32_w64);
}
