// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for division fixup, exceptional operands, and FP-status expectations.

#include "fp/div_fixup_test.h"
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
TEST(FpDivFixupF16W32, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 32>(rj_arith_v_div_fixup_f16_w32, /*gfx12=*/false,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 64>(rj_arith_v_div_fixup_f16_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF16W64, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<16, 64>(rj_arith_v_div_fixup_f16_w64, /*gfx12=*/false,
                                                 status_cases_f16);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 32>(rj_arith_v_div_fixup_f32_w32_gfx12, /*gfx12=*/true,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W32, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 32>(rj_arith_v_div_fixup_f32_w32, /*gfx12=*/false,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 64>(rj_arith_v_div_fixup_f32_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF32W64, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<32, 64>(rj_arith_v_div_fixup_f32_w64, /*gfx12=*/false,
                                                 status_cases_f32);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 32>(rj_arith_v_div_fixup_f64_w32_gfx12, /*gfx12=*/true,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W32, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 32>(rj_arith_v_div_fixup_f64_w32, /*gfx12=*/false,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 64>(rj_arith_v_div_fixup_f64_w64_gfx12, /*gfx12=*/true,
                                                 status_cases_f64);
}

// Check division fixup results and status against the family reference cases.
TEST(FpDivFixupF64W64, Arithmetic) {
  rocjitsu::arith::test::check_div_fixup<64, 64>(rj_arith_v_div_fixup_f64_w64, /*gfx12=*/false,
                                                 status_cases_f64);
}
