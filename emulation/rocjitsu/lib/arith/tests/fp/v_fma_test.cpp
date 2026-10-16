// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for fused arithmetic across FP widths, aliases, and guest controls.

#include "fp/fma_test.h"
#include "fp/half_fma_test.h"
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
TEST(FpFmaF16W32, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<32, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w32, /*gfx12=*/false, status_cases_f16);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w64_gfx12, /*gfx12=*/true, status_cases_f16);
}

// Check FP16 fused arithmetic, halfword selection, aliases, and FP-status cases.
TEST(FpFmaF16W64, Arithmetic) {
  rocjitsu::arith::test::check_half_fma<64, /*Packed=*/false, /*Accumulate=*/false>(
      rj_arith_v_fma_f16_w64, /*gfx12=*/false, status_cases_f16);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 32>(rj_arith_v_fma_f32_w32_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W32, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 32>(rj_arith_v_fma_f32_w32, /*gfx12=*/false);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 64>(rj_arith_v_fma_f32_w64_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF32W64, Arithmetic) {
  rocjitsu::arith::test::check_fma<32, 64>(rj_arith_v_fma_f32_w64, /*gfx12=*/false);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 32>(rj_arith_v_fma_f64_w32_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W32, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 32>(rj_arith_v_fma_f64_w32, /*gfx12=*/false);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 64>(rj_arith_v_fma_f64_w64_gfx12, /*gfx12=*/true);
}

// Check fused arithmetic, modifiers, aliases, and supported FP-status behavior.
TEST(FpFmaF64W64, Arithmetic) {
  rocjitsu::arith::test::check_fma<64, 64>(rj_arith_v_fma_f64_w64, /*gfx12=*/false);
}
