// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for mixed FP16/FP32 FMA entry points.

#include "fp/mixed_fma_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<0, 32>(rj_arith_v_fma_mix_f32_w32_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixF32W32, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<0, 32>(rj_arith_v_fma_mix_f32_w32, /*gfx12=*/false);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<0, 64>(rj_arith_v_fma_mix_f32_w64_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixF32W64, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<0, 64>(rj_arith_v_fma_mix_f32_w64, /*gfx12=*/false);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixloF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<1, 32>(rj_arith_v_fma_mixlo_f16_w32_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixloF16W32, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<1, 32>(rj_arith_v_fma_mixlo_f16_w32, /*gfx12=*/false);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixloF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<1, 64>(rj_arith_v_fma_mixlo_f16_w64_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixloF16W64, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<1, 64>(rj_arith_v_fma_mixlo_f16_w64, /*gfx12=*/false);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixhiF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<2, 32>(rj_arith_v_fma_mixhi_f16_w32_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixhiF16W32, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<2, 32>(rj_arith_v_fma_mixhi_f16_w32, /*gfx12=*/false);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixhiF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<2, 64>(rj_arith_v_fma_mixhi_f16_w64_gfx12, /*gfx12=*/true);
}

// Check mixed-precision results, halfword preservation, and guest FP controls.
TEST(FpFmaMixhiF16W64, Arithmetic) {
  rocjitsu::arith::test::check_mixed_fma<2, 64>(rj_arith_v_fma_mixhi_f16_w64, /*gfx12=*/false);
}
