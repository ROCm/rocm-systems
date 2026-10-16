// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for dX9 zero-product fused arithmetic entry points.

#include "fp/dx9_fma_test.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmaDx9ZeroF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<32, /*Accumulate=*/false>(
      rj_arith_v_fma_dx9_zero_f32_w32_gfx12);
}

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmaDx9ZeroF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<32, /*Accumulate=*/false>(rj_arith_v_fma_dx9_zero_f32_w32);
}

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmaDx9ZeroF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<64, /*Accumulate=*/false>(
      rj_arith_v_fma_dx9_zero_f32_w64_gfx12);
}

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmaDx9ZeroF32W64, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<64, /*Accumulate=*/false>(rj_arith_v_fma_dx9_zero_f32_w64);
}

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmacDx9ZeroF32W32, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<32, /*Accumulate=*/true>(rj_arith_v_fmac_dx9_zero_f32_w32);
}

// Check fused arithmetic with DX9 zero-product rules, routing, and aliases.
TEST(FpFmacDx9ZeroF32W64, Arithmetic) {
  rocjitsu::arith::test::check_dx9_fma<64, /*Accumulate=*/true>(rj_arith_v_fmac_dx9_zero_f32_w64);
}
