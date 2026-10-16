// SPDX-License-Identifier: MIT

// FMA arithmetic and storage contracts for both wave sizes and element types.

#include "fp/fma_test.h"
#include "rj_arith/rj_arith.h"

// Check FP16 arithmetic for Wave32.
TEST(FpFmaF16W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma16<32>(rj_arith_v_fma_f16_w32_gfx12);
}

// Check FP16 arithmetic for Wave64.
TEST(FpFmaF16W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma16<64>(rj_arith_v_fma_f16_w64_gfx12);
}

// Check FP32 arithmetic for Wave32.
TEST(FpFmaF32W32Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma32<32>(rj_arith_v_fma_f32_w32_gfx12);
}

// Check FP32 arithmetic for Wave64.
TEST(FpFmaF32W64Gfx12, Arithmetic) {
  rocjitsu::arith::test::check_fma32<64>(rj_arith_v_fma_f32_w64_gfx12);
}

// Check the gfx11 arithmetic and unsupported status contract.
TEST(FpFmaF16W32, Arithmetic) {
  rocjitsu::arith::test::check_fma16<32, true>(rj_arith_v_fma_f16_w32);
}

// Check the gfx11 arithmetic and unsupported status contract.
TEST(FpFmaF16W64, Arithmetic) {
  rocjitsu::arith::test::check_fma16<64, true>(rj_arith_v_fma_f16_w64);
}

// Check the gfx11 arithmetic and unsupported status contract.
TEST(FpFmaF32W32, Arithmetic) {
  rocjitsu::arith::test::check_fma32<32, true>(rj_arith_v_fma_f32_w32);
}

// Check the gfx11 arithmetic and unsupported status contract.
TEST(FpFmaF32W64, Arithmetic) {
  rocjitsu::arith::test::check_fma32<64, true>(rj_arith_v_fma_f32_w64);
}
