// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP32 narrowing to FP8/BF8, including stochastic rounding.

#include "conversion/fp8_narrow_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check FP8/BF8 narrowing, packing, and destination preservation for the supplied entry point.
template <int Lanes, bool Stochastic, bool Bf8, class Call> void check_fp8_narrow(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  // Midpoints and neighbors exercise discarded mantissa bits; seed extremes
  // distinguish truncation-before-seed-addition from ordinary stochastic rounding.
  const uint32_t edges[] = {0x3f880000U, 0x3f87ffffU, 0x3f880001U, 0x3f980000U,
                            0x3fa00000U, 0x3f9fffffU, 0x3fa00001U, 0x3b000001U,
                            0x37800001U, 0x43e80000U, 0x47700000U};
  for (int lane = 0; lane < 11; ++lane) {
    input.v[0][lane + 16] = edges[lane];
  }
  if constexpr (Stochastic) {
    for (int lane = 0; lane < 16; ++lane) {
      input.v[2][lane] = lane % 2 ? UINT32_MAX : 0;
    }
  }
  uint64_t modifiers = RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_A |
                       (Stochastic ? RJ_ARITH_CVT_BYTE_3
                                   : RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_HIGH_D);
  check_conversion<Lanes, 2, 1, 1, true>(
      call, input,
      [](uint64_t a, uint32_t b, uint32_t, uint32_t old, uint64_t flags, uint32_t mode) {
        return fp8_narrow_result((Stochastic ? 2 : 0) + (Bf8 ? 1 : 0), uint32_t(a), b, old, mode,
                                 flags & RJ_ARITH_FP16_OVFL);
      },
      conversion_bits, modifiers, true);
}

} // namespace

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkFp8F32W32, Arithmetic) {
  check_fp8_narrow<32, false, false>(rj_arith_v_cvt_pk_fp8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkBf8F32W32, Arithmetic) {
  check_fp8_narrow<32, false, true>(rj_arith_v_cvt_pk_bf8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrFp8F32W32, Arithmetic) {
  check_fp8_narrow<32, true, false>(rj_arith_v_cvt_sr_fp8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrBf8F32W32, Arithmetic) {
  check_fp8_narrow<32, true, true>(rj_arith_v_cvt_sr_bf8_f32_w32);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkFp8F32W64, Arithmetic) {
  check_fp8_narrow<64, false, false>(rj_arith_v_cvt_pk_fp8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtPkBf8F32W64, Arithmetic) {
  check_fp8_narrow<64, false, true>(rj_arith_v_cvt_pk_bf8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrFp8F32W64, Arithmetic) {
  check_fp8_narrow<64, true, false>(rj_arith_v_cvt_sr_fp8_f32_w64);
}

// Check FP8/BF8 narrowing, packing, and destination preservation.
TEST(ConversionCvtSrBf8F32W64, Arithmetic) {
  check_fp8_narrow<64, true, true>(rj_arith_v_cvt_sr_bf8_f32_w64);
}
