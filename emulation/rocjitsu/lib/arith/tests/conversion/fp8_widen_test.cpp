// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP8/BF8 widening, including packed source selections.

#include "conversion/fp8_conversion_reference.h"
#include "conversion/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check FP8/BF8 widening and packed source selection for the supplied entry point.
template <int Lanes, bool Packed, bool Bf8, class Call> void check_fp8_widen(Call call) {
  auto input = conversion_inputs<Lanes>(32);
  // Keep every class in both selected bytes, including NaNs and signed zeros.
  const uint32_t edge[] = {0x807f7c00U, 0xfffe7e01U, 0x01807fffU, 0x7c007e80U};
  for (int lane = 0; lane < 4; ++lane) {
    input.v[0][lane] = edge[lane];
  }
  check_conversion<Lanes, 1, 1, Packed ? 2 : 1, !Packed>(
      call, input,
      [](uint64_t a, uint32_t, uint32_t, uint32_t, uint64_t, uint32_t mode) {
        int shift = Packed ? (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0) : 8 * ((mode >> 16) & 3);
        uint64_t result = fp8_conversion_reference(Bf8, uint8_t(a >> shift));
        if constexpr (Packed) {
          result |= uint64_t(fp8_conversion_reference(Bf8, uint8_t(a >> (shift + 8)))) << 32;
        }
        return result;
      },
      conversion_bits, Packed ? RJ_ARITH_ALU_HIGH_A : RJ_ARITH_CVT_BYTE_3, true);
}

} // namespace

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Fp8W32, Arithmetic) {
  check_fp8_widen<32, false, false>(rj_arith_v_cvt_f32_fp8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Bf8W32, Arithmetic) {
  check_fp8_widen<32, false, true>(rj_arith_v_cvt_f32_bf8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Fp8W32, Arithmetic) {
  check_fp8_widen<32, true, false>(rj_arith_v_cvt_pk_f32_fp8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Bf8W32, Arithmetic) {
  check_fp8_widen<32, true, true>(rj_arith_v_cvt_pk_f32_bf8_w32);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Fp8W64, Arithmetic) {
  check_fp8_widen<64, false, false>(rj_arith_v_cvt_f32_fp8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtF32Bf8W64, Arithmetic) {
  check_fp8_widen<64, false, true>(rj_arith_v_cvt_f32_bf8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Fp8W64, Arithmetic) {
  check_fp8_widen<64, true, false>(rj_arith_v_cvt_pk_f32_fp8_w64);
}

// Check FP8/BF8 widening and packed source selection.
TEST(ConversionCvtPkF32Bf8W64, Arithmetic) {
  check_fp8_widen<64, true, true>(rj_arith_v_cvt_pk_f32_bf8_w64);
}
