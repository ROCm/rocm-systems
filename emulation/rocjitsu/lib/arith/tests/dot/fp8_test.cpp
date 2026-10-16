// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP8/BF8 DOT4 and DOT8 arithmetic.

#include "common/floating_point.h"
#include "conversion/fp8_conversion_reference.h"
#include "dot/helpers.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing for the supplied entry point.
template <int Lanes, bool Bf8A, bool Bf8B, class Call> void check_fp8_dot(Call call) {
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int reg = 0; reg < 2; ++reg) {
      bool bf8 = reg ? Bf8B : Bf8A;
      uint32_t word = 0;
      for (int byte = 0; byte < 4; ++byte) {
        // Magnitudes below four; all mantissa bits participate.
        uint32_t raw = uint32_t(random()) >> 23;
        int fraction = bf8 ? 2 : 3, bias = bf8 ? 15 : 7;
        uint8_t value = uint8_t((raw & 0x80) | ((bias - 2 + ((raw >> fraction) & 3)) << fraction) |
                                (raw & ((1U << fraction) - 1)));
        word |= uint32_t(value) << (8 * byte);
      }
      input.v[reg][lane] = word;
    }
    input.v[2][lane] = random_f32_bits(random);
    input.v[3][lane] = random_f32_bits(random);
  }
  input.v[0][2] = (input.v[0][2] & 0xffffff00U) | (Bf8A ? kBF8QuietNaNBits : kFP8QuietNaNBits);
  if constexpr (Bf8A) {
    input.v[0][3] = (input.v[0][3] & 0xffffff00U) | kBF8PositiveInfinityBits;
  }
  input.v[1][4] = (input.v[1][4] & 0xffffff00U) | (Bf8B ? kBF8QuietNaNBits : kFP8QuietNaNBits);
  input.v[1][5] =
      (input.v[1][5] & 0xffffff00U) | (Bf8B ? kBF8PositiveInfinityBits : kFP8QuietNaNBits);
  input.v[2][8] = kF32QuietNaNBits;
  input.v[2][9] = kF32PositiveInfinityBits;
  // Zero/subnormal and top-finite encodings across SIMD block boundaries.
  input.v[0][7] = 0x80010001U;
  input.v[1][7] = 0x01008001U;
  input.v[0][15] = Bf8A ? 0x7b7b7b7bU : 0x7e7e7e7eU;
  input.v[1][15] = 0x01010101U;
  input.v[2][7] = input.v[2][15] = 0;
  check_loose_dot<Lanes, false, false>(
      call, input,
      [](uint32_t actual, uint32_t a, uint32_t b, uint32_t c, uint32_t, uint64_t, uint32_t mode) {
        if (mode & RJ_ARITH_DOT_ABS_C) {
          c &= 0x7fffffffU;
        }
        if (mode & RJ_ARITH_DOT_NEG_C) {
          c ^= 0x80000000U;
        }
        double sum = bit_cast<float>(c);
        for (int byte = 0; byte < 4; ++byte) {
          sum += double(bit_cast<float>(fp8_conversion_reference(Bf8A, uint8_t(a >> (8 * byte))))) *
                 double(bit_cast<float>(fp8_conversion_reference(Bf8B, uint8_t(b >> (8 * byte)))));
        }
        // Inherited bounded-factor DOT4 tolerance, also checked by this sample.
        expect_f32(actual, float(sum), RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/4e-6, /*rtol=*/2e-7);
      },
      RJ_ARITH_DOT_ABS_C | RJ_ARITH_DOT_NEG_C, false);
}

} // namespace

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Fp8Fp8W32, Arithmetic) {
  check_fp8_dot<32, false, false>(rj_arith_v_dot4_f32_fp8_fp8_w32);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Fp8Bf8W32, Arithmetic) {
  check_fp8_dot<32, false, true>(rj_arith_v_dot4_f32_fp8_bf8_w32);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Bf8Fp8W32, Arithmetic) {
  check_fp8_dot<32, true, false>(rj_arith_v_dot4_f32_bf8_fp8_w32);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Bf8Bf8W32, Arithmetic) {
  check_fp8_dot<32, true, true>(rj_arith_v_dot4_f32_bf8_bf8_w32);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Fp8Fp8W64, Arithmetic) {
  check_fp8_dot<64, false, false>(rj_arith_v_dot4_f32_fp8_fp8_w64);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Fp8Bf8W64, Arithmetic) {
  check_fp8_dot<64, false, true>(rj_arith_v_dot4_f32_fp8_bf8_w64);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Bf8Fp8W64, Arithmetic) {
  check_fp8_dot<64, true, false>(rj_arith_v_dot4_f32_bf8_fp8_w64);
}

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing.
TEST(Dot4F32Bf8Bf8W64, Arithmetic) {
  check_fp8_dot<64, true, true>(rj_arith_v_dot4_f32_bf8_bf8_w64);
}
