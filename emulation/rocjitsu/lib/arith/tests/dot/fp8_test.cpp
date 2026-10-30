// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for FP8/BF8 DOT4 and DOT8 arithmetic.

#include "common/floating_point.h"
#include "conversion/fp8_conversion_reference.h"
#include "dot/helpers.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Check FP8/BF8 dot arithmetic, modifiers, and aliasing for the supplied entry point.
template <int Lanes, bool Bf8A, bool Bf8B, typename Call> void check_fp8_dot(Call call) {
  // One captured rounding/invalid-product witness per format pairing (#13691).
  constexpr uint32_t a =
      Bf8A ? (Bf8B ? 0x526563b0U : 0xd2e6da86U) : (Bf8B ? 0x003040feU : 0xd1539ce7U);
  constexpr uint32_t b =
      Bf8A ? (Bf8B ? 0x2bb0cb26U : 0xc5364dadU) : (Bf8B ? 0xfcbc8100U : 0x622dbba1U);
  constexpr uint32_t c = Bf8A ? (Bf8B ? 0xcc743234U : 0x2b21369bU) : (Bf8B ? 0U : 0xccb00900U);
  constexpr uint32_t expected =
      Bf8A ? (Bf8B ? 0xcc743e9bU : 0xc5184000U) : (Bf8B ? 0xffc00a3dU : 0xccb0092bU);
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
        continue;
      }
      Registers<4, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        r.v[0][lane] = a;
        r.v[1][lane] = b;
        r.v[2][lane] = c;
      }
      ASSERT_EQ(call(cpu | semantics, ~WaveMask<Lanes>(0), 0, r.v + 2, r.v, r.v + 1, r.v + 2),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(r.v[2][lane], expected);
      }
    }
  }
  // Minimum subnormal times one is exactly representable in FP32. Check bits:
  // the broad DOT4 tolerance below can hide a decoder that flushes this term.
  // All lanes stay finite so this also exercises the ordinary SIMD path.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      Registers<4, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        bool tiny_a = lane & 1;
        uint32_t sign = lane & 2 ? 0x80U : 0;
        r.v[0][lane] = (tiny_a ? 1U : (Bf8A ? 0x3cU : 0x38U)) | sign;
        r.v[1][lane] = tiny_a ? (Bf8B ? 0x3cU : 0x38U) : 1U;
        r.v[2][lane] = 0;
      }
      ASSERT_EQ(call(cpu | semantics, ~WaveMask<Lanes>(0), 0, r.v + 3, r.v, r.v + 1, r.v + 2),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < Lanes; ++lane) {
        // E4M3FN: 2^-9; E5M2: 2^-16. Alternate the subnormal operand and sign.
        uint32_t expected = (lane & 1 ? Bf8A : Bf8B) ? 0x37800000U : 0x3b000000U;
        expected |= lane & 2 ? 0x80000000U : 0;
        EXPECT_EQ(r.v[3][lane], expected) << cpu << '/' << semantics << '/' << lane;
      }
    }
  }
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
        a = ((a >> (mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0)) & 0xffffU) |
            ((a >> (mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16)) << 16);
        b = ((b >> (mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0)) & 0xffffU) |
            ((b >> (mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16)) << 16);
        if (mode & RJ_ARITH_DOT_ABS_C) {
          c &= 0x7fffffffU;
        }
        if (mode & RJ_ARITH_DOT_NEG_C) {
          c ^= 0x80000000U;
        }
        double sum = std::bit_cast<float>(c);
        for (int byte = 0; byte < 4; ++byte) {
          sum +=
              double(
                  std::bit_cast<float>(fp8_conversion_reference(Bf8A, uint8_t(a >> (8 * byte))))) *
              double(
                  std::bit_cast<float>(fp8_conversion_reference(Bf8B, uint8_t(b >> (8 * byte)))));
        }
        // Inherited bounded-factor DOT4 tolerance, also checked by this sample.
        expect_f32(actual, float(sum), RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/4e-6, /*rtol=*/2e-7);
      },
      RJ_ARITH_DOT_ABS_C | RJ_ARITH_DOT_NEG_C | RJ_ARITH_DOT_LO_A_HIGH | RJ_ARITH_DOT_LO_B_HIGH |
          RJ_ARITH_DOT_HI_A_LOW | RJ_ARITH_DOT_HI_B_LOW,
      true);
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
