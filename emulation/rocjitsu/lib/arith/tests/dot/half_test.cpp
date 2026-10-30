// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for DOT2 with FP16/BF16 accumulators and results.

#include "common/floating_point.h"
#include "common/half_reference.h"
#include "conversion/conversion64_reference.h"
#include "dot/helpers.h"
#include "rj_arith/rj_arith.h"
#include <bit>

#include <gtest/gtest.h>
#include <stdint.h>

namespace {

using namespace rocjitsu::arith::test;

// Decode FP16/BF16 dot input bits with source modifiers and the format's denormal policy.
template <bool Bf16> double half_dot_input(uint16_t bits, uint32_t mode) {
  if (mode & RJ_ARITH_ALU_ABS_A) {
    bits &= 0x7fff;
  }
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits ^= 0x8000;
  }
  if constexpr (Bf16) {
    if (!(bits & 0x7f80)) {
      bits &= 0x8000;
    }
    return bf16_value(bits);
  } else {
    return half_value(bits);
  }
}

// Check 16-bit dot accumulation, packing, and source modifiers for the supplied entry point.
template <int Lanes, bool Bf16, bool Gfx11, typename Call> void check_half_dot(Call call) {
  // Selected gfx1100/gfx1201 witnesses from rocjitsu #13691 distinguish
  // the two accumulator floors without a large fixture corpus.
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      Registers<4, Lanes> r;
      for (int lane = 0; lane < Lanes; ++lane) {
        r.v[0][lane] = Bf16 ? 0x0cca8580U : 0x3e006400U;
        r.v[1][lane] = Bf16 ? 0x2f01c580U : 0x3c003c00U;
        r.v[2][lane] = Bf16 ? 0x00008005U : 0x00008001U;
      }
      uint32_t status = 0x55;
      ASSERT_EQ(
          call(cpu | semantics, ~WaveMask<Lanes>(0), 0, r.v + 2, r.v, r.v + 1, r.v + 2, &status),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(status, 0x55U);
      for (int lane = 0; lane < Lanes; ++lane) {
        EXPECT_EQ(r.v[2][lane], Bf16 ? (Gfx11 ? 0x0b80U : 0x0b7fU) : (Gfx11 ? 0x6402U : 0x6401U));
      }
    }
  }
  auto input = dot2_inputs<Lanes, Bf16, true>();
  // A finite overflow witness distinguishes FP16_OVFL from infinity input.
  uint16_t largest = Bf16 ? 0x7f7f : 0x7bff, one = Bf16 ? 0x3f80 : 0x3c00;
  input.v[0][0] = largest | (uint32_t(largest) << 16);
  input.v[1][0] = one | (uint32_t(one) << 16);
  input.v[2][0] = 0;
  // The fixed BF16 policy flushes these inputs regardless of guest flags.
  input.v[0][1] = 0x00010001U;
  input.v[1][1] = one | (uint32_t(one) << 16);
  input.v[2][1] = 0;
  if constexpr (Bf16) {
    // Normal inputs producing a subnormal result must flush on BF16 DOT2.
    input.v[0][12] = 0x00000080U;
    input.v[1][12] = 0x3f003f00U;
    input.v[2][12] = 0;
  }
  // SIMD-width boundaries: signed zero, signaling payload, and minimum normal
  // operands on both sides of an AVX2/AVX512 block boundary.
  for (int lane : {7, 8, 15, 16}) {
    uint16_t value = lane == 7   ? 0x8000
                     : lane == 8 ? (Bf16 ? 0x7f81 : 0x7c01)
                                 : (Bf16 ? 0x0080 : 0x0400);
    input.v[0][lane] = uint32_t(value) * 0x10001U;
    input.v[1][lane] = uint32_t(one) * 0x10001U;
    input.v[2][lane] = 0;
  }
  check_loose_dot<Lanes, false, true>(
      call, input,
      [](uint32_t actual, uint32_t a, uint32_t b, uint32_t c, uint32_t old, uint64_t flags,
         uint32_t mode) {
        int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
        uint32_t keep = ~(0xffffU << shift);
        EXPECT_EQ(actual & keep, old & keep);
        double sum =
            half_dot_input<Bf16>(uint16_t(c >> (mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0)), mode >> 2);
        for (int half = 0; half < 2; ++half) {
          sum += half_dot_input<Bf16>(uint16_t(a >> (16 * half)), mode) *
                 half_dot_input<Bf16>(uint16_t(b >> (16 * half)), mode >> 1);
        }
        if constexpr (Bf16) {
          uint16_t result = uint16_t(actual >> shift);
          if (!(result & 0x7f80)) {
            EXPECT_EQ(result & 0x7fff, 0); // Do not hide a flushing failure in atol.
          }
        }
        float expected;
        if constexpr (Bf16) {
          uint16_t rounded =
              uint16_t(conversion_reencode(std::bit_cast<uint32_t>(float(sum)), 23, 127, 7, 127));
          if (!(rounded & 0x7f80)) {
            rounded &= 0x8000;
          }
          expected = bf16_value(rounded);
        } else {
          expected = float(half_value(half_bits(sum, flags & RJ_ARITH_FP16_OVFL)));
        }
        float value =
            Bf16 ? bf16_value(uint16_t(actual >> shift)) : f16_value(uint16_t(actual >> shift));
        expect_f32(std::bit_cast<uint32_t>(value), expected, RJ_ARITH_SEMANTICS_LOOSE,
                   /*atol=*/Bf16 ? 0x1p-126 : 0x1p-24, /*rtol=*/Bf16 ? 4e-3 : 1e-3);
      },
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_A |
          RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D,
      true,
      RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
          RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS);
}

} // namespace

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2F16F16W32Gfx11, Arithmetic) {
  check_half_dot<32, false, true>(rj_arith_v_dot2_f16_f16_w32);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2F16F16W32Gfx12, Arithmetic) {
  check_half_dot<32, false, false>(rj_arith_v_dot2_f16_f16_w32_gfx12);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2Bf16Bf16W32Gfx11, Arithmetic) {
  check_half_dot<32, true, true>(rj_arith_v_dot2_bf16_bf16_w32);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2Bf16Bf16W32Gfx12, Arithmetic) {
  check_half_dot<32, true, false>(rj_arith_v_dot2_bf16_bf16_w32_gfx12);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2F16F16W64Gfx11, Arithmetic) {
  check_half_dot<64, false, true>(rj_arith_v_dot2_f16_f16_w64);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2F16F16W64Gfx12, Arithmetic) {
  check_half_dot<64, false, false>(rj_arith_v_dot2_f16_f16_w64_gfx12);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2Bf16Bf16W64Gfx11, Arithmetic) {
  check_half_dot<64, true, true>(rj_arith_v_dot2_bf16_bf16_w64);
}

// Check 16-bit dot accumulation, packing, and source modifiers.
TEST(Dot2Bf16Bf16W64Gfx12, Arithmetic) {
  check_half_dot<64, true, false>(rj_arith_v_dot2_bf16_bf16_w64_gfx12);
}
