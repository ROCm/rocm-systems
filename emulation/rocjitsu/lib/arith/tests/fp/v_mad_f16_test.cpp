// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Legacy half MAD's FP32 intermediate, output controls, wave masks and aliases.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "fp/reference.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <gtest/gtest.h>
#include <random>
#include <type_traits>

namespace {

// Both MAD spellings inherit the same approximation; selected goldens pin its rounding stages.
template <int Lanes, bool Gfx90a = false, typename Call> void check_mad(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = (uint32_t(random() >> 15) << 16) | uint32_t(random() >> 15);
    }
  }
  // Single-round half FMA produces 0xe5a1; MAD's rounded FP32 intermediate produces 0xe5a2.
  input.v[0][0] = 0x47824782;
  input.v[1][0] = 0xda00da00;
  input.v[2][0] = 0x03e103e1;
  const auto cpus = supported_cpu_levels();
  for (auto semantics : semantics_modes()) {
    for (auto cpu : cpus) {
      auto actual = input;
      uint32_t status = 0x80000000;
      int result = call(cpu | semantics, Mask(1), 0, actual.v + 3, actual.v, actual.v + 1,
                        actual.v + 2, &status);
      if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
          semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
        EXPECT_EQ(result, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
        actual.expect_preserved(input, 0, 0);
      } else {
        ASSERT_EQ(result, RJ_ARITH_SUCCESS);
        EXPECT_EQ(actual.v[3][0], (input.v[3][0] & 0xffff0000) | 0xe5a2);
      }
      EXPECT_EQ(status, 0x80000000U);
    }
  }
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, Mask(0), mode, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_A |
          RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
          RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP);
  if constexpr (Gfx90a) {
    // FP16_OVFL saturates the product before the separately rounded add.
    for (auto cpu : cpus) {
      for (bool overflow : {false, true}) {
        auto actual = input;
        actual.v[0][0] = actual.v[1][0] = 0x7bff;
        actual.v[2][0] = 0;
        const uint64_t flags =
            cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (overflow ? RJ_ARITH_FP16_OVFL : 0);
        ASSERT_EQ(
            call(flags, Mask(1), 0, actual.v + 3, actual.v, actual.v + 1, actual.v + 2, nullptr),
            RJ_ARITH_SUCCESS);
        EXPECT_EQ(uint16_t(actual.v[3][0]), overflow ? 0x7bff : 0x7c00);
      }
    }
    // Selected inherited-model goldens distinguish two half roundings from FMA/FP32 MAD.
    const uint16_t operands[][3] = {{0x5c01, 0x3bfe, 0xdc00},
                                    {0x3c00, 0x3c00, 0x1000},
                                    {0x0001, 0x3c00, 0x0000},
                                    {0x7bff, 0x4000, 0x0000}};
    const uint16_t goldens[][4] = {{0x0000, 0x3c00, 0x0000, 0x7c00},
                                   {0x0000, 0x3c01, 0x0000, 0x7c00},
                                   {0xb000, 0x3c00, 0x0000, 0x7bff},
                                   {0xb000, 0x3c00, 0x0000, 0x7bff}};
    for (int rounding = 0; rounding < 4; ++rounding) {
      for (auto cpu : cpus) {
        auto actual = input;
        for (int lane = 0; lane < 4; ++lane) {
          for (int operand = 0; operand < 3; ++operand) {
            actual.v[operand][lane] = operands[lane][operand];
          }
        }
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24),
                       Mask(15), 0, actual.v + 3, actual.v, actual.v + 1, actual.v + 2, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 4; ++lane) {
          EXPECT_EQ(uint16_t(actual.v[3][lane]), goldens[rounding][lane]);
        }
      }
    }
  }
  // Independent numeric reference: decode halves, explicitly round the sum to FP32,
  // then use the test-only nearest-half encoder. This is an approximation contract.
  for (auto cpu : cpus) {
    auto actual = input;
    ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_LOOSE, ~Mask(0), 0, actual.v + 3, actual.v,
                   actual.v + 1, actual.v + 2, nullptr),
              RJ_ARITH_SUCCESS);
    for (int lane = 0; lane < Lanes; ++lane) {
      float x = float(half_value(uint16_t(input.v[0][lane])));
      float y = float(half_value(uint16_t(input.v[1][lane])));
      float z = float(half_value(uint16_t(input.v[2][lane])));
      const float intermediate = x * y + z;
      expect_fp(uint16_t(actual.v[3][lane]), half_bits(intermediate), 16, RJ_ARITH_SEMANTICS_LOOSE,
                /*atol=*/0x1p-14, /*rtol=*/0x1p-9);
    }
  }
  for (auto fp : {uint64_t(0), RJ_ARITH_FP_ROUND_UP, RJ_ARITH_FP_ROUND_DOWN, RJ_ARITH_FP_IEEE_MODE,
                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_ROUND_ZERO,
                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP16_OVFL,
                  RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_FP_IEEE_MODE}) {
    for (uint32_t mode :
         {0U, RJ_ARITH_ALU_OMOD_2, RJ_ARITH_ALU_OMOD_4, RJ_ARITH_ALU_OMOD_HALF,
          RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B,
          RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_ALU_HIGH_D}) {
      for (int destination : {0, 1, 2, 3}) {
        const Mask exec_mask = Mask(0xaaaaaaaaaaaaaaaaULL);
        auto expected = input;
        ASSERT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY | fp, exec_mask, mode,
                       expected.v + destination, expected.v, expected.v + 1, expected.v + 2,
                       nullptr),
                  RJ_ARITH_SUCCESS);
        for (auto cpu : cpus) {
          auto actual = input;
          ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | fp, exec_mask, mode,
                         actual.v + destination, actual.v, actual.v + 1, actual.v + 2, nullptr),
                    RJ_ARITH_SUCCESS);
          actual.expect_preserved(input, destination, 1);
          for (int lane = 0; lane < Lanes; ++lane) {
            if (!((exec_mask >> lane) & 1)) {
              EXPECT_EQ(actual.v[destination][lane], input.v[destination][lane]);
              continue;
            }
            int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
            EXPECT_EQ((actual.v[destination][lane] ^ input.v[destination][lane]) &
                          (shift ? 0xffffU : 0xffff0000U),
                      0U);
            if constexpr (Gfx90a) {
              EXPECT_EQ(actual.v[destination][lane], expected.v[destination][lane]);
            } else {
              expect_fp(actual.v[destination][lane] >> shift,
                        expected.v[destination][lane] >> shift, 16, RJ_ARITH_SEMANTICS_LOOSE,
                        /*atol=*/0x1p-14, /*rtol=*/0x1p-9);
            }
          }
        }
      }
    }
  }
}

} // namespace

TEST(FpMadF16W32, Arithmetic) {
  check_mad<32>(rj_arith_v_mad_f16_w32);
  check_mad<32>(rj_arith_v_mad_legacy_f16_w32);
}

TEST(FpMadF16W64, Arithmetic) {
  check_mad<64>(rj_arith_v_mad_f16_w64);
  check_mad<64>(rj_arith_v_mad_legacy_f16_w64);
}

TEST(FpMadF16W32Gfx90a, Arithmetic) {
  check_mad<32, true>(rj_arith_v_mad_f16_w32_gfx90a);
  check_mad<32, true>(rj_arith_v_mad_legacy_f16_w32_gfx90a);
}

TEST(FpMadF16W64Gfx90a, Arithmetic) {
  check_mad<64, true>(rj_arith_v_mad_f16_w64_gfx90a);
  check_mad<64, true>(rj_arith_v_mad_legacy_f16_w64_gfx90a);
}
