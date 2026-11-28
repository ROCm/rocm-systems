// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Mixed BF16 FMA: inherited rounding examples, CPU-path agreement and overlap.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using rocjitsu::arith::test::Registers;
using Call = decltype(&rj_arith_v_fma_mix_f32_bf16_w32);

// Compare a Loose result at destination precision and retain the unselected BF16 half.
void expect_loose(uint32_t actual, uint32_t expected, uint32_t original, int kind) {
  const int shift = kind == 2 ? 16 : 0;
  const uint32_t actual_bits = kind ? ((actual >> shift) & 0xffffU) << 16 : actual;
  const uint32_t expected_bits = kind ? ((expected >> shift) & 0xffffU) << 16 : expected;
  rocjitsu::arith::test::expect_f32(actual_bits, std::bit_cast<float>(expected_bits),
                                    RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-40, /*rtol=*/1e-2);
  if (kind) {
    const uint32_t keep = kind == 1 ? 0xffff0000U : 0xffffU;
    EXPECT_EQ(actual & keep, original & keep);
  }
}

// Check one result format against selected inherited goldens and all CPU paths.
void check(Call call, int kind) {
  const auto cpus = rocjitsu::arith::test::supported_cpu_levels();
  struct Golden {
    uint32_t a, b, c;
    uint16_t result[4];
  };
  // Selected from rocjitsu's FmaMixBf16ResultsHonorRoundModeAndClamp regression.
  const Golden goldens[] = {
      {0x3f70df0d, 0x3f8de289, 0, {0x3f85, 0x3f86, 0x3f85, 0x3f85}},
      {0x3f800000, 0x3f800000, 1, {0x3f80, 0x3f81, 0x3f80, 0x3f80}},
      {0xff7f8000, 0x3f800000, 0, {0xff80, 0xff7f, 0xff80, 0xff7f}},
      {0x80018000, 0x3f800000, 0, {0x8002, 0x8001, 0x8002, 0x8001}},
      {0x3f800000, 0x3f800000, 0xbf800000, {0, 0, 0x8000, 0}},
      {0x7fc00001, 0x3f800000, 0, {0x7fc1, 0x7fc1, 0x7fc1, 0x7fc1}},
  };
  for (auto cpu : cpus) {
    for (auto semantics : rocjitsu::arith::test::semantics_modes()) {
      for (unsigned rounding = 0; rounding < 4; ++rounding) {
        Registers<4, 32> input;
        for (int lane = 0; lane < 32; ++lane) {
          const auto &g = goldens[lane % 6];
          input.v[0][lane] = g.a;
          input.v[1][lane] = g.b;
          input.v[2][lane] = g.c;
          input.v[3][lane] = 0x12345678;
        }
        auto output = input;
        uint32_t status = 0x80000000;
        const int result = call(cpu | semantics | (uint64_t(rounding) << 24), ~0U, 0, output.v + 3,
                                input.v, input.v + 1, input.v + 2, &status);
        const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                            semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
        EXPECT_EQ(result, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
        EXPECT_EQ(status, 0x80000000U);
        for (int lane = 0; lane < 32; ++lane) {
          const auto &g = goldens[lane % 6];
          uint32_t expected = std::bit_cast<uint32_t>(std::fma(
              std::bit_cast<float>(g.a), std::bit_cast<float>(g.b), std::bit_cast<float>(g.c)));
          const bool loose = !strict && semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY;
          if (kind) {
            const auto bits = g.result[loose ? 0 : rounding];
            expected = kind == 1 ? 0x12340000U | bits : (uint32_t(bits) << 16) | 0x5678U;
          }
          if (loose) {
            expect_loose(output.v[3][lane], expected, 0x12345678U, kind);
          } else {
            EXPECT_EQ(output.v[3][lane], strict ? 0x12345678U : expected);
          }
        }
      }
    }
  }
  // Fully specified engine/high-bit extraction keeps inputs stable across libraries.
  // This differential check supplements the independent goldens with wide encodings.
  std::minstd_rand random;
  for (int sample = 0; sample < 128; ++sample) {
    Registers<4, 32> input;
    for (int reg = 0; reg < 4; ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        const uint32_t lo = uint32_t(random()) >> 15;
        const uint32_t hi = uint32_t(random()) >> 15;
        input.v[reg][lane] = lo | (hi << 16);
      }
    }
    const uint32_t mode =
        (uint32_t(random()) >> 15) &
        (RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_A |
         RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A |
         RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_BF16_A | RJ_ARITH_MIX_BF16_B |
         RJ_ARITH_MIX_BF16_C);
    const int aliasing_offset = sample % 4;
    const uint32_t exec_mask = sample & 1 ? 0xa5a55a5aU : ~0U;
    const uint64_t semantics =
        sample & 4 ? RJ_ARITH_SEMANTICS_COMPATIBILITY : RJ_ARITH_SEMANTICS_LOOSE;
    const uint64_t flags = semantics | (uint64_t((sample / 8) % 4) << 24);
    auto expected = input;
    ASSERT_EQ(call(flags, exec_mask, mode, expected.v + aliasing_offset, expected.v, expected.v + 1,
                   expected.v + 2, nullptr),
              RJ_ARITH_SUCCESS);
    for (auto cpu : cpus) {
      SCOPED_TRACE(::testing::Message() << sample << '/' << cpu);
      auto output = input;
      ASSERT_EQ(call(flags | cpu, exec_mask, mode, output.v + aliasing_offset, output.v,
                     output.v + 1, output.v + 2, nullptr),
                RJ_ARITH_SUCCESS);
      for (int reg = 0; reg < 4; ++reg) {
        for (int lane = 0; lane < 32; ++lane) {
          if (semantics == RJ_ARITH_SEMANTICS_LOOSE && reg == aliasing_offset &&
              ((exec_mask >> lane) & 1)) {
            expect_loose(output.v[reg][lane], expected.v[reg][lane], input.v[reg][lane], kind);
          } else {
            EXPECT_EQ(output.v[reg][lane], expected.v[reg][lane]);
          }
          if (reg != aliasing_offset || !((exec_mask >> lane) & 1)) {
            EXPECT_EQ(output.v[reg][lane], input.v[reg][lane]);
          }
        }
      }
    }
  }
  rocjitsu::arith::test::expect_alu_invalid_flags<true>(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_A |
          RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A |
          RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_BF16_A | RJ_ARITH_MIX_BF16_B |
          RJ_ARITH_MIX_BF16_C);
}
} // namespace

TEST(FmaMixBf16, Float) { check(rj_arith_v_fma_mix_f32_bf16_w32, 0); }
TEST(FmaMixBf16, Low) { check(rj_arith_v_fma_mixlo_bf16_w32, 1); }
TEST(FmaMixBf16, High) { check(rj_arith_v_fma_mixhi_bf16_w32, 2); }
