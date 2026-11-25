// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx908 MIX's two rounding stages, SIMD parity, masks, overlap and validation.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <bit>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Compare selected SIMD results with the inherited scalar model; goldens pin
// independent FP32 and FP16 rounding.
template <int Lanes, int Kind, typename Call> void check(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<4, Lanes> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4; ++reg) {
    for (int lane = 0; lane < Lanes; ++lane) {
      input.v[reg][lane] = random_f32_bits(random);
    }
  }
  input.v[0][0] = input.v[1][0] = 0x3f800000;
  input.v[2][0] = Kind ? 0x3a000000 : 0x33800000; // Half an output ULP at one.
  // FP32 upward rounding moves a half tie just above the midpoint.
  input.v[0][2] = 0x3f801000;
  input.v[1][2] = 0x3f800000;
  input.v[2][2] = 0x33800000;
  input.v[0][1] = kF32SignalingNaNBits;
  input.v[1][1] = 0x3f800000;
  input.v[2][1] = 0;
  const auto cpus = supported_cpu_levels();
  for (unsigned rounding = 0; rounding < 4; ++rounding) {
    for (unsigned intermediate = 0; intermediate < (Kind ? 4U : 1U); ++intermediate) {
      const uint64_t fp = (uint64_t(rounding) << 24) | (uint64_t(intermediate) << 28);
      for (int aliasing_offset : {0, 1, 2, 3}) {
        auto expected = input;
        const uint32_t *a = expected.v[0], *b = expected.v[1], *c = expected.v[2];
        const Mask exec_mask = Mask(1) | (Mask(1) << (Lanes - 1)) | Mask(0xaaae);
        uint32_t status = 0x1234;
        ASSERT_EQ(call(fp | RJ_ARITH_SEMANTICS_COMPATIBILITY, exec_mask, 0,
                       expected.v + aliasing_offset, &a, &b, &c, &status),
                  RJ_ARITH_SUCCESS);
        ASSERT_EQ(status, 0x1234U);
        if (aliasing_offset == 3) {
          const uint32_t word = expected.v[3][0];
          if constexpr (Kind == 0) {
            EXPECT_EQ(word, rounding == 1 ? 0x3f800001U : 0x3f800000U);
          } else {
            EXPECT_EQ((word >> (Kind == 2 ? 16 : 0)) & 0xffffU, rounding == 1 ? 0x3c01U : 0x3c00U);
            EXPECT_EQ((expected.v[3][2] >> (Kind == 2 ? 16 : 0)) & 0xffffU,
                      rounding == 1 || (!rounding && intermediate == 1) ? 0x3c01U : 0x3c00U);
            EXPECT_EQ((expected.v[3][1] >> (Kind == 2 ? 16 : 0)) & 0x7e00U, 0x7e00U);
          }
        }
        for (auto semantics : semantics_modes()) {
          for (auto cpu : cpus) {
            auto actual = input;
            a = actual.v[0];
            b = actual.v[1];
            c = actual.v[2];
            const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
            const int result = call(fp | semantics | cpu, exec_mask, 0, actual.v + aliasing_offset,
                                    &a, &b, &c, &status);
            ASSERT_EQ(result, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
            EXPECT_EQ(status, 0x1234U);
            actual.expect_preserved(input, aliasing_offset, strict ? 0 : 1);
            for (int lane = 0; lane < Lanes; ++lane) {
              const uint32_t got = actual.v[aliasing_offset][lane];
              const uint32_t want = (strict || !((exec_mask >> lane) & 1))
                                        ? input.v[aliasing_offset][lane]
                                        : expected.v[aliasing_offset][lane];
              if (strict || !((exec_mask >> lane) & 1) || semantics != RJ_ARITH_SEMANTICS_LOOSE) {
                EXPECT_EQ(got, want);
              } else if constexpr (Kind == 0) {
                expect_f32(got, std::bit_cast<float>(want), semantics, /*atol=*/1e-5,
                           /*rtol=*/1e-5);
              } else {
                constexpr int shift = Kind == 2 ? 16 : 0;
                EXPECT_EQ((got ^ want) & ~(0xffffU << shift), 0U);
                expect_f32(std::bit_cast<uint32_t>(f16_value(uint16_t(got >> shift))),
                           f16_value(uint16_t(want >> shift)), semantics, /*atol=*/0.002,
                           /*rtol=*/0.002);
              }
            }
          }
        }
      }
    }
  }
  constexpr uint64_t modes = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                             RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                             RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B |
                             RJ_ARITH_ALU_HIGH_C | RJ_ARITH_MIX_F16_A | RJ_ARITH_MIX_F16_B |
                             RJ_ARITH_MIX_F16_C;
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t mode) {
        return call(flags, Mask(-1), mode, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      modes);
}

} // namespace

TEST(MixedMadGfx908, F32Wave32) { check<32, 0>(rj_arith_v_mad_mix_f32_w32_gfx908); }
TEST(MixedMadGfx908, F32Wave64) { check<64, 0>(rj_arith_v_mad_mix_f32_w64_gfx908); }
TEST(MixedMadGfx908, LowWave32) { check<32, 1>(rj_arith_v_mad_mixlo_f16_w32_gfx908); }
TEST(MixedMadGfx908, LowWave64) { check<64, 1>(rj_arith_v_mad_mixlo_f16_w64_gfx908); }
TEST(MixedMadGfx908, HighWave32) { check<32, 2>(rj_arith_v_mad_mixhi_f16_w32_gfx908); }
TEST(MixedMadGfx908, HighWave64) { check<64, 2>(rj_arith_v_mad_mixhi_f16_w64_gfx908); }
