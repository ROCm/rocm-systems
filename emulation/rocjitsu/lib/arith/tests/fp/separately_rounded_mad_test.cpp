// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older FP32 MAD's two rounding steps, forced flushing, modifiers and overlaps.

#include "common/contracts.h"
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_fp.h"

#include <bit>
#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace {

// Check the inherited two-operation model; finite default-mode arithmetic has
// an ordinary volatile multiply/add reference, deliberately preventing FMA.
template <int Lanes, bool Legacy, typename Call> void check(Call call) {
  using namespace rocjitsu::arith::test;
  using Mask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;
  Registers<4, Lanes> input;
  std::minstd_rand random;
  uint32_t expected[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    for (int operand = 0; operand < 3; ++operand) {
      input.v[operand][lane] = 0x3f000000U | (uint32_t(random()) >> 8);
    }
    volatile float product =
        std::bit_cast<float>(input.v[0][lane]) * std::bit_cast<float>(input.v[1][lane]);
    expected[lane] =
        std::bit_cast<uint32_t>(float(product) + std::bit_cast<float>(input.v[2][lane]));
  }
  const struct {
    int lane;
    uint32_t a, b, c, result;
  } special[] = {
      {0, 0x3f800001, 0x3f7ffffe, 0xbf800000, 0}, // FMA would retain -2^-46.
      {1, 0x00800000, 0x3f000000, 0, 0},          // Flush the rounded product.
      {2, 0x80000000, 0x40000000, 0x80000000, Legacy ? 0U : 0x80000000U},
      {3, 0, 0x7f800000, 0x3f800000, Legacy ? 0x3f800000U : 0xffc00000U},
      {4, 0x7f800123, 0x40000000, 0, 0x7fc00123},
      {5, 0x7f800000, 0x40000000, 0x3f800000, 0x7f800000},
      {6, 1, 0x3f800000, 0x00800000, 0x00800000},
  };
  for (const auto &s : special) {
    input.v[0][s.lane] = s.a;
    input.v[1][s.lane] = s.b;
    input.v[2][s.lane] = s.c;
    expected[s.lane] = s.result;
  }
  const auto cpus = supported_cpu_levels();
  for (uint64_t cpu : cpus) {
    for (uint64_t semantics : semantics_modes()) {
      for (int aliasing_offset : {0, 1, 2, 3}) {
        for (Mask exec_mask : {Mask(0), Mask(1), ~Mask(0), Mask(1) << (Lanes - 1)}) {
          auto r = input;
          const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
          uint32_t state = 0x80000000;
          const int status =
              call(cpu | semantics, exec_mask, 0, r.v + aliasing_offset, &a, &b, &c, &state);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            EXPECT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
            r.expect_preserved(input, 0, 0);
          } else {
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            r.expect_preserved(input, aliasing_offset, 1);
            for (int lane = 0; lane < Lanes; ++lane) {
              if (!((exec_mask >> lane) & 1)) {
                EXPECT_EQ(r.v[aliasing_offset][lane], input.v[aliasing_offset][lane]);
              } else if (semantics == RJ_ARITH_SEMANTICS_LOOSE) {
                expect_f32(r.v[aliasing_offset][lane], std::bit_cast<float>(expected[lane]),
                           semantics, /*atol=*/1e-6, /*rtol=*/1e-6);
              } else {
                EXPECT_EQ(r.v[aliasing_offset][lane], expected[lane]) << "lane=" << lane;
              }
            }
          }
          EXPECT_EQ(state, 0x80000000U);
        }
      }
    }
    {
      auto r = input;
      r.v[0][0] = 0xbf000000;
      r.v[1][0] = 0x3e800000;
      r.v[2][0] = 0x3e000000;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, Mask(1),
                     RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C, r.v + 3, &a, &b,
                     &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], 0xbe800000U);
    }
    for (bool preserve_nan : {false, true}) {
      auto r = input;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                         (preserve_nan ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0),
                     Mask(1) << 4, RJ_ARITH_ALU_CLAMP, r.v + 3, &a, &b, &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][4], preserve_nan ? 0x7fc00123U : 0U);
    }
    // Effective OMOD can overflow after exact MAD arithmetic. Toward-zero
    // scaling saturates to max finite rather than returning infinity.
    for (int rounding : {0, 3}) {
      auto r = input;
      r.v[0][0] = 0x7f7fffff;
      r.v[1][0] = 0x3f800000;
      r.v[2][0] = 0;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                         (uint64_t(rounding) << 24),
                     Mask(1), RJ_ARITH_ALU_OMOD_2, r.v + 3, &a, &b, &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], rounding == 0 ? 0x7f800000U : 0x7f7fffffU);
    }
    // A product just above one differs under upward rounding, before C cancels one.
    for (int rounding = 0; rounding < 4; ++rounding) {
      auto r = input;
      r.v[0][0] = r.v[1][0] = 0x3f800001;
      r.v[2][0] = 0xbf800000;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | (uint64_t(rounding) << 24), Mask(1),
                     0, r.v + 3, &a, &b, &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], rounding == 1 ? 0x34c00000U : 0x34800000U);
    }
    // Packing precedes flushing: directed rounding rescues these tiny products.
    for (bool negative : {false, true}) {
      auto r = input;
      r.v[0][0] = negative ? 0x80800000U : 0x00800000U;
      r.v[1][0] = 0x3f7fffffU;
      r.v[2][0] = 0;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY |
                         (negative ? RJ_ARITH_FP_ROUND_DOWN : RJ_ARITH_FP_ROUND_UP),
                     Mask(1), 0, r.v + 3, &a, &b, &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], negative ? 0x80800000U : 0x00800000U);
    }
    // OMOD gates on the original guest denormal/IEEE state; source modifiers
    // apply before both operations and CLAMP uses the guest DX10 NaN policy.
    for (uint64_t mode : {uint64_t(0), RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                          RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
      auto r = input;
      r.v[0][0] = 0xbf000000;
      r.v[1][0] = 0x3f800000;
      r.v[2][0] = 0;
      const uint32_t *a = r.v[0], *b = r.v[1], *c = r.v[2];
      ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY | mode, Mask(1),
                     RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_OMOD_2, r.v + 3, &a, &b, &c, nullptr),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(r.v[3][0], mode == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 0x3f800000U : 0x3f000000U);
    }
  }
  const uint64_t supported = RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C |
                             RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C |
                             RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
  expect_invalid_flags(
      [&](uint64_t flags, uint64_t modifiers) {
        return call(flags, Mask(0), modifiers, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      supported);
}

} // namespace

TEST(MadF32W32, Arithmetic) { check<32, false>(rj_arith_v_mad_f32_w32); }
TEST(MadF32W64, Arithmetic) { check<64, false>(rj_arith_v_mad_f32_w64); }
TEST(MadLegacyF32W32, Arithmetic) { check<32, true>(rj_arith_v_mad_legacy_f32_w32); }
TEST(MadLegacyF32W64, Arithmetic) { check<64, true>(rj_arith_v_mad_legacy_f32_w64); }
