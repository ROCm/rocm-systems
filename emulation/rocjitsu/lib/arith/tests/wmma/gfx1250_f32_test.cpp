// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 gfx1250 WMMA approximation, C modifiers, semantics and overlapping VGPRs.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
TEST(WmmaGfx1250F32, Arithmetic) {
  // One register bank permits D to overlap any input, or occupy independent storage.
  std::array<std::array<uint32_t, 32>, 20> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = random_f32_bits(random);
    }
  }
  for (int base : {0, 2, 4}) {
    input[base][0] = kF32QuietNaNBits;
    input[base][1] = kF32PositiveInfinityBits;
  }
  const auto levels = supported_cpu_levels();
  for (uint64_t controls :
       {0U, RJ_ARITH_WMMA_NEG_C, RJ_ARITH_WMMA_ABS_C, RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C}) {
    float expected[8][32];
    for (int reg = 0; reg < 8; ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        int row = reg + 8 * (lane / 16), col = lane % 16;
        float acc = std::bit_cast<float>(input[4 + reg][lane]);
        if (controls & RJ_ARITH_WMMA_ABS_C) {
          acc = std::fabs(acc);
        }
        if (controls & RJ_ARITH_WMMA_NEG_C) {
          acc = -acc;
        }
        for (int k = 0; k < 4; ++k) {
          float a = std::bit_cast<float>(input[k % 2][row + 16 * (k / 2)]);
          float b = std::bit_cast<float>(input[2 + k % 2][col + 16 * (k / 2)]);
          acc = std::fma(a, b, acc);
        }
        expected[reg][lane] = acc;
      }
    }
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        for (int aliasing_offset : {0, 2, 4, 12}) {
          auto storage = input;
          const uint32_t *a[2], *b[2], *c[8];
          uint32_t *d[8];
          for (int reg = 0; reg < 2; ++reg) {
            a[reg] = storage[reg].data();
            b[reg] = storage[2 + reg].data();
          }
          for (int reg = 0; reg < 8; ++reg) {
            c[reg] = storage[4 + reg].data();
            d[reg] = storage[aliasing_offset + reg].data();
          }
          int status = rj_arith_v_wmma_f32_16x16x4_f32_w32(cpu | semantics, controls, d, a, b, c);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(storage, input);
            continue;
          }
          ASSERT_EQ(status, RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < 8; ++reg) {
            for (int lane = 0; lane < 32; ++lane) {
              // Full FP32 inputs share MFMA's bound for fused versus unfused cancellation.
              expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/0.002, /*rtol=*/1e-5);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(rj_arith_v_wmma_f32_16x16x4_f32_w32(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr,
                                                nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
} // namespace
