// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TANH tests retain the inherited libm model, destination rounding and storage contract.
#include "common/cpu.h"
#include "common/half_reference.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;

// Compare finite TANH to host float libm, with explicit NaN/Inf and modifier expectations.
template <bool Half> void check() {
  auto call = Half ? rj_arith_v_tanh_f16_w32_gfx1250 : rj_arith_v_tanh_f32_w32_gfx1250;
  std::array<uint32_t, 32> input;
  std::minstd_rand random;
  for (auto &bits : input) {
    const float x = float(int(random() >> 16) - 16384) / 2048.0f;
    bits = Half ? half_bits(x) : std::bit_cast<uint32_t>(x);
  }
  input[0] = Half ? 0x7c00U : 0x7f800000U;
  input[1] = Half ? 0x7d23U : 0x7f812345U;
  input[2] = Half ? 0x8000U : 0x80000000U;
  input[3] = 1;
  input[4] = Half ? 0x8400U : 0x80800000U;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : semantics_modes()) {
      for (int omod = 0; omod < 4; ++omod) {
        for (bool clamp : {false, true}) {
          auto actual = input;
          const uint32_t *a[] = {actual.data()};
          uint32_t *d[] = {actual.data()};
          constexpr uint32_t exec_mask = 0xaaaaaabfU;
          const uint64_t mode =
              uint64_t(omod) * RJ_ARITH_ALU_OMOD_2 | (clamp ? RJ_ARITH_ALU_CLAMP : 0);
          uint32_t status = 0x55;
          const bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
          ASSERT_EQ(call(cpu | semantics, exec_mask, mode, d, a, &status),
                    strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
          EXPECT_EQ(status, 0x55U);
          for (int lane = 0; lane < 32; ++lane) {
            if (strict || !(exec_mask & (1U << lane))) {
              EXPECT_EQ(actual[lane], input[lane]);
              continue;
            }
            uint32_t expected;
            if (lane == 1) {
              expected = clamp ? 0 : input[lane] | (Half ? 0x200U : 0x400000U);
            } else {
              const float x = Half ? float(half_value(uint16_t(input[lane])))
                                   : std::bit_cast<float>(input[lane]);
              float y = std::tanh(x);
              if constexpr (Half) {
                y = float(half_value(half_bits(y)));
              }
              if (omod) {
                const float minimum = Half ? 0x1p-14f : 0x1p-126f;
                y = std::abs(y) < minimum ? 0.0f : y * (omod == 3 ? 0.5f : omod == 2 ? 4.0f : 2.0f);
                if (std::abs(y) < minimum) {
                  y = std::copysign(0.0f, y);
                }
              }
              if (clamp) {
                y = y <= 0 ? 0 : y > 1 ? 1 : y;
              }
              expected = Half ? half_bits(y) : std::bit_cast<uint32_t>(y);
            }
            if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY && lane != 1) {
              const double got =
                  Half ? half_value(uint16_t(actual[lane])) : std::bit_cast<float>(actual[lane]);
              const double want =
                  Half ? half_value(uint16_t(expected)) : std::bit_cast<float>(expected);
              EXPECT_NEAR(got, want, Half ? 0.002 : 2e-6) << "lane " << lane;
            } else {
              EXPECT_EQ(actual[lane], expected) << "lane " << lane;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(call(0, 0, 0, nullptr, nullptr, nullptr), RJ_ARITH_SUCCESS);
  EXPECT_EQ(call(0, 0, 1ULL << 63, nullptr, nullptr, nullptr), RJ_ARITH_ERROR_INVALID_FLAGS);
}
} // namespace

// Validate F16 narrowing before OMOD, NaNs, aliases, masks and semantics.
TEST(Gfx1250Tanh, F16) { check<true>(); }
// Validate the FP32 libm model with the same output and storage policies.
TEST(Gfx1250Tanh, F32) { check<false>(); }

// Bound approximation error on a dense grid spanning the nonsaturated domain.
TEST(Gfx1250Tanh, LooseAccuracy) {
  for (uint64_t cpu : supported_cpu_levels()) {
    double max_error = 0;
    for (int first = 0; first < 65536; first += 32) {
      uint32_t input[32], output[32];
      for (int lane = 0; lane < 32; ++lane) {
        input[lane] = std::bit_cast<uint32_t>(float(first + lane - 32768) / 2048.0f);
      }
      const uint32_t *a[] = {input};
      uint32_t *d[] = {output};
      ASSERT_EQ(rj_arith_v_tanh_f32_w32_gfx1250(cpu, UINT32_MAX, 0, d, a, nullptr),
                RJ_ARITH_SUCCESS);
      for (int lane = 0; lane < 32; ++lane) {
        const float reference = std::tanh(std::bit_cast<float>(input[lane]));
        max_error =
            std::max(max_error, std::abs(double(std::bit_cast<float>(output[lane])) - reference));
      }
    }
    EXPECT_LT(max_error, 3e-7) << "cpu " << cpu;
  }
}
