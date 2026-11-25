// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 BF16 math checks inherited FP32 behavior, truncation, modifiers and storage.

#include "common/cpu.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Call = int (*)(uint64_t, uint32_t, uint64_t, uint32_t *const *, const uint32_t *const *,
                     uint32_t *);

// Compare inherited numeric results with the established FP32 API and explicit truncation.
void check(Call call, Call reference) {
  std::minstd_rand random;
  std::array<uint32_t, 32> input;
  for (auto &word : input) {
    word = uint32_t(random() >> 15);
  }
  input[0] = 0x4040; // Reciprocal of three distinguishes truncation from BF16 RNE.
  input[6] = 0x7f81;
  input[2] = 0x7f80;
  input[8] = 0x8000;
  input[4] = 1;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      for (uint64_t mode : std::array<uint64_t, 3>{0, RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_OMOD_2,
                                                   RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_CLAMP}) {
        std::array<uint32_t, 32> wide, result{};
        for (int lane = 0; lane < 32; ++lane) {
          wide[lane] = input[lane] << 16;
          if (mode & RJ_ARITH_ALU_ABS_A) {
            wide[lane] &= 0x7fffffffU;
          }
          if (mode & RJ_ARITH_ALU_NEG_A) {
            wide[lane] ^= 0x80000000U;
          }
        }
        const uint32_t *source[] = {wide.data()};
        uint32_t *dest[] = {result.data()};
        if (reference) {
          ASSERT_EQ(reference(cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, UINT32_MAX,
                              0, dest, source, nullptr),
                    RJ_ARITH_SUCCESS);
        } else {
          for (int lane = 0; lane < 32; ++lane) {
            result[lane] =
                (wide[lane] & 0x7fffffffU) > 0x7f800000U
                    ? wide[lane] | 0x400000U
                    : std::bit_cast<uint32_t>(std::tanh(std::bit_cast<float>(wide[lane])));
          }
        }
        for (bool high : {false, true}) {
          for (bool alias : {false, true}) {
            auto storage = input;
            for (auto &word : storage) {
              word = high ? (word << 16) | 0xa5a5U : word | 0xa5a50000U;
            }
            std::array<uint32_t, 32> output;
            output.fill(0xdeadbeefU);
            uint32_t *d[] = {alias ? storage.data() : output.data()};
            const uint32_t *a[] = {storage.data()};
            const auto before = alias ? storage : output;
            uint32_t status = 0x12345678;
            constexpr uint32_t exec_mask = 0x55555555U;
            ASSERT_EQ(call(cpu | semantics, exec_mask,
                           mode | (high ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0), d, a,
                           &status),
                      RJ_ARITH_SUCCESS);
            EXPECT_EQ(status, 0x12345678U);
            for (int lane = 0; lane < 32; ++lane) {
              float value = std::bit_cast<float>(result[lane]);
              if (mode & RJ_ARITH_ALU_OMOD_2) {
                value *= 2.0f;
              }
              if (mode & RJ_ARITH_ALU_CLAMP) {
                value = std::isnan(value) || value < 0 ? 0 : value > 1 ? 1 : value;
              }
              uint16_t half = uint16_t(std::bit_cast<uint32_t>(value) >> 16);
              if ((mode & RJ_ARITH_ALU_OMOD_MASK) && (half & 0x7fffU) < 0x80U) {
                half = 0;
              }
              uint32_t expected = exec_mask & (1U << lane)
                                      ? (before[lane] & (high ? 0xffffU : 0xffff0000U)) |
                                            (uint32_t(half) << (high ? 16 : 0))
                                      : before[lane];
              EXPECT_EQ(d[0][lane], expected) << "lane " << lane;
            }
          }
        }
      }
    }
  }
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    EXPECT_EQ(call(semantics, 0, 0, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
  }
  EXPECT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, 0, nullptr, nullptr, nullptr),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(call(0, 0, 1ULL << 63, nullptr, nullptr, nullptr), RJ_ARITH_ERROR_INVALID_FLAGS);
}
} // namespace

// Exercise rcp across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Rcp) { check(rj_arith_v_rcp_bf16_w32_gfx1250, rj_arith_v_rcp_f32_w32_gfx12); }

// Exercise rsq across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Rsq) { check(rj_arith_v_rsq_bf16_w32_gfx1250, rj_arith_v_rsq_f32_w32_gfx12); }

// Exercise sqrt across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Sqrt) {
  check(rj_arith_v_sqrt_bf16_w32_gfx1250, rj_arith_v_sqrt_f32_w32_gfx12);
}

// Exercise exp across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Exp) { check(rj_arith_v_exp_bf16_w32_gfx1250, rj_arith_v_exp_f32_w32_gfx12); }

// Exercise log across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Log) { check(rj_arith_v_log_bf16_w32_gfx1250, rj_arith_v_log_f32_w32_gfx12); }

// Exercise sin across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Sin) { check(rj_arith_v_sin_bf16_w32_gfx1250, rj_arith_v_sin_f32_w32_gfx12); }

// Exercise cos across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Cos) { check(rj_arith_v_cos_bf16_w32_gfx1250, rj_arith_v_cos_f32_w32_gfx12); }

// Exercise tanh across supported CPUs and available semantics.
TEST(Gfx1250Bf16Math, Tanh) { check(rj_arith_v_tanh_bf16_w32_gfx1250, nullptr); }

// A literal witness prevents accidentally replacing truncation with BF16 RNE.
TEST(Gfx1250Bf16Math, TruncatingReciprocal) {
  uint32_t input[32] = {0x4040}, output[32]{};
  const uint32_t *a[] = {input};
  uint32_t *d[] = {output};
  ASSERT_EQ(rj_arith_v_rcp_bf16_w32_gfx1250(RJ_ARITH_SEMANTICS_COMPATIBILITY, 1, 0, d, a, nullptr),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(output[0], 0x3eaaU);
}

// Exhaust the compact BF16 input domain against the scalar inherited model.
// Vary OMOD and selectors by block; the focused tests above cover aliases and EXEC.
TEST(Gfx1250Bf16Math, VectorModelParity) {
  const Call operations[] = {
      rj_arith_v_rcp_bf16_w32_gfx1250,  rj_arith_v_rsq_bf16_w32_gfx1250,
      rj_arith_v_sqrt_bf16_w32_gfx1250, rj_arith_v_exp_bf16_w32_gfx1250,
      rj_arith_v_log_bf16_w32_gfx1250,  rj_arith_v_sin_bf16_w32_gfx1250,
      rj_arith_v_cos_bf16_w32_gfx1250,
  };
  for (Call call : operations) {
    for (uint64_t cpu : supported_cpu_levels()) {
      if (cpu == RJ_ARITH_CPU_BASELINE) {
        continue;
      }
      for (int first = 0; first < 65536; first += 32) {
        uint32_t input[32], expected[32]{}, actual[32]{};
        const bool high = first & 32;
        const uint64_t mode = uint64_t((first / 64) & 3) * RJ_ARITH_ALU_OMOD_2 |
                              (high ? RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_D : 0);
        for (int lane = 0; lane < 32; ++lane) {
          input[lane] = uint32_t(first + lane) << (high ? 16 : 0);
        }
        const uint32_t *a[] = {input};
        uint32_t *d[] = {expected};
        ASSERT_EQ(call(RJ_ARITH_SEMANTICS_COMPATIBILITY, UINT32_MAX, mode, d, a, nullptr),
                  RJ_ARITH_SUCCESS);
        d[0] = actual;
        ASSERT_EQ(call(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, UINT32_MAX, mode, d, a, nullptr),
                  RJ_ARITH_SUCCESS);
        for (int lane = 0; lane < 32; ++lane) {
          ASSERT_EQ(actual[lane], expected[lane]) << "input " << first + lane << " cpu " << cpu;
        }
      }
    }
  }
}
