// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 FP64 WMMA register mapping, modifiers and aliasing under the inherited FMA model.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Storage = Registers<40, 32>;

// Read a double from two logical register words, independently of their allocation order.
double read(const Storage &r, int reg, int lane) {
  return std::bit_cast<double>(uint64_t(r.v[reg][lane]) | (uint64_t(r.v[reg + 1][lane]) << 32));
}

TEST(WmmaF64Gfx1251, Arithmetic) {
  Storage input;
  std::minstd_rand random;
  for (int reg = 0; reg < 40; reg += 2) {
    for (int lane = 0; lane < 32; ++lane) {
      const auto first_draw = random();
      uint64_t bits = (uint64_t(first_draw >> 5) << 26) | (random() >> 5);
      bits |= uint64_t(1020 + (random() >> 28)) << 52;
      bits |= uint64_t(random() >> 30) << 63;
      input.v[reg][lane] = uint32_t(bits);
      input.v[reg + 1][lane] = uint32_t(bits >> 32);
    }
  }
  for (int reg : {0, 4, 8}) {
    input.v[reg][0] = 1;
    input.v[reg + 1][0] = 0x7ff80000U;
    input.v[reg][1] = 0;
    input.v[reg + 1][1] = 0x7ff00000U;
  }
  const auto cpus = supported_cpu_levels();
  for (uint64_t modifiers : {0U, 1U, 2U, 4U, 32U, 36U, 39U}) {
    double expected[8][32];
    for (int row = 0; row < 16; ++row) {
      for (int col = 0; col < 16; ++col) {
        const int reg = 2 * (row % 8), lane = col + 16 * (row / 8);
        double acc = read(input, 8 + reg, lane);
        if (modifiers & RJ_ARITH_WMMA_ABS_C) {
          acc = std::abs(acc);
        }
        if (modifiers & RJ_ARITH_WMMA_NEG_C) {
          acc = -acc;
        }
        for (int k = 0; k < 4; ++k) {
          double a = read(input, 2 * (k % 2), row + 16 * (k / 2));
          double b = read(input, 4 + 2 * (k % 2), col + 16 * (k / 2));
          acc = std::fma((modifiers & RJ_ARITH_WMMA_NEG_LO_A) ? -a : a,
                         (modifiers & RJ_ARITH_WMMA_NEG_LO_B) ? -b : b, acc);
        }
        expected[reg / 2][lane] = acc;
      }
    }
    for (uint64_t cpu : cpus) {
      for (uint64_t semantics : semantics_modes()) {
        for (int aliasing_offset : {0, 4, 8, 24}) {
          auto storage = input;
          const uint32_t *a[4], *b[4], *c[16];
          uint32_t *d[16];
          for (int reg = 0; reg < 4; ++reg) {
            a[reg] = storage.v[reg];
            b[reg] = storage.v[4 + reg];
          }
          for (int reg = 0; reg < 16; ++reg) {
            c[reg] = storage.v[8 + reg];
            d[reg] = storage.v[aliasing_offset + reg];
          }
          const int status =
              rj_arith_v_wmma_f64_16x16x4_f64_w32(cpu | semantics, modifiers, d, a, b, c);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            storage.expect_preserved(input, 0, 0);
            continue;
          }
          ASSERT_EQ(status, RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < 16; reg += 2) {
            for (int lane = 0; lane < 32; ++lane) {
              const double actual = read(storage, aliasing_offset + reg, lane);
              const double reference = expected[reg / 2][lane];
              if (std::isnan(reference)) {
                EXPECT_TRUE(std::isnan(actual));
              } else if (std::isinf(reference)) {
                EXPECT_EQ(actual, reference);
              } else {
                EXPECT_NEAR(actual, reference, 1e-12 + 1e-12 * std::abs(reference));
              }
            }
          }
          storage.expect_preserved(input, aliasing_offset, 16);
        }
      }
    }
  }
}

TEST(WmmaF64Gfx1251, ApiContract) {
  expect_invalid_flags(
      [](uint64_t flags, uint64_t modifiers) {
        return rj_arith_v_wmma_f64_16x16x4_f64_w32(flags, modifiers, nullptr, nullptr, nullptr,
                                                   nullptr);
      },
      RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B | RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_ABS_C);
}
} // namespace
