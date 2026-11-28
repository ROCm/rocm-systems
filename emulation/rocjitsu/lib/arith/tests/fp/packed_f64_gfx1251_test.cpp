// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP64 arithmetic/selection under inherited semantics, including masked aliases and errors.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace rocjitsu::arith::test;
using Storage = Registers<16, 32>;
// Read one double from independently allocated low/high VGPR planes.
double read(const Storage &r, int reg, int lane) {
  return std::bit_cast<double>(uint64_t(r.v[reg][lane]) | (uint64_t(r.v[reg + 1][lane]) << 32));
}
// Bind the entry point's arity while keeping operand/semantics coverage in one runner.
template <int Op>
int invoke(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
           const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
           uint32_t *status) {
  if constexpr (Op == 0) {
    return rj_arith_v_pk_add_f64_w32(flags, exec_mask, mode, d, a, b, status);
  }
  if constexpr (Op == 1) {
    return rj_arith_v_pk_mul_f64_w32(flags, exec_mask, mode, d, a, b, status);
  }
  if constexpr (Op == 2) {
    return rj_arith_v_pk_fma_f64_w32(flags, exec_mask, mode, d, a, b, c, status);
  }
  if constexpr (Op == 3) {
    return rj_arith_v_pk_min_num_f64_w32(flags, exec_mask, mode, d, a, b, status);
  }
  if constexpr (Op == 4) {
    return rj_arith_v_pk_max_num_f64_w32(flags, exec_mask, mode, d, a, b, status);
  }
}
// Check the inherited nearest/preserved-denormal model; decoded tests cover guest MODE boundaries.
template <int Op> void arithmetic() {
  Storage input;
  std::minstd_rand random;
  for (int reg = 0; reg < 16; reg += 2) {
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
    input.v[reg][0] = 0x1234U + reg;
    input.v[reg + 1][0] = 0x7ff80000U;
    input.v[reg][1] = 0;
    input.v[reg + 1][1] = 0x7ff00000U;
    input.v[reg][2] = 0;
    input.v[reg + 1][2] = reg ? 0 : 0x80000000U;
  }
  if constexpr (Op == 2) {
    // The inherited FMA model prioritizes invalid 0*Inf over C's quiet NaN.
    input.v[0][3] = input.v[1][3] = input.v[4][3] = 0;
    input.v[5][3] = 0x7ff00000U;
    input.v[8][3] = 0xabcdU;
    input.v[9][3] = 0x7ff80000U;
  }
  const auto cpus = supported_cpu_levels();
  for (uint64_t mode : {0U, RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_HI_B, RJ_ARITH_PK_CLAMP,
                        RJ_ARITH_PK_CLAMP | RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_HI_B}) {
    double expected[2][32];
    for (int element = 0; element < 2; ++element) {
      for (int lane = 0; lane < 32; ++lane) {
        const uint32_t signs = uint32_t(mode >> (element * 3));
        double x = read(input, element * 2, lane), y = read(input, 4 + element * 2, lane),
               z = read(input, 8 + element * 2, lane);
        if (signs & 1) {
          x = -x;
        }
        if (signs & 2) {
          y = -y;
        }
        double value;
        if constexpr (Op == 0) {
          value = x + y;
        }
        if constexpr (Op == 1) {
          value = x * y;
        }
        if constexpr (Op == 2) {
          value = std::fma(x, y, z);
        }
        if constexpr (Op >= 3) {
          if (std::isnan(x)) {
            value = std::isnan(y) ? std::bit_cast<double>(0x7ff8000000000000ULL) : y;
          } else if (std::isnan(y)) {
            value = x;
          } else if (x == 0 && y == 0) {
            value = std::bit_cast<double>(
                Op == 3 ? (std::bit_cast<uint64_t>(x) | std::bit_cast<uint64_t>(y))
                        : (std::bit_cast<uint64_t>(x) & std::bit_cast<uint64_t>(y)));
          } else {
            value = Op == 3 ? std::fmin(x, y) : std::fmax(x, y);
          }
        }
        if (mode & RJ_ARITH_PK_CLAMP) {
          value = std::isnan(value) || value <= 0 ? 0 : value > 1 ? 1 : value;
        }
        expected[element][lane] = value;
      }
    }
    for (uint64_t cpu : cpus) {
      for (uint64_t semantics : semantics_modes()) {
        for (uint32_t exec_mask : {0U, 1U, 0x80000000U, 0x55aa55aaU, 0xffffffffU}) {
          for (int aliasing_offset : {0, 1, 4, 8, 12}) {
            auto storage = input;
            const uint32_t *x[4], *y[4], *z[4];
            uint32_t *d[4];
            for (int reg = 0; reg < 4; ++reg) {
              x[reg] = storage.v[reg];
              y[reg] = storage.v[4 + reg];
              z[reg] = storage.v[8 + reg];
              d[reg] = storage.v[aliasing_offset + reg];
            }
            uint32_t status = 0xbadc0deU;
            int error = invoke<Op>(cpu | semantics, exec_mask, mode, d, x, y, z, &status);
            ASSERT_EQ(status, 0xbadc0deU);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              ASSERT_EQ(error, semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                   ? RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS
                                   : RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              storage.expect_preserved(input, 0, 0);
              continue;
            }
            ASSERT_EQ(error, RJ_ARITH_SUCCESS);
            for (int element = 0; element < 2; ++element) {
              for (int lane = 0; lane < 32; ++lane) {
                double value = read(storage, aliasing_offset + 2 * element, lane);
                double reference = exec_mask >> lane & 1
                                       ? expected[element][lane]
                                       : read(input, aliasing_offset + 2 * element, lane);
                if (std::isnan(reference)) {
                  EXPECT_TRUE(std::isnan(value));
                  if constexpr (Op < 3) {
                    // Compatibility retains A's payload even when packed arithmetic calls
                    // a public entry point whose Loose path permits native NaN selection.
                    if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && mode == 0 &&
                        (exec_mask & 1) && element == 0 && lane == 0) {
                      EXPECT_EQ(std::bit_cast<uint64_t>(value), 0x7ff8000000001234ULL);
                    }
                  }
                  if constexpr (Op == 2) {
                    if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY && mode == 0 &&
                        (exec_mask & 8) && element == 0 && lane == 3) {
                      EXPECT_EQ(std::bit_cast<uint64_t>(value), 0xfff8000000000000ULL);
                    }
                  }
                } else if (std::isinf(reference)) {
                  EXPECT_EQ(value, reference);
                } else if (reference == 0) {
                  EXPECT_EQ(std::bit_cast<uint64_t>(value), std::bit_cast<uint64_t>(reference));
                } else {
                  EXPECT_NEAR(value, reference, 1e-12 + 1e-12 * std::abs(reference));
                }
              }
            }
            storage.expect_preserved(input, aliasing_offset, 4);
          }
        }
      }
    }
  }
}
TEST(PackedF64Gfx1251, Add) { arithmetic<0>(); }
TEST(PackedF64Gfx1251, Multiply) { arithmetic<1>(); }
TEST(PackedF64Gfx1251, Fma) { arithmetic<2>(); }
TEST(PackedF64Gfx1251, MinimumNumber) { arithmetic<3>(); }
TEST(PackedF64Gfx1251, MaximumNumber) { arithmetic<4>(); }
// Check every unsupported modifier and reserved semantics before pointer access.
template <int Op> void contract() {
  constexpr uint64_t supported = RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B |
                                 RJ_ARITH_PK_NEG_HI_A | RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_CLAMP |
                                 (Op == 2 ? RJ_ARITH_PK_NEG_LO_C | RJ_ARITH_PK_NEG_HI_C : 0);
  expect_invalid_flags(
      [](uint64_t flags, uint64_t mode) {
        return invoke<Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, nullptr);
      },
      supported);
}
TEST(PackedF64Gfx1251, ApiContract) {
  contract<0>();
  contract<1>();
  contract<2>();
  contract<3>();
  contract<4>();
}
} // namespace
