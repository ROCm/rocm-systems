// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed U64 arithmetic, mathematical saturation and masked overlapping register tuples.
#include "common/contracts.h"
#include "common/cpu.h"
#include "common/registers.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_integer.h"
#include <gtest/gtest.h>
#include <random>
namespace {
using namespace rocjitsu::arith::test;
using Storage = Registers<16, 32>;
// Assemble independent VGPR planes without assuming their physical allocation order.
uint64_t read(const Storage &r, int reg, int lane) {
  return uint64_t(r.v[reg][lane]) | (uint64_t(r.v[reg + 1][lane]) << 32);
}
// Exercise one operation with a wide-integer reference and shared alias/EXEC/CPU coverage.
template <int Op> void arithmetic() {
  Storage input;
  std::minstd_rand random;
  for (int reg = 0; reg < 16; ++reg) {
    for (int lane = 0; lane < 32; ++lane) {
      const auto first_draw = random();
      input.v[reg][lane] = (uint32_t(first_draw >> 7) << 8) | (random() >> 23);
    }
  }
  constexpr uint64_t edges[] = {0, 1, UINT64_MAX, 0x8000000000000000ULL, 0x7fffffffffffffffULL};
  for (int i = 0; i < 5; ++i) {
    for (int reg : {0, 2, 4, 6}) {
      uint64_t x = edges[(i + (reg / 2)) % 5];
      input.v[reg][i] = uint32_t(x);
      input.v[reg + 1][i] = uint32_t(x >> 32);
    }
  }
  if constexpr (Op == 2) {
    for (int reg = 4; reg < 6; ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        input.v[reg][lane] = (random() >> 24) % 5;
      }
    }
  }
  const auto cpus = supported_cpu_levels();
  for (uint64_t semantics : semantics_modes()) {
    for (uint64_t cpu : cpus) {
      for (uint32_t exec_mask : {0U, 1U, 0x80000000U, 0x55aa55aaU, 0xffffffffU}) {
        // Four independent sign bits and CLAMP are cheap to cover in the same runner.
        for (unsigned signs = 0; signs < (Op == 2 ? 1U : 16U); ++signs) {
          for (bool clamp : {false, true}) {
            if constexpr (Op == 2) {
              if (clamp) {
                continue;
              }
            }
            const uint64_t mode =
                (signs & 3) | ((signs & 12) << 1) | (clamp ? RJ_ARITH_PK_CLAMP : 0);
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
              int status;
              if constexpr (Op == 0) {
                status = rj_arith_v_pk_add_nc_u64_w32(cpu | semantics, exec_mask, mode, d, x, y);
              } else if constexpr (Op == 1) {
                status = rj_arith_v_pk_sub_nc_u64_w32(cpu | semantics, exec_mask, mode, d, x, y);
              } else {
                status =
                    rj_arith_v_pk_lshl_add_u64_w32(cpu | semantics, exec_mask, mode, d, x, y, z);
              }
              ASSERT_EQ(status, RJ_ARITH_SUCCESS);
              for (int element = 0; element < 2; ++element) {
                for (int lane = 0; lane < 32; ++lane) {
                  const int reg = element * 2;
                  uint64_t expected = read(input, aliasing_offset + reg, lane);
                  if (exec_mask >> lane & 1) {
                    uint64_t lhs = read(input, reg, lane), rhs = read(input, 4 + reg, lane);
                    if constexpr (Op == 2) {
                      expected = (lhs << input.v[4 + element][lane]) + read(input, 8 + reg, lane);
                    } else {
                      // Wide signed reference keeps mathematical source negation before saturation.
                      __int128_t left = lhs, right = rhs;
                      if (mode & (RJ_ARITH_PK_NEG_LO_A << (3 * element))) {
                        left = -left;
                      }
                      if (mode & (RJ_ARITH_PK_NEG_LO_B << (3 * element))) {
                        right = -right;
                      }
                      __int128_t value = Op == 0 ? left + right : left - right;
                      expected = clamp && value < 0                        ? 0
                                 : clamp && value > __int128_t(UINT64_MAX) ? UINT64_MAX
                                                                           : uint64_t(value);
                    }
                  }
                  ASSERT_EQ(read(storage, aliasing_offset + reg, lane), expected)
                      << "element=" << element << ", lane=" << lane << ", mode=" << mode;
                }
              }
              storage.expect_preserved(input, aliasing_offset, 4);
            }
          }
        }
      }
    }
  }
}
TEST(PackedU64Gfx1251, Add) { arithmetic<0>(); }
TEST(PackedU64Gfx1251, Subtract) { arithmetic<1>(); }
TEST(PackedU64Gfx1251, ShiftAdd) { arithmetic<2>(); }
TEST(PackedU64Gfx1251, ApiContract) {
  constexpr uint64_t binary = RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B | RJ_ARITH_PK_NEG_HI_A |
                              RJ_ARITH_PK_NEG_HI_B | RJ_ARITH_PK_CLAMP;
  expect_invalid_flags(
      [](uint64_t f, uint64_t m) {
        return rj_arith_v_pk_add_nc_u64_w32(f, 0, m, nullptr, nullptr, nullptr);
      },
      binary);
  expect_invalid_flags(
      [](uint64_t f, uint64_t m) {
        return rj_arith_v_pk_sub_nc_u64_w32(f, 0, m, nullptr, nullptr, nullptr);
      },
      binary);
  expect_invalid_flags(
      [](uint64_t f, uint64_t m) {
        return rj_arith_v_pk_lshl_add_u64_w32(f, 0, m, nullptr, nullptr, nullptr, nullptr);
      },
      0);
}
} // namespace
