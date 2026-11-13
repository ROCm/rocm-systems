// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse matrix API policy, SIMD parity and aliasing; decoded tests cover physical layout
// independently.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_wmma.h"
#include <array>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_swmmac_f32_16x16x64_f16_w32);

// Compare every CPU path to the inherited scalar FMA sequence, with aliased tied inputs.
template <int Bits, int Form> void arithmetic(Entry entry) {
  constexpr int A = 8, B = 16, D = (Form == 0 || Form == 3) ? 8 : 4, Index = A + B,
                Dest = Index + 2, Count = Dest + D;
  std::array<std::array<uint32_t, 32>, Count> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = uint32_t(random() >> 15) << 16 | (random() >> 15);
    }
  }
  constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
  for (int reg = Index; reg < Dest; ++reg) {
    for (auto &word : input[reg]) {
      word = 0;
      for (int n = 0; n < 8; ++n) {
        word |= pairs[(random() >> 16) % 6] << (4 * n);
      }
    }
  }
  auto invoke = [&](uint64_t flags, auto &storage, int offset) {
    const uint32_t *a[A], *b[B], *index[] = {storage[Index].data(), storage[Index + 1].data()};
    uint32_t *d[D];
    for (int i = 0; i < A; ++i) {
      a[i] = storage[i].data();
    }
    for (int i = 0; i < B; ++i) {
      b[i] = storage[A + i].data();
    }
    for (int i = 0; i < D; ++i) {
      d[i] = storage[offset + i].data();
    }
    return entry(flags, 0, d, a, b, index);
  };
  const auto levels = supported_cpu_levels();
  for (uint64_t fp : std::array<uint64_t, 2>{0, RJ_ARITH_FP16_OVFL}) {
    for (int aliasing_offset : {0, A, Index, Dest}) {
      auto expected = input;
      ASSERT_EQ(invoke(fp | RJ_ARITH_SEMANTICS_COMPATIBILITY, expected, aliasing_offset),
                RJ_ARITH_SUCCESS);
      for (uint64_t cpu : levels) {
        for (uint64_t semantics : semantics_modes()) {
          auto actual = input;
          int status = invoke(cpu | semantics | fp, actual, aliasing_offset);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(actual, input);
          } else {
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            EXPECT_EQ(actual, expected);
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x64_f16_w32) {
  arithmetic<16, 0>(rj_arith_v_swmmac_f32_16x16x64_f16_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x64_bf16_w32) {
  arithmetic<16, 0>(rj_arith_v_swmmac_f32_16x16x64_bf16_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f16_16x16x64_f16_w32) {
  arithmetic<16, 1>(rj_arith_v_swmmac_f16_16x16x64_f16_w32);
}
TEST(SwmmacGfx1250Float, swmmac_bf16_16x16x64_bf16_w32) {
  arithmetic<16, 2>(rj_arith_v_swmmac_bf16_16x16x64_bf16_w32);
}
TEST(SwmmacGfx1250Float, swmmac_bf16f32_16x16x64_bf16_w32) {
  arithmetic<16, 3>(rj_arith_v_swmmac_bf16f32_16x16x64_bf16_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x128_fp8_fp8_w32) {
  arithmetic<8, 0>(rj_arith_v_swmmac_f32_16x16x128_fp8_fp8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f16_16x16x128_fp8_fp8_w32) {
  arithmetic<8, 1>(rj_arith_v_swmmac_f16_16x16x128_fp8_fp8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x128_fp8_bf8_w32) {
  arithmetic<8, 0>(rj_arith_v_swmmac_f32_16x16x128_fp8_bf8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f16_16x16x128_fp8_bf8_w32) {
  arithmetic<8, 1>(rj_arith_v_swmmac_f16_16x16x128_fp8_bf8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x128_bf8_fp8_w32) {
  arithmetic<8, 0>(rj_arith_v_swmmac_f32_16x16x128_bf8_fp8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f16_16x16x128_bf8_fp8_w32) {
  arithmetic<8, 1>(rj_arith_v_swmmac_f16_16x16x128_bf8_fp8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f32_16x16x128_bf8_bf8_w32) {
  arithmetic<8, 0>(rj_arith_v_swmmac_f32_16x16x128_bf8_bf8_w32);
}
TEST(SwmmacGfx1250Float, swmmac_f16_16x16x128_bf8_bf8_w32) {
  arithmetic<8, 1>(rj_arith_v_swmmac_f16_16x16x128_bf8_bf8_w32);
}
} // namespace
