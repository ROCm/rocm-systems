// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse matrix API policy, SIMD parity and aliasing; decoded tests cover physical layout
// independently.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/registers.h"
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
            for (int reg = 0; reg < Count; ++reg) {
              for (int lane = 0; lane < 32; ++lane) {
                if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY || reg < aliasing_offset ||
                    reg >= aliasing_offset + D) {
                  EXPECT_EQ(actual[reg][lane], expected[reg][lane]);
                } else if constexpr (Form == 0) {
                  expect_f32(actual[reg][lane], std::bit_cast<float>(expected[reg][lane]),
                             RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-5, /*rtol=*/1e-5);
                } else {
                  for (int shift : {0, 16}) {
                    auto decode = [](uint16_t bits) {
                      return Form == 1 ? f16_value(bits) : bf16_value(bits);
                    };
                    expect_f32(
                        std::bit_cast<uint32_t>(decode(uint16_t(actual[reg][lane] >> shift))),
                        decode(uint16_t(expected[reg][lane] >> shift)), RJ_ARITH_SEMANTICS_LOOSE,
                        /*atol=*/1e-5, /*rtol=*/Form == 1 ? 0.001 : 0.008);
                  }
                }
              }
            }
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
// Compatibility fixes NaN payloads; Loose only promises a NaN classification.
void expect_nan_word(uint32_t actual, uint32_t expected, uint64_t semantics, bool packed = false) {
  if (semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    EXPECT_EQ(actual, expected);
  } else {
    EXPECT_TRUE(std::isnan(packed ? bf16_value(uint16_t(actual)) : std::bit_cast<float>(actual)));
    if (packed) {
      EXPECT_TRUE(std::isnan(bf16_value(uint16_t(actual >> 16))));
    }
  }
}

TEST(SwmmacGfx1250Float, SelectedNanPriorityAndPackedPayload) {
  constexpr int A = 0, B = 8, Index = 24, D = 25;
  Registers<D + 8, 32> initial;
  for (int reg = A; reg < D + 8; ++reg) {
    std::fill_n(initial.v[reg], 32, reg < B ? 0x3f803f80U : 0u);
  }
  // The final K quartet uses A register 7 and B registers 14 and 15.
  // Row 0 selects NaNs in both inputs at the final term. Row 1's final B
  // NaN supersedes its previous A NaN. Row 2 skips both B NaNs and keeps C.
  // Row 3 selects K=63 before K=60, so the earlier B NaN survives.
  initial.v[A + 7][16] = 0xff913f80U;
  initial.v[A + 7][17] = 0x3f807f93U;
  initial.v[B + 14][16] = 0x7f980000U;
  initial.v[B + 15][16] = 0xffa20000U;
  initial.v[Index][16] = 0xf0000000U;
  initial.v[Index][17] = 0xc0000000U;
  initial.v[Index][19] = 0x30000000U;
  initial.v[D + 2][0] = 0x7f805701U;
  // Row 4 column 1 adds opposite infinities with no NaN input. The finite
  // ties in column 2 round to the neighboring even BF16 in packed results.
  initial.v[B][1] = 0x00007f80U;
  initial.v[D + 4][1] = 0xff800000U;
  initial.v[D + 5][2] = 0x3f808000U;
  initial.v[D + 6][2] = 0x3f818000U;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
      auto full = initial;
      ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x64_bf16_w32(cpu | semantics, 0, full.v + D, full.v + A,
                                                        full.v + B, full.v + Index),
                RJ_ARITH_SUCCESS);
      expect_nan_word(full.v[D][0], 0xffd10000U, semantics);
      expect_nan_word(full.v[D + 1][0], 0xffe20000U, semantics);
      expect_nan_word(full.v[D + 2][0], 0x7fc05701U, semantics);
      expect_nan_word(full.v[D + 3][0], 0xffe20000U, semantics);
      expect_nan_word(full.v[D + 4][1], 0xffc00000U, semantics);

      auto packed = initial;
      ASSERT_EQ(rj_arith_v_swmmac_bf16f32_16x16x64_bf16_w32(
                    cpu | semantics, 0, packed.v + D, packed.v + A, packed.v + B, packed.v + Index),
                RJ_ARITH_SUCCESS);
      expect_nan_word(packed.v[D][0], 0xffe2ffd1U, semantics, true);
      expect_nan_word(packed.v[D + 1][0], 0xffe27fc1U, semantics, true);
      expect_nan_word((packed.v[D + 2][1] & 0xffffU) << 16, 0xffc00000U, semantics);
      EXPECT_EQ(packed.v[D + 2][2] >> 16, 0x3f80U);
      EXPECT_EQ(packed.v[D + 3][2] & 0xffffU, 0x3f82U);
    }
  }
}
TEST(SwmmacGfx1250Float, Bf8NanConversion) {
  constexpr int A = 0, B = 8, Index = 24, D = 26;
  Registers<D + 8, 32> initial;
  for (auto *reg : initial.v) {
    std::fill_n(reg, 32, 0u);
  }
  // The last term of row 0 selects a negative E5M2 signaling NaN in A.
  // Row 1 selects a positive B NaN with a different nonzero fraction.
  initial.v[A + 7][16] = 0xfd000000U;
  initial.v[B + 15][16] = 0x7f000000U;
  initial.v[Index + 1][17] = 0xc0000000U;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics : {RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_SEMANTICS_LOOSE}) {
      auto wave = initial;
      ASSERT_EQ(rj_arith_v_swmmac_f32_16x16x128_bf8_bf8_w32(cpu | semantics, 0, wave.v + D,
                                                            wave.v + A, wave.v + B, wave.v + Index),
                RJ_ARITH_SUCCESS);
      expect_nan_word(wave.v[D][0], 0xffc00000U, semantics);
      expect_nan_word(wave.v[D + 1][0], 0x7fc00000U, semantics);
    }
  }
}
} // namespace
