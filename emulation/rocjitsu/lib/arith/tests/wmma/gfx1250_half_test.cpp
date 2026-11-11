// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 split-K layouts, accumulator controls and overlap across CPU levels.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_wmma_f32_16x16x32_f16_w32);

// Exercise one matrix instruction with common data, including special lanes and
// aliases.
template <bool Bf16, bool Packed = false, bool Mixed = false> void arithmetic(Entry entry) {
  constexpr bool PackedC = Packed && !Mixed;
  constexpr bool DotFirst = Bf16 && !PackedC;
  constexpr float inline_value = 0x1.45f306p-3f;
  std::array<std::array<uint32_t, 32>, 32> input{};
  std::minstd_rand random;
  for (int reg = 0; reg < 24; ++reg) {
    for (auto &word : input[reg]) {
      if (reg < 16 || PackedC) {
        auto half = [&] { return Bf16 ? random_bf16_bits(random) : random_f16_bits(random); };
        word = uint32_t(half()) | uint32_t(half()) << 16;
      } else {
        word = random_f32_bits(random);
      }
    }
  }
  for (int reg : {0, 8}) {
    input[reg][0] = Bf16 ? kBF16QuietNaNBits : kF16QuietNaNBits;
    input[reg][1] = Bf16 ? kBF16PositiveInfinityBits : kF16PositiveInfinityBits;
  }
  // At (row 3, column 7), a dot of 32 ones survives addition to 2^25,
  // whereas adding each product to C individually loses every one to rounding.
  // This distinguishes BF16's dot-before-C policy from FP16's C-first
  // reduction.
  uint32_t ones = Bf16 ? 0x3f803f80U : 0x3c003c00U;
  for (int reg = 0; reg < 8; ++reg) {
    input[reg][3] = input[reg][19] = ones;
    input[8 + reg][7] = input[8 + reg][23] = ones;
  }
  if constexpr (!PackedC) {
    input[16 + 3][7] = std::bit_cast<uint32_t>(0x1p25f);
  }
  const auto levels = supported_cpu_levels();
  std::vector<uint64_t> variants{0, RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C |
                                        RJ_ARITH_WMMA_NEG_LO_A};
  if constexpr (PackedC) {
    variants.push_back(RJ_ARITH_WMMA_INLINE_C);
  }
  for (uint64_t controls : variants) {
    for (uint64_t semantics : semantics_modes()) {
      const bool compatibility = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY;
      float expected[16][16];
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          int reg = row % 8, lane = col + 16 * (row / 8);
          float initial;
          if constexpr (PackedC) {
            uint16_t half = uint16_t(input[16 + reg / 2][lane] >> (16 * (row % 2)));
            initial =
                controls & RJ_ARITH_WMMA_INLINE_C
                    ? (compatibility ? inline_value
                       : Bf16 ? bf16_value(uint16_t(std::bit_cast<uint32_t>(inline_value) >> 16))
                              : float(half_value(half_bits(inline_value))))
                : Bf16 ? bf16_value(half)
                       : float(half_value(half));
          } else {
            initial = std::bit_cast<float>(input[16 + reg][lane]);
            if (controls & RJ_ARITH_WMMA_ABS_C) {
              initial = std::fabs(initial);
            }
            if (controls & RJ_ARITH_WMMA_NEG_C) {
              initial = -initial;
            }
          }
          if constexpr (PackedC) {
            if (!compatibility) {
              if (controls & RJ_ARITH_WMMA_ABS_C) {
                initial = std::fabs(initial);
              }
              if (controls & RJ_ARITH_WMMA_NEG_C) {
                initial = -initial;
              }
            }
          }
          float sum = DotFirst ? 0.0f : initial;
          auto read = [&](int base, int index, int k) {
            // Two K blocks per half-wave, with another pair in VGPRs 4..7.
            int source_reg = 4 * (k / 16) + (k % 8) / 2;
            int source_lane = index + 16 * (k / 8 % 2);
            uint16_t half = uint16_t(input[base + source_reg][source_lane] >> (16 * (k % 2)));
            return Bf16 ? std::bit_cast<float>(uint32_t(half) << 16) : float(half_value(half));
          };
          for (int k = 0; k < 32; ++k) {
            sum = std::fma(read(0, row, k), read(8, col, k), sum);
          }
          float value = DotFirst ? sum + initial : sum;
          uint32_t bits = std::bit_cast<uint32_t>(value);
          uint16_t bf16 = uint16_t(bits >> 16);
          if (!compatibility && std::isfinite(value)) {
            bf16 += (bits & 0xffffU) > 0x8000U || ((bits & 0xffffU) == 0x8000U && (bf16 & 1));
          }
          expected[row][col] =
              Packed ? (Bf16 ? bf16_value(bf16) : float(half_value(half_bits(value)))) : value;
        }
      }
      for (uint64_t cpu : levels) {
        for (int aliasing_offset : {0, 8, 16, 24}) {
          auto storage = input;
          if (controls & RJ_ARITH_WMMA_INLINE_C) {
            storage[16][0] = compatibility ? std::bit_cast<uint32_t>(inline_value)
                             : Bf16        ? uint16_t(std::bit_cast<uint32_t>(inline_value) >> 16)
                                           : half_bits(inline_value);
          }
          const auto before = storage;
          const uint32_t *a[8], *b[8], *c[8];
          uint32_t *d[8];
          for (int reg = 0; reg < 8; ++reg) {
            a[reg] = storage[reg].data();
            b[reg] = storage[8 + reg].data();
            c[reg] = storage[16 + reg].data();
            d[reg] = storage[aliasing_offset + reg].data();
          }
          int status = entry(cpu | semantics, controls, d, a, b, c);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(storage, before);
            continue;
          }
          ASSERT_EQ(status, RJ_ARITH_SUCCESS);
          for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 16; ++col) {
              int reg = row % 8, lane = col + 16 * (row / 8);
              uint32_t actual;
              if constexpr (Packed) {
                uint16_t half = uint16_t(d[reg / 2][lane] >> (16 * (row % 2)));
                actual = std::bit_cast<uint32_t>(Bf16 ? bf16_value(half) : float(half_value(half)));
              } else {
                actual = d[reg][lane];
              }
              // One final destination-format ULP allows the inherited
              // fused/unfused difference.
              expect_f32(actual, expected[row][col], RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/Packed ? (Bf16 ? 0.008 : 0.001) : 1e-5,
                         /*rtol=*/Packed ? (Bf16 ? 0.008 : 0.001) : 1e-5);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

// These selected constants distinguish full-precision inline C from prematurely
// packed C.
template <bool Bf16> void inline_precision(Entry entry) {
  for (uint64_t cpu : supported_cpu_levels()) {
    std::array<std::array<uint32_t, 32>, 28> words{};
    const uint32_t *a[8], *b[8], *c[4];
    uint32_t *d[4];
    for (int reg = 0; reg < 8; ++reg) {
      a[reg] = words[reg].data();
      b[reg] = words[8 + reg].data();
    }
    for (int reg = 0; reg < 4; ++reg) {
      c[reg] = words[16 + reg].data();
      d[reg] = words[24 + reg].data();
    }
    words[0][0] = Bf16 ? 0x3800U : 0x0440U; // 2^-15 or 0x1.1p-14.
    words[8][0] = Bf16 ? 0x3f80U : 0x3c00U; // 1.
    words[16][0] = std::bit_cast<uint32_t>(0x1.45f306p-3f);
    ASSERT_EQ(entry(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, RJ_ARITH_WMMA_INLINE_C, d, a, b, c),
              RJ_ARITH_SUCCESS);
    uint16_t expected = Bf16 ? 0x3e23U : half_bits(0x1.46p-3);
    EXPECT_EQ(uint16_t(d[0][0]), expected);
  }
}

// Selected legacy BF16F32 cases: operand NaN precedence, quieting and Inf*0.
void mixed_nan_priority() {
  struct Case {
    uint16_t a, b;
    uint32_t c;
    uint16_t expected;
  };
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto test :
         {Case{0x7f81, 0x7fc2, 0x7fc30000U, 0x7fc1}, Case{0x3f80, 0x7fc2, 0x7fc30000U, 0x7fc2},
          Case{0, 0, 0x7fc30000U, 0x7fc3}, Case{0x7f80, 0, 0, 0xffc0}}) {
      std::array<std::array<uint32_t, 32>, 24> words{};
      const uint32_t *a[8], *b[8], *c[8];
      uint32_t *d[4];
      for (int reg = 0; reg < 8; ++reg) {
        a[reg] = words[reg].data();
        b[reg] = words[8 + reg].data();
        c[reg] = words[16 + reg].data();
        if (reg < 4) {
          d[reg] = words[16 + reg].data();
        }
      }
      words[0][0] = test.a;
      words[8][0] = test.b;
      words[16][0] = test.c;
      ASSERT_EQ(rj_arith_v_wmma_bf16f32_16x16x32_bf16_w32(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 0,
                                                          d, a, b, c),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(uint16_t(d[0][0]), test.expected);
    }
  }
}

// Exact FP32 sums isolate final packing from accumulation order and SIMD width.
TEST(WmmaGfx1250Half, FinalPacking) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (uint64_t semantics :
         {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
      bool compatibility = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY;
      for (int form = 0; form < 3; ++form) {
        for (bool large : {false, true}) {
          std::array<std::array<uint32_t, 32>, 28> words{};
          const uint32_t *a[8], *b[8], *c[8];
          uint32_t *d[4];
          for (int reg = 0; reg < 8; ++reg) {
            a[reg] = words[reg].data();
            b[reg] = words[8 + reg].data();
            c[reg] = words[16 + reg].data();
            if (reg < 4) {
              d[reg] = words[24 + reg].data();
            }
          }
          if (form == 0) {
            for (int reg = 0; reg < 16; ++reg) {
              words[reg].fill(0x5c005c00U);
            }
          } else {
            words[0][0] = large ? 0x7b00U : 0x3bc0U; // 2^119 or 3/512.
            words[8][0] = 0x3f80U;                   // 1.
            words[16][0] =
                form == 1 ? (large ? 0x7f7fU : 0x3f80U) : (large ? 0x7f7f0000U : 0x3f800000U);
          }
          Entry entry = form == 0   ? rj_arith_v_wmma_f16_16x16x32_f16_w32
                        : form == 1 ? rj_arith_v_wmma_bf16_16x16x32_bf16_w32
                                    : rj_arith_v_wmma_bf16f32_16x16x32_bf16_w32;
          for (bool saturate : {false, true}) {
            uint64_t flags = cpu | semantics | (saturate ? RJ_ARITH_FP16_OVFL : 0);
            ASSERT_EQ(entry(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
            uint16_t expected = form == 0 ? (saturate && !compatibility ? 0x7bff : 0x7c00)
                                : large   ? (compatibility || saturate ? 0x7f7f : 0x7f80)
                                          : (compatibility ? 0x3f80 : 0x3f81);
            EXPECT_EQ(uint16_t(d[0][0]), expected)
                << cpu << '/' << semantics << '/' << form << '/' << saturate;
          }
          if (form == 0) {
            // CDNA5 MODE Table 7 saturation includes signed infinities, from
            // either C or a product. This is not a physical gfx1250 capture.
            for (bool product : {false, true}) {
              for (int reg = 0; reg < 16; ++reg) {
                words[reg].fill(product ? (reg < 8 ? 0x7c007c00U : 0x3c003c00U) : 0);
              }
              for (int reg = 16; reg < 20; ++reg) {
                words[reg].fill(product ? 0 : 0xfc007c00U);
              }
              for (bool saturate : {false, true}) {
                auto flags = cpu | semantics | (saturate ? RJ_ARITH_FP16_OVFL : 0);
                ASSERT_EQ(entry(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
                auto expected = saturate && !compatibility ? 0xfbff7bffU : 0xfc007c00U;
                // The C case covers both signs; products here are positive infinity.
                if (!product) {
                  EXPECT_EQ(d[0][0], expected) << cpu << '/' << semantics;
                } else {
                  EXPECT_EQ(uint16_t(d[0][0]), uint16_t(expected)) << cpu << '/' << semantics;
                }
              }
            }
          }
          if (form != 0) {
            // Finite operands can already overflow the FP32 accumulator.
            words[0][0] = 0x7f7fU;
            words[8][0] = 0x4000U;
            words[16][0] = 0;
            for (bool saturate : {false, true}) {
              auto flags = cpu | semantics | (saturate ? RJ_ARITH_FP16_OVFL : 0);
              ASSERT_EQ(entry(flags, 0, d, a, b, c), RJ_ARITH_SUCCESS);
              EXPECT_EQ(uint16_t(d[0][0]), saturate && !compatibility ? 0x7f7fU : 0x7f80U)
                  << cpu << '/' << semantics << '/' << form;
            }
          }
          if (form == 1 && compatibility) {
            // max_bf16 * 2 - max_bf16 stays finite only with a fused update.
            words[0][0] = 0x7f7fU;
            words[8][0] = 0x4000U;
            words[16][0] = 0xff7fU;
            ASSERT_EQ(entry(cpu | semantics, 0, d, a, b, c), RJ_ARITH_SUCCESS);
            EXPECT_EQ(uint16_t(d[0][0]), 0x7f7fU) << cpu;
          }
        }
      }
    }
  }
}

TEST(WmmaGfx1250Half, F16Arithmetic) { arithmetic<false>(rj_arith_v_wmma_f32_16x16x32_f16_w32); }
TEST(WmmaGfx1250Half, Bf16Arithmetic) { arithmetic<true>(rj_arith_v_wmma_f32_16x16x32_bf16_w32); }
TEST(WmmaGfx1250Half, PackedF16Arithmetic) {
  arithmetic<false, true>(rj_arith_v_wmma_f16_16x16x32_f16_w32);
  inline_precision<false>(rj_arith_v_wmma_f16_16x16x32_f16_w32);
}
TEST(WmmaGfx1250Half, PackedBf16Arithmetic) {
  arithmetic<true, true>(rj_arith_v_wmma_bf16_16x16x32_bf16_w32);
  inline_precision<true>(rj_arith_v_wmma_bf16_16x16x32_bf16_w32);
}
TEST(WmmaGfx1250Half, MixedBf16F32Arithmetic) {
  arithmetic<true, true, true>(rj_arith_v_wmma_bf16f32_16x16x32_bf16_w32);
  mixed_nan_priority();
}
} // namespace
