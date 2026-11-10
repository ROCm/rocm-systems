// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 byte-input WMMA layouts, final narrowing and whole-wave overlap.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_wmma_f32_16x16x64_fp8_fp8_w32);

// Numeric decoder independent of the implementation's scalar/SIMD bit construction.
float decode(uint8_t code, bool bf8, bool fnuz) {
  if (fnuz && code == 0x80) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  int frac_bits = bf8 ? 2 : 3, bias = (bf8 ? 15 : 7) + int(fnuz);
  int exponent = (code & 0x7f) >> frac_bits, fraction = code & ((1 << frac_bits) - 1);
  if (!fnuz && exponent == (bf8 ? 31 : 15)) {
    if (bf8 && !fraction) {
      return std::copysign(std::numeric_limits<float>::infinity(), code & 0x80 ? -1.0f : 1.0f);
    }
    if (bf8 || fraction == 7) {
      return std::numeric_limits<float>::quiet_NaN();
    }
  }
  float magnitude = std::ldexp(float(exponent ? (1 << frac_bits) + fraction : fraction),
                               (exponent ? exponent : 1) - bias - frac_bits);
  return code & 0x80 ? -magnitude : magnitude;
}

// Exercise one instruction on all CPU levels with shared deterministic data and aliases.
template <int K, bool Bf8A, bool Bf8B, bool Packed> void arithmetic(Entry entry) {
  constexpr int Inputs = K / 8, Outputs = Packed ? 4 : 8;
  std::array<std::array<uint32_t, 32>, 2 * Inputs + 16> input{};
  std::minstd_rand random;
  for (int reg = 0; reg < 2 * Inputs + Outputs; ++reg) {
    for (auto &word : input[reg]) {
      if (reg >= 2 * Inputs) {
        word = Packed ? uint32_t(half_bits(0.125)) | uint32_t(half_bits(-0.25)) << 16
                      : random_f32_bits(random);
      } else {
        for (int byte = 0; byte < 4; ++byte) {
          uint32_t code = random() >> 23;
          word |= ((code & 0x80U) | ((code & 0x7fU) % 0x40U)) << (8 * byte);
        }
      }
    }
  }
  input[0][0] = Bf8A ? 0x7d : 0x7f;
  input[Inputs][0] = Bf8B ? 0x7d : 0x7f;
  // A finite overflowing dot distinguishes normal narrowing from FP16_OVFL saturation.
  for (int reg = 0; reg < Inputs; ++reg) {
    input[reg][3] = input[reg][19] = Bf8A ? 0x7b7b7b7bU : 0x7e7e7e7eU;
    input[Inputs + reg][7] = input[Inputs + reg][23] = Bf8B ? 0x7b7b7b7bU : 0x7e7e7e7eU;
  }
  // C contains one infinity and one NaN; both are retained by the reduction.
  input[2 * Inputs][1] = Packed ? 0x7c00U : kF32PositiveInfinityBits;
  input[2 * Inputs][2] = Packed ? 0x7e00U : kF32QuietNaNBits;
  const auto levels = supported_cpu_levels();
  for (int variant : {0, 1, 2}) {
    uint64_t controls = variant == 1             ? RJ_ARITH_WMMA_ABS_C | RJ_ARITH_WMMA_NEG_C
                        : variant == 2 && Packed ? RJ_ARITH_WMMA_INLINE_C
                                                 : 0;
    uint64_t fp = variant == 1 ? RJ_ARITH_FP16_OVFL : 0;
    for (uint64_t semantics : semantics_modes()) {
      float expected[16][16];
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          int lane = col + 16 * (row / 8), reg = row % 8;
          uint32_t word = input[2 * Inputs + (Packed ? reg / 2 : reg)][lane];
          float sum = Packed ? float(half_value(uint16_t(word >> (16 * (row % 2)))))
                             : std::bit_cast<float>(word);
          if (controls & RJ_ARITH_WMMA_INLINE_C) {
            sum = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY
                      ? 0x1.45f306p-3f
                      : float(half_value(half_bits(0x1.45f306p-3f)));
          }
          if ((!Packed || semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) &&
              (controls & RJ_ARITH_WMMA_ABS_C)) {
            sum = std::fabs(sum);
          }
          if ((!Packed || semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) &&
              (controls & RJ_ARITH_WMMA_NEG_C)) {
            sum = -sum;
          }
          auto read = [&](int base, int index, int k, bool bf8) {
            int src_reg = 4 * (k / 32) + (k % 16) / 4;
            int src_lane = index + 16 * ((k / 16) % 2);
            return decode(uint8_t(input[base + src_reg][src_lane] >> (8 * (k % 4))), bf8, false);
          };
          for (int k = 0; k < K; ++k) {
            sum = std::fma(read(0, row, k, Bf8A), read(Inputs, col, k, Bf8B), sum);
          }
          if constexpr (Packed) {
            if (fp && std::fabs(sum) > 65504.0f) {
              sum = std::copysign(65504.0f, sum);
            }
            sum = float(half_value(half_bits(sum)));
          }
          expected[row][col] = sum;
        }
      }
      for (uint64_t cpu : levels) {

        for (int aliasing_offset : {0, Inputs, 2 * Inputs, 2 * Inputs + 8}) {
          auto storage = input;
          if (controls & RJ_ARITH_WMMA_INLINE_C) {
            storage[2 * Inputs][0] = semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY
                                         ? std::bit_cast<uint32_t>(0x1.45f306p-3f)
                                         : half_bits(0x1.45f306p-3f);
          }
          auto before = storage;
          const uint32_t *a[Inputs], *b[Inputs], *c[Outputs];
          uint32_t *d[Outputs];
          for (int i = 0; i < Inputs; ++i) {
            a[i] = storage[i].data();
            b[i] = storage[Inputs + i].data();
          }
          for (int i = 0; i < Outputs; ++i) {
            c[i] = storage[2 * Inputs + i].data();
            d[i] = storage[aliasing_offset + i].data();
          }
          int status = entry(cpu | semantics | fp, controls, d, a, b, c);
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
              uint32_t actual =
                  Packed ? std::bit_cast<uint32_t>(
                               float(half_value(uint16_t(d[reg / 2][lane] >> (16 * (row % 2))))))
                         : d[reg][lane];
              expect_f32(actual, expected[row][col], RJ_ARITH_SEMANTICS_LOOSE,
                         /*atol=*/Packed ? 0.001 : 1e-4, /*rtol=*/Packed ? 0.001 : 1e-5);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
} // namespace
TEST(WmmaGfx1250Fp8, f32_64_fp8_fp8) {
  arithmetic<64, false, false, false>(rj_arith_v_wmma_f32_16x16x64_fp8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f16_64_fp8_fp8) {
  arithmetic<64, false, false, true>(rj_arith_v_wmma_f16_16x16x64_fp8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f32_64_fp8_bf8) {
  arithmetic<64, false, true, false>(rj_arith_v_wmma_f32_16x16x64_fp8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f16_64_fp8_bf8) {
  arithmetic<64, false, true, true>(rj_arith_v_wmma_f16_16x16x64_fp8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f32_64_bf8_fp8) {
  arithmetic<64, true, false, false>(rj_arith_v_wmma_f32_16x16x64_bf8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f16_64_bf8_fp8) {
  arithmetic<64, true, false, true>(rj_arith_v_wmma_f16_16x16x64_bf8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f32_64_bf8_bf8) {
  arithmetic<64, true, true, false>(rj_arith_v_wmma_f32_16x16x64_bf8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f16_64_bf8_bf8) {
  arithmetic<64, true, true, true>(rj_arith_v_wmma_f16_16x16x64_bf8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f32_128_fp8_fp8) {
  arithmetic<128, false, false, false>(rj_arith_v_wmma_f32_16x16x128_fp8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f16_128_fp8_fp8) {
  arithmetic<128, false, false, true>(rj_arith_v_wmma_f16_16x16x128_fp8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f32_128_fp8_bf8) {
  arithmetic<128, false, true, false>(rj_arith_v_wmma_f32_16x16x128_fp8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f16_128_fp8_bf8) {
  arithmetic<128, false, true, true>(rj_arith_v_wmma_f16_16x16x128_fp8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f32_128_bf8_fp8) {
  arithmetic<128, true, false, false>(rj_arith_v_wmma_f32_16x16x128_bf8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f16_128_bf8_fp8) {
  arithmetic<128, true, false, true>(rj_arith_v_wmma_f16_16x16x128_bf8_fp8_w32);
}
TEST(WmmaGfx1250Fp8, f32_128_bf8_bf8) {
  arithmetic<128, true, true, false>(rj_arith_v_wmma_f32_16x16x128_bf8_bf8_w32);
}
TEST(WmmaGfx1250Fp8, f16_128_bf8_bf8) {
  arithmetic<128, true, true, true>(rj_arith_v_wmma_f16_16x16x128_bf8_bf8_w32);
}
