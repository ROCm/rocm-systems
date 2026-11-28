// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense MFMA FP8 format distinctions, source controls and whole-wave overlap.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64);

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

// Validate both format combinations and aliasing with the same deterministic input wave.
template <int M, bool Bf8A, bool Bf8B, bool Fnuz> void arithmetic(Entry entry) {
  constexpr int K = 512 / M, Outputs = M * M / 64;
  std::array<std::array<uint32_t, 64>, 4 + 2 * Outputs> input{};
  std::minstd_rand random;
  for (int reg = 0; reg < 4 + Outputs; ++reg) {
    for (auto &word : input[reg]) {
      if (reg >= 4) {
        word = random_f32_bits(random);
        continue;
      }
      for (int byte = 0; byte < 4; ++byte) {
        uint32_t code = random() >> 23;
        word |= ((code & 0x80) | ((code & 0x7f) % 0x70)) << (8 * byte);
      }
    }
  }
  input[0][0] = Fnuz ? 0x80 : Bf8A ? 0x7d : 0x7f;
  input[2][0] = Fnuz ? 0x80 : Bf8B ? 0x7d : 0x7f;
  // FNUZ 0xff is finite; the corresponding OCP BF8 code is a NaN.
  input[0][1] = input[2][1] = 0xff;
  input[5][5] = kF32QuietNaNBits;
  input[6][6] = kF32PositiveInfinityBits;
  const auto levels = supported_cpu_levels();
  for (uint64_t controls : {0ULL, 4ULL | (13ULL << 3) | (7ULL << 7), 7ULL}) {
    float expected[Outputs][64];
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        int reg = row / (4 * (64 / M)) * 4 + row % 4;
        int lane = row / 4 % (64 / M) * M + col;
        float sum = std::bit_cast<float>(input[4 + reg][lane]);
        auto read = [&](int base, int index, int k) {
          constexpr int per_lane = K / (64 / M);
          int source_lane = k / per_lane * M + index;
          int cbsz = int(controls & 7), blgp = int(controls >> 7 & 7);
          if (base == 0 && cbsz > 0 && cbsz <= 4) {
            int size = 64 >> cbsz;
            source_lane = source_lane % size + size * (int(controls >> 3 & 15) % (1 << cbsz));
          }
          if (base == 2 && blgp) {
            if (blgp <= 2) {
              source_lane = source_lane % 32 + (blgp - 1) * 32;
            } else if (blgp == 3) {
              source_lane = (source_lane + 16) % 64;
            } else {
              source_lane = source_lane % 16 + (blgp - 4) * 16;
            }
          }
          uint8_t code = uint8_t(input[base + k % per_lane / 4][source_lane] >> (8 * (k % 4)));
          return decode(code, base == 0 ? Bf8A : Bf8B, Fnuz);
        };
        for (int k = 0; k < K; ++k) {
          sum = std::fma(read(0, row, k), read(2, col, k), sum);
        }
        expected[reg][lane] = sum;
      }
    }
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        // Qualified gfx942 and gfx950 exact cases with default controls have captured
        // expectations in gfx942_exact_test and gfx950_exact_test.
        if (!controls && semantics != RJ_ARITH_SEMANTICS_LOOSE &&
            semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
          continue;
        }
        for (int aliasing_offset : {0, 2, 4, 4 + Outputs}) {
          auto storage = input;
          const uint32_t *a[2], *b[2], *c[Outputs];
          uint32_t *d[Outputs];
          for (int reg = 0; reg < 2; ++reg) {
            a[reg] = storage[reg].data();
            b[reg] = storage[2 + reg].data();
          }
          for (int reg = 0; reg < Outputs; ++reg) {
            c[reg] = storage[4 + reg].data();
            d[reg] = storage[aliasing_offset + reg].data();
          }
          int status = entry(cpu | semantics, controls, d, a, b, c);
          if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
              semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(storage, input);
            continue;
          }
          ASSERT_EQ(status, RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < Outputs; ++reg) {
            for (int lane = 0; lane < 64; ++lane) {
              expect_f32(d[reg][lane], expected[reg][lane], RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-5,
                         /*rtol=*/1e-5);
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 63, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(MfmaFp8, mfma_f32_16x16x32_fp8_fp8_w64) {
  arithmetic<16, false, false, true>(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64);
}
TEST(MfmaFp8, mfma_f32_16x16x32_fp8_fp8_w64_gfx950) {
  arithmetic<16, false, false, false>(rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_16x16x32_fp8_bf8_w64) {
  arithmetic<16, false, true, true>(rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64);
}
TEST(MfmaFp8, mfma_f32_16x16x32_fp8_bf8_w64_gfx950) {
  arithmetic<16, false, true, false>(rj_arith_v_mfma_f32_16x16x32_fp8_bf8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_16x16x32_bf8_fp8_w64) {
  arithmetic<16, true, false, true>(rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64);
}
TEST(MfmaFp8, mfma_f32_16x16x32_bf8_fp8_w64_gfx950) {
  arithmetic<16, true, false, false>(rj_arith_v_mfma_f32_16x16x32_bf8_fp8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_16x16x32_bf8_bf8_w64) {
  arithmetic<16, true, true, true>(rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64);
}
TEST(MfmaFp8, mfma_f32_16x16x32_bf8_bf8_w64_gfx950) {
  arithmetic<16, true, true, false>(rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_32x32x16_fp8_fp8_w64) {
  arithmetic<32, false, false, true>(rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64);
}
TEST(MfmaFp8, mfma_f32_32x32x16_fp8_fp8_w64_gfx950) {
  arithmetic<32, false, false, false>(rj_arith_v_mfma_f32_32x32x16_fp8_fp8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_32x32x16_fp8_bf8_w64) {
  arithmetic<32, false, true, true>(rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64);
}
TEST(MfmaFp8, mfma_f32_32x32x16_fp8_bf8_w64_gfx950) {
  arithmetic<32, false, true, false>(rj_arith_v_mfma_f32_32x32x16_fp8_bf8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_32x32x16_bf8_fp8_w64) {
  arithmetic<32, true, false, true>(rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64);
}
TEST(MfmaFp8, mfma_f32_32x32x16_bf8_fp8_w64_gfx950) {
  arithmetic<32, true, false, false>(rj_arith_v_mfma_f32_32x32x16_bf8_fp8_w64_gfx950);
}
TEST(MfmaFp8, mfma_f32_32x32x16_bf8_bf8_w64) {
  arithmetic<32, true, true, true>(rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64);
}
TEST(MfmaFp8, mfma_f32_32x32x16_bf8_bf8_w64_gfx950) {
  arithmetic<32, true, true, false>(rj_arith_v_mfma_f32_32x32x16_bf8_bf8_w64_gfx950);
}

TEST(MfmaFp8, ExhaustiveByteDecodeAndSubnormalCFallback) {
  // Verify every 8-bit code in [0, 255] across OCP/FNUZ FP8/BF8 against the independent
  // decoder, and verify subnormal-C preservation on all supported CPU levels.
  struct Variant {
    Entry entry;
    bool bf8, fnuz;
  };
  const Variant variants[] = {
      {rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64, false, true},
      {rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64, true, true},
      {rj_arith_v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950, false, false},
      {rj_arith_v_mfma_f32_16x16x32_bf8_bf8_w64_gfx950, true, false},
  };
  for (const Variant v : variants) {
    const uint8_t one_code = v.bf8 ? (v.fnuz ? 0x40 : 0x3c) : (v.fnuz ? 0x40 : 0x38);
    // Test 4 sweeps of 64 codes each so every 8-bit code is checked against decode().
    for (int sweep = 0; sweep < 4; ++sweep) {
      std::array<std::array<uint32_t, 64>, 2> a_buf{}, b_buf{};
      for (int lane = 0; lane < 64; ++lane) {
        const uint8_t code = uint8_t(sweep * 64 + lane);
        // Only byte 0 of reg 0 (k = 0, 8, 16, 24) is populated; remaining bytes are 0.
        a_buf[0][lane] = uint32_t(code);
        b_buf[0][lane] = uint32_t(one_code);
      }
      // Compute independent expected matrix: row r sums the 4 groups g in 0..3 at lane g*16+r.
      float expected[4][64]{};
      for (int row = 0; row < 16; ++row) {
        float sum = 0.0f;
        for (int g = 0; g < 4; ++g) {
          const uint8_t code = uint8_t(sweep * 64 + g * 16 + row);
          sum = std::fma(decode(code, v.bf8, v.fnuz), 1.0f, sum);
        }
        const int reg = row % 4;
        const int base_lane = (row / 4) * 16;
        for (int col = 0; col < 16; ++col) {
          expected[reg][base_lane + col] = sum;
        }
      }
      std::array<std::array<uint32_t, 64>, 4> zero_c{};
      const uint32_t *a_ptrs[] = {a_buf[0].data(), a_buf[1].data()};
      const uint32_t *b_ptrs[] = {b_buf[0].data(), b_buf[1].data()};
      const uint32_t *c_ptrs[] = {zero_c[0].data(), zero_c[1].data(), zero_c[2].data(),
                                  zero_c[3].data()};
      for (uint64_t cpu : supported_cpu_levels()) {
        std::array<std::array<uint32_t, 64>, 4> out{};
        uint32_t *d_ptrs[] = {out[0].data(), out[1].data(), out[2].data(), out[3].data()};
        ASSERT_EQ(v.entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0, d_ptrs, a_ptrs, b_ptrs, c_ptrs),
                  RJ_ARITH_SUCCESS);
        for (int r = 0; r < 4; ++r) {
          for (int lane = 0; lane < 64; ++lane) {
            expect_f32(out[r][lane], expected[r][lane], RJ_ARITH_SEMANTICS_LOOSE, /*atol=*/1e-5,
                       /*rtol=*/1e-5);
          }
        }
      }
    }
    // Verify subnormal C is preserved bit-for-bit when A and B are zero (vdpbf16ps fallback).
    std::array<std::array<uint32_t, 64>, 2> z_buf{};
    std::array<std::array<uint32_t, 64>, 4> c_buf{};
    for (int r = 0; r < 4; ++r) {
      for (int lane = 0; lane < 64; ++lane) {
        c_buf[r][lane] = (lane & 1) ? 0x80000001u : 0x00000001u;
      }
    }
    const uint32_t *z_ptrs[] = {z_buf[0].data(), z_buf[1].data()};
    const uint32_t *c_ptrs[] = {c_buf[0].data(), c_buf[1].data(), c_buf[2].data(), c_buf[3].data()};
    for (uint64_t cpu : supported_cpu_levels()) {
      std::array<std::array<uint32_t, 64>, 4> out{};
      uint32_t *d_ptrs[] = {out[0].data(), out[1].data(), out[2].data(), out[3].data()};
      ASSERT_EQ(v.entry(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0, d_ptrs, z_ptrs, z_ptrs, c_ptrs),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(out, c_buf);
    }
  }
}
} // namespace
