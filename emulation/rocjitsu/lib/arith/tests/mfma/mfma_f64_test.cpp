// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP64 MFMA layout, sign controls, approximate arithmetic and whole-wave aliasing.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_mfma_f64_16x16x4_f64_w64);

// Exercise each supported CPU level and semantics using one deterministic register bank.
template <int M, int Batch> void arithmetic(Entry entry) {
  constexpr int Outputs = 2 * M * M * Batch / 64;
  std::array<std::array<uint32_t, 64>, 4 + 2 * Outputs> input;
  std::minstd_rand random;
  for (int reg = 0; reg < 4 + 2 * Outputs; reg += 2) {
    for (int lane = 0; lane < 64; ++lane) {
      uint64_t bits = (uint64_t(random() >> 5) << 26) | (random() >> 5);
      bits |= uint64_t(1020 + (random() >> 27) % 8) << 52;
      bits |= uint64_t(random() >> 30) << 63;
      input[reg][lane] = uint32_t(bits);
      input[reg + 1][lane] = uint32_t(bits >> 32);
    }
  }
  for (int reg : {0, 2, 4}) {
    input[reg][0] = 1;
    input[reg + 1][0] = 0x7ff80000U;
    input[reg][1] = 0;
    input[reg + 1][1] = 0x7ff00000U;
  }
  const auto levels = supported_cpu_levels();
  for (uint64_t controls : {0ULL, 1ULL, 2ULL, 4ULL, 7ULL}) {
    double expected[Outputs / 2][64];
    auto read = [&](int reg, int lane, bool neg) {
      double v = std::bit_cast<double>(uint64_t(input[reg][lane]) |
                                       (uint64_t(input[reg + 1][lane]) << 32));
      return neg ? -v : v;
    };
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          // Legacy output mapping uses contiguous row groups, unlike FP32 MFMA.
          int block_groups = (64 + M * M - 1) / (M * M);
          int reg = batch * (M * M / 64) + row / (64 / M);
          int lane = batch % block_groups * M + row % (64 / M) * block_groups * M + col;
          double sum = read(4 + 2 * reg, lane, controls & 4);
          for (int k = 0; k < 4; ++k) {
            sum = std::fma(read(0, batch * M + k * M * Batch + row, controls & 1),
                           read(2, batch * M + k * M * Batch + col, controls & 2), sum);
          }
          expected[reg][lane] = sum;
        }
      }
    }
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        for (int aliasing_offset : {0, 2, 4, 4 + Outputs}) {
          auto storage = input;
          const uint32_t *a[] = {storage[0].data(), storage[1].data()};
          const uint32_t *b[] = {storage[2].data(), storage[3].data()};
          const uint32_t *c[Outputs];
          uint32_t *d[Outputs];
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
          for (int reg = 0; reg < Outputs; reg += 2) {
            for (int lane = 0; lane < 64; ++lane) {
              double actual = std::bit_cast<double>(uint64_t(d[reg][lane]) |
                                                    (uint64_t(d[reg + 1][lane]) << 32));
              double reference = expected[reg / 2][lane];
              if (std::isnan(reference)) {
                EXPECT_TRUE(std::isnan(actual));
              } else if (std::isinf(reference)) {
                EXPECT_EQ(actual, reference);
              } else {
                // Loose arithmetic allows small numerical discrepancies.
                EXPECT_NEAR(actual, reference, 1e-10 + 1e-12 * std::abs(reference));
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 3, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
// Fused cancellation must survive both intermediate overflow and product rounding.
TEST(MfmaF64, FusedCompatibility) {
  for (uint64_t cpu : supported_cpu_levels()) {
    for (bool overflow : {false, true}) {
      uint32_t storage[20][64] = {};
      auto set = [&](int reg, int lane, double value) {
        uint64_t bits = std::bit_cast<uint64_t>(value);
        storage[reg][lane] = uint32_t(bits);
        storage[reg + 1][lane] = uint32_t(bits >> 32);
      };
      set(0, 0, overflow ? 0x1.fffffffffffffp1023 : 1 + 0x1p-27);
      set(2, 0, overflow ? 2 : 1 - 0x1p-27);
      set(4, 0, overflow ? -0x1.fffffffffffffp1023 : -1);
      const uint32_t *a[] = {storage[0], storage[1]}, *b[] = {storage[2], storage[3]};
      const uint32_t *c[8];
      uint32_t *d[8];
      for (int reg = 0; reg < 8; ++reg) {
        c[reg] = storage[4 + reg];
        d[reg] = storage[12 + reg];
      }
      ASSERT_EQ(rj_arith_v_mfma_f64_16x16x4_f64_w64(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, 0, d, a,
                                                    b, c),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(std::bit_cast<double>(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32)),
                overflow ? 0x1.fffffffffffffp1023 : -0x1p-54);
    }
  }
}
TEST(MfmaF64, mfma_f64_16x16x4_f64_w64) { arithmetic<16, 1>(rj_arith_v_mfma_f64_16x16x4_f64_w64); }
TEST(MfmaF64, mfma_f64_4x4x4_4b_f64_w64) { arithmetic<4, 4>(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64); }

TEST(MfmaF64, Gfx950ExactControlsAndGuards) {
  const auto levels = supported_cpu_levels();
  uint32_t storage[20][64] = {};
  auto set = [&](int reg, int lane, uint64_t bits) {
    storage[reg][lane] = uint32_t(bits);
    storage[reg + 1][lane] = uint32_t(bits >> 32);
  };
  for (int lane = 0; lane < 64; ++lane) {
    set(0, lane, std::bit_cast<uint64_t>(-2.0));
    set(2, lane, std::bit_cast<uint64_t>(-3.0));
    for (int r = 0; r < 8; r += 2) {
      set(4 + r, lane, std::bit_cast<uint64_t>(-5.0));
    }
  }
  const uint32_t *a[] = {storage[0], storage[1]}, *b[] = {storage[2], storage[3]};
  const uint32_t *c[8];
  uint32_t *d[8];
  for (int r = 0; r < 8; ++r) {
    c[r] = storage[4 + r];
    d[r] = storage[12 + r];
  }
  // CBSZ abs mask (controls[5:3] in 0..4) with BLGP=0 succeeds in mode 2 (and Compatibility /
  // Loose / allow-Loose), while mode 3 returns UNSUPPORTED_FP_STATUS without touching D:
  // cbsz=1 (|A|): 4 * (|+2| * -3) + (-5) = -29; cbsz=4 (|C|): 4 * (-2 * -3) + |+5| = 29.
  for (uint64_t cpu : levels) {
    for (uint64_t sem : semantics_modes()) {
      for (int r = 0; r < 8; ++r) {
        std::fill_n(storage[12 + r], 64, 0xdeadbeefU);
      }
      const int st16 = rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(cpu | sem, 1ULL << 3, d, a, b, c);
      if (sem == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
        ASSERT_EQ(st16, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        EXPECT_EQ(storage[12][0], 0xdeadbeefU);
      } else {
        ASSERT_EQ(st16, RJ_ARITH_SUCCESS);
        EXPECT_EQ(std::bit_cast<double>(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32)), -29.0);
      }
      for (int r = 0; r < 2; ++r) {
        std::fill_n(storage[12 + r], 64, 0xdeadbeefU);
      }
      const int st4 = rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(cpu | sem, 4ULL << 3, d, a, b, c);
      if (sem == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
        ASSERT_EQ(st4, RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
        EXPECT_EQ(storage[12][0], 0xdeadbeefU);
      } else {
        ASSERT_EQ(st4, RJ_ARITH_SUCCESS);
        EXPECT_EQ(std::bit_cast<double>(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32)), 29.0);
      }
    }
    // BLGP != 0 or CBSZ >= 5 works in Loose/Compatibility/mode 1 and rejects strict mode 2.
    ASSERT_EQ(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY,
                                                          7ULL, d, a, b, c),
              RJ_ARITH_SUCCESS);
    // 4 * (+2 * +3) + (+5) = 29.
    EXPECT_EQ(std::bit_cast<double>(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32)), 29.0);
    storage[12][0] = 0xdeadbeefU;
    EXPECT_EQ(rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(
                  cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 1ULL, d, a, b, c),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(storage[12][0], 0xdeadbeefU);
    EXPECT_EQ(rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(
                  cpu | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 5ULL << 3, d, a, b, c),
              RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
    EXPECT_EQ(storage[12][0], 0xdeadbeefU);
    EXPECT_EQ(rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(cpu | RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 6,
                                                         nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
  // MI350X wave2_3f capture row (v_mfma_f64_4x4x4_4b_f64 wave 1 lane 0):
  // cbsz=0 -> 0xc02f5485490e821b, cbsz=1 (|A|) -> 0x40268d6532f05171, cbsz=4 (|C|) ->
  // 0xc020cea71cc4b49c.
  const uint64_t hw_a[4] = {0xbfc102ae2ca7d1e0ULL, 0x40019c4b13efb996ULL, 0xc0068c117c0faea0ULL,
                            0xc005f48f08b09f5eULL};
  const uint64_t hw_b[4] = {0x3fdb6575b1ccf3b0ULL, 0x3fe4e0f5aa49fd98ULL, 0x3ffe5c089178c8d0ULL,
                            0x4007832372b79c7eULL};
  for (int lane = 0; lane < 64; ++lane) {
    set(0, lane, 0ULL);
    set(2, lane, 0ULL);
    set(4, lane, 0ULL);
  }
  for (int k = 0; k < 4; ++k) {
    set(0, k * 16, hw_a[k]);
    set(2, k * 16, hw_b[k]);
  }
  set(4, 0, 0xc00d0bbc58939afeULL);
  for (const auto &[cbsz, expected] :
       {std::pair{0ULL, 0xc02f5485490e821bULL}, std::pair{1ULL, 0x40268d6532f05171ULL},
        std::pair{4ULL, 0xc020cea71cc4b49cULL}}) {
    ASSERT_EQ(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(
                  RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, cbsz << 3, d, a, b, c),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32), expected);
  }
  // 0 * Inf overrides qNaN(C) -> 0xfff8000000000000 in mode 2; mode 3 rejects both shapes
  // with UNSUPPORTED_FP_STATUS and leaves D intact.
  for (int lane = 0; lane < 64; ++lane) {
    set(0, lane, 0ULL);
    set(2, lane, 0x7ff0000000000000ULL);
    set(4, lane, 0x7ff8000000000123ULL);
  }
  for (int r = 0; r < 8; ++r) {
    std::fill_n(storage[12 + r], 64, 0xdeadbeefU);
  }
  EXPECT_EQ(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0,
                                                        d, a, b, c),
            RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
  EXPECT_EQ(storage[12][0], 0xdeadbeefU);
  EXPECT_EQ(storage[13][0], 0xdeadbeefU);
  EXPECT_EQ(rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, 0, d,
                                                       a, b, c),
            RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS);
  EXPECT_EQ(storage[12][0], 0xdeadbeefU);
  ASSERT_EQ(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(
                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32), 0xfff8000000000000ULL);
  for (int lane = 0; lane < 64; ++lane) {
    set(0, lane, 0x7ff0000000000055ULL);
  }
  ASSERT_EQ(rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(
                RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS, 0, d, a, b, c),
            RJ_ARITH_SUCCESS);
  EXPECT_EQ(uint64_t(d[0][0]) | (uint64_t(d[1][0]) << 32), 0x7ff8000000000055ULL);
}
} // namespace
