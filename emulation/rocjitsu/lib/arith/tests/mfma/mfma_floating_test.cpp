// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Dense floating MFMA contracts, layout, source controls and whole-wave aliasing.
#include "common/cpu.h"
#include "common/floating_point.h"
#include "common/half_reference.h"
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
using Entry = decltype(&rj_arith_v_mfma_f32_16x16x16_f16_w64);

// Compare a full instruction against the established approximate reduction,
// including batched layouts and sources that overlap the destination.
template <int M, int K, int Batch, bool Bf16, bool FullF32 = false, int ABlockLanes = 0>
void arithmetic(Entry entry, uint32_t factor_mask = UINT32_MAX, Entry gfx942 = nullptr) {
  constexpr int InputRegs = M * K * Batch / (FullF32 ? 64 : 128);
  constexpr int OutputRegs = M * M * Batch / 64;
  constexpr int Count = 2 * InputRegs + 2 * OutputRegs;
  std::array<std::array<uint32_t, 64>, Count> input;
  std::minstd_rand random;
  for (int r = 0; r < Count; ++r) {
    for (auto &word : input[r]) {
      if (r < 2 * InputRegs && !FullF32) {
        auto half = [&] { return Bf16 ? random_bf16_bits(random) : random_f16_bits(random); };
        const uint16_t low = half();
        const uint16_t high = half();
        word = uint32_t(low) | uint32_t(high) << 16;
      } else {
        word = random_f32_bits(random);
      }
    }
  }
  // One NaN and infinity in each factor also exercise SIMD exceptional lanes.
  for (int r : {0, InputRegs}) {
    input[r][0] = FullF32 ? kF32QuietNaNBits : Bf16 ? kBF16QuietNaNBits : kF16QuietNaNBits;
    input[r][1] = FullF32 ? kF32PositiveInfinityBits
                  : Bf16  ? kBF16PositiveInfinityBits
                          : kF16PositiveInfinityBits;
  }
  const auto levels = supported_cpu_levels();
  for (uint64_t controls :
       {0ULL, 1ULL | (1ULL << 3) | (3ULL << 7), 4ULL | (13ULL << 3) | (7ULL << 7), 7ULL}) {
    std::array<std::array<float, 64>, OutputRegs> expected{};
    for (int batch = 0; batch < Batch; ++batch) {
      for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {
          const int groups = (64 + M * M / 4 - 1) / (M * M / 4);
          const int reg = batch * (M * M / 64) + row / (4 * (64 / M)) * 4 + row % 4;
          const int lane = batch % groups * M + row / 4 % (64 / M) * groups * M + col;
          float sum = std::bit_cast<float>(input[2 * InputRegs + reg][lane]);
          // Reference indexing follows the legacy GFX9 executor; do not call
          // the implementation's layout helpers from this test.
          auto read = [&](int operand, int index, int k) {
            const int per_lane = K / (64 / (M * Batch));
            int source_lane = batch * M + k / per_lane * M * Batch + index;
            int cbsz = int(controls & 7), blgp = int(controls >> 7 & 7);
            if (operand == 0 && cbsz > 0 && cbsz <= 4) {
              int size = ABlockLanes ? ABlockLanes : 64 >> cbsz;
              if (size <= (64 >> cbsz)) {
                int group = size << cbsz;
                source_lane = source_lane / group * group + source_lane % size +
                              size * (int(controls >> 3 & 15) % (1 << cbsz));
              }
            }
            if (operand == 1 && blgp) {
              if (blgp <= 2) {
                source_lane = source_lane % 32 + (blgp - 1) * 32;
              } else if (blgp == 3) {
                source_lane = (source_lane + 16) % 64;
              } else {
                source_lane = source_lane % 16 + (blgp - 4) * 16;
              }
            }
            if constexpr (FullF32) {
              return std::bit_cast<float>(input[operand * InputRegs + k % per_lane][source_lane] &
                                          factor_mask);
            } else {
              uint16_t value = uint16_t(
                  input[operand * InputRegs + k % per_lane / 2][source_lane] >> (16 * (k % 2)));
              return Bf16 ? std::bit_cast<float>(uint32_t(value) << 16) : float(half_value(value));
            }
          };
          for (int k = 0; k < K; ++k) {
            sum = std::fma(read(0, row, k), read(1, col, k), sum);
          }
          expected[reg][lane] = sum;
        }
      }
    }
    for (Entry target : {entry, gfx942}) {
      if (!target) {
        continue;
      }
      for (uint64_t cpu : levels) {
        for (uint64_t semantics : semantics_modes()) {
          // Qualified gfx942 exact outputs have captured expectations in gfx942_exact_test.
          if (target == gfx942 && !controls && semantics != RJ_ARITH_SEMANTICS_LOOSE &&
              semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
            continue;
          }
          for (int aliasing_offset : {0, InputRegs, 2 * InputRegs, 2 * InputRegs + OutputRegs}) {
            auto storage = input;
            const uint32_t *a[InputRegs], *b[InputRegs], *c[OutputRegs];
            uint32_t *d[OutputRegs];
            for (int r = 0; r < InputRegs; ++r) {
              a[r] = storage[r].data();
              b[r] = storage[InputRegs + r].data();
            }
            for (int r = 0; r < OutputRegs; ++r) {
              c[r] = storage[2 * InputRegs + r].data();
              d[r] = storage[aliasing_offset + r].data();
            }
            int status = target(cpu | semantics, controls, d, a, b, c);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              EXPECT_EQ(storage, input);
              continue;
            }
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            // Keep the inherited absolute bound for FP32 accumulation; half-input
            // products retain the tighter bound.
            for (int r = 0; r < OutputRegs; ++r) {
              for (int lane = 0; lane < 64; ++lane) {
                expect_f32(d[r][lane], expected[r][lane], RJ_ARITH_SEMANTICS_LOOSE,
                           /*atol=*/FullF32 ? 0.002 : 1e-5f,
                           /*rtol=*/1e-5f);
              }
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 10, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}

// A single nonzero product must cancel before intermediate FP32 overflow.
TEST(MfmaFloating, OlderFusedCancellation) {
  struct Case {
    Entry entry;
    bool bf16;
  };
  for (const Case test : {
           Case{rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908, false},
           Case{rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a, false},
           Case{rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908, true},
           Case{rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a, true},
       }) {
    std::array<uint32_t, 64> left{}, right{};
    std::array<std::array<uint32_t, 64>, 4> result;
    const uint32_t expected = test.bf16 ? 0x7f7f0000U : 0x7f7fffffU;
    for (int lane = 0; lane < 16; ++lane) {
      // k=0 is the low half for BF16; every other reduction term is zero.
      left[lane] = test.bf16 ? 0x7f7fU : expected;
      right[lane] = test.bf16 ? 0x4000U : 0x40000000U;
    }
    const uint32_t *a[] = {left.data()}, *b[] = {right.data()}, *c[4];
    uint32_t *d[4];
    for (int reg = 0; reg < 4; ++reg) {
      c[reg] = result[reg].data();
      d[reg] = result[reg].data();
    }
    for (uint64_t cpu : supported_cpu_levels()) {
      for (uint64_t semantics : {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_COMPATIBILITY}) {
        for (auto &reg : result) {
          reg.fill(expected | 0x80000000U);
        }
        ASSERT_EQ(test.entry(cpu | semantics, 0, d, a, b, c), RJ_ARITH_SUCCESS);
        for (const auto &reg : result) {
          for (uint32_t value : reg) {
            EXPECT_EQ(value, expected) << "cpu=" << cpu << " bf16=" << test.bf16;
          }
        }
      }
    }
  }
}

TEST(MfmaF32, F32ClampsFiniteOverflowAndPreservesInfinities) {
  std::array<uint32_t, 64> a, b;
  std::array<std::array<uint32_t, 64>, 4> result;
  const uint32_t *sources_a[] = {a.data()};
  const uint32_t *sources_b[] = {b.data()};
  uint32_t *dest[] = {result[0].data(), result[1].data(), result[2].data(), result[3].data()};
  const uint32_t *acc[] = {dest[0], dest[1], dest[2], dest[3]};

  struct Case {
    uint32_t a, b;
    uint32_t initial, unclamped, clamped;
  };
  for (const Case test : {
           Case{0x7f7fffff, 0x40000000, 0, 0x7f800000, 0x7f7fffff},
           Case{0xff7fffff, 0x40000000, 0, 0xff800000, 0xff7fffff},
           Case{0x7f800000, 0x40000000, 0, 0x7f800000, 0x7f800000},
           Case{0x40000000, 0xff800000, 0, 0xff800000, 0xff800000},
           Case{0x7f7fffff, 0x40000000, 0xff800000, 0xff800000, 0xff800000},
       }) {
    a.fill(test.a);
    b.fill(test.b);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (bool clamp : {false, true}) {
        for (auto &reg : result) {
          reg.fill(test.initial);
        }
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | (clamp ? RJ_ARITH_FP16_OVFL : 0);
        ASSERT_EQ(rj_arith_v_mfma_f32_4x4x1_16b_f32_w64(flags, 0, dest, sources_a, sources_b, acc),
                  RJ_ARITH_SUCCESS);
        for (const auto &reg : result) {
          for (uint32_t value : reg) {
            EXPECT_EQ(value, clamp ? test.clamped : test.unclamped);
          }
        }
      }
    }
  }
}

TEST(MfmaHalf, Gfx950Bf16ClampsFiniteOverflowAndPreservesInfinities) {
  std::array<uint32_t, 64> a, b;
  std::array<std::array<uint32_t, 64>, 4> result;
  const uint32_t *sources_a[] = {a.data(), a.data()};
  const uint32_t *sources_b[] = {b.data(), b.data()};
  uint32_t *dest[] = {result[0].data(), result[1].data(), result[2].data(), result[3].data()};
  const uint32_t *acc[] = {dest[0], dest[1], dest[2], dest[3]};

  struct Case {
    uint16_t a, b;
    uint32_t initial, unclamped, clamped;
  };
  for (const Case test : {
           Case{0x7f7f, 0x4000, 0, 0x7f800000, 0x7f7fffff},
           Case{0xff7f, 0x4000, 0, 0xff800000, 0xff7fffff},
           Case{0x7f80, 0x4000, 0, 0x7f800000, 0x7f800000},
           Case{0x4000, 0xff80, 0, 0xff800000, 0xff800000},
           Case{0x7f7f, 0x4000, 0xff800000, 0xff800000, 0xff800000},
       }) {
    a.fill(uint32_t(test.a) * 0x10001u);
    b.fill(uint32_t(test.b) * 0x10001u);
    for (uint64_t cpu : supported_cpu_levels()) {
      for (bool clamp : {false, true}) {
        for (auto &reg : result) {
          reg.fill(test.initial);
        }
        const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | (clamp ? RJ_ARITH_FP16_OVFL : 0);
        ASSERT_EQ(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950(flags, 0, dest, sources_a,
                                                                sources_b, acc),
                  RJ_ARITH_SUCCESS);
        for (const auto &reg : result) {
          for (uint32_t value : reg) {
            EXPECT_EQ(value, clamp ? test.clamped : test.unclamped);
          }
        }
      }
    }
  }
  // BLGP=2 broadcasts the upper 32 B lanes. Column 0 overflows from finite
  // inputs in lane 32, while column 1 reads an actual infinity from lane 33.
  a.fill(0x7f7f7f7fu);
  b.fill(0x40004000u);
  b[0] = b[33] = 0x7f807f80u;
  for (uint64_t cpu : supported_cpu_levels()) {
    for (auto &reg : result) {
      reg.fill(0);
    }
    const uint64_t flags = cpu | RJ_ARITH_SEMANTICS_LOOSE | RJ_ARITH_FP16_OVFL;
    ASSERT_EQ(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950(flags, 2u << 7, dest, sources_a,
                                                            sources_b, acc),
              RJ_ARITH_SUCCESS);
    EXPECT_EQ(result[0][0], 0x7f7fffffu);
    EXPECT_EQ(result[0][1], 0x7f800000u);
  }
}

TEST(MfmaHalf, mfma_f32_32x32x4_2b_f16) {
  arithmetic<32, 4, 2, false>(rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx950, UINT32_MAX,
                              rj_arith_v_mfma_f32_32x32x4_2b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_f16) {
  arithmetic<16, 4, 4, false>(rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx950, UINT32_MAX,
                              rj_arith_v_mfma_f32_16x16x4_4b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_f16) {
  arithmetic<4, 4, 16, false>(rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx950, UINT32_MAX,
                              rj_arith_v_mfma_f32_4x4x4_16b_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x8_f16) {
  arithmetic<32, 8, 1, false>(rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx950, UINT32_MAX,
                              rj_arith_v_mfma_f32_32x32x8_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x16_f16) {
  arithmetic<16, 16, 1, false>(rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx950, UINT32_MAX,
                               rj_arith_v_mfma_f32_16x16x16_f16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x32_f16) {
  arithmetic<16, 32, 1, false>(rj_arith_v_mfma_f32_16x16x32_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x16_f16) {
  arithmetic<32, 16, 1, false>(rj_arith_v_mfma_f32_32x32x16_f16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x4_2b_bf16) {
  arithmetic<32, 4, 2, true>(rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx950, UINT32_MAX,
                             rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_bf16) {
  arithmetic<16, 4, 4, true>(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx950, UINT32_MAX,
                             rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_bf16) {
  arithmetic<4, 4, 16, true>(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx950, UINT32_MAX,
                             rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x8_bf16) {
  arithmetic<32, 8, 1, true>(rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx950, UINT32_MAX,
                             rj_arith_v_mfma_f32_32x32x8_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x16_bf16) {
  arithmetic<16, 16, 1, true>(rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx950, UINT32_MAX,
                              rj_arith_v_mfma_f32_16x16x16_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_16x16x32_bf16) {
  arithmetic<16, 32, 1, true>(rj_arith_v_mfma_f32_16x16x32_bf16_w64);
}
TEST(MfmaHalf, mfma_f32_32x32x16_bf16) {
  arithmetic<32, 16, 1, true>(rj_arith_v_mfma_f32_32x32x16_bf16_w64);
}

TEST(MfmaFullF32, mfma_f32_16x16x8_xf32) {
  arithmetic<16, 8, 1, false, true>(rj_arith_v_mfma_f32_16x16x8_xf32_w64_gfx950, 0xffffe000u,
                                    rj_arith_v_mfma_f32_16x16x8_xf32_w64);
}
TEST(MfmaFullF32, mfma_f32_32x32x4_xf32) {
  arithmetic<32, 4, 1, false, true>(rj_arith_v_mfma_f32_32x32x4_xf32_w64_gfx950, 0xffffe000u,
                                    rj_arith_v_mfma_f32_32x32x4_xf32_w64);
}
TEST(MfmaF32, mfma_f32_32x32x1_2b_f32_w64) {
  arithmetic<32, 1, 2, false, true>(rj_arith_v_mfma_f32_32x32x1_2b_f32_w64);
}
TEST(MfmaF32, mfma_f32_16x16x1_4b_f32_w64) {
  arithmetic<16, 1, 4, false, true>(rj_arith_v_mfma_f32_16x16x1_4b_f32_w64);
}
TEST(MfmaF32, mfma_f32_4x4x1_16b_f32_w64) {
  arithmetic<4, 1, 16, false, true>(rj_arith_v_mfma_f32_4x4x1_16b_f32_w64);
}
TEST(MfmaF32, mfma_f32_32x32x2_f32_w64) {
  arithmetic<32, 2, 1, false, true>(rj_arith_v_mfma_f32_32x32x2_f32_w64);
}
TEST(MfmaF32, mfma_f32_16x16x4_f32_w64) {
  arithmetic<16, 4, 1, false, true>(rj_arith_v_mfma_f32_16x16x4_f32_w64);
}
} // namespace

TEST(MfmaF32, mfma_f32_32x32x1_2b_f32_w64_gfx908) {
  arithmetic<32, 1, 2, false, true, 0>(rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx908);
}

TEST(MfmaF32, mfma_f32_16x16x1_4b_f32_w64_gfx908) {
  arithmetic<16, 1, 4, false, true, 0>(rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx908);
}

TEST(MfmaF32, mfma_f32_4x4x1_16b_f32_w64_gfx908) {
  arithmetic<4, 1, 16, false, true, 0>(rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx908);
}

TEST(MfmaF32, mfma_f32_32x32x2_f32_w64_gfx908) {
  arithmetic<32, 2, 1, false, true, 0>(rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx908);
}

TEST(MfmaF32, mfma_f32_16x16x4_f32_w64_gfx908) {
  arithmetic<16, 4, 1, false, true, 0>(rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908);
}

TEST(MfmaF32, mfma_f32_32x32x1_2b_f32_w64_gfx90a) {
  arithmetic<32, 1, 2, false, true, 32>(rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx90a);
}

TEST(MfmaF32, mfma_f32_16x16x1_4b_f32_w64_gfx90a) {
  arithmetic<16, 1, 4, false, true, 16>(rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx90a);
}

TEST(MfmaF32, mfma_f32_4x4x1_16b_f32_w64_gfx90a) {
  arithmetic<4, 1, 16, false, true, 4>(rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx90a);
}

TEST(MfmaF32, mfma_f32_32x32x2_f32_w64_gfx90a) {
  arithmetic<32, 2, 1, false, true, 32>(rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx90a);
}

TEST(MfmaF32, mfma_f32_16x16x4_f32_w64_gfx90a) {
  arithmetic<16, 4, 1, false, true, 16>(rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a);
}

// Input flushing, output flushing and accumulator inputs are distinct policy stages.
TEST(MfmaF32, OlderDenormalPoliciesAndIgnoredOverflowMode) {
  std::array<uint32_t, 64> a, b, initial;
  std::array<std::array<uint32_t, 64>, 4> output;
  const uint32_t *pa[] = {a.data()}, *pb[] = {b.data()};
  const uint32_t *pc[] = {initial.data(), initial.data(), initial.data(), initial.data()};
  uint32_t *d[] = {output[0].data(), output[1].data(), output[2].data(), output[3].data()};
  struct Case {
    uint32_t a, b, c, preserved, input_flushed, output_flushed, both_flushed;
  };
  for (auto entry :
       {rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908, rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a}) {
    const bool guest = entry == rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a;
    for (const Case test :
         {Case{0x00400000U, 0x40800000U, 0, 0x02000000U, 0, 0x02000000U, 0},
          Case{0x9f000000U, 0x1f000000U, 0, 0x80200000U, 0x80200000U, 0x80000000U, 0x80000000U},
          Case{0, 0x3f800000U, 0x00400000U, 0x00400000U, 0, 0, 0},
          Case{0x7f7fffffU, 0x40000000U, 0, 0x7f800000U, 0x7f800000U, 0x7f800000U, 0x7f800000U}}) {
      a.fill(test.a);
      b.fill(test.b);
      initial.fill(test.c);
      for (uint64_t flush : std::array<uint64_t, 4>{
               0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS, RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
               RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
        const uint32_t expected = !guest || !flush                             ? test.preserved
                                  : flush == RJ_ARITH_FP_FLUSH_INPUT_DENORMALS ? test.input_flushed
                                  : flush == RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS
                                      ? test.output_flushed
                                      : test.both_flushed;
        for (uint64_t cpu : supported_cpu_levels()) {
          for (uint64_t semantics : semantics_modes()) {
            for (auto &reg : output) {
              reg.fill(0xdeadbeefU);
            }
            int status = entry(cpu | semantics | flush | RJ_ARITH_FP16_OVFL, 0, d, pa, pb, pc);
            bool strict = semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL ||
                          semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
            ASSERT_EQ(status, strict ? RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS : RJ_ARITH_SUCCESS);
            for (const auto &reg : output) {
              for (uint32_t word : reg) {
                ASSERT_EQ(word, strict ? 0xdeadbeefU : expected);
              }
            }
          }
        }
      }
    }
  }
}

TEST(MfmaHalf, mfma_f32_32x32x2_2b_bf16_w64_gfx908) {
  arithmetic<32, 2, 2, true, false, 0>(rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx908);
}
TEST(MfmaHalf, mfma_f32_16x16x2_4b_bf16_w64_gfx908) {
  arithmetic<16, 2, 4, true, false, 0>(rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx908);
}
TEST(MfmaHalf, mfma_f32_4x4x2_16b_bf16_w64_gfx908) {
  arithmetic<4, 2, 16, true, false, 0>(rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx908);
}
TEST(MfmaHalf, mfma_f32_32x32x4_bf16_w64_gfx908) {
  arithmetic<32, 4, 1, true, false, 0>(rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx908);
}
TEST(MfmaHalf, mfma_f32_16x16x8_bf16_w64_gfx908) {
  arithmetic<16, 8, 1, true, false, 0>(rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908);
}
TEST(MfmaHalf, mfma_f32_32x32x4_2b_f16_w64_gfx90a) {
  arithmetic<32, 4, 2, false, false, 32>(rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_f16_w64_gfx90a) {
  arithmetic<16, 4, 4, false, false, 16>(rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_f16_w64_gfx90a) {
  arithmetic<4, 4, 16, false, false, 4>(rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_32x32x8_f16_w64_gfx90a) {
  arithmetic<32, 8, 1, false, false, 32>(rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x16_f16_w64_gfx90a) {
  arithmetic<16, 16, 1, false, false, 16>(rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_32x32x2_2b_bf16_w64_gfx90a) {
  arithmetic<32, 2, 2, true, false, 32>(rj_arith_v_mfma_f32_32x32x2_2b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x2_4b_bf16_w64_gfx90a) {
  arithmetic<16, 2, 4, true, false, 16>(rj_arith_v_mfma_f32_16x16x2_4b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_4x4x2_16b_bf16_w64_gfx90a) {
  arithmetic<4, 2, 16, true, false, 4>(rj_arith_v_mfma_f32_4x4x2_16b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_32x32x4_bf16_w64_gfx90a) {
  arithmetic<32, 4, 1, true, false, 32>(rj_arith_v_mfma_f32_32x32x4_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x8_bf16_w64_gfx90a) {
  arithmetic<16, 8, 1, true, false, 16>(rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_32x32x4_2b_bf16_w64_gfx90a) {
  arithmetic<32, 4, 2, true, false, 32>(rj_arith_v_mfma_f32_32x32x4_2b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x4_4b_bf16_w64_gfx90a) {
  arithmetic<16, 4, 4, true, false, 16>(rj_arith_v_mfma_f32_16x16x4_4b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_4x4x4_16b_bf16_w64_gfx90a) {
  arithmetic<4, 4, 16, true, false, 4>(rj_arith_v_mfma_f32_4x4x4_16b_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_32x32x8_bf16_w64_gfx90a) {
  arithmetic<32, 8, 1, true, false, 32>(rj_arith_v_mfma_f32_32x32x8_bf16_w64_gfx90a);
}
TEST(MfmaHalf, mfma_f32_16x16x16_bf16_w64_gfx90a) {
  arithmetic<16, 16, 1, true, false, 16>(rj_arith_v_mfma_f32_16x16x16_bf16_w64_gfx90a);
}

// Half-input gfx90a MFMA flushes in each operand's original format, even when
// the guest requests preservation. Overflow saturation remains ignored.
TEST(MfmaHalf, OlderFixedDenormalPolicies) {
  struct Case {
    bool bf16;
    uint16_t a, b;
    uint32_t c, preserved;
  };
  const Case cases[] = {
      {false, 0x0001, 0x3c00, 0, 0x35800000}, // 16 * 2^-24.
      {false, 0, 0x3c00, 0x00400000, 0x00400000},
      {true, 0x0040, 0x4080, 0, 0x02800000}, // 8 * 2^-127 * 4.
      {true, 0x0080, 0x3d80, 0, 0x00400000}, // 8 * 2^-126 * 2^-4.
      {true, 0, 0x3f80, 0x00400000, 0x00400000},
  };
  for (const auto &test : cases) {
    for (bool gfx90a : {false, true}) {
      Entry entry = test.bf16 ? (gfx90a ? rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx90a
                                        : rj_arith_v_mfma_f32_16x16x8_bf16_w64_gfx908)
                              : (gfx90a ? rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx90a
                                        : rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx908);
      std::array<std::array<uint32_t, 64>, 2> a, b;
      std::array<std::array<uint32_t, 64>, 4> output;
      std::array<uint32_t, 64> initial;
      for (auto &reg : a) {
        reg.fill(uint32_t(test.a) * 0x10001U);
      }
      for (auto &reg : b) {
        reg.fill(uint32_t(test.b) * 0x10001U);
      }
      initial.fill(test.c);
      const uint32_t *pa[] = {a[0].data(), a[1].data()}, *pb[] = {b[0].data(), b[1].data()};
      const uint32_t *pc[] = {initial.data(), initial.data(), initial.data(), initial.data()};
      uint32_t *d[] = {output[0].data(), output[1].data(), output[2].data(), output[3].data()};
      for (uint64_t cpu : supported_cpu_levels()) {
        for (uint64_t semantics : semantics_modes()) {
          for (uint64_t controls :
               std::array<uint64_t, 2>{0, RJ_ARITH_FP16_OVFL | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                              RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS}) {
            const int status = entry(cpu | semantics | controls, 0, d, pa, pb, pc);
            if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
              ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
              continue;
            }
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            for (const auto &reg : output) {
              for (uint32_t word : reg) {
                EXPECT_EQ(word, gfx90a ? 0U : test.preserved);
              }
            }
          }
        }
      }
    }
  }
}

// All gfx908 FP16 shapes retain the existing scalar/SIMD approximation.
TEST(MfmaHalf, Gfx908F16) {
  arithmetic<32, 4, 2, false>(rj_arith_v_mfma_f32_32x32x4_2b_f16_w64_gfx908);
  arithmetic<16, 4, 4, false>(rj_arith_v_mfma_f32_16x16x4_4b_f16_w64_gfx908);
  arithmetic<4, 4, 16, false>(rj_arith_v_mfma_f32_4x4x4_16b_f16_w64_gfx908);
  arithmetic<32, 8, 1, false>(rj_arith_v_mfma_f32_32x32x8_f16_w64_gfx908);
  arithmetic<16, 16, 1, false>(rj_arith_v_mfma_f32_16x16x16_f16_w64_gfx908);
}
