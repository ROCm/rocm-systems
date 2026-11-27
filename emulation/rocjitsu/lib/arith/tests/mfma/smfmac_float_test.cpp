// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse matrix API policy, SIMD parity and aliasing; decoded tests cover physical layout
// independently.
#include "common/cpu.h"
#include "common/semantics.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include <array>
#include <gtest/gtest.h>
#include <random>

namespace {
using namespace rocjitsu::arith::test;
using Entry = decltype(&rj_arith_v_smfmac_f32_16x16x32_f16_w64);

// Compare every CPU path to the inherited scalar FMA sequence, with aliased tied inputs.
template <int M, int K, int Bits> void arithmetic(Entry entry, Entry gfx942 = nullptr) {
  constexpr int A = M * K * Bits / 4096, B = 2 * A, D = M * M / 64, Index = A + B, Dest = Index + 1,
                Count = Dest + D;
  std::array<std::array<uint32_t, 64>, Count> input;
  std::minstd_rand random;
  for (auto &reg : input) {
    for (auto &word : reg) {
      word = uint32_t(random() >> 15) << 16 | (random() >> 15);
    }
  }
  // Exercise all legal selector pairs, including high metadata nibbles.
  constexpr uint32_t pairs[] = {4, 8, 12, 9, 13, 14};
  for (auto &word : input[Index]) {
    word = 0;
    for (int n = 0; n < 8; ++n) {
      word |= pairs[(random() >> 16) % 6] << (4 * n);
    }
  }
  auto invoke = [&](Entry target, uint64_t flags, auto &storage, int offset,
                    uint64_t controls = 0) {
    const uint32_t *a[A], *b[B], *index[] = {storage[Index].data()};
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
    return target(flags, controls, d, a, b, index);
  };
  const auto levels = supported_cpu_levels();
  for (Entry target : {entry, gfx942}) {
    if (!target) {
      continue;
    }
    for (int aliasing_offset : {0, A, Index, Dest}) {
      auto expected = input;
      ASSERT_EQ(invoke(target, RJ_ARITH_SEMANTICS_COMPATIBILITY, expected, aliasing_offset),
                RJ_ARITH_SUCCESS);
      auto exact_expected = input;
      if (target == gfx942) {
        // Verify aliased tied destinations match a disjoint accumulator copy in exact mode.
        auto disjoint = input;
        for (int reg = 0; reg < D; ++reg) {
          disjoint[Dest + reg] = input[aliasing_offset + reg];
        }
        ASSERT_EQ(invoke(target, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, disjoint, Dest),
                  RJ_ARITH_SUCCESS);
        for (int reg = 0; reg < D; ++reg) {
          exact_expected[aliasing_offset + reg] = disjoint[Dest + reg];
        }
      }
      for (uint64_t cpu : levels) {
        for (uint64_t semantics : semantics_modes()) {
          auto actual = input;
          int status = invoke(target, cpu | semantics, actual, aliasing_offset);
          if (target == gfx942 && semantics != RJ_ARITH_SEMANTICS_LOOSE &&
              semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            EXPECT_EQ(actual, exact_expected);
          } else if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
                     semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
            ASSERT_EQ(status, RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS);
            EXPECT_EQ(actual, input);
          } else {
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            EXPECT_EQ(actual, expected);
          }
        }
        if (target == gfx942) {
          // RJ_ARITH_FP16_OVFL applies per-step clamping in exact modes while Loose ignores it.
          auto exact_ovfl = input;
          {
            auto disjoint = input;
            for (int reg = 0; reg < D; ++reg) {
              disjoint[Dest + reg] = input[aliasing_offset + reg];
            }
            ASSERT_EQ(invoke(target, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL | RJ_ARITH_FP16_OVFL,
                             disjoint, Dest),
                      RJ_ARITH_SUCCESS);
            for (int reg = 0; reg < D; ++reg) {
              exact_ovfl[aliasing_offset + reg] = disjoint[Dest + reg];
            }
          }
          for (uint64_t semantics : semantics_modes()) {
            auto actual = input;
            int status =
                invoke(target, cpu | semantics | RJ_ARITH_FP16_OVFL, actual, aliasing_offset);
            ASSERT_EQ(status, RJ_ARITH_SUCCESS);
            if (semantics == RJ_ARITH_SEMANTICS_LOOSE ||
                semantics == RJ_ARITH_SEMANTICS_COMPATIBILITY) {
              EXPECT_EQ(actual, expected);
            } else {
              EXPECT_EQ(actual, exact_ovfl);
            }
          }
        }
      }
    }
    // Selecting a metadata set is equivalent to shifting it to the first set.
    for (uint64_t controls : {8ULL, 16ULL, 24ULL, 9ULL}) {
      // One nibble selects two factors; each lane holds M*K/256 nibbles.
      constexpr int metadata_bits = 4 * (M * K / 256);
      const int set = (controls & 3) ? 0 : (controls >> 3) % (32 / metadata_bits);
      const int shift = metadata_bits * set;
      for (uint64_t mode : {uint64_t(RJ_ARITH_SEMANTICS_COMPATIBILITY),
                            uint64_t(target == gfx942 ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
                                                      : RJ_ARITH_SEMANTICS_COMPATIBILITY)}) {
        auto expected = input;
        for (auto &word : expected[Index]) {
          word >>= shift;
        }
        ASSERT_EQ(invoke(target, mode, expected, Dest), RJ_ARITH_SUCCESS);
        for (uint64_t cpu : levels) {
          auto actual = input;
          ASSERT_EQ(invoke(target, cpu | mode, actual, Dest, controls), RJ_ARITH_SUCCESS);
          for (int reg = 0; reg < D; ++reg) {
            EXPECT_EQ(actual[Dest + reg], expected[Dest + reg]);
          }
        }
      }
    }
    EXPECT_EQ(target(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 7, nullptr, nullptr, nullptr, nullptr),
              RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}
// The last NaN factor wins across terms. Within one term A precedes B,
// and an initial accumulator payload survives if both factors stay numeric.
TEST(SparseFloatMatrix, OrderedBf16NanPayloads) {
  using Wave = std::array<uint32_t, 64>;
  for (uint64_t cpu : supported_cpu_levels()) {
    std::array<Wave, 2> left{};
    std::array<Wave, 4> right{}, accumulator{};
    Wave metadata;
    metadata.fill(0x44444444);
    for (auto &reg : left) {
      reg.fill(0x3f803f80);
    }
    for (auto &reg : right) {
      reg.fill(0x3f803f80);
    }
    for (auto &reg : accumulator) {
      reg.fill(0x7fa12345);
    }
    // In row 0, the earlier A NaN is replaced by either a later B NaN or
    // by the A NaN at the last term (k=29). Row 1 has no NaN in A.
    left[0][0] = 0x3f807fc1;
    left[1][48] = 0xffa23f80;
    // k=28 and k=29 occupy the low and high halves of B word 2.
    right[2][48] = 0x7fa47fc3;
    const uint32_t *a[] = {left[0].data(), left[1].data()};
    const uint32_t *b[] = {right[0].data(), right[1].data(), right[2].data(), right[3].data()};
    const uint32_t *index[] = {metadata.data()};
    uint32_t *d[] = {accumulator[0].data(), accumulator[1].data(), accumulator[2].data(),
                     accumulator[3].data()};
    ASSERT_EQ(
        rj_arith_v_smfmac_f32_16x16x32_bf16_w64(cpu | RJ_ARITH_SEMANTICS_LOOSE, 0, d, a, b, index),
        RJ_ARITH_SUCCESS);
    EXPECT_EQ(accumulator[0][0], 0xffe20000u);
    EXPECT_EQ(accumulator[0][1], 0xffe20000u);
    EXPECT_EQ(accumulator[1][0], 0x7fe40000u);
    EXPECT_EQ(accumulator[1][1], 0x7fe12345u);
  }
}

TEST(SparseFloatMatrix, smfmac_f32_16x16x32_f16_w64) {
  arithmetic<16, 32, 16>(rj_arith_v_smfmac_f32_16x16x32_f16_w64_gfx950,
                         rj_arith_v_smfmac_f32_16x16x32_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x32_bf16_w64) {
  arithmetic<16, 32, 16>(rj_arith_v_smfmac_f32_16x16x32_bf16_w64_gfx950,
                         rj_arith_v_smfmac_f32_16x16x32_bf16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x16_f16_w64) {
  arithmetic<32, 16, 16>(rj_arith_v_smfmac_f32_32x32x16_f16_w64_gfx950,
                         rj_arith_v_smfmac_f32_32x32x16_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x16_bf16_w64) {
  arithmetic<32, 16, 16>(rj_arith_v_smfmac_f32_32x32x16_bf16_w64_gfx950,
                         rj_arith_v_smfmac_f32_32x32x16_bf16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_f16_w64) {
  arithmetic<16, 64, 16>(rj_arith_v_smfmac_f32_16x16x64_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf16_w64) {
  arithmetic<16, 64, 16>(rj_arith_v_smfmac_f32_16x16x64_bf16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_f16_w64) {
  arithmetic<32, 32, 16>(rj_arith_v_smfmac_f32_32x32x32_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf16_w64) {
  arithmetic<32, 32, 16>(rj_arith_v_smfmac_f32_32x32x32_bf16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_fp8_fp8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950,
                        rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_fp8_bf8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950,
                        rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_fp8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950,
                        rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_bf8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950,
                        rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_fp8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950,
                        rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_bf8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950,
                        rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_fp8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950,
                        rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_bf8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950,
                        rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x128_fp8_fp8_w64) {
  arithmetic<16, 128, 8>(rj_arith_v_smfmac_f32_16x16x128_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x128_fp8_bf8_w64) {
  arithmetic<16, 128, 8>(rj_arith_v_smfmac_f32_16x16x128_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x128_bf8_fp8_w64) {
  arithmetic<16, 128, 8>(rj_arith_v_smfmac_f32_16x16x128_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x128_bf8_bf8_w64) {
  arithmetic<16, 128, 8>(rj_arith_v_smfmac_f32_16x16x128_bf8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x64_fp8_fp8_w64) {
  arithmetic<32, 64, 8>(rj_arith_v_smfmac_f32_32x32x64_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x64_fp8_bf8_w64) {
  arithmetic<32, 64, 8>(rj_arith_v_smfmac_f32_32x32x64_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x64_bf8_fp8_w64) {
  arithmetic<32, 64, 8>(rj_arith_v_smfmac_f32_32x32x64_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x64_bf8_bf8_w64) {
  arithmetic<32, 64, 8>(rj_arith_v_smfmac_f32_32x32x64_bf8_bf8_w64);
}

// MI300X hardware-captured outputs (AMD Instinct MI300X VF, gfx942, ROCm 7.2.4 /
// HIP 7.2.53211, HW_REG_MODE.FP16_OVFL bit 23 = 0 vs 1) for non-BF16 sparse forms.
// These establish both tied-accumulator clamping with zero products (+Inf -> 0x7f7fffff,
// qNaN -> 0x00000000) and multi-step NaN flushing where step 0 produces NaN and step 1
// accumulates a finite dot product.
TEST(SparseFloatMatrix, Gfx942Fp16OvflMatchesCapturedSparseF16AndFp8) {
  for (uint64_t semantics : {RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
                             RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL}) {
    for (Entry fn :
         {rj_arith_v_smfmac_f32_16x16x32_f16_w64, rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64}) {
      std::array<std::array<uint32_t, 64>, 2> a_zero{};
      std::array<std::array<uint32_t, 64>, 4> b_zero{}, d_buf{};
      std::array<uint32_t, 64> idx_zero{};
      const uint32_t *a[] = {a_zero[0].data(), a_zero[1].data()};
      const uint32_t *b[] = {b_zero[0].data(), b_zero[1].data(), b_zero[2].data(),
                             b_zero[3].data()};
      const uint32_t *index[] = {idx_zero.data()};
      uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};

      // Zero products with tied +Inf (lane 0) and qNaN (lane 1).
      d_buf[0][0] = 0x7f800000U;
      d_buf[0][1] = 0x7fc00000U;
      ASSERT_EQ(fn(semantics, 0, d, a, b, index), RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7f800000U);
      EXPECT_EQ(d_buf[0][1], 0xffc00000U);

      d_buf[0][0] = 0x7f800000U;
      d_buf[0][1] = 0x7fc00000U;
      ASSERT_EQ(fn(semantics | RJ_ARITH_FP16_OVFL, 0, d, a, b, index), RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[0][0], 0x7f7fffffU);
      EXPECT_EQ(d_buf[0][1], 0x00000000U);
    }

    // Captured MI300X sparse F16 row (wave 3, row 2, col 1 -> out_reg 2, out_lane 1):
    // FP16_OVFL=0 yields 0xffc00000, while FP16_OVFL=1 flushes step 0's NaN so step 1 yields
    // 0xbf2d5f6d.
    {
      std::array<std::array<uint32_t, 64>, 2> a_buf{};
      std::array<std::array<uint32_t, 64>, 4> b_buf{}, d_buf{};
      std::array<uint32_t, 64> idx_buf{};
      constexpr int kALanes[4] = {2, 18, 34, 50};
      constexpr uint32_t kAWords[4][2] = {{0x3bdbb8e5U, 0x387bb8ceU},
                                          {0xfc002896U, 0x392ea45bU},
                                          {0x3baab547U, 0x0000b4dfU},
                                          {0x80013930U, 0xb3ed305cU}};
      constexpr uint32_t kIdx[4] = {0x9dc449cdU, 0x44e9d484U, 0xd4484dc9U, 0x48e899ceU};
      constexpr int kBLanes[4] = {1, 17, 33, 49};
      constexpr uint32_t kBWords[4][4] = {{0xb61f3bacU, 0x3b1ab89dU, 0xae043bfcU, 0xba61b811U},
                                          {0x00017c00U, 0x3906aa12U, 0x3aeabb8eU, 0x300e3176U},
                                          {0xae7aaebaU, 0xbadab583U, 0xb8938001U, 0xaecb3adcU},
                                          {0xb82bb56aU, 0xb9deba9fU, 0xb567b1b2U, 0xb9d3b2bfU}};
      for (int g = 0; g < 4; ++g) {
        a_buf[0][kALanes[g]] = kAWords[g][0];
        a_buf[1][kALanes[g]] = kAWords[g][1];
        idx_buf[kALanes[g]] = kIdx[g];
        for (int r = 0; r < 4; ++r) {
          b_buf[r][kBLanes[g]] = kBWords[g][r];
        }
      }
      const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
      const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data(), b_buf[2].data(), b_buf[3].data()};
      const uint32_t *index[] = {idx_buf.data()};
      uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};

      d_buf[2][1] = 0x429dc064U;
      ASSERT_EQ(rj_arith_v_smfmac_f32_16x16x32_f16_w64(semantics, 0, d, a, b, index),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[2][1], 0xffc00000U);

      d_buf[2][1] = 0x429dc064U;
      ASSERT_EQ(
          rj_arith_v_smfmac_f32_16x16x32_f16_w64(semantics | RJ_ARITH_FP16_OVFL, 0, d, a, b, index),
          RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[2][1], 0xbf2d5f6dU);
    }

    // Captured MI300X sparse FP8 row (wave 0, row 2, col 0 -> out_reg 2, out_lane 0):
    // FP16_OVFL=0 yields 0xffc00000, while FP16_OVFL=1 flushes step 0's NaN so step 1 yields
    // 0xc5d62f21.
    {
      std::array<std::array<uint32_t, 64>, 2> a_buf{};
      std::array<std::array<uint32_t, 64>, 4> b_buf{}, d_buf{};
      std::array<uint32_t, 64> idx_buf{};
      constexpr int kALanes[4] = {2, 18, 34, 50};
      constexpr uint32_t kAWords[4][2] = {{0x80fc7134U, 0xb2625a3aU},
                                          {0x774b7df7U, 0xa75803adU},
                                          {0xbf28c1a1U, 0xa16e223dU},
                                          {0xed6a262bU, 0x00697839U}};
      constexpr uint32_t kIdx[4] = {0xe849c999U, 0xed84494cU, 0xc89c88ecU, 0x9894ed9dU};
      constexpr int kBLanes[4] = {0, 16, 32, 48};
      constexpr uint32_t kBWords[4][4] = {{0xda7f6e05U, 0x6fd1133bU, 0x6aef84e7U, 0x7adc0cd2U},
                                          {0xe0f8f8dcU, 0xc1bd68ecU, 0x14b995f4U, 0xe6008d67U},
                                          {0x5ae9e4a6U, 0x840e0142U, 0xc1b02f2bU, 0x38efaca8U},
                                          {0xfd411268U, 0x55589a98U, 0xd2e1392bU, 0xa1fa451cU}};
      for (int g = 0; g < 4; ++g) {
        a_buf[0][kALanes[g]] = kAWords[g][0];
        a_buf[1][kALanes[g]] = kAWords[g][1];
        idx_buf[kALanes[g]] = kIdx[g];
        for (int r = 0; r < 4; ++r) {
          b_buf[r][kBLanes[g]] = kBWords[g][r];
        }
      }
      const uint32_t *a[] = {a_buf[0].data(), a_buf[1].data()};
      const uint32_t *b[] = {b_buf[0].data(), b_buf[1].data(), b_buf[2].data(), b_buf[3].data()};
      const uint32_t *index[] = {idx_buf.data()};
      uint32_t *d[] = {d_buf[0].data(), d_buf[1].data(), d_buf[2].data(), d_buf[3].data()};

      d_buf[2][0] = 0xc2e6d38cU;
      ASSERT_EQ(rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64(semantics, 0, d, a, b, index),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[2][0], 0xffc00000U);

      d_buf[2][0] = 0xc2e6d38cU;
      ASSERT_EQ(rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64(semantics | RJ_ARITH_FP16_OVFL, 0, d, a,
                                                           b, index),
                RJ_ARITH_SUCCESS);
      EXPECT_EQ(d_buf[2][0], 0xc5d62f21U);
    }
  }
}
} // namespace
