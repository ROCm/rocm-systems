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
template <int M, int K, int Bits> void arithmetic(Entry entry) {
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
  auto invoke = [&](uint64_t flags, auto &storage, int offset, uint64_t controls = 0) {
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
    return entry(flags, controls, d, a, b, index);
  };
  const auto levels = supported_cpu_levels();
  for (int aliasing_offset : {0, A, Index, Dest}) {
    auto expected = input;
    ASSERT_EQ(invoke(RJ_ARITH_SEMANTICS_COMPATIBILITY, expected, aliasing_offset),
              RJ_ARITH_SUCCESS);
    for (uint64_t cpu : levels) {
      for (uint64_t semantics : semantics_modes()) {
        auto actual = input;
        int status = invoke(cpu | semantics, actual, aliasing_offset);
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
  // Selecting a metadata set is equivalent to shifting it to the first set.
  for (uint64_t controls : {8ULL, 16ULL, 24ULL, 9ULL}) {
    // One nibble selects two factors; each lane holds M*K/256 nibbles.
    constexpr int metadata_bits = 4 * (M * K / 256);
    const int set = (controls & 3) ? 0 : (controls >> 3) % (32 / metadata_bits);
    const int shift = metadata_bits * set;
    auto expected = input;
    for (auto &word : expected[Index]) {
      word >>= shift;
    }
    ASSERT_EQ(invoke(RJ_ARITH_SEMANTICS_COMPATIBILITY, expected, Dest), RJ_ARITH_SUCCESS);
    for (uint64_t cpu : levels) {
      auto actual = input;
      ASSERT_EQ(invoke(cpu | RJ_ARITH_SEMANTICS_COMPATIBILITY, actual, Dest, controls),
                RJ_ARITH_SUCCESS);
      for (int reg = 0; reg < D; ++reg) {
        EXPECT_EQ(actual[Dest + reg], expected[Dest + reg]);
      }
    }
  }
  EXPECT_EQ(entry(RJ_ARITH_SEMANTICS_LOOSE, 1ULL << 7, nullptr, nullptr, nullptr, nullptr),
            RJ_ARITH_ERROR_INVALID_FLAGS);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x32_f16_w64) {
  arithmetic<16, 32, 16>(rj_arith_v_smfmac_f32_16x16x32_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x32_bf16_w64) {
  arithmetic<16, 32, 16>(rj_arith_v_smfmac_f32_16x16x32_bf16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x16_f16_w64) {
  arithmetic<32, 16, 16>(rj_arith_v_smfmac_f32_32x32x16_f16_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x16_bf16_w64) {
  arithmetic<32, 16, 16>(rj_arith_v_smfmac_f32_32x32x16_bf16_w64);
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
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_fp8_fp8_w64_gfx950) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_fp8_bf8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_fp8_bf8_w64_gfx950) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_fp8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_fp8_w64_gfx950) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_bf8_w64) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_16x16x64_bf8_bf8_w64_gfx950) {
  arithmetic<16, 64, 8>(rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_fp8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_fp8_w64_gfx950) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_bf8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_fp8_bf8_w64_gfx950) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_fp8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_fp8_w64_gfx950) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_bf8_w64) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64);
}
TEST(SparseFloatMatrix, smfmac_f32_32x32x32_bf8_bf8_w64_gfx950) {
  arithmetic<32, 32, 8>(rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950);
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
} // namespace
