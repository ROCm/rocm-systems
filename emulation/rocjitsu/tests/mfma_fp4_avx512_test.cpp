// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/mfma_fp4_avx512.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mfma_fp4_vnni.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mfma_scale_simd.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <limits>
#include <random>

namespace {
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__FMA__)
using namespace rocjitsu::amdgpu;
constexpr std::array<float, 16> fp4 = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct RestoreEnvironment {
  fenv_t state;
  int error = errno;
  RestoreEnvironment() { std::fegetenv(&state); }
  ~RestoreEnvironment() {
    std::fesetenv(&state);
    errno = error;
  }
};

TEST(MfmaFp4Avx512, RejectsExceptionalScaleRangesBeforeExecution) {
  std::array<uint8_t, 64> a, b;
  a.fill(65);
  b.fill(65);
  EXPECT_TRUE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
  for (unsigned lane = 0; lane < 64; ++lane) {
    a[lane] = 64;
    EXPECT_FALSE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
    a[lane] = 65;
  }
  a.fill(185);
  b.fill(186);
  EXPECT_TRUE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
  for (unsigned lane = 0; lane < 64; ++lane) {
    b[lane] = 187;
    EXPECT_FALSE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
    b[lane] = 186;
  }
  a.fill(0);
  b.fill(255);
  EXPECT_FALSE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
  std::swap(a, b);
  EXPECT_FALSE(mfma_fp4_avx512_normal_scales(a.data(), b.data()));
}

TEST(MfmaFp4Avx512, SpecializedScalingMatchesGenericPolicy) {
#if __has_include(<experimental/simd>)
  RestoreEnvironment restore;
  using F = util::native<float>;
  static_assert(F::size() == 16);
  const std::array<float, 16> input = {0.0f,  -0.0f,  0.25f,   -0.25f,  1152.0f, -1152.0f,
                                       1.0f,  -1.0f,  0.5f,    -0.5f,   3.0f,    -3.0f,
                                       12.0f, -12.0f, 123.25f, -123.25f};
  F sums;
  sums.copy_from(input.data(), util::stdx::element_aligned);
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (uint32_t denorm = 0; denorm < 4; ++denorm)
      for (uint32_t a = 0; a < 256; ++a)
        for (uint32_t offset = 0; offset < 256; offset += 16) {
          std::array<uint8_t, 16> b;
          for (uint32_t lane = 0; lane < 16; ++lane)
            b[lane] = offset + lane;
          auto environment = [&] {
            std::fesetenv(FE_DFL_ENV);
            std::fesetround(mode);
            _mm_setcsr((_mm_getcsr() & ~0x8040u) | ((denorm & 1) ? 0x8000u : 0u) |
                       ((denorm & 2) ? 0x40u : 0u));
            std::feraiseexcept(FE_DIVBYZERO);
            errno = EDOM;
          };
          environment();
          const F expected = mfma_scale_e8m0_simd(sums, a, b.data());
          const int flags = std::fetestexcept(FE_ALL_EXCEPT), error = errno;
          environment();
          const F actual =
              std::bit_cast<F>(mfma_fp4_scale_avx512(std::bit_cast<__m512>(sums), a, b.data()));
          EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
          EXPECT_EQ(errno, error);
          for (uint32_t lane = 0; lane < 16; ++lane)
            ASSERT_EQ(std::bit_cast<uint32_t>(float(actual[lane])),
                      std::bit_cast<uint32_t>(float(expected[lane])))
                << "a=" << a << " b=" << uint32_t(b[lane]) << " mode=" << mode
                << " denorm=" << denorm << " lane=" << lane;
        }
#else
  GTEST_SKIP() << "experimental SIMD is unavailable";
#endif
}

TEST(MfmaFp4Avx512, PackedDecodeMatchesEveryNibbleAndMatrixPosition) {
  std::mt19937 rng(950);
  std::array<float, 2048> a, b;
  for (unsigned trial = 0; trial < 16; ++trial)
    for (unsigned reg = 0; reg < 4; ++reg) {
      std::array<uint32_t, 64> words;
      for (auto &word : words)
        word = rng();
      words[0] = 0x76543210;
      words[1] = 0xfedcba98;
      mfma_fp4_stage_a_avx512(words.data(), reg, a.data());
      mfma_fp4_stage_b_avx512(words.data(), reg, b.data());
      for (unsigned lane = 0; lane < 64; ++lane)
        for (unsigned element = 0; element < 8; ++element) {
          const auto expected = std::bit_cast<uint32_t>(fp4[(words[lane] >> (element * 4)) & 15]);
          const unsigned index = lane % 16, k = (lane / 16) * 32 + reg * 8 + element;
          ASSERT_EQ(std::bit_cast<uint32_t>(a[index * 128 + k]), expected);
          ASSERT_EQ(std::bit_cast<uint32_t>(b[k * 16 + index]), expected);
        }
    }
}

[[gnu::noinline]] void scalar_product(const float *a, const float *b, float *c, const uint8_t *as,
                                      const uint8_t *bs) {
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned col = 0; col < 16; ++col)
      for (unsigned block = 0; block < 4; ++block) {
        float sum = 0;
        for (unsigned k = block * 32; k < (block + 1) * 32; ++k)
          sum = std::fma(a[row * 128 + k], b[k * 16 + col], sum);
        const float scaled =
            as[block * 16 + row] == 255 || bs[block * 16 + col] == 255
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(sum, int(as[block * 16 + row]) + int(bs[block * 16 + col]) - 254);
        const auto previous = std::bit_cast<uint32_t>(c[row * 16 + col]);
        // Keep arithmetic exceptions and explicitly model accumulator-first
        // NaN propagation instead of depending on C++ operand commutation.
        volatile float next = c[row * 16 + col] + scaled;
        c[row * 16 + col] = (previous & 0x7fffffffu) > 0x7f800000u
                                ? std::bit_cast<float>(previous | 0x00400000u)
                                : next;
      }
}

TEST(MfmaFp4Avx512, ProductsScalingAndAccumulatorBitsMatchAcrossHostModes) {
  RestoreEnvironment restore;
  std::mt19937 rng(128);
  alignas(64) std::array<float, 2048> a, b;
  std::array<uint8_t, 64> as, bs;
  constexpr std::array<uint32_t, 12> acc = {0,          0x80000000, 0x3f800001, 0xbf800001,
                                            0x00000001, 0x80000001, 0x7f7fffff, 0xff7fffff,
                                            0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321};
  for (unsigned seed = 0; seed < 48; ++seed) {
    for (auto &v : a)
      v = fp4[rng() % 16];
    for (auto &v : b)
      v = fp4[rng() % 16];
    // Explicit signed-zero/cancellation witnesses as well as random dots.
    for (unsigned k = 0; k < 128; ++k) {
      a[k] = -0.0f;
      b[k * 16] = 1.0f;
      a[128 + k] = (k & 1) ? -1.0f : 1.0f;
    }
    for (unsigned i = 0; i < 64; ++i) {
      as[i] = seed % 3 == 0 ? 65 : seed % 3 == 1 ? 185 : 120 + rng() % 15;
      bs[i] = seed % 3 == 0 ? 65 : seed % 3 == 1 ? 186 : 120 + rng() % 15;
    }
    if (seed >= 16)
      for (unsigned i = 0; i < 64; ++i) {
        as[i] = (i + seed) % 7 == 0 ? 255 : rng() % 256;
        bs[i] = (i + seed) % 11 == 0 ? 255 : rng() % 256;
      }
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
      for (unsigned denorm = 0; denorm < 4; ++denorm) {
        SCOPED_TRACE(testing::Message()
                     << "seed=" << seed << " mode=" << mode << " denorm=" << denorm);
        std::array<float, 256> expected, actual;
        for (unsigned i = 0; i < 256; ++i)
          actual[i] = expected[i] = std::bit_cast<float>(acc[(i + seed) % acc.size()]);
        auto environment = [&] {
          std::fesetenv(FE_DFL_ENV);
          std::fesetround(mode);
          _mm_setcsr((_mm_getcsr() & ~0x8040u) | ((denorm & 1) ? 0x8000u : 0u) |
                     ((denorm & 2) ? 0x40u : 0u));
          std::feraiseexcept(FE_DIVBYZERO);
          errno = EDOM;
        };
        environment();
        scalar_product(a.data(), b.data(), expected.data(), as.data(), bs.data());
        const int flags = std::fetestexcept(FE_ALL_EXCEPT), error = errno;
        environment();
        if (mfma_fp4_avx512_normal_scales(as.data(), bs.data()))
          mfma_fp4_16x16x128_avx512(a.data(), b.data(), actual.data(), as.data(), bs.data());
        else
          mfma_fp4_16x16x128_avx512<false>(a.data(), b.data(), actual.data(), as.data(), bs.data());
        const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT), actual_error = errno;
        for (unsigned i = 0; i < 256; ++i)
          ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i])) << i;
        EXPECT_EQ(actual_flags, flags);
        EXPECT_EQ(actual_error, error);
      }
  }
}
#if defined(__AVX512BW__)
TEST(MfmaFp4Avx512, VnniPackingProductsAndHostModeGate) {
  if (!__builtin_cpu_supports("avx512vnni"))
    GTEST_SKIP() << "AVX-512 VNNI unavailable on this host";
  RestoreEnvironment restore;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::fesetround(mode);
    EXPECT_EQ(mfma_fp4_vnni_available(), mode == FE_TONEAREST);
  }
  std::mt19937 rng(32128);
  for (unsigned seed = 0; seed < 48; ++seed) {
    alignas(64) uint32_t a[512], b[512];
    alignas(64) float af[2048], bf[2048];
    for (unsigned reg = 0; reg < 4; ++reg) {
      std::array<uint32_t, 64> pa, pb;
      for (unsigned i = 0; i < 64; ++i) {
        pa[i] = rng();
        pb[i] = rng();
      }
      pa[0] = 0x88888888;
      pa[1] = 0x2222aaaa;
      pb[0] = 0x22222222;
      mfma_fp4_stage_a_avx512(pa.data(), reg, af);
      mfma_fp4_stage_b_avx512(pb.data(), reg, bf);
      mfma_fp4_stage_vnni(pa.data(), pb.data(), reg, a, b);
    }
    for (unsigned row = 0; row < 16; ++row)
      for (unsigned k = 0; k < 128; ++k) {
        const int av = (a[row * 32 + k / 4] >> (8 * (k % 4))) & 255;
        const int bv = static_cast<int8_t>(b[(k / 4) * 16 + row] >> (8 * (k % 4)));
        ASSERT_EQ(av - 12, af[row * 128 + k] * 2);
        ASSERT_EQ(bv, bf[k * 16 + row] * 2);
      }
    std::array<uint8_t, 64> as, bs;
    for (unsigned i = 0; i < 64; ++i) {
      as[i] = seed % 3 == 0 ? 65 : seed % 3 == 1 ? 185 : 120 + rng() % 15;
      bs[i] = seed % 3 == 0 ? 65 : seed % 3 == 1 ? 186 : 120 + rng() % 15;
    }
    if (seed >= 16)
      for (unsigned i = 0; i < 64; ++i) {
        as[i] = (i + seed) % 7 == 0 ? 255 : rng() % 256;
        bs[i] = (i + seed) % 11 == 0 ? 255 : rng() % 256;
      }
    for (unsigned denorm = 0; denorm < 4; ++denorm) {
      std::array<float, 256> expected, actual;
      constexpr uint32_t values[] = {0,          0x80000000, 1,          0x80000001,
                                     0x3f800001, 0xbf800001, 0x7fc12345, 0xffc54321,
                                     0x7f812345, 0xff854321, 0x7f7fffff, 0xff7fffff};
      for (unsigned i = 0; i < 256; ++i)
        actual[i] = expected[i] = std::bit_cast<float>(values[(i + seed) % std::size(values)]);
      auto environment = [&] {
        std::fesetenv(FE_DFL_ENV);
        _mm_setcsr((_mm_getcsr() & ~0x8040u) | ((denorm & 1) ? 0x8000u : 0u) |
                   ((denorm & 2) ? 0x40u : 0u));
        std::feraiseexcept(FE_DIVBYZERO);
        errno = EDOM;
      };
      environment();
      scalar_product(af, bf, expected.data(), as.data(), bs.data());
      const int flags = std::fetestexcept(FE_ALL_EXCEPT), error = errno;
      environment();
      if (mfma_fp4_avx512_normal_scales(as.data(), bs.data()))
        mfma_fp4_16x16x128_vnni(a, b, actual.data(), as.data(), bs.data());
      else
        mfma_fp4_16x16x128_vnni<false>(a, b, actual.data(), as.data(), bs.data());
      const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT), actual_error = errno;
      for (unsigned i = 0; i < 256; ++i)
        ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i]))
            << "seed=" << seed << " denorm=" << denorm << " element=" << i;
      EXPECT_EQ(actual_flags, flags);
      EXPECT_EQ(actual_error, error);
    }
  }
}
#endif
#else
TEST(MfmaFp4Avx512, RequiresAvx512Build) {
  GTEST_SKIP() << "AVX-512 kernel unavailable in this build";
}
#endif
} // namespace
