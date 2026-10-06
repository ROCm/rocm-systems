// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simd_correctness/mma_exact_test_support.h"

#include <array>
#include <cerrno>
#include <cfenv>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;

struct HostEnvironment {
  fenv_t saved;
  const int saved_errno = errno;
  HostEnvironment() { std::fegetenv(&saved); }
  ~HostEnvironment() {
    std::fesetenv(&saved);
    errno = saved_errno;
  }
};

void set_environment(int rounding, uint32_t denormal_mode) {
  std::fesetenv(FE_DFL_ENV);
  std::fesetround(rounding);
#if defined(__SSE__)
  _mm_setcsr((_mm_getcsr() & ~0x8040u) | ((denormal_mode & 1u) ? 0x8000u : 0u) |
             ((denormal_mode & 2u) ? 0x40u : 0u));
#else
  (void)denormal_mode;
#endif
  std::feraiseexcept(FE_DIVBYZERO);
  errno = EDOM;
}

[[gnu::noinline]] float scalar_scale(float value, uint8_t a, uint8_t b) {
  if (a == 255 || b == 255)
    return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(value, int(a) + int(b) - 254);
}

TEST(MfmaScaleSimd, MatchesScalarBitsExceptionsAndErrnoAcrossHostModes) {
#if __has_include(<experimental/simd>)
  HostEnvironment restore;
  constexpr size_t W = util::native<float>::size();
  constexpr std::array<uint32_t, 24> values = {
      0u,          0x80000000u, 0x3f800000u, 0xbf800000u, 0x3f800001u, 0xbf800001u,
      0x00800000u, 0x80800000u, 0x00800001u, 0x007fffffu, 0x807fffffu, 0x00000001u,
      0x80000001u, 0x7f7fffffu, 0xff7fffffu, 0x7f000001u, 0x01000001u, 0x3effffffu,
      0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u, 0x7f812345u, 0xff854321u};
  constexpr std::array<uint8_t, 15> scales = {0,   1,   2,   63,  125, 126, 127, 128,
                                              129, 190, 251, 252, 253, 254, 255};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (uint32_t denormal_mode = 0; denormal_mode < 4; ++denormal_mode)
      for (uint8_t a : scales)
        for (uint32_t offset = 0; offset < values.size(); ++offset) {
          std::array<float, W> input;
          std::array<uint8_t, W> b;
          for (uint32_t lane = 0; lane < W; ++lane) {
            input[lane] = std::bit_cast<float>(values[(offset + lane) % values.size()]);
            b[lane] = scales[(offset * 7 + lane) % scales.size()];
          }
          std::array<uint32_t, W> expected;
          set_environment(rounding, denormal_mode);
          for (uint32_t lane = 0; lane < W; ++lane)
            expected[lane] = std::bit_cast<uint32_t>(scalar_scale(input[lane], a, b[lane]));
          const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const int expected_errno = errno;

          set_environment(rounding, denormal_mode);
          util::native<float> sums;
          sums.copy_from(input.data(), util::stdx::element_aligned);
          const auto result = amdgpu::mfma_scale_e8m0_simd(sums, a, b.data());
          const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const int actual_errno = errno;
          std::array<uint32_t, W> actual;
          for (uint32_t lane = 0; lane < W; ++lane)
            actual[lane] = std::bit_cast<uint32_t>(float(result[lane]));
          ASSERT_EQ(actual, expected) << "round=" << rounding << " denorm=" << denormal_mode
                                      << " a=" << int(a) << " offset=" << offset;
          ASSERT_EQ(actual_flags, expected_flags);
          ASSERT_EQ(actual_errno, expected_errno);
        }
#else
  GTEST_SKIP() << "experimental SIMD is unavailable";
#endif
}

TEST(MfmaScaleSimd, NaNScaleDoesNotInspectSignalingInput) {
#if __has_include(<experimental/simd>)
  HostEnvironment restore;
  using F = util::native<float>;
  std::array<float, F::size()> input;
  std::array<uint8_t, F::size()> scales;
  input.fill(1.0f);
  scales.fill(127);
  // Keep the exceptional value dynamic so constant folding cannot bypass
  // classification in the helper under test.
  volatile uint32_t signaling_bits = 0xff8d2490u;
  input[1] = std::bit_cast<float>(uint32_t(signaling_bits));
  scales[1] = 255;
  F sums;
  sums.copy_from(input.data(), util::stdx::element_aligned);
  set_environment(FE_TONEAREST, 0);
  const F result = amdgpu::mfma_scale_e8m0_simd(sums, 176, scales.data());
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
  EXPECT_EQ(errno, EDOM);
  EXPECT_EQ(std::bit_cast<uint32_t>(float(result[1])),
            std::bit_cast<uint32_t>(std::numeric_limits<float>::quiet_NaN()));
#else
  GTEST_SKIP() << "experimental/simd unavailable";
#endif
}

TEST(MfmaScaleSimd, FullAccumulationPreservesNaNPayloadsAndHostModes) {
  SKIP_IF_NO_SIMD();
  HostEnvironment restore;
  mma_exact::ForceScalarGuard restore_scalar;
  constexpr uint32_t A = 0, B = 16, ACC = 32, DST = 64, SA = 96, SB = 97;
  constexpr std::array<uint32_t, 10> accumulator = {
      0u,          0x80000000u, 0x3f800001u, 0xbf800001u, 0x7fc12345u,
      0xffc54321u, 0x7f812345u, 0xff854321u, 0x7f800000u, 0xff800000u};
  constexpr std::array<uint8_t, 8> scales = {0, 1, 126, 127, 128, 253, 254, 255};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    for (uint32_t denormal_mode = 0; denormal_mode < 4; ++denormal_mode)
      for (uint32_t dim : {16u, 32u})
        for (auto format : {mma_exact::Fmt::RAW4, mma_exact::Fmt::FP8, mma_exact::Fmt::BF8}) {
          SCOPED_TRACE(testing::Message() << "round=" << rounding << " denorm=" << denormal_mode
                                          << " dim=" << dim << " format=" << int(format));
          mma_exact::ExactFixture scalar(ROCJITSU_CODE_ARCH_CDNA4, 64);
          mma_exact::ExactFixture simd(ROCJITSU_CODE_ARCH_CDNA4, 64);
          ASSERT_NE(scalar.wf, nullptr);
          ASSERT_NE(simd.wf, nullptr);
          const uint32_t bits = mma_exact::fmt_bits(format);
          auto seed = [&](mma_exact::ExactFixture &fx) {
            fx.seed(A, bits, format, mma_exact::Mode::RandomInt, 15);
            fx.seed(B, bits, format, mma_exact::Mode::RandomInt, 81);
            for (uint32_t lane = 0; lane < 64; ++lane) {
              if (format != mma_exact::Fmt::RAW4 && lane % 4 == 0) {
                fx.cu->write_vgpr(fx.vbase + A, lane, 0x7fff7d7fu);
                fx.cu->write_vgpr(fx.vbase + B, lane, 0xfe7d7cfcu);
              }
              for (uint32_t reg = 0; reg < dim * dim / 64; ++reg)
                fx.cu->write_vgpr(fx.vbase + ACC + reg, lane,
                                  accumulator[(lane + reg) % accumulator.size()]);
              fx.cu->write_vgpr(fx.vbase + SA, lane, scales[lane % scales.size()]);
              fx.cu->write_vgpr(fx.vbase + SB, lane, scales[(lane * 3 + 1) % scales.size()]);
            }
          };
          auto run = [&](mma_exact::ExactFixture &fx) {
            auto execute = [&](auto extract) {
              amdgpu::exec_f32_scaled_mixed(*fx.cu, dim, dim, 2048 / dim, 1, bits, bits,
                                            fx.vbase + DST, fx.vbase + A, fx.vbase + B,
                                            fx.vbase + ACC, extract, extract, amdgpu::ACC_FROM_VGPR,
                                            fx.vbase, 256 + SA, 256 + SB, 0, 0);
            };
            if (format == mma_exact::Fmt::RAW4)
              execute(amdgpu::extract_fp4);
            else if (format == mma_exact::Fmt::FP8)
              execute(amdgpu::extract_fp8);
            else
              execute(amdgpu::extract_bf8);
          };
          seed(scalar);
          seed(simd);
          util::set_force_scalar_for_testing(true);
          set_environment(rounding, denormal_mode);
          run(scalar);
          const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const int expected_errno = errno;
          util::set_force_scalar_for_testing(false);
          set_environment(rounding, denormal_mode);
          run(simd);
          const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
          const int actual_errno = errno;
          EXPECT_EQ(scalar.snapshot(DST, dim * dim / 64), simd.snapshot(DST, dim * dim / 64));
          EXPECT_EQ(actual_flags, expected_flags);
          EXPECT_EQ(actual_errno, expected_errno);
        }
}

} // namespace
