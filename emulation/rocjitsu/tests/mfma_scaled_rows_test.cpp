// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simd_correctness/mma_exact_test_support.h"

#include <array>
#include <cfenv>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace {
using namespace rocjitsu;

struct HostEnvironment {
  fenv_t saved;
  HostEnvironment() { std::fegetenv(&saved); }
  ~HostEnvironment() { std::fesetenv(&saved); }
};

void set_host_mode(int rounding, uint32_t denorm) {
  std::fesetenv(FE_DFL_ENV);
  std::fesetround(rounding);
#if defined(__SSE__)
  _mm_setcsr((_mm_getcsr() & ~0x8040u) | ((denorm & 1u) ? 0x8000u : 0u) |
             ((denorm & 2u) ? 0x40u : 0u));
#else
  (void)denorm;
#endif
  std::feraiseexcept(FE_DIVBYZERO);
}

#if __has_include(<experimental/simd>)
// The previous SIMD loop evaluated one row at a time in increasing K order.
[[gnu::noinline]] util::native<float> one_row_product(const float *a, const float *b,
                                                      uint32_t stride, uint32_t count) {
  util::native<float> sum(0.0f);
  for (uint32_t k = 0; k < count; ++k) {
    util::native<float> av(a[k]);
    util::native<float> bv;
    bv.copy_from(b + k * stride, util::stdx::vector_aligned);
    sum = util::stdx::fma(av, bv, sum);
  }
  return sum;
}
#endif

TEST(MfmaScaledRows, Fp4BlockProductsPreserveOrderAndHostModes) {
#if __has_include(<experimental/simd>)
  HostEnvironment restore;
  constexpr uint32_t W = util::native<float>::size();
  constexpr uint32_t K = 32;
  // Every decoded E2M1 value, including both signs of zero.
  constexpr std::array<uint32_t, 16> patterns = {
      0u,          0x3f000000u, 0x3f800000u, 0x3fc00000u, 0x40000000u, 0x40400000u,
      0x40800000u, 0x40c00000u, 0x80000000u, 0xbf000000u, 0xbf800000u, 0xbfc00000u,
      0xc0000000u, 0xc0400000u, 0xc0800000u, 0xc0c00000u};
  alignas(64) float a[4 * K];
  alignas(64) float b[K * W];
  for (uint32_t offset = 0; offset < patterns.size(); ++offset) {
    for (uint32_t r = 0; r < 4; ++r)
      for (uint32_t k = 0; k < K; ++k)
        a[r * K + k] = std::bit_cast<float>(patterns[(r * 3 + k + offset) % patterns.size()]);
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t lane = 0; lane < W; ++lane)
        b[k * W + lane] = std::bit_cast<float>(patterns[(k * 5 + lane + offset) % patterns.size()]);
    for (uint32_t count : {1u, 3u, 31u, 32u})
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
        for (uint32_t denorm = 0; denorm < 4; ++denorm) {
          SCOPED_TRACE(testing::Message() << "offset=" << offset << " count=" << count
                                          << " round=" << rounding << " denorm=" << denorm);
          set_host_mode(rounding, denorm);
          std::array<util::native<float>, 4> expected;
          for (uint32_t r = 0; r < 4; ++r)
            expected[r] = one_row_product(a + r * K, b, W, count);
          const int expected_flags = std::fetestexcept(FE_ALL_EXCEPT);
          set_host_mode(rounding, denorm);
          const auto actual = amdgpu::mfma_fp4_block_product<4>(a, K, b, W, count);
          const int actual_flags = std::fetestexcept(FE_ALL_EXCEPT);
          for (uint32_t r = 0; r < 4; ++r)
            for (uint32_t lane = 0; lane < W; ++lane)
              ASSERT_EQ(std::bit_cast<uint32_t>(float(actual[r][lane])),
                        std::bit_cast<uint32_t>(float(expected[r][lane])))
                  << "row=" << r << " lane=" << lane;
          EXPECT_EQ(actual_flags, expected_flags);
        }
  }
#else
  GTEST_SKIP() << "experimental SIMD is unavailable";
#endif
}

TEST(MfmaScaledRows, FiniteScalesPreserveMixedInputNaNsAndAliases) {
  SKIP_IF_NO_SIMD();
  mma_exact::ForceScalarGuard restore_scalar;
  constexpr uint32_t A = 0, B = 16, ACC = 32;
  for (uint32_t dim : {16u, 32u})
    for (uint32_t destination : {0u, 16u, 32u, 64u})
      for (bool reverse : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "dim=" << dim << " dst=" << destination << " reverse=" << reverse);
        mma_exact::ExactFixture scalar(ROCJITSU_CODE_ARCH_CDNA4, 64);
        mma_exact::ExactFixture simd(ROCJITSU_CODE_ARCH_CDNA4, 64);
        ASSERT_NE(scalar.wf, nullptr);
        ASSERT_NE(simd.wf, nullptr);
        for (auto *fx : {&scalar, &simd}) {
          fx->seed(A, 8, reverse ? mma_exact::Fmt::BF8 : mma_exact::Fmt::FP8,
                   mma_exact::Mode::RandomInt, 31);
          fx->seed(B, 8, reverse ? mma_exact::Fmt::FP8 : mma_exact::Fmt::BF8,
                   mma_exact::Mode::RandomInt, 79);
          for (uint32_t lane = 0; lane < 64; ++lane) {
            if ((lane & 3u) == 0)
              fx->cu->write_vgpr(fx->vbase + A, lane, reverse ? 0xfd7e7dfeu : 0xff7fff7fu);
            if ((lane & 3u) == 1)
              fx->cu->write_vgpr(fx->vbase + B, lane, reverse ? 0x7fff7fffu : 0xfc7d7cfeu);
            for (uint32_t reg = 0; reg < dim * dim / 64; ++reg)
              fx->cu->write_vgpr(fx->vbase + ACC + reg, lane,
                                 lane & 1u ? 0xffc54321u : 0x7fc12345u);
          }
        }
        auto run = [&](mma_exact::ExactFixture &fx) {
          auto execute = [&](auto extract_a, auto extract_b) {
            amdgpu::exec_f32_scaled_mixed(
                *fx.cu, dim, dim, 2048 / dim, 1, 8, 8, fx.vbase + destination, fx.vbase + A,
                fx.vbase + B, fx.vbase + ACC, extract_a, extract_b, amdgpu::ACC_FROM_VGPR, fx.vbase,
                /*+1.0f=*/242, /*+1.0f=*/242, 0, 0);
          };
          if (reverse)
            execute(amdgpu::extract_bf8, amdgpu::extract_fp8);
          else
            execute(amdgpu::extract_fp8, amdgpu::extract_bf8);
        };
        util::set_force_scalar_for_testing(true);
        run(scalar);
        util::set_force_scalar_for_testing(false);
        run(simd);
        EXPECT_EQ(simd.snapshot(destination, dim * dim / 64),
                  scalar.snapshot(destination, dim * dim / 64));
      }
}

} // namespace
