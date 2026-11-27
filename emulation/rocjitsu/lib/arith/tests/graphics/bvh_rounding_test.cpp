// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Check BVH's vector rounding against native scalar RTZ arithmetic, including
// widely separated exponents, gradual underflow and overflow. Only the reference
// changes host controls; the vector operations execute under the default state.
#include "common/cpu.h"
#include "graphics/x86_64/bvh_rounding.h"
#include <immintrin.h>

#include <gtest/gtest.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <random>

namespace {

// Volatile reads keep each reference operation inside the selected environment.
[[gnu::noinline]] float reference(float a, float b, int operation) {
  volatile float left = a, right = b;
  switch (operation) {
  case 0:
    return left + right;
  case 1:
    return left - right;
  default:
    return left * right;
  }
}

TEST(BvhRounding, RoundTowardZero) {
  if (!rocjitsu::arith::test::cpu_level_supported(
#if defined(RJ_ARITH_BVH_AVX512)
          RJ_ARITH_CPU_X86_64_V4,
#else
          RJ_ARITH_CPU_X86_64_V3,
#endif
          rj_arith_init_cpu_flags())) {
    GTEST_SKIP();
  }
  ASSERT_EQ(std::fegetround(), FE_TONEAREST);
  using namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE::bvh::bvh_rounding;
  std::minstd_rand rng;
  const auto word = [&] {
    const uint32_t high = rng(), low = rng();
    return (high << 1) | (low >> 30);
  };
  constexpr uint32_t special[] = {0,          0x80000000, 1,          0x80000001, 0x007fffff,
                                  0x00800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
                                  0x7f800000, 0xff800000, 0x7f812345, 0x7fc12345};
  constexpr int count = std::size(special);
  for (int group = 0; group < count * count + 1024; ++group) {
    SCOPED_TRACE(group);
    float a[3], b[3], expected[3][3], actual[3][4];
    for (int lane = 0; lane < 3; ++lane) {
      a[lane] = std::bit_cast<float>(group < count * count ? special[(group / count + lane) % count]
                                                           : word());
      b[lane] =
          std::bit_cast<float>(group < count * count ? special[(group + lane) % count] : word());
    }
    ASSERT_EQ(std::fesetround(FE_TOWARDZERO), 0);
    for (int operation = 0; operation < 3; ++operation) {
      for (int lane = 0; lane < 3; ++lane) {
        expected[operation][lane] = reference(a[lane], b[lane], operation);
      }
    }
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const auto x = pack(a[0], a[1], a[2]), y = pack(b[0], b[1], b[2]);
    const auto controls = _mm_getcsr() & ~_MM_EXCEPT_MASK;
    store(actual[0], add(x, y));
    store(actual[1], subtract(x, y));
    store(actual[2], multiply(x, y));
    EXPECT_EQ(_mm_getcsr() & ~_MM_EXCEPT_MASK, controls); // Arithmetic sticky flags may change.
    for (int operation = 0; operation < 3; ++operation) {
      for (int lane = 0; lane < 3; ++lane) {
        if (std::isnan(expected[operation][lane])) {
          EXPECT_TRUE(std::isnan(actual[operation][lane]));
        } else {
          EXPECT_EQ(std::bit_cast<uint32_t>(actual[operation][lane]),
                    std::bit_cast<uint32_t>(expected[operation][lane]))
              << "operation=" << operation << " lane=" << lane
              << " a_bits=" << std::bit_cast<uint32_t>(a[lane])
              << " b_bits=" << std::bit_cast<uint32_t>(b[lane]);
        }
      }
    }
  }
}

} // namespace
