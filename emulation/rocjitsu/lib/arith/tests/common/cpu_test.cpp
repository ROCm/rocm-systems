// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU feature/OS gating, reserved flag bits, shared enumerations, and readable names.

#include "common/cpu.h"
#include "common/cpu_decode.h"
#include "common/internal.h"
#include "common/semantics.h"
#include "rj_arith/rj_arith.h"

#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>

// Check that host detection returns only defined CPU-level bits.
TEST(Cpu, DetectionReturnsOnlyKnownCpuBits) {
  const uint64_t flags = rj_arith_init_cpu_flags();
  EXPECT_LE(flags & RJ_ARITH_CPU_MASK, RJ_ARITH_CPU_X86_64_ZEN4);
  EXPECT_EQ(flags & ~RJ_ARITH_CPU_MASK, 0ULL);
}

// Check CPU-level masking independently of reserved feature bits.
TEST(Cpu, LevelAndReservedFeatures) {
  EXPECT_EQ(RJ_ARITH_CPU_MASK, 0xfULL);
  EXPECT_EQ(RJ_ARITH_CPU_FEATURES_MASK, 0xfff0ULL);
  for (uint64_t level : {RJ_ARITH_CPU_BASELINE, RJ_ARITH_CPU_X86_64_V3, RJ_ARITH_CPU_X86_64_V4,
                         RJ_ARITH_CPU_X86_64_ZEN4}) {
    EXPECT_EQ(rocjitsu::arith::validate(level, 0), RJ_ARITH_SUCCESS);
    for (int bit = 4; bit < 16; ++bit) {
      EXPECT_EQ(rocjitsu::arith::validate(level | (1ULL << bit), 0), RJ_ARITH_ERROR_INVALID_FLAGS);
    }
  }
  for (uint64_t level = RJ_ARITH_CPU_X86_64_ZEN4 + 1; level <= RJ_ARITH_CPU_MASK; ++level) {
    EXPECT_EQ(rocjitsu::arith::validate(level, 0), RJ_ARITH_ERROR_INVALID_FLAGS);
  }
}

// Check that usable SIMD levels require both CPU features and OS vector-state support.
TEST(Cpu, FeatureAndOsGating) {
  rocjitsu::arith::CpuState s;
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_BASELINE);
  s.leaf1_ecx = s.leaf7_ebx = s.leaf7_ecx = s.leaf71_eax = s.ext1_ecx = UINT32_MAX;
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_BASELINE);
  s.xcr0 = 0x6;
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_X86_64_V3);
  s.xcr0 = 0xe6;
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_X86_64_ZEN4);
  s.leaf71_eax = 0;
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_X86_64_V4);
  s.leaf7_ebx &= ~(1u << 16);
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_X86_64_V3);
  s.leaf1_ecx &= ~(1u << 27);
  EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_BASELINE);
}

// Remove each required feature in turn and verify that the detected CPU level falls.
TEST(Cpu, EveryRequiredFeatureIsChecked) {
  rocjitsu::arith::CpuState full{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0xe6};
  for (int bit : {0, 9, 12, 13, 19, 20, 22, 23, 26, 27, 28, 29}) {
    auto s = full;
    s.leaf1_ecx &= ~(1u << bit);
    EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_BASELINE);
  }
  for (int bit : {1, 6, 8, 9, 10, 11, 12, 14}) {
    auto s = full;
    s.leaf7_ecx &= ~(1u << bit);
    EXPECT_EQ(rocjitsu::arith::decode_cpu(s), RJ_ARITH_CPU_X86_64_V4);
  }
}

// Check the shared CPU-level and semantics enumerations.
TEST(Cpu, SharedEnumerationHelpers) {
  std::vector<uint64_t> supported = rocjitsu::arith::test::supported_cpu_levels();
  ASSERT_FALSE(supported.empty());
  EXPECT_EQ(supported.front(), RJ_ARITH_CPU_BASELINE);
  uint64_t detected = rj_arith_init_cpu_flags();
  EXPECT_EQ(supported.back() & RJ_ARITH_CPU_MASK, detected & RJ_ARITH_CPU_MASK);
  for (uint64_t level : supported) {
    EXPECT_EQ(level & ~RJ_ARITH_CPU_MASK, 0U);
    EXPECT_TRUE(
        rocjitsu::arith::test::cpu_level_supported(level, detected | RJ_ARITH_CPU_FEATURES_MASK));
  }
  const auto semantics = rocjitsu::arith::test::semantics_modes();
  ASSERT_EQ(semantics.size(), 5U);
  EXPECT_EQ(semantics.front(), RJ_ARITH_SEMANTICS_LOOSE);
  EXPECT_EQ(semantics.back(), RJ_ARITH_SEMANTICS_COMPATIBILITY);
}

// Check readable names for supported and invalid CPU levels.
TEST(Cpu, Names) {
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_BASELINE), "scalar");
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_X86_64_V3), "x86-64-v3");
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_X86_64_V4), "x86-64-v4");
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_X86_64_ZEN4), "x86-64-zen4");
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_X86_64_ZEN4 | RJ_ARITH_CPU_FEATURES_MASK |
                                 RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL),
               "x86-64-zen4");
  EXPECT_STREQ(rj_arith_cpu_name(RJ_ARITH_CPU_MASK), "unknown");
}
