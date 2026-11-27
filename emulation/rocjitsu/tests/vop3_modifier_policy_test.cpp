// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar NaN and signed-zero clamp policy still used by gfx10 interpolation.

#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/isa_family.h"

#include <bit>
#include <cstdint>
#include <gtest/gtest.h>

TEST(Vop3ModifierPolicy, NanClampFollowsArchitectureAndDx10Mode) {
  using rocjitsu::amdgpu::floating_clamp_nan_to_zero;

  EXPECT_FALSE(floating_clamp_nan_to_zero(rocjitsu::IsaFamily::Cdna4, false));
  EXPECT_TRUE(floating_clamp_nan_to_zero(rocjitsu::IsaFamily::Cdna4, true));
  EXPECT_TRUE(floating_clamp_nan_to_zero(rocjitsu::IsaFamily::Rdna4, false));
  EXPECT_TRUE(floating_clamp_nan_to_zero(rocjitsu::IsaFamily::Cdna5, false));
}

TEST(Vop3ModifierPolicy, ScalarClampHandlesQuietSignalingNanAndSignedZero) {
  using rocjitsu::amdgpu::clamp_floating_result;

  constexpr uint32_t kQuietNan = 0x7FC12345u;
  constexpr uint32_t kSignalingNan = 0x7F812345u;
  constexpr uint32_t kNegativeZero = 0x80000000u;
  for (uint32_t bits : {kQuietNan, kSignalingNan}) {
    const float value = std::bit_cast<float>(bits);
    EXPECT_EQ(std::bit_cast<uint32_t>(clamp_floating_result(value, false)), bits);
    EXPECT_EQ(std::bit_cast<uint32_t>(clamp_floating_result(value, true)), 0u);
  }
  EXPECT_EQ(
      std::bit_cast<uint32_t>(clamp_floating_result(std::bit_cast<float>(kNegativeZero), false)),
      0u);
}
