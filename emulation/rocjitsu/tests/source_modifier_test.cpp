// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/source_modifier.h"

#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>

namespace {

namespace src = rocjitsu::amdgpu::source_modifier;
namespace fmt = rocjitsu::amdgpu::fp_format;

TEST(SourceModifierTest, RawBitsApplyAbsBeforeNeg) {
  // ABS clears the sign before NEG sets it again.
  EXPECT_EQ(src::apply<fmt::F32>(0x80000002u, true, true), 0x80000002u);
  EXPECT_EQ(src::apply<fmt::F32>(0x80000002u, true, false), 0x00000002u);
}

TEST(SourceModifierTest, FloatAdaptersPreserveNanBitsAndSignedZero) {
  // ABS then NEG leaves this negative signaling NaN's encoding intact.
  const float nan32 = std::bit_cast<float>(0xff800042u);
  EXPECT_EQ(std::bit_cast<uint32_t>(src::apply_to_float(nan32, 0, 1u, 1u)), 0xff800042u);
  // Source 1 uses bit 1 of ABS; neither the payload nor quiet bit changes.
  const double nan64 = std::bit_cast<double>(uint64_t{0xfff0000000000042});
  EXPECT_EQ(std::bit_cast<uint64_t>(src::apply_to_float(nan64, 1, 2u, 0u)),
            uint64_t{0x7ff0000000000042});
  EXPECT_EQ(std::bit_cast<uint32_t>(src::apply_to_float(0.0f, 2, 0u, 4u)), 0x80000000u);
}

} // namespace
