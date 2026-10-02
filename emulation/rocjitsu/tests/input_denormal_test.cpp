// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

namespace denorm = rocjitsu::amdgpu::input_denormal;
namespace fmt = rocjitsu::amdgpu::fp_format;

constexpr denorm::Policy kKeep{false};
constexpr denorm::Policy kFlush{true};

TEST(InputDenormalTest, FlushKeepsSignAndSpecials) {
  EXPECT_EQ(denorm::flush_input<fmt::F32>(0x807fffffu, kFlush), 0x80000000u);
  EXPECT_EQ(denorm::flush_input<fmt::F32>(0x007fffffu, kFlush), 0x00000000u);
  EXPECT_EQ(denorm::flush_input<fmt::F32>(0x00800000u, kFlush), 0x00800000u);
  EXPECT_EQ(denorm::flush_input<fmt::F32>(0x7f800001u, kFlush), 0x7f800001u);
  EXPECT_EQ(denorm::flush_input<fmt::F32>(0x807fffffu, kKeep), 0x807fffffu);
  EXPECT_EQ(denorm::flush_input<fmt::F16>(0x83ffu, kFlush), 0x8000u);
  EXPECT_EQ(denorm::flush_input<fmt::F64>(uint64_t{0x800fffffffffffff}, kFlush),
            uint64_t{0x8000000000000000});
}

TEST(InputDenormalTest, FlushesWhenInputDenormalsAreDisabled) {
  EXPECT_TRUE(denorm::Policy::make(0u).flush_inputs);
  EXPECT_TRUE(denorm::Policy::make(2u).flush_inputs);
  EXPECT_FALSE(denorm::Policy::make(1u).flush_inputs);
  EXPECT_FALSE(denorm::Policy::make(3u).flush_inputs);
}

} // namespace
