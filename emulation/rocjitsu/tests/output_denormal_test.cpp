// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

namespace output = rocjitsu::amdgpu::output_denormal;
namespace fmt = rocjitsu::amdgpu::fp_format;

TEST(OutputDenormalTest, FlushesWhenOutputDenormalsAreDisabled) {
  EXPECT_TRUE(output::Policy::make(0u).flush_outputs);
  EXPECT_TRUE(output::Policy::make(1u).flush_outputs);
  EXPECT_FALSE(output::Policy::make(2u).flush_outputs);
  EXPECT_FALSE(output::Policy::make(3u).flush_outputs);
  EXPECT_EQ(output::flush_output<fmt::F32>(0x807fffffu, output::Policy::make(1u)), 0x80000000u);
  EXPECT_EQ(output::flush_output<fmt::F32>(0x807fffffu, output::Policy::make(2u)), 0x807fffffu);
}

} // namespace
