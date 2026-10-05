// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_denormal.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <limits>

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

TEST(DenormalTest, FlushIsPolicyFree) {
  namespace flush = rocjitsu::amdgpu::denormal;
  EXPECT_EQ(flush::flush<fmt::F32>(0x00000001u), 0x00000000u);
  EXPECT_EQ(flush::flush<fmt::F32>(0x807fffffu), 0x80000000u);
  EXPECT_EQ(flush::flush<fmt::F32>(0x80800000u), 0x80800000u);
  EXPECT_EQ(flush::flush<fmt::F32>(0xff800000u), 0xff800000u);
  EXPECT_EQ(flush::flush<fmt::F32>(0x7fc00001u), 0x7fc00001u);
  EXPECT_EQ(flush::flush<fmt::F16>(0x0001u), 0x0000u);
  EXPECT_EQ(flush::flush<fmt::F16>(0x8400u), 0x8400u);
  EXPECT_EQ(flush::flush<fmt::F64>(uint64_t{0x0000000000000001}), uint64_t{0});
  EXPECT_EQ(flush::flush<fmt::F64>(uint64_t{0x8010000000000000}), uint64_t{0x8010000000000000});
}

TEST(DenormalTest, FlushValueUsesTheEncoding) {
  namespace flush = rocjitsu::amdgpu::denormal;
  const float tiny = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(std::bit_cast<uint32_t>(flush::flush_value(-tiny)), 0x80000000u);
  EXPECT_EQ(flush::flush_value(1.5f), 1.5f);
  const double tiny64 = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(std::bit_cast<uint64_t>(flush::flush_value(-tiny64)), uint64_t{0x8000000000000000});
  EXPECT_EQ(flush::flush_value(-2.0), -2.0);

  using V = util::native<float>;
  V lanes([&](auto i) { return i % 2 == 0 ? -tiny : float(i); });
  const V flushed = flush::flush_value(lanes);
  for (std::size_t i = 0; i < V::size(); ++i)
    EXPECT_EQ(std::bit_cast<uint32_t>(float(flushed[i])),
              i % 2 == 0 ? 0x80000000u : std::bit_cast<uint32_t>(float(i)));
}

TEST(OutputDenormalTest, FlushesWhenOutputDenormalsAreDisabled) {
  namespace output = rocjitsu::amdgpu::output_denormal;
  EXPECT_TRUE(output::Policy::make(0u).flush_outputs);
  EXPECT_TRUE(output::Policy::make(1u).flush_outputs);
  EXPECT_FALSE(output::Policy::make(2u).flush_outputs);
  EXPECT_FALSE(output::Policy::make(3u).flush_outputs);
  EXPECT_EQ(output::flush_output<fmt::F32>(0x807fffffu, output::Policy::make(1u)), 0x80000000u);
  EXPECT_EQ(output::flush_output<fmt::F32>(0x807fffffu, output::Policy::make(2u)), 0x807fffffu);
}

} // namespace
