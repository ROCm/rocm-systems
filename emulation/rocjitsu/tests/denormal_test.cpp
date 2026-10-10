// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_format.h"
#include "util/simd.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

namespace flush = rocjitsu::amdgpu::denormal;
namespace fmt = rocjitsu::amdgpu::fp_format;

TEST(DenormalTest, FlushIsPolicyFree) {
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

// Every class other than a subnormal passes through bit for bit, including NaN
// payloads, on SIMD lanes as on scalar ones.
TEST(DenormalTest, FlushValueSimdKeepsEveryOtherClass) {
  using V = util::native<float>;
  constexpr std::pair<uint32_t, uint32_t> kCases[] = {
      {0x00000001u, 0x00000000u}, {0x80000001u, 0x80000000u}, {0x007fffffu, 0x00000000u},
      {0x807fffffu, 0x80000000u}, {0x00000000u, 0x00000000u}, {0x80000000u, 0x80000000u},
      {0x00800000u, 0x00800000u}, {0x3f800000u, 0x3f800000u}, {0x7f800000u, 0x7f800000u},
      {0xff800000u, 0xff800000u}, {0x7fc00000u, 0x7fc00000u}, {0x7f800001u, 0x7f800001u},
  };
  for (const auto &[in, out] : kCases) {
    const V flushed = flush::flush_value(V(std::bit_cast<float>(in)));
    for (std::size_t i = 0; i < V::size(); ++i)
      EXPECT_EQ(std::bit_cast<uint32_t>(float(flushed[i])), out) << std::hex << in;
  }
}

} // namespace
