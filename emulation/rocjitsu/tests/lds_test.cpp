// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

using rocjitsu::amdgpu::Lds;

TEST(LdsTest, ConstructionKeepsLogicalCapacityUnmaterialized) {
  Lds lds(320);

  EXPECT_EQ(lds.size_bytes(), 320u * 1024u);
  EXPECT_EQ(lds.materialized_size_bytes(), 0u);
  EXPECT_EQ(lds.read32(0), 0u);
  EXPECT_EQ(lds.read32(320u * 1024u - sizeof(uint32_t)), 0u);
  EXPECT_EQ(lds.materialized_size_bytes(), 0u);
}

TEST(LdsTest, ZeroRangeMaterializesOnlyAWorkingPrefix) {
  Lds lds(320);

  lds.zero_range(512, 256);
  EXPECT_GE(lds.materialized_size_bytes(), 768u);
  EXPECT_LT(lds.materialized_size_bytes(), lds.size_bytes());

  lds.write32(700, 0xA5A5A5A5u);
  lds.zero_range(512, 256);
  EXPECT_EQ(lds.read32(700), 0u);
  EXPECT_LT(lds.materialized_size_bytes(), lds.size_bytes());
}

TEST(LdsTest, ReadsAcrossBackingBoundaryPreserveImplicitZeros) {
  Lds lds(64);
  lds.zero_range(0, 256);
  const auto boundary = lds.materialized_size_bytes();
  ASSERT_GE(boundary, 256u);
  ASSERT_LT(boundary, lds.size_bytes());
  lds.write8(static_cast<uint32_t>(boundary - 1), 0xAB);

  EXPECT_EQ(lds.read16(static_cast<uint32_t>(boundary - 1)), 0x00ABu);
  std::array<uint8_t, 4> bytes{0xFF, 0xFF, 0xFF, 0xFF};
  static_cast<const Lds &>(lds).read(static_cast<uint32_t>(boundary - 1), bytes.data(),
                                     static_cast<uint32_t>(bytes.size()));
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0xAB, 0, 0, 0}));
  EXPECT_EQ(lds.materialized_size_bytes(), boundary);
}

TEST(LdsTest, VectorAccessesCrossBackingBoundaryLazily) {
  Lds lds(64);
  lds.zero_range(0, 256);
  const auto boundary = lds.materialized_size_bytes();
  ASSERT_LT(boundary, lds.size_bytes());
  lds.write8(static_cast<uint32_t>(boundary - 1), 0xAB);

  std::array<uint64_t, 64> addrs{};
  addrs[0] = boundary - 1;
  std::array<uint8_t, 4> bytes{0xFF, 0xFF, 0xFF, 0xFF};
  lds.vector_load(addrs.data(), 1, bytes.size(), 1, bytes.data());
  EXPECT_EQ(bytes, (std::array<uint8_t, 4>{0xAB, 0, 0, 0}));
  EXPECT_EQ(lds.materialized_size_bytes(), boundary);

  bytes = {0x11, 0x22, 0x33, 0x44};
  lds.vector_store(addrs.data(), 1, bytes.size(), 1, bytes.data());
  EXPECT_GT(lds.materialized_size_bytes(), boundary);
  std::array<uint8_t, 4> stored{};
  static_cast<const Lds &>(lds).read(static_cast<uint32_t>(boundary - 1), stored.data(),
                                     static_cast<uint32_t>(stored.size()));
  EXPECT_EQ(stored, bytes);
}

TEST(LdsTest, WritesGrowBackingAndOutOfBoundsWritesDoNot) {
  Lds lds(64);
  const uint32_t last_word = static_cast<uint32_t>(lds.size_bytes() - sizeof(uint32_t));

  lds.write32(last_word, 0x12345678u);
  EXPECT_EQ(lds.read32(last_word), 0x12345678u);
  EXPECT_EQ(lds.materialized_size_bytes(), lds.size_bytes());

  Lds untouched(64);
  untouched.write32(static_cast<uint32_t>(untouched.size_bytes() - 2), 0xFFFFFFFFu);
  EXPECT_EQ(untouched.materialized_size_bytes(), 0u);
}

} // namespace
