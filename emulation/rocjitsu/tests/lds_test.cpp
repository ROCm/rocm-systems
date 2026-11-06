// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

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

TEST(LdsTest, VectorShapesPreserveMasksOffsetsAndElementBounds) {
  constexpr std::array<std::array<uint32_t, 2>, 8> shapes = {
      {{1, 1}, {2, 1}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {8, 1}, {3, 2}}};
  for (const std::array<uint32_t, 2> &shape : shapes) {
    SCOPED_TRACE(testing::Message() << "bytes=" << shape[0] << " count=" << shape[1]);
    Lds lds(1);
    const uint32_t elem_size = shape[0], count = shape[1], stride = elem_size * count;
    const uint32_t capacity = static_cast<uint32_t>(lds.size_bytes());
    for (uint32_t byte = 0; byte < capacity; ++byte)
      lds.write8(byte, static_cast<uint8_t>(byte * 13u + 7u));
    constexpr uint32_t offset = 4;
    const uint32_t tail = count > 1 ? capacity - elem_size * (count - 1) : capacity - elem_size + 1;
    std::array<uint64_t, 64> addresses{};
    addresses[0] = 1;
    addresses[1] = (uint64_t{1} << 32) | 17u;
    addresses[63] = tail - offset;
    constexpr uint64_t lanes = 3u | (uint64_t{1} << 63);
    std::vector<uint8_t> output(64 * stride, 0xA5);
    lds.vector_load(addresses.data(), lanes, elem_size, count, output.data(), offset);
    for (uint32_t lane = 0; lane < 64; ++lane)
      for (uint32_t byte = 0; byte < stride; ++byte) {
        uint8_t expected = 0xA5;
        if (lanes & (uint64_t{1} << lane)) {
          const uint64_t base = uint64_t{static_cast<uint32_t>(addresses[lane])} + offset;
          const uint64_t element = base + (byte / elem_size) * elem_size;
          expected =
              element + elem_size <= capacity ? static_cast<uint8_t>((base + byte) * 13u + 7u) : 0;
        }
        EXPECT_EQ(output[lane * stride + byte], expected) << "lane=" << lane << " byte=" << byte;
      }
    std::vector<uint8_t> tail_input(64 * stride, 0xCC);
    lds.vector_store(addresses.data(), uint64_t{1} << 63, elem_size, count, tail_input.data(),
                     offset);
    for (uint32_t byte = 0; byte < stride && tail + byte < capacity; ++byte) {
      const bool complete = tail + (byte / elem_size) * elem_size + elem_size <= capacity;
      EXPECT_EQ(lds.read8(tail + byte),
                complete ? 0xCC : static_cast<uint8_t>((tail + byte) * 13u + 7u));
    }
    EXPECT_EQ(lds.read8(tail - 1), static_cast<uint8_t>((tail - 1) * 13u + 7u));
    // Every active store aliases the same unaligned range; the last active lane wins.
    addresses.fill(1);
    std::vector<uint8_t> input(64 * stride);
    for (uint32_t lane = 0; lane < 64; ++lane)
      for (uint32_t byte = 0; byte < stride; ++byte)
        input[lane * stride + byte] = static_cast<uint8_t>(lane + byte);
    lds.vector_store(addresses.data(), lanes, elem_size, count, input.data(), offset);
    EXPECT_EQ(lds.read8(offset), static_cast<uint8_t>(offset * 13u + 7u));
    for (uint32_t byte = 0; byte < stride; ++byte)
      EXPECT_EQ(lds.read8(1 + offset + byte), static_cast<uint8_t>(63 + byte));
    EXPECT_EQ(lds.read8(1 + offset + stride),
              static_cast<uint8_t>((1 + offset + stride) * 13u + 7u));
    Lds untouched(1);
    untouched.vector_store(addresses.data(), 0, elem_size, count, input.data(), offset);
    untouched.vector_load(addresses.data(), 0, elem_size, count, output.data(), offset);
    EXPECT_EQ(untouched.materialized_size_bytes(), 0u);
    addresses.fill(capacity - elem_size + 1);
    untouched.vector_store(addresses.data(), lanes, elem_size, count, input.data());
    EXPECT_EQ(untouched.materialized_size_bytes(), 0u);
  }
}

} // namespace
