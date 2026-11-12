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

TEST(LdsTest, FourDwordTransfersPreserveMaskAddressingAndLaneOrder) {
  Lds lds(4);
  lds.zero_range(0, 64);
  std::array<uint64_t, 64> addrs{};
  addrs[1] = (1ULL << 32) + 1;
  addrs[63] = (2ULL << 32) + 9;
  constexpr uint64_t mask = (1ULL << 1) | (1ULL << 63);
  constexpr uint32_t offset = 16;
  std::array<uint8_t, 64 * 16> src{};
  for (uint32_t i = 0; i < 16; ++i) {
    src[16 + i] = 0x10 + i;
    src[63 * 16 + i] = 0x80 + i;
  }
  lds.vector_store(addrs.data(), mask, 4, 4, src.data(), offset);

  // The later lane overwrites the overlapping eight bytes.
  for (uint32_t addr = 0; addr < 64; ++addr) {
    const uint8_t expected = addr >= 25 && addr < 41   ? src[63 * 16 + addr - 25]
                             : addr >= 17 && addr < 25 ? src[16 + addr - 17]
                                                       : 0;
    EXPECT_EQ(lds.read8(addr), expected) << addr;
  }

  std::array<uint8_t, 64 * 16> dst;
  dst.fill(0xEE);
  lds.vector_load(addrs.data(), mask, 4, 4, dst.data(), offset);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    for (uint32_t i = 0; i < 16; ++i) {
      const uint8_t expected =
          mask & (1ULL << lane) ? lds.read8(static_cast<uint32_t>(addrs[lane]) + offset + i) : 0xEE;
      EXPECT_EQ(dst[lane * 16 + i], expected) << lane << ":" << i;
    }
  }
}

TEST(LdsTest, FourDwordTransfersClipAtWholeElements) {
  for (uint32_t valid_bytes : {0u, 3u, 8u, 15u, 16u}) {
    Lds lds(4);
    lds.zero_range(0, 4096);
    std::array<uint64_t, 64> addrs{};
    addrs[0] = lds.size_bytes() - valid_bytes;
    std::array<uint8_t, 16> src;
    src.fill(0xAB);
    lds.vector_store(addrs.data(), 1, 4, 4, src.data());
    for (uint32_t i = 0; i < valid_bytes; ++i)
      EXPECT_EQ(lds.read8(static_cast<uint32_t>(addrs[0]) + i), i < valid_bytes / 4 * 4 ? 0xAB : 0)
          << valid_bytes << ":" << i;

    // An incomplete dword must read as zero even when its in-bounds bytes are nonzero.
    for (uint32_t i = valid_bytes / 4 * 4; i < valid_bytes; ++i)
      lds.write8(static_cast<uint32_t>(addrs[0]) + i, 0xD7);

    std::array<uint8_t, 16> dst;
    dst.fill(0xEE);
    lds.vector_load(addrs.data(), 1, 4, 4, dst.data());
    for (uint32_t i = 0; i < 16; ++i)
      EXPECT_EQ(dst[i], i < valid_bytes / 4 * 4 ? 0xAB : 0) << valid_bytes << ":" << i;
  }
}

TEST(LdsTest, FourDwordTransfersCrossUnmaterializedBacking) {
  Lds lds(64);
  lds.zero_range(0, 256);
  const auto boundary = lds.materialized_size_bytes();
  ASSERT_LT(boundary + 16, lds.size_bytes());
  lds.write8(static_cast<uint32_t>(boundary - 1), 0xAB);
  std::array<uint64_t, 64> addrs{};
  addrs[0] = boundary - 7;
  std::array<uint8_t, 16> bytes;
  bytes.fill(0xEE);
  lds.vector_load(addrs.data(), 1, 4, 4, bytes.data());
  for (uint32_t i = 0; i < 16; ++i)
    EXPECT_EQ(bytes[i], i == 6 ? 0xAB : 0);
  EXPECT_EQ(lds.materialized_size_bytes(), boundary);

  bytes.fill(0xCD);
  lds.vector_store(addrs.data(), 1, 4, 4, bytes.data());
  EXPECT_GT(lds.materialized_size_bytes(), boundary);
  bytes.fill(0);
  lds.vector_load(addrs.data(), 1, 4, 4, bytes.data());
  for (uint8_t byte : bytes)
    EXPECT_EQ(byte, 0xCD);
}

} // namespace
