// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <algorithm>
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

TEST(LdsVectorAccessTest, FullWaveContiguousLoadAndStore) {
  Lds lds(4);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kNumElems = 2;
  constexpr uint32_t kStride = kElemSize * kNumElems;

  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64 * kNumElems> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = lane * kStride;
    for (uint32_t elem = 0; elem < kNumElems; ++elem)
      input[lane * kNumElems + elem] = 0x10000000u | (lane << 8) | elem;
  }

  lds.vector_store(addrs.data(), ~uint64_t{0}, kElemSize, kNumElems,
                   reinterpret_cast<const uint8_t *>(input.data()));
  EXPECT_GE(lds.materialized_size_bytes(), input.size() * sizeof(uint32_t));

  std::array<uint32_t, 64 * kNumElems> output{};
  lds.vector_load(addrs.data(), ~uint64_t{0}, kElemSize, kNumElems,
                  reinterpret_cast<uint8_t *>(output.data()));
  EXPECT_EQ(output, input);
}

TEST(LdsVectorAccessTest, FullWaveContiguousAccessHonorsBaseOffsetAtExactEnd) {
  Lds lds(1);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kBaseOffset = 1024 - 64 * kElemSize;
  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = lane * kElemSize;
    input[lane] = 0xe9000000u | lane;
  }

  lds.vector_store(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                   reinterpret_cast<const uint8_t *>(input.data()), kBaseOffset);
  EXPECT_EQ(lds.materialized_size_bytes(), lds.size_bytes());

  std::array<uint32_t, 64> output{};
  lds.vector_load(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                  reinterpret_cast<uint8_t *>(output.data()), kBaseOffset);
  EXPECT_EQ(output, input);
}

TEST(LdsVectorAccessTest, FullWaveUnmaterializedLoadReturnsLogicalZeros) {
  Lds lds(4);
  constexpr uint32_t kStride = sizeof(uint32_t);
  constexpr uint32_t kBase = 1024;

  std::array<uint64_t, 64> addrs{};
  for (uint32_t lane = 0; lane < addrs.size(); ++lane)
    addrs[lane] = kBase + lane * kStride;

  std::array<uint32_t, 64> output{};
  output.fill(0xdeadbeefu);
  lds.vector_load(addrs.data(), ~uint64_t{0}, kStride, 1,
                  reinterpret_cast<uint8_t *>(output.data()));

  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](uint32_t value) { return value == 0; }));
  EXPECT_EQ(lds.materialized_size_bytes(), 0u);
}

TEST(LdsVectorAccessTest, ZeroStrideIsANoOp) {
  Lds lds(1);
  lds.write32(0, 0x12345678);
  std::array<uint64_t, 64> addrs{};
  std::array<uint8_t, 64> data{};
  data.fill(0xa5);

  lds.vector_store(addrs.data(), ~uint64_t{0}, sizeof(uint32_t), 0, data.data());
  lds.vector_load(addrs.data(), ~uint64_t{0}, sizeof(uint32_t), 0, data.data());

  EXPECT_EQ(lds.read32(0), 0x12345678u);
  for (uint8_t value : data)
    EXPECT_EQ(value, 0xa5);
}

TEST(LdsVectorAccessTest, PartialMaskPreservesInactiveLanes) {
  Lds lds(1);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint64_t kMask = (uint64_t{1} << 0) | (uint64_t{1} << 7) | (uint64_t{1} << 63);

  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = lane * kElemSize;
    lds.write32(static_cast<uint32_t>(addrs[lane]), 0x11110000u | lane);
    input[lane] = 0x22220000u | lane;
  }

  lds.vector_store(addrs.data(), kMask, kElemSize, 1,
                   reinterpret_cast<const uint8_t *>(input.data()));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const uint32_t expected = (kMask & (uint64_t{1} << lane)) ? input[lane] : 0x11110000u | lane;
    EXPECT_EQ(lds.read32(static_cast<uint32_t>(addrs[lane])), expected);
  }

  std::array<uint32_t, 64> output{};
  output.fill(0xdeadbeef);
  lds.vector_load(addrs.data(), kMask, kElemSize, 1, reinterpret_cast<uint8_t *>(output.data()));
  for (uint32_t lane = 0; lane < 64; ++lane) {
    const uint32_t expected = (kMask & (uint64_t{1} << lane)) ? input[lane] : 0xdeadbeef;
    EXPECT_EQ(output[lane], expected);
  }
}

TEST(LdsVectorAccessTest, NonContiguousAddressesUseScalarSemantics) {
  Lds lds(1);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = ((lane * 13) % 64) * kElemSize;
    input[lane] = 0xabc00000u | lane;
  }

  lds.vector_store(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                   reinterpret_cast<const uint8_t *>(input.data()));
  std::array<uint32_t, 64> output{};
  lds.vector_load(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                  reinterpret_cast<uint8_t *>(output.data()));

  EXPECT_EQ(output, input);
}

TEST(LdsVectorAccessTest, OutOfBoundsLanesLoadZeroAndDropStores) {
  Lds lds(1);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kBase = 896;
  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = kBase + lane * kElemSize;
    input[lane] = 0x70000000u | lane;
  }

  lds.vector_store(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                   reinterpret_cast<const uint8_t *>(input.data()));
  std::array<uint32_t, 64> output{};
  output.fill(0xffffffff);
  lds.vector_load(addrs.data(), ~uint64_t{0}, kElemSize, 1,
                  reinterpret_cast<uint8_t *>(output.data()));

  for (uint32_t lane = 0; lane < 64; ++lane)
    EXPECT_EQ(output[lane], lane < 32 ? input[lane] : 0u);
}

TEST(LdsVectorAccessTest, NonzeroBaseOffsetAppliesOnScalarFallback) {
  Lds lds(1);
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kBaseOffset = 256;
  constexpr uint64_t kMask = (uint64_t{1} << 1) | (uint64_t{1} << 5);
  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> input{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = lane * kElemSize;
    input[lane] = 0xb0000000u | lane;
  }

  lds.vector_store(addrs.data(), kMask, kElemSize, 1,
                   reinterpret_cast<const uint8_t *>(input.data()), kBaseOffset);
  EXPECT_EQ(lds.read32(kBaseOffset + static_cast<uint32_t>(addrs[1])), input[1]);
  EXPECT_EQ(lds.read32(kBaseOffset + static_cast<uint32_t>(addrs[5])), input[5]);
  EXPECT_EQ(lds.read32(kBaseOffset + static_cast<uint32_t>(addrs[0])), 0u);

  std::array<uint32_t, 64> output{};
  output.fill(0xcccccccc);
  lds.vector_load(addrs.data(), kMask, kElemSize, 1, reinterpret_cast<uint8_t *>(output.data()),
                  kBaseOffset);
  EXPECT_EQ(output[1], input[1]);
  EXPECT_EQ(output[5], input[5]);
  EXPECT_EQ(output[0], 0xccccccccu);
}

} // namespace
