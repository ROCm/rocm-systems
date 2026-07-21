// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

using rocjitsu::amdgpu::Lds;

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

  std::array<uint32_t, 64 * kNumElems> output{};
  lds.vector_load(addrs.data(), ~uint64_t{0}, kElemSize, kNumElems,
                  reinterpret_cast<uint8_t *>(output.data()));
  EXPECT_EQ(output, input);
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
