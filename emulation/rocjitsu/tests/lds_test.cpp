// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

using rocjitsu::amdgpu::Lds;

// Per-lane, per-element semantics that vector_load/vector_store must preserve:
// an element reaching past capacity reads as zero and is not written.
void reference_load(const Lds &lds, const std::array<uint64_t, 64> &addrs, uint64_t mask,
                    uint32_t elem_size, uint32_t num_elems, uint8_t *dst) {
  const uint32_t stride = elem_size * num_elems;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(mask >> lane & 1))
      continue;
    for (uint32_t e = 0; e < num_elems; ++e) {
      const uint64_t ea = static_cast<uint32_t>(addrs[lane]) + uint64_t{e} * elem_size;
      uint8_t *out = dst + lane * stride + e * elem_size;
      if (ea + elem_size > lds.size_bytes())
        std::fill_n(out, elem_size, 0);
      else
        lds.read(static_cast<uint32_t>(ea), out, elem_size);
    }
  }
}

void reference_store(Lds &lds, const std::array<uint64_t, 64> &addrs, uint64_t mask,
                     uint32_t elem_size, uint32_t num_elems, const uint8_t *src) {
  const uint32_t stride = elem_size * num_elems;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(mask >> lane & 1))
      continue;
    for (uint32_t e = 0; e < num_elems; ++e) {
      const uint64_t ea = static_cast<uint32_t>(addrs[lane]) + uint64_t{e} * elem_size;
      if (ea + elem_size <= lds.size_bytes())
        lds.write(static_cast<uint32_t>(ea), src + lane * stride + e * elem_size, elem_size);
    }
  }
}

std::vector<uint8_t> contents(const Lds &lds) {
  std::vector<uint8_t> bytes(lds.size_bytes());
  lds.read(0u, bytes.data(), static_cast<uint32_t>(bytes.size()));
  return bytes;
}

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

TEST(LdsTest, OverlappingVectorStoresKeepTheHighestLane) {
  Lds lds(4);
  std::array<uint64_t, 64> addrs{};
  std::array<uint32_t, 64> values{};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    addrs[lane] = 256 + (lane % 4) * 4;
    values[lane] = 0x1000u + lane;
  }
  lds.vector_store(addrs.data(), ~uint64_t{0}, 4, 1,
                   reinterpret_cast<const uint8_t *>(values.data()));
  for (uint32_t slot = 0; slot < 4; ++slot)
    EXPECT_EQ(lds.read32(256 + slot * 4), 0x1000u + 60 + slot);
}

// Descending lanes are contiguous only out of lane order, and the lane at
// capacity - 12 holds one element inside capacity and one reaching past it.
TEST(LdsTest, DescendingLanesAtCapacityKeepPerElementBounds) {
  Lds lds(1);
  const uint64_t cap = lds.size_bytes();
  std::array<uint64_t, 64> addrs{};
  addrs[0] = cap - 12;
  addrs[1] = cap - 28;
  addrs[2] = cap - 44;
  std::array<uint8_t, 64 * 16> src{};
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i + 1);

  lds.vector_store(addrs.data(), 0b111, 8, 2, src.data());
  Lds expected(1);
  reference_store(expected, addrs, 0b111, 8, 2, src.data());
  EXPECT_EQ(contents(lds), contents(expected));
  EXPECT_EQ(lds.read32(static_cast<uint32_t>(cap - 4)), 0u);

  std::array<uint8_t, 64 * 16> loaded{};
  std::array<uint8_t, 64 * 16> reference{};
  lds.vector_load(addrs.data(), 0b111, 8, 2, loaded.data());
  reference_load(expected, addrs, 0b111, 8, 2, reference.data());
  EXPECT_EQ(loaded, reference);
}

// Compare the block-copy path with per-element semantics across contiguous,
// permuted, strided, overlapping, sparse and capacity-crossing operands.
TEST(LdsTest, VectorAccessesMatchPerElementSemantics) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 8> kShapes{
      {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {4, 2}, {4, 3}, {4, 4}, {8, 2}}};
  std::mt19937_64 rng(0x1d5c0a1e5ce);
  for (int iteration = 0; iteration < 4000; ++iteration) {
    const auto [elem_size, num_elems] = kShapes[rng() % kShapes.size()];
    const uint32_t stride = elem_size * num_elems;
    Lds lds(2);
    const uint64_t cap = lds.size_bytes();
    std::array<uint64_t, 64> addrs{};
    const uint32_t pattern = rng() % 5;
    const uint64_t window = pattern == 4 ? 64 : 64 * stride;
    const uint64_t base =
        rng() % 2 ? cap - window + rng() % (2 * stride) - stride : rng() % (cap / 2);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      switch (pattern) {
      case 0: // contiguous
        addrs[lane] = base + uint64_t{lane} * stride;
        break;
      case 1: // permuted within groups of eight lanes
        addrs[lane] = base + uint64_t{(lane & ~7u) | ((lane ^ 5u) & 7u)} * stride;
        break;
      case 2: // descending
        addrs[lane] = base + uint64_t{63 - lane} * stride;
        break;
      case 3: // strided with gaps
        addrs[lane] = base + uint64_t{lane} * stride * 2;
        break;
      default: // random, frequently overlapping
        addrs[lane] = base + rng() % window;
        break;
      }
    }
    const uint64_t mask = rng() % 3 == 0 ? rng() & rng() : ~uint64_t{0};

    // Leave some iterations unmaterialized so block copies cross the lazy backing.
    Lds expected(2);
    if (rng() % 4 != 0) {
      std::vector<uint8_t> initial(cap);
      for (auto &b : initial)
        b = static_cast<uint8_t>(rng());
      lds.write(0u, initial.data(), static_cast<uint32_t>(cap));
      expected.write(0u, initial.data(), static_cast<uint32_t>(cap));
    }

    std::vector<uint8_t> src(64 * stride);
    for (auto &b : src)
      b = static_cast<uint8_t>(rng());
    lds.vector_store(addrs.data(), mask, elem_size, num_elems, src.data());
    reference_store(expected, addrs, mask, elem_size, num_elems, src.data());
    ASSERT_EQ(contents(lds), contents(expected)) << "store iteration " << iteration;
    ASSERT_EQ(lds.materialized_size_bytes(), expected.materialized_size_bytes())
        << "store iteration " << iteration;

    std::vector<uint8_t> loaded(64 * stride, 0xCC);
    std::vector<uint8_t> reference(64 * stride, 0xCC);
    lds.vector_load(addrs.data(), mask, elem_size, num_elems, loaded.data());
    reference_load(expected, addrs, mask, elem_size, num_elems, reference.data());
    ASSERT_EQ(loaded, reference) << "load iteration " << iteration;
  }
}

} // namespace
