// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

using rocjitsu::amdgpu::Lds;

// Model bytes rather than replaying the implementation's lane/element traversal.
// Neither oracle calls Lds accessors or shares traversal helpers with production.
std::vector<uint8_t> reference_load(const std::vector<uint8_t> &memory,
                                    const std::array<uint64_t, 64> &addrs, uint64_t mask,
                                    uint32_t elem_size, uint32_t num_elems) {
  const uint32_t stride = elem_size * num_elems;
  std::vector<uint8_t> result(64 * stride, 0xCC);
  for (size_t out = 0; out < result.size(); ++out) {
    const size_t lane = out / stride;
    if (!(mask >> lane & 1))
      continue;
    const size_t within_lane = out % stride;
    const uint64_t element =
        uint64_t{static_cast<uint32_t>(addrs[lane])} + (within_lane / elem_size) * elem_size;
    result[out] =
        element + elem_size <= memory.size() ? memory[element + within_lane % elem_size] : 0;
  }
  return result;
}

std::vector<uint8_t> reference_store(const std::vector<uint8_t> &memory,
                                     const std::array<uint64_t, 64> &addrs, uint64_t mask,
                                     uint32_t elem_size, uint32_t num_elems,
                                     const std::vector<uint8_t> &src) {
  const uint32_t stride = elem_size * num_elems;
  auto result = memory;
  // For each LDS byte, find the highest active lane with a valid element covering it.
  // This defines the winner directly, independently of the store traversal order.
  for (size_t address = 0; address < result.size(); ++address) {
    for (int lane = 63; lane >= 0; --lane) {
      if (!(mask >> lane & 1))
        continue;
      const uint64_t base = uint64_t{static_cast<uint32_t>(addrs[lane])};
      if (address < base || address - base >= stride)
        continue;
      const size_t within_lane = address - base;
      const uint64_t element = base + (within_lane / elem_size) * elem_size;
      if (element + elem_size <= memory.size()) {
        result[address] = src[lane * stride + within_lane];
        break;
      }
    }
  }
  return result;
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

TEST(LdsTest, ContiguousAndSparseLanesMatchExplicitBytes) {
  constexpr std::array<uint8_t, 16> initial{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  constexpr std::array<uint8_t, 16> values{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  struct Case {
    uint64_t mask;
    std::array<uint8_t, 16> loaded;
    std::array<uint8_t, 16> stored;
  };
  const std::array<Case, 2> cases{{
      {0b1111, initial, values},
      {0b1011,
       {1, 2, 3, 4, 5, 6, 7, 8, 0xCC, 0xCC, 0xCC, 0xCC, 13, 14, 15, 16},
       {16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 4, 3, 2, 1}},
  }};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.mask);
    Lds lds(1);
    lds.write(0u, initial.data(), initial.size());
    std::array<uint64_t, 64> addrs{0, 4, 8, 12};
    std::array<uint8_t, 64 * 4> loaded;
    loaded.fill(0xCC);
    lds.vector_load(addrs.data(), test.mask, 2, 2, loaded.data());
    std::array<uint8_t, 64 * 4> expected_load;
    expected_load.fill(0xCC);
    std::copy(test.loaded.begin(), test.loaded.end(), expected_load.begin());
    EXPECT_EQ(loaded, expected_load);

    std::array<uint8_t, 64 * 4> src{};
    std::copy(values.begin(), values.end(), src.begin());
    lds.vector_store(addrs.data(), test.mask, 2, 2, src.data());
    std::vector<uint8_t> expected_store(lds.size_bytes(), 0);
    std::copy(test.stored.begin(), test.stored.end(), expected_store.begin());
    EXPECT_EQ(contents(lds), expected_store);
  }
}

TEST(LdsTest, EmptyMaskLeavesMemoryAndOutputUntouched) {
  Lds lds(1);
  std::array<uint64_t, 64> addrs{};
  std::array<uint8_t, 64 * 4> bytes;
  bytes.fill(0xCC);
  const auto expected = bytes;
  lds.vector_load(addrs.data(), 0, 4, 1, bytes.data());
  EXPECT_EQ(bytes, expected);
  lds.vector_store(addrs.data(), 0, 4, 1, bytes.data());
  EXPECT_EQ(lds.materialized_size_bytes(), 0u);
  EXPECT_EQ(contents(lds), std::vector<uint8_t>(lds.size_bytes(), 0));
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
  std::vector<uint8_t> expected(cap, 0);
  std::copy_n(src.begin(), 8, expected.begin() + cap - 12);
  std::copy_n(src.begin() + 16, 16, expected.begin() + cap - 28);
  std::copy_n(src.begin() + 32, 16, expected.begin() + cap - 44);
  EXPECT_EQ(contents(lds), expected);

  // A straddling element must zero-fill even when its in-range prefix is nonzero.
  lds.write32(static_cast<uint32_t>(cap - 4), 0xA5A5A5A5u);
  std::array<uint8_t, 64 * 16> loaded;
  loaded.fill(0xCC);
  lds.vector_load(addrs.data(), 0b111, 8, 2, loaded.data());
  auto expected_load = src;
  std::fill(expected_load.begin() + 8, expected_load.begin() + 16, 0);
  std::fill(expected_load.begin() + 48, expected_load.end(), 0xCC);
  EXPECT_EQ(loaded, expected_load);
}

// vector_load_span() and vector_load32() may write inactive lanes, but every
// active lane must hold vector_load()'s bytes: whole contiguous waves,
// independent 16-lane segments, b64 rows where lane l + 16 reads 8 bytes after
// lane l, broadcast and scattered lanes, at the capacity and the lazy backing,
// and with offsets that wrap 2^32.
TEST(LdsTest, SpanLoadsMatchVectorLoadOnActiveLanes) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 8> kShapes{
      {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {4, 2}, {4, 3}, {4, 4}, {8, 2}}};
  std::mt19937_64 rng(0x5ba7c0de);
  for (int iteration = 0; iteration < 20000; ++iteration) {
    const auto [elem_size, num_elems] = kShapes[rng() % kShapes.size()];
    const uint32_t stride = elem_size * num_elems;
    Lds lds(4);
    const uint32_t cap = static_cast<uint32_t>(lds.size_bytes());
    // Back a random prefix (possibly none) with random bytes.
    std::vector<uint8_t> initial(rng() % 4 == 0 ? 0 : rng() % (cap + 1));
    for (auto &b : initial)
      b = static_cast<uint8_t>(rng());
    if (!initial.empty())
      lds.write(0u, initial.data(), static_cast<uint32_t>(initial.size()));
    const uint32_t backed = static_cast<uint32_t>(lds.materialized_size_bytes());
    const uint32_t edges[] = {0, backed, cap, cap / 2};
    const auto near_edge = [&] {
      return edges[rng() % 4] - static_cast<uint32_t>(rng() % (80 * stride)) +
             static_cast<uint32_t>(rng() % 24);
    };
    std::array<uint32_t, 64> target{};
    switch (rng() % 5) {
    case 0: { // one contiguous wave
      const uint32_t base = near_edge();
      for (uint32_t lane = 0; lane < 64; ++lane)
        target[lane] = base + lane * stride;
      break;
    }
    case 1: // independent 16-lane segments
      for (uint32_t segment = 0; segment < 64; segment += 16) {
        const uint32_t base = rng() % 3 ? near_edge() : static_cast<uint32_t>(rng());
        for (uint32_t lane = 0; lane < 16; ++lane)
          target[segment + lane] = base + lane * stride;
      }
      break;
    case 2: // b64-style rows: lane l + 16 reads 8 bytes after lane l, mostly backed
      for (uint32_t lane = 0; lane < 16; ++lane) {
        const bool inside = backed >= 16 && rng() % 64 != 0;
        target[lane] = inside ? static_cast<uint32_t>(rng() % (backed - 15)) : near_edge();
        target[lane + 16] = target[lane] + 8;
        target[lane + 32] = near_edge();
        target[lane + 48] = target[lane + 32] + 8;
      }
      break;
    case 3: { // broadcast
      const uint32_t base = near_edge();
      target.fill(base);
      break;
    }
    default: // scattered
      for (auto &address : target)
        address = rng() % 8 ? near_edge() : static_cast<uint32_t>(rng());
      break;
    }
    uint64_t mask = rng();
    switch (rng() % 7) {
    case 6:
      mask = 0xFFFF'FFFFull;
      break; // a 32-lane span, where b64 rows pair up
    case 0:
      mask = ~uint64_t{0};
      break;
    case 1:
      mask = 0xFFFF'FFFFull;
      break;
    case 2:
      mask = 0xFFFF'FFFF'FFFFull;
      break;
    case 3:
      mask = uint64_t{1} << (rng() % 64);
      break;
    case 4:
      mask &= rng() & rng();
      break;
    default:
      break;
    }
    const uint32_t offsets[] = {0, 37, 0xFFFF'FFF0u};
    const uint32_t offset = offsets[rng() % 3];
    std::array<uint32_t, 64> lanes{};
    std::array<uint64_t, 64> addrs{};
    for (uint32_t lane = 0; lane < 64; ++lane) {
      lanes[lane] = target[lane] - offset;
      addrs[lane] = target[lane];
    }
    std::vector<uint8_t> expected(64 * stride, 0xCC), span(64 * stride, 0x5A),
        native(64 * stride, 0xA5);
    lds.vector_load(addrs.data(), mask, elem_size, num_elems, expected.data());
    lds.vector_load_span(addrs.data(), mask, elem_size, num_elems, span.data());
    lds.vector_load32(lanes.data(), offset, mask, elem_size, num_elems, native.data());
    for (uint64_t active = mask; active; active &= active - 1) {
      const uint32_t at = static_cast<uint32_t>(std::countr_zero(active)) * stride;
      ASSERT_TRUE(std::equal(span.begin() + at, span.begin() + at + stride, expected.begin() + at))
          << "vector_load_span iteration " << iteration << " lane " << at / stride;
      ASSERT_TRUE(
          std::equal(native.begin() + at, native.begin() + at + stride, expected.begin() + at))
          << "vector_load32 iteration " << iteration << " lane " << at / stride;
    }
  }
}

// Compare the block-copy path with per-element semantics across contiguous,
// permuted, strided, overlapping, sparse and capacity-crossing operands.
TEST(LdsTest, VectorAccessesMatchPerElementSemantics) {
  constexpr std::array<std::pair<uint32_t, uint32_t>, 8> kShapes{
      {{1, 1}, {2, 1}, {4, 1}, {8, 1}, {4, 2}, {4, 3}, {4, 4}, {8, 2}}};
  enum class Pattern { Contiguous, Permuted, Descending, Strided, Overlapping };
  constexpr std::array kPatterns{Pattern::Contiguous, Pattern::Permuted, Pattern::Descending,
                                 Pattern::Strided, Pattern::Overlapping};
  std::mt19937_64 rng(0x1d5c0a1e5ce);
  for (int iteration = 0; iteration < 4000; ++iteration) {
    const auto [elem_size, num_elems] = kShapes[rng() % kShapes.size()];
    const uint32_t stride = elem_size * num_elems;
    Lds lds(2);
    const uint64_t cap = lds.size_bytes();
    std::array<uint64_t, 64> addrs{};
    const Pattern pattern = kPatterns[rng() % kPatterns.size()];
    const uint64_t window = pattern == Pattern::Overlapping ? 64 : 64 * stride;
    const uint64_t base =
        rng() % 2 ? cap - window + rng() % (2 * stride) - stride : rng() % (cap / 2);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      switch (pattern) {
      case Pattern::Contiguous:
        addrs[lane] = base + uint64_t{lane} * stride;
        break;
      case Pattern::Permuted: // permuted within groups of eight lanes
        addrs[lane] = base + uint64_t{(lane & ~7u) | ((lane ^ 5u) & 7u)} * stride;
        break;
      case Pattern::Descending:
        addrs[lane] = base + uint64_t{63 - lane} * stride;
        break;
      case Pattern::Strided: // strided with gaps
        addrs[lane] = base + uint64_t{lane} * stride * 2;
        break;
      case Pattern::Overlapping:
        addrs[lane] = base + rng() % window;
        break;
      }
    }
    const uint64_t mask = rng() % 3 == 0 ? rng() & rng() : ~uint64_t{0};

    // Leave some iterations unmaterialized so block copies cross the lazy backing.
    std::vector<uint8_t> initial(cap, 0);
    if (rng() % 4 != 0) {
      for (auto &b : initial)
        b = static_cast<uint8_t>(rng());
      lds.write(0u, initial.data(), static_cast<uint32_t>(cap));
    }

    // Test loads before stores so matching load/store mistakes cannot cancel out.
    std::vector<uint8_t> loaded(64 * stride, 0xCC);
    lds.vector_load(addrs.data(), mask, elem_size, num_elems, loaded.data());
    ASSERT_EQ(loaded, reference_load(initial, addrs, mask, elem_size, num_elems))
        << "initial load iteration " << iteration;

    std::vector<uint8_t> src(64 * stride);
    for (auto &b : src)
      b = static_cast<uint8_t>(rng());
    const auto expected = reference_store(initial, addrs, mask, elem_size, num_elems, src);
    lds.vector_store(addrs.data(), mask, elem_size, num_elems, src.data());
    ASSERT_EQ(contents(lds), expected) << "store iteration " << iteration;

    std::fill(loaded.begin(), loaded.end(), 0xCC);
    lds.vector_load(addrs.data(), mask, elem_size, num_elems, loaded.data());
    ASSERT_EQ(loaded, reference_load(expected, addrs, mask, elem_size, num_elems))
        << "load after store iteration " << iteration;
  }
}

} // namespace
