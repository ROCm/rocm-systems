// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd_shard_test.cpp
/// @brief Round-robin grid sharding across the XCDs of a partition.

#include "rocjitsu/vm/amdgpu/dispatch_entry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

namespace {

using rocjitsu::amdgpu::XcdShard;

/// Enumerate every grid-wide chunk ordinal owned by @p shard.
std::vector<uint32_t> owned_ordinals(XcdShard shard, uint32_t total_chunks) {
  std::vector<uint32_t> ordinals;
  for (uint32_t i = 0; i < shard.owned_chunks(total_chunks); ++i)
    ordinals.push_back(shard.nth_owned_chunk(i));
  return ordinals;
}

} // namespace

TEST(XcdShardTest, DefaultShardOwnsWholeGrid) {
  XcdShard shard;
  EXPECT_TRUE(shard.is_whole_grid());
  EXPECT_EQ(shard.owned_chunks(256), 256u);
  EXPECT_EQ(shard.nth_owned_chunk(0), 0u);
  EXPECT_EQ(shard.nth_owned_chunk(255), 255u);
}

// The permutation is the contract, not just the split: kernels that swizzle
// their workgroup index for cache locality assume chunk c lands on XCD
// c % num_xcds.
TEST(XcdShardTest, ChunkOrdinalMapsToRankModuloStride) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kTotalChunks = 256;

  for (uint32_t rank = 0; rank < kStride; ++rank) {
    XcdShard shard{rank, kStride};
    for (uint32_t ordinal : owned_ordinals(shard, kTotalChunks))
      ASSERT_EQ(ordinal % kStride, rank) << "rank " << rank << " ordinal " << ordinal;
  }
}

TEST(XcdShardTest, ShardsPartitionTheGridExactly) {
  constexpr uint32_t kStride = 8;

  // Include grids that do not divide evenly and grids smaller than the stride.
  for (uint32_t total_chunks : {0u, 1u, 7u, 8u, 9u, 100u, 256u}) {
    std::multiset<uint32_t> seen;
    uint32_t summed = 0;
    for (uint32_t rank = 0; rank < kStride; ++rank) {
      XcdShard shard{rank, kStride};
      summed += shard.owned_chunks(total_chunks);
      for (uint32_t ordinal : owned_ordinals(shard, total_chunks))
        seen.insert(ordinal);
    }

    EXPECT_EQ(summed, total_chunks) << "total_chunks " << total_chunks;
    ASSERT_EQ(seen.size(), total_chunks) << "total_chunks " << total_chunks;
    for (uint32_t ordinal = 0; ordinal < total_chunks; ++ordinal)
      EXPECT_EQ(seen.count(ordinal), 1u) << "ordinal " << ordinal << " of " << total_chunks;
  }
}

// A grid with fewer chunks than XCDs leaves the high-rank shards empty. Those
// shards must not be handed to a command processor: an entry with zero
// workgroups is indistinguishable from a barrier packet.
TEST(XcdShardTest, ShardSmallerThanStrideLeavesHighRanksEmpty) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kTotalChunks = 3;

  for (uint32_t rank = 0; rank < kTotalChunks; ++rank)
    EXPECT_EQ(XcdShard({rank, kStride}).owned_chunks(kTotalChunks), 1u) << "rank " << rank;
  for (uint32_t rank = kTotalChunks; rank < kStride; ++rank)
    EXPECT_EQ(XcdShard({rank, kStride}).owned_chunks(kTotalChunks), 0u) << "rank " << rank;
}

TEST(XcdShardTest, UnevenGridDistributesRemainderToLowRanks) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kTotalChunks = 100; // 12 each, remainder 4

  for (uint32_t rank = 0; rank < 4; ++rank)
    EXPECT_EQ(XcdShard({rank, kStride}).owned_chunks(kTotalChunks), 13u) << "rank " << rank;
  for (uint32_t rank = 4; rank < kStride; ++rank)
    EXPECT_EQ(XcdShard({rank, kStride}).owned_chunks(kTotalChunks), 12u) << "rank " << rank;
}
