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

using rocjitsu::amdgpu::DispatchEntry;
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
  EXPECT_EQ(shard.stride, 1u);
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

// A grid with fewer chunks than XCDs leaves the high-rank shards owning nothing.
// Those empty shards are still handed to their command processors: they are what
// keeps every XCD's view of the queue in step for barrier ordering.
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

// A dispatch entry walks its own share: dispatched_wgs indexes into the shard,
// and chunk_ordinal_for maps that back to a grid-wide ordinal.
TEST(XcdShardTest, EntryWalksItsOwnShareOfTheGrid) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kGridWgs = 256;

  std::multiset<uint32_t> seen;
  for (uint32_t rank = 0; rank < kStride; ++rank) {
    DispatchEntry entry;
    entry.grid_wgs_x = kGridWgs;
    entry.total_wgs = kGridWgs;
    entry.apply_shard({rank, kStride}, kGridWgs);
    EXPECT_EQ(entry.total_wgs, kGridWgs / kStride) << "rank " << rank;

    for (uint32_t i = 0; i < entry.total_wgs; ++i) {
      uint32_t wg = entry.chunk_ordinal_for(i);
      ASSERT_EQ(wg % kStride, rank) << "rank " << rank << " wg " << wg;
      seen.insert(wg);
    }
  }
  EXPECT_EQ(seen.size(), kGridWgs);
}

// A clustered dispatch shards by whole clusters so cluster peers stay
// co-resident on the XCD whose LDS they share.
TEST(XcdShardTest, ClusteredEntryShardsByWholeClusters) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kClusterSize = 4;
  constexpr uint32_t kGridWgs = 256;

  DispatchEntry entry;
  entry.grid_wgs_x = kGridWgs;
  entry.cluster_size_x = kClusterSize;
  entry.cluster_count_x = kGridWgs / kClusterSize;
  entry.total_wgs = kGridWgs;
  ASSERT_EQ(entry.dispatch_chunk_wgs(), kClusterSize);

  entry.apply_shard({3, kStride}, kGridWgs);
  EXPECT_EQ(entry.total_wgs, kGridWgs / kStride);

  // Every owned cluster ordinal belongs to this rank, and the workgroups of a
  // cluster are never split across ranks.
  for (uint32_t i = 0; i < entry.total_wgs / kClusterSize; ++i)
    EXPECT_EQ(entry.chunk_ordinal_for(i) % kStride, 3u);
}

// An unsharded entry must walk the grid exactly as it did before sharding
// existed: chunk index i is workgroup i.
TEST(XcdShardTest, UnshardedEntryWalksTheGridUnchanged) {
  DispatchEntry entry;
  entry.grid_wgs_x = 100;
  entry.total_wgs = 100;
  for (uint32_t i = 0; i < entry.total_wgs; ++i)
    EXPECT_EQ(entry.chunk_ordinal_for(i), i);
}
