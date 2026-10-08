// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd_shard_test.cpp
/// @brief Round-robin splitting of an ordinal range across a participating set.

#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/xcd_shard.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
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

TEST(XcdShardTest, DefaultShardIsUnshardedAndOwnsEveryOrdinal) {
  XcdShard shard;
  EXPECT_EQ(shard.rank(), 0u);
  EXPECT_EQ(shard.stride(), 1u);
  EXPECT_TRUE(shard.is_unsharded());
  EXPECT_EQ(shard.owned_chunks(256), 256u);
  EXPECT_EQ(shard.nth_owned_chunk(0), 0u);
  EXPECT_EQ(shard.nth_owned_chunk(255), 255u);
}

// The predicate asks whether anything is being split, which is not the same
// question as whether this shard happens to cover the range it is asked about.
// A one-chunk range is the case that separates them: rank 0 owns all of it and
// rank 1 owns none, yet neither is unsharded.
TEST(XcdShardTest, IsUnshardedAsksAboutTheShardNotItsCoverage) {
  EXPECT_TRUE(XcdShard(0, 1).is_unsharded());
  EXPECT_FALSE(XcdShard(0, 8).is_unsharded());
  EXPECT_FALSE(XcdShard(7, 8).is_unsharded());

  EXPECT_EQ(XcdShard(0, 8).owned_chunks(1), 1u);
  EXPECT_EQ(XcdShard(1, 8).owned_chunks(1), 0u);
  EXPECT_FALSE(XcdShard(0, 8).is_unsharded());
  EXPECT_FALSE(XcdShard(1, 8).is_unsharded());
}

// The permutation is the contract, not just the split: a caller that swizzles
// its chunk index for locality assumes chunk c belongs to the member at
// c % participants. What that member stands for is the caller's business.
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

// A range with fewer chunks than participants leaves the high-rank shards
// empty. What a caller does with an empty share is its own policy; this type
// only promises the count.
TEST(XcdShardTest, ShardSmallerThanStrideLeavesHighRanksEmpty) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kTotalChunks = 3;

  for (uint32_t rank = 0; rank < kTotalChunks; ++rank)
    EXPECT_EQ(XcdShard(rank, kStride).owned_chunks(kTotalChunks), 1u) << "rank " << rank;
  for (uint32_t rank = kTotalChunks; rank < kStride; ++rank)
    EXPECT_EQ(XcdShard(rank, kStride).owned_chunks(kTotalChunks), 0u) << "rank " << rank;
}

TEST(XcdShardTest, UnevenGridDistributesRemainderToLowRanks) {
  constexpr uint32_t kStride = 8;
  constexpr uint32_t kTotalChunks = 100; // 12 each, remainder 4

  for (uint32_t rank = 0; rank < 4; ++rank)
    EXPECT_EQ(XcdShard(rank, kStride).owned_chunks(kTotalChunks), 13u) << "rank " << rank;
  for (uint32_t rank = 4; rank < kStride; ++rank)
    EXPECT_EQ(XcdShard(rank, kStride).owned_chunks(kTotalChunks), 12u) << "rank " << rank;
}

// A maximal grid must not wrap. The usual (n + stride - 1) / stride ceiling
// overflows uint32 here and reports zero chunks, which would silently drop
// almost the entire grid rather than fail loudly.
TEST(XcdShardTest, MaximalGridDoesNotOverflow) {
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  constexpr uint32_t kStride = 8;

  EXPECT_EQ(XcdShard(0, kStride).owned_chunks(kMax), 536870912u);

  // The shares of a maximal grid still partition it exactly.
  uint64_t summed = 0;
  for (uint32_t rank = 0; rank < kStride; ++rank)
    summed += XcdShard(rank, kStride).owned_chunks(kMax);
  EXPECT_EQ(summed, static_cast<uint64_t>(kMax));

  // Counting correctly is only half of it: the mapper multiplies the shard-local
  // index by the stride, which is the other place a maximal grid can wrap. Pin
  // the largest ordinal each rank produces against the same arithmetic done in
  // 64 bits, without enumerating four billion chunks. Rank 0 ends at 4294967288.
  EXPECT_EQ(XcdShard(0, kStride).nth_owned_chunk(536870911u), 4294967288u);
  for (uint32_t rank = 0; rank < kStride; ++rank) {
    XcdShard shard{rank, kStride};
    const uint64_t owned = shard.owned_chunks(kMax);
    ASSERT_GT(owned, 0u) << "rank " << rank;
    const uint64_t expected = uint64_t{rank} + (owned - 1) * uint64_t{kStride};
    EXPECT_LT(expected, uint64_t{kMax}) << "rank " << rank;
    EXPECT_EQ(shard.nth_owned_chunk(static_cast<uint32_t>(owned - 1)), expected) << "rank " << rank;
  }
}

// Stride is a count of participants, so non-power-of-two splits have to work;
// nothing here may assume a maskable stride.
//
// Counts alone are too weak to catch a mapping regression, and every exact
// permutation check above uses stride 8. Stepping ordinals by 2 instead of by
// the stride would leave the summed count at 13 for stride 3 while duplicating
// 2, 4, 6 and 8 and omitting 9 through 12 -- so enumerate the ordinals here and
// require each one in [0, total) exactly once.
//
// Exact coverage alone is still not the round-robin contract: contiguous rank
// windows also cover [0, total) exactly once. Each ordinal must land on the rank
// congruent to it, or a stride could quietly switch from round-robin to slices.
TEST(XcdShardTest, NonPowerOfTwoStridePartitionsExactly) {
  for (uint32_t stride : {1u, 3u, 5u, 6u, 7u, 12u}) {
    for (uint32_t total : {0u, 1u, 13u, 100u, 1024u}) {
      std::multiset<uint32_t> seen;
      uint64_t summed = 0;
      for (uint32_t rank = 0; rank < stride; ++rank) {
        XcdShard shard{rank, stride};
        summed += shard.owned_chunks(total);
        for (uint32_t ordinal : owned_ordinals(shard, total)) {
          ASSERT_EQ(ordinal % stride, rank)
              << "ordinal " << ordinal << " stride " << stride << " rank " << rank;
          seen.insert(ordinal);
        }
      }

      EXPECT_EQ(summed, total) << "stride " << stride << " total " << total;
      ASSERT_EQ(seen.size(), total) << "stride " << stride << " total " << total;
      for (uint32_t ordinal = 0; ordinal < total; ++ordinal)
        EXPECT_EQ(seen.count(ordinal), 1u)
            << "ordinal " << ordinal << " stride " << stride << " total " << total;
    }
  }
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
    entry.apply_shard(XcdShard(rank, kStride));
    EXPECT_EQ(entry.total_wgs, kGridWgs / kStride) << "rank " << rank;

    for (uint32_t i = 0; i < entry.total_wgs; ++i) {
      uint32_t wg = entry.chunk_ordinal_for(i);
      ASSERT_EQ(wg % kStride, rank) << "rank " << rank << " wg " << wg;
      seen.insert(wg);
    }
  }
  // Size alone would still pass if some ordinals were duplicated and others
  // missing, since the insert count is fixed at kStride * share. Require every
  // ordinal of the grid exactly once.
  ASSERT_EQ(seen.size(), kGridWgs);
  for (uint32_t wg = 0; wg < kGridWgs; ++wg)
    EXPECT_EQ(seen.count(wg), 1u) << "workgroup " << wg << " not covered exactly once";
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
  // grid_wgs_y and grid_wgs_z default to 1, so their cluster counts must be 1
  // as well. Leaving them zero makes cluster_grid_is_complete() false and trips
  // the assertion in apply_shard() under any assertion-enabled build.
  entry.cluster_count_y = 1;
  entry.cluster_count_z = 1;
  entry.total_wgs = kGridWgs;
  ASSERT_TRUE(entry.cluster_grid_is_complete());
  ASSERT_EQ(entry.dispatch_chunk_wgs(), kClusterSize);

  entry.apply_shard(XcdShard(3, kStride));
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

// The reason the packet kind is stored rather than inferred. A grid smaller
// than the participating set leaves high ranks with nothing to run, and an
// empty share is exactly what `total_wgs == 0` used to mean for a barrier. The
// kind has to survive that, or the command processor retires the kernel early
// and fires a completion signal the grid has not earned.
TEST(XcdShardTest, EmptyKernelShareIsStillAKernel) {
  DispatchEntry entry;
  entry.kind = rocjitsu::amdgpu::DispatchPacketKind::Kernel;
  entry.grid_wgs_x = 3;
  entry.total_wgs = 3;

  entry.apply_shard(XcdShard(7, 8));
  ASSERT_EQ(entry.total_wgs, 0u) << "rank 7 of a 3-chunk grid owns nothing";
  EXPECT_FALSE(entry.is_non_kernel());
  EXPECT_TRUE(entry.fully_dispatched());
  EXPECT_TRUE(entry.fully_completed());

  DispatchEntry barrier;
  barrier.kind = rocjitsu::amdgpu::DispatchPacketKind::NonKernel;
  EXPECT_TRUE(barrier.is_non_kernel());
}
