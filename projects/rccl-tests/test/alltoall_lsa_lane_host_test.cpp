/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the LSA AllToAll lane policy in alltoall_lsa_lane_policy.h.
// No GPU or MPI required.

#include <gtest/gtest.h>

#include "alltoall_lsa_lane_policy.h"

using alltoall_lsa::Algo;
using alltoall_lsa::capable;
using alltoall_lsa::defaultThreshold;
using alltoall_lsa::flagBlocks;
using alltoall_lsa::launchConfig;
using alltoall_lsa::select;
using alltoall_lsa::sizeOk;
using alltoall_lsa::threshold;

namespace {

constexpr size_t MiB = (size_t)1024 * 1024;

TEST(AlltoAllLsaLanePolicy, CapableNeedsAnLsaTeamOverEveryRank) {
  EXPECT_TRUE(capable(4, 4));
  EXPECT_TRUE(capable(16, 16));
  EXPECT_TRUE(capable(64, 64));
  EXPECT_FALSE(capable(8, 4));    // two hosts without MNNVL: two LSA teams of 4
  EXPECT_FALSE(capable(1, 1));    // nothing to exchange
  EXPECT_FALSE(capable(65, 65));  // past the 64-CTA flag pool
}

TEST(AlltoAllLsaLanePolicy, SizeGateUsesTotalBytes) {
  EXPECT_FALSE(sizeOk(4, 0, 256 * 1024));
  EXPECT_FALSE(sizeOk(4, 16, 0));
  EXPECT_TRUE(sizeOk(4, 16, 64 * 1024));
  EXPECT_FALSE(sizeOk(4, 64 * 1024, 64 * 1024));
  EXPECT_TRUE(sizeOk(64, 16, 64 * 1024));
  EXPECT_FALSE(sizeOk(65, 16, 64 * MiB));
}

// The lane copies 16-byte vectors; any other per-peer size stays on gin.put.
TEST(AlltoAllLsaLanePolicy, SizeGateNeedsSixteenByteMultiples) {
  EXPECT_FALSE(sizeOk(4, 4, 64 * 1024));
  EXPECT_FALSE(sizeOk(4, 20, 64 * 1024));
  EXPECT_FALSE(sizeOk(4, 16 * 1024 + 8, 64 * 1024));
  EXPECT_TRUE(sizeOk(4, 32, 64 * 1024));
  EXPECT_EQ(select(true, 2, 20, defaultThreshold(2)), Algo::Put);
}

TEST(AlltoAllLsaLanePolicy, LaunchConfigScalesChunksWithSize) {
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  launchConfig(8, 8 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 2);
  EXPECT_EQ(threads, 256);
  EXPECT_EQ(chunkBytes, 4u * 1024u);

  launchConfig(8, 32 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 4);
  EXPECT_EQ(threads, 256);

  launchConfig(8, 512 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 8);
  EXPECT_EQ(threads, 256);
}

// Chunks per peer are capped so nRanks * chunks stays within the 64-CTA pool.
TEST(AlltoAllLsaLanePolicy, LaunchConfigCapsChunksAtTheCtaPool) {
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  launchConfig(16, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 4);
  EXPECT_EQ(chunkBytes, 32u * 1024u);
  launchConfig(64, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 1);
  launchConfig(65, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 1);
}

TEST(AlltoAllLsaLanePolicy, FlagBlocksCapAt64) {
  EXPECT_EQ(flagBlocks(4), 32);
  EXPECT_EQ(flagBlocks(16), 64);
}

TEST(AlltoAllLsaLanePolicy, ThresholdIsZeroUnlessOptedIn) {
  EXPECT_EQ(threshold(false, -1, 4), 0u);
  EXPECT_EQ(threshold(false, 65536, 4), 0u);
  // A total size of nRanks^2 MiB, except no cap at 16 ranks, the largest count measured.
  EXPECT_EQ(threshold(true, -1, 2), 4 * MiB);
  EXPECT_EQ(threshold(true, -1, 4), 16 * MiB);
  EXPECT_EQ(threshold(true, -1, 8), 64 * MiB);
  EXPECT_EQ(threshold(true, -1, 15), 225 * MiB);
  EXPECT_EQ(threshold(true, -1, 16), ~(size_t)0);
  EXPECT_EQ(threshold(true, -1, 17), 289 * MiB);
  EXPECT_EQ(threshold(true, -1, 32), 1024 * MiB);
  EXPECT_EQ(threshold(true, -1, 64), 4096 * MiB);
  EXPECT_EQ(threshold(true, (int64_t)256 * MiB, 16), 256 * MiB);
  EXPECT_EQ(threshold(true, -1, 1), 0u);
  EXPECT_EQ(threshold(true, 65536, 4), 65536u);
  EXPECT_EQ(threshold(true, 0, 4), 0u);
}

TEST(AlltoAllLsaLanePolicy, LsaWhenOnAndUnderThresholdElsePut) {
  const size_t thr4 = defaultThreshold(4);
  // 32 KiB/peer * 4 = 128 KiB total: LSA
  EXPECT_EQ(select(true, 4, 32 * 1024, thr4), Algo::Lsa);
  // 4 MiB/peer * 4 = 16 MiB total: still LSA, at the default threshold
  EXPECT_EQ(select(true, 4, 4 * MiB, thr4), Algo::Lsa);
  // One more vector per peer: gin.put
  EXPECT_EQ(select(true, 4, 4 * MiB + 16, thr4), Algo::Put);
  // 2 ranks: 2 MiB/peer (4 MiB total) is LSA, 4 MiB/peer gin.put
  EXPECT_EQ(select(true, 2, 2 * MiB, defaultThreshold(2)), Algo::Lsa);
  EXPECT_EQ(select(true, 2, 4 * MiB, defaultThreshold(2)), Algo::Put);
  // 8 ranks: 8 MiB/peer (64 MiB total) is LSA, one more vector per peer gin.put
  EXPECT_EQ(select(true, 8, 8 * MiB, defaultThreshold(8)), Algo::Lsa);
  EXPECT_EQ(select(true, 8, 8 * MiB + 16, defaultThreshold(8)), Algo::Put);
  // 16 ranks: no cap, so 32 MiB/peer (512 MiB total) and 256 MiB/peer (4 GiB) stay on LSA
  EXPECT_EQ(select(true, 16, 32 * MiB, defaultThreshold(16)), Algo::Lsa);
  EXPECT_EQ(select(true, 16, 256 * MiB, defaultThreshold(16)), Algo::Lsa);
  // Lane off (a rank did not opt in, is not capable, or disagrees on the threshold): put
  EXPECT_EQ(select(false, 4, 32 * 1024, thr4), Algo::Put);
  // Rank count above the 64-CTA pool falls through to put even when the lane is on.
  EXPECT_EQ(select(true, 65, 32 * 1024, thr4), Algo::Put);
}

}  // namespace
