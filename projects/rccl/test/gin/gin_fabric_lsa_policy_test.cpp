/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"

using gin::fabric::ginFabricLsaA2ACapablePredicate;
using gin::fabric::ginFabricLsaA2ALaunchConfig;
using gin::fabric::ginFabricLsaA2AFlagBlocks;
using gin::fabric::ginFabricLsaAlltoAllSizeOk;
using gin::fabric::ginFabricA2ASelectAlgo;
using gin::fabric::GinFabricA2AAlgo;
using gin::fabric::ginFabricLsaA2AThreshold;
using gin::fabric::ginFabricLsaA2ADefaultThreshold;

TEST(GinFabricLsaPolicy, CapablePredicateRequiresFabricLsaTeam) {
  EXPECT_FALSE(ginFabricLsaA2ACapablePredicate(false, true, true, 4, 4, 4));
  EXPECT_FALSE(ginFabricLsaA2ACapablePredicate(true, false, true, 4, 4, 4));
  EXPECT_FALSE(ginFabricLsaA2ACapablePredicate(true, true, false, 4, 4, 4));
  EXPECT_FALSE(ginFabricLsaA2ACapablePredicate(true, true, true, 2, 4, 4));
  EXPECT_FALSE(ginFabricLsaA2ACapablePredicate(true, true, true, 4, 4, 2));
  EXPECT_TRUE(ginFabricLsaA2ACapablePredicate(true, true, true, 4, 4, 4));
}

TEST(GinFabricLsaPolicy, SizeGateUsesTotalBytes) {
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 0, 256 * 1024));
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 16, 0));
  EXPECT_TRUE(ginFabricLsaAlltoAllSizeOk(4, 16, 64 * 1024));
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 64 * 1024, 64 * 1024));
  EXPECT_TRUE(ginFabricLsaAlltoAllSizeOk(64, 16, 64 * 1024));
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(65, 16, 64 * 1024 * 1024));
}

// The lane copies 16-byte vectors; any other per-peer size stays on gin.put.
TEST(GinFabricLsaPolicy, SizeGateNeedsSixteenByteMultiples) {
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 4, 64 * 1024));
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 20, 64 * 1024));
  EXPECT_FALSE(ginFabricLsaAlltoAllSizeOk(4, 16 * 1024 + 8, 64 * 1024));
  EXPECT_TRUE(ginFabricLsaAlltoAllSizeOk(4, 32, 64 * 1024));
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 2, 20, ginFabricLsaA2ADefaultThreshold(2)), GinFabricA2AAlgo::Put);
}

TEST(GinFabricLsaPolicy, LaunchConfigScalesChunksWithSize) {
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  ginFabricLsaA2ALaunchConfig(8, 8 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 2);
  EXPECT_EQ(threads, 256);
  EXPECT_EQ(chunkBytes, 4u * 1024u);

  ginFabricLsaA2ALaunchConfig(8, 32 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 4);
  EXPECT_EQ(threads, 256);

  ginFabricLsaA2ALaunchConfig(8, 512 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 8);
  EXPECT_EQ(threads, 256);
}

// Chunks per peer are capped so nRanks * chunks stays within the 64-CTA pool.
TEST(GinFabricLsaPolicy, LaunchConfigCapsChunksAtTheCtaPool) {
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  ginFabricLsaA2ALaunchConfig(16, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 4);
  EXPECT_EQ(chunkBytes, 32u * 1024u);
  ginFabricLsaA2ALaunchConfig(64, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 1);
  ginFabricLsaA2ALaunchConfig(65, 128 * 1024, &chunks, &threads, &chunkBytes);
  EXPECT_EQ(chunks, 1);
}

TEST(GinFabricLsaPolicy, FlagBlocksCapAt64) {
  EXPECT_EQ(ginFabricLsaA2AFlagBlocks(4), 32);
  EXPECT_EQ(ginFabricLsaA2AFlagBlocks(16), 64);
}

TEST(GinFabricLsaPolicy, ThresholdIsZeroUnlessOptedIn) {
  EXPECT_EQ(ginFabricLsaA2AThreshold(false, -1, 4), 0u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(false, 65536, 4), 0u);
  // A total size of nRanks^2 MiB.
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 2), 4u * 1024 * 1024);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 4), 16u * 1024 * 1024);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 8), 64u * 1024 * 1024);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 16), 256u * 1024 * 1024);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 64), (size_t)4 * 1024 * 1024 * 1024);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 1), 0u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, 65536, 4), 65536u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, 0, 4), 0u);
}

TEST(GinFabricA2ASelect, LsaWhenOptedInAndUnderThresholdElsePut) {
  const size_t lsaThr = ginFabricLsaA2ADefaultThreshold(4);
  // 32 KiB/peer * 4 = 128 KiB total: LSA
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // 4 MiB/peer * 4 = 16 MiB total: still LSA, at the default threshold
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 4 * 1024 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // One more vector per peer: gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 4 * 1024 * 1024 + 16, lsaThr), GinFabricA2AAlgo::Put);
  // 2 ranks: 2 MiB/peer (4 MiB total) is LSA, 4 MiB/peer gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 2, 2 * 1024 * 1024, ginFabricLsaA2ADefaultThreshold(2)),
            GinFabricA2AAlgo::Lsa);
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 2, 4 * 1024 * 1024, ginFabricLsaA2ADefaultThreshold(2)),
            GinFabricA2AAlgo::Put);
  // 16 ranks: 16 MiB/peer (256 MiB total) is LSA, 32 MiB/peer gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 16, 16 * 1024 * 1024, ginFabricLsaA2ADefaultThreshold(16)),
            GinFabricA2AAlgo::Lsa);
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 16, 32 * 1024 * 1024, ginFabricLsaA2ADefaultThreshold(16)),
            GinFabricA2AAlgo::Put);
  // LSA not armed: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 0, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Context failed its layout check: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(false, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Rank count above the 64-CTA LSA pool falls through to put even when LSA is armed.
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 65, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
}
