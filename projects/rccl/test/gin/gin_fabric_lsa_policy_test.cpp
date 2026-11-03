/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <string>

#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"
#include "gin/gin_anvil_reach.h"

using gin::fabric::ginFabricLsaA2ACapablePredicate;
using gin::fabric::ginFabricLsaA2ALaunchConfig;
using gin::fabric::ginFabricLsaA2ALsaBarrierPool;
using gin::fabric::ginFabricLsaAlltoAllSizeOk;
using gin::fabric::ginFabricA2ASelectAlgo;
using gin::fabric::GinFabricA2AAlgo;
using gin::fabric::ginFabricLsaA2AThreshold;
using gin::fabric::kGinFabricLsaA2ADefaultThreshold;

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
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 2, 20, kGinFabricLsaA2ADefaultThreshold), GinFabricA2AAlgo::Put);
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

TEST(GinFabricLsaPolicy, LsaBarrierPoolCapsAt64) {
  EXPECT_EQ(ginFabricLsaA2ALsaBarrierPool(4), 32);
  EXPECT_EQ(ginFabricLsaA2ALsaBarrierPool(16), 64);
}

TEST(GinFabricLsaPolicy, ThresholdIsZeroUnlessOptedIn) {
  EXPECT_EQ(ginFabricLsaA2AThreshold(false, -1, 4), 0u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(false, 65536, 4), 0u);
  EXPECT_EQ(kGinFabricLsaA2ADefaultThreshold, 2u * 1024 * 1024);
  // A total size, the same at every rank count.
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 4), kGinFabricLsaA2ADefaultThreshold);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 16), kGinFabricLsaA2ADefaultThreshold);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, -1, 1), 0u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, 65536, 4), 65536u);
  EXPECT_EQ(ginFabricLsaA2AThreshold(true, 0, 4), 0u);
}

TEST(GinFabricA2ASelect, LsaWhenOptedInAndUnderThresholdElsePut) {
  const size_t lsaThr = kGinFabricLsaA2ADefaultThreshold;
  // 32 KiB/peer * 4 = 128 KiB total: LSA
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // 512 KiB/peer * 4 = 2 MiB total: still LSA, at the default threshold
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 512 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // 1 MiB/peer * 4 = 4 MiB total: gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 1024 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // 128 KiB/peer * 16 = 2 MiB total: LSA; 256 KiB/peer * 16 = 4 MiB: gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 16, 128 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 16, 256 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // LSA not armed: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 0, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Context failed its layout check: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(false, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Rank count above the 64-CTA LSA pool falls through to put even when LSA is armed.
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 65, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
}

TEST(GinAnvilSdmaReach, SingleHostOnEveryArch) {
  for (GinAnvilArch arch : {GinAnvilArch::Other, GinAnvilArch::Mi300, GinAnvilArch::Mi350, GinAnvilArch::Mi450}) {
    EXPECT_EQ(ginAnvilSdmaReachPredicate(arch, 1, 8, false), GinAnvilSdmaReach::Host);
    EXPECT_EQ(ginAnvilSdmaReachPredicate(arch, 1, 64, true), GinAnvilSdmaReach::Host);
  }
}

TEST(GinAnvilSdmaReach, Mi300AndMi350StopAtTheHost) {
  for (GinAnvilArch arch : {GinAnvilArch::Mi300, GinAnvilArch::Mi350, GinAnvilArch::Other}) {
    EXPECT_EQ(ginAnvilSdmaReachPredicate(arch, 2, 16, false), GinAnvilSdmaReach::None);
    EXPECT_EQ(ginAnvilSdmaReachPredicate(arch, 2, 16, true), GinAnvilSdmaReach::None);
    EXPECT_STREQ(ginAnvilSdmaNoReachReason(arch, 16, false), "SDMA reaches only GPUs on the same host");
  }
}

TEST(GinAnvilSdmaReach, Mi450CrossesHostsOnlyInFabricPod) {
  EXPECT_EQ(ginAnvilSdmaReachPredicate(GinAnvilArch::Mi450, 2, 8, true), GinAnvilSdmaReach::FabricPod);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(GinAnvilArch::Mi450, 4, NCCL_GIN_ANVIL_IPC_MAX_RANKS, true),
            GinAnvilSdmaReach::FabricPod);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(GinAnvilArch::Mi450, 2, 8, false), GinAnvilSdmaReach::None);
  EXPECT_EQ(ginAnvilSdmaReachPredicate(GinAnvilArch::Mi450, 4, NCCL_GIN_ANVIL_IPC_MAX_RANKS + 1, true),
            GinAnvilSdmaReach::None);
  EXPECT_STRNE(ginAnvilSdmaNoReachReason(GinAnvilArch::Mi450, 8, false),
               ginAnvilSdmaNoReachReason(GinAnvilArch::Mi450, NCCL_GIN_ANVIL_IPC_MAX_RANKS + 1, false));
}

// A pod the topology allows but a rank did not opt in to names the variables to set.
TEST(GinAnvilSdmaReach, Mi450PodWithoutOptInNamesTheOptIns) {
  const char* reason = ginAnvilSdmaNoReachReason(GinAnvilArch::Mi450, 8, true);
  EXPECT_NE(std::string(reason).find("RCCL_GIN_ANVIL_FABRIC_POD=1"), std::string::npos);
  EXPECT_NE(std::string(reason).find("RCCL_GIN_ANVIL_FABRIC_MEM=1"), std::string::npos);
}
