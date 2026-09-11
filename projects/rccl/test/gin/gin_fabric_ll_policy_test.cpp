/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "algorithms/dda/fabric/fabric_init.h"
#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"

using gin::fabric::GinFabricA2ACommState;
using gin::fabric::ginFabricA2ALaneTryBuild;
using gin::fabric::ginFabricLlAlltoAllBlocksPerPeer;
using gin::fabric::ginFabricLlAlltoAllEligible;
using gin::fabric::ginFabricLlA2AScratchBytes;
using gin::fabric::ginFabricLlLaneResourcesOk;
using gin::fabric::kGinFabricLlAgMaxBlocksPerPeer;
using gin::fabric::kGinFabricLlA2APktsPerBlock;
using gin::fabric::kGinFabricLlMaxBytes;
using gin::fabric::parseGinFabricLLThresholdEnv;
using gin::fabric::pickGinFabricLLThresholdAlltoAll;
using gin::fabric::resolveGinFabricLLThresholdAlltoAll;

TEST(GinFabricLLPolicy, AlltoAllSetOverridesDdaFallback) {
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(true, 0, 32768u), 0u);
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(true, 524288, 32768u), 524288u);
}

TEST(GinFabricLLPolicy, UnsetAlltoAllUsesDdaFallback) {
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(false, 0, 32768u), 32768u);
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(false, 999, 524288u), 524288u);
}

TEST(GinFabricLLPolicy, ResolveAlltoAllEnvThenDdaFallback) {
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  unsetenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(32768u), 32768u);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "524288", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(32768u), 524288u);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "0", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(32768u), 0u);

  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  setenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "65536", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(32768u), 65536u);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "524288", 1);
  setenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "65536", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(32768u), 524288u);

  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  unsetenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
}

TEST(GinFabricLLPolicy, ParseRejectsEmptyAndNegative) {
  unsigned long long v = 99;
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "-1", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", " -1", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "512abc", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "999999999999999999999999999999", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "524288", 1);
  EXPECT_TRUE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  EXPECT_EQ(v, 524288ull);
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
}

TEST(GinFabricLLPolicy, EligibilityMatchesLaneAndSizeGate) {
  ncclGinFabricA2ALane lane{};
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 16, 4, true));
  lane.enabled = 1;
  lane.peerScratch = reinterpret_cast<void**>(0x1);
  lane.llEpoch = reinterpret_cast<uint32_t*>(0x2);
  lane.llThreshold = 64 * 1024;
  lane.scratchBytes = ginFabricLlA2AScratchBytes(4);
  EXPECT_TRUE(ginFabricLlAlltoAllEligible(lane, 4, 16, 4, true));
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 16, 4, false));
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 3, 4, true));
  lane.llThreshold = 0;
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 16, 4, true));
  (void)kGinFabricLlMaxBytes;
}

TEST(GinFabricLLPolicy, LaneResourcesRejectZeroThreshold) {
  EXPECT_FALSE(ginFabricLlLaneResourcesOk(4, ginFabricLlA2AScratchBytes(4), 0));
}

TEST(GinFabricLLPolicy, BlocksPerPeerCoversFastDivideAndClamp) {
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(8), 1);
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(kGinFabricLlA2APktsPerBlock * 8), 1);
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer((kGinFabricLlA2APktsPerBlock + 1) * 8), 2);
  const size_t hugePk = (static_cast<size_t>(kGinFabricLlAgMaxBlocksPerPeer) + 2) * kGinFabricLlA2APktsPerBlock * 8;
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(hugePk), kGinFabricLlAgMaxBlocksPerPeer);
}

TEST(GinFabricLLPolicy, LaneBuildRequiresResourcesAndDdaLL) {
  GinFabricA2ACommState comm{};
  comm.fabricMemHandler = reinterpret_cast<void*>(0x1);
  comm.peerPtrsDev = reinterpret_cast<void**>(0x2);
  comm.llEpochDev = reinterpret_cast<uint32_t*>(0x3);
  comm.scratch = reinterpret_cast<void*>(0x4);
  comm.scratchBytes = ginFabricLlA2AScratchBytes(4);
  comm.llEpochLen = 4;
  comm.nRanks = 4;
  ncclGinFabricA2ALane lane{};
  EXPECT_TRUE(ginFabricA2ALaneTryBuild(comm, true, 64 * 1024, &lane));
  EXPECT_EQ(lane.enabled, 1);
  EXPECT_EQ(lane.llThreshold, 64u * 1024u);
  EXPECT_FALSE(ginFabricA2ALaneTryBuild(comm, false, 64 * 1024, &lane));
  EXPECT_FALSE(ginFabricA2ALaneTryBuild(comm, true, 0, &lane));
}

TEST(GinFabricLLPolicy, FabricMemPredicate) {
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(false, 4, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 2, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 4, 4, false));
  EXPECT_TRUE(ginAnvilUseFabricMemPredicate(true, 4, 4, true));
}
