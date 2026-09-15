/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"

using gin::fabric::GinFabricA2ACommState;
using gin::fabric::ginAnvilUseFabricMemPredicate;
using gin::fabric::ginFabricA2ALaneTryBuild;
using gin::fabric::ginFabricLlA2ACarveFits;
using gin::fabric::ginFabricLlA2ACarveOffset;
using gin::fabric::ginFabricLlAlltoAllBlocksPerPeer;
using gin::fabric::ginFabricLlAlltoAllEligible;
using gin::fabric::ginFabricLlAlltoAllSizeOk;
using gin::fabric::ginFabricLlA2AGinRegionBytes;
using gin::fabric::ginFabricLlA2AScratchBytes;
using gin::fabric::ginFabricLlLaneResourcesOk;
using gin::fabric::kGinFabricLlAgMaxBlocksPerPeer;
using gin::fabric::kGinFabricLlAlltoAllThresholdDefault;
using gin::fabric::kGinFabricLlA2APktsPerBlock;
using gin::fabric::kGinFabricLlMaxBytes;
using gin::fabric::kGinFabricLlMaxNranks;
using gin::fabric::parseGinFabricLLThresholdEnv;
using gin::fabric::pickGinFabricLLThresholdAlltoAll;
using gin::fabric::resolveGinFabricLLThresholdAlltoAll;

TEST(GinFabricLLPolicy, AlltoAllSetOverridesDefault) {
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(true, 0), 0u);
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(true, 524288), 524288u);
}

TEST(GinFabricLLPolicy, UnsetAlltoAllUses256KiBDefault) {
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(false, 0), kGinFabricLlAlltoAllThresholdDefault);
  EXPECT_EQ(pickGinFabricLLThresholdAlltoAll(false, 999), kGinFabricLlAlltoAllThresholdDefault);
}

TEST(GinFabricLLPolicy, ResolveAlltoAllEnvThen256KiBDefault) {
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  unsetenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(), kGinFabricLlAlltoAllThresholdDefault);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "524288", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(), 524288u);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "0", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(), 0u);

  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  setenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "65536", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(), 65536u);

  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "524288", 1);
  setenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "65536", 1);
  EXPECT_EQ(resolveGinFabricLLThresholdAlltoAll(), 524288u);

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
}

// The LL protocol expands 8 payload bytes into a 16-byte packet, so a chunk is
// only eligible while perChunkBytes * 2 stays within kGinFabricLlMaxBytes. Both
// terms are exercised here: a chunk just under the cap passes and the first one
// over it is rejected, with the threshold and scratch raised out of the way so
// this cap is the only gate that can decide either case.
TEST(GinFabricLLPolicy, EligibilityRejectsChunksOverTheLLPacketCap) {
  const size_t maxChunk = kGinFabricLlMaxBytes / 2;
  ncclGinFabricA2ALane lane{};
  lane.enabled = 1;
  lane.peerScratch = reinterpret_cast<void**>(0x1);
  lane.llEpoch = reinterpret_cast<uint32_t*>(0x2);
  lane.scratchBytes = ginFabricLlA2AScratchBytes(4);
  lane.llThreshold = 4 * maxChunk * 2;

  EXPECT_TRUE(ginFabricLlAlltoAllEligible(lane, 4, maxChunk, 1, true));
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, maxChunk + 16, 1, true));
  // count == 0 is rejected before any size arithmetic runs.
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 0, 1, true));
}

// Device-side gate, same caps, reached without a lane struct.
TEST(GinFabricLLPolicy, SizeGateRejectsZeroCountAndOverCapChunks) {
  const size_t scratch4 = ginFabricLlA2AScratchBytes(4);
  const size_t maxChunk = kGinFabricLlMaxBytes / 2;
  const size_t threshold = 4 * maxChunk * 2;

  EXPECT_TRUE(ginFabricLlAlltoAllSizeOk(4, maxChunk, threshold, scratch4));
  EXPECT_FALSE(ginFabricLlAlltoAllSizeOk(4, maxChunk + 16, threshold, scratch4));
  EXPECT_FALSE(ginFabricLlAlltoAllSizeOk(4, 0, threshold, scratch4));
  // Total over the threshold falls back to gin.put even when the chunk fits.
  EXPECT_FALSE(ginFabricLlAlltoAllSizeOk(4, 4096, 4 * 4096 - 16, scratch4));
}

TEST(GinFabricLLPolicy, LaneResourcesRejectZeroThreshold) {
  EXPECT_FALSE(ginFabricLlLaneResourcesOk(4, ginFabricLlA2AScratchBytes(4), 0));
}

TEST(GinFabricLLPolicy, LaneResourcesRejectUndersizedScratchAndRankBounds) {
  const size_t gin4 = ginFabricLlA2AGinRegionBytes(4, 64 * 1024);
  ASSERT_GT(gin4, 0u);
  EXPECT_TRUE(ginFabricLlLaneResourcesOk(4, gin4, 64 * 1024));
  EXPECT_FALSE(ginFabricLlLaneResourcesOk(4, gin4 - 1, 64 * 1024));
  EXPECT_FALSE(ginFabricLlLaneResourcesOk(1, gin4, 64 * 1024));
  EXPECT_FALSE(ginFabricLlLaneResourcesOk(kGinFabricLlMaxNranks + 1, gin4, 64 * 1024));
}

TEST(GinFabricLLPolicy, BlocksPerPeerCoversFastDivideAndClamp) {
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(8), 1);
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(kGinFabricLlA2APktsPerBlock * 8), 1);
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer((kGinFabricLlA2APktsPerBlock + 1) * 8), 2);
  const size_t hugePk = (static_cast<size_t>(kGinFabricLlAgMaxBlocksPerPeer) + 2) * kGinFabricLlA2APktsPerBlock * 8;
  EXPECT_EQ(ginFabricLlAlltoAllBlocksPerPeer(hugePk), kGinFabricLlAgMaxBlocksPerPeer);
}

TEST(GinFabricLLPolicy, CarveOffsetPlacesGinRegionAtScratchTail) {
  const size_t ddaHead = ginFabricLlA2AScratchBytes(4);
  const size_t ginTail = ginFabricLlA2AGinRegionBytes(4, 64 * 1024);
  ASSERT_GT(ginTail, 0u);
  ASSERT_LT(ginTail, ddaHead);
  const size_t alloc = ddaHead + ginTail;
  EXPECT_EQ(ginFabricLlA2ACarveOffset(alloc, ginTail), ddaHead);
  EXPECT_FALSE(ginFabricLlA2ACarveFits(4, ddaHead, ddaHead, 64 * 1024));
  EXPECT_TRUE(ginFabricLlA2ACarveFits(4, alloc, ddaHead, 64 * 1024));
}

TEST(GinFabricLLPolicy, LaneBuildRequiresResourcesAndDdaLL) {
  const size_t ddaHead = ginFabricLlA2AScratchBytes(4);
  const size_t ginTail = ginFabricLlA2AGinRegionBytes(4, 64 * 1024);
  GinFabricA2ACommState comm{};
  comm.fabricMemHandler = reinterpret_cast<void*>(0x1);
  comm.peerPtrsDev = reinterpret_cast<void**>(0x2);
  comm.llEpochDev = reinterpret_cast<uint32_t*>(0x3);
  comm.scratch = reinterpret_cast<void*>(0x4);
  comm.scratchBytes = ddaHead;
  comm.scratchAllocBytes = ddaHead + ginTail;
  comm.llEpochLen = 4;
  comm.nRanks = 4;
  ncclGinFabricA2ALane lane{};
  EXPECT_TRUE(ginFabricA2ALaneTryBuild(comm, true, 64 * 1024, &lane));
  EXPECT_EQ(lane.enabled, 1);
  EXPECT_EQ(lane.llThreshold, 64u * 1024u);
  EXPECT_EQ(lane.scratchBytes, ginTail);
  EXPECT_FALSE(ginFabricA2ALaneTryBuild(comm, false, 64 * 1024, &lane));
  EXPECT_FALSE(ginFabricA2ALaneTryBuild(comm, true, 0, &lane));
  comm.scratchAllocBytes = ddaHead;
  EXPECT_FALSE(ginFabricA2ALaneTryBuild(comm, true, 64 * 1024, &lane));
}

TEST(GinFabricLLPolicy, FabricMemPredicate) {
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(false, 4, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 2, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 4, 4, false));
  EXPECT_TRUE(ginAnvilUseFabricMemPredicate(true, 4, 4, true));
}
