/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"

using gin::fabric::parseGinFabricLLThresholdEnv;
using gin::fabric::pickGinFabricLLThresholdAlltoAll;
using gin::fabric::resolveGinFabricLLThresholdAlltoAll;
using gin::fabric::ginFabricLlAlltoAllEligible;
using gin::fabric::ginFabricLlA2AScratchBytes;
using gin::fabric::kGinFabricLlMaxBytes;

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
  lane.llThreshold = 32;
  EXPECT_FALSE(ginFabricLlAlltoAllEligible(lane, 4, 16, 4, true));
  (void)kGinFabricLlMaxBytes;
}
