/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "nccl_device/gin/anvil_sdma/gin_fabric_lsa_policy.h"

using gin::fabric::ginAnvilUseFabricMemPredicate;
using gin::fabric::ginFabricLsaA2ACapablePredicate;
using gin::fabric::ginFabricLsaA2AEnvMode;
using gin::fabric::ginFabricLsaA2ALaunchConfig;
using gin::fabric::ginFabricLsaA2ALsaBarrierPool;
using gin::fabric::ginFabricLsaAlltoAllSizeOk;
using gin::fabric::ginFabricA2ASelectAlgo;
using gin::fabric::GinFabricA2AAlgo;
using gin::fabric::ginFabricLsaA2AWantEnabled;
using gin::fabric::kGinFabricLsaA2ADefaultMaxPerPeer;
using gin::fabric::parseGinFabricThresholdEnv;
using gin::fabric::resolveGinFabricLsaThresholdAlltoAll;

TEST(GinFabricLsaPolicy, FabricMemPredicate) {
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(false, 4, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 2, 4, true));
  EXPECT_FALSE(ginAnvilUseFabricMemPredicate(true, 4, 4, false));
  EXPECT_TRUE(ginAnvilUseFabricMemPredicate(true, 4, 4, true));
}

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
  EXPECT_EQ(threads, 512);
}

TEST(GinFabricLsaPolicy, LsaBarrierPoolCapsAt64) {
  EXPECT_EQ(ginFabricLsaA2ALsaBarrierPool(4), 32);
  EXPECT_EQ(ginFabricLsaA2ALsaBarrierPool(16), 64);
}

TEST(GinFabricLsaPolicy, OptInEnv) {
  unsetenv("RCCL_GIN_FABRIC_LSA_A2A");
  unsetenv("NCCL_GIN_FABRIC_LSA_A2A");
  EXPECT_EQ(ginFabricLsaA2AEnvMode(), -1);
  EXPECT_FALSE(ginFabricLsaA2AWantEnabled(true));
  EXPECT_FALSE(ginFabricLsaA2AWantEnabled(false));
  setenv("RCCL_GIN_FABRIC_LSA_A2A", "1", 1);
  EXPECT_EQ(ginFabricLsaA2AEnvMode(), 1);
  EXPECT_TRUE(ginFabricLsaA2AWantEnabled(true));
  EXPECT_FALSE(ginFabricLsaA2AWantEnabled(false));
  setenv("RCCL_GIN_FABRIC_LSA_A2A", "0", 1);
  EXPECT_EQ(ginFabricLsaA2AEnvMode(), 0);
  EXPECT_FALSE(ginFabricLsaA2AWantEnabled(true));
  unsetenv("RCCL_GIN_FABRIC_LSA_A2A");
  setenv("NCCL_GIN_FABRIC_LSA_A2A", "1", 1);
  EXPECT_EQ(ginFabricLsaA2AEnvMode(), 1);
  unsetenv("NCCL_GIN_FABRIC_LSA_A2A");
}

TEST(GinFabricLsaPolicy, ParseThresholdRejectsEmptyNegativeAndJunk) {
  const char* name = "RCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL";
  unsigned long long v = 99;
  unsetenv(name);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, "-1", 1);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, " -1", 1);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, "512abc", 1);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, "", 1);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, "999999999999999999999999999999", 1);
  EXPECT_FALSE(parseGinFabricThresholdEnv(name, &v));
  setenv(name, "524288", 1);
  EXPECT_TRUE(parseGinFabricThresholdEnv(name, &v));
  EXPECT_EQ(v, 524288ull);
  unsetenv(name);
}

TEST(GinFabricLsaPolicy, DefaultLsaThresholdIsZeroUnlessOptedIn) {
  unsetenv("RCCL_GIN_FABRIC_LSA_A2A");
  unsetenv("NCCL_GIN_FABRIC_LSA_A2A");
  unsetenv("RCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL");
  unsetenv("NCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL");
  EXPECT_EQ(resolveGinFabricLsaThresholdAlltoAll(4), 0u);
  setenv("RCCL_GIN_FABRIC_LSA_A2A", "1", 1);
  EXPECT_EQ(resolveGinFabricLsaThresholdAlltoAll(4), 4 * kGinFabricLsaA2ADefaultMaxPerPeer);
  EXPECT_EQ(resolveGinFabricLsaThresholdAlltoAll(1), 0u);
  setenv("NCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL", "65536", 1);
  EXPECT_EQ(resolveGinFabricLsaThresholdAlltoAll(4), 65536u);
  unsetenv("NCCL_GIN_FABRIC_LSA_THRESHOLD_ALLTOALL");
  unsetenv("RCCL_GIN_FABRIC_LSA_A2A");
}

TEST(GinFabricA2ASelect, LsaWhenOptedInAndUnderThresholdElsePut) {
  const size_t lsaThr = 4 * kGinFabricLsaA2ADefaultMaxPerPeer;
  // 32 KiB/peer * 4 = 128 KiB total: LSA
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // 4 MiB/peer * 4 = 16 MiB total: still LSA, below the 8 MiB/peer default
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 4 * 1024 * 1024, lsaThr), GinFabricA2AAlgo::Lsa);
  // 16 MiB/peer * 4 = 64 MiB total: gin.put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 4, 16 * 1024 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // LSA not armed: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 0, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Context failed its layout check: put
  EXPECT_EQ(ginFabricA2ASelectAlgo(false, 1, 4, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
  // Rank count above the 64-CTA LSA pool falls through to put even when LSA is armed.
  EXPECT_EQ(ginFabricA2ASelectAlgo(true, 1, 65, 32 * 1024, lsaThr), GinFabricA2AAlgo::Put);
}
