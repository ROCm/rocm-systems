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
  unsetenv("NCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
}

TEST(GinFabricLLPolicy, ParseRejectsEmptyAndNegative) {
  unsigned long long v = 99;
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  setenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", "-1", 1);
  EXPECT_FALSE(parseGinFabricLLThresholdEnv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL", &v));
  unsetenv("RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL");
}
