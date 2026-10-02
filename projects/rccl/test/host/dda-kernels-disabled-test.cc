/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only checks for src/algorithms/dda/dda_kernels_disabled.cc, the stubs
// linked when DDA kernels are compiled out. Compiled into rccl-UnitTestsMicroInit,
// which does not link the real kernels or fakes/dda_fakes.cc. WARN goes through
// the ncclDebugLog already provided by fakes/nccl_fakes.cc.

#include <gtest/gtest.h>

#include <cstdint>

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "../common/LogCapture.hpp"

TEST(DdaKernelsDisabled, EligibilityIsFalseAndBlocksAreZero) {
  using RedElig = bool (*)(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t);
  using Elig = bool (*)(ncclComm*, const void*, void*, size_t, ncclDataType_t);
  using Blocks = uint32_t (*)(ncclComm*, size_t, ncclDataType_t);

  const RedElig redElig[] = {
      ncclAllReduceDdaIpcEligible,         ncclAllReduceDdaFabricEligible,     ncclAllReduceDdaFabricLLEligible,
      ddaLLArOneShotEligible,              ddaLLArTwoShotEligible,             ddaLL128ArOneShotEligible,
      ddaLL128ArTwoShotEligible,           ncclAllReduceDdaFabricLL128Eligible, ncclReduceScatterDdaIpcEligible,
      ncclReduceScatterDdaFabricEligible,  ncclReduceScatterDdaFabricLLEligible,
      ncclReduceScatterDdaFabricLL128Eligible,
  };
  for (RedElig fn : redElig) {
    EXPECT_FALSE(fn(nullptr, nullptr, nullptr, 1, ncclFloat, ncclSum));
  }

  const Elig elig[] = {
      ncclAllGatherDdaIpcEligible,        ncclAllGatherDdaFabricEligible,        ncclAllGatherDdaFabricLLEligible,
      ncclAllGatherDdaFabricLL128Eligible, ncclAllToAllDdaIpcEligible,           ncclAllToAllDdaFabricEligible,
      ncclAllToAllDdaFabricLLEligible,    ncclAllToAllDdaFabricLL128Eligible,
  };
  for (Elig fn : elig) {
    EXPECT_FALSE(fn(nullptr, nullptr, nullptr, 1, ncclFloat));
  }

  const Blocks blocks[] = {
      ncclAllReduceDdaIpcBlocks,       ncclAllReduceDdaFabricBlocks,       ncclAllReduceDdaFabricLLBlocks,
      ncclAllReduceDdaFabricLL128Blocks, ncclAllGatherDdaIpcBlocks,        ncclAllGatherDdaFabricBlocks,
      ncclAllGatherDdaFabricLLBlocks,  ncclAllGatherDdaFabricLL128Blocks,  ncclAllToAllDdaIpcBlocks,
      ncclAllToAllDdaFabricBlocks,     ncclAllToAllDdaFabricLLBlocks,      ncclAllToAllDdaFabricLL128Blocks,
  };
  for (Blocks fn : blocks) {
    EXPECT_EQ(0u, fn(nullptr, 1, ncclFloat));
  }
}

TEST(DdaKernelsDisabled, LaunchReturnsInternalError) {
  using RedLaunch =
      ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t);
  using Launch = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t);

  const RedLaunch redLaunch[] = {
      ncclAllReduceDdaIpc,        ncclAllReduceDdaFabric,        ncclAllReduceDdaFabricLL,
      ncclAllReduceDdaFabricLL128, ncclReduceScatterDdaIpc,      ncclReduceScatterDdaFabric,
      ncclReduceScatterDdaFabricLL, ncclReduceScatterDdaFabricLL128,
  };
  bool sawDisabledWarning = false;
  for (RedLaunch fn : redLaunch) {
    ncclResult_t res = ncclSuccess;
    const std::string log =
        RcclUnitTesting::CaptureLog([&]() { res = fn(nullptr, nullptr, 1, ncclFloat, ncclSum, nullptr, nullptr); });
    EXPECT_EQ(ncclInternalError, res);
    sawDisabledWarning = sawDisabledWarning || RcclUnitTesting::LogHas(log, "disabled at compile time");
  }

  const Launch launch[] = {
      ncclAllGatherDdaIpc,     ncclAllGatherDdaFabric,     ncclAllGatherDdaFabricLL,     ncclAllGatherDdaFabricLL128,
      ncclAllToAllDdaIpc,      ncclAllToAllDdaFabric,      ncclAllToAllDdaFabricLL,      ncclAllToAllDdaFabricLL128,
  };
  for (Launch fn : launch) {
    ncclResult_t res = ncclSuccess;
    const std::string log =
        RcclUnitTesting::CaptureLog([&]() { res = fn(nullptr, nullptr, 1, ncclFloat, nullptr, nullptr); });
    EXPECT_EQ(ncclInternalError, res);
    sawDisabledWarning = sawDisabledWarning || RcclUnitTesting::LogHas(log, "disabled at compile time");
  }
  EXPECT_TRUE(sawDisabledWarning);
}
