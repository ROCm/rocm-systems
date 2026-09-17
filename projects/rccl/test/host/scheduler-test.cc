/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; UUTs are #include'd. Line citations use src/scheduler/ numbers; hipify adds 1.

#include <gtest/gtest.h>

#include <memory>

#include "../common/LogCapture.hpp"
#include "fakes/scheduler_fakes.h"

// Local to this TU only: struct ncclComm is never shared across a TU boundary here, so no ABI mismatch.
#define ENABLE_WARP_SPEED

// Hipify renames the allgatherv one to *_tmp.cc: src/enqueue/task_sched/allgatherv_sched.cc has the basename.
#include ALLGATHERV_SCHED_CC_PATH
#include SYMMETRIC_SCHED_CC_PATH

namespace {
constexpr int kBaselineChannels = 4;
}  // namespace

TEST(SchedulerMicrotest, AgvChannelCount_MultiplierAtMost1_ReturnsTunedChannelsUnchanged) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 1;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_COLL);
  int result = -1;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { result = agvChannelCount(comm.get(), kBaselineChannels); });
  EXPECT_EQ(result, kBaselineChannels);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "AllGatherV: WarpSpeed not supported"));
}

TEST(SchedulerMicrotest, AgvChannelCount_MultiplierAbove1_DividesTunedChannels) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 2;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_COLL);
  int result = -1;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { result = agvChannelCount(comm.get(), kBaselineChannels); });
  EXPECT_EQ(result, kBaselineChannels / 2);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "AllGatherV: WarpSpeed not supported"));
}

TEST(SchedulerMicrotest, AgvChannelCount_MultiplierAbove1_FloorsResultAtOne) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 8;
  EXPECT_EQ(agvChannelCount(comm.get(), /*tunedChannels=*/1), 1);
}

TEST(SchedulerMicrotest, SymkRedOp_Avg_ReturnsDevSumPostDivRegardlessOfInputDevOp) {
  EXPECT_EQ(symkRedOp(ncclAvg, ncclDevSum), ncclDevSumPostDiv);
  EXPECT_EQ(symkRedOp(ncclAvg, ncclDevMinMax), ncclDevSumPostDiv);
}

TEST(SchedulerMicrotest, SymkRedOp_NonAvg_ReturnsDevRedOpUnchanged) {
  EXPECT_EQ(symkRedOp(ncclSum, ncclDevSum), ncclDevSum);
  EXPECT_EQ(symkRedOp(ncclMax, ncclDevMinMax), ncclDevMinMax);
  EXPECT_EQ(symkRedOp(ncclProd, ncclDevProd), ncclDevProd);
}
