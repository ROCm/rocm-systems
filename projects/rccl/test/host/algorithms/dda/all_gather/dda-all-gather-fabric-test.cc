/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_gather/dda_all_gather_fabric.cu: eligibility, the scratch check,
// the grid, and which kernel is launched with what.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_GATHER_FABRIC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::FabricGpuBarrier;
using nccl_dda_detail::DdaFabricBarrierState;
using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllGatherFabric<int8_t, N>);
}

class DdaAllGatherFabricMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    comm().ddaFabricBarrierState = barrier_;
    comm().rank = 3;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t sendcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllGatherDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, sendcount, datatype);
  }

  ncclResult_t Run(size_t sendcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllGatherDdaFabric(kSendbuff, kRecvbuff, sendcount, datatype, mock_.get(), kStream);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
};

TEST_F(DdaAllGatherFabricMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllGatherDdaFabricEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaFabricBarrierState = missing == 1 ? nullptr : barrier_;
    comm().ddaScratch = missing == 2 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 3 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllGatherFabricMicrotest, Eligible_PerRankBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(3));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

// Eligibility compares one rank's bytes with scratch, but the launcher needs room for all nRanks; a fix flips this pin.
TEST_F(DdaAllGatherFabricMicrotest, Eligible_KnownBug_ScratchCheckUsesPerRankBytesNotGatheredTotal) {
  comm().ddaScratchBytes = 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_EQ(Run(16), ncclInvalidArgument);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_EachMissingFabricResourceReturnsInvalidUsage) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaFabricBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_EQ(Run(4), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_GatheredTotalMayFillScratchExactly) {
  comm().nRanks = 4;
  comm().ddaScratchBytes = 4 * 64;
  EXPECT_EQ(Run(16), ncclSuccess);
  EXPECT_EQ(Run(20), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {5, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    g_ddaLaunches.clear();
    comm().nRanks = nRanks;
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_PassesBuffersByteCountRankAndBarrier) {
  ASSERT_EQ(Run(8, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel<8>());
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 7u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<size_t>(2), 16u);
  EXPECT_EQ(launch.Arg<const int8_t*>(3), kSendbuff);
  EXPECT_EQ(launch.Arg<int>(4), 3);
  EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
}

// Grids spelled out from getGridAndBlockDims (16 B per thread, 512 threads, 64-thread warps).
TEST_F(DdaAllGatherFabricMicrotest, Launch_GridMatchesBlocksAndCapsAtFabricMaxBlocks) {
  struct Case {
    size_t count;
    int maxBlocks;
    unsigned blocks, threads;
  };
  const Case cases[] = {{4, 256, 1, 64}, {512, 256, 1, 128}, {2048, 256, 1, 512}, {1u << 18, 256, 128, 512},
                        {1u << 18, 4, 4, 512}, {2048 + 4, 256, 2, 320}};
  for (const Case& c : cases) {
    g_ddaLaunches.clear();
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    ASSERT_EQ(Run(c.count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(g_ddaLaunches[0].grid.y, 1u);
    EXPECT_EQ(g_ddaLaunches[0].block.x, c.threads) << "count " << c.count;
    EXPECT_EQ(ncclAllGatherDdaFabricBlocks(mock_.get(), c.count, ncclFloat32), c.blocks);
  }
}

TEST_F(DdaAllGatherFabricMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllGatherDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat16), 16u);
  EXPECT_EQ(ncclAllGatherDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat32), 32u);
}

TEST_F(DdaAllGatherFabricMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
