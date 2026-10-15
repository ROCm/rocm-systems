/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_fabric.cu: eligibility, the scratch check, the
// staging copy ahead of the launch, the grid, and the kernel launched.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALLTOALL_FABRIC_CC_PATH

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
  return KernelPtr(&dda::common::ddaAllToAllFabric<int8_t, N>);
}

class DdaAlltoallFabricMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    RecordDdaCopies(&copies_);
    comm().ddaFabricBarrierState = barrier_;
    comm().rank = 3;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabric(kSendbuff, kRecvbuff, count, datatype, mock_.get(), kStream);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaAlltoallFabricMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllToAllDdaFabricEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaFabricBarrierState = missing == 1 ? nullptr : barrier_;
    comm().ddaScratch = missing == 2 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 3 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_PerPeerBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaAlltoallFabricMicrotest, Eligible_WholeSendBufferMustFitScratch) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_TRUE(Eligible(32, ncclFloat16));
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_EachMissingFabricResourceReturnsInvalidUsage) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaFabricBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_EQ(Run(4), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_WholeSendBufferMayFillScratchExactly) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_EQ(Run(16), ncclSuccess);
  EXPECT_EQ(Run(20), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(copies_.size(), 1u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_StagesWholeSendBufferIntoScratchBeforeTheKernel) {
  ASSERT_EQ(Run(8, ncclFloat16), ncclSuccess);
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, 8u * 16);
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_StagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_EQ(comm().addonStopEvent, kStopEvent);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {5, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    g_ddaLaunches.clear();
    comm().nRanks = nRanks;
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(4), nRanks);
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_PassesBuffersPerPeerBytesRankAndBarrier) {
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
  ASSERT_EQ(launch.args.size(), 6u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<size_t>(2), 16u);
  EXPECT_EQ(launch.Arg<int>(3), 3);
  EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(5), barrier_->barrierHost));
}

// Grids spelled out from getGridAndBlockDims over one peer's bytes (16 B per thread, 512 threads, 64-thread warps).
TEST_F(DdaAlltoallFabricMicrotest, Launch_GridFollowsPerPeerBytesAndCapsAtFabricMaxBlocks) {
  struct Case {
    size_t count;
    int maxBlocks;
    unsigned blocks, threads;
  };
  const Case cases[] = {{4, 256, 1, 64}, {512, 256, 1, 128}, {2048, 256, 1, 512}, {2052, 256, 2, 320},
                        {1u << 18, 256, 128, 512}, {1u << 18, 4, 4, 512}};
  for (const Case& c : cases) {
    g_ddaLaunches.clear();
    comm().ddaFabricMaxBlocks = c.maxBlocks;
    ASSERT_EQ(Run(c.count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(g_ddaLaunches[0].block.x, c.threads) << "count " << c.count;
    EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), c.count, ncclFloat32), c.blocks);
  }
}

TEST_F(DdaAlltoallFabricMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat16), 16u);
  EXPECT_EQ(ncclAllToAllDdaFabricBlocks(mock_.get(), 1u << 16, ncclFloat32), 32u);
}

TEST_F(DdaAlltoallFabricMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
