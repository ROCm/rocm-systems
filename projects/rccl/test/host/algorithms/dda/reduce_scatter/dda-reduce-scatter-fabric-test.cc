/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu: eligibility, the scratch
// check and staging copy, the datatype dispatch, the grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_REDUCE_SCATTER_FABRIC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::FabricGpuBarrier;
using nccl_dda_detail::DdaFabricBarrierState;
using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

template <typename T, int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabric<T, N, false>);
}

class DdaReduceScatterFabricMicrotest : public ::testing::Test {
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

  bool Eligible(size_t recvcount, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclReduceScatterDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, recvcount, datatype, op);
  }

  ncclResult_t Run(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclReduceScatterDdaFabric(kSendbuff, kRecvbuff, recvcount, datatype, ncclSum, mock_.get(), kStream);
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount) {
    EXPECT_EQ(launch.kernel, (Kernel<T, 8>()));
    ASSERT_EQ(launch.args.size(), 8u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<size_t>(2), recvcount);
    EXPECT_EQ(launch.Arg<const T*>(3), kSendbuff);
    EXPECT_EQ(launch.Arg<int>(4), 3);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
    EXPECT_EQ(launch.Arg<std::nullptr_t>(7), nullptr);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = LeakedBarrierState<DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclReduceScatterDdaFabricEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaFabricBarrierState = missing == 1 ? nullptr : barrier_;
    comm().ddaScratch = missing == 2 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 3 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_PerRankBytesMustBeMultipleOf16) {
  comm().nRanks = 4;
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaReduceScatterFabricMicrotest, Eligible_WholeSendBufferMustFitScratch) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_TRUE(Eligible(32, ncclFloat16));
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_EachMissingFabricResourceReturnsInvalidUsage) {
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

TEST_F(DdaReduceScatterFabricMicrotest, Launch_ScratchCheckUsesTheDispatchedElementSize) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_EQ(Run(32, ncclFloat16), ncclSuccess);
  EXPECT_EQ(Run(17, ncclFloat32), ncclInvalidArgument);
  EXPECT_EQ(Run(33, ncclBfloat16), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_EachDatatypeLaunchesItsOwnKernel) {
  ASSERT_EQ(Run(16, ncclFloat32), ncclSuccess);
  ASSERT_EQ(Run(16, ncclFloat16), ncclSuccess);
  ASSERT_EQ(Run(16, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 3u);
  ExpectLaunchOf<float>(g_ddaLaunches[0], 16);
  ExpectLaunchOf<half>(g_ddaLaunches[1], 16);
  ExpectLaunchOf<bf16>(g_ddaLaunches[2], 16);
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_StagesWholeSendBufferIntoScratchBeforeTheKernel) {
  ASSERT_EQ(Run(8, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, 8u * 8 * 2);
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_StagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
  EXPECT_EQ(comm().addonStopEvent, kStopEvent);
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {{4, Kernel<float, 4>()},
                                               {8, Kernel<float, 8>()},
                                               {2, Kernel<float, 0>()},
                                               {5, Kernel<float, 0>()},
                                               {144, Kernel<float, 0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    g_ddaLaunches.clear();
    comm().nRanks = nRanks;
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(5), nRanks);
  }
}

// Sized on this rank's shard in elements, capped by the VMM block budget rather than ddaFabricMaxBlocks.
TEST_F(DdaReduceScatterFabricMicrotest, Launch_GridFollowsShardAndCapsAtVmmBlocks) {
  int budget = 256;
  g_rcclDdaFabricVmmBlocks = [&](const ncclComm* c) {
    EXPECT_EQ(c, mock_.get());
    return budget;
  };
  comm().ddaFabricMaxBlocks = 1;
  struct Case {
    size_t count;
    ncclDataType_t datatype;
    int budget;
    unsigned blocks, threads;
  };
  const Case cases[] = {{4, ncclFloat32, 256, 1, 64},      {2048, ncclFloat32, 256, 1, 512},
                        {2052, ncclFloat32, 256, 2, 320},  {4096, ncclFloat16, 256, 1, 512},
                        {1u << 18, ncclFloat32, 256, 128, 512}, {1u << 18, ncclFloat32, 5, 5, 512}};
  for (const Case& c : cases) {
    g_ddaLaunches.clear();
    budget = c.budget;
    ASSERT_EQ(Run(c.count, c.datatype), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(g_ddaLaunches[0].block.x, c.threads) << "count " << c.count;
  }
}

TEST_F(DdaReduceScatterFabricMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
