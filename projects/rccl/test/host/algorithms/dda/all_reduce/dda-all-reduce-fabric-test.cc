/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_fabric.cu: eligibility, the flat/tree
// choice at 256 KiB, the tree's staging copy, the datatype dispatch, the grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_REDUCE_FABRIC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::FabricGpuBarrier;
using nccl_dda_detail::DdaFabricBarrierState;
using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// Spelled out rather than read from kDdaFlatTreeThresholdBytes, so a moved threshold fails here.
constexpr size_t kTreeThresholdBytes = 256 * 1024;
constexpr size_t kFp32AtThreshold = kTreeThresholdBytes / 4;
constexpr size_t kFp16AtThreshold = kTreeThresholdBytes / 2;

template <typename T, int N>
const void* Flat() {
  return KernelPtr(&dda::common::ddaAllReduceFlatFabric<T, N, false>);
}

template <typename T, int N>
const void* Tree() {
  return KernelPtr(&dda::common::ddaAllReduceTreeFabric<T, N, false>);
}

class DdaAllReduceFabricMicrotest : public ::testing::Test {
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

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclAllReduceDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype, op);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllReduceDdaFabric(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t count, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(count, datatype), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t count) {
    EXPECT_EQ(launch.kernel, (Flat<T, 8>()));
    ASSERT_EQ(launch.args.size(), 8u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<size_t>(2), count);
    EXPECT_EQ(launch.Arg<const T*>(3), kSendbuff);
    EXPECT_EQ(launch.Arg<int>(4), 3);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
    EXPECT_EQ(launch.Arg<std::nullptr_t>(7), nullptr);
  }

  DdaFabricMockComm mock_;
  DdaFabricBarrierState* const barrier_ = SharedBarrierState<DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaAllReduceFabricMicrotest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllReduceDdaFabricEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_EachMissingFabricResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaFabricBarrierState = missing == 1 ? nullptr : barrier_;
    comm().ddaScratch = missing == 2 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 3 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllReduceFabricMicrotest, Eligible_PayloadUpToScratchAndMultipleOf16) {
  comm().ddaScratchBytes = 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_FALSE(Eligible(3));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
}

// Below and at 256 KiB the flat kernel runs on any count; above it each rank's slice must be 16-byte aligned.
TEST_F(DdaAllReduceFabricMicrotest, Eligible_AboveThresholdPerRankSliceMustBe16ByteAligned) {
  EXPECT_TRUE(Eligible(12));
  EXPECT_TRUE(Eligible(kFp32AtThreshold));
  EXPECT_FALSE(Eligible(kFp32AtThreshold + 4));
  EXPECT_FALSE(Eligible(kFp32AtThreshold + 8));
  EXPECT_TRUE(Eligible(kFp32AtThreshold + 32));
  EXPECT_TRUE(Eligible(kFp16AtThreshold + 64, ncclBfloat16));
  comm().nRanks = 3;
  EXPECT_TRUE(Eligible(kFp32AtThreshold));
  EXPECT_FALSE(Eligible(kFp16AtThreshold + 32, ncclBfloat16));
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_EachMissingFabricResourceReturnsInvalidUsage) {
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

TEST_F(DdaAllReduceFabricMicrotest, Launch_ScratchCheckUsesTheDispatchedElementSize) {
  comm().ddaScratchBytes = 64;
  EXPECT_EQ(Run(32, ncclFloat16), ncclSuccess);
  EXPECT_EQ(Run(17, ncclFloat32), ncclInvalidArgument);
  EXPECT_EQ(Run(33, ncclBfloat16), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_EachDatatypeLaunchesItsOwnKernel) {
  ExpectLaunchOf<float>(RunOne(16, ncclFloat32), 16);
  ExpectLaunchOf<half>(RunOne(16, ncclFloat16), 16);
  ExpectLaunchOf<bf16>(RunOne(16, ncclBfloat16), 16);
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_TakesTheStopEventOnTheStream) {
  const DdaLaunch& launch = RunOne(4);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_FlatAtThresholdTreeAboveWhenRanksDivideCount) {
  EXPECT_EQ(RunOne(kFp32AtThreshold).kernel, (Flat<float, 8>()));
  EXPECT_TRUE(copies_.empty());
  EXPECT_EQ(RunOne(kFp32AtThreshold + 8).kernel, (Tree<float, 8>()));
  EXPECT_EQ(RunOne(kFp32AtThreshold + 4).kernel, (Flat<float, 8>()));
  EXPECT_EQ(RunOne(kFp16AtThreshold, ncclFloat16).kernel, (Flat<half, 8>()));
  EXPECT_EQ(RunOne(kFp16AtThreshold + 8, ncclBfloat16).kernel, (Tree<bf16, 8>()));
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_TreeStagesSendBufferIntoScratchBeforeTheKernel) {
  const size_t count = kFp32AtThreshold + 8;
  const DdaLaunch& launch = RunOne(count);
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, count * 4);
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
  ASSERT_EQ(launch.args.size(), 8u);
  EXPECT_EQ(launch.Arg<float**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<float*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<size_t>(2), count);
  EXPECT_EQ(launch.Arg<const float*>(3), kSendbuff);
  EXPECT_EQ(launch.Arg<int>(4), 3);
  EXPECT_EQ(launch.Arg<int>(5), 8);
  EXPECT_TRUE(SameBarrier(launch.Arg<FabricGpuBarrier>(6), barrier_->barrierHost));
  EXPECT_EQ(launch.Arg<std::nullptr_t>(7), nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_TreeStagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(kFp32AtThreshold + 8), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  struct Case {
    int nRanks;
    const void* flat;
    const void* tree;
  };
  const Case cases[] = {{4, Flat<float, 4>(), Tree<float, 4>()},
                        {8, Flat<float, 8>(), Tree<float, 8>()},
                        {2, Flat<float, 0>(), Tree<float, 0>()},
                        {6, Flat<float, 0>(), Tree<float, 0>()},
                        {144, Flat<float, 0>(), Tree<float, 0>()}};
  // 144 * 456 floats is just over 256 KiB and divides by every rank count here.
  for (const Case& c : cases) {
    comm().nRanks = c.nRanks;
    EXPECT_EQ(RunOne(4).kernel, c.flat) << "nRanks " << c.nRanks;
    const DdaLaunch& tree = RunOne(144 * 456);
    EXPECT_EQ(tree.kernel, c.tree) << "nRanks " << c.nRanks;
    EXPECT_EQ(tree.Arg<int>(5), c.nRanks);
  }
}

// Sized on the whole message in elements, capped by the VMM block budget rather than ddaFabricMaxBlocks.
TEST_F(DdaAllReduceFabricMicrotest, Launch_GridMatchesBlocksAndCapsAtVmmBlocks) {
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
    budget = c.budget;
    const DdaLaunch& launch = RunOne(c.count, c.datatype);
    EXPECT_EQ(launch.grid.x, c.blocks) << "count " << c.count;
    EXPECT_EQ(launch.block.x, c.threads) << "count " << c.count;
    EXPECT_EQ(ncclAllReduceDdaFabricBlocks(mock_.get(), c.count, c.datatype), c.blocks);
  }
}

TEST_F(DdaAllReduceFabricMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
