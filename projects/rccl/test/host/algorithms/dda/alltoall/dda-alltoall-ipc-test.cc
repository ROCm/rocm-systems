/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_ipc.cu: eligibility, the scratch check, the
// in-kernel versus pre-kernel staging copy, the 24-block grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../common/DdaIpcTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALLTOALL_IPC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::IpcGpuBarrier;
using nccl_dda_detail::DdaIpcBarrierState;
using RcclUnitTesting::DdaIpcMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// One 512-thread block of 16 B loads covers 8 KiB per peer; below that the kernel stages the copy itself.
constexpr size_t kSingleBlockFloats = 8192 / 4;

template <bool kStagingCopyInKernel>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllToAllIpc<int8_t, 8, false, kStagingCopyInKernel>);
}

class DdaAlltoallIpcMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    RecordDdaCopies(&copies_);
    comm().ddaIpcBarrierState = barrier_;
    comm().rank = 4;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaIpcEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaIpc(kSendbuff, kRecvbuff, count, datatype, mock_.get(), kStream);
  }

  DdaIpcMockComm mock_;
  DdaIpcBarrierState* const barrier_ = SharedBarrierState<DdaIpcBarrierState>(0xa5);
  std::vector<DdaCopy> copies_;
};

TEST_F(DdaAlltoallIpcMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllToAllDdaIpcEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_EachMissingIpcResourceRejects) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setIpcResourcesPresent(true);
    comm().ddaIpcMemHandler = missing == 0 ? nullptr : comm().ddaIpcMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaIpcBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_SingleNodeOfExactlyEightRanks) {
  comm().nNodes = 2;
  EXPECT_FALSE(Eligible(4));
  comm().nNodes = 1;
  comm().nRanks = 7;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 9;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_PerPeerBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaAlltoallIpcMicrotest, Eligible_WholeSendBufferMustFitScratch) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_TRUE(Eligible(16));
  EXPECT_FALSE(Eligible(20));
  EXPECT_TRUE(Eligible(32, ncclFloat16));
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_EachMissingIpcResourceReturnsInvalidUsage) {
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setIpcResourcesPresent(true);
    comm().ddaIpcMemHandler = missing == 0 ? nullptr : comm().ddaIpcMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaIpcBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_EQ(Run(4), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_WholeSendBufferMayFillScratchExactly) {
  comm().ddaScratchBytes = 8 * 64;
  EXPECT_EQ(Run(16), ncclSuccess);
  EXPECT_EQ(Run(20), ncclInvalidArgument);
  EXPECT_EQ(g_ddaLaunches.size(), 1u);
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_SingleBlockPayloadStagesInsideTheKernel) {
  ASSERT_EQ(Run(kSingleBlockFloats - 4), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(g_ddaLaunches[0].kernel, Kernel<true>());
  EXPECT_EQ(g_ddaLaunches[0].grid.x, 1u);
  EXPECT_TRUE(copies_.empty());
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_LargerPayloadIsStagedIntoScratchBeforeTheKernel) {
  ASSERT_EQ(Run(kSingleBlockFloats), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(g_ddaLaunches[0].kernel, Kernel<false>());
  ASSERT_EQ(copies_.size(), 1u);
  EXPECT_EQ(copies_[0].dst, comm().ddaScratch);
  EXPECT_EQ(copies_[0].src, kSendbuff);
  EXPECT_EQ(copies_[0].bytes, 8 * kSingleBlockFloats * 4);
  EXPECT_EQ(copies_[0].kind, hipMemcpyDeviceToDevice);
  EXPECT_EQ(copies_[0].stream, kStream);
  EXPECT_EQ(copies_[0].launchesBefore, 0u);
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_StagingCopyErrorIsReportedWithoutLaunching) {
  g_hipMemcpyAsync = [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; };
  EXPECT_EQ(Run(kSingleBlockFloats), ncclUnhandledCudaError);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_PassesBuffersPerPeerBytesRankAndBarrier) {
  for (size_t count : {size_t{8}, kSingleBlockFloats * 2}) {
    g_ddaLaunches.clear();
    comm().addonStopEvent = kStopEvent;
    ASSERT_EQ(Run(count, ncclFloat16), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    const DdaLaunch& launch = g_ddaLaunches[0];
    EXPECT_EQ(launch.stream, kStream);
    EXPECT_EQ(launch.startEvent, nullptr);
    EXPECT_EQ(launch.stopEvent, kStopEvent) << "count " << count;
    EXPECT_EQ(launch.sharedMemBytes, 0u);
    EXPECT_EQ(launch.flags, 0u);
    ASSERT_EQ(launch.args.size(), 6u);
    EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<size_t>(2), count * 2);
    EXPECT_EQ(launch.Arg<const int8_t*>(3), kSendbuff);
    EXPECT_EQ(launch.Arg<int>(4), 4);
    EXPECT_TRUE(SameBarrier(launch.Arg<IpcGpuBarrier>(5), barrier_->barrierHost));
    EXPECT_EQ(comm().addonStopEvent, nullptr);
  }
}

// The 24-block cap is spelled out rather than read from DDA_IPC_MAXBLOCKS, so a changed cap fails here.
TEST_F(DdaAlltoallIpcMicrotest, Launch_GridFollowsPerPeerBytesAndCapsAt24) {
  const std::pair<size_t, unsigned> cases[] = {{4, 1}, {2048, 1}, {2052, 2}, {1u << 14, 8}, {1u << 16, 24}};
  for (const auto& [count, blocks] : cases) {
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, blocks) << "count " << count;
    EXPECT_EQ(ncclAllToAllDdaIpcBlocks(nullptr, count, ncclFloat32), blocks);
  }
}

TEST_F(DdaAlltoallIpcMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllToAllDdaIpcBlocks(nullptr, 1u << 13, ncclFloat16), 2u);
  EXPECT_EQ(ncclAllToAllDdaIpcBlocks(nullptr, 1u << 13, ncclFloat32), 4u);
}

TEST_F(DdaAlltoallIpcMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
