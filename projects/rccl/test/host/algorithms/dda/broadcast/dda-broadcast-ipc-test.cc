/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/broadcast/dda_broadcast_ipc.cu: eligibility, the all-gather grid at one
// slice, and the launch with its alignment variants.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "../common/DdaIpcTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_BROADCAST_IPC_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using dda::common::IpcGpuBarrier;
using nccl_dda_detail::DdaIpcBarrierState;
using RcclUnitTesting::DdaIpcMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);

// Bytes counted in int8, so the 128 B unit (8 slices of 16 B) reads directly.
static_assert(kDdaBcIpcGranuleBytes == 128, "the sizes below are written in 128 B units");
class DdaBroadcastIpcMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    comm().ddaIpcBarrierState = barrier_;
    comm().rank = 6;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool BcEligible(size_t bytes) {
    return ncclBroadcastDdaIpcEligible(mock_.get(), bytes, ncclInt8);
  }

  ncclResult_t BcRun(size_t bytes, int root, const void* send = kSendbuff, void* recv = kRecvbuff) {
    return ncclBroadcastDdaIpc(send, recv, bytes, ncclInt8, root, mock_.get(), kStream);
  }

  DdaIpcMockComm mock_;
  DdaIpcBarrierState* const barrier_ = SharedBarrierState<DdaIpcBarrierState>(0xa5);
};

template <bool kSendAligned, bool kRecvAligned>
const void* BcKernel() {
  return KernelPtr(&dda::common::ddaBroadcastIpc2Phase<int8_t, 8, kSendAligned, kRecvAligned>);
}

TEST_F(DdaBroadcastIpcMicrotest, Eligible_WholeUnitsOfEightSlicesAnyDatatype) {
  EXPECT_TRUE(BcEligible(128));
  EXPECT_FALSE(BcEligible(0));
  EXPECT_FALSE(BcEligible(112));
  EXPECT_FALSE(BcEligible(128 + 16));
  EXPECT_TRUE(ncclBroadcastDdaIpcEligible(mock_.get(), 32, ncclFloat32));
  EXPECT_TRUE(ncclBroadcastDdaIpcEligible(mock_.get(), 16, ncclFloat64));
  EXPECT_FALSE(ncclBroadcastDdaIpcEligible(nullptr, 128, ncclInt8));
}

TEST_F(DdaBroadcastIpcMicrotest, Eligible_WholeMessageMustFitScratch) {
  comm().ddaScratchBytes = 1024;
  EXPECT_TRUE(BcEligible(1024));
  EXPECT_FALSE(BcEligible(1024 + 128));
}

TEST_F(DdaBroadcastIpcMicrotest, Eligible_SingleNodeEightRanksWithEveryIpcResource) {
  comm().nNodes = 2;
  EXPECT_FALSE(BcEligible(128));
  comm().nNodes = 1;
  comm().nRanks = 4;
  EXPECT_FALSE(BcEligible(128));
  comm().nRanks = 8;
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setIpcResourcesPresent(true);
    comm().ddaIpcMemHandler = missing == 0 ? nullptr : comm().ddaIpcMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaIpcBarrierState = missing == 3 ? nullptr : barrier_;
    EXPECT_FALSE(BcEligible(128)) << "missing resource " << missing;
    EXPECT_EQ(BcRun(128, 0), ncclInvalidUsage) << "missing resource " << missing;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

// The all-gather grid at one slice per rank, at most 24 blocks.
TEST_F(DdaBroadcastIpcMicrotest, Blocks_AllGatherGridAtOneSlice) {
  const std::pair<size_t, unsigned> cases[] = {{128, 1}, {256 << 10, 4}, {1 << 20, 16}, {8 << 20, 24}};
  for (const auto& [bytes, blocks] : cases) {
    g_ddaLaunches.clear();
    EXPECT_EQ(ncclBroadcastDdaIpcBlocks(nullptr, bytes, ncclInt8), blocks) << bytes;
    ASSERT_EQ(BcRun(bytes, 0), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, blocks) << bytes;
  }
}

TEST_F(DdaBroadcastIpcMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclBroadcastDdaIpcBlocks(nullptr, 64 << 10, ncclInt8), 1u);
  EXPECT_EQ(ncclBroadcastDdaIpcBlocks(nullptr, 64 << 10, ncclFloat32), 4u);
}

TEST_F(DdaBroadcastIpcMicrotest, Launch_SliceIsBytesNotElements) {
  ASSERT_EQ(ncclBroadcastDdaIpc(kSendbuff, kRecvbuff, 256, ncclFloat32, 3, mock_.get(), kStream), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  EXPECT_EQ(g_ddaLaunches[0].Arg<size_t>(2), 128u);
}

TEST_F(DdaBroadcastIpcMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(BcRun(128, 0), ncclUnhandledCudaError);
}

TEST_F(DdaBroadcastIpcMicrotest, Launch_RejectsBadRootAndIneligibleSize) {
  EXPECT_EQ(BcRun(128, -1), ncclInvalidUsage);
  EXPECT_EQ(BcRun(128, 8), ncclInvalidUsage);
  EXPECT_EQ(BcRun(144, 0), ncclInvalidUsage);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaBroadcastIpcMicrotest, Launch_PassesScratchBuffersSliceRankRootAndBarrier) {
  ASSERT_EQ(BcRun(1024, 3), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  ASSERT_EQ(launch.args.size(), 7u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<size_t>(2), 128u);
  EXPECT_EQ(launch.Arg<const int8_t*>(3), kSendbuff);
  EXPECT_EQ(launch.Arg<int>(4), 6);
  EXPECT_EQ(launch.Arg<int>(5), 3);
  EXPECT_TRUE(SameBarrier(launch.Arg<IpcGpuBarrier>(6), barrier_->barrierHost));
}

// sendbuff alignment counts on the root only; recvbuff alignment on every rank.
TEST_F(DdaBroadcastIpcMicrotest, Launch_VariantFollowsThisRanksBufferAlignment) {
  char* const odd = static_cast<char*>(kRecvbuff) + 4;
  const struct {
    int root;
    const void* send;
    void* recv;
    const void* kernel;
  } cases[] = {{6, kSendbuff, kRecvbuff, BcKernel<true, true>()},
               {6, odd, kRecvbuff, BcKernel<false, true>()},
               {6, kSendbuff, odd, BcKernel<true, false>()},
               {6, odd, odd, BcKernel<false, false>()},
               {0, odd, kRecvbuff, BcKernel<true, true>()},
               {0, kSendbuff, odd, BcKernel<true, false>()}};
  for (const auto& c : cases) {
    g_ddaLaunches.clear();
    ASSERT_EQ(BcRun(128, c.root, c.send, c.recv), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, c.kernel) << "root " << c.root << " send " << c.send << " recv " << c.recv;
  }
}

}  // namespace
