/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu: eligibility against the
// bank-sized LL slot, the block-per-peer grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALLTOALL_FABRIC_LL_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A 16 B packet carries 8 B, so the LL staging cap of 32 MiB per chunk leaves 16 MiB of payload.
constexpr size_t kMaxPerPeerBytes = 16u << 20;
// 8 ranks, bank of 8 * 32 packets: a slot of 32 packets carries 256 B per peer.
constexpr size_t kSlotScratch = 2 * 8 * 32 * 16;

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricLL<int8_t, N>);
}

class DdaAlltoallFabricLLMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    comm().rank = 1;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricLL(kSendbuff, kRecvbuff, count, datatype, mock_.get(), kStream);
  }

  DdaFabricMockComm mock_;
};

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllToAllDdaFabricLLEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
  comm().ddaFabricBarrierState = nullptr;
  EXPECT_TRUE(Eligible(4));
  for (int missing = 0; missing < 3; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
  EXPECT_TRUE(Eligible(16, ncclFloat8e4m3));
  EXPECT_TRUE(Eligible(16, ncclFloat8e5m2));
  g_rcclDdaDatatypeSupported = [](const ncclComm*, ncclDataType_t datatype) { return datatype == ncclFloat8e5m2; };
  EXPECT_FALSE(Eligible(8, ncclFloat16));
  EXPECT_FALSE(Eligible(16, ncclFloat8e4m3));
  EXPECT_TRUE(Eligible(16, ncclFloat8e5m2));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_PerPeerBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_PerPeerBytesUpToHalfTheStagingCap) {
  EXPECT_TRUE(Eligible(kMaxPerPeerBytes / 4));
  EXPECT_FALSE(Eligible(kMaxPerPeerBytes / 4 + 4));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Eligible_PerPeerPacketsMustFitTheSlot) {
  comm().ddaScratchBytes = kSlotScratch;
  EXPECT_TRUE(Eligible(64));
  EXPECT_FALSE(Eligible(68));
  // The same bank over 4 ranks doubles the slot to 64 packets, 512 B per peer.
  comm().nRanks = 4;
  EXPECT_TRUE(Eligible(128));
  EXPECT_FALSE(Eligible(132));
  comm().nRanks = 8;
  // 31 packets per peer in the bank floor to a 16-packet slot.
  comm().ddaScratchBytes = 2 * 8 * 31 * 16;
  EXPECT_TRUE(Eligible(32));
  EXPECT_FALSE(Eligible(36));
  // A bank under one 16-packet granule per peer leaves no slot at all.
  comm().ddaScratchBytes = 2 * 8 * 15 * 16;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLLMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAlltoallFabricLLMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {3, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    g_ddaLaunches.clear();
    comm().nRanks = nRanks;
    ASSERT_EQ(Run(4), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(g_ddaLaunches[0].grid.x, static_cast<unsigned>(nRanks));
    EXPECT_EQ(g_ddaLaunches[0].Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAlltoallFabricLLMicrotest, Launch_PassesBuffersPerPeerBytesEpochAndBank) {
  comm().ddaScratchBytes = 4096 + 40;
  ASSERT_EQ(Run(8, ncclFloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel<8>());
  EXPECT_EQ(launch.block.x, 256u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 10u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const int8_t*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 16u);
  EXPECT_EQ(launch.Arg<int>(4), 1);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  // Bank 1 starts at half the scratch floored to 128 B, the offset every DDA LL/LL128 tier uses.
  EXPECT_EQ(launch.Arg<size_t>(8), 2048u);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(9).flag, comm().abortFlagDev);
}

// One block per peer up to 256 packets (2 KiB) per peer, then one per 256 packets, at most 8 per peer.
TEST_F(DdaAlltoallFabricLLMicrotest, Launch_BlocksPerPeerFollowPacketCountCappedAtEight) {
  comm().nRanks = 4;
  const std::pair<size_t, unsigned> cases[] = {{4, 1}, {512, 1}, {516, 2}, {1024, 2}, {4096, 8}, {1u << 20, 8}};
  for (const auto& [count, bpp] : cases) {
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, 4u);
    EXPECT_EQ(g_ddaLaunches[0].grid.y, bpp) << "count " << count;
    EXPECT_EQ(ncclAllToAllDdaFabricLLBlocks(mock_.get(), count, ncclFloat32), 4 * bpp);
  }
}

// The whole grid must be co-resident, so blocks per peer are also capped at the comm block budget / nRanks.
TEST_F(DdaAlltoallFabricLLMicrotest, Launch_BlocksPerPeerCappedByTheCuBudget) {
  comm().nRanks = 4;
  const std::pair<int, unsigned> cases[] = {{256, 8}, {20, 5}, {4, 1}, {3, 1}};
  for (const auto& [budget, bpp] : cases) {
    comm().ddaFabricMaxBlocks = budget;
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(4096), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.y, bpp) << "budget " << budget;
  }
}

TEST_F(DdaAlltoallFabricLLMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllToAllDdaFabricLLBlocks(mock_.get(), 1024, ncclFloat16), 8u);
  EXPECT_EQ(ncclAllToAllDdaFabricLLBlocks(mock_.get(), 1024, ncclFloat32), 16u);
}

TEST_F(DdaAlltoallFabricLLMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
