/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu: eligibility against the LL
// threshold and the per-rank slot of all_gather_dda_fabric_ll.h, the block-per-peer grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_GATHER_FABRIC_LL_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// 8 ranks, bank of 8 * 32 packets: a slot of 32 packets carries 256 B per rank.
constexpr size_t kSlotScratch = 2 * 8 * 32 * 16;

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllGatherFabricLL<int8_t, N>);
}

class DdaAllGatherFabricLLMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    comm().rank = 5;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t sendcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllGatherDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, sendcount, datatype);
  }

  ncclResult_t Run(size_t sendcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllGatherDdaFabricLL(kSendbuff, kRecvbuff, sendcount, datatype, mock_.get(), kStream);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_DdaLLZeroRejects) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
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

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_PerRankBytesMustBeMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_GatheredTotalUpToAllGatherLLThreshold) {
  ncclFunc_t seenFunc = ncclFuncSendRecv;
  g_rcclDdaLLThreshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    seenFunc = func;
    return size_t{8 * 64};
  };
  EXPECT_TRUE(Eligible(16));
  EXPECT_EQ(seenFunc, ncclFuncAllGather);
  EXPECT_FALSE(Eligible(20));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Eligible_PerRankPacketsMustFitTheSlot) {
  comm().ddaScratchBytes = kSlotScratch;
  EXPECT_TRUE(Eligible(64));
  EXPECT_FALSE(Eligible(68));
  // 31 packets per rank in the bank floor to a 16-packet slot.
  comm().ddaScratchBytes = 2 * 8 * 31 * 16;
  EXPECT_TRUE(Eligible(32));
  EXPECT_FALSE(Eligible(36));
  // A bank under one 16-packet granule per rank leaves no slot at all.
  comm().ddaScratchBytes = 2 * 8 * 15 * 16;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLLMicrotest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllGatherFabricLLMicrotest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {7, Kernel<0>()}, {144, Kernel<0>()}};
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

TEST_F(DdaAllGatherFabricLLMicrotest, Launch_PassesBuffersBytesEpochAndBank) {
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
  ASSERT_EQ(launch.args.size(), 9u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const int8_t*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 16u);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 2064u);
}

// One block per peer up to 256 packets (2 KiB) per rank, then one per 256 packets, at most 8 per peer.
TEST_F(DdaAllGatherFabricLLMicrotest, Launch_BlocksPerPeerFollowPacketCountCappedAtEight) {
  comm().nRanks = 4;
  const std::pair<size_t, unsigned> cases[] = {{4, 1}, {512, 1}, {516, 2}, {1024, 2}, {4096, 8}, {1u << 20, 8}};
  for (const auto& [count, bpp] : cases) {
    g_ddaLaunches.clear();
    ASSERT_EQ(Run(count), ncclSuccess);
    ASSERT_EQ(g_ddaLaunches.size(), 1u);
    EXPECT_EQ(g_ddaLaunches[0].grid.x, 4u);
    EXPECT_EQ(g_ddaLaunches[0].grid.y, bpp) << "count " << count;
    EXPECT_EQ(ncclAllGatherDdaFabricLLBlocks(mock_.get(), count, ncclFloat32), 4 * bpp);
  }
}

TEST_F(DdaAllGatherFabricLLMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllGatherDdaFabricLLBlocks(mock_.get(), 1024, ncclFloat16), 8u);
  EXPECT_EQ(ncclAllGatherDdaFabricLLBlocks(mock_.get(), 1024, ncclFloat32), 16u);
}

TEST_F(DdaAllGatherFabricLLMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
