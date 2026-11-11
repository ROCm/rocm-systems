/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/broadcast/dda_broadcast_fabric_ll.cu: the block budget, the
// threads override, the root/non-root grids, rank-uniform eligibility, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_BROADCAST_FABRIC_LL_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
constexpr int kRank = 5;
constexpr int kOtherRoot = 0;
// 16 KiB is 2048 packets: eight 256-thread chunks.
constexpr size_t kEightChunks = 16384;

template <bool kAligned>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaBroadcastFabricLL<kAligned>);
}

void* Addr(uintptr_t a) { return reinterpret_cast<void*>(a); }

class DdaBroadcastFabricLLMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    comm().rank = kRank;
    comm().WarpSize = 64;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  // int8 so count is bytes; broadcast is datatype-agnostic.
  bool Eligible(size_t count, ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLLEligible(mock_.get(), count, datatype);
  }

  uint32_t Blocks(size_t count, ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLLBlocks(mock_.get(), count, datatype);
  }

  ncclResult_t Run(int root, size_t count, const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff,
                   ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLL(sendbuff, recvbuff, count, datatype, root, mock_.get(), kStream);
  }

  DdaLaunch RunOne(int root, size_t count, const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff) {
    g_ddaLaunches.clear();
    comm().addonStopEvent = kStopEvent;
    EXPECT_EQ(Run(root, count, sendbuff, recvbuff), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.empty() ? DdaLaunch{} : g_ddaLaunches.back();
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaBroadcastFabricLLMicrotest, BlockBudget_FabricMaxBlocksElseCuCount) {
  comm().cuCount = 80;
  comm().ddaFabricMaxBlocks = 200;
  EXPECT_EQ(ncclBroadcastDdaBlockBudget(mock_.get()), 200u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(ncclBroadcastDdaBlockBudget(mock_.get()), 80u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, BlockBudget_EnvLowersButNeverRaises) {
  comm().ddaFabricMaxBlocks = 200;
  const std::pair<int64_t, size_t> cases[] = {{0, 200},   {1, 1},     {32, 32}, {199, 199},
                                              {200, 200}, {300, 200}, {-5, 200}};
  for (const auto& [env, budget] : cases) {
    params_["RCCL_DDA_BC_MAX_BLOCKS"] = env;
    EXPECT_EQ(ncclBroadcastDdaBlockBudget(mock_.get()), budget) << "DDA_BC_MAX_BLOCKS " << env;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, BlockBudget_NeverBelowOne) {
  comm().ddaFabricMaxBlocks = 0;
  comm().cuCount = 0;
  EXPECT_EQ(ncclBroadcastDdaBlockBudget(mock_.get()), 1u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_ThreadsOverrideMustBeAWarpMultipleUpTo512) {
  const std::pair<int64_t, unsigned> wave64[] = {{256, 256}, {64, 64},   {128, 128}, {512, 512}, {32, 256},
                                                 {96, 256},  {576, 256}, {1024, 256}, {0, 256},  {-64, 256}};
  for (const auto& [env, threads] : wave64) {
    params_["RCCL_DDA_LL_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, threads) << "wave64 DDA_LL_BC_THREADS " << env;
  }
  comm().WarpSize = 32;
  const std::pair<int64_t, unsigned> wave32[] = {{32, 32}, {96, 96}, {16, 256}, {48, 256}};
  for (const auto& [env, threads] : wave32) {
    params_["RCCL_DDA_LL_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, threads) << "wave32 DDA_LL_BC_THREADS " << env;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_NonRootChunksArePacketsPerThreadClampedOneToEight) {
  const std::pair<size_t, unsigned> cases[] = {{16, 1},   {2048, 1},     {2064, 2},     {4096, 2},
                                               {4112, 3}, {kEightChunks, 8}, {65536, 8}};
  for (const auto& [bytes, chunks] : cases) {
    const DdaLaunch launch = RunOne(kOtherRoot, bytes);
    EXPECT_EQ(launch.grid.x, chunks) << "bytes " << bytes;
    EXPECT_EQ(launch.grid.y, 1u) << "bytes " << bytes;
  }
  params_["RCCL_DDA_LL_BC_THREADS"] = 512;
  EXPECT_EQ(RunOne(kOtherRoot, 4096).grid.x, 1u);
  EXPECT_EQ(RunOne(kOtherRoot, 4112).grid.x, 2u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_NonRootChunksCappedByBudget) {
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 3;
  EXPECT_EQ(RunOne(kOtherRoot, kEightChunks).grid.x, 3u);
  EXPECT_EQ(RunOne(kOtherRoot, 2064).grid.x, 2u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_RootGetsOneColumnPerPeerAndChunkRows) {
  const std::tuple<int, size_t, unsigned, unsigned> cases[] = {
      {8, 16, 7, 1}, {8, 2064, 7, 2}, {8, kEightChunks, 7, 8}, {2, kEightChunks, 1, 8}, {144, kEightChunks, 143, 1}};
  for (const auto& [nRanks, bytes, x, y] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch launch = RunOne(kRank, bytes);
    EXPECT_EQ(launch.grid.x, x) << "nRanks " << nRanks << " bytes " << bytes;
    EXPECT_EQ(launch.grid.y, y) << "nRanks " << nRanks << " bytes " << bytes;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_RootSplitsBudgetAcrossPeersAtLeastOneRow) {
  const std::pair<int64_t, unsigned> cases[] = {{21, 3}, {14, 2}, {13, 1}, {3, 1}};
  for (const auto& [budget, rows] : cases) {
    params_["RCCL_DDA_BC_MAX_BLOCKS"] = budget;
    const DdaLaunch launch = RunOne(kRank, kEightChunks);
    EXPECT_EQ(launch.grid.x, 7u) << "budget " << budget;
    EXPECT_EQ(launch.grid.y, rows) << "budget " << budget;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Blocks_IsTheRootGridSize) {
  EXPECT_EQ(Blocks(16), 7u);
  EXPECT_EQ(Blocks(kEightChunks), 56u);
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 14;
  EXPECT_EQ(Blocks(kEightChunks), 14u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(Blocks(4096, ncclInt8), 14u);
  EXPECT_EQ(Blocks(4096, ncclFloat32), 56u);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_BaselineSmallPayload) {
  EXPECT_TRUE(Eligible(16));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_NeedsScratchPeersAndEpoch) {
  EXPECT_FALSE(ncclBroadcastDdaFabricLLEligible(nullptr, 16, ncclInt8));
  for (int missing = 0; missing < 3; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaScratch = missing == 0 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 1 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaLLEpochDev = missing == 2 ? nullptr : comm().ddaLLEpochDev;
    EXPECT_FALSE(Eligible(16)) << "missing resource " << missing;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(16));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(16));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(16));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(16));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_BytesMultipleOf16ForAnyDatatype) {
  EXPECT_FALSE(Eligible(8));
  EXPECT_FALSE(Eligible(24));
  EXPECT_FALSE(Eligible(2, ncclFloat32));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  for (const auto& [count, datatype] : {std::pair{16, ncclInt8}, std::pair{4, ncclUint32}, std::pair{4, ncclFloat32},
                                        std::pair{8, ncclBfloat16}, std::pair{2, ncclFloat64}}) {
    EXPECT_TRUE(Eligible(count, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_UpToBroadcastLLThreshold) {
  ncclFunc_t seenFunc = ncclFuncSendRecv;
  g_rcclDdaLLThreshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    seenFunc = func;
    return size_t{1024};
  };
  EXPECT_TRUE(Eligible(1024));
  EXPECT_EQ(seenFunc, ncclFuncBroadcast);
  EXPECT_FALSE(Eligible(1040));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_PayloadMustFitBankAfterTokenLines) {
  // A 1152 B bank is 72 lines: 8 rank tokens leave 64 packets, 512 B.
  comm().ddaScratchBytes = 2 * 1152;
  EXPECT_TRUE(Eligible(512));
  EXPECT_FALSE(Eligible(528));
  // The bank floors to 128 B.
  comm().ddaScratchBytes = 2 * 1152 + 254;
  EXPECT_TRUE(Eligible(512));
  EXPECT_FALSE(Eligible(528));
  // Four rank tokens leave 68 packets, 544 B.
  comm().nRanks = 4;
  EXPECT_TRUE(Eligible(544));
  EXPECT_FALSE(Eligible(560));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_RootGridMustFitEpoch) {
  // Root grid 7 x 8; non-root 8.
  comm().ddaLLEpochLen = 56;
  EXPECT_TRUE(Eligible(kEightChunks));
  comm().ddaLLEpochLen = 55;
  EXPECT_FALSE(Eligible(kEightChunks));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_NonRootGridMustFitEpoch) {
  // Budget 5 on 4 ranks: root grid 3 x 1, non-root 5.
  comm().nRanks = 4;
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 5;
  comm().ddaLLEpochLen = 5;
  EXPECT_TRUE(Eligible(kEightChunks));
  comm().ddaLLEpochLen = 4;
  EXPECT_FALSE(Eligible(kEightChunks));
}

TEST_F(DdaBroadcastFabricLLMicrotest, Eligible_SameAnswerOnEveryRank) {
  for (int epochLen : {55, 56}) {
    comm().ddaLLEpochLen = epochLen;
    comm().rank = 0;
    const bool expected = Eligible(kEightChunks);
    for (int rank = 1; rank < comm().nRanks; ++rank) {
      comm().rank = rank;
      EXPECT_EQ(Eligible(kEightChunks), expected) << "epochLen " << epochLen << " rank " << rank;
    }
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_AlignedKernelFromOwnBuffersOnly) {
  struct Case {
    int root;
    uintptr_t send, recv;
    bool aligned;
  };
  const Case cases[] = {
      {kOtherRoot, 0x1001, 0x2000, true},  // a non-root never reads sendbuff
      {kOtherRoot, 0x1000, 0x2002, false},
      {kRank, 0x1000, 0x2000, true},
      {kRank, 0x1000, 0x2004, true},  // 4-byte words suffice
      {kRank, 0x1001, 0x2000, false},
      {kRank, 0x1004, 0x2001, false},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(RunOne(c.root, 16, Addr(c.send), Addr(c.recv)).kernel, c.aligned ? Kernel<true>() : Kernel<false>())
        << "root " << c.root << " send " << std::hex << c.send << " recv " << c.recv;
  }
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_PassesBuffersBytesRanksEpochAndBank) {
  comm().ddaScratchBytes = 4096 + 40;
  ASSERT_EQ(Run(/*root=*/2, 8, kSendbuff, kRecvbuff, ncclFloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel<true>());
  EXPECT_EQ(launch.grid.x, 1u);
  EXPECT_EQ(launch.block.x, 256u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 11u);
  EXPECT_EQ(launch.Arg<char* const*>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<char*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const char*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 16u);
  EXPECT_EQ(launch.Arg<int>(4), kRank);
  EXPECT_EQ(launch.Arg<int>(5), 8);
  EXPECT_EQ(launch.Arg<int>(6), 2);
  EXPECT_EQ(launch.Arg<uint32_t*>(7), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(8), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(9), 2048u);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

TEST_F(DdaBroadcastFabricLLMicrotest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(kOtherRoot, 16), ncclUnhandledCudaError);
}

}  // namespace
