/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/broadcast/dda_broadcast_fabric_ll128.cu: the per-arch slice and
// threads override, the root/non-root grids, rank-uniform eligibility, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_BROADCAST_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
constexpr int kRank = 5;
constexpr int kOtherRoot = 0;
// Payload per slice: gfx9 64B lines on wave64, gfx1250 128B lines on wave32.
constexpr size_t kGfx9Slice = 3584;
constexpr size_t kGfx1250Slice = 1920;
// One group per block at the one-shot 256 threads: 4 gfx9 slices, 8 gfx1250 slices.
constexpr size_t kGfx9Group = 4 * kGfx9Slice;
constexpr size_t kGfx1250Group = 8 * kGfx1250Slice;
constexpr char kOneShotThreshold[] = "RCCL_DDA_LL128_BC_ONESHOT_THRESHOLD";
constexpr size_t kDefaultOneShotThreshold = 256 << 10;
constexpr size_t kTwoShotThreshold = 64 << 10;
// Two-shot on 8 ranks: 140 gfx9 slices, 20 per chunk.
constexpr size_t kTwoShotBytes = 7 * 20 * kGfx9Slice;

template <bool kAligned>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaBroadcastFabricLL128<kAligned>);
}

template <bool kAligned>
const void* TwoShotKernel() {
  return KernelPtr(&dda::common::ddaBroadcastFabricLL128TwoShot<kAligned>);
}

void* Addr(uintptr_t a) { return reinterpret_cast<void*>(a); }

class DdaBroadcastFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    // LL128 is off with no arch table; open the threshold so it is not under test.
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{64} << 20; };
    topo_->ll128Enabled = true;
    comm().topo = topo_.get();
    SetArch("gfx950");
    comm().WarpSize = 64;
    // One-shot unless a test opts into two-shot.
    params_[kOneShotThreshold] = int64_t{1} << 40;
    comm().rank = kRank;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  void SetArch(const char* arch) { comm().archName = const_cast<char*>(arch); }

  // Two-shot above 64 KiB, with the block pinned to 8 gfx9 warps.
  void UseTwoShot() {
    params_[kOneShotThreshold] = kTwoShotThreshold;
    params_["RCCL_DDA_LL128_BC_THREADS"] = 512;
  }

  // int8 so count is bytes; broadcast is datatype-agnostic.
  bool Eligible(size_t count, ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLL128Eligible(mock_.get(), count, datatype);
  }

  uint32_t Blocks(size_t count, ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLL128Blocks(mock_.get(), count, datatype);
  }

  ncclResult_t Run(int root, size_t count, const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff,
                   ncclDataType_t datatype = ncclInt8) {
    return ncclBroadcastDdaFabricLL128(sendbuff, recvbuff, count, datatype, root, mock_.get(), kStream);
  }

  DdaLaunch RunOne(int root, size_t count, const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff) {
    g_ddaLaunches.clear();
    comm().addonStopEvent = kStopEvent;
    EXPECT_EQ(Run(root, count, sendbuff, recvbuff), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.empty() ? DdaLaunch{} : g_ddaLaunches.back();
  }

  DdaFabricMockComm mock_;
  std::unique_ptr<ncclTopoSystem> topo_ = std::make_unique<ncclTopoSystem>();
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ThreadsOverrideMustBeAnArchWarpMultipleUpTo512) {
  const std::pair<int64_t, unsigned> wave64[] = {{512, 512}, {64, 64},   {128, 128}, {32, 256},
                                                 {96, 256},  {576, 256}, {0, 256},   {-64, 256}};
  for (const auto& [env, threads] : wave64) {
    params_["RCCL_DDA_LL128_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, threads) << "gfx950 DDA_LL128_BC_THREADS " << env;
  }
  // The warp follows the arch, not comm->WarpSize.
  SetArch("gfx1250");
  comm().WarpSize = 64;
  const std::pair<int64_t, unsigned> wave32[] = {{32, 32}, {96, 96}, {16, 256}, {48, 256}, {1024, 256}};
  for (const auto& [env, threads] : wave32) {
    params_["RCCL_DDA_LL128_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, threads) << "gfx1250 DDA_LL128_BC_THREADS " << env;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_NonRootGroupsAreWarpsOfArchSlices) {
  const std::pair<size_t, unsigned> gfx9[] = {
      {16, 1}, {kGfx9Group, 1}, {kGfx9Group + 16, 2}, {8 * kGfx9Group, 8}, {20 * kGfx9Group, 20}};
  for (const auto& [bytes, groups] : gfx9) {
    const DdaLaunch launch = RunOne(kOtherRoot, bytes);
    EXPECT_EQ(launch.grid.x, groups) << "gfx950 bytes " << bytes;
    EXPECT_EQ(launch.grid.y, 1u) << "gfx950 bytes " << bytes;
  }
  // Eight warps per block at 512 threads.
  params_["RCCL_DDA_LL128_BC_THREADS"] = 512;
  EXPECT_EQ(RunOne(kOtherRoot, 8 * kGfx9Slice).grid.x, 1u);
  EXPECT_EQ(RunOne(kOtherRoot, 8 * kGfx9Slice + 16).grid.x, 2u);
  params_.erase("RCCL_DDA_LL128_BC_THREADS");
  SetArch("gfx1250");
  EXPECT_EQ(RunOne(kOtherRoot, kGfx1250Group).grid.x, 1u);
  EXPECT_EQ(RunOne(kOtherRoot, kGfx1250Group + 16).grid.x, 2u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_NonRootGroupsCappedByBudget) {
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 3;
  EXPECT_EQ(RunOne(kOtherRoot, 20 * kGfx9Group).grid.x, 3u);
  EXPECT_EQ(RunOne(kOtherRoot, 2 * kGfx9Group).grid.x, 2u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_RootGetsOneColumnPerPeerAndGroupRows) {
  const std::tuple<int, size_t, unsigned, unsigned> cases[] = {
      {8, 16, 7, 1},       {8, 8 * kGfx9Group, 7, 8},   {8, 40 * kGfx9Group, 7, 36},
      {2, 8 * kGfx9Group, 1, 8}, {144, 8 * kGfx9Group, 143, 1}};
  for (const auto& [nRanks, bytes, x, y] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch launch = RunOne(kRank, bytes);
    EXPECT_EQ(launch.grid.x, x) << "nRanks " << nRanks << " bytes " << bytes;
    EXPECT_EQ(launch.grid.y, y) << "nRanks " << nRanks << " bytes " << bytes;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_RootSplitsBudgetAcrossPeersAtLeastOneRow) {
  const std::pair<int64_t, unsigned> cases[] = {{21, 3}, {14, 2}, {13, 1}, {3, 1}};
  for (const auto& [budget, rows] : cases) {
    params_["RCCL_DDA_BC_MAX_BLOCKS"] = budget;
    const DdaLaunch launch = RunOne(kRank, 8 * kGfx9Group);
    EXPECT_EQ(launch.grid.x, 7u) << "budget " << budget;
    EXPECT_EQ(launch.grid.y, rows) << "budget " << budget;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Blocks_IsTheRootGridSize) {
  EXPECT_EQ(Blocks(16), 7u);
  EXPECT_EQ(Blocks(8 * kGfx9Group), 56u);
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 14;
  EXPECT_EQ(Blocks(8 * kGfx9Group), 14u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(Blocks(2 * kGfx9Group, ncclInt8), 14u);
  EXPECT_EQ(Blocks(2 * kGfx9Group, ncclFloat32), 56u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Blocks_FollowArchSliceSize) {
  EXPECT_EQ(Blocks(kGfx9Group + 16), 14u);
  SetArch("gfx1250");
  EXPECT_EQ(Blocks(kGfx9Group + 16), 7u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_BaselineOnGfx9AndGfx1250) {
  EXPECT_TRUE(Eligible(16));
  SetArch("gfx1250");
  EXPECT_TRUE(Eligible(16));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NeedsScratchPeersAndEpoch) {
  EXPECT_FALSE(ncclBroadcastDdaFabricLL128Eligible(nullptr, 16, ncclInt8));
  for (int missing = 0; missing < 4; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaScratch = missing == 0 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 1 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaLLEpochDev = missing == 2 ? nullptr : comm().ddaLLEpochDev;
    comm().ddaLLEpochLen = missing == 3 ? 0 : comm().ddaLLEpochLen;
    EXPECT_FALSE(Eligible(16)) << "missing resource " << missing;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_Gfx9NeedsLL128EnabledGfx1250DoesNot) {
  topo_->ll128Enabled = false;
  EXPECT_FALSE(Eligible(16));
  SetArch("gfx942");
  EXPECT_FALSE(Eligible(16));
  SetArch("gfx1250");
  EXPECT_TRUE(Eligible(16));
  comm().topo = nullptr;
  EXPECT_TRUE(Eligible(16));
  SetArch("gfx950");
  EXPECT_FALSE(Eligible(16));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(16));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(16));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(16));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(16));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_BytesMultipleOf16ForAnyDatatype) {
  EXPECT_FALSE(Eligible(8));
  EXPECT_FALSE(Eligible(24));
  EXPECT_FALSE(Eligible(2, ncclFloat32));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  for (const auto& [count, datatype] : {std::pair{16, ncclInt8}, std::pair{4, ncclUint32}, std::pair{4, ncclFloat32},
                                        std::pair{8, ncclBfloat16}, std::pair{2, ncclFloat64}}) {
    EXPECT_TRUE(Eligible(count, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_UpToBroadcastLL128Threshold) {
  ncclFunc_t seenFunc = ncclFuncSendRecv;
  g_rcclDdaLL128Threshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    seenFunc = func;
    return size_t{1024};
  };
  EXPECT_TRUE(Eligible(1024));
  EXPECT_EQ(seenFunc, ncclFuncBroadcast);
  EXPECT_FALSE(Eligible(1040));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_SlicesMustFitBankAfterTokenRegion) {
  // gfx9: 512-word slices, 8 rank tokens pad to one slice; a 1536-word bank leaves two.
  comm().ddaScratchBytes = 2 * 1536 * 8;
  EXPECT_TRUE(Eligible(2 * kGfx9Slice));
  EXPECT_FALSE(Eligible(2 * kGfx9Slice + 16));
  // gfx1250: 256-word slices, a 768-word bank leaves two.
  SetArch("gfx1250");
  comm().ddaScratchBytes = 2 * 768 * 8;
  EXPECT_TRUE(Eligible(2 * kGfx1250Slice));
  EXPECT_FALSE(Eligible(2 * kGfx1250Slice + 16));
  // 144 rank tokens pad to two slices, leaving one.
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(kGfx1250Slice));
  EXPECT_FALSE(Eligible(kGfx1250Slice + 16));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_RootGridMustFitEpoch) {
  // Root grid 7 x 8; non-root 8.
  comm().ddaLLEpochLen = 56;
  EXPECT_TRUE(Eligible(8 * kGfx9Group));
  comm().ddaLLEpochLen = 55;
  EXPECT_FALSE(Eligible(8 * kGfx9Group));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NonRootGridMustFitEpoch) {
  // Budget 5 on 4 ranks: root grid 3 x 1, non-root 5.
  comm().nRanks = 4;
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 5;
  comm().ddaLLEpochLen = 5;
  EXPECT_TRUE(Eligible(8 * kGfx9Group));
  comm().ddaLLEpochLen = 4;
  EXPECT_FALSE(Eligible(8 * kGfx9Group));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_SameAnswerOnEveryRank) {
  for (int epochLen : {55, 56}) {
    comm().ddaLLEpochLen = epochLen;
    comm().rank = 0;
    const bool expected = Eligible(8 * kGfx9Group);
    for (int rank = 1; rank < comm().nRanks; ++rank) {
      comm().rank = rank;
      EXPECT_EQ(Eligible(8 * kGfx9Group), expected) << "epochLen " << epochLen << " rank " << rank;
    }
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_AlignedKernelFrom16ByteOwnBuffersOnly) {
  struct Case {
    int root;
    uintptr_t send, recv;
    bool aligned;
  };
  const Case cases[] = {
      {kOtherRoot, 0x1008, 0x2000, true},  // a non-root never reads sendbuff
      {kOtherRoot, 0x1000, 0x2008, false},
      {kRank, 0x1000, 0x2000, true},
      {kRank, 0x1000, 0x2010, true},
      {kRank, 0x1008, 0x2000, false},
      {kRank, 0x1000, 0x2004, false},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(RunOne(c.root, 16, Addr(c.send), Addr(c.recv)).kernel, c.aligned ? Kernel<true>() : Kernel<false>())
        << "root " << c.root << " send " << std::hex << c.send << " recv " << c.recv;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_PassesBuffersBytesRanksEpochAndBank) {
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

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(kOtherRoot, 16), ncclUnhandledCudaError);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotAboveOneShotThreshold) {
  params_[kOneShotThreshold] = kTwoShotThreshold;
  for (int root : {kOtherRoot, kRank}) {
    EXPECT_EQ(RunOne(root, kTwoShotThreshold).kernel, Kernel<true>()) << "root " << root;
    EXPECT_EQ(RunOne(root, kTwoShotThreshold + 16).kernel, TwoShotKernel<true>()) << "root " << root;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_OneShotUpToDefaultThreshold) {
  params_.erase(kOneShotThreshold);
  EXPECT_EQ(RunOne(kOtherRoot, kDefaultOneShotThreshold).kernel, Kernel<true>());
  EXPECT_EQ(RunOne(kOtherRoot, kDefaultOneShotThreshold + 16).kernel, TwoShotKernel<true>());
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_OneShotDefaultsTo256Threads) {
  params_[kOneShotThreshold] = kTwoShotThreshold;
  EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, 256u);
  EXPECT_EQ(RunOne(kRank, kTwoShotThreshold).block.x, 256u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotThreadsFollowChunkSizeUnlessOverridden) {
  params_[kOneShotThreshold] = kTwoShotThreshold;
  // Chunk is bytes / (nRanks - 1): 128 up to 512 KiB, 256 up to 4 MiB, 512 above.
  const std::tuple<int, size_t, unsigned> cases[] = {
      {8, 7 * (512 << 10), 128},      {8, 7 * (512 << 10) + 16, 256}, {8, 7 * (4 << 20), 256},
      {8, 7 * (4 << 20) + 16, 512},   {2, 512 << 10, 128},            {2, (512 << 10) + 16, 256}};
  for (const auto& [nRanks, bytes, threads] : cases) {
    comm().nRanks = nRanks;
    EXPECT_EQ(RunOne(kOtherRoot, bytes).block.x, threads) << "nRanks " << nRanks << " bytes " << bytes;
    EXPECT_EQ(RunOne(kRank, bytes).block.x, threads) << "nRanks " << nRanks << " bytes " << bytes;
  }
  comm().nRanks = 8;
  params_["RCCL_DDA_LL128_BC_THREADS"] = 64;
  EXPECT_EQ(RunOne(kOtherRoot, 7 * (4 << 20) + 16).block.x, 64u);
  EXPECT_EQ(RunOne(kOtherRoot, 7 * (512 << 10)).block.x, 64u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotRootColumnsEachCoverOneChunk) {
  UseTwoShot();
  // Rows of 8 warps over ceil(slices / 7) slices per chunk.
  const std::pair<size_t, unsigned> cases[] = {
      {kTwoShotBytes, 3}, {kTwoShotBytes + 16, 3}, {7 * 24 * kGfx9Slice, 3}, {7 * 25 * kGfx9Slice, 4}};
  for (const auto& [bytes, rows] : cases) {
    const DdaLaunch launch = RunOne(kRank, bytes);
    EXPECT_EQ(launch.grid.x, 7u) << "bytes " << bytes;
    EXPECT_EQ(launch.grid.y, rows) << "bytes " << bytes;
    EXPECT_EQ(Blocks(bytes), 7 * rows) << "bytes " << bytes;
  }
  comm().nRanks = 2;
  EXPECT_EQ(RunOne(kRank, kTwoShotBytes).grid.y, 18u);
  comm().nRanks = 8;
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 14;
  EXPECT_EQ(RunOne(kRank, kTwoShotBytes).grid.y, 2u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotNonRootCoversEveryChunk) {
  UseTwoShot();
  // 7 chunks of 20 or 21 slices in 8-warp blocks.
  const std::pair<size_t, unsigned> cases[] = {{kTwoShotBytes, 18}, {kTwoShotBytes + 16, 19}};
  for (const auto& [bytes, blocks] : cases) {
    const DdaLaunch launch = RunOne(kOtherRoot, bytes);
    EXPECT_EQ(launch.grid.x, blocks) << "bytes " << bytes;
    EXPECT_EQ(launch.grid.y, 1u) << "bytes " << bytes;
  }
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 5;
  EXPECT_EQ(RunOne(kOtherRoot, kTwoShotBytes).grid.x, 5u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotEmptyChunksStillGetAColumnAndATask) {
  UseTwoShot();
  // 19 slices over 31 peers: chunks of 1 slice, the last 12 empty.
  comm().nRanks = 32;
  const DdaLaunch root = RunOne(kRank, kTwoShotThreshold + 16);
  EXPECT_EQ(root.kernel, TwoShotKernel<true>());
  EXPECT_EQ(root.grid.x, 31u);
  EXPECT_EQ(root.grid.y, 1u);
  EXPECT_EQ(RunOne(kOtherRoot, kTwoShotThreshold + 16).grid.x, 4u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotTwoRanksNonRootCoversTheOneChunk) {
  UseTwoShot();
  comm().nRanks = 2;
  const DdaLaunch launch = RunOne(kOtherRoot, kTwoShotBytes);
  EXPECT_EQ(launch.kernel, TwoShotKernel<true>());
  EXPECT_EQ(launch.grid.x, 18u);
  EXPECT_EQ(launch.grid.y, 1u);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotPassesBuffersBytesRanksEpochAndBank) {
  UseTwoShot();
  comm().ddaScratchBytes = 4096 + 40;
  ASSERT_EQ(Run(/*root=*/2, kTwoShotBytes / 2, kSendbuff, kRecvbuff, ncclFloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, TwoShotKernel<true>());
  EXPECT_EQ(launch.grid.x, 18u);
  EXPECT_EQ(launch.block.x, 512u);
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
  EXPECT_EQ(launch.Arg<size_t>(3), kTwoShotBytes);
  EXPECT_EQ(launch.Arg<int>(4), kRank);
  EXPECT_EQ(launch.Arg<int>(5), 8);
  EXPECT_EQ(launch.Arg<int>(6), 2);
  EXPECT_EQ(launch.Arg<uint32_t*>(7), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(8), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(9), 2048u);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_TwoShotRootGridMustFitEpoch) {
  UseTwoShot();
  // Root grid 7 x 3; non-root 18.
  comm().ddaLLEpochLen = 21;
  EXPECT_TRUE(Eligible(kTwoShotBytes));
  comm().ddaLLEpochLen = 20;
  EXPECT_FALSE(Eligible(kTwoShotBytes));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_TwoShotNonRootGridMustFitEpoch) {
  UseTwoShot();
  // Budget 5 on 4 ranks, 3 chunks of 16 slices: root grid 3 x 1, non-root 5.
  comm().nRanks = 4;
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 5;
  comm().ddaLLEpochLen = 5;
  EXPECT_TRUE(Eligible(3 * 16 * kGfx9Slice));
  comm().ddaLLEpochLen = 4;
  EXPECT_FALSE(Eligible(3 * 16 * kGfx9Slice));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_TwoShotAlignedKernelFrom16ByteOwnBuffersOnly) {
  UseTwoShot();
  struct Case {
    int root;
    uintptr_t send, recv;
    bool aligned;
  };
  const Case cases[] = {
      {kOtherRoot, 0x1008, 0x2000, true},
      {kOtherRoot, 0x1000, 0x2008, false},
      {kRank, 0x1000, 0x2010, true},
      {kRank, 0x1008, 0x2000, false},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(RunOne(c.root, kTwoShotBytes, Addr(c.send), Addr(c.recv)).kernel,
              c.aligned ? TwoShotKernel<true>() : TwoShotKernel<false>())
        << "root " << c.root << " send " << std::hex << c.send << " recv " << c.recv;
  }
}

}  // namespace
