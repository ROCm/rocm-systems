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
// One group per block at the default 512 threads: 8 gfx9 slices, 16 gfx1250 slices.
constexpr size_t kGfx9Group = 8 * kGfx9Slice;
constexpr size_t kGfx1250Group = 16 * kGfx1250Slice;

template <bool kAligned>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaBroadcastFabricLL128<kAligned>);
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
    SetArch("gfx950");
    comm().WarpSize = 64;
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
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaBroadcastFabricLL128Microtest, Launch_ThreadsOverrideMustBeAnArchWarpMultipleUpTo512) {
  const std::pair<int64_t, unsigned> wave64[] = {{512, 512}, {64, 64},   {256, 256}, {32, 512},
                                                 {96, 512},  {576, 512}, {0, 512},   {-64, 512}};
  for (const auto& [env, threads] : wave64) {
    params_["RCCL_DDA_LL128_BC_THREADS"] = env;
    EXPECT_EQ(RunOne(kOtherRoot, 16).block.x, threads) << "gfx950 DDA_LL128_BC_THREADS " << env;
  }
  // The warp follows the arch, not comm->WarpSize.
  SetArch("gfx1250");
  comm().WarpSize = 64;
  const std::pair<int64_t, unsigned> wave32[] = {{32, 32}, {96, 96}, {16, 512}, {48, 512}, {1024, 512}};
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
  // Four warps per block at 256 threads.
  params_["RCCL_DDA_LL128_BC_THREADS"] = 256;
  EXPECT_EQ(RunOne(kOtherRoot, 4 * kGfx9Slice).grid.x, 1u);
  EXPECT_EQ(RunOne(kOtherRoot, 4 * kGfx9Slice + 16).grid.x, 2u);
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

// gfx9 stays off until a cross-GPU gfx9 data test exists.
TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_Gfx1250Only) {
  for (const char* arch : {"gfx950", "gfx942"}) {
    SetArch(arch);
    EXPECT_FALSE(Eligible(16)) << arch;
  }
  SetArch("gfx1250");
  EXPECT_TRUE(Eligible(16));
}

// IPC scratch has no fabric handler.
TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_RejectsIpcScratch) {
  SetArch("gfx1250");
  mock_.setFabricResourcesPresent(true);
  comm().ddaFabricMemHandler = nullptr;
  comm().ddaIpcMemHandler = reinterpret_cast<ncclIpcMemHandler*>(0x1);
  EXPECT_FALSE(Eligible(16));
  comm().ddaIpcMemHandler = nullptr;
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NeedsScratchPeersAndEpoch) {
  SetArch("gfx1250");
  EXPECT_FALSE(ncclBroadcastDdaFabricLL128Eligible(nullptr, 16, ncclInt8));
  for (int missing = 0; missing < 5; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 4 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 0 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 1 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaLLEpochDev = missing == 2 ? nullptr : comm().ddaLLEpochDev;
    comm().ddaLLEpochLen = missing == 3 ? 0 : comm().ddaLLEpochLen;
    EXPECT_FALSE(Eligible(16)) << "missing resource " << missing;
  }
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  SetArch("gfx1250");
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
  SetArch("gfx1250");
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_BytesMultipleOf16ForAnyDatatype) {
  SetArch("gfx1250");
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
  SetArch("gfx1250");
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
  // 256-word slices, 8 rank tokens pad to one slice; a 768-word bank leaves two.
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
  SetArch("gfx1250");
  comm().ddaLLEpochLen = 56;
  EXPECT_TRUE(Eligible(8 * kGfx1250Group));
  comm().ddaLLEpochLen = 55;
  EXPECT_FALSE(Eligible(8 * kGfx1250Group));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_NonRootGridMustFitEpoch) {
  // Budget 5 on 4 ranks: root grid 3 x 1, non-root 5.
  SetArch("gfx1250");
  comm().nRanks = 4;
  params_["RCCL_DDA_BC_MAX_BLOCKS"] = 5;
  comm().ddaLLEpochLen = 5;
  EXPECT_TRUE(Eligible(8 * kGfx1250Group));
  comm().ddaLLEpochLen = 4;
  EXPECT_FALSE(Eligible(8 * kGfx1250Group));
}

TEST_F(DdaBroadcastFabricLL128Microtest, Eligible_SameAnswerOnEveryRank) {
  SetArch("gfx1250");
  for (int epochLen : {55, 56}) {
    comm().ddaLLEpochLen = epochLen;
    comm().rank = 0;
    const bool expected = Eligible(8 * kGfx1250Group);
    for (int rank = 1; rank < comm().nRanks; ++rank) {
      comm().rank = rank;
      EXPECT_EQ(Eligible(8 * kGfx1250Group), expected) << "epochLen " << epochLen << " rank " << rank;
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

}  // namespace
