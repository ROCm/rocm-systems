/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu: eligibility against the
// LL128 threshold and slot, the block size and grid budget from the arch table, the peer-column grid, and the launch.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_ALL_GATHER_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A slice carries 1920 payload bytes in 2 KiB on the wire.
constexpr size_t kSliceData = 1920;
constexpr size_t kSliceWire = 2048;
constexpr size_t kMiB = size_t{1} << 20;

// fp32 count per rank that gathers totalBytes across the fixture's four ranks.
constexpr size_t CountFor(size_t totalBytes) { return totalBytes / (4 * sizeof(float)); }

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllGatherFabricLL128<int8_t, N>);
}

const void* StagedKernel() {
  return KernelPtr(&dda::common::ddaAllGatherFabricLL128Staged<int8_t>);
}

class DdaAllGatherFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    // The LL128 tier is off without an arch table; give it a cap well above anything tested.
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{1} << 30; };
    comm().nRanks = 4;
    comm().rank = 2;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t sendcount, ncclDataType_t datatype = ncclFloat32, const void* sendbuff = kSendbuff,
                void* recvbuff = kRecvbuff) {
    return ncclAllGatherDdaFabricLL128Eligible(mock_.get(), sendbuff, recvbuff, sendcount, datatype);
  }

  ncclResult_t Run(size_t sendcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllGatherDdaFabricLL128(kSendbuff, kRecvbuff, sendcount, datatype, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t sendcount) {
    g_ddaLaunches.clear();
    comm().addonStopEvent = kStopEvent;  // Each launch takes it.
    EXPECT_EQ(Run(sendcount), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

// LL128 is gated by RCCL_DDA_LL128 alone; RCCL_DDA_LL only controls the LL tier.
TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_DdaLLZeroDoesNotGateLL128) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_NeedsHandlerScratchPeersAndEpoch) {
  comm().ddaFabricBarrierState = nullptr;
  EXPECT_TRUE(Eligible(4));
  for (int missing = 0; missing < 5; ++missing) {
    mock_.setFabricResourcesPresent(true);
    comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
    comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
    comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
    comm().ddaLLEpochDev = missing == 3 ? nullptr : comm().ddaLLEpochDev;
    comm().ddaLLEpochLen = missing == 4 ? 0 : comm().ddaLLEpochLen;
    EXPECT_FALSE(Eligible(4)) << "missing resource " << missing;
  }
  comm().ddaLLEpochLen = 1;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
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

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_PerRankBytesAndBothBuffersAligned) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_FALSE(Eligible(4, ncclFloat32, reinterpret_cast<void*>(0x1008)));
  EXPECT_FALSE(Eligible(4, ncclFloat32, kSendbuff, reinterpret_cast<void*>(0x2008)));
  EXPECT_TRUE(Eligible(4, ncclFloat32, reinterpret_cast<void*>(0x1010), reinterpret_cast<void*>(0x2030)));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_GatheredTotalUpToAllGatherLL128Threshold) {
  ncclFunc_t seenFunc = ncclFuncSendRecv;
  g_rcclDdaLL128Threshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    seenFunc = func;
    return size_t{4 * 64};
  };
  EXPECT_TRUE(Eligible(16));
  EXPECT_EQ(seenFunc, ncclFuncAllGather);
  EXPECT_FALSE(Eligible(20));
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{0}; };
  EXPECT_FALSE(Eligible(4));
}

// The slot is a whole number of 2 KiB slices out of a bank split four ways.
TEST_F(DdaAllGatherFabricLL128Microtest, Eligible_PerRankBytesMustFitTheSlotSlices) {
  comm().ddaScratchBytes = 2 * 4 * 3 * kSliceWire;
  EXPECT_TRUE(Eligible(3 * kSliceData / 4));
  EXPECT_FALSE(Eligible(3 * kSliceData / 4 + 4));
  comm().ddaScratchBytes = 2 * 4 * (4 * kSliceWire - 16);
  EXPECT_FALSE(Eligible(3 * kSliceData / 4 + 4));
  comm().ddaScratchBytes = 2 * 4 * kSliceWire - 32;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {6, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.grid.x, static_cast<unsigned>(nRanks - 1));
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_PassesBuffersBytesEpochSlicesAndBank) {
  comm().ddaScratchBytes = 4096 + 40;
  ASSERT_EQ(Run(1000, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel<4>());
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 11u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const int8_t*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 2000u);
  EXPECT_EQ(launch.Arg<int>(4), 2);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 2u);
  EXPECT_EQ(launch.Arg<size_t>(9), 2048u);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

// A warp per slice: 512 threads give 16 warps, so a column grows past one block at 17 slices.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_BlocksPerPeerFollowSlicesCappedByShareOfMaxBlocks) {
  const std::pair<size_t, unsigned> cases[] = {
      {4, 1}, {16 * kSliceData / 4, 1}, {17 * kSliceData / 4, 2}, {100 * 16 * kSliceData / 4, 85}};
  for (const auto& [count, bpp] : cases) {
    const DdaLaunch& launch = RunOne(count);
    EXPECT_EQ(launch.grid.x, 3u);
    EXPECT_EQ(launch.grid.y, bpp) << "count " << count;
    EXPECT_EQ(launch.block.x, 512u);
    EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32), 3 * bpp);
  }
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_MaxBlocksBelowPeerCountStillGivesEachPeerABlock) {
  for (int maxBlocks : {0, 2}) {
    comm().ddaFabricMaxBlocks = maxBlocks;
    const DdaLaunch& launch = RunOne(1u << 20);
    EXPECT_EQ(launch.grid.x, 3u) << "maxBlocks " << maxBlocks;
    EXPECT_EQ(launch.grid.y, 1u) << "maxBlocks " << maxBlocks;
  }
  comm().ddaFabricMaxBlocks = 7;
  EXPECT_EQ(RunOne(1u << 20).grid.y, 2u);
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_ThreadsOverrideMustBeAWarpMultipleUpTo1024) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {32, 32}, {1024, 1024}, {16, 512},
                                                {48, 512},  {1056, 512}, {0, 512}, {-32, 512}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_AG_THREADS"] = env;
    EXPECT_EQ(RunOne(4).block.x, threads) << "DDA_LL128_AG_THREADS " << env;
  }
  // 256 threads are 8 warps, so 9 slices take a second block.
  params_["RCCL_DDA_LL128_AG_THREADS"] = 256;
  EXPECT_EQ(RunOne(9 * kSliceData / 4).grid.y, 2u);
  EXPECT_EQ(RunOne(8 * kSliceData / 4).grid.y, 1u);
}

// The arch table picks the block size for the gathered total, 512 when it sets none.
// A valid DDA_LL128_AG_THREADS overrides both.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_ThreadsFromArchTableUnlessOverridden) {
  size_t total = 0;
  g_rcclDdaFabricAgLL128Threads = [&total](const ncclComm*, size_t totalBytes) {
    total = totalBytes;
    return 256;
  };
  EXPECT_EQ(RunOne(CountFor(8 * kMiB)).block.x, 256u);
  EXPECT_EQ(total, 8 * kMiB);
  params_["RCCL_DDA_LL128_AG_THREADS"] = 512;
  EXPECT_EQ(RunOne(CountFor(8 * kMiB)).block.x, 512u);
  params_.erase("RCCL_DDA_LL128_AG_THREADS");
  g_rcclDdaFabricAgLL128Threads = [](const ncclComm*, size_t) { return 0; };
  EXPECT_EQ(RunOne(CountFor(8 * kMiB)).block.x, 512u);
}

// The arch table's block cap for the gathered total bounds the grid. One warp per block lets it bind on four ranks.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_BlockBudgetFromArchTable) {
  params_["RCCL_DDA_LL128_AG_THREADS"] = 32;
  size_t total = 0;
  g_rcclDdaFabricAgLL128Blocks = [&total](const ncclComm*, size_t totalBytes) {
    total = totalBytes;
    return 75;
  };
  const DdaLaunch& launch = RunOne(CountFor(8 * kMiB));
  EXPECT_EQ(total, 8 * kMiB);
  EXPECT_EQ(launch.grid.x, 3u);
  EXPECT_EQ(launch.grid.y, 25u);
  EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), CountFor(8 * kMiB), ncclFloat32), 75u);
}

TEST_F(DdaAllGatherFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 2, ncclFloat16), 6u);
  EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 2, ncclFloat32), 9u);
}

// The knob at 0 or negative: a gfx1250 wave32 comm launches the unstaged kernels with no LDS.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireOffAtZeroOrNegative) {
  comm().WarpSize = 32;
  const std::pair<int, const void*> cases[] = {{4, Kernel<4>()}, {8, Kernel<8>()}, {6, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    for (const int64_t env : {int64_t{-1}, int64_t{0}}) {
      params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = env;
      for (const size_t count : {size_t{4}, 8 * kMiB / 4}) {  // 16 B and 8 MiB per rank
        const DdaLaunch& launch = RunOne(count);
        EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks << " env " << env << " count " << count;
        EXPECT_EQ(launch.sharedMemBytes, 0u) << "nRanks " << nRanks << " env " << env << " count " << count;
        EXPECT_EQ(launch.args.size(), 11u);
      }
    }
  }
}

// With the knob on, a gfx1250 wave32 comm launches the staged kernel on the same grid, block and arguments, plus
// its LDS and pushPeers.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireOnTheSameGrid) {
  comm().WarpSize = 32;
  const std::pair<int, const void*> cases[] = {{4, Kernel<4>()}, {8, Kernel<8>()}, {6, Kernel<0>()}, {32, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    for (const size_t perRank : {size_t{1} * kMiB, size_t{2} * kMiB, size_t{8} * kMiB}) {
      SCOPED_TRACE(testing::Message() << nRanks << "r perRank " << perRank);
      params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = 0;
      const DdaLaunch coop = RunOne(perRank / 4);
      EXPECT_EQ(coop.kernel, kernel);
      params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = 1;
      const DdaLaunch launch = RunOne(perRank / 4);
      EXPECT_EQ(launch.kernel, StagedKernel());
      EXPECT_EQ(launch.grid.x, coop.grid.x);
      EXPECT_EQ(launch.grid.y, coop.grid.y);
      EXPECT_EQ(launch.block.x, coop.block.x);
      EXPECT_EQ(launch.sharedMemBytes, dda::common::ddaLL128StageBytes(launch.block.x));
      EXPECT_EQ(launch.stream, kStream);
      EXPECT_EQ(launch.stopEvent, kStopEvent);
      ASSERT_EQ(launch.args.size(), 12u);
      EXPECT_EQ(launch.Arg<int8_t**>(0), coop.Arg<int8_t**>(0));
      EXPECT_EQ(launch.Arg<int8_t*>(1), coop.Arg<int8_t*>(1));
      EXPECT_EQ(launch.Arg<const int8_t*>(2), coop.Arg<const int8_t*>(2));
      EXPECT_EQ(launch.Arg<size_t>(3), perRank);
      EXPECT_EQ(launch.Arg<int>(4), coop.Arg<int>(4));
      EXPECT_EQ(launch.Arg<int>(5), nRanks);
      EXPECT_EQ(launch.Arg<uint32_t*>(6), coop.Arg<uint32_t*>(6));
      EXPECT_EQ(launch.Arg<int>(7), coop.Arg<int>(7));
      EXPECT_EQ(launch.Arg<size_t>(8), coop.Arg<size_t>(8));
      EXPECT_EQ(launch.Arg<size_t>(9), coop.Arg<size_t>(9));
      EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, coop.Arg<dda::common::DdaKernelParams>(10).flag);
      EXPECT_EQ(launch.Arg<int>(11), dda::common::kDdaLL128AGStagePushPeers);
    }
  }
}

// The knob is in pushed bytes a block: 32r (31x8) at 1 MiB a rank and 4r (3x85) at 85 x 128 KiB both push 128 KiB.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireFromPayloadBytesPerBlock) {
  comm().WarpSize = 32;
  constexpr size_t kPerBlock = 128 * 1024;
  const struct {
    int nRanks;
    const void* kernel;
    unsigned bpp;
  } cases[] = {{32, Kernel<0>(), 8}, {4, Kernel<4>(), 85}};
  for (const auto& c : cases) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r");
    comm().nRanks = c.nRanks;
    const size_t perRank = kPerBlock * c.bpp;
    params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = (int64_t)kPerBlock;
    const DdaLaunch& at = RunOne(perRank / 4);
    ASSERT_EQ(at.grid.x, (unsigned)(c.nRanks - 1));
    ASSERT_EQ(at.grid.y, c.bpp);
    ASSERT_EQ(at.block.x, 512u);
    EXPECT_EQ(at.kernel, StagedKernel()) << "exactly the knob's bytes per block";
    EXPECT_EQ(RunOne((perRank - 16) / 4).kernel, c.kernel) << "one 16 B chunk a rank short of the knob";
    params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = (int64_t)kPerBlock + 1;
    EXPECT_EQ(RunOne(perRank / 4).kernel, c.kernel) << "one byte a block above";
  }
}

// The staged kernel needs async LDS stores (gfx1250), an LL128 warp that is one wave, and blocks of 512 threads.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireOnlyOnGfx1250Wave32At512Threads) {
  params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = 1;
  const size_t count = 8 * kMiB / 4;
  EXPECT_EQ(RunOne(count).kernel, Kernel<4>()) << "WarpSize 0";
  comm().WarpSize = 64;
  EXPECT_EQ(RunOne(count).kernel, Kernel<4>()) << "an LL128 warp must be one wave";
  comm().WarpSize = 32;
  EXPECT_EQ(RunOne(count).kernel, StagedKernel());
  comm().archName = const_cast<char*>("gfx1201");
  EXPECT_EQ(RunOne(count).kernel, Kernel<4>()) << "the async LDS stores are gfx1250 only";
  comm().archName = nullptr;
  EXPECT_EQ(RunOne(count).kernel, Kernel<4>()) << "no arch";
  comm().archName = const_cast<char*>("gfx1250");
  for (const auto& [threads, staged] : {std::pair<int64_t, bool>{128, false}, {256, false}, {512, true}, {544, false},
                                        {1024, false}}) {
    params_["RCCL_DDA_LL128_AG_THREADS"] = threads;
    const DdaLaunch& launch = RunOne(count);
    EXPECT_EQ(launch.kernel, staged ? StagedKernel() : Kernel<4>()) << "threads " << threads;
    EXPECT_EQ(launch.sharedMemBytes, staged ? dda::common::ddaLL128StageBytes((unsigned)threads) : 0u)
      << "threads " << threads;
  }
}

// Unset, the knob stages from 80 KiB a block with 4 peers a fill: 4r (3x85) at 85 x 80 KiB a rank, not 16 B less.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireDefaults) {
  comm().WarpSize = 32;
  params_.erase("RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES");
  comm().nRanks = 4;
  constexpr size_t kPerRank = 85 * 80 * 1024;
  const DdaLaunch& at = RunOne(kPerRank / 4);
  ASSERT_EQ(at.grid.x, 3u);
  ASSERT_EQ(at.grid.y, 85u);
  ASSERT_EQ(at.block.x, 512u);
  EXPECT_EQ(at.kernel, StagedKernel()) << "exactly 80 KiB a block";
  EXPECT_EQ(at.Arg<int>(11), 4);
  EXPECT_EQ(RunOne((kPerRank - 16) / 4).kernel, Kernel<4>()) << "one 16 B chunk a rank short of 80 KiB a block";
  EXPECT_EQ(RunOne(size_t{4}).kernel, Kernel<4>()) << "16 B a rank stays unstaged";
}

// A 256-thread block from the arch table keeps today's kernel on the same grid, at a size that stages at 512 threads.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_AsyncWireNotAt256ThreadsFromArchTable) {
  comm().WarpSize = 32;
  params_.erase("RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES");
  const size_t count = 8 * kMiB / 4;  // 8 MiB a rank: about 96 KiB a block on 4r
  const DdaLaunch staged = RunOne(count);
  ASSERT_EQ(staged.block.x, 512u);
  ASSERT_EQ(staged.kernel, StagedKernel());
  g_rcclDdaFabricAgLL128Threads = [](const ncclComm*, size_t) { return 256; };
  const DdaLaunch& launch = RunOne(count);
  ASSERT_EQ(launch.block.x, 256u);
  EXPECT_EQ(launch.grid.x, staged.grid.x);
  EXPECT_EQ(launch.grid.y, staged.grid.y) << "the same bytes a block";
  EXPECT_EQ(launch.kernel, Kernel<4>());
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.args.size(), 11u);
}

// Peer groups: ceil(nPeers / pushPeers) groups, group g holds peer indices g, g + groups, ...; sizes cover nPeers.
TEST_F(DdaAllGatherFabricLL128Microtest, StageGroups_CoverEveryPeerOnce) {
  using dda::common::ddaLL128AGStageGroupPeers;
  using dda::common::ddaLL128AGStageGroups;
  EXPECT_EQ(ddaLL128AGStageGroups(31, 15), 3);
  EXPECT_EQ(ddaLL128AGStageGroupPeers(0, 3, 31), 11);
  EXPECT_EQ(ddaLL128AGStageGroupPeers(2, 3, 31), 10);
  EXPECT_EQ(ddaLL128AGStageGroups(31, dda::common::kDdaLL128AGStagePushPeers), 8);
  for (int nPeers = 1; nPeers <= 143; ++nPeers) {
    for (int pushPeers = 1; pushPeers <= dda::common::kDdaLL128AGStageMaxPushPeers; ++pushPeers) {
      const int groups = ddaLL128AGStageGroups(nPeers, pushPeers);
      int sum = 0;
      for (int g = 0; g < groups; ++g) {
        const int cnt = ddaLL128AGStageGroupPeers(g, groups, nPeers);
        EXPECT_GE(cnt, 1) << nPeers << " peers, " << pushPeers << " a fill, group " << g;
        EXPECT_LE(cnt, pushPeers) << nPeers << " peers, " << pushPeers << " a fill, group " << g;
        sum += cnt;
      }
      EXPECT_EQ(sum, nPeers) << nPeers << " peers, " << pushPeers << " a fill";
    }
  }
}

// The staged phase 1 walks each row's slices: over every row, each slice exactly once, in the row phase 2 polls it in.
TEST_F(DdaAllGatherFabricLL128Microtest, StageRows_CoverEverySliceInItsPollingRow) {
  using dda::common::ddaLL128AGRowSlice;
  using dda::common::ddaLL128AGRowSlices;
  for (const uint32_t nwarps : {4u, 16u}) {
    for (const uint32_t rows : {1u, 2u, 3u, 8u, 85u}) {
      for (const uint32_t slicesTotal : {1u, 5u, 37u, 301u, 2000u}) {
        SCOPED_TRACE(testing::Message() << slicesTotal << " slices, " << rows << " rows, " << nwarps << " warps");
        std::vector<uint32_t> seen;
        for (uint32_t row = 0; row < rows; ++row) {
          for (uint32_t k = 0; k < ddaLL128AGRowSlices(slicesTotal, row, rows, nwarps); ++k) {
            const uint32_t s = ddaLL128AGRowSlice(k, row, rows, nwarps);
            EXPECT_EQ(s / nwarps % rows, row) << "slice " << s;
            seen.push_back(s);
          }
        }
        std::sort(seen.begin(), seen.end());
        ASSERT_EQ(seen.size(), slicesTotal);
        for (uint32_t s = 0; s < slicesTotal; ++s) {
          EXPECT_EQ(seen[s], s);
        }
      }
    }
  }
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
  comm().WarpSize = 32;
  params_["RCCL_DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES"] = 1;
  EXPECT_EQ(Run(4), ncclUnhandledCudaError) << "staged launch";
}

}  // namespace
