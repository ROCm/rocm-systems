/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu: eligibility against
// the scratch bank's per-rank slot and the LL128 cap, the datatype dispatch, the peer-parallel and batched split, each
// kernel's slice grid and its caps, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_REDUCE_SCATTER_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A slice is 16 lines on the wire and carries 1920 bytes, 480 floats.
constexpr size_t kSliceFloats = 480;
// Scratch holding exactly one slice per rank at 8 ranks: two banks of 8 x 2 KiB.
constexpr size_t kOneSliceScratch8 = 2 * 8 * 2048;

template <typename T, int N, bool kAligned = true>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabricLL128<T, N, kAligned>);
}

template <typename T, int N, bool kAligned = true>
const void* PeerParKernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabricLL128PeerPar<T, N, kAligned>);
}

class DdaReduceScatterFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    // The LL128 tier is off without an arch table; give it a cap well above anything tested.
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{1} << 30; };
    comm().rank = 7;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t recvcount, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, recvcount, datatype, op);
  }

  ncclResult_t Run(size_t recvcount, ncclDataType_t datatype = ncclFloat32, const void* sendbuff = kSendbuff,
                   void* recvbuff = kRecvbuff) {
    return ncclReduceScatterDdaFabricLL128(sendbuff, recvbuff, recvcount, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t recvcount, ncclDataType_t datatype = ncclFloat32, const void* sendbuff = kSendbuff,
                          void* recvbuff = kRecvbuff) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(recvcount, datatype, sendbuff, recvbuff), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  // Messages up to `bytes` take the peer-parallel kernel, as an arch table cap would send them.
  void PeerParallelUpTo(size_t bytes) {
    g_rcclDdaFabricRsLL128PeerMaxBytes = [bytes](const ncclComm*) { return bytes; };
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount, const void* kernel = Kernel<T, 8>()) {
    EXPECT_EQ(launch.kernel, kernel);
    ASSERT_EQ(launch.args.size(), 11u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<const T*>(2), kSendbuff);
    EXPECT_EQ(launch.Arg<size_t>(3), recvcount * sizeof(T));
    EXPECT_EQ(launch.Arg<int>(4), 7);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
    EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
    EXPECT_EQ(launch.Arg<size_t>(8), (recvcount * sizeof(T) + 1919) / 1920);
    EXPECT_EQ(launch.Arg<size_t>(9), (comm().ddaScratchBytes / 2) / 128 * 128);
    EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}

// LL128 is gated by RCCL_DDA_LL128 alone; RCCL_DDA_LL only controls the LL tier.
TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_DdaLLZeroDoesNotGateLL128) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
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

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_NeedsTheEpochArray) {
  comm().ddaLLEpochLen = 0;
  EXPECT_FALSE(Eligible(4));
  comm().ddaLLEpochLen = 1;
  EXPECT_TRUE(Eligible(4));
  comm().ddaLLEpochDev = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
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

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ShardBytesMultipleOf16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(6));
  EXPECT_FALSE(Eligible(12, ncclBfloat16));
}

// Buffer alignment differs between ranks and every rank must pick the same tier, so it never decides eligibility.
TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_WhateverTheBufferAlignment) {
  void* const misaligned = reinterpret_cast<void*>(0x1008);
  EXPECT_TRUE(ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), misaligned, kRecvbuff, 4, ncclFloat32, ncclSum));
  EXPECT_TRUE(ncclReduceScatterDdaFabricLL128Eligible(mock_.get(), kSendbuff, misaligned, 4, ncclFloat32, ncclSum));
}

// The cap is checked against the whole message, nRanks shards, as the selector does.
TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_WholeMessageWithinTheLL128Cap) {
  size_t cap = 8 * 16;
  g_rcclDdaLL128Threshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    EXPECT_EQ(func, ncclFuncReduceScatter);
    return cap;
  };
  EXPECT_TRUE(Eligible(4));
  cap = 8 * 16 - 1;
  EXPECT_FALSE(Eligible(4));
  cap = 0;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_RankSlotMustHoldEverySlice) {
  comm().ddaScratchBytes = kOneSliceScratch8;
  EXPECT_TRUE(Eligible(kSliceFloats));
  EXPECT_FALSE(Eligible(kSliceFloats + 4));
  comm().ddaScratchBytes = kOneSliceScratch8 - 1;
  EXPECT_FALSE(Eligible(4));
  comm().ddaScratchBytes = 2 * kOneSliceScratch8;
  EXPECT_TRUE(Eligible(kSliceFloats + 4));
  comm().nRanks = 4;
  comm().ddaScratchBytes = kOneSliceScratch8 / 2;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_EachDatatypeLaunchesItsOwnKernel) {
  ExpectLaunchOf<float>(RunOne(16, ncclFloat32), 16);
  ExpectLaunchOf<half>(RunOne(16, ncclFloat16), 16);
  ExpectLaunchOf<bf16>(RunOne(16, ncclBfloat16), 16);
  ExpectLaunchOf<dda::common::fp8e4m3>(RunOne(16, ncclFloat8e4m3), 16);
  ExpectLaunchOf<dda::common::fp8e5m2>(RunOne(16, ncclFloat8e5m2), 16);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_TakesTheStopEventOnTheStream) {
  const DdaLaunch& launch = RunOne(4);
  EXPECT_EQ(launch.block.x, 512u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {{4, Kernel<float, 4>()},
                                               {8, Kernel<float, 8>()},
                                               {2, Kernel<float, 0>()},
                                               {5, Kernel<float, 0>()},
                                               {144, Kernel<float, 0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

// 512 threads are 16 warps, one slice each: a block per 16 slices of the shard.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_GridFollowsShardSlicesWithinCaps) {
  EXPECT_EQ(RunOne(8, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * kSliceFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * kSliceFloats + 4).grid.x, 2u);
  EXPECT_EQ(RunOne(32 * kSliceFloats, ncclBfloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 100u);
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 6u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(16 * 100 * kSliceFloats).grid.x, 1u);
}

// The reported block count is the launch grid of either kernel, caps included.
TEST_F(DdaReduceScatterFabricLL128Microtest, Blocks_MatchTheLaunchedGrid) {
  struct Case {
    size_t count;
    ncclDataType_t datatype;
  };
  const Case cases[] = {{8, ncclFloat16},
                        {16 * kSliceFloats + 4, ncclFloat32},
                        {32 * kSliceFloats, ncclBfloat16},
                        {16 * 100 * kSliceFloats, ncclFloat32}};
  for (size_t peerMax : {size_t{0}, size_t{1} << 30}) {
    PeerParallelUpTo(peerMax);
    for (int maxBlocks : {256, 6}) {
      comm().ddaFabricMaxBlocks = maxBlocks;
      for (const Case& c : cases) {
        EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), c.count, c.datatype),
                  RunOne(c.count, c.datatype).grid.x)
          << "count " << c.count << " maxBlocks " << maxBlocks << " peerMax " << peerMax;
      }
    }
  }
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsOverrideMustBeAWarpMultipleUpTo512) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {32, 32},   {512, 512}, {16, 512},
                                                {48, 512},  {544, 512}, {0, 512},   {-32, 512}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_RS_THREADS"] = env;
    EXPECT_EQ(RunOne(8).block.x, threads) << "DDA_LL128_RS_THREADS " << env;
  }
  // 256 threads are 8 warps, so 9 slices take a second block.
  params_["RCCL_DDA_LL128_RS_THREADS"] = 256;
  EXPECT_EQ(RunOne(9 * kSliceFloats).grid.x, 2u);
  EXPECT_EQ(RunOne(8 * kSliceFloats).grid.x, 1u);
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 9 * kSliceFloats, ncclFloat32), 2u);
}

// The arch table picks the block size, 512 when it sets none or one the launch bound rules out.
// A valid DDA_LL128_RS_THREADS overrides it.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsFromArchTableUnlessOverridden) {
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 256; };
  // 256 threads are 8 warps, so 9 slices take a second block.
  EXPECT_EQ(RunOne(9 * kSliceFloats).block.x, 256u);
  EXPECT_EQ(RunOne(9 * kSliceFloats).grid.x, 2u);
  EXPECT_EQ(ncclReduceScatterDdaFabricLL128Blocks(mock_.get(), 9 * kSliceFloats, ncclFloat32), 2u);
  params_["RCCL_DDA_LL128_RS_THREADS"] = 512;
  EXPECT_EQ(RunOne(9 * kSliceFloats).block.x, 512u);
  params_.erase("RCCL_DDA_LL128_RS_THREADS");
  for (int tuned : {0, 48, 1024}) {
    g_rcclDdaFabricRsLL128Threads = [tuned](const ncclComm*) { return tuned; };
    EXPECT_EQ(RunOne(8).block.x, 512u) << "table " << tuned;
  }
}

// The table's cap is the whole message: up to it the peer-parallel kernel runs, past it the batched one.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_UpToTheTableCapBatchedAbove) {
  EXPECT_EQ(RunOne(kSliceFloats).kernel, (Kernel<float, 8>()));
  PeerParallelUpTo(8 * kSliceFloats * sizeof(float));
  EXPECT_EQ(RunOne(kSliceFloats).kernel, (PeerParKernel<float, 8>()));
  EXPECT_EQ(RunOne(kSliceFloats + 4).kernel, (Kernel<float, 8>()));
  EXPECT_EQ(RunOne(2 * kSliceFloats, ncclFloat16).kernel, (PeerParKernel<half, 8>()));
  EXPECT_EQ(RunOne(2 * kSliceFloats + 8, ncclFloat16).kernel, (Kernel<half, 8>()));
}

// Both kernels come in an aligned and an any-address instance; my own buffers pick which.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_AlignedKernelFrom16ByteOwnBuffersOnly) {
  void* const send8 = reinterpret_cast<void*>(0x1008);
  void* const recv4 = reinterpret_cast<void*>(0x2004);
  void* const send16 = reinterpret_cast<void*>(0x1010);
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, send16, kRecvbuff).kernel, (Kernel<float, 8, true>()));
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, send8, kRecvbuff).kernel, (Kernel<float, 8, false>()));
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, kSendbuff, recv4).kernel, (Kernel<float, 8, false>()));
  comm().nRanks = 6;
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, send8, kRecvbuff).kernel, (Kernel<float, 0, false>()));
  comm().nRanks = 8;
  PeerParallelUpTo(8 * kSliceFloats * sizeof(float));
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, send16, kRecvbuff).kernel, (PeerParKernel<float, 8, true>()));
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, send8, kRecvbuff).kernel, (PeerParKernel<float, 8, false>()));
  EXPECT_EQ(RunOne(kSliceFloats, ncclFloat32, kSendbuff, recv4).kernel, (PeerParKernel<float, 8, false>()));
}

// DDA_LL128_RS_PEER_MAX_BYTES overrides the table's cap. -1, the default, reads the table, and 0 or another negative
// keeps the batched kernel.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_EnvOverridesTheTable) {
  PeerParallelUpTo(size_t{1} << 30);
  for (int64_t env : {int64_t{0}, int64_t{-5}}) {
    params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = env;
    EXPECT_EQ(RunOne(4).kernel, (Kernel<float, 8>())) << "DDA_LL128_RS_PEER_MAX_BYTES " << env;
  }
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = 8 * 16;
  EXPECT_EQ(RunOne(4).kernel, (PeerParKernel<float, 8>()));
  EXPECT_EQ(RunOne(8).kernel, (Kernel<float, 8>()));
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = -1;
  EXPECT_EQ(RunOne(8).kernel, (PeerParKernel<float, 8>()));
  PeerParallelUpTo(0);
  EXPECT_EQ(RunOne(8).kernel, (Kernel<float, 8>()));
  params_["RCCL_DDA_LL128_RS_PEER_MAX_BYTES"] = 8 * 32;
  EXPECT_EQ(RunOne(8).kernel, (PeerParKernel<float, 8>()));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_EachDatatypeLaunchesItsOwnKernel) {
  PeerParallelUpTo(size_t{1} << 30);
  ExpectLaunchOf<float>(RunOne(16, ncclFloat32), 16, PeerParKernel<float, 8>());
  ExpectLaunchOf<half>(RunOne(16, ncclFloat16), 16, PeerParKernel<half, 8>());
  ExpectLaunchOf<bf16>(RunOne(16, ncclBfloat16), 16, PeerParKernel<bf16, 8>());
  ExpectLaunchOf<dda::common::fp8e4m3>(RunOne(16, ncclFloat8e4m3), 16, PeerParKernel<dda::common::fp8e4m3, 8>());
  ExpectLaunchOf<dda::common::fp8e5m2>(RunOne(16, ncclFloat8e5m2), 16, PeerParKernel<dda::common::fp8e5m2, 8>());
}

TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_FourAndEightRanksUnrolledOthersRuntime) {
  PeerParallelUpTo(size_t{1} << 30);
  const std::pair<int, const void*> cases[] = {{4, PeerParKernel<float, 4>()},
                                               {8, PeerParKernel<float, 8>()},
                                               {2, PeerParKernel<float, 0>()},
                                               {5, PeerParKernel<float, 0>()},
                                               {144, PeerParKernel<float, 0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

// A block per slice within the block cap, capped at ddaFabricMaxBlocks and at the epoch cells.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_GridFollowsShardSlicesWithinCaps) {
  PeerParallelUpTo(size_t{1} << 30);
  EXPECT_EQ(RunOne(8, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(kSliceFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(kSliceFloats + 4).grid.x, 2u);
  const DdaLaunch& uncapped = RunOne(100 * kSliceFloats);
  EXPECT_EQ(uncapped.grid.x, 100u);
  EXPECT_EQ(uncapped.block.x, 256u);
  comm().ddaFabricMaxBlocks = 6;
  const DdaLaunch& capped = RunOne(100 * kSliceFloats);
  EXPECT_EQ(capped.grid.x, 6u);
  EXPECT_EQ(capped.block.x, 512u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(100 * kSliceFloats).grid.x, 1u);
}

// A warp group is a warp per rank, up to the 16 warps of 512 threads.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockIsOneWarpPerRank) {
  PeerParallelUpTo(size_t{1} << 30);
  const std::pair<int, unsigned> cases[] = {{2, 64u},   {3, 96u},   {4, 128u},  {8, 256u},
                                            {12, 384u}, {16, 512u}, {32, 512u}, {144, 512u}};
  for (const auto& [nRanks, threads] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(kSliceFloats);
    EXPECT_EQ(launch.block.x, threads) << "nRanks " << nRanks;
    EXPECT_EQ(launch.grid.x, 1u) << "nRanks " << nRanks;
  }
}

// Past the block cap a block doubles its groups, up to 512 threads: 40 slices at 4 ranks.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockTakesMoreGroupsOnlyPastTheCap) {
  PeerParallelUpTo(size_t{1} << 30);
  comm().nRanks = 4;
  const std::pair<int, std::pair<unsigned, unsigned>> cases[] = {{40, {40u, 128u}}, {20, {20u, 256u}}, {6, {6u, 512u}}};
  for (const auto& [maxBlocks, want] : cases) {
    comm().ddaFabricMaxBlocks = maxBlocks;
    const DdaLaunch& launch = RunOne(40 * kSliceFloats);
    EXPECT_EQ(launch.grid.x, want.first) << "maxBlocks " << maxBlocks;
    EXPECT_EQ(launch.block.x, want.second) << "maxBlocks " << maxBlocks;
  }
}

// DDA_LL128_RS_THREADS and the table's block size apply to the batched kernel only.
TEST_F(DdaReduceScatterFabricLL128Microtest, PeerParallel_BlockIgnoresTheBatchedThreadSettings) {
  params_["RCCL_DDA_LL128_RS_THREADS"] = 32;
  g_rcclDdaFabricRsLL128Threads = [](const ncclComm*) { return 128; };
  EXPECT_EQ(RunOne(4).block.x, 32u);
  PeerParallelUpTo(size_t{1} << 30);
  EXPECT_EQ(RunOne(4).block.x, 256u);
  params_.erase("RCCL_DDA_LL128_RS_THREADS");
  EXPECT_EQ(RunOne(4).block.x, 256u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

// Walks phase 2's poll batches as the kernel does; returns the valid peers in order and counts invalid entries.
template <int kBatch>
std::vector<int> WalkPollBatches(int selfRank, int nRanks, int& invalid, int& invalidNotSelf) {
  std::vector<int> visited;
  int peer = dda::common::ddaLL128RsNextRank(selfRank, nRanks);
  for (int base = 1; base < nRanks; base += kBatch) {
    int peers[kBatch];
    bool valid[kBatch];
    peer = dda::common::ddaLL128RsBatchPeers(peer, base, nRanks, peers, valid);
    for (int k = 0; k < kBatch; ++k) {
      if (valid[k]) {
        visited.push_back(peers[k]);
      } else {
        ++invalid;
        invalidNotSelf += peers[k] != selfRank;
      }
    }
  }
  return visited;
}

template <int kBatch>
void ExpectPollWalkCoversPeersInRankOrder() {
  for (int nRanks = 2; nRanks <= 144; ++nRanks) {
    for (int self = 0; self < nRanks; ++self) {
      std::vector<int> expected;
      for (int r = 1; r < nRanks; ++r) {
        expected.push_back((self + r) % nRanks);
      }
      int invalid = 0;
      int invalidNotSelf = 0;
      ASSERT_EQ(WalkPollBatches<kBatch>(self, nRanks, invalid, invalidNotSelf), expected)
        << "batch " << kBatch << " nRanks " << nRanks << " self " << self;
      ASSERT_EQ(invalid, (kBatch - (nRanks - 1) % kBatch) % kBatch) << "batch " << kBatch << " nRanks " << nRanks;
      ASSERT_EQ(invalidNotSelf, 0) << "batch " << kBatch << " nRanks " << nRanks << " self " << self;
    }
  }
}

// Every peer is polled once in rank order after self; a past-the-end entry names self, whose own slot is never written.
TEST_F(DdaReduceScatterFabricLL128Microtest, PollBatch_WalkVisitsEveryPeerOnceInRankOrder) {
  static_assert(dda::common::kDdaLL128RsPollBatch<float> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<half> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<bf16> == 8);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3> == 4);
  static_assert(dda::common::kDdaLL128RsPollBatch<dda::common::fp8e5m2> == 4);
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<float>>();
  ExpectPollWalkCoversPeersInRankOrder<dda::common::kDdaLL128RsPollBatch<dda::common::fp8e4m3>>();
  // The 4- and 8-rank instances cap the batch at their peer count.
  ExpectPollWalkCoversPeersInRankOrder<3>();
  ExpectPollWalkCoversPeersInRankOrder<7>();
}

}  // namespace
