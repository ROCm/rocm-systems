/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu: eligibility against the
// slot carved from one scratch bank, the epoch cells and the buffer alignment, the warp-per-slice grid and its
// caps, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "../common/DdaFabricTestHelpers.hpp"
#include "../common/ScopedEnv.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_ALLTOALL_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;
using RcclUnitTesting::ScopedEnv;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A 2 KiB wire slice carries 1920 bytes, 480 floats; one warp moves one slice.
constexpr size_t kSliceFloats = 480;
// 256 threads = 8 warps per block.
constexpr size_t kWarpsPerBlock = 8;

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricLL128V2<int8_t, N, 1>);
}

class DdaAlltoallFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
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

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllToAllDdaFabricLL128(kSendbuff, kRecvbuff, count, datatype, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t count) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(count), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
  ScopedEnv maxBpp_{"RCCL_DDA_LL128_A2A_MAXBPP", nullptr};
};

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

// LL128 is gated by RCCL_DDA_LL128 alone; RCCL_DDA_LL only controls the LL tier.
TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_DdaLLZeroDoesNotGateLL128) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllToAllDdaFabricLL128Eligible(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
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

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_OnlyDatatypesTheDdaGateAdmits) {
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

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_PerPeerBytesMultipleOf16) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible(16, ncclBfloat16));
}

// Each bank holds nRanks slots of whole 2 KiB slices; a slot of S slices carries S * 1920 bytes.
TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_PerPeerBytesUpToTheSlotCapacity) {
  constexpr size_t kSlices = 10;
  comm().ddaScratchBytes = 2 * 4 * kSlices * 2048;
  EXPECT_TRUE(Eligible(kSlices * kSliceFloats));
  EXPECT_FALSE(Eligible(kSlices * kSliceFloats + 4));
  comm().ddaScratchBytes = 2 * 4 * kSlices * 2048 - 1;
  EXPECT_FALSE(Eligible(kSlices * kSliceFloats));
  EXPECT_TRUE(Eligible((kSlices - 1) * kSliceFloats));
}

// The grid has nRanks - 1 columns and at least one block per column, each owning one epoch cell.
TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_NeedsAnEpochCellPerPeer) {
  comm().ddaLLEpochLen = 3;
  EXPECT_TRUE(Eligible(4));
  comm().ddaLLEpochLen = 2;
  EXPECT_FALSE(Eligible(4));
  comm().ddaLLEpochLen = 256;
  comm().ddaLLEpochDev = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_Needs16ByteAlignedBuffers) {
  char* const base = reinterpret_cast<char*>(0x10000);
  EXPECT_TRUE(ncclAllToAllDdaFabricLL128Eligible(mock_.get(), base, base + 16, 4, ncclFloat32));
  EXPECT_FALSE(ncclAllToAllDdaFabricLL128Eligible(mock_.get(), base + 8, base + 16, 4, ncclFloat32));
  EXPECT_FALSE(ncclAllToAllDdaFabricLL128Eligible(mock_.get(), base, base + 4, 4, ncclFloat32));
  EXPECT_FALSE(ncclAllToAllDdaFabricLL128Eligible(mock_.get(), base + 4, base + 4, 4, ncclFloat32));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
  const std::pair<int, const void*> cases[] = {
      {4, Kernel<4>()}, {8, Kernel<8>()}, {2, Kernel<0>()}, {3, Kernel<0>()}, {144, Kernel<0>()}};
  for (const auto& [nRanks, kernel] : cases) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, kernel) << "nRanks " << nRanks;
    EXPECT_EQ(launch.grid.x, static_cast<unsigned>(nRanks - 1)) << "no self column";
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_PassesBuffersSlicesBankAndAbortFlag) {
  ASSERT_EQ(Run(8, ncclBfloat16), ncclSuccess);
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
  EXPECT_EQ(launch.Arg<size_t>(3), 16u);
  EXPECT_EQ(launch.Arg<int>(4), 2);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 1u);
  EXPECT_EQ(launch.Arg<size_t>(9), dda::common::ddaBankSize(comm().ddaScratchBytes));
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

// One warp per slice: blocks per peer = ceil(slices / 8), capped by the CU budget per peer.
TEST_F(DdaAlltoallFabricLL128Microtest, Launch_BlocksPerPeerFollowSlicesPerWarp) {
  const size_t perBlock = kWarpsPerBlock * kSliceFloats;
  const unsigned cap = 256 / 3;
  const std::pair<size_t, unsigned> cases[] = {
      {4, 1}, {perBlock, 1}, {perBlock + 4, 2}, {cap * perBlock, cap}, {(cap + 20) * perBlock, cap}};
  for (const auto& [count, bpp] : cases) {
    const DdaLaunch& launch = RunOne(count);
    EXPECT_EQ(launch.grid.x, 3u);
    EXPECT_EQ(launch.grid.y, bpp) << "count " << count;
    EXPECT_EQ(launch.block.x, 256u);
    EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32), 3 * bpp);
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_GridStaysWithinTheEpochCellsAndCuBudget) {
  const size_t big = 1000 * kWarpsPerBlock * kSliceFloats;
  comm().ddaLLEpochLen = 13;
  EXPECT_EQ(RunOne(big).grid.y, 4u);
  comm().ddaLLEpochLen = 3;
  EXPECT_EQ(RunOne(big).grid.y, 1u);
  comm().ddaLLEpochLen = 256;
  comm().ddaFabricMaxBlocks = 30;
  EXPECT_EQ(RunOne(big).grid.y, 10u);
  comm().ddaFabricMaxBlocks = 2;
  EXPECT_EQ(RunOne(big).grid.y, 1u);
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_MaxBlocksPerPeerFromEnv) {
  const size_t big = 64 * kWarpsPerBlock * kSliceFloats;
  {
    ScopedEnv maxBpp("RCCL_DDA_LL128_A2A_MAXBPP", "3");
    EXPECT_EQ(RunOne(big).grid.y, 3u);
    ScopedEnv one("RCCL_DDA_LL128_A2A_MAXBPP", "1");
    EXPECT_EQ(RunOne(big).grid.y, 1u);
  }
  {
    ScopedEnv aboveCap("RCCL_DDA_LL128_A2A_MAXBPP", "1000");
    EXPECT_EQ(RunOne(big).grid.y, 64u);
  }
  for (const char* bad : {"", "0", "-2", "x"}) {
    ScopedEnv maxBpp("RCCL_DDA_LL128_A2A_MAXBPP", bad);
    EXPECT_EQ(RunOne(big).grid.y, 64u) << "env '" << bad << "'";
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_ThreadsOverrideMustBeAWarpMultipleUpTo1024) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {32, 32},  {1024, 1024}, {16, 256},
                                                {48, 256},  {1056, 256}, {0, 256}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_A2A_THREADS"] = env;
    EXPECT_EQ(RunOne(4).block.x, threads) << "DDA_LL128_A2A_THREADS " << env;
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  const size_t count = 2 * kWarpsPerBlock * kSliceFloats;
  EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), count, ncclFloat16), 3u);
  EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32), 6u);
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
