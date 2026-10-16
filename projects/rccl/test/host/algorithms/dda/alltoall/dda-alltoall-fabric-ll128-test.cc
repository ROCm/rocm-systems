/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu: eligibility against the fixed
// LL128 slot stride, the grid-shape overrides and epoch clamp, and the launch.

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
// Two banks of nRanks slots of ceil(512 KiB / 120 B) = 4370 128-byte lines.
constexpr size_t kScratchPerRank = 2 * 4370 * 128;
// A 128-byte line carries 120 bytes, 30 floats.
constexpr size_t kLineFloats = 30;

template <int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllToAllFabricLL128<int8_t, N>);
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
  ScopedEnv lpb_{"RCCL_DDA_LL128_A2A_LPB", nullptr};
  ScopedEnv maxBpp_{"RCCL_DDA_LL128_A2A_MAXBPP", nullptr};
};

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_BaselineFp32) {
  EXPECT_TRUE(Eligible(4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_DdaLLZeroRejects) {
  params_["RCCL_DDA_LL"] = 0;
  EXPECT_FALSE(Eligible(4));
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

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_PerPeerBytesMultipleOf16UpTo512KiB) {
  EXPECT_FALSE(Eligible(2));
  EXPECT_FALSE(Eligible(12, ncclFloat16));
  EXPECT_TRUE(Eligible((512u << 10) / 4));
  EXPECT_FALSE(Eligible((512u << 10) / 4 + 4));
}

TEST_F(DdaAlltoallFabricLL128Microtest, Eligible_ScratchMustHoldBothBanksOfFixedSlots) {
  comm().ddaScratchBytes = 4 * kScratchPerRank;
  EXPECT_TRUE(Eligible(4));
  comm().ddaScratchBytes = 4 * kScratchPerRank - 1;
  EXPECT_FALSE(Eligible(4));
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
    EXPECT_EQ(launch.grid.x, static_cast<unsigned>(nRanks));
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_PassesBuffersPerPeerBytesAndEpoch) {
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
  ASSERT_EQ(launch.args.size(), 8u);
  EXPECT_EQ(launch.Arg<int8_t**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<int8_t*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const int8_t*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 16u);
  EXPECT_EQ(launch.Arg<int>(4), 2);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
}

// One block per peer up to 4 lines per peer, then one per 4 lines, at most 8 per peer.
TEST_F(DdaAlltoallFabricLL128Microtest, Launch_BlocksPerPeerFollowLinesCappedAtEight) {
  const std::pair<size_t, unsigned> cases[] = {
      {4, 1}, {4 * kLineFloats, 1}, {4 * kLineFloats + 2, 2}, {32 * kLineFloats, 8}, {33 * kLineFloats, 8}};
  for (const auto& [count, bpp] : cases) {
    const DdaLaunch& launch = RunOne(count);
    EXPECT_EQ(launch.grid.x, 4u);
    EXPECT_EQ(launch.grid.y, bpp) << "count " << count;
    EXPECT_EQ(launch.block.x, 1024u);
    EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32), 4 * bpp);
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_GridStaysWithinTheEpochCells) {
  comm().ddaLLEpochLen = 13;
  EXPECT_EQ(RunOne(32 * kLineFloats).grid.y, 3u);
  comm().ddaLLEpochLen = 3;
  EXPECT_EQ(RunOne(32 * kLineFloats).grid.y, 1u);
  comm().ddaLLEpochLen = 32;
  EXPECT_EQ(RunOne(32 * kLineFloats).grid.y, 8u);
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_LinesPerBlockAndMaxBlocksPerPeerFromEnv) {
  {
    ScopedEnv lpb("RCCL_DDA_LL128_A2A_LPB", "2");
    EXPECT_EQ(RunOne(5 * kLineFloats).grid.y, 3u);
    EXPECT_EQ(RunOne(2 * kLineFloats).grid.y, 1u);
  }
  {
    ScopedEnv maxBpp("RCCL_DDA_LL128_A2A_MAXBPP", "3");
    EXPECT_EQ(RunOne(32 * kLineFloats).grid.y, 3u);
    ScopedEnv one("RCCL_DDA_LL128_A2A_MAXBPP", "1");
    EXPECT_EQ(RunOne(32 * kLineFloats).grid.y, 1u);
  }
  for (const char* bad : {"", "0", "-2", "x"}) {
    ScopedEnv lpb("RCCL_DDA_LL128_A2A_LPB", bad);
    ScopedEnv maxBpp("RCCL_DDA_LL128_A2A_MAXBPP", bad);
    EXPECT_EQ(RunOne(5 * kLineFloats).grid.y, 2u) << "env '" << bad << "'";
    EXPECT_EQ(RunOne(64 * kLineFloats).grid.y, 8u) << "env '" << bad << "'";
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_ThreadsOverrideMustBeALineMultipleUpTo1024) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {16, 16},  {1024, 1024}, {8, 1024},
                                                {24, 1024}, {1040, 1024}, {0, 1024}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_A2A_THREADS"] = env;
    EXPECT_EQ(RunOne(4).block.x, threads) << "DDA_LL128_A2A_THREADS " << env;
  }
}

TEST_F(DdaAlltoallFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), 8 * kLineFloats, ncclFloat16), 4u);
  EXPECT_EQ(ncclAllToAllDdaFabricLL128Blocks(mock_.get(), 8 * kLineFloats, ncclFloat32), 8u);
}

TEST_F(DdaAlltoallFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
