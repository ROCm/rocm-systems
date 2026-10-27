/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu: eligibility against the
// LL128 threshold and slot, the DDA_LL128_AG_THREADS block size, the peer-column grid, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

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

const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllGatherFabricLL128<int8_t>);
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

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_OneKernelForEveryRankCount) {
  for (int nRanks : {2, 4, 6, 8, 144}) {
    comm().nRanks = nRanks;
    const DdaLaunch& launch = RunOne(4);
    EXPECT_EQ(launch.kernel, Kernel()) << "nRanks " << nRanks;
    EXPECT_EQ(launch.grid.x, static_cast<unsigned>(nRanks - 1));
    EXPECT_EQ(launch.Arg<int>(5), nRanks);
  }
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_PassesBuffersBytesEpochSlicesAndBank) {
  comm().ddaScratchBytes = 4096 + 40;
  ASSERT_EQ(Run(1000, ncclBfloat16), ncclSuccess);
  ASSERT_EQ(g_ddaLaunches.size(), 1u);
  const DdaLaunch& launch = g_ddaLaunches[0];
  EXPECT_EQ(launch.kernel, Kernel());
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
  EXPECT_EQ(launch.Arg<size_t>(9), 2064u);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

// A warp per slice: 512 threads give 16 warps, so a column grows past one block at 17 slices.
// Above 2 MiB in total the grid budget drops to RCCL_DDA_LL128_AG_MAXBLOCKS (75), 25 per peer here.
TEST_F(DdaAllGatherFabricLL128Microtest, Launch_BlocksPerPeerFollowSlicesCappedByShareOfMaxBlocks) {
  const std::pair<size_t, unsigned> cases[] = {
      {4, 1}, {16 * kSliceData / 4, 1}, {17 * kSliceData / 4, 2}, {100 * 16 * kSliceData / 4, 25}};
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

TEST_F(DdaAllGatherFabricLL128Microtest, Blocks_SizedOnBytesNotElements) {
  EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 2, ncclFloat16), 6u);
  EXPECT_EQ(ncclAllGatherDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 2, ncclFloat32), 9u);
}

TEST_F(DdaAllGatherFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
