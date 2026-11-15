/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu: eligibility against
// the fixed per-shard cap and its scratch layout, the datatype dispatch, the line grid and its caps, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

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
// A 128B line carries 15 payload words, 30 floats.
constexpr size_t kLineFloats = 30;
// 1024 threads are 64 lines of 16 lanes, so one block covers 64 lines.
constexpr size_t kBlockFloats = 64 * kLineFloats;
constexpr size_t kMaxShardFloats = dda::common::kDdaLL128RsMaxBytes / sizeof(float);
// Two banks of nRanks fixed-stride slots.
constexpr size_t ScratchFor(int nRanks) {
  return 2 * (size_t)nRanks * dda::common::kDdaLL128RsSlotStrideLines * sizeof(dda::common::LLLine128);
}

template <typename T, int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaReduceScatterFabricLL128<T, N>);
}

class DdaReduceScatterFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    comm().rank = 7;
    comm().addonStopEvent = kStopEvent;
    comm().ddaFabricMaxBlocks = 256;
    comm().ddaLLEpochLen = 256;
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

  ncclResult_t Run(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    return ncclReduceScatterDdaFabricLL128(kSendbuff, kRecvbuff, recvcount, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t recvcount, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(recvcount, datatype), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t recvcount) {
    EXPECT_EQ(launch.kernel, (Kernel<T, 8>()));
    ASSERT_EQ(launch.args.size(), 8u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<const T*>(2), kSendbuff);
    EXPECT_EQ(launch.Arg<size_t>(3), recvcount);
    EXPECT_EQ(launch.Arg<int>(4), 7);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
    EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
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

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ShardBytesMultipleOf8) {
  EXPECT_TRUE(Eligible(2));
  EXPECT_TRUE(Eligible(4, ncclFloat16));
  EXPECT_FALSE(Eligible(1));
  EXPECT_FALSE(Eligible(3));
  EXPECT_FALSE(Eligible(6, ncclBfloat16));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ShardWithinTheFixedCap) {
  EXPECT_TRUE(Eligible(kMaxShardFloats));
  EXPECT_FALSE(Eligible(kMaxShardFloats + 2));
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Eligible_ScratchMustHoldBothBanksOfSlots) {
  comm().ddaScratchBytes = ScratchFor(8);
  EXPECT_TRUE(Eligible(4));
  comm().ddaScratchBytes = ScratchFor(8) - 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 4;
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
  EXPECT_EQ(launch.block.x, 1024u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_ThreadsOverrideNeedsAMultipleOf16UpTo1024) {
  params_["RCCL_DDA_LL128_RS_THREADS"] = 256;
  EXPECT_EQ(RunOne(4).block.x, 256u);
  params_["RCCL_DDA_LL128_RS_THREADS"] = 200;
  EXPECT_EQ(RunOne(4).block.x, 1024u);
  params_["RCCL_DDA_LL128_RS_THREADS"] = 2048;
  EXPECT_EQ(RunOne(4).block.x, 1024u);
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

// A block per 64 lines of the shard, within ddaFabricMaxBlocks and the epoch cells, and at least one.
TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_GridFollowsShardLinesWithinCaps) {
  EXPECT_EQ(RunOne(4, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOne(kBlockFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(kBlockFloats + 2).grid.x, 2u);
  EXPECT_EQ(RunOne(100 * kBlockFloats).grid.x, 100u);
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(RunOne(100 * kBlockFloats).grid.x, 6u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(100 * kBlockFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(100 * kBlockFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(100 * kBlockFloats).grid.x, 1u);
}

TEST_F(DdaReduceScatterFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
