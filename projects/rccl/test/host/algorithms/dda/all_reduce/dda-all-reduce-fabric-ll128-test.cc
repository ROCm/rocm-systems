/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu, whose *_old entry points
// nothing calls any more: eligibility against the LL128 threshold and the per-call compact scratch, the
// DDA_LL128_AR_THREADS block size, the line grid and its caps, and the launch.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include "fakes/param_redirect.h"

#include DDA_ALL_REDUCE_FABRIC_LL128_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// A 128-byte line carries 120 bytes, 30 floats.
constexpr size_t kLineFloats = 30;

template <typename T, int N>
const void* Kernel() {
  return KernelPtr(&dda::common::ddaAllReduceFlatLL128<T, N>);
}

class DdaAllReduceFabricLL128Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    // The LL128 tier is off without an arch table; give it a cap well above anything tested.
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return size_t{1} << 30; };
    comm().rank = 6;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(size_t count, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum) {
    return ncclAllReduceDdaFabricLL128Eligible_old(mock_.get(), kSendbuff, kRecvbuff, count, datatype, op);
  }

  ncclResult_t Run(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllReduceDdaFabricLL128_old(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& RunOne(size_t count, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(Run(count, datatype), ncclSuccess);
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  template <typename T>
  void ExpectLaunchOf(const DdaLaunch& launch, size_t count, size_t lines) {
    EXPECT_EQ(launch.kernel, (Kernel<T, 8>()));
    ASSERT_EQ(launch.args.size(), 10u);
    EXPECT_EQ(launch.Arg<T**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<T*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<const T*>(2), kSendbuff);
    EXPECT_EQ(launch.Arg<size_t>(3), count);
    EXPECT_EQ(launch.Arg<int>(4), 6);
    EXPECT_EQ(launch.Arg<int>(5), 8);
    EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
    EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
    EXPECT_EQ(launch.Arg<size_t>(8), lines);
    EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(9).flag, comm().abortFlagDev);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_BaselineFp32Sum) {
  EXPECT_TRUE(Eligible(4));
}


TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_NeedsCommAndBootstrap) {
  EXPECT_FALSE(ncclAllReduceDdaFabricLL128Eligible_old(nullptr, kSendbuff, kRecvbuff, 4, ncclFloat32, ncclSum));
  comm().bootstrap = nullptr;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_NeedsHandlerScratchAndPeersButNotBarrier) {
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

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_ZeroCountRejects) {
  EXPECT_FALSE(Eligible(0));
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_RanksFromTwoTo144) {
  comm().nRanks = 1;
  EXPECT_FALSE(Eligible(4));
  comm().nRanks = 2;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 144;
  EXPECT_TRUE(Eligible(4));
  comm().nRanks = 145;
  EXPECT_FALSE(Eligible(4));
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_OnlySum) {
  for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
    EXPECT_FALSE(Eligible(4, ncclFloat32, op)) << "op " << op;
  }
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_OnlyFloat32Float16Bfloat16) {
  EXPECT_TRUE(Eligible(8, ncclFloat16));
  EXPECT_TRUE(Eligible(8, ncclBfloat16));
  for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
    EXPECT_FALSE(Eligible(16, datatype)) << "datatype " << datatype;
  }
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_MessageBytesMultipleOf8) {
  EXPECT_TRUE(Eligible(2));
  EXPECT_TRUE(Eligible(4, ncclFloat16));
  EXPECT_FALSE(Eligible(1));
  EXPECT_FALSE(Eligible(6, ncclBfloat16));
}

TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_UpToAllReduceLL128ThresholdAndZeroDisables) {
  ncclFunc_t seenFunc = ncclFuncSendRecv;
  size_t threshold = 1024;
  g_rcclDdaLL128Threshold = [&](const ncclComm* c, ncclFunc_t func) {
    EXPECT_EQ(c, mock_.get());
    seenFunc = func;
    return threshold;
  };
  EXPECT_TRUE(Eligible(256));
  EXPECT_EQ(seenFunc, ncclFuncAllReduce);
  EXPECT_FALSE(Eligible(258));
  threshold = 0;
  EXPECT_FALSE(Eligible(2));
}

// Two banks of nRanks slots sized to this message's 128-byte lines.
TEST_F(DdaAllReduceFabricLL128Microtest, Eligible_ScratchMustHoldThisMessagesLines) {
  comm().ddaScratchBytes = 2 * 8 * 128;
  EXPECT_TRUE(Eligible(kLineFloats));
  EXPECT_FALSE(Eligible(kLineFloats + 2));
  comm().ddaScratchBytes = 2 * 8 * 128 - 1;
  EXPECT_FALSE(Eligible(kLineFloats));
  comm().nRanks = 4;
  comm().ddaScratchBytes = 2 * 4 * 3 * 128;
  EXPECT_TRUE(Eligible(3 * kLineFloats));
  EXPECT_FALSE(Eligible(3 * kLineFloats + 2));
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_UnsupportedDatatypeReturnsInvalidArgument) {
  for (ncclDataType_t datatype : {ncclInt8, ncclInt32, ncclUint64, ncclFloat64}) {
    EXPECT_EQ(Run(16, datatype), ncclInvalidArgument) << "datatype " << datatype;
  }
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_EachDatatypeLaunchesItsOwnKernelWithItsLines) {
  ExpectLaunchOf<float>(RunOne(kLineFloats + 2, ncclFloat32), kLineFloats + 2, 2);
  ExpectLaunchOf<half>(RunOne(kLineFloats * 2, ncclFloat16), kLineFloats * 2, 1);
  ExpectLaunchOf<bf16>(RunOne(kLineFloats * 2 + 4, ncclBfloat16), kLineFloats * 2 + 4, 2);
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_TakesTheStopEventOnTheStream) {
  const DdaLaunch& launch = RunOne(4);
  EXPECT_EQ(launch.block.x, 1024u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_FourAndEightRanksUnrolledOthersRuntime) {
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

// 1024 threads are 64 sixteen-lane groups, one line each: a block per 64 lines of the message.
TEST_F(DdaAllReduceFabricLL128Microtest, Launch_GridFollowsMessageLinesWithinCaps) {
  const std::pair<size_t, unsigned> cases[] = {{1, 1}, {64 * kLineFloats, 1}, {64 * kLineFloats + 2, 2},
                                               {64 * 100 * kLineFloats, 100}};
  for (const auto& [count, blocks] : cases) {
    EXPECT_EQ(RunOne(count).grid.x, blocks) << "count " << count;
    EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks_old(mock_.get(), count, ncclFloat32), blocks);
  }
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks_old(mock_.get(), 64 * kLineFloats, ncclFloat16), 1u);
  comm().ddaFabricMaxBlocks = 6;
  EXPECT_EQ(RunOne(64 * 100 * kLineFloats).grid.x, 6u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOne(64 * 100 * kLineFloats).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaLLEpochLen = 9;
  EXPECT_EQ(RunOne(64 * 100 * kLineFloats).grid.x, 9u);
  comm().ddaLLEpochLen = 0;
  EXPECT_EQ(RunOne(64 * 100 * kLineFloats).grid.x, 1u);
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_ThreadsOverrideMustBeALineMultipleUpTo1024) {
  const std::pair<int64_t, unsigned> cases[] = {{256, 256}, {16, 16},  {1024, 1024}, {8, 1024},
                                                {24, 1024}, {1040, 1024}, {0, 1024}};
  for (const auto& [env, threads] : cases) {
    params_["RCCL_DDA_LL128_AR_THREADS"] = env;
    EXPECT_EQ(RunOne(4).block.x, threads) << "DDA_LL128_AR_THREADS " << env;
  }
  // 256 threads are 16 groups, so 17 lines take a second block.
  params_["RCCL_DDA_LL128_AR_THREADS"] = 256;
  EXPECT_EQ(RunOne(16 * kLineFloats).grid.x, 1u);
  EXPECT_EQ(RunOne(17 * kLineFloats).grid.x, 2u);
}

TEST_F(DdaAllReduceFabricLL128Microtest, Launch_LaunchErrorIsReported) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(Run(4), ncclUnhandledCudaError);
}

}  // namespace
