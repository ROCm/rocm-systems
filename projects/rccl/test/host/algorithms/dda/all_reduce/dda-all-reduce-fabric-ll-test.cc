/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu: the LL and LL128 one-shot
// and two-shot eligibility, their slot sizing from the all_reduce_dda_ll*.h headers, tier selection, the grids,
// and the launches.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"

#include DDA_ALL_REDUCE_FABRIC_LL_CC_PATH

#undef hipExtLaunchKernelGGL

namespace {

using RcclUnitTesting::DdaFabricMockComm;

void* const kSendbuff = reinterpret_cast<void*>(0x1000);
void* const kRecvbuff = reinterpret_cast<void*>(0x2000);
const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x88);
const hipEvent_t kStopEvent = reinterpret_cast<hipEvent_t>(0x77);
// An LL128 slice carries 1920 payload bytes in 2 KiB on the wire.
constexpr size_t kSliceData = 1920;
constexpr size_t kSliceWire = 2048;

template <typename T, int N>
const void* LLOneShot() {
  return KernelPtr(&dda::common::ddaAllReduceFlatLL<T, N>);
}
template <typename T, int N>
const void* LLTwoShot() {
  return KernelPtr(&dda::common::ddaAllReduceTwoShotLL<T, N>);
}
template <typename T, int N>
const void* LL128OneShot() {
  return KernelPtr(&dda::common::ddaAllReduceFlatLL128<T, N>);
}
template <typename T, int N>
const void* LL128TwoShot() {
  return KernelPtr(&dda::common::ddaAllReduceTwoShotLL128<T, N>);
}

using EligibleFn = bool (*)(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t);

class DdaAllReduceFabricLLMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hipGetLastError = [] { return hipSuccess; };
    g_loadParam = [this](const char* env, int64_t deftVal) {
      auto it = params_.find(env);
      return it == params_.end() ? deftVal : it->second;
    };
    comm().rank = 5;
    comm().addonStopEvent = kStopEvent;
  }

  void TearDown() override {
    ResetDdaCollSeams();
    ResetHipFakes();
    ResetNcclFakes();
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(EligibleFn fn, size_t count, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum,
                const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff) {
    return fn(mock_.get(), sendbuff, recvbuff, count, datatype, op);
  }

  ncclResult_t RunLL(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllReduceDdaFabricLL(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream);
  }

  ncclResult_t RunLL128(size_t count, ncclDataType_t datatype = ncclFloat32) {
    return ncclAllReduceDdaFabricLL128(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream);
  }

  const DdaLaunch& Only() {
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  const DdaLaunch& RunOneLL(size_t count, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(RunLL(count, datatype), ncclSuccess);
    return Only();
  }

  const DdaLaunch& RunOneLL128(size_t count, ncclDataType_t datatype = ncclFloat32) {
    g_ddaLaunches.clear();
    EXPECT_EQ(RunLL128(count, datatype), ncclSuccess);
    return Only();
  }

  // One-shot thresholds low enough that 1 KiB messages and up run two-shot.
  void OnlyTwoShot() {
    params_["RCCL_DDA_LL_ONESHOT_THRESHOLD"] = 512;
    params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = 512;
  }

  // Common to all four tiers: everything the shape checks look at but the size.
  void ExpectShapeRejections(EligibleFn fn) {
    EXPECT_TRUE(Eligible(fn, 64));
    EXPECT_FALSE(Eligible(fn, 0));
    EXPECT_FALSE(fn(nullptr, kSendbuff, kRecvbuff, 64, ncclFloat32, ncclSum));
    for (ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg}) {
      EXPECT_FALSE(Eligible(fn, 64, ncclFloat32, op)) << "op " << op;
    }
    EXPECT_TRUE(Eligible(fn, 128, ncclFloat16));
    EXPECT_TRUE(Eligible(fn, 128, ncclBfloat16));
    for (ncclDataType_t datatype : {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclFloat64}) {
      EXPECT_FALSE(Eligible(fn, 256, datatype)) << "datatype " << datatype;
    }
    EXPECT_TRUE(Eligible(fn, 256, ncclFloat8e4m3));
    EXPECT_TRUE(Eligible(fn, 256, ncclFloat8e5m2));
    g_rcclDdaDatatypeSupported = [](const ncclComm*, ncclDataType_t datatype) { return datatype == ncclFloat8e5m2; };
    EXPECT_FALSE(Eligible(fn, 128, ncclFloat16));
    EXPECT_FALSE(Eligible(fn, 256, ncclFloat8e4m3));
    EXPECT_TRUE(Eligible(fn, 256, ncclFloat8e5m2));
    ResetDdaCollSeams();
    for (int nRanks : {1, 145}) {
      comm().nRanks = nRanks;
      EXPECT_FALSE(Eligible(fn, 64)) << "nRanks " << nRanks;
    }
    comm().nRanks = 2;
    EXPECT_TRUE(Eligible(fn, 64));
    comm().nRanks = 144;
    EXPECT_TRUE(Eligible(fn, 144 * 4));
    comm().nRanks = 8;
    comm().bootstrap = nullptr;
    EXPECT_FALSE(Eligible(fn, 64));
    comm().bootstrap = &mock_.bootstrapPlaceholder;
    comm().ddaFabricBarrierState = nullptr;
    EXPECT_TRUE(Eligible(fn, 64));
    for (int missing = 0; missing < 3; ++missing) {
      mock_.setFabricResourcesPresent(true);
      comm().ddaFabricMemHandler = missing == 0 ? nullptr : comm().ddaFabricMemHandler;
      comm().ddaScratch = missing == 1 ? nullptr : comm().ddaScratch;
      comm().ddaPeerPtrsDev = missing == 2 ? nullptr : comm().ddaPeerPtrsDev;
      EXPECT_FALSE(Eligible(fn, 64)) << "missing resource " << missing;
    }
    mock_.setFabricResourcesPresent(true);
  }

  DdaFabricMockComm mock_;
  std::map<std::string, int64_t> params_;
};

TEST_F(DdaAllReduceFabricLLMicrotest, LLOneShotEligible_SharedShapeChecks) {
  ExpectShapeRejections(ddaLLArOneShotEligible);
  comm().ddaLLEpochDev = nullptr;
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 64));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLTwoShotEligible_SharedShapeChecks) {
  ExpectShapeRejections(ddaLLArTwoShotEligible);
  comm().ddaLLEpochDev = nullptr;
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 64));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128OneShotEligible_SharedShapeChecksAndEpoch) {
  ExpectShapeRejections(ddaLL128ArOneShotEligible);
  comm().ddaLLEpochLen = 0;
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 64));
  comm().ddaLLEpochLen = 1;
  comm().ddaLLEpochDev = nullptr;
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 64));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128TwoShotEligible_SharedShapeChecksAndEpoch) {
  ExpectShapeRejections(ddaLL128ArTwoShotEligible);
  comm().ddaLLEpochLen = 0;
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 64));
  comm().ddaLLEpochLen = 1;
  comm().ddaLLEpochDev = nullptr;
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 64));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLOneShotEligible_UpToOneShotThresholdOnWholeMessage) {
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, (1u << 20) / 4));
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, (1u << 20) / 4 + 4));
  params_["RCCL_DDA_LL_ONESHOT_THRESHOLD"] = 1024;
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, 260));
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 512, ncclFloat16));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLOneShotEligible_UpToOneShotThresholdOfThisComm) {
  const ncclComm* seen = nullptr;
  g_rcclDdaLLOneShotThreshold = [&](const ncclComm* c) {
    seen = c;
    return size_t{4096};
  };
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 1024));
  EXPECT_EQ(seen, mock_.get());
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, 1028));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLOneShotEligible_WholeMessageIn16ByteUnitsWithinTheSlot) {
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, 6));
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 12));
  // 8 ranks, a 4 KiB bank: 32 packets, 256 B, per slot.
  comm().ddaScratchBytes = 8192;
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 64));
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, 68));
  // 31 packets per rank floor to a 16-packet slot.
  comm().ddaScratchBytes = 2 * 8 * 31 * 16;
  EXPECT_TRUE(Eligible(ddaLLArOneShotEligible, 32));
  EXPECT_FALSE(Eligible(ddaLLArOneShotEligible, 36));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLTwoShotEligible_UpToTwoShotThresholdOfThisComm) {
  const ncclComm* seen = nullptr;
  size_t threshold = 2u << 20;
  g_rcclDdaLLTwoShotThreshold = [&](const ncclComm* c) {
    seen = c;
    return threshold;
  };
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, (2u << 20) / 4));
  EXPECT_EQ(seen, mock_.get());
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, (2u << 20) / 4 + 32));
  threshold = 1024;
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, 288));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LLTwoShotEligible_EachShardIn16ByteUnitsWithinHalfASlot) {
  comm().nRanks = 3;
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, 4));
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 12));
  comm().nRanks = 8;
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, 16));
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 32));
  // 8 ranks, a 4 KiB bank holding two staging areas: 16 packets, 128 B, per shard.
  comm().ddaScratchBytes = 8192;
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, 288));
  // 31 packets per shard floor to 16.
  comm().ddaScratchBytes = 2 * 2 * 8 * 31 * 16;
  EXPECT_TRUE(Eligible(ddaLLArTwoShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLLArTwoShotEligible, 288));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128OneShotEligible_UpToOneShotThresholdOnWholeMessage) {
  EXPECT_TRUE(Eligible(ddaLL128ArOneShotEligible, (4u << 20) / 4));
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, (4u << 20) / 4 + 4));
  params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = 1024;
  EXPECT_TRUE(Eligible(ddaLL128ArOneShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 260));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128OneShotEligible_AlignedWholeMessageWithinTheSlotSlices) {
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 6));
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 64, ncclFloat32, ncclSum, reinterpret_cast<void*>(0x1008)));
  EXPECT_FALSE(
      Eligible(ddaLL128ArOneShotEligible, 64, ncclFloat32, ncclSum, kSendbuff, reinterpret_cast<void*>(0x2001)));
  // 8 ranks, a bank of two 2 KiB slices per rank.
  comm().ddaScratchBytes = 2 * 8 * 2 * kSliceWire;
  EXPECT_TRUE(Eligible(ddaLL128ArOneShotEligible, 2 * kSliceData / 4));
  EXPECT_FALSE(Eligible(ddaLL128ArOneShotEligible, 2 * kSliceData / 4 + 4));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128TwoShotEligible_UpToTwoShotThresholdOfThisComm) {
  const ncclComm* seen = nullptr;
  size_t threshold = 64u << 20;
  g_rcclDdaLL128TwoShotThreshold = [&](const ncclComm* c) {
    seen = c;
    return threshold;
  };
  EXPECT_TRUE(Eligible(ddaLL128ArTwoShotEligible, (64u << 20) / 4));
  EXPECT_EQ(seen, mock_.get());
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, (64u << 20) / 4 + 32));
  threshold = 1024;
  EXPECT_TRUE(Eligible(ddaLL128ArTwoShotEligible, 256));
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 288));
}

TEST_F(DdaAllReduceFabricLLMicrotest, LL128TwoShotEligible_AlignedShardsWithinHalfTheSlotSlices) {
  comm().nRanks = 3;
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 4));
  EXPECT_TRUE(Eligible(ddaLL128ArTwoShotEligible, 12));
  comm().nRanks = 8;
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 16));
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 64, ncclFloat32, ncclSum, reinterpret_cast<void*>(0x1004)));
  EXPECT_FALSE(
      Eligible(ddaLL128ArTwoShotEligible, 64, ncclFloat32, ncclSum, kSendbuff, reinterpret_cast<void*>(0x2002)));
  // 8 ranks, a bank holding two staging areas of two 2 KiB slices per rank.
  comm().ddaScratchBytes = 2 * 2 * 8 * 2 * kSliceWire;
  EXPECT_TRUE(Eligible(ddaLL128ArTwoShotEligible, 8 * 2 * kSliceData / 4));
  EXPECT_FALSE(Eligible(ddaLL128ArTwoShotEligible, 8 * (2 * kSliceData + 16) / 4));
}

TEST_F(DdaAllReduceFabricLLMicrotest, TierEligible_EitherShotQualifies) {
  EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, 64, ncclFloat32, ncclSum));
  EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, 64, ncclFloat32, ncclSum));
  OnlyTwoShot();
  EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, 256, ncclFloat32, ncclSum));
  EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, 256, ncclFloat32, ncclSum));
  // 260 floats do not split into 16-byte shards across 8 ranks.
  EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, 260, ncclFloat32, ncclSum));
  EXPECT_FALSE(ncclAllReduceDdaFabricLL128Eligible(mock_.get(), kSendbuff, kRecvbuff, 260, ncclFloat32, ncclSum));
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_OneShotFirstThenTwoShotElseInternalError) {
  EXPECT_EQ(RunOneLL(64).kernel, (LLOneShot<float, 8>()));
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL(256).kernel, (LLTwoShot<float, 8>()));
  g_ddaLaunches.clear();
  EXPECT_EQ(RunLL(260), ncclInternalError);
  EXPECT_EQ(RunLL(256, ncclInt32), ncclInternalError);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

// The tier gate and the launcher read one one-shot cap: past it the two-shot
// runs, or no LL tier claims the message.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_PastThisCommsOneShotCapRunsTwoShot) {
  comm().nRanks = 16;
  g_rcclDdaLLOneShotThreshold = [](const ncclComm*) { return size_t{4096}; };
  EXPECT_EQ(RunOneLL(1024).kernel, (LLOneShot<float, 0>()));
  EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, 1088, ncclFloat32, ncclSum));
  EXPECT_EQ(RunOneLL(1088).kernel, (LLTwoShot<float, 0>()));
  // 16-byte aligned, but its shards are not.
  EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mock_.get(), kSendbuff, kRecvbuff, 1028, ncclFloat32, ncclSum));
  g_ddaLaunches.clear();
  EXPECT_EQ(RunLL(1028), ncclInternalError);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_OneShotFirstThenTwoShotElseInternalError) {
  EXPECT_EQ(RunOneLL128(64).kernel, (LL128OneShot<float, 8>()));
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL128(256).kernel, (LL128TwoShot<float, 8>()));
  g_ddaLaunches.clear();
  EXPECT_EQ(RunLL128(260), ncclInternalError);
  EXPECT_EQ(RunLL128(256, ncclInt32), ncclInternalError);
  EXPECT_TRUE(g_ddaLaunches.empty());
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_EachDatatypeLaunchesItsOwnKernelForBothShots) {
  EXPECT_EQ(RunOneLL(64, ncclFloat16).kernel, (LLOneShot<half, 8>()));
  EXPECT_EQ(RunOneLL(64, ncclBfloat16).kernel, (LLOneShot<bf16, 8>()));
  EXPECT_EQ(RunOneLL(256, ncclFloat8e4m3).kernel, (LLOneShot<dda::common::fp8e4m3, 8>()));
  EXPECT_EQ(RunOneLL(256, ncclFloat8e5m2).kernel, (LLOneShot<dda::common::fp8e5m2, 8>()));
  EXPECT_EQ(RunOneLL(64, ncclFloat16).Arg<const half*>(2), kSendbuff);
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL(512, ncclFloat16).kernel, (LLTwoShot<half, 8>()));
  EXPECT_EQ(RunOneLL(512, ncclBfloat16).kernel, (LLTwoShot<bf16, 8>()));
  EXPECT_EQ(RunOneLL(2048, ncclFloat8e4m3).kernel, (LLTwoShot<dda::common::fp8e4m3, 8>()));
  EXPECT_EQ(RunOneLL(2048, ncclFloat8e5m2).kernel, (LLTwoShot<dda::common::fp8e5m2, 8>()));
  EXPECT_EQ(RunOneLL(512, ncclBfloat16).Arg<bf16*>(1), kRecvbuff);
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_EachDatatypeLaunchesItsOwnKernelForBothShots) {
  EXPECT_EQ(RunOneLL128(64, ncclFloat16).kernel, (LL128OneShot<half, 8>()));
  EXPECT_EQ(RunOneLL128(64, ncclBfloat16).kernel, (LL128OneShot<bf16, 8>()));
  EXPECT_EQ(RunOneLL128(256, ncclFloat8e4m3).kernel, (LL128OneShot<dda::common::fp8e4m3, 8>()));
  EXPECT_EQ(RunOneLL128(256, ncclFloat8e5m2).kernel, (LL128OneShot<dda::common::fp8e5m2, 8>()));
  EXPECT_EQ(RunOneLL128(64, ncclFloat16).Arg<const half*>(2), kSendbuff);
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL128(512, ncclFloat16).kernel, (LL128TwoShot<half, 8>()));
  EXPECT_EQ(RunOneLL128(512, ncclBfloat16).kernel, (LL128TwoShot<bf16, 8>()));
  EXPECT_EQ(RunOneLL128(2048, ncclFloat8e4m3).kernel, (LL128TwoShot<dda::common::fp8e4m3, 8>()));
  EXPECT_EQ(RunOneLL128(2048, ncclFloat8e5m2).kernel, (LL128TwoShot<dda::common::fp8e5m2, 8>()));
  EXPECT_EQ(RunOneLL128(512, ncclBfloat16).Arg<bf16*>(1), kRecvbuff);
}

TEST_F(DdaAllReduceFabricLLMicrotest, Run_FourAndEightRanksUnrolledOthersRuntimeInEveryTier) {
  struct Case {
    int nRanks;
    const void* kernels[4];
  };
  const Case cases[] = {
      {4, {LLOneShot<float, 4>(), LLTwoShot<float, 4>(), LL128OneShot<float, 4>(), LL128TwoShot<float, 4>()}},
      {8, {LLOneShot<float, 8>(), LLTwoShot<float, 8>(), LL128OneShot<float, 8>(), LL128TwoShot<float, 8>()}},
      {2, {LLOneShot<float, 0>(), LLTwoShot<float, 0>(), LL128OneShot<float, 0>(), LL128TwoShot<float, 0>()}},
      {16, {LLOneShot<float, 0>(), LLTwoShot<float, 0>(), LL128OneShot<float, 0>(), LL128TwoShot<float, 0>()}}};
  for (const Case& c : cases) {
    params_.clear();
    comm().nRanks = c.nRanks;
    EXPECT_EQ(RunOneLL(64).kernel, c.kernels[0]) << "nRanks " << c.nRanks;
    EXPECT_EQ(RunOneLL128(64).kernel, c.kernels[2]) << "nRanks " << c.nRanks;
    OnlyTwoShot();
    EXPECT_EQ(RunOneLL(1024).kernel, c.kernels[1]) << "nRanks " << c.nRanks;
    EXPECT_EQ(Only().Arg<int>(5), c.nRanks);
    EXPECT_EQ(RunOneLL128(1024).kernel, c.kernels[3]) << "nRanks " << c.nRanks;
    EXPECT_EQ(Only().Arg<int>(5), c.nRanks);
  }
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_OneShotPassesWholeMessageEpochAndBank) {
  comm().ddaScratchBytes = (64u << 20) + 40;
  const DdaLaunch& launch = RunOneLL(64);
  EXPECT_EQ(launch.block.x, 512u);
  EXPECT_EQ(launch.stream, kStream);
  EXPECT_EQ(launch.startEvent, nullptr);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  EXPECT_EQ(comm().addonStopEvent, nullptr);
  EXPECT_EQ(launch.sharedMemBytes, 0u);
  EXPECT_EQ(launch.flags, 0u);
  ASSERT_EQ(launch.args.size(), 10u);
  EXPECT_EQ(launch.Arg<float**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<float*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const float*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 64u);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 32u << 20);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(9).flag, comm().abortFlagDev);
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_TwoShotPassesWholeCountEpochAndBank) {
  OnlyTwoShot();
  comm().ddaScratchBytes = (64u << 20) + 40;
  const DdaLaunch& launch = RunOneLL(256);
  EXPECT_EQ(launch.block.x, 1024u);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  ASSERT_EQ(launch.args.size(), 10u);
  EXPECT_EQ(launch.Arg<float**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<float*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const float*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 256u);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 32u << 20);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(9).flag, comm().abortFlagDev);
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_OneShotPassesBytesSlicesEpochAndBank) {
  comm().ddaScratchBytes = (64u << 20) + 40;
  const DdaLaunch& launch = RunOneLL128(1000, ncclBfloat16);
  EXPECT_EQ(launch.block.x, 512u);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  ASSERT_EQ(launch.args.size(), 11u);
  EXPECT_EQ(launch.Arg<bf16**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<bf16*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const bf16*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 2000u);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 2u);
  EXPECT_EQ(launch.Arg<size_t>(9), 32u << 20);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_TwoShotPassesShardBytesAndSlices) {
  OnlyTwoShot();
  comm().ddaScratchBytes = (64u << 20) + 40;
  const DdaLaunch& launch = RunOneLL128(8 * 1000, ncclBfloat16);
  EXPECT_EQ(launch.block.x, 512u);
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  ASSERT_EQ(launch.args.size(), 11u);
  EXPECT_EQ(launch.Arg<bf16**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<bf16*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const bf16*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 2000u);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), 2u);
  EXPECT_EQ(launch.Arg<size_t>(9), 32u << 20);
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

// One-shot LL: a block per 512 packets of the message, capped at ddaFabricMaxBlocks.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_OneShotGridFollowsMessagePackets) {
  EXPECT_EQ(RunOneLL(1024).grid.x, 1u);
  EXPECT_EQ(RunOneLL(1028).grid.x, 2u);
  EXPECT_EQ(RunOneLL(1024, ncclFloat16).grid.x, 1u);
  EXPECT_EQ(RunOneLL(1u << 16).grid.x, 64u);
  EXPECT_EQ(ncclAllReduceDdaFabricLLBlocks(mock_.get(), 1u << 16, ncclFloat32), 64u);
  comm().ddaFabricMaxBlocks = 5;
  EXPECT_EQ(RunOneLL(1u << 16).grid.x, 5u);
  EXPECT_EQ(ncclAllReduceDdaFabricLLBlocks(mock_.get(), 1u << 16, ncclFloat32), 5u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOneLL(1u << 16).grid.x, 1u);
  EXPECT_EQ(ncclAllReduceDdaFabricLLBlocks(mock_.get(), 1u << 16, ncclFloat32), 1u);
}

// Two-shot LL: a block per 1024 packets of one shard.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_TwoShotGridFollowsShardPackets) {
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL(8 * 2048).grid.x, 1u);
  EXPECT_EQ(RunOneLL(8 * 2052).grid.x, 2u);
  EXPECT_EQ(RunOneLL(8 * 6144).grid.x, 3u);
  comm().ddaFabricMaxBlocks = 2;
  EXPECT_EQ(RunOneLL(8 * 6144).grid.x, 2u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOneLL(8 * 6144).grid.x, 1u);
}

// One-shot LL128: a warp per slice, 16 warps a block.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_OneShotGridFollowsMessageSlices) {
  EXPECT_EQ(RunOneLL128(16 * kSliceData / 4).grid.x, 1u);
  EXPECT_EQ(RunOneLL128(17 * kSliceData / 4).grid.x, 2u);
  EXPECT_EQ(RunOneLL128(17 * kSliceData / 4).Arg<size_t>(8), 17u);
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 4, ncclFloat32), 2u);
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 4, ncclFloat16), 1u);
  comm().ddaFabricMaxBlocks = 1;
  EXPECT_EQ(RunOneLL128(17 * kSliceData / 4).grid.x, 1u);
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 4, ncclFloat32), 1u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOneLL128(17 * kSliceData / 4).grid.x, 1u);
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), 17 * kSliceData / 4, ncclFloat32), 1u);
}

// Two-shot LL128: sized on one shard's slices.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_TwoShotGridFollowsShardSlices) {
  OnlyTwoShot();
  EXPECT_EQ(RunOneLL128(8 * 16 * kSliceData / 4).grid.x, 1u);
  EXPECT_EQ(RunOneLL128(8 * 17 * kSliceData / 4).grid.x, 2u);
  EXPECT_EQ(Only().Arg<size_t>(8), 17u);
  comm().ddaFabricMaxBlocks = 1;
  EXPECT_EQ(RunOneLL128(8 * 17 * kSliceData / 4).grid.x, 1u);
  comm().ddaFabricMaxBlocks = 0;
  EXPECT_EQ(RunOneLL128(8 * 17 * kSliceData / 4).grid.x, 1u);
}

TEST_F(DdaAllReduceFabricLLMicrotest, Blocks_EmptyMessageStillReportsOneBlock) {
  EXPECT_EQ(ncclAllReduceDdaFabricLLBlocks(mock_.get(), 0, ncclFloat32), 1u);
  EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), 0, ncclFloat32), 1u);
}

TEST_F(DdaAllReduceFabricLLMicrotest, Run_LaunchErrorIsReportedFromEveryTier) {
  g_hipGetLastError = [] { return hipErrorLaunchFailure; };
  EXPECT_EQ(RunLL(64), ncclUnhandledCudaError);
  EXPECT_EQ(RunLL128(64), ncclUnhandledCudaError);
  OnlyTwoShot();
  EXPECT_EQ(RunLL(256), ncclUnhandledCudaError);
  EXPECT_EQ(RunLL128(256), ncclUnhandledCudaError);
  EXPECT_EQ(g_ddaLaunches.size(), 4u);
}

}  // namespace
