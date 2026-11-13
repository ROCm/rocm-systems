/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu: the LL and LL128 one-shot
// and two-shot eligibility, their slot sizing from the all_reduce_dda_ll*.h headers, tier selection, the grids,
// and the launches.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../common/DdaFabricTestHelpers.hpp"
#include "algorithms/dda/DdaCollSeams.h"
#include "algorithms/dda/dda_init_detail.h"

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
const void* LLSpread() {
  return KernelPtr(&dda::common::ddaAllReduceSpreadTwoShotLL<T, N>);
}
// The LL one-shot or spread instance of T the launcher picks for nRanks.
template <typename T>
const void* LLTier(bool spread, int nRanks) {
  if (nRanks == 4) {
    return spread ? LLSpread<T, 4>() : LLOneShot<T, 4>();
  }
  if (nRanks == 8) {
    return spread ? LLSpread<T, 8>() : LLOneShot<T, 8>();
  }
  return spread ? LLSpread<T, 0>() : LLOneShot<T, 0>();
}
// The LL one-shot or spread instance the launcher picks for datatype and nRanks.
const void* LLTypeTier(bool spread, ncclDataType_t datatype, int nRanks) {
  switch (datatype) {
  case ncclFloat16:
    return LLTier<half>(spread, nRanks);
  case ncclBfloat16:
    return LLTier<bf16>(spread, nRanks);
  case ncclFloat8e4m3:
    return LLTier<dda::common::fp8e4m3>(spread, nRanks);
  case ncclFloat8e5m2:
    return LLTier<dda::common::fp8e5m2>(spread, nRanks);
  default:
    return LLTier<float>(spread, nRanks);
  }
}
// Floats in a message of `bytes`.
constexpr size_t Floats(size_t bytes) {
  return bytes / sizeof(float);
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
    ResetLdsDevs<float>();
    ResetLdsDevs<half>();
    ResetLdsDevs<bf16>();
    ResetLdsDevs<dda::common::fp8e4m3>();
    ResetLdsDevs<dda::common::fp8e5m2>();
  }

  // Clears every hand-off kernel's LDS cache of T.
  template <typename T>
  static void ResetLdsDevs() {
    ddaArSplitHandoffLdsDevs<T> = 0;
    ddaArSplitHandoffLdsDevs<T, kDdaArSplitBWide> = 0;
    ddaArSplitHandoffLdsDevs<T, kDdaArSplitBWideChunked> = 0;
  }

  ncclComm& comm() { return mock_.comm; }

  bool Eligible(EligibleFn fn, size_t count, ncclDataType_t datatype = ncclFloat32, ncclRedOp_t op = ncclSum,
                const void* sendbuff = kSendbuff, void* recvbuff = kRecvbuff) {
    return fn(mock_.get(), sendbuff, recvbuff, count, datatype, op);
  }

  ncclResult_t RunLL(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {}) {
    return ncclAllReduceDdaFabricLL(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream, cap);
  }

  ncclResult_t RunLL128(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {}) {
    return ncclAllReduceDdaFabricLL128(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream, cap);
  }

  const DdaLaunch& Only() {
    EXPECT_EQ(g_ddaLaunches.size(), 1u);
    return g_ddaLaunches.at(0);
  }

  const DdaLaunch& RunOneLL(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {}) {
    g_ddaLaunches.clear();
    EXPECT_EQ(RunLL(count, datatype, cap), ncclSuccess);
    return Only();
  }

  const DdaLaunch& RunOneLL128(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {}) {
    g_ddaLaunches.clear();
    EXPECT_EQ(RunLL128(count, datatype, cap), ncclSuccess);
    return Only();
  }

  // One-shot thresholds low enough that 1 KiB messages and up run two-shot.
  void OnlyTwoShot() {
    params_["RCCL_DDA_LL_ONESHOT_THRESHOLD"] = 512;
    params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = 512;
  }

  // Whether `tier` is the one that launches `count` elements: one-shot first, else two-shot.
  template <typename Tier>
  bool TierRuns(const Tier& tier, size_t count, ncclDataType_t datatype) {
    const bool oneShot = Eligible(tier.ll128 ? ddaLL128ArOneShotEligible : ddaLLArOneShotEligible, count, datatype);
    const bool twoShot = Eligible(tier.ll128 ? ddaLL128ArTwoShotEligible : ddaLLArTwoShotEligible, count, datatype);
    return tier.twoShot ? !oneShot && twoShot : oneShot;
  }

  // Launches `count` elements on `tier`'s path (LL or LL128) under `cap`.
  template <typename Tier>
  DdaLaunch RunTier(const Tier& tier, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap = {}) {
    return tier.ll128 ? RunOneLL128(count, datatype, cap) : RunOneLL(count, datatype, cap);
  }

  // Calls fn(maxBlocks, tier, nRanks, type, bytes, count) for every sweep case a tier runs; 20+ cases per tier.
  template <typename Fn>
  void ForEachTierCase(Fn fn);

  // Blocks `tier`'s path reports for `count` elements under `cap`.
  template <typename Tier>
  uint32_t ReportTier(const Tier& tier, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap) {
    return tier.ll128 ? ncclAllReduceDdaFabricLL128Blocks(mock_.get(), count, datatype, cap)
                      : ncclAllReduceDdaFabricLLBlocks(mock_.get(), count, datatype, cap);
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

  // The selector's question: whether the peer-split takes `count` elements, and the bound it runs within under `cap`.
  bool TakesSplit(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {}, bool vmmFirst = false,
                  rcclDdaLaunchCap* bound = nullptr) {
    rcclDdaLaunchCap unused = {};
    return ncclAllReduceDdaFabricLL128SplitEligible(mock_.get(), kSendbuff, kRecvbuff, count, datatype, ncclSum, cap,
                                                    vmmFirst, bound != nullptr ? bound : &unused);
  }

  // TakesSplit for a float message of `bytes`.
  bool TakesSplitBytes(size_t bytes, bool vmmFirst = false) {
    return TakesSplit(bytes / sizeof(float), ncclFloat32, {}, vmmFirst);
  }

  // The LL128 launch of the selector's decision: the peer-split within its bound where it takes the message.
  const DdaLaunch& RunSelectedLL128(size_t count, ncclDataType_t datatype = ncclFloat32, rcclDdaLaunchCap cap = {},
                                    bool vmmFirst = false) {
    rcclDdaLaunchCap bound = {};
    const bool split = TakesSplit(count, datatype, cap, vmmFirst, &bound);
    g_ddaLaunches.clear();
    EXPECT_EQ(ncclAllReduceDdaFabricLL128(kSendbuff, kRecvbuff, count, datatype, ncclSum, mock_.get(), kStream, bound,
                                          split),
              ncclSuccess);
    return Only();
  }

  // The launch the split replaces: the LL, else the LL128, launcher with RCCL_DDA_LL128_ASYNC_WIRE=0 (no window).
  DdaLaunch RunReplaced(bool ll, size_t count, rcclDdaLaunchCap cap = {}) {
    params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 0;
    const DdaLaunch launch = ll ? RunOneLL(count, ncclFloat32, cap) : RunOneLL128(count, ncclFloat32, cap);
    params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
    return launch;
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

// Grid sweep inputs: every LL datatype, odd and even per-rank sizes (16B multiples), and 2 to 16 ranks.
constexpr ncclDataType_t kLLTypes[] = {ncclFloat32, ncclFloat16, ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
constexpr size_t kShardBytes[] = {16, 48, 1040, 17 * kSliceData, 3 * 8192 + 16, 65536 + 112, 262144};
constexpr int kSweepRanks[] = {2, 3, 4, 8, 16};

// The fixed shape of each tier before caps existed: threads per block and threads per unit of work.
struct FixedTier {
  bool ll128;
  bool twoShot;
  unsigned threads;
  unsigned unitThreads;
};
constexpr FixedTier kTiers[] = {
    {false, false, 512, 1}, {false, true, 1024, 1}, {true, false, 512, 32}, {true, true, 512, 32}};

// Units of work a tier covers: LL packets (8B) or LL128 slices, of the message (one-shot) or one shard (two-shot).
size_t TierUnits(const FixedTier& tier, size_t bytes, int nRanks) {
  const size_t span = tier.twoShot ? bytes / (size_t)nRanks : bytes;
  return tier.ll128 ? dda::common::ddaLL128Slices(span) : span >> 3;
}

// Blocks of the fixed shape: ceil(units / units per block), at most ddaFabricMaxBlocks, at least one.
unsigned FixedBlocks(size_t units, unsigned threads, unsigned unitThreads, int maxBlocks) {
  const size_t perBlock = threads / unitThreads;
  size_t blocks = (units + perBlock - 1) / perBlock;
  blocks = std::min(blocks, (size_t)std::max(maxBlocks, 1));
  return (unsigned)std::max<size_t>(blocks, 1);
}

template <typename Fn>
void DdaAllReduceFabricLLMicrotest::ForEachTierCase(Fn fn) {
  for (int maxBlocks : {256, 5}) {
    comm().ddaFabricMaxBlocks = maxBlocks;
    for (const FixedTier& tier : kTiers) {
      params_.clear();
      if (tier.twoShot) {
        OnlyTwoShot();
      }
      int cases = 0;
      for (int nRanks : kSweepRanks) {
        comm().nRanks = nRanks;
        for (ncclDataType_t type : kLLTypes) {
          for (size_t shard : kShardBytes) {
            const size_t bytes = shard * (size_t)nRanks;
            const size_t count = bytes / ncclTypeSize(type);
            if (!TierRuns(tier, count, type)) {
              continue;
            }
            ++cases;
            SCOPED_TRACE(testing::Message() << "ll128=" << tier.ll128 << " twoShot=" << tier.twoShot
                                            << " maxBlocks=" << maxBlocks << " nRanks=" << nRanks << " type=" << type
                                            << " bytes=" << bytes);
            fn(maxBlocks, tier, nRanks, type, bytes, count);
          }
        }
      }
      EXPECT_GE(cases, 20) << "ll128=" << tier.ll128 << " twoShot=" << tier.twoShot;
    }
  }
}

// Without a cap every tier launches its fixed shape, and its reporter reports the grid it launches.
TEST_F(DdaAllReduceFabricLLMicrotest, Run_NoCapKeepsTheFixedShapeInEveryTier) {
  ForEachTierCase([&](int maxBlocks, const FixedTier& tier, int nRanks, ncclDataType_t type, size_t bytes,
                      size_t count) {
    const DdaLaunch launch = RunTier(tier, count, type);
    EXPECT_EQ(launch.block.x, tier.threads);
    EXPECT_EQ(launch.grid.x, FixedBlocks(TierUnits(tier, bytes, nRanks), tier.threads, tier.unitThreads, maxBlocks));
    const DdaLaunch zero = RunTier(tier, count, type, {0, 0});
    EXPECT_EQ(zero.grid.x, launch.grid.x);
    EXPECT_EQ(zero.block.x, launch.block.x);
    EXPECT_EQ(ReportTier(tier, count, type, {0, 0}), launch.grid.x * launch.grid.y);
  });
}

// A capped launch stays within the cap's blocks and its threads (whole 64-thread granules, at least one) in every tier,
// keeps its kernel, and its reporter reports the grid it launches; the next uncapped call is the fixed shape again.
// Caps above and below ddaFabricMaxBlocks both stay within it. A narrowed block runs the LL one-shot for the spread.
TEST_F(DdaAllReduceFabricLLMicrotest, Run_CapBoundsTheGridAndTheReporterInEveryTier) {
  const rcclDdaLaunchCap caps[] = {{1, 64}, {3, 192}, {17, 256},   {3, 0},  {0, 192}, {17, 96}, {2, 1024},
                                   {1, 0},  {0, 64},  {1000, 4096}, {0, 480}, {0, 992}, {1, 32},  {0, 1}};
  ForEachTierCase([&](int maxBlocks, const FixedTier& tier, int nRanks, ncclDataType_t type, size_t bytes,
                      size_t count) {
    const DdaLaunch fixed = RunTier(tier, count, type);
    for (const rcclDdaLaunchCap& cap : caps) {
      SCOPED_TRACE(testing::Message() << "cap={" << cap.blocks << "," << cap.threads << "}");
      const DdaLaunch capped = RunTier(tier, count, type, cap);
      const bool narrowedSpread = fixed.kernel == LLTypeTier(true, type, nRanks) && capped.block.x < tier.threads;
      EXPECT_EQ(capped.kernel, narrowedSpread ? LLTypeTier(false, type, nRanks) : fixed.kernel);
      EXPECT_GE(capped.grid.x, 1u);
      EXPECT_LE(capped.grid.x, (unsigned)maxBlocks);
      if (cap.blocks > 0) {
        EXPECT_LE(capped.grid.x, cap.blocks);
      }
      EXPECT_LE(capped.block.x, tier.threads);
      if (cap.threads >= 64) {
        EXPECT_LE(capped.block.x, cap.threads);
      }
      EXPECT_EQ(capped.block.x % 64, 0u);
      const unsigned threads = capped.block.x;
      EXPECT_EQ(threads, cap.threads > 0 && cap.threads < tier.threads ? std::max(cap.threads / 64 * 64, 64u)
                                                                         : tier.threads);
      const int blockBound = cap.blocks > 0 ? std::min<int>(cap.blocks, maxBlocks) : maxBlocks;
      const size_t units = TierUnits(tier, bytes, nRanks);
      EXPECT_EQ(capped.grid.x, FixedBlocks(units, threads, tier.unitThreads, blockBound))
          << "the capped width sizes the grid";
      EXPECT_EQ(ReportTier(tier, count, type, cap), capped.grid.x * capped.grid.y)
          << "the reporter reports the capped grid";
    }
    const DdaLaunch next = RunTier(tier, count, type);
    EXPECT_EQ(next.grid.x, fixed.grid.x);
    EXPECT_EQ(next.block.x, fixed.block.x);
  });
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

// Spread once the one-shot's block wire, 16 B x min(packets, 512) x (n - 1), reaches 96 KiB or its saved wire 4 MiB.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadWhereOneShotBlockWireOrSavedWireIsLarge) {
  comm().ddaScratchBytes = 1u << 30;
  struct Case {
    int nRanks;
    size_t bytes;
    bool spread;
  };
  const Case cases[] = {
      {16, 2048, false},       // 256 packets x 16 B x 15 = 60 KiB a block
      {16, 4096, true},        // 512 packets: 120 KiB a block
      {32, 1024, false},       // 128 packets x 16 B x 31 = 62 KiB
      {32, 2048, true},        // 124 KiB
      {12, 4096, false},       // 88 KiB, and 2 x 11 x 10 / 12 x 4 KiB of saved wire
      {64, 1024, true},        // 126 KiB
      {8, 256 << 10, false},   // 56 KiB a block, 2.6 MiB saved
      {8, 512 << 10, true},    // 5.25 MiB saved
      {4, 1 << 20, false},     // 3 MiB saved
      {2, 1 << 20, false},     // nothing saved at 2 ranks
  };
  for (const Case& c : cases) {
    comm().nRanks = c.nRanks;
    EXPECT_EQ(RunOneLL(Floats(c.bytes)).kernel, LLTier<float>(c.spread, c.nRanks)) << c.nRanks << " ranks, " << c.bytes;
  }
}

// Both rules at their exact edges and just below: 96 KiB of block wire at 13 ranks, 4 MiB of saved wire.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadRuleEdges) {
  EXPECT_TRUE(ddaLLArSpreadWins(4096, 13));       // 512 packets x 16 B x 12 = 96 KiB
  EXPECT_FALSE(ddaLLArSpreadWins(4080, 13));      // 510 packets: 95.625 KiB
  EXPECT_TRUE(ddaLLArSpreadWins(3 << 20, 3));     // 2 x 2 x 1 / 3 x 3 MiB = 4 MiB
  EXPECT_FALSE(ddaLLArSpreadWins(1398101, 4));    // 2 x 3 x 2 / 4 x bytes = 4 MiB - 1
  comm().nRanks = 13;
  comm().ddaScratchBytes = 1u << 30;
  EXPECT_EQ(RunOneLL(Floats(4096)).kernel, (LLSpread<float, 0>()));
  EXPECT_EQ(RunOneLL(Floats(4080)).kernel, (LLOneShot<float, 0>()));
}

// The spread runs on exactly the one-shot's grid, which the reporter also returns.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadTakesTheOneShotGridAndArguments) {
  comm().nRanks = 32;
  comm().ddaScratchBytes = (256u << 20) + 40;
  for (size_t bytes : {size_t{2048}, size_t{64} << 10, size_t{1} << 20}) {
    comm().addonStopEvent = kStopEvent;
    const DdaLaunch& launch = RunOneLL(Floats(bytes));
    ASSERT_EQ(launch.kernel, (LLSpread<float, 0>())) << bytes;
    EXPECT_EQ(launch.grid.x, std::max<size_t>(1, bytes / 8 / 512)) << bytes;
    EXPECT_EQ(launch.grid.x, ncclAllReduceDdaFabricLLBlocks(mock_.get(), Floats(bytes), ncclFloat32)) << bytes;
    EXPECT_EQ(launch.block.x, 512u);
    EXPECT_EQ(launch.sharedMemBytes, 0u);
    EXPECT_EQ(launch.stopEvent, kStopEvent);
    ASSERT_EQ(launch.args.size(), 10u);
    EXPECT_EQ(launch.Arg<float**>(0), comm().ddaPeerPtrsDev);
    EXPECT_EQ(launch.Arg<float*>(1), kRecvbuff);
    EXPECT_EQ(launch.Arg<const float*>(2), kSendbuff);
    EXPECT_EQ(launch.Arg<size_t>(3), Floats(bytes));
    EXPECT_EQ(launch.Arg<int>(4), 5);
    EXPECT_EQ(launch.Arg<int>(5), 32);
    EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
    EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
    EXPECT_EQ(launch.Arg<size_t>(8), 128u << 20);
    EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(9).flag, comm().abortFlagDev);
  }
  comm().ddaFabricMaxBlocks = 5;
  EXPECT_EQ(RunOneLL(Floats(1 << 20)).grid.x, 5u);
}

// Shards need not be equal: a rank count that does not divide the message gives the last ranks shorter shards.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadTakesUnevenShards) {
  comm().nRanks = 24;
  comm().ddaScratchBytes = 1u << 30;
  EXPECT_EQ(RunOneLL(Floats(8192)).kernel, (LLSpread<float, 0>()));   // 1024 packets: 23 shards of 43, one of 35
  comm().nRanks = 64;
  EXPECT_EQ(RunOneLL(Floats(784)).kernel, (LLSpread<float, 0>()));    // 98 packets: 49 shards of 2, 15 empty
  comm().nRanks = 12;
  EXPECT_EQ(RunOneLL(Floats(512 << 10)).kernel, (LLSpread<float, 0>()));
}

// The shard must fit the two-shot slot: a scratch whose two-shot slot is below a shard keeps the one-shot.
TEST_F(DdaAllReduceFabricLLMicrotest, LLSpreadSelected_ShardMustFitTheTwoShotSlot) {
  comm().nRanks = 32;
  comm().ddaScratchBytes = 2 * 32 * 16 * 2 * 16;  // bank 16 KiB: one-shot slot 32 packets, two-shot slot 16
  ASSERT_EQ(dda::common::ddaLLArSlotPkts(dda::common::ddaBankSize(comm().ddaScratchBytes), 32), 32u);
  ASSERT_EQ(dda::common::ddaLLArTwoShotSlotPkts(dda::common::ddaBankSize(comm().ddaScratchBytes), 32), 16u);
  constexpr unsigned kFull = dda::common::kDdaLLSpreadThreads;
  EXPECT_TRUE(ddaLLArSpreadSelected(mock_.get(), 32 * 16 * 8, kFull)) << "16-packet shards fill the 16-packet slot";
  EXPECT_FALSE(ddaLLArSpreadSelected(mock_.get(), 32 * 16 * 8 + 16, kFull)) << "17-packet shards, rounded up";
  EXPECT_FALSE(ddaLLArSpreadSelected(mock_.get(), 1u << 20, kFull));
  comm().ddaScratchBytes = 1u << 30;
  EXPECT_TRUE(ddaLLArSpreadSelected(mock_.get(), 1u << 20, kFull));
  EXPECT_FALSE(ddaLLArSpreadSelected(mock_.get(), 1u << 20, kFull - 64)) << "a narrowed block keeps the one-shot";
}

// A cap that narrows the blocks keeps the one-shot; one that lowers only the blocks runs the spread on the capped grid.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadNeedsFullWidthBlocksUnderACap) {
  comm().nRanks = 32;
  comm().ddaScratchBytes = 1u << 30;
  const size_t count = Floats(1 << 20);  // 131072 packets: 256 one-shot blocks uncapped
  ASSERT_EQ(RunOneLL(count).kernel, (LLSpread<float, 0>()));
  for (uint32_t threads : {64u, 256u, 448u, 511u}) {
    const DdaLaunch& narrowed = RunOneLL(count, ncclFloat32, {0, threads});
    EXPECT_EQ(narrowed.kernel, (LLOneShot<float, 0>())) << threads;
    EXPECT_LT(narrowed.block.x, 512u) << threads;
  }
  for (const rcclDdaLaunchCap cap : {rcclDdaLaunchCap{3, 0}, rcclDdaLaunchCap{17, 512}, rcclDdaLaunchCap{1, 4096}}) {
    const DdaLaunch& capped = RunOneLL(count, ncclFloat32, cap);
    EXPECT_EQ(capped.kernel, (LLSpread<float, 0>())) << cap.blocks;
    EXPECT_EQ(capped.block.x, 512u) << cap.blocks;
    EXPECT_EQ(capped.grid.x, cap.blocks) << cap.blocks;
    EXPECT_EQ(ncclAllReduceDdaFabricLLBlocks(mock_.get(), count, ncclFloat32, cap), cap.blocks) << cap.blocks;
  }
}

// Every datatype has its own spread kernel.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL_SpreadEachDatatypeLaunchesItsOwnKernel) {
  comm().nRanks = 16;
  comm().ddaScratchBytes = 1u << 30;
  EXPECT_EQ(RunOneLL(4096 / 2, ncclFloat16).kernel, (LLSpread<half, 0>()));
  EXPECT_EQ(RunOneLL(4096 / 2, ncclBfloat16).kernel, (LLSpread<bf16, 0>()));
  EXPECT_EQ(RunOneLL(4096, ncclFloat8e4m3).kernel, (LLSpread<dda::common::fp8e4m3, 0>()));
  EXPECT_EQ(RunOneLL(4096, ncclFloat8e5m2).kernel, (LLSpread<dda::common::fp8e5m2, 0>()));
}

// ---- LDS-staged LL128 wire writer ----

template <typename T, int N, int W>
const void* LL128TwoShotW() {
  return KernelPtr(&dda::common::ddaAllReduceTwoShotLL128<T, N, W>);
}

constexpr int kAsync = dda::common::kDdaLL128WireAsync;
constexpr size_t kMiB = 1024 * 1024;

// RCCL_DDA_LL128_ASYNC_WIRE unset or 1: a gfx1250 wave32 comm takes the staged writer from a 2 MiB shard; 0 keeps coop.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_StagedWriterOnByDefault) {
  comm().WarpSize = 32;
  for (const bool unset : {true, false}) {
    if (!unset) {
      params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
    }
    for (const size_t count : {16 * kMiB / 4, 32 * kMiB / 4}) {  // 8 ranks: 2 and 4 MiB shards, two-shot
      const DdaLaunch& launch = RunOneLL128(count);
      EXPECT_EQ(launch.kernel, (LL128TwoShotW<float, 8, kAsync>())) << "unset " << unset << " count " << count;
      EXPECT_EQ(launch.sharedMemBytes, launch.block.x / 32 * 2 * kSliceWire) << "unset " << unset << " count " << count;
    }
  }
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 0;
  for (const size_t count : {16 * kMiB / 4, 32 * kMiB / 4}) {
    const DdaLaunch& launch = RunOneLL128(count);
    EXPECT_EQ(launch.kernel, (LL128TwoShot<float, 8>())) << "0 turns it off, count " << count;
    EXPECT_EQ(launch.sharedMemBytes, 0u) << "0 turns it off, count " << count;
  }
}

// With RCCL_DDA_LL128_ASYNC_WIRE=1, from a 2 MiB shard a gfx1250 wave32 comm takes the staged writer on the same grid.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_StagedWriterFromTwoMegShardsOnTheSameGrid) {
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
  const size_t big = 16 * kMiB / 4;  // 8 ranks: a 2 MiB shard, two-shot
  const DdaLaunch coop = RunOneLL128(big);
  EXPECT_EQ(coop.kernel, (LL128TwoShot<float, 8>())) << "WarpSize 0: the cooperative writer";
  EXPECT_EQ(coop.sharedMemBytes, 0u);
  comm().WarpSize = 32;
  const DdaLaunch staged = RunOneLL128(big);
  EXPECT_EQ(staged.kernel, (LL128TwoShotW<float, 8, kAsync>()));
  EXPECT_EQ(staged.grid.x, coop.grid.x);
  EXPECT_EQ(staged.block.x, coop.block.x);
  EXPECT_EQ(staged.sharedMemBytes, staged.block.x / 32 * 2 * 2048);
  EXPECT_EQ(staged.Arg<size_t>(3), coop.Arg<size_t>(3)) << "same shard bytes";
  EXPECT_EQ(staged.Arg<size_t>(8), coop.Arg<size_t>(8)) << "same slices";

  const DdaLaunch& below = RunOneLL128(16 * kMiB / 4 - 8 * 4);  // one 16 B chunk short of a 2 MiB shard
  EXPECT_EQ(below.kernel, (LL128TwoShot<float, 8>()));
  EXPECT_EQ(below.sharedMemBytes, 0u);
  EXPECT_EQ(RunOneLL128(8 * kMiB / 4).kernel, (LL128TwoShot<float, 8>())) << "1 MiB shard";

  comm().WarpSize = 64;
  const DdaLaunch& wave64 = RunOneLL128(big);
  EXPECT_EQ(wave64.kernel, (LL128TwoShot<float, 8>())) << "an LL128 warp must be one wave";
  EXPECT_EQ(wave64.sharedMemBytes, 0u);

  comm().WarpSize = 32;
  comm().archName = const_cast<char*>("gfx1201");
  const DdaLaunch& noAsync = RunOneLL128(big);
  EXPECT_EQ(noAsync.kernel, (LL128TwoShot<float, 8>())) << "the async LDS stores are gfx1250 only";
  EXPECT_EQ(noAsync.sharedMemBytes, 0u);
}

// The one-shot keeps the cooperative writer at every size: it sends the whole message to each peer.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_OneShotKeepsTheCooperativeWriter) {
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
  comm().WarpSize = 32;
  comm().nRanks = 2;
  const DdaLaunch& two = RunOneLL128(4 * kMiB / 4);
  EXPECT_EQ(two.kernel, (LL128OneShot<float, 0>()));
  EXPECT_EQ(two.sharedMemBytes, 0u);
}

// Every rank count and datatype gets its own staged instantiation.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_StagedKernelsPerRankCountAndDatatype) {
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
  comm().WarpSize = 32;
  const struct {
    int nRanks;
    const void* two;
  } cases[] = {{4, LL128TwoShotW<float, 4, kAsync>()},
               {8, LL128TwoShotW<float, 8, kAsync>()},
               {16, LL128TwoShotW<float, 0, kAsync>()},
               {32, LL128TwoShotW<float, 0, kAsync>()}};
  for (const auto& c : cases) {
    comm().nRanks = c.nRanks;
    EXPECT_EQ(RunOneLL128(c.nRanks * 2 * kMiB / 4).kernel, c.two) << c.nRanks;
  }
  comm().nRanks = 8;
  EXPECT_EQ(RunOneLL128(16 * kMiB / 2, ncclBfloat16).kernel, (LL128TwoShotW<bf16, 8, kAsync>()));
  EXPECT_EQ(RunOneLL128(16 * kMiB / 2, ncclFloat16).kernel, (LL128TwoShotW<half, 8, kAsync>()));
  EXPECT_EQ(RunOneLL128(16 * kMiB, ncclFloat8e4m3).kernel, (LL128TwoShotW<dda::common::fp8e4m3, 8, kAsync>()));
  EXPECT_EQ(RunOneLL128(16 * kMiB, ncclFloat8e5m2).kernel, (LL128TwoShotW<dda::common::fp8e5m2, 8, kAsync>()));
}

// ---- Peer-split LL128 two-shot ----

constexpr int kWide = dda::common::kDdaLL128FlagBytesWide;
constexpr int kNarrow = dda::common::kDdaLL128FlagBytesNarrow;

template <typename T, int kFlagBytes = kWide, bool kHandoff = false, DdaArSplitB kB = kDdaArSplitBPairwise>
const void* LL128Split() {
  return KernelPtr(&dda::common::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, kHandoff, kB>);
}

// The narrow hand-off kernel of T with B form b.
template <typename T>
const void* LL128SplitHandoff(DdaArSplitB b) {
  return b == kDdaArSplitBWide          ? LL128Split<T, kNarrow, true, kDdaArSplitBWide>()
         : b == kDdaArSplitBWideChunked ? LL128Split<T, kNarrow, true, kDdaArSplitBWideChunked>()
                                        : LL128Split<T, kNarrow, true>();
}

// Slices of one shard of a `bytes` message on nRanks, for a flag of flagBytes.
size_t ShardSlices(size_t bytes, int nRanks, int flagBytes = kWide) {
  const size_t shard = bytes / (size_t)nRanks;
  return flagBytes == kNarrow ? dda::common::ddaLL128SlicesOf<kNarrow>(shard) : dda::common::ddaLL128Slices(shard);
}

// The split's grid: the replaced blocks to its flag's slices, a warp per unit a block of the 8 B flag's grid
// owns, to min(512, the replaced threads).
void ExpectSplitGrid(const DdaLaunch& split, rcclDdaLaunchCap replaced, size_t bytes, int nRanks, int flagBytes) {
  const size_t wide = ShardSlices(bytes, nRanks);
  const size_t blocks = std::min<size_t>(replaced.blocks, wide);
  const size_t warps = std::min<size_t>((wide + blocks - 1) / blocks * (nRanks - 1),
                                        std::min(kDdaArLL128Threads, replaced.threads) / kDdaLL128Warp);
  EXPECT_EQ(split.grid.x, std::min(blocks, ShardSlices(bytes, nRanks, flagBytes))) << "the replaced launch's blocks";
  EXPECT_EQ(split.block.x, warps * kDdaLL128Warp) << "a warp per unit a block owns";
}

rcclDdaLaunchCap GridOf(const DdaLaunch& l) {
  return {l.grid.x, l.block.x};
}

// A narrow split launch of T: the hand-off kernel and its LDS area where ddaArSplitTakesHandoff says, with the B form
// and its stage ddaArSplitB says, else my own slot. The rules are pinned on their own by RuleEdges and the tier table.
template <typename T>
void ExpectNarrowSplit(const DdaLaunch& l, int nRanks) {
  const bool handoff = ddaArSplitTakesHandoff(nRanks, l.Arg<size_t>(8), l.grid.x, l.block.x);
  const DdaArSplitB b =
    handoff ? ddaArSplitB(nRanks, l.Arg<size_t>(3), l.Arg<size_t>(8), l.grid.x) : kDdaArSplitBPairwise;
  EXPECT_EQ(l.kernel, handoff ? LL128SplitHandoff<T>(b) : (LL128Split<T, kNarrow>()));
  EXPECT_EQ(l.sharedMemBytes, dda::common::ddaLL128ArSplitSharedBytes(l.block.x, nRanks, handoff, b));
}

// The gfx1250 tables' tiers (LL128 to 256 MiB, LL to nRanks x 2 MiB) on wave32 with RCCL_DDA_LL128_ASYNC_WIRE=1.
void SplitTiersOn(ncclComm& c, std::map<std::string, int64_t>& params) {
  params["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return 256 * kMiB; };
  g_rcclDdaLLThreshold = [](const ncclComm* cm, ncclFunc_t) { return (size_t)cm->nRanks * 2 * kMiB; };
  g_rcclDdaLLTwoShotThreshold = [](const ncclComm* cm) { return (size_t)cm->nRanks * 2 * kMiB; };
  c.WarpSize = kDdaLL128Warp;
}

// gfx1250 SPX tiers as in rccl_arch_thresholds.cc and rccl_wrap.cc (LL, LL two-shot, LL128), 192 CUs, their scratch
// and VMM grid.
void ProductionTiersOn(ncclComm& c, std::map<std::string, int64_t>& params, int nRanks) {
  SplitTiersOn(c, params);
  auto ll = [](const ncclComm* cm) {
    return cm->nRanks < 8 ? 32 * kMiB : std::min<size_t>((size_t)cm->nRanks * kMiB, 256 * kMiB);
  };
  g_rcclDdaLLThreshold = [ll](const ncclComm* cm, ncclFunc_t) { return ll(cm); };
  g_rcclDdaLLTwoShotThreshold = [ll](const ncclComm* cm) { return cm->nRanks < 8 ? 2 * kMiB : ll(cm); };
  g_rcclDdaLL128TwoShotThreshold = [](const ncclComm*) { return 256 * kMiB; };
  g_rcclDdaFabricVmmBlocks = [](const ncclComm* cm) {
    const int cap = cm->ddaFabricMaxBlocks;
    return cm->nRanks < 8 ? cap : std::min(cap, std::min(128, std::max(32, 2048 / cm->nRanks)));
  };
  c.nRanks = nRanks;
  c.cuCount = 192;
  c.nChannels = 192;
  c.ddaFabricMaxBlocks = 192;
  c.ddaScratchBytes = nccl_dda_detail::ddaFabricScratchSizing(nRanks, -1, 1, 256 * kMiB, 1, 1, 256 * kMiB);
}

// The selector's VMM-first window on gfx1250 SPX (rcclDdaArVmmFirstTab): from 32 ranks, 2 MiB < bytes <= n x 512 KiB.
bool ProductionVmmFirst(int nRanks, size_t bytes) {
  return nRanks >= 32 && bytes > 2 * kMiB && bytes <= (size_t)nRanks * 512 * 1024;
}

// From 2 MiB the split runs a warp per (slice, peer) unit within the LL two-shot's blocks; Blocks reports its grid.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_PeerSplitTakesTheWindowWithinTheReplacedGrid) {
  SplitTiersOn(comm(), params_);
  const struct {
    int nRanks;
    size_t bytes;
  } cells[] = {{4, 2 * kMiB}, {8, 2 * kMiB}, {8, 4 * kMiB}, {16, 2 * kMiB}, {16, 8 * kMiB}, {32, 16 * kMiB}};
  for (const auto& c : cells) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r " << c.bytes);
    comm().nRanks = c.nRanks;
    const size_t shard = c.bytes / c.nRanks;
    const DdaLaunch replaced = RunReplaced(/*ll=*/true, c.bytes / 4);
    ASSERT_EQ(replaced.block.x, kDdaArLLTwoShotThreads) << "the LL two-shot";
    const DdaLaunch& l = RunSelectedLL128(c.bytes / 4);
    ExpectNarrowSplit<float>(l, c.nRanks);
    ExpectSplitGrid(l, GridOf(replaced), c.bytes, c.nRanks, kNarrow);
    EXPECT_EQ(l.Arg<size_t>(3), shard) << "shard bytes";
    EXPECT_EQ(l.Arg<size_t>(8), ShardSlices(c.bytes, c.nRanks, kNarrow)) << "shard slices";
    rcclDdaLaunchCap bound = {};
    ASSERT_TRUE(TakesSplit(c.bytes / 4, ncclFloat32, {}, /*vmmFirst=*/false, &bound));
    EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), c.bytes / 4, ncclFloat32, bound, /*split=*/true),
              l.grid.x);
  }
  comm().nRanks = 8;  // 2 MiB hands off to wide B
  EXPECT_EQ(RunSelectedLL128(2 * kMiB / 2, ncclBfloat16).kernel, LL128SplitHandoff<bf16>(kDdaArSplitBWide));
  EXPECT_EQ(RunSelectedLL128(2 * kMiB / 2, ncclFloat16).kernel, LL128SplitHandoff<half>(kDdaArSplitBWide));
  EXPECT_EQ(RunSelectedLL128(2 * kMiB, ncclFloat8e4m3).kernel,
            LL128SplitHandoff<dda::common::fp8e4m3>(kDdaArSplitBWide));
  EXPECT_EQ(RunSelectedLL128(2 * kMiB, ncclFloat8e5m2).kernel,
            LL128SplitHandoff<dda::common::fp8e5m2>(kDdaArSplitBWide));
}

// The window: above two ranks, from 2 MiB with no shard cap, up to the LL128 cap and past a pinned LL128 one-shot.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_Window) {
  SplitTiersOn(comm(), params_);
  EXPECT_TRUE(TakesSplitBytes(2 * kMiB)) << "8r: the 2 MiB cutoff";
  EXPECT_FALSE(TakesSplitBytes(2 * kMiB - 8 * 16)) << "8r just below 2 MiB";
  EXPECT_TRUE(TakesSplitBytes(8 * kMiB)) << "8r 8 MiB: 1 MiB shard";
  EXPECT_TRUE(TakesSplitBytes(64 * kMiB)) << "8r 64 MiB: 8 MiB shard, past the LL tier";
  comm().nRanks = 4;
  EXPECT_TRUE(TakesSplitBytes(4 * kMiB)) << "4r 4 MiB: 1 MiB shard";
  comm().nRanks = 3;
  EXPECT_TRUE(TakesSplitBytes(3 * kMiB)) << "3 ranks: two peers";
  comm().nRanks = 2;
  EXPECT_FALSE(TakesSplitBytes(3 * kMiB)) << "2 ranks: one peer";
  comm().nRanks = 8;
  comm().WarpSize = 2 * kDdaLL128Warp;
  EXPECT_FALSE(TakesSplitBytes(2 * kMiB)) << "an LL128 warp must be one wave";
  comm().WarpSize = kDdaLL128Warp;
  comm().archName = const_cast<char*>("gfx1201");
  EXPECT_FALSE(TakesSplitBytes(2 * kMiB)) << "no async LDS stores";
  comm().archName = const_cast<char*>("gfx1250");
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return 0; };
  EXPECT_FALSE(TakesSplitBytes(2 * kMiB)) << "LL128 tier off";
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return 8 * kMiB; };
  EXPECT_TRUE(TakesSplitBytes(8 * kMiB)) << "at the LL128 cap";
  EXPECT_FALSE(TakesSplitBytes(8 * kMiB + 8 * 16)) << "past the LL128 cap";
  g_rcclDdaArLL128OneShotPinnedByEnv = [] { return true; };
  params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = 3 * kMiB;
  EXPECT_FALSE(TakesSplitBytes(3 * kMiB)) << "the pinned LL128 one-shot's cap";
  EXPECT_TRUE(Eligible(&ddaLL128ArOneShotEligible, 3 * kMiB / 4)) << "the pinned LL128 one-shot keeps it";
  EXPECT_TRUE(TakesSplitBytes(3 * kMiB + 8 * 16)) << "past the pinned LL128 one-shot's cap";
}

// The split launch passes the two-shot's arguments: peers, buffers, shard bytes and slices, rank, epoch and bank.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_PeerSplitPassesTheTwoShotsArguments) {
  SplitTiersOn(comm(), params_);
  const DdaLaunch& launch = RunSelectedLL128(2 * kMiB / 2, ncclBfloat16);
  EXPECT_EQ(launch.kernel, LL128SplitHandoff<bf16>(kDdaArSplitBWide));
  EXPECT_EQ(launch.stopEvent, kStopEvent);
  // 16 warps' two windows, the hand-off area's 32 slices and ready words, wide B's 48 copies, the peer table.
  EXPECT_EQ(launch.sharedMemBytes, 16 * 2 * kSliceWire + 32 * (kSliceWire + 4) + 48 * kSliceWire + 8 * sizeof(void*));
  ASSERT_EQ(launch.args.size(), 11u);
  EXPECT_EQ(launch.Arg<bf16**>(0), comm().ddaPeerPtrsDev);
  EXPECT_EQ(launch.Arg<bf16*>(1), kRecvbuff);
  EXPECT_EQ(launch.Arg<const bf16*>(2), kSendbuff);
  EXPECT_EQ(launch.Arg<size_t>(3), 2 * kMiB / 8);
  EXPECT_EQ(launch.Arg<int>(4), 5);
  EXPECT_EQ(launch.Arg<int>(5), 8);
  EXPECT_EQ(launch.Arg<uint32_t*>(6), comm().ddaLLEpochDev);
  EXPECT_EQ(launch.Arg<int>(7), comm().ddaLLEpochLen);
  EXPECT_EQ(launch.Arg<size_t>(8), ShardSlices(2 * kMiB, 8, kNarrow));
  EXPECT_EQ(launch.Arg<size_t>(9), dda::common::ddaBankSize(comm().ddaScratchBytes));
  EXPECT_EQ(launch.Arg<dda::common::DdaKernelParams>(10).flag, comm().abortFlagDev);
}

// The split's bound is the replaced launch's grid: the LL two-shot in the LL tier, else the LL128 one-shot or two-shot.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_BoundIsTheReplacedLaunchersGrid) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;  // 8 ranks: 256 KiB shards
  auto bound = [&] {
    rcclDdaLaunchCap b = {};
    EXPECT_TRUE(TakesSplit(count, ncclFloat32, {}, /*vmmFirst=*/false, &b));
    return b;
  };
  auto expectSplitWithin = [&](const DdaLaunch& replaced, const char* tier) {
    SCOPED_TRACE(tier);
    EXPECT_EQ(bound().blocks, replaced.grid.x);
    EXPECT_EQ(bound().threads, replaced.block.x);
    const DdaLaunch& l = RunSelectedLL128(count);
    ExpectNarrowSplit<float>(l, 8);
    ExpectSplitGrid(l, GridOf(replaced), 2 * kMiB, 8, kNarrow);
  };
  const DdaLaunch ll = RunReplaced(/*ll=*/true, count);
  EXPECT_EQ(ll.kernel, (LLTwoShot<float, 8>()));
  expectSplitWithin(ll, "LL two-shot");

  g_rcclDdaLLThreshold = [](const ncclComm*, ncclFunc_t) { return kMiB; };
  const DdaLaunch oneShot = RunReplaced(/*ll=*/false, count);
  EXPECT_EQ(oneShot.kernel, (LL128OneShot<float, 8>())) << "LL tier short of the message";
  expectSplitWithin(oneShot, "LL128 one-shot");

  params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = kMiB;
  const DdaLaunch twoShot = RunReplaced(/*ll=*/false, count);
  EXPECT_EQ(twoShot.kernel, (LL128TwoShot<float, 8>()));
  expectSplitWithin(twoShot, "LL128 two-shot");
}

// Past the LL tier the split replaces the LL128 two-shot and fills its grid.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_PastTheLLTierFillsTheLL128TwoShotGrid) {
  SplitTiersOn(comm(), params_);
  const size_t bytes = 64 * kMiB;  // 8 ranks: 8 MiB shards, past the 16 MiB LL tier
  const DdaLaunch replaced = RunReplaced(/*ll=*/false, bytes / 4);
  EXPECT_EQ(replaced.kernel, (LL128TwoShot<float, 8>()));
  const DdaLaunch& l = RunSelectedLL128(bytes / 4);
  ExpectNarrowSplit<float>(l, 8);
  ExpectSplitGrid(l, GridOf(replaced), bytes, 8, kNarrow);
}

// A block owns whole slices, so a bound with more blocks than the shard has slices launches one block per slice.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_NoMoreBlocksThanShardSlices) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;
  for (const int nRanks : {32, 64}) {
    SCOPED_TRACE(testing::Message() << nRanks << "r 2 MiB");
    comm().nRanks = nRanks;
    const size_t slices = ShardSlices(2 * kMiB, nRanks, kNarrow);
    const rcclDdaLaunchCap vmm = ncclAllReduceDdaFabricGrid(mock_.get(), count, ncclFloat32, {});
    ASSERT_GT(vmm.blocks, slices) << "the VMM grid the split replaces has more blocks than slices";
    ASSERT_GT(slices * (nRanks - 1), slices * (kDdaArLL128Threads / kDdaLL128Warp)) << "and so do the units";
    const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, {}, /*vmmFirst=*/true);
    ExpectNarrowSplit<float>(l, nRanks);
    EXPECT_EQ(l.grid.x, slices);
    ExpectSplitGrid(l, vmm, 2 * kMiB, nRanks, kNarrow);
    rcclDdaLaunchCap bound = {};
    ASSERT_TRUE(TakesSplit(count, ncclFloat32, {}, /*vmmFirst=*/true, &bound));
    EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), count, ncclFloat32, bound, /*split=*/true), slices);
  }
  // 144 ranks of 15360 B shards: 8 slices on the VMM grid are 8 blocks, a small grid, so the 8 B flag.
  comm().nRanks = 144;
  const DdaLaunch& l = RunSelectedLL128(144 * 15360 / 4, ncclFloat32, {}, /*vmmFirst=*/true);
  EXPECT_EQ(l.kernel, (LL128Split<float>()));
  EXPECT_EQ(l.grid.x, 8u);
}

// Without a bound the split takes a warp per unit; a cap past the fabric's blocks or off a warp multiple is clamped.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_GridWithoutABoundOrPastTheFabric) {
  comm().nRanks = 4;
  comm().ddaFabricMaxBlocks = 256;
  const auto none = ddaArLL128SplitGrid(&comm(), 2 * kMiB, {});
  EXPECT_EQ(none.first.x, 52u) << "274 slices x 3 peers on 16 warps a block";
  EXPECT_EQ(none.second.x, 512u);
  comm().ddaFabricMaxBlocks = 40;
  const auto capped = ddaArLL128SplitGrid(&comm(), 2 * kMiB, {1000, 100});
  EXPECT_EQ(capped.first.x, 40u);
  EXPECT_EQ(capped.second.x, 96u);
}

// The VMM-first window's sizes stay in the split's window, so the selector leaves them to the LL128 tier.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_TakesVmmFirstSizes) {
  SplitTiersOn(comm(), params_);
  comm().nRanks = 32;
  EXPECT_TRUE(TakesSplitBytes(4 * kMiB, /*vmmFirst=*/true)) << "32r 4 MiB, a VMM-first size";
  EXPECT_TRUE(TakesSplitBytes(16 * kMiB, /*vmmFirst=*/true));
  EXPECT_FALSE(TakesSplitBytes(kMiB, /*vmmFirst=*/true)) << "below 2 MiB";
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 0;
  EXPECT_FALSE(TakesSplitBytes(4 * kMiB, /*vmmFirst=*/true)) << "staged writer off: the VMM-first window keeps it";
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
  comm().WarpSize = 2 * kDdaLL128Warp;
  EXPECT_FALSE(TakesSplitBytes(4 * kMiB, /*vmmFirst=*/true)) << "an LL128 warp must be one wave";
}

// Declining, the predicate leaves the bound at the cap, so the LL128 launch it leads to stays within the cap.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_DeclinedBoundIsTheCap) {
  SplitTiersOn(comm(), params_);
  const size_t count = kMiB / 4;  // 8 ranks, below the 2 MiB window
  const rcclDdaLaunchCap cap = {3, 256};
  const DdaLaunch uncapped = RunOneLL128(count);
  ASSERT_TRUE(uncapped.grid.x > cap.blocks || uncapped.block.x > cap.threads) << "the cap binds";
  rcclDdaLaunchCap bound = {};
  EXPECT_FALSE(TakesSplit(count, ncclFloat32, cap, /*vmmFirst=*/false, &bound));
  EXPECT_EQ(bound.blocks, 3u);
  EXPECT_EQ(bound.threads, 256u);
  const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, cap);
  EXPECT_NE(l.kernel, (LL128Split<float>()));
  EXPECT_NE(l.kernel, (LL128Split<float, kNarrow>()));
  EXPECT_LE(l.grid.x, cap.blocks);
  EXPECT_LE(l.block.x, cap.threads);
}

// In the VMM-first window the split replaces the VMM launch, so its bound is the VMM grid: its blocks and threads.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_InTheVmmFirstWindowTakesTheVmmGrid) {
  SplitTiersOn(comm(), params_);
  comm().nRanks = 32;
  const size_t count = 4 * kMiB / 4;  // 128 KiB shards
  const unsigned llTwoShot = RunReplaced(/*ll=*/true, count).grid.x;
  EXPECT_EQ(RunSelectedLL128(count).grid.x, llTwoShot) << "outside the VMM-first window: the LL two-shot's grid";
  ASSERT_TRUE(ncclAllReduceDdaFabricEligible(mock_.get(), kSendbuff, kRecvbuff, count, ncclFloat32, ncclSum));
  const rcclDdaLaunchCap vmm = ncclAllReduceDdaFabricGrid(mock_.get(), count, ncclFloat32, {});
  ASSERT_GT(vmm.blocks, llTwoShot);
  const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, {}, /*vmmFirst=*/true);
  ExpectNarrowSplit<float>(l, 32);
  ExpectSplitGrid(l, vmm, 4 * kMiB, 32, kNarrow);
  // Without a VMM-eligible comm (no fabric barrier) the selector would not run VMM: the LL two-shot's grid stays.
  auto* const barrier = comm().ddaFabricBarrierState;
  comm().ddaFabricBarrierState = nullptr;
  EXPECT_EQ(RunSelectedLL128(count, ncclFloat32, {}, /*vmmFirst=*/true).grid.x, llTwoShot);
  comm().ddaFabricBarrierState = barrier;
  // A smaller VMM grid than the LL two-shot's still bounds the split.
  g_rcclDdaFabricVmmBlocks = [](const ncclComm*) { return 4; };
  ASSERT_LT(ncclAllReduceDdaFabricBlocks(mock_.get(), count, ncclFloat32), llTwoShot);
  EXPECT_EQ(RunSelectedLL128(count, ncclFloat32, {}, /*vmmFirst=*/true).grid.x, 4u);

  // Under a launch cap the split stays within the capped VMM launch: its blocks and its threads per block.
  g_rcclDdaFabricVmmBlocks = [](const ncclComm* c) { return c->ddaFabricMaxBlocks; };
  comm().ddaFabricBarrierState = SharedBarrierState<nccl_dda_detail::DdaFabricBarrierState>(0x5a);
  std::vector<DdaCopy> copies;
  RecordDdaCopies(&copies);
  const rcclDdaLaunchCap caps[] = {{3, 256}, {0, 128}, {200, 0}, {7, 1024}};
  for (const rcclDdaLaunchCap& cap : caps) {
    SCOPED_TRACE(testing::Message() << "cap={" << cap.blocks << "," << cap.threads << "}");
    g_ddaLaunches.clear();
    ASSERT_EQ(ncclAllReduceDdaFabric(kSendbuff, kRecvbuff, count, ncclFloat32, ncclSum, mock_.get(), kStream, cap),
              ncclSuccess);
    const DdaLaunch vmmLaunch = Only();
    const DdaLaunch& capped = RunSelectedLL128(count, ncclFloat32, cap, /*vmmFirst=*/true);
    const bool small = cap.blocks != 0 && cap.blocks <= 8;  // a cap of 8 blocks or fewer keeps the 8 B flag
    if (small) {
      EXPECT_EQ(capped.kernel, (LL128Split<float>()));
    } else {
      ExpectNarrowSplit<float>(capped, 32);
    }
    ExpectSplitGrid(capped, GridOf(vmmLaunch), 4 * kMiB, 32, small ? kWide : kNarrow);
  }
}

// A user-set LL or LL128 one-shot threshold keeps the window's messages wherever that tier can take them.
TEST_F(DdaAllReduceFabricLLMicrotest, Eligible_PeerSplitWindowLeavesAPinnedTier) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;
  g_rcclDdaArLL128OneShotPinnedByEnv = [] { return true; };
  EXPECT_FALSE(TakesSplit(count)) << "LL128 one-shot pinned";
  EXPECT_TRUE(Eligible(&ddaLLArTwoShotEligible, count)) << "LL unpinned, LL128 one-shot pinned: the split is off";
  EXPECT_TRUE(Eligible(&ddaLL128ArOneShotEligible, count));
  g_rcclDdaArLL128OneShotPinnedByEnv = [] { return false; };
  g_rcclDdaArLLTierPinnedByEnv = [] { return true; };
  EXPECT_FALSE(TakesSplit(count)) << "LL pinned, the LL two-shot can take it";
  EXPECT_TRUE(Eligible(&ddaLLArTwoShotEligible, count));
  EXPECT_TRUE(Eligible(&ddaLL128ArOneShotEligible, count));
  g_rcclDdaLLTwoShotThreshold = [](const ncclComm*) { return kMiB; };
  EXPECT_TRUE(TakesSplit(count)) << "past both pinned LL shots the split still runs";
  EXPECT_FALSE(Eligible(&ddaLL128ArOneShotEligible, count));
  params_["RCCL_DDA_LL_ONESHOT_THRESHOLD"] = 2 * kMiB;
  ASSERT_TRUE(Eligible(&ddaLLArOneShotEligible, count));
  EXPECT_FALSE(TakesSplit(count)) << "the pinned LL one-shot can take it";
}

// A pinned LL tier keeps only the messages within its threshold: past it the split still takes them.
TEST_F(DdaAllReduceFabricLLMicrotest, Eligible_PinnedLLTierBelowTheMessageLeavesItToTheSplit) {
  SplitTiersOn(comm(), params_);
  comm().nRanks = 4;
  const size_t count = 2 * kMiB / 4;
  g_rcclDdaArLLTierPinnedByEnv = [] { return true; };
  EXPECT_FALSE(TakesSplit(count)) << "the pinned LL tier takes 2 MiB";
  g_rcclDdaLLThreshold = [](const ncclComm*, ncclFunc_t) { return kMiB / 2; };
  EXPECT_TRUE(TakesSplit(count)) << "RCCL_DDA_LL_THRESHOLD=512K: the LL tier cannot take 2 MiB";
}

// RCCL_DDA_LL_TWOSHOT_THRESHOLD set at 32 ranks: the split leaves the VMM-first window, so the selector runs VMM there.
TEST_F(DdaAllReduceFabricLLMicrotest, Eligible_PinnedLLTwoShotLeavesTheVmmFirstWindowToVmm) {
  SplitTiersOn(comm(), params_);
  comm().nRanks = 32;
  const size_t count = 4 * kMiB / 4;
  EXPECT_TRUE(TakesSplit(count, ncclFloat32, {}, /*vmmFirst=*/true)) << "unpinned: the split takes the VMM-first size";
  g_rcclDdaArLLTierPinnedByEnv = [] { return true; };
  EXPECT_FALSE(TakesSplit(count, ncclFloat32, {}, /*vmmFirst=*/true)) << "pinned: the VMM-first window keeps it";
  EXPECT_TRUE(Eligible(&ddaLLArTwoShotEligible, count)) << "the LL entry agrees with the selector";
}

// The LL two-shot and the LL128 one-shot leave window messages to the LL128 two-shot, as the selector does.
TEST_F(DdaAllReduceFabricLLMicrotest, Eligible_PeerSplitWindowLeavesTheLLTwoShotAndTheLL128OneShot) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;
  EXPECT_FALSE(Eligible(&ddaLLArTwoShotEligible, count));
  EXPECT_FALSE(Eligible(&ncclAllReduceDdaFabricLLEligible, count));
  EXPECT_FALSE(Eligible(&ddaLL128ArOneShotEligible, count));
  EXPECT_TRUE(Eligible(&ddaLL128ArTwoShotEligible, count));
  EXPECT_TRUE(Eligible(&ncclAllReduceDdaFabricLL128Eligible, count));
  comm().WarpSize = 2 * kDdaLL128Warp;
  EXPECT_TRUE(Eligible(&ddaLLArTwoShotEligible, count)) << "no split on wave64: the LL two-shot keeps it";
}

// RCCL_DDA_LL128_ASYNC_WIRE unset or 1 splits window messages; 0 keeps them on the LL and cooperative LL128 tiers.
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_PeerSplitFollowsTheStagedWireFlag) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;  // 8 ranks: 256 KiB shards
  for (const bool unset : {true, false}) {
    SCOPED_TRACE(testing::Message() << "unset " << unset);
    if (unset) {
      params_.erase("RCCL_DDA_LL128_ASYNC_WIRE");
    } else {
      params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 1;
    }
    EXPECT_TRUE(TakesSplit(count));
    EXPECT_FALSE(Eligible(&ddaLLArTwoShotEligible, count));
    EXPECT_FALSE(Eligible(&ddaLL128ArOneShotEligible, count));
    ExpectNarrowSplit<float>(RunSelectedLL128(count), 8);
  }
  params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 0;
  EXPECT_FALSE(TakesSplit(count));
  EXPECT_TRUE(Eligible(&ddaLLArTwoShotEligible, count));
  EXPECT_TRUE(Eligible(&ddaLL128ArOneShotEligible, count));
  EXPECT_EQ(RunOneLL(count).kernel, (LLTwoShot<float, 8>()));
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128OneShot<float, 8>()));
  params_["RCCL_DDA_LL128_ONESHOT_THRESHOLD"] = kMiB;
  const DdaLaunch& two = RunSelectedLL128(count);
  EXPECT_EQ(two.kernel, (LL128TwoShot<float, 8>()));
  EXPECT_EQ(two.sharedMemBytes, 0u);
}

// Under a launch cap the split stays within the blocks and threads of the capped launch it replaces (the LL two-shot).
TEST_F(DdaAllReduceFabricLLMicrotest, RunLL128_PeerSplitStaysWithinTheCappedReplacedLaunch) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;  // 8 ranks: 256 KiB shards
  const rcclDdaLaunchCap caps[] = {{3, 192}, {17, 0}, {0, 256}, {1000, 1024}, {2, 64}, {0, 96}};
  for (const rcclDdaLaunchCap& cap : caps) {
    SCOPED_TRACE(testing::Message() << "cap={" << cap.blocks << "," << cap.threads << "}");
    const DdaLaunch replaced = RunReplaced(/*ll=*/true, count, cap);
    EXPECT_EQ(replaced.kernel, (LLTwoShot<float, 8>()));
    const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, cap);
    const bool small = cap.blocks != 0 && cap.blocks <= 8;  // a cap of 8 blocks or fewer keeps the 8 B flag
    if (small) {
      EXPECT_EQ(l.kernel, (LL128Split<float>()));
      EXPECT_EQ(l.sharedMemBytes, dda::common::ddaLL128ArSplitSharedBytes(l.block.x, 8));
    } else {
      ExpectNarrowSplit<float>(l, 8);
    }
    EXPECT_LE(l.block.x, replaced.block.x);
    ExpectSplitGrid(l, GridOf(replaced), 2 * kMiB, 8, small ? kWide : kNarrow);
  }
}

// Every even cell on gfx1250 SPX tiers, 4-32 ranks and 2-128 MiB: grid, flag, hand-off and B form, within the launch
// the selector makes without the split.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_RuleGridAndHandoffOnProductionTiers) {
  const struct {
    int nRanks;
    size_t mib;
    unsigned grid;
    unsigned block;
    bool handoff;  // off: 4r 2M-4M under a unit a warp, past 17 slices a block, and 32r 2M on 8 blocks
    DdaArSplitB b = kDdaArSplitBPairwise;  // wide B to 256 KiB shards: 8r 2M wide, 32r 4M-8M chunked
  } cells[] = {
    {4, 2, 64, 480, false},    {4, 4, 137, 384, false},   {4, 8, 69, 512, true},     {4, 16, 137, 512, true},
    {4, 32, 192, 512, false},  {4, 64, 192, 512, false},  {4, 128, 192, 512, false},
    {8, 2, 32, 512, true, kDdaArSplitBWide},
    {8, 4, 64, 512, true},     {8, 8, 128, 512, true},    {8, 16, 69, 512, true},    {8, 32, 137, 512, true},
    {8, 64, 192, 512, false},  {8, 128, 192, 512, false}, {16, 2, 16, 512, true},    {16, 4, 32, 512, true},
    {16, 8, 64, 512, true},    {16, 16, 128, 512, true},  {16, 32, 69, 512, true},   {16, 64, 137, 512, true},
    {16, 128, 192, 512, false}, {32, 2, 8, 512, false},
    {32, 4, 64, 512, true, kDdaArSplitBWideChunked},      {32, 8, 64, 512, true, kDdaArSplitBWideChunked},
    {32, 16, 64, 512, true},   {32, 32, 128, 512, true},  {32, 64, 69, 512, true},   {32, 128, 137, 512, true}};
  for (const auto& c : cells) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r " << c.mib << " MiB");
    const size_t count = c.mib * kMiB / 4;
    ProductionTiersOn(comm(), params_, c.nRanks);
    const bool vmmFirst = ProductionVmmFirst(c.nRanks, c.mib * kMiB);
    // The selector without the split: VMM in its first window, else the LL tier where an LL shot takes it, else LL128.
    rcclDdaLaunchCap noSplit = ncclAllReduceDdaFabricGrid(mock_.get(), count, ncclFloat32, {});
    if (!vmmFirst) {
      params_["RCCL_DDA_LL128_ASYNC_WIRE"] = 0;
      const bool ll = c.mib * kMiB <= rcclDdaLLThreshold(&comm(), ncclFuncAllReduce) &&
                      Eligible(&ncclAllReduceDdaFabricLLEligible, count);
      noSplit = GridOf(RunReplaced(ll, count));
    }
    ASSERT_TRUE(TakesSplit(count, ncclFloat32, {}, vmmFirst));
    const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, {}, vmmFirst);
    EXPECT_EQ(l.grid.x, c.grid);
    EXPECT_EQ(l.block.x, c.block);
    EXPECT_EQ(l.kernel, c.handoff     ? LL128SplitHandoff<float>(c.b)
                        : c.grid <= 8 ? (LL128Split<float>())
                                      : (LL128Split<float, kNarrow>()));
    EXPECT_LE(l.grid.x, noSplit.blocks);
    EXPECT_LE(l.block.x, noSplit.threads);
  }
}

// The hand-off rule's edges: more than 8 blocks, at most 17 slices (34816 B) a block, at least one unit a warp.
TEST(DdaArSplitHandoff, RuleEdges) {
  EXPECT_FALSE(ddaArSplitTakesHandoff(32, 72, 8, 512)) << "8 blocks";
  EXPECT_TRUE(ddaArSplitTakesHandoff(32, 72, 9, 512)) << "9 blocks";
  EXPECT_TRUE(ddaArSplitTakesHandoff(4, 17 * 64, 64, 512)) << "17 slices a block";
  EXPECT_FALSE(ddaArSplitTakesHandoff(4, 17 * 64 + 1, 64, 512)) << "18 slices in a block";
  EXPECT_TRUE(ddaArSplitTakesHandoff(5, 256, 64, 512)) << "1024 units on 1024 warps";
  EXPECT_FALSE(ddaArSplitTakesHandoff(5, 255, 64, 512)) << "1020 units on 1024 warps";
  // 4r 4M on 137 x 384: 0.965 units a warp on 1984 B slices (the launch), 0.998 on 1920 B ones; one more slice is 1.0.
  EXPECT_FALSE(ddaArSplitTakesHandoff(4, 529, 137, 384));
  EXPECT_FALSE(ddaArSplitTakesHandoff(4, 547, 137, 384));
  EXPECT_TRUE(ddaArSplitTakesHandoff(4, 548, 137, 384));
}

// Wide B's edges: 8 and 9 copies, 8 and 9 slices a block, the 256 KiB shard; chunked: 16 and 17 copies and blocks,
// 5 and 6 slices a block; the shard cap holds for both.
TEST(DdaArSplitWideB, RuleEdges) {
  constexpr size_t kShard = kDdaArSplitWideMaxShardBytes;
  EXPECT_EQ(kShard, 256u * 1024);
  EXPECT_EQ(ddaArSplitB(9, kShard, 8 * 32, 32), kDdaArSplitBWide) << "8 copies, 8 slices a block";
  EXPECT_EQ(ddaArSplitB(2, kShard, 1, 1), kDdaArSplitBWide) << "one copy";
  EXPECT_EQ(ddaArSplitB(10, kShard, 8 * 32, 32), kDdaArSplitBPairwise) << "9 copies";
  EXPECT_EQ(ddaArSplitB(9, kShard, 8 * 32 + 1, 32), kDdaArSplitBPairwise) << "9 slices in a block";
  EXPECT_EQ(ddaArSplitB(9, kShard + 16, 8 * 32, 32), kDdaArSplitBPairwise) << "past the shard cap";
  EXPECT_EQ(ddaArSplitB(18, kShard, 5 * 17, 17), kDdaArSplitBWideChunked) << "17 copies, 17 blocks, 5 slices a block";
  EXPECT_EQ(ddaArSplitB(17, kShard, 5 * 17, 17), kDdaArSplitBPairwise) << "16 copies";
  EXPECT_EQ(ddaArSplitB(18, kShard, 5 * 16, 16), kDdaArSplitBPairwise) << "16 blocks";
  EXPECT_EQ(ddaArSplitB(18, kShard, 5 * 17 + 1, 17), kDdaArSplitBPairwise) << "6 slices in a block";
  EXPECT_EQ(ddaArSplitB(18, kShard + 16, 5 * 17, 17), kDdaArSplitBPairwise) << "past the shard cap";
  EXPECT_EQ(ddaArSplitB(144, kShard, 1, 1), kDdaArSplitBPairwise) << "one block";
  EXPECT_EQ(ddaArSplitB(8, kShard, 8, 0), kDdaArSplitBPairwise) << "no block";
}

// The hand-off kernel's dynamic LDS limit is raised once per device; a refused raise keeps my own slot.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_HandoffRaisesTheLdsLimitOncePerDevice) {
  SplitTiersOn(comm(), params_);
  const size_t count = 4 * kMiB / 4;  // 8 ranks: hands off, pairwise B
  int dev = 7;
  hipError_t getDevice = hipSuccess;
  hipError_t answer = hipErrorInvalidValue;
  int calls = 0;
  g_hipGetDevice = [&](int* d) {
    *d = dev;
    return getDevice;
  };
  g_hipFuncSetAttribute = [&](const void* kernel, hipFuncAttribute attr, int value) {
    ++calls;
    EXPECT_EQ(kernel, (LL128Split<float, kNarrow, true>()));
    EXPECT_EQ(attr, hipFuncAttributeMaxDynamicSharedMemorySize);
    EXPECT_EQ(value,
              (int)dda::common::ddaLL128ArSplitSharedBytes(kDdaArLL128Threads, dda::common::kDdaMaxNranks, true));
    return answer;
  };
  const DdaLaunch refused = RunSelectedLL128(count);
  EXPECT_EQ(refused.kernel, (LL128Split<float, kNarrow>())) << "refused: my own slot";
  EXPECT_EQ(refused.sharedMemBytes, dda::common::ddaLL128ArSplitSharedBytes(refused.block.x, 8));
  answer = hipSuccess;
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128Split<float, kNarrow, true>()));
  answer = hipErrorInvalidValue;
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128Split<float, kNarrow, true>())) << "raised on device 7";
  EXPECT_EQ(calls, 2);
  dev = 3;
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128Split<float, kNarrow>())) << "device 3 asks again";
  EXPECT_EQ(calls, 3);
  dev = 64;
  answer = hipSuccess;
  RunSelectedLL128(count);
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128Split<float, kNarrow, true>()));
  EXPECT_EQ(calls, 5) << "past 64 devices: every launch";
  getDevice = hipErrorInvalidDevice;
  EXPECT_EQ(RunSelectedLL128(count).kernel, (LL128Split<float, kNarrow>())) << "no device: my own slot";
  EXPECT_EQ(calls, 5);
  // Each type's kernel is raised on its own: bf16 on device 7 still asks.
  getDevice = hipSuccess;
  dev = 7;
  g_hipFuncSetAttribute = [&](const void* kernel, hipFuncAttribute, int) {
    ++calls;
    EXPECT_EQ(kernel, (LL128Split<bf16, kNarrow, true>()));
    return hipSuccess;
  };
  EXPECT_EQ(RunSelectedLL128(count * 2, ncclBfloat16).kernel, (LL128Split<bf16, kNarrow, true>()));
  EXPECT_EQ(RunSelectedLL128(count * 2, ncclBfloat16).kernel, (LL128Split<bf16, kNarrow, true>()));
  EXPECT_EQ(calls, 6) << "bf16 cached its own raise";
}

// B's form reads the launched grid: 128 ranks of 16400 B shards launch 9 of the VMM bound's 32 blocks, too few to
// chunk.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_WideBReadsTheLaunchedGrid) {
  ProductionTiersOn(comm(), params_, 128);
  const size_t count = 128 * 16400 / 4;
  rcclDdaLaunchCap bound = {};
  ASSERT_TRUE(TakesSplit(count, ncclFloat32, {}, /*vmmFirst=*/true, &bound));
  EXPECT_EQ(bound.blocks, 32u);
  const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, {}, /*vmmFirst=*/true);
  EXPECT_EQ(l.grid.x, 9u);
  EXPECT_EQ(l.kernel, LL128SplitHandoff<float>(kDdaArSplitBPairwise));
}

// Wide and chunked B raise their own kernel's limit, with the stage, once per device and type; refused, B is pairwise.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_WideBRaisesItsOwnLdsLimit) {
  std::vector<std::pair<const void*, int>> raised;
  hipError_t wideAnswer = hipErrorInvalidValue;
  g_hipFuncSetAttribute = [&](const void* kernel, hipFuncAttribute attr, int value) {
    EXPECT_EQ(attr, hipFuncAttributeMaxDynamicSharedMemorySize);
    raised.emplace_back(kernel, value);
    return kernel == LL128SplitHandoff<float>(kDdaArSplitBPairwise) ||
               kernel == LL128SplitHandoff<bf16>(kDdaArSplitBPairwise)
             ? hipSuccess
             : wideAnswer;
  };
  // 8 ranks at 2 MiB run wide B, 32 ranks at 4 MiB chunked wide B; each first refused, then granted.
  for (const auto& form : {std::make_pair(8, kDdaArSplitBWide), std::make_pair(32, kDdaArSplitBWideChunked)}) {
    SCOPED_TRACE(testing::Message() << form.first << "r");
    ProductionTiersOn(comm(), params_, form.first);
    const size_t count = (form.first == 8 ? 2 : 4) * kMiB / 4;
    const bool vmmFirst = form.first == 32;
    wideAnswer = hipErrorInvalidValue;
    const DdaLaunch refused = RunSelectedLL128(count, ncclFloat32, {}, vmmFirst);
    EXPECT_EQ(refused.kernel, LL128SplitHandoff<float>(kDdaArSplitBPairwise)) << "refused: pairwise B";
    EXPECT_EQ(refused.sharedMemBytes, dda::common::ddaLL128ArSplitSharedBytes(refused.block.x, form.first, true));
    wideAnswer = hipSuccess;
    const DdaLaunch granted = RunSelectedLL128(count, ncclFloat32, {}, vmmFirst);
    EXPECT_EQ(granted.kernel, LL128SplitHandoff<float>(form.second));
    EXPECT_EQ(granted.sharedMemBytes,
              dda::common::ddaLL128ArSplitSharedBytes(granted.block.x, form.first, true, form.second));
    EXPECT_EQ(RunSelectedLL128(count, ncclFloat32, {}, vmmFirst).kernel, LL128SplitHandoff<float>(form.second))
        << "raised";
    EXPECT_EQ(RunSelectedLL128(count * 2, ncclBfloat16, {}, vmmFirst).kernel, LL128SplitHandoff<bf16>(form.second));
    EXPECT_EQ(RunSelectedLL128(count * 2, ncclBfloat16, {}, vmmFirst).kernel, LL128SplitHandoff<bf16>(form.second))
        << "bf16 raised";
  }
  const int handoffLds =
    (int)dda::common::ddaLL128ArSplitSharedBytes(kDdaArLL128Threads, dda::common::kDdaMaxNranks, true);
  const int wideLds = (int)dda::common::ddaLL128ArSplitSharedBytes(kDdaArLL128Threads, dda::common::kDdaMaxNranks,
                                                                   true, kDdaArSplitBWide);
  const std::vector<std::pair<const void*, int>> want = {
    {LL128SplitHandoff<float>(kDdaArSplitBPairwise), handoffLds},
    {LL128SplitHandoff<float>(kDdaArSplitBWide), wideLds},
    {LL128SplitHandoff<float>(kDdaArSplitBWide), wideLds},
    {LL128SplitHandoff<bf16>(kDdaArSplitBPairwise), handoffLds},
    {LL128SplitHandoff<bf16>(kDdaArSplitBWide), wideLds},
    {LL128SplitHandoff<float>(kDdaArSplitBWideChunked), wideLds},
    {LL128SplitHandoff<float>(kDdaArSplitBWideChunked), wideLds},
    {LL128SplitHandoff<bf16>(kDdaArSplitBWideChunked), wideLds}};
  EXPECT_EQ(raised, want) << "each form refused, then raised once a type";
}

// ---- The peer-split's narrow 4 B LL128 flag ----

// Epoch 7 as the kernels build the 8 B flag word (the epoch in both halves), and its narrow flag dword (xor the key).
constexpr uint64_t kFlag7 = 0x0000000700000007ull;
constexpr uint32_t kNarrowWire = 0x7FF5A5A2u;

// One slice of distinct user dwords through every lane's narrow load, pack and wire hi word.
std::vector<uint64_t> NarrowWireOf(const std::vector<uint32_t>& user) {
  namespace dc = dda::common;
  std::vector<uint64_t> wire(dc::kDdaLL128WireWordsPerSlice);
  for (int lane = 0; lane < dc::kDdaLL128Warp; ++lane) {
    const bool flagLane = dc::ddaLL128IsFlagLane(lane);
    uint64_t regs[dc::kDdaLL128WordsPerThread];
    for (int g = 0; g < dc::kDdaLL128Pairs; ++g) {
      const int ix = dc::ddaLL128NarrowLoadChunkIx(g, lane, flagLane);
      EXPECT_GE(ix, 0);
      EXPECT_LT(16 * ix, dc::kDdaLL128NarrowDataBytesPerSlice) << "lane " << lane << " pair " << g;
      std::memcpy(&regs[2 * g], &user[(size_t)4 * ix], 16);
    }
    dc::ddaLL128NarrowPack(regs, flagLane);
    for (int g = 0; g < dc::kDdaLL128Pairs; ++g) {
      wire[(size_t)g * 2 * dc::kDdaLL128Warp + 2 * lane] = regs[2 * g];
      wire[(size_t)g * 2 * dc::kDdaLL128Warp + 2 * lane + 1] = dc::ddaLL128NarrowWireHi(regs[2 * g + 1], kFlag7,
                                                                                          flagLane);
    }
  }
  return wire;
}

std::vector<uint32_t> DistinctSliceDwords() {
  std::vector<uint32_t> user(dda::common::kDdaLL128NarrowDataBytesPerSlice / 4);
  for (size_t i = 0; i < user.size(); ++i) {
    user[i] = (uint32_t)(i + 1) * 0x01010101u;
  }
  return user;
}

// Every line: 31 user dwords and the narrow flag in its last dword; the slice: each user dword exactly once.
TEST(DdaLL128NarrowLayout, EachLineCarries31UserDwordsAndTheFlagLast) {
  const std::vector<uint32_t> user = DistinctSliceDwords();
  const std::vector<uint64_t> wire = NarrowWireOf(user);
  std::map<uint32_t, int> seen;
  for (size_t line = 0; line < kSliceWire / 128; ++line) {
    const uint32_t* d = reinterpret_cast<const uint32_t*>(wire.data()) + line * 32;
    EXPECT_EQ(d[31], kNarrowWire) << "line " << line;
    for (int k = 0; k < 31; ++k) {
      ++seen[d[k]];
    }
  }
  ASSERT_EQ(seen.size(), user.size()) << "lines carry other dwords than the slice's";
  for (const uint32_t v : user) {
    EXPECT_EQ(seen[v], 1) << "user dword " << std::hex << v;
  }
}

// Unpack of the wire registers gives back every user chunk at its index; every flag lane's hi word shows the flag.
TEST(DdaLL128NarrowLayout, UnpackRestoresTheSlice) {
  namespace dc = dda::common;
  const std::vector<uint32_t> user = DistinctSliceDwords();
  const std::vector<uint64_t> wire = NarrowWireOf(user);
  std::vector<uint32_t> back(user.size(), 0);
  for (int lane = 0; lane < dc::kDdaLL128Warp; ++lane) {
    const bool flagLane = dc::ddaLL128IsFlagLane(lane);
    uint64_t regs[dc::kDdaLL128WordsPerThread];
    for (int g = 0; g < dc::kDdaLL128Pairs; ++g) {
      regs[2 * g] = wire[(size_t)g * 2 * dc::kDdaLL128Warp + 2 * lane];
      regs[2 * g + 1] = wire[(size_t)g * 2 * dc::kDdaLL128Warp + 2 * lane + 1];
      if (flagLane) {
        EXPECT_TRUE(dc::ddaLL128NarrowFlagSeen(regs[2 * g + 1], kFlag7)) << "lane " << lane << " pair " << g;
      }
    }
    dc::ddaLL128NarrowUnpack(regs, flagLane);
    for (int g = 0; g < dc::kDdaLL128Pairs; ++g) {
      if (!flagLane || g < dc::kDdaLL128NarrowFullPairs) {
        std::memcpy(&back[(size_t)4 * dc::ddaLL128NarrowChunkIx(g, lane)], &regs[2 * g], 16);
      }
    }
  }
  EXPECT_EQ(back, user);
}

// Pairs 0..2 are 32 consecutive chunks; pair 3 the data lanes' 28 in lane order; a flag lane's pair 3 loads lane - 1's.
TEST(DdaLL128NarrowLayout, ChunksAreLaneContiguous) {
  namespace dc = dda::common;
  int next3 = 3 * dc::kDdaLL128Warp;
  for (int lane = 0; lane < dc::kDdaLL128Warp; ++lane) {
    for (int g = 0; g < dc::kDdaLL128NarrowFullPairs; ++g) {
      EXPECT_EQ(dc::ddaLL128NarrowChunkIx(g, lane), g * dc::kDdaLL128Warp + lane);
    }
    if (dc::ddaLL128IsFlagLane(lane)) {
      EXPECT_EQ(dc::ddaLL128NarrowLoadChunkIx(3, lane, true), dc::ddaLL128NarrowChunkIx(3, lane - 1));
    } else {
      EXPECT_EQ(dc::ddaLL128NarrowChunkIx(3, lane), next3++);
    }
  }
  EXPECT_EQ(next3, dc::kDdaLL128NarrowDataBytesPerSlice / 16);
}

// Only the top dword is the flag: any data dword below it; the plain epoch, as stale user data may hold, never matches.
TEST(DdaLL128NarrowLayout, FlagIsTheTopDwordOnly) {
  namespace dc = dda::common;
  EXPECT_TRUE(dc::ddaLL128NarrowFlagSeen(0x7FF5A5A2DEADBEEFull, kFlag7));
  EXPECT_TRUE(dc::ddaLL128NarrowFlagSeen(0x7FF5A5A27FF5A5A2ull, kFlag7));
  EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen(0x0000000700000007ull, kFlag7)) << "the plain epoch";
  EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen(0x7FF5A5A07FF5A5A2ull, kFlag7)) << "data equal to the flag, stale flag";
  EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen(0xFFF5A5A2DEADBEEFull, kFlag7));
  EXPECT_EQ(dc::ddaLL128NarrowWireHi(0x12345678DEADBEEFull, kFlag7, true), 0x7FF5A5A2DEADBEEFull);
  EXPECT_EQ(dc::ddaLL128NarrowWireHi(0x12345678DEADBEEFull, kFlag7, false), 0x12345678DEADBEEFull);
}

// A stale line of a ddaBankSize-banked tier holds an older same-parity epoch, or 0, at bytes 124..127; neither matches.
TEST(DdaLL128NarrowLayout, NarrowFlagNeverMatchesAStaleEpoch) {
  namespace dc = dda::common;
  EXPECT_EQ(7 * sizeof(dc::LLPacket16) + offsetof(dc::LLPacket16, flag1), (size_t)124);
  for (const uint64_t e : {2ull, 3ull, 7ull, 0x10000ull, 0x7FF5A5A5ull, 0x7FF5A5A7ull, 0xFFFFFFFEull, 0xFFFFFFFFull}) {
    const uint64_t flag = (e << 32) | e;
    EXPECT_NE((dc::ddaLL128NarrowFlagOf(flag) >> 32) & 1, e & 1) << "epoch " << e;
    for (uint64_t back = 0; back <= 64; back += 2) {
      const uint64_t old = e - back;  // the same bank's epochs, this one and older
      EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen((old << 32) | old, flag)) << "epoch " << e << " stale " << old;
    }
    EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen(e, flag)) << "epoch " << e << ": an 8 B flag in the low dword, 0 above";
    EXPECT_FALSE(dc::ddaLL128NarrowFlagSeen(0, flag)) << "epoch " << e << ": a zeroed line";
  }
  EXPECT_EQ(dc::ddaLL128NarrowFlagOf(0x7FF5A5A57FF5A5A5ull), 0x8000000000000000ull) << "the key's own epoch";
}

// Slices by 1984 data bytes: every one full but the last, which holds the rest; the 8 B flag's sizing unchanged.
TEST(DdaLL128NarrowLayout, SlicesAreSizedByDataBytes) {
  namespace dc = dda::common;
  EXPECT_EQ(dc::ddaLL128DataBytesPerSliceOf<kNarrow>(), 1984);
  EXPECT_EQ(dc::ddaLL128DataBytesPerSliceOf<kWide>(), (int)kSliceData);
  for (const size_t bytes : {(size_t)16, (size_t)1984 - 16, (size_t)1984, (size_t)1984 + 16, (size_t)2 * 1984 + 112,
                             (size_t)128 * 1024, (size_t)2 * kMiB / 16, (size_t)8 * kMiB, (size_t)8 * kMiB + 112}) {
    SCOPED_TRACE(testing::Message() << bytes);
    const size_t n = dc::ddaLL128SlicesOf<kNarrow>(bytes);
    EXPECT_EQ(n, (bytes + 1983) / 1984);
    size_t sum = 0;
    for (size_t s = 0; s < n; ++s) {
      const int b = dc::ddaLL128SliceBytesOf<kNarrow>(bytes, s);
      EXPECT_EQ(b, s + 1 < n ? 1984 : (int)(bytes - s * 1984));
      sum += (size_t)b;
    }
    EXPECT_EQ(sum, bytes);
    EXPECT_LE(n, dc::ddaLL128Slices(bytes)) << "never more slices than the 8 B flag's";
    EXPECT_EQ(dc::ddaLL128SlicesOf<kWide>(bytes), dc::ddaLL128Slices(bytes));
    EXPECT_EQ(dc::ddaLL128SliceBytesOf<kWide>(bytes, 0), dc::ddaLL128SliceBytes(bytes, 0));
  }
}

// Past 8 blocks a split launch runs the 4 B flag on the 8 B flag's grid, down to the 1984 B slices; Blocks agrees.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_NarrowFlagPastEightBlocks) {
  SplitTiersOn(comm(), params_);
  comm().ddaFabricMaxBlocks = 256;
  comm().ddaScratchBytes = (size_t)1 << 30;
  for (const int n : {4, 8, 16, 32, 64}) {
    comm().nRanks = n;
    for (const size_t bytes : {4 * kMiB, 16 * kMiB, 64 * kMiB}) {
      SCOPED_TRACE(testing::Message() << n << "r " << bytes);
      rcclDdaLaunchCap bound = {};
      ASSERT_TRUE(TakesSplit(bytes / 4, ncclFloat32, {}, false, &bound));
      const DdaLaunch& l = RunSelectedLL128(bytes / 4);
      const size_t wideGrid = std::min<size_t>(bound.blocks, ShardSlices(bytes, n));
      if (wideGrid <= 8) {
        EXPECT_EQ(n, 64) << "only 64r 4 MiB is a small grid here";
        EXPECT_EQ(l.kernel, (LL128Split<float>()));
        EXPECT_EQ(l.grid.x, wideGrid);
        continue;
      }
      ExpectNarrowSplit<float>(l, n);
      EXPECT_EQ(l.Arg<size_t>(8), ShardSlices(bytes, n, kNarrow)) << "shard slices";
      EXPECT_EQ(l.Arg<size_t>(3), bytes / (size_t)n) << "shard bytes";
      ExpectSplitGrid(l, bound, bytes, n, kNarrow);
      EXPECT_EQ(ncclAllReduceDdaFabricLL128Blocks(mock_.get(), bytes / 4, ncclFloat32, bound, /*split=*/true),
                l.grid.x);
    }
  }
  // 64 ranks, 2 MiB without a bound: 32 KiB shards are 18 slices of 1920 B but 17 of 1984 B, so 17 blocks.
  comm().nRanks = 64;
  int flagBytes = 0;
  EXPECT_EQ(ShardSlices(2 * kMiB, 64), 18u);
  EXPECT_EQ(ddaArLL128SplitGrid(&comm(), 2 * kMiB, {}, &flagBytes).first.x, 17u);
  EXPECT_EQ(flagBytes, kNarrow);
  comm().nRanks = 16;
  ExpectNarrowSplit<bf16>(RunSelectedLL128(4 * kMiB / 2, ncclBfloat16), 16);
  ExpectNarrowSplit<dda::common::fp8e4m3>(RunSelectedLL128(4 * kMiB, ncclFloat8e4m3), 16);
}

// A split launch whose 8 B flag grid has 8 blocks or fewer keeps the 8 B flag, its slices, grid and my own slot; 9 take
// the 4 B flag and the hand-off.
TEST_F(DdaAllReduceFabricLLMicrotest, Split_SmallGridKeepsTheWideFlag) {
  SplitTiersOn(comm(), params_);
  const size_t count = 2 * kMiB / 4;  // 8 ranks: 256 KiB shards, 137 slices of 1920 B, 133 of 1984 B
  for (const unsigned blocks : {1u, 4u, 8u, 9u, 12u}) {
    SCOPED_TRACE(testing::Message() << blocks << " blocks");
    const rcclDdaLaunchCap cap = {blocks, 0};
    const bool narrow = blocks > 8;
    const DdaLaunch& l = RunSelectedLL128(count, ncclFloat32, cap);
    EXPECT_EQ(l.grid.x, blocks);
    EXPECT_EQ(l.kernel, narrow ? (LL128Split<float, kNarrow, true>()) : (LL128Split<float>()));
    EXPECT_EQ(l.Arg<size_t>(8), ShardSlices(2 * kMiB, 8, narrow ? kNarrow : kWide));
    EXPECT_EQ(RunSelectedLL128(count * 2, ncclBfloat16, cap).kernel,
              narrow ? (LL128Split<bf16, kNarrow, true>()) : (LL128Split<bf16>()));
  }
  // 32 ranks, 2 MiB and 64 ranks, 2 MiB on the replaced launch's 8 and 4 blocks: the 8 B flag and its slices.
  for (const auto& c : {std::pair<int, unsigned>{32, 8}, {64, 4}}) {
    comm().nRanks = c.first;
    const DdaLaunch& l = RunSelectedLL128(count);
    EXPECT_EQ(l.kernel, (LL128Split<float>())) << c.first << "r";
    EXPECT_EQ(l.grid.x, c.second) << c.first << "r";
    EXPECT_EQ(l.Arg<size_t>(8), ShardSlices(2 * kMiB, c.first)) << c.first << "r";
  }
  // 8 ranks of 300000 B shards on 9 blocks: 17 slices of 1984 B a block (18 of 1920 B) hand off.
  comm().nRanks = 8;
  const DdaLaunch& seventeen = RunSelectedLL128(8 * 300000 / 4, ncclFloat32, {9, 0});
  EXPECT_EQ(seventeen.grid.x, 9u);
  EXPECT_EQ(seventeen.Arg<size_t>(8), 152u);
  EXPECT_EQ(seventeen.kernel, (LL128Split<float, kNarrow, true>()));
  // The rule reads the 8 B flag's grid: 136 ranks of 15424 B shards are 9 slices of 1920 B, 8 of 1984 B.
  comm().nRanks = 136;
  int flagBytes = 0;
  EXPECT_EQ(ddaArLL128SplitGrid(&comm(), 136 * 15424, {}, &flagBytes).first.x, 8u);
  EXPECT_EQ(flagBytes, kNarrow);
}

}  // namespace
