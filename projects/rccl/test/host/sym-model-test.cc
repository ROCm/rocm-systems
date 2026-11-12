/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the REAL symmetric cost model (src/tuning/sym_model.cc and
// src/tuning/sym_model/*.cc), focused on how it costs the Tensor Data Mover (Tma) kernels on
// gfx1250 now that NCCL_SYM_TMA_ENABLE defaults to 1. Linked into rccl-UnitTestsMicroSymKernels,
// whose sym-kernels-test.cc TU supplies the real src/sym_kernels.cc.

#include <gtest/gtest.h>

#include <cfloat>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "SymTdmTestHelpers.h"
#include "fakes/devcomm_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"
#include "tuning.h"

#include SYM_MODEL_CC_PATH
#include SYM_MODEL_GIN_CC_PATH
#include SYM_MODEL_LSA_CC_PATH
#include SYM_MODEL_LSA_BASE_CC_PATH
#include SYM_MODEL_LSA_A2A_CC_PATH

namespace {

constexpr int kRanks = 4;
constexpr ncclDataType_t kTy = ncclFloat32;

ncclSymkKernelId TmaKernelFor(ncclFunc_t func) {
  switch (func) {
  case ncclFuncAllReduce:
    return ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST;
  case ncclFuncAllGather:
    return ncclSymkKernelId_AllGather_TmaST;
  default:
    return ncclSymkKernelId_ReduceScatter_TmaLD;
  }
}

const char* FuncName(ncclFunc_t func) {
  switch (func) {
  case ncclFuncAllReduce:
    return "AllReduce";
  case ncclFuncAllGather:
    return "AllGather";
  default:
    return "ReduceScatter";
  }
}

struct Pick {
  int kernelId = ncclSymkKernelId_Count;
  float selectionTimeUs = 0.0f;
  int nChannels = 0;
  int nWarps = 0;
};

class SymModelTdmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kRanks;
    comm_->nNodes = 1;
    comm_->symmetricSupport = true;
    comm_->isAllDirectNvlink = true;
    comm_->symkState.hasLsaMultimem = false;
    SetTdmCapableArch(comm_.get());
  }
  void TearDown() override {
    ResetDevcommFakes();
    ResetNcclFakes();
  }

  void SetArch(const char* arch, int compCap, int warpSize) { SetSymkArch(comm_.get(), arch, compCap, warpSize); }

  // ncclTuningSymkModelSim reads nBytes as the per-rank bytes for AllGather/ReduceScatter and the
  // message bytes for AllReduce, exactly as symmetric_sched.cc fills it for a single work.
  ncclTuningInput_t Input(ncclFunc_t func, size_t nBytes) const {
    ncclTuningInput_t in{};
    in.comm = comm_.get();
    in.tuningMask = ~uint64_t(0);
    in.func = func;
    in.redOp = ncclSum;
    in.devRedOp = ncclDevSum;
    in.datatype = kTy;
    in.nBytes = nBytes;
    in.count = in.countMax = nBytes / ncclTypeSize(kTy);
    in.nWorks = 1;
    in.winRegType = ncclSymSendRegRecvReg;
    in.symAligned16B = true;
    in.minCTAs = 1;
    in.maxCTAs = ncclSymkMaxBlocks;
    return in;
  }

  static ncclTuningResult_t Evaluate(ncclTuningInput_t in, int kernelId) {
    ncclTuningResult_t r{};
    r.symKernelId = kernelId;
    r.valid = 1;
    EXPECT_EQ(ncclTuningSymkModelSim(&in, &r), ncclSuccess);
    return r;
  }

  // Mirrors tuning.cc: the valid kernel with the lowest selection time wins.
  static Pick Select(const ncclTuningInput_t& in) {
    Pick best;
    float bestTime = FLT_MAX;
    for (int k = 0; k < ncclSymkKernelId_Count; ++k) {
      ncclTuningResult_t r = Evaluate(in, k);
      if (!r.valid) continue;
      float t = r.selectionTimeUs > 0.0f ? r.selectionTimeUs : r.timeUs;
      if (t < bestTime) {
        bestTime = t;
        best = {k, t, r.nChannels, r.nWarps};
      }
    }
    return best;
  }

  // Large messages, where the Tma kernels would otherwise win, must land on a vector kernel.
  void ExpectLargeMessagesStayOnVectorKernels(const std::function<void(ncclTuningInput_t&)>& mutate = {}) {
    for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
      ncclTuningInput_t in = Input(func, size_t(256) << 20);
      if (mutate) mutate(in);
      Pick p = Select(in);
      ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << FuncName(func);
      EXPECT_FALSE(rcclSymkKernelIdIsTma(p.kernelId))
          << FuncName(func) << " picked " << ncclSymkKernelIdToString(p.kernelId);
    }
  }

  std::unique_ptr<ncclComm> comm_;
};

// For cases that need a Tma kernel to survive the mask, which drops them without the SDK's TDM header.
class SymModelTdmReachableTest : public SymModelTdmTest {
 protected:
  void SetUp() override {
    SymModelTdmTest::SetUp();
    if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  }
};

// ---- Default enablement: NCCL_SYM_TMA_ENABLE left at its compile-time default ----

TEST_F(SymModelTdmReachableTest, DefaultParam_Gfx1250_TmaAvailableButNotForced) {
  EXPECT_TRUE(ncclSymkTmaAvailable(comm_.get()));
  EXPECT_FALSE(ncclSymkTmaForced(comm_.get())) << "the default must leave the choice to the cost model";
}

TEST_F(SymModelTdmTest, DefaultParam_NonTdmArch_TmaUnavailable) {
  struct {
    const char* arch;
    int compCap;
    int warpSize;
  } const archs[] = {{"gfx950", 95, 64}, {"gfx942", 94, 64}, {"gfx90a", 90, 64}, {"gfx1200", 120, 32}};
  for (const auto& a : archs) {
    SetArch(a.arch, a.compCap, a.warpSize);
    EXPECT_FALSE(ncclSymkTmaAvailable(comm_.get())) << a.arch;
  }
  comm_->archName = nullptr;
  EXPECT_FALSE(ncclSymkTmaAvailable(comm_.get())) << "unknown arch";
}

TEST_F(SymModelTdmTest, DefaultParam_Gfx950_LargeMessagesStayOnVectorKernels) {
  SetArch("gfx950", 95, /*warpSize=*/64);
  ExpectLargeMessagesStayOnVectorKernels();
}

// ---- Cost model integration: which kernel the model picks, and how it launches it ----

struct SelectionCase {
  ncclFunc_t func;
  size_t nBytes;  // per-rank for AllGather/ReduceScatter, message for AllReduce
  bool expectTma;
};

std::ostream& operator<<(std::ostream& os, const SelectionCase& c) {
  return os << FuncName(c.func) << "_" << c.nBytes;
}

class SymModelTdmSelectionTest : public SymModelTdmTest, public ::testing::WithParamInterface<SelectionCase> {};

TEST_P(SymModelTdmSelectionTest, DefaultParam_PicksTmaOnlyForLargeMessages) {
  const SelectionCase& c = GetParam();
  if (c.expectTma && !TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  Pick p = Select(Input(c.func, c.nBytes));
  ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << "no symmetric kernel was valid";
  EXPECT_EQ(rcclSymkKernelIdIsTma(p.kernelId), c.expectTma) << "picked " << ncclSymkKernelIdToString(p.kernelId);
  if (c.expectTma) {
    EXPECT_EQ(p.kernelId, TmaKernelFor(c.func));
    // TDM throughput keeps scaling with resident blocks, and symCheckTmaLaunch() needs all 16 warps.
    EXPECT_EQ(p.nChannels, ncclSymkMaxBlocks);
    EXPECT_EQ(p.nWarps, ncclSymkWarpsPerBlock);
  }
}

// Endpoints only: the crossover between them is a tuning choice, not a contract.
INSTANTIATE_TEST_SUITE_P(SymTdmEndpoints, SymModelTdmSelectionTest,
                         ::testing::Values(SelectionCase{ncclFuncAllReduce, 64 << 10, false},
                                           SelectionCase{ncclFuncAllReduce, size_t(1) << 30, true},
                                           SelectionCase{ncclFuncAllGather, 16 << 10, false},
                                           SelectionCase{ncclFuncAllGather, size_t(256) << 20, true},
                                           SelectionCase{ncclFuncReduceScatter, 16 << 10, false},
                                           SelectionCase{ncclFuncReduceScatter, size_t(256) << 20, true}),
                         [](const ::testing::TestParamInfo<SelectionCase>& info) {
                           return std::string(FuncName(info.param.func)) + "_" +
                                  std::to_string(info.param.nBytes >> 10) + "KiB";
                         });

// The model must not offer a Tma kernel whose deep loop would not run even at the minimum block count.
TEST_F(SymModelTdmReachableTest, DefaultParam_BelowDeepLoopBar_TmaKernelInvalid) {
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    ncclSymkKernelId k = TmaKernelFor(func);
    size_t nBytes = 4 << 10;
    ASSERT_FALSE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, /*nBlocks=*/1)) << FuncName(func);
    EXPECT_FALSE(Evaluate(Input(func, nBytes), k).valid) << FuncName(func);
  }
}

// When nMaxBlocks would starve the deep loop the model trims blocks rather than dropping the kernel.
TEST_F(SymModelTdmReachableTest, DefaultParam_JustAboveDeepLoopBar_LaunchesOnlyBlocksTheDeepLoopCanFeed) {
  ncclSymkKernelId k = ncclSymkKernelId_ReduceScatter_TmaLD;
  size_t nBytes = size_t(ncclSymkDeepMaxBytePerChunk) * kRanks * 2;  // deep-eligible at 2 blocks, not at 4
  ASSERT_TRUE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, 2));
  ASSERT_FALSE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, 4));
  ncclTuningResult_t r = Evaluate(Input(ncclFuncReduceScatter, nBytes), k);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, 2);
}

// Large messages: the TDM terms (higher bandwidth for extra setup latency) must beat the vector kernel.
TEST_F(SymModelTdmReachableTest, DefaultParam_LargeMessage_TmaCheaperThanVectorCounterpart) {
  struct {
    ncclFunc_t func;
    ncclSymkKernelId vector;
    size_t nBytes;
  } const cases[] = {
      {ncclFuncAllReduce, ncclSymkKernelId_AllReduce_RSxLD_AGxST, size_t(1) << 30},
      {ncclFuncAllGather, ncclSymkKernelId_AllGather_ST, size_t(256) << 20},
      {ncclFuncReduceScatter, ncclSymkKernelId_ReduceScatter_LD, size_t(256) << 20},
  };
  for (const auto& c : cases) {
    ncclTuningResult_t tma = Evaluate(Input(c.func, c.nBytes), TmaKernelFor(c.func));
    ncclTuningResult_t vec = Evaluate(Input(c.func, c.nBytes), c.vector);
    ASSERT_TRUE(tma.valid && vec.valid) << FuncName(c.func);
    EXPECT_LT(tma.selectionTimeUs, vec.selectionTimeUs) << FuncName(c.func);
  }
}

// ---- Gates that keep the default-on Tma kernels out of reach ----
// Each skips without the TDM header: the mask already drops Tma there, so the gate under test would not run.

TEST_F(SymModelTdmReachableTest, TmaParamDisabled_LargeMessagePicksVectorKernel) {
  ScopedHook loadParam(g_loadParam, SymTmaEnable(0));
  ExpectLargeMessagesStayOnVectorKernels();
}

TEST_F(SymModelTdmReachableTest, DefaultParam_NotSymAligned16B_LargeMessagePicksVectorKernel) {
  ExpectLargeMessagesStayOnVectorKernels([](ncclTuningInput_t& in) { in.symAligned16B = false; });
}

// Deep-loop eligibility is per work (countMax), not per batch: many small grouped works must not reach Tma.
TEST_F(SymModelTdmReachableTest, DefaultParam_GroupOfSmallWorks_TmaKernelInvalid) {
  ncclTuningInput_t in = Input(ncclFuncAllReduce, size_t(256) << 20);
  in.countMax = (4 << 10) / ncclTypeSize(kTy);
  in.nWorks = 1024;
  EXPECT_FALSE(Evaluate(in, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST).valid);
}

// Without registered windows on both sides only LL kernels are valid, Tma included.
TEST_F(SymModelTdmReachableTest, DefaultParam_UnregisteredWindows_TmaKernelInvalid) {
  ncclTuningInput_t in = Input(ncclFuncAllReduce, size_t(1) << 30);
  in.winRegType = ncclSymSendNonregRecvNonreg;
  EXPECT_FALSE(Evaluate(in, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST).valid);
}

// The model's one-sided contract: AllGather needs only the receive window registered and
// ReduceScatter only the send window; AllReduce needs both. (RCCL never reaches these one-sided
// cases with symAligned16B set: ncclSymAligned16B requires both windows to be symmetric.)
TEST_F(SymModelTdmReachableTest, DefaultParam_OneSidedRegistration_TmaValidOnlyOnTheRequiredSide) {
  struct {
    ncclFunc_t func;
    ncclSymRegType_t reg;
    bool valid;
  } const cases[] = {
      {ncclFuncAllGather, ncclSymSendNonregRecvReg, true},     {ncclFuncAllGather, ncclSymSendRegRecvNonreg, false},
      {ncclFuncReduceScatter, ncclSymSendRegRecvNonreg, true}, {ncclFuncReduceScatter, ncclSymSendNonregRecvReg, false},
      {ncclFuncAllReduce, ncclSymSendNonregRecvReg, false},    {ncclFuncAllReduce, ncclSymSendRegRecvNonreg, false},
  };
  for (const auto& c : cases) {
    ncclTuningInput_t in = Input(c.func, size_t(256) << 20);
    in.winRegType = c.reg;
    EXPECT_EQ(Evaluate(in, TmaKernelFor(c.func)).valid != 0, c.valid) << FuncName(c.func) << " winRegType " << c.reg;
  }
}

TEST_F(SymModelTdmTest, AgHierCtas_SymCtasCannotExceedTheDefaultCap) {
  g_loadParam = ParamOverride({{"SYM_CTAS", 96}});
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_HierLsaST), ncclSymkAgHierMaxBlocks);
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_ST), ncclSymkMaxBlocks);
}

TEST_F(SymModelTdmTest, AgHierCtas_EnvSetsTheCountAboveTheDefaultCap) {
  g_loadParam = ParamOverride({{"SYM_AG_HIER_CTAS", 96}});
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_HierLsaST), 96);
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_ST), 0);
}

TEST_F(SymModelTdmTest, AgHierCtas_SymCtasReducesFromTheOverriddenCap) {
  g_loadParam = ParamOverride({{"SYM_AG_HIER_CTAS", 96}, {"SYM_CTAS", 40}});
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_HierLsaST), 40);
  g_loadParam = ParamOverride({{"SYM_AG_HIER_CTAS", 96}, {"SYM_CTAS", 128}});
  EXPECT_EQ(ncclSymkModelCtasEnvOverride(ncclSymkKernelId_AllGather_HierLsaST), 96);
}

// ---- Hierarchical AllGather: gfx1250 partition mode, the only place the mask offers it ----

// g_loadParam stub: SYM_CTAS reads as ncclSymkMaxBlocks; every other param keeps its default.
int64_t SymCtasAtMaxBlocks(const char* env, int64_t deftVal) {
  return std::string(env) == "SYM_CTAS" ? ncclSymkMaxBlocks : deftVal;
}

// g_loadParam stub: SYM_CTAS reads 100, above both hierarchical block caps.
int64_t SymCtas100(const char* env, int64_t deftVal) {
  return std::string(env) == "SYM_CTAS" ? 100 : deftVal;
}

// g_loadParam stub: SYM_CTAS reads 21, under the TDM block cap but not a multiple of the package split.
int64_t SymCtas21(const char* env, int64_t deftVal) {
  return std::string(env) == "SYM_CTAS" ? 21 : deftVal;
}

// Packages of two ranks, so the fixture's four ranks make two packages.
class SymModelTdmHierTest : public SymModelTdmReachableTest {
 protected:
  void SetUp() override {
    SymModelTdmReachableTest::SetUp();
    comm_->hasMloPart = true;
    comm_->symkState.kcomm.hierScaleInSize = 2;
  }

  // What NCCL_SYM_KERNEL=AllGather_HierLsaTmaST leaves to the model.
  ncclTuningResult_t EvaluateForcedHierTma(size_t nBytes) {
    comm_->tuningContext.forced[ncclFuncAllGather] = 1;
    return Evaluate(Input(ncclFuncAllGather, nBytes), ncclSymkKernelId_AllGather_HierLsaTmaST);
  }
};

TEST_F(SymModelTdmHierTest, ForcedHierTma_LaunchesItsBlockCapAtFullWidth) {
  ncclTuningResult_t r = EvaluateForcedHierTma(size_t(64) << 20);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, ncclSymkAgHierTmaMaxBlocks);
  EXPECT_EQ(r.nWarps, ncclSymkWarpsPerBlock);
}

// Three packages split 48 blocks as 9 for the copying group and 19 for each pushing group, leaving one idle.
TEST_F(SymModelTdmHierTest, ForcedHierTma_ThreePackages_LaunchesOnlyTheBlocksTheGroupsUse) {
  comm_->nRanks = 6;
  ncclTuningResult_t r = EvaluateForcedHierTma(size_t(64) << 20);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, 47);
}

// NCCL_SYM_CTAS replaces the model's block bounds, so it has to respect the same cap.
TEST_F(SymModelTdmHierTest, ForcedHierTma_SymCtasAboveTheCap_LaunchesTheCap) {
  ScopedHook loadParam(g_loadParam, SymCtasAtMaxBlocks);
  ncclTuningResult_t r = EvaluateForcedHierTma(size_t(64) << 20);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, ncclSymkAgHierTmaMaxBlocks);
}

// Each hierarchical kernel keeps its own cap under NCCL_SYM_CTAS.
TEST_F(SymModelTdmHierTest, SymCtasAboveBothCaps_EachKernelLaunchesItsOwnCap) {
  ScopedHook loadParam(g_loadParam, SymCtas100);
  comm_->tuningContext.forced[ncclFuncAllGather] = 1;
  ncclTuningResult_t tma = Evaluate(Input(ncclFuncAllGather, size_t(64) << 20), ncclSymkKernelId_AllGather_HierLsaTmaST);
  ASSERT_TRUE(tma.valid);
  EXPECT_EQ(tma.nChannels, ncclSymkAgHierTmaMaxBlocks);
  ncclTuningResult_t vec = Evaluate(Input(ncclFuncAllGather, size_t(64) << 20), ncclSymkKernelId_AllGather_HierLsaST);
  ASSERT_TRUE(vec.valid);
  EXPECT_EQ(vec.nChannels, ncclSymkAgHierMaxBlocks);
}

// 21 blocks on three packages split as 4 for the copying group and 8 for each pushing group, so 20 launch.
TEST_F(SymModelTdmHierTest, ForcedHierTma_SymCtasTrimmedByThePackageSplit) {
  ScopedHook loadParam(g_loadParam, SymCtas21);
  comm_->nRanks = 6;
  ncclTuningResult_t r = EvaluateForcedHierTma(size_t(64) << 20);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, 20);
}

// Sixteen ranks in packages of two make eight packages; the TDM kernel launches 3 + 7 * 6 = 45 blocks, the vector one 4 + 7 * 9 = 67.
TEST_F(SymModelTdmHierTest, EightPackages_BlockCounts) {
  ScopedHook loadParam(g_loadParam, SymCtas100);  // The vector kernel only reaches its cap when asked to.
  comm_->nRanks = 16;
  comm_->tuningContext.forced[ncclFuncAllGather] = 1;
  ncclTuningResult_t tma = EvaluateForcedHierTma(size_t(64) << 20);
  ASSERT_TRUE(tma.valid);
  EXPECT_EQ(tma.nChannels, 45);
  ncclTuningResult_t vec = Evaluate(Input(ncclFuncAllGather, size_t(64) << 20), ncclSymkKernelId_AllGather_HierLsaST);
  ASSERT_TRUE(vec.valid);
  EXPECT_EQ(vec.nChannels, 67);
}

// The by-size block count never undercuts NCCL_MIN_CTAS.
TEST_F(SymModelTdmHierTest, VectorKernel_HonorsMinCtas) {
  comm_->nRanks = 16;
  comm_->tuningContext.forced[ncclFuncAllGather] = 1;
  ncclTuningInput_t in = Input(ncclFuncAllGather, size_t(1) << 20);
  in.minCTAs = ncclSymkAgHierMaxBlocks;
  in.maxCTAs = MAXCHANNELS;
  ncclTuningResult_t r = Evaluate(in, ncclSymkKernelId_AllGather_HierLsaST);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, ncclSymkAgHierLaunchBlocks(comm_.get(), ncclSymkAgHierMaxBlocks));
}

// The vector kernels hold every size range, so neither TDM kernel runs unless asked for.
TEST_F(SymModelTdmHierTest, DefaultParam_AllGatherStaysOnVectorKernels) {
  for (size_t nBytes : {size_t(16) << 10, size_t(1) << 20, size_t(64) << 20}) {
    Pick p = Select(Input(ncclFuncAllGather, nBytes));
    ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << nBytes;
    EXPECT_FALSE(rcclSymkKernelIdIsTma(p.kernelId)) << nBytes << " B picked " << ncclSymkKernelIdToString(p.kernelId);
  }
}

// NCCL_SYM_TMA_ENABLE=2 leaves only the TDM kernels, which take the size ranges of their vector twins.
TEST_F(SymModelTdmHierTest, TmaForced_TdmKernelsSplitAllGatherLikeTheirVectorTwins) {
  ScopedHook loadParam(g_loadParam, SymTmaEnable(2));
  struct {
    size_t nBytes;  // per rank; the thresholds count bus bytes, four times this
    ncclSymkKernelId expected;
  } const cases[] = {
      {16 << 10, ncclSymkKernelId_AllGather_TmaST},  // LL's range, and there is no TDM LL
      {1 << 20, ncclSymkKernelId_AllGather_TmaST},
      {size_t(64) << 20, ncclSymkKernelId_AllGather_HierLsaTmaST},
  };
  for (const auto& c : cases) {
    Pick p = Select(Input(ncclFuncAllGather, c.nBytes));
    EXPECT_EQ(p.kernelId, c.expected) << c.nBytes << " B picked " << ncclSymkKernelIdToString(p.kernelId);
  }
}

// An algorithm selection naming only the TDM hierarchical kernel reaches it without NCCL_SYM_KERNEL.
TEST_F(SymModelTdmHierTest, SelectionOfOnlyHierTma_KeepsItValid) {
  ncclTuningInput_t in = Input(ncclFuncAllGather, size_t(1) << 20);
  in.tuningMask = uint64_t(1) << (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AllGather_HierLsaTmaST);
  EXPECT_TRUE(Evaluate(in, ncclSymkKernelId_AllGather_HierLsaTmaST).valid);
}

}  // namespace
