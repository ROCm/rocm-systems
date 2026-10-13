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
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/devcomm_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"
#include "tdm/tdmCopy.h"  // TDM_TOOLCHAIN_AVAILABLE
#include "tuning.h"

#include SYM_MODEL_CC_PATH
#include SYM_MODEL_GIN_CC_PATH
#include SYM_MODEL_LSA_CC_PATH
#include SYM_MODEL_LSA_BASE_CC_PATH
#include SYM_MODEL_LSA_A2A_CC_PATH

// gin.cc's RailA2A helpers name ncclTeamRail; this binary has no other definition of it.
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) {
  ncclTeam_t t{};
  t.nRanks = 1;
  return t;
}

namespace {

constexpr int kRanks = 4;
constexpr int kGfx1250CompCap = 125;  // HIP reports gfx1250 as compute capability 12.5
constexpr ncclDataType_t kTy = ncclFloat32;

std::function<int64_t(const char*, int64_t)> SymTmaEnable(int64_t value) {
  return [value](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? value : deftVal;
  };
}

bool TmaKernel(int id) { return (ncclSymkTmaKernelMask() >> id) & 1; }

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
    comm_->WarpSize = 32;
    SetArch("gfx1250", kGfx1250CompCap);
  }
  void TearDown() override {
    ResetDevcommFakes();
    ResetNcclFakes();
  }

  void SetArch(const char* arch, int compCap) {
    comm_->archName = const_cast<char*>(arch);
    comm_->compCap = comm_->minCompCap = comm_->maxCompCap = compCap;
  }

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

  static bool TmaReachableOnThisBuild() { return TDM_TOOLCHAIN_AVAILABLE; }

  std::unique_ptr<ncclComm> comm_;
};

// ---- Default enablement: NCCL_SYM_TMA_ENABLE left at its compile-time default ----

TEST_F(SymModelTdmTest, DefaultParam_Gfx1250_TmaAvailableButNotForced) {
  if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  EXPECT_TRUE(ncclSymkTmaAvailable(comm_.get()));
  EXPECT_FALSE(ncclSymkTmaForced(comm_.get())) << "the default must leave the choice to the cost model";
}

TEST_F(SymModelTdmTest, DefaultParam_NonTdmArch_TmaUnavailable) {
  for (const char* arch : {"gfx950", "gfx942", "gfx90a", "gfx1200"}) {
    SetArch(arch, 95);
    EXPECT_FALSE(ncclSymkTmaAvailable(comm_.get())) << arch;
  }
  comm_->archName = nullptr;
  EXPECT_FALSE(ncclSymkTmaAvailable(comm_.get())) << "unknown arch";
}

TEST_F(SymModelTdmTest, DefaultParam_Gfx950_LargeMessagesStayOnVectorKernels) {
  SetArch("gfx950", 95);
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    Pick p = Select(Input(func, size_t(256) << 20));
    ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << FuncName(func);
    EXPECT_FALSE(TmaKernel(p.kernelId)) << FuncName(func) << " picked " << ncclSymkKernelIdToString(p.kernelId);
  }
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
  EXPECT_EQ(TmaKernel(p.kernelId), c.expectTma) << "picked " << ncclSymkKernelIdToString(p.kernelId);
  if (c.expectTma) {
    EXPECT_EQ(p.kernelId, TmaKernelFor(c.func));
    // TDM throughput keeps scaling with resident blocks, and symCheckTmaLaunch() needs all 16 warps.
    EXPECT_EQ(p.nChannels, ncclSymkMaxBlocks);
    EXPECT_EQ(p.nWarps, ncclSymkWarpsPerBlock);
  }
}

// Endpoints only: the crossover between them is a tuning choice, not a contract.
INSTANTIATE_TEST_SUITE_P(Endpoints, SymModelTdmSelectionTest,
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
TEST_F(SymModelTdmTest, DefaultParam_BelowDeepLoopBar_TmaKernelInvalid) {
  if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    ncclSymkKernelId k = TmaKernelFor(func);
    size_t nBytes = 4 << 10;
    ASSERT_FALSE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, /*nBlocks=*/1)) << FuncName(func);
    EXPECT_FALSE(Evaluate(Input(func, nBytes), k).valid) << FuncName(func);
  }
}

// When nMaxBlocks would starve the deep loop the model trims blocks rather than dropping the kernel.
TEST_F(SymModelTdmTest, DefaultParam_JustAboveDeepLoopBar_LaunchesOnlyBlocksTheDeepLoopCanFeed) {
  if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  ncclSymkKernelId k = ncclSymkKernelId_ReduceScatter_TmaLD;
  size_t nBytes = size_t(ncclSymkDeepMaxBytePerChunk) * kRanks * 2;  // deep-eligible at 2 blocks, not at 4
  ASSERT_TRUE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, 2));
  ASSERT_FALSE(ncclSymkTmaDeepEligible(comm_.get(), k, nBytes, 4));
  ncclTuningResult_t r = Evaluate(Input(ncclFuncReduceScatter, nBytes), k);
  ASSERT_TRUE(r.valid);
  EXPECT_EQ(r.nChannels, 2);
}

// Large messages: the TDM terms (higher bandwidth for extra setup latency) must beat the vector kernel.
TEST_F(SymModelTdmTest, DefaultParam_LargeMessage_TmaCheaperThanVectorCounterpart) {
  if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
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

TEST_F(SymModelTdmTest, TmaParamDisabled_LargeMessagePicksVectorKernel) {
  ScopedHook loadParam(g_loadParam, SymTmaEnable(0));
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    Pick p = Select(Input(func, size_t(256) << 20));
    ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << FuncName(func);
    EXPECT_FALSE(TmaKernel(p.kernelId)) << FuncName(func) << " picked " << ncclSymkKernelIdToString(p.kernelId);
  }
}

TEST_F(SymModelTdmTest, DefaultParam_NotSymAligned16B_LargeMessagePicksVectorKernel) {
  for (ncclFunc_t func : {ncclFuncAllReduce, ncclFuncAllGather, ncclFuncReduceScatter}) {
    ncclTuningInput_t in = Input(func, size_t(256) << 20);
    in.symAligned16B = false;
    Pick p = Select(in);
    ASSERT_NE(p.kernelId, ncclSymkKernelId_Count) << FuncName(func);
    EXPECT_FALSE(TmaKernel(p.kernelId)) << FuncName(func) << " picked " << ncclSymkKernelIdToString(p.kernelId);
  }
}

// Deep-loop eligibility is per work (countMax), not per batch: many small grouped works must not reach Tma.
TEST_F(SymModelTdmTest, DefaultParam_GroupOfSmallWorks_TmaKernelInvalid) {
  if (!TmaReachableOnThisBuild()) GTEST_SKIP() << "SDK ships no TDM descriptor header";
  ncclTuningInput_t in = Input(ncclFuncAllReduce, size_t(256) << 20);
  in.countMax = (4 << 10) / ncclTypeSize(kTy);
  in.nWorks = 1024;
  EXPECT_FALSE(Evaluate(in, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST).valid);
}

// Without registered windows on both sides only LL kernels are valid, Tma included.
TEST_F(SymModelTdmTest, DefaultParam_UnregisteredWindows_TmaKernelInvalid) {
  ncclTuningInput_t in = Input(ncclFuncAllReduce, size_t(1) << 30);
  in.winRegType = ncclSymSendNonregRecvNonreg;
  EXPECT_FALSE(Evaluate(in, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST).valid);
}

}  // namespace
