/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the REAL src/sym_kernels.cc; sym-kernels-index-test.cc covers the GENERATED half.

#include <gtest/gtest.h>

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

// Neuter __global__: a real definition needs hipLaunchKernel/__hipRegisterFunction, unavailable under -no-hip-rt.
#undef __global__
#define __global__

#include "fakes/sym_kernels_test_stubs.h"

#include SYM_KERNELS_HOST_CC_PATH

#undef __global__

#include SYM_KERNELS_CC_PATH

namespace {

bool KernelBitSet(uint32_t mask, ncclSymkKernelId id) { return (mask >> static_cast<int>(id)) & 1u; }

class SymKernelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 4;
    comm_->symkState.hasLsaMultimem = false;
    for (int i = 0; i < ncclSymkKernelCount; ++i) {
      ncclSymkKernelMaxDynamicSmem[i] = 0;
    }
  }
  void TearDown() override {
    ResetDevcommFakes();
    ResetNcclFakes();
  }

  std::unique_ptr<ncclComm> comm_;
};

// ---- ncclSymkMaxChunkElts: kernelMask_AR|kernelMask_RS bit test (isReduce) + eltSize/accMult arithmetic ----

struct MaxChunkEltsCase {
  std::string name;
  ncclSymkKernelId id;
  int red;
  ncclDataType_t ty;
  int poisonSmem;
  int expectedElts;
};

std::ostream& operator<<(std::ostream& os, const MaxChunkEltsCase& c) { return os << c.name; }

class SymKernelMaxChunkEltsTest : public SymKernelMicrotest,
                                  public ::testing::WithParamInterface<MaxChunkEltsCase> {};

TEST_P(SymKernelMaxChunkEltsTest, ComputesExpectedChunkElts) {
  const MaxChunkEltsCase& c = GetParam();
  int index = ncclSymkGetKernelIndex(c.id, c.red, c.ty);
  ASSERT_GE(index, 0);
  ncclSymkKernelMaxDynamicSmem[index] = c.poisonSmem;
  EXPECT_EQ(ncclSymkMaxChunkElts(comm_.get(), c.id, c.red, c.ty), c.expectedElts);
}

const std::vector<MaxChunkEltsCase>& MaxChunkEltsCases() {
  static const std::vector<MaxChunkEltsCase> kCases = {
      // AllGather is not in kernelMask_AR|kernelMask_RS: accMult stays 1 no matter how small eltSize is.
      {"AllGatherLL_f8e4m3_NonReduceAccMultOne", ncclSymkKernelId_AllGather_LL, ncclDevSum, ncclFloat8e4m3, 100, 100},
      {"AllGatherST_f32_NonReduceEltSizeFour", ncclSymkKernelId_AllGather_ST, ncclDevSum, ncclFloat32, 400, 100},
      // AllReduce is in kernelMask_AR; f32's eltSize==4 pins the "< 4", not "<= 4", accMult boundary.
      {"AllReduceAGxLLR_f32_EltSizeFourBoundaryAccMultOne", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum,
       ncclFloat32, 800, 200},
      {"AllReduceAGxLLR_f16_EltSizeTwoAccMultTwo", ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSum, ncclFloat16, 800,
       200},
      {"AllReduceRSxLDAGxST_f8e4m3_EltSizeOneAccMultTwo", ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclDevSum,
       ncclFloat8e4m3, 600, 300},
      // ReduceScatter is in kernelMask_RS, not kernelMask_AR; the bf16 case below pins the OR.
      {"ReduceScatterLL_f32_EltSizeFourBoundaryAccMultOne", ncclSymkKernelId_ReduceScatter_LL, ncclDevSum,
       ncclFloat32, 400, 100},
      {"ReduceScatterLD_bf16_EltSizeTwoAccMultTwo", ncclSymkKernelId_ReduceScatter_LD, ncclDevSumPostDiv,
       ncclBfloat16, 800, 200},
  };
  return kCases;
}

INSTANTIATE_TEST_SUITE_P(SymAllChunkEltsCases, SymKernelMaxChunkEltsTest, ::testing::ValuesIn(MaxChunkEltsCases()),
                         [](const ::testing::TestParamInfo<MaxChunkEltsCase>& info) { return info.param.name; });

TEST_F(SymKernelMicrotest, MaxChunkElts_UnhandledCombination_ReturnsZeroRegardlessOfSmem) {
  ncclSymkKernelMaxDynamicSmem[0] = 12345;  // must be irrelevant: kernelIndex<0 short-circuits before it is read
  EXPECT_EQ(ncclSymkMaxChunkElts(comm_.get(), ncclSymkKernelId_AllReduce_AGxLL_R, ncclDevSumPostDiv, ncclFloat32), 0);
}

// ---- ncclSymkMask: kmask assembly (2GB/64GB thresholds, Tma gating, Gin gating; hasLsaMultimem false throughout) ----

class SymKernelMaskTest : public SymKernelMicrotest {
 protected:
  static constexpr ncclDataType_t kTy = ncclFloat32;

  // AllReduce's nBusBytes multiplier is 1, so nElts*4 (f32) controls the byte count exactly, cell-aligned.
  static size_t NEltsForBytes(size_t bytes) { return bytes / ncclTypeSize(kTy); }
};

TEST_F(SymKernelMaskTest, JustBelow2GB_LLKernelSurvives) {
  size_t nElts = NEltsForBytes((size_t(2) << 30) - NCCL_SYM_KERNEL_CELL_SIZE);
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, nElts);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_AGxLL_R));
}

TEST_F(SymKernelMaskTest, At2GB_LLKernelCleared) {
  size_t nElts = NEltsForBytes(size_t(2) << 30);
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, nElts);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_AGxLL_R));
}

TEST_F(SymKernelMaskTest, JustBelow64GB_NonLLKernelSurvives) {
  comm_->minCompCap = 100;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? 1 : 0;
  });
  size_t nElts = NEltsForBytes(32 * (size_t(2) << 30) - NCCL_SYM_KERNEL_CELL_SIZE);
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, nElts, /*symAligned16B=*/true);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST));
}

TEST_F(SymKernelMaskTest, At64GB_EntireMaskZeroed) {
  comm_->minCompCap = 100;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? 1 : 0;
  });
  size_t nElts = NEltsForBytes(32 * (size_t(2) << 30));
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, nElts, /*symAligned16B=*/true);
  EXPECT_EQ(kmask, 0u);
}

TEST_F(SymKernelMaskTest, TmaAvailable_CompCapAtBoundaryAndAligned_TmaKernelSurvives) {
  comm_->minCompCap = 100;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? 1 : 0;
  });
  uint32_t kmask =
      ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024, /*symAligned16B=*/true);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST));
}

TEST_F(SymKernelMaskTest, CompCapJustBelowBoundary_TmaKernelCleared) {
  comm_->minCompCap = 99;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? 1 : 0;
  });
  uint32_t kmask =
      ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024, /*symAligned16B=*/true);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST));
}

TEST_F(SymKernelMaskTest, TmaParamDisabled_TmaKernelCleared) {
  comm_->minCompCap = 100;
  ScopedHook loadParam(g_loadParam, [](const char*, int64_t) -> int64_t { return 0; });
  uint32_t kmask =
      ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024, /*symAligned16B=*/true);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST));
}

TEST_F(SymKernelMaskTest, NotSymAligned16B_TmaKernelCleared) {
  comm_->minCompCap = 100;
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_TMA_ENABLE" ? 1 : 0;
  });
  uint32_t kmask =
      ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024, /*symAligned16B=*/false);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST));
}

TEST_F(SymKernelMaskTest, NeedGinFalse_HasGinTrue_GinKernelClearedNonGinKernelSurvives) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t c) {
    ncclTeam_t t{};
    t.nRanks = c->nRanks;  // LSA spans the whole communicator: single-node, needGin false
    return t;
  });
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_GIN_KERNELS_ENABLE" ? 1 : 0;
  });
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD));
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

TEST_F(SymKernelMaskTest, NeedGinFalse_HasGinFalse_GinKernelClearedNonGinKernelSurvives) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t c) {
    ncclTeam_t t{};
    t.nRanks = c->nRanks;
    return t;
  });
  ScopedHook loadParam(g_loadParam, [](const char*, int64_t) -> int64_t { return 0; });
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD));
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

TEST_F(SymKernelMaskTest, NeedGinTrue_HasGinTrue_GinKernelSurvivesNonGinKernelCleared) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 2;  // LSA spans fewer ranks than the communicator: multi-node, needGin true
    return t;
  });
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t) -> int64_t {
    return std::string(env) == "SYM_GIN_KERNELS_ENABLE" ? 1 : 0;
  });
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD));
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

TEST_F(SymKernelMaskTest, NeedGinTrue_HasGinFalse_EverythingCleared) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 2;
    return t;
  });
  ScopedHook loadParam(g_loadParam, [](const char*, int64_t) -> int64_t { return 0; });
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD));
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

TEST_F(SymKernelMaskTest, HasLsaMultimemTrue_STMCKernelSurvives) {
  comm_->symkState.hasLsaMultimem = true;
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_AGxLLMC_R));
}

TEST_F(SymKernelMaskTest, HasLsaMultimemFalse_STMCKernelCleared) {
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, kTy, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_AGxLLMC_R));
}

TEST_F(SymKernelMaskTest, HasLDMC_ValidRedAndType_Survives) {
  comm_->symkState.hasLsaMultimem = true;
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat16, /*nElts=*/1024);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC));
}

TEST_F(SymKernelMaskTest, HasLDMC_UnsupportedRed_Cleared) {
  comm_->symkState.hasLsaMultimem = true;
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevProd, ncclFloat16, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC));
}

TEST_F(SymKernelMaskTest, HasLDMC_F8Type_CompCapBelowBoundary_Cleared) {
  comm_->symkState.hasLsaMultimem = true;
  comm_->compCap = 99;
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat8e4m3, /*nElts=*/1024);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC));
}

TEST_F(SymKernelMaskTest, HasLDMC_F8Type_CompCapAtBoundary_Survives) {
  comm_->symkState.hasLsaMultimem = true;
  comm_->compCap = 100;
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat8e4m3, /*nElts=*/1024);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC));
}

TEST_F(SymKernelMaskTest, ReduceScatterNRanksMultiplier_JustBelowPerRank2GB_LLKernelSurvives) {
  size_t nElts = NEltsForBytes((size_t(2) << 30) / comm_->nRanks - NCCL_SYM_KERNEL_CELL_SIZE);
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, nElts);
  EXPECT_TRUE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

TEST_F(SymKernelMaskTest, ReduceScatterNRanksMultiplier_AtPerRank2GB_LLKernelCleared) {
  size_t nElts = NEltsForBytes((size_t(2) << 30) / comm_->nRanks);
  uint32_t kmask = ncclSymkMask(comm_.get(), ncclFuncReduceScatter, ncclDevSum, kTy, nElts);
  EXPECT_FALSE(KernelBitSet(kmask, ncclSymkKernelId_ReduceScatter_LL));
}

}  // namespace
