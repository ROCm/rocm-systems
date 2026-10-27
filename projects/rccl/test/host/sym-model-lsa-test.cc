/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the LSA CTA selection in src/tuning/sym_model/lsa.cc.

#include <gtest/gtest.h>

#include <memory>

#include "comm.h"
#include "core.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"
#include "tuning.h"

// sym_kernels.cc classifies AllGather_ST, the one kernel driven here, as AG only: not LL, AR, RS or Tma.
int LsaMicroAGKernelMask() { return 1 << ncclSymkKernelId_AllGather_ST; }
int LsaMicroNoKernelMask() { return 0; }
bool LsaMicroTmaAvailable(struct ncclComm*) { return false; }
bool LsaMicroTmaForced(struct ncclComm*) { return false; }
bool LsaMicroTmaDeepEligible(struct ncclComm*, ncclSymkKernelId, size_t, int) { return false; }
const char* LsaMicroKernelIdToString(int) { return "micro-sym-kernel"; }
int LsaMicroCtasEnvOverride() { return 0; }  // sym_model.cc with NCCL_SYM_CTAS unset

// Redirect the sym_kernels.cc and sym_model.cc dependencies. Keep all declaring headers before this block.
#define ncclSymkAGKernelMask LsaMicroAGKernelMask
#define ncclSymkARKernelMask LsaMicroNoKernelMask
#define ncclSymkRSKernelMask LsaMicroNoKernelMask
#define ncclSymkLLKernelMask LsaMicroNoKernelMask
#define ncclSymkTmaKernelMask LsaMicroNoKernelMask
#define ncclSymkTmaAvailable LsaMicroTmaAvailable
#define ncclSymkTmaForced LsaMicroTmaForced
#define ncclSymkTmaDeepEligible LsaMicroTmaDeepEligible
#define ncclSymkKernelIdToString LsaMicroKernelIdToString
#define ncclSymkModelCtasEnvOverride LsaMicroCtasEnvOverride

#include SYM_MODEL_LSA_A2A_CC_PATH
#include SYM_MODEL_LSA_BASE_CC_PATH
#include SYM_MODEL_LSA_CC_PATH

#undef ncclSymkModelCtasEnvOverride
#undef ncclSymkKernelIdToString
#undef ncclSymkTmaDeepEligible
#undef ncclSymkTmaForced
#undef ncclSymkTmaAvailable
#undef ncclSymkTmaKernelMask
#undef ncclSymkLLKernelMask
#undef ncclSymkRSKernelMask
#undef ncclSymkARKernelMask
#undef ncclSymkAGKernelMask

namespace {

constexpr ncclSymkKernelId kKernel = ncclSymkKernelId_AllGather_ST;

class SymModelLsaMicrotest : public ::testing::Test {
 protected:
  // Nothing here installs a hook, so resetting on entry is enough.
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 8;
  }

  ncclTuningInput_t MakeInput(size_t nBytes, int minCTAs, int maxCTAs) const {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = ncclFuncAllGather;
    input.datatype = ncclInt8;
    input.nBytes = input.count = input.countMax = nBytes;
    input.nWorks = 1;
    input.minCTAs = minCTAs;
    input.maxCTAs = maxCTAs;
    return input;
  }

  // The estimate CTA selection compares: the model at one CTA count, without the per-CTA resource penalty.
  static double CtaTimeUs(ncclTuningInput_t* input, int nBlocks) {
    ncclSymkLsaEstimate estimate{};
    EXPECT_TRUE(ncclSymkLsaBaseModel(input, kKernel, input->nBytes, nBlocks, &estimate));
    return estimate.ctaSelectionTimeUs;
  }

  std::unique_ptr<ncclComm> comm_;
};

// The legacy rule the refactor keeps: the fewest candidate CTAs (1, then even counts) whose unpenalized estimate is
// within withinPeakFactor of the widest count's. The resource penalty reaches only selectionTimeUs.
TEST_F(SymModelLsaMicrotest, SelectsFewestCtasWithinPeakOfUnpenalizedEstimate) {
  const double withinPeak = rcclSymkLsaWithinPeakFactor(kKernel);
  int narrowed = 0;
  for (size_t nBytes = size_t(1) << 10; nBytes <= (size_t(4) << 20); nBytes *= 4) {
    SCOPED_TRACE(nBytes);
    ncclTuningInput_t input = MakeInput(nBytes, 1, ncclSymkMaxBlocks);
    const double widest = CtaTimeUs(&input, ncclSymkMaxBlocks);
    ASSERT_LE(widest, CtaTimeUs(&input, 1)) << "a per-CTA penalty in the compared estimate makes widening look slower";
    int expected = ncclSymkMaxBlocks;
    for (int candidate = 1; candidate < ncclSymkMaxBlocks; candidate += candidate == 1 ? 1 : 2) {
      if (CtaTimeUs(&input, candidate) <= withinPeak * static_cast<float>(widest)) {
        expected = candidate;
        break;
      }
    }
    float timeUs = 0.0f, selectionTimeUs = 0.0f;
    int nBlocks = 0;
    ASSERT_EQ(ncclSuccess, ncclSymkLsaModel(&input, kKernel, nBytes, &timeUs, &selectionTimeUs, &nBlocks));
    EXPECT_EQ(expected, nBlocks);
    EXPECT_FLOAT_EQ(static_cast<float>(CtaTimeUs(&input, nBlocks)), timeUs);
    EXPECT_FLOAT_EQ(timeUs * (1.0f + 0.025f * nBlocks), selectionTimeUs);
    narrowed += expected < ncclSymkMaxBlocks;
  }
  EXPECT_GT(narrowed, 0) << "no size exercised the CTA search";
}

// Pinned bounds leave a single candidate, so no comparison runs, yet the estimate and its penalty are still reported.
TEST_F(SymModelLsaMicrotest, PinnedBoundsReportTheEstimateAtThatCount) {
  ncclTuningInput_t input = MakeInput(size_t(64) << 10, 6, 6);
  float timeUs = 0.0f, selectionTimeUs = 0.0f;
  int nBlocks = 0;
  ASSERT_EQ(ncclSuccess, ncclSymkLsaModel(&input, kKernel, input.nBytes, &timeUs, &selectionTimeUs, &nBlocks));
  EXPECT_EQ(6, nBlocks);
  EXPECT_FLOAT_EQ(static_cast<float>(CtaTimeUs(&input, 6)), timeUs);
  EXPECT_FLOAT_EQ(timeUs * (1.0f + 0.025f * 6), selectionTimeUs);
}

}  // namespace
