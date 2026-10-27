/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/cost_model.cc with its Ring and Tree models.

#include <gtest/gtest.h>

#include <memory>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// The other models are not under test; tuning_general.cc resolves from tuning-general-test.cc.
ncclResult_t CostModelMicroInitStub(struct ncclComm*, int, int*) { return ncclSuccess; }
ncclResult_t CostModelMicroSimStub(struct ncclTuningInput_t*, struct ncclTuningResult_t*) { return ncclSuccess; }
static const char* CostModelMicroFuncStr[NCCL_NUM_FUNCTIONS + 4] = {"Broadcast", "Reduce", "AllGather",
                                                                    "ReduceScatter", "AllReduce"};
static const char* CostModelMicroSymkKernelIdToString(int) { return "micro-sym-kernel"; }

// Keep all declaring headers before this block.
#define ncclTuningCollnetModelInit CostModelMicroInitStub
#define ncclTuningNvlsModelInit CostModelMicroInitStub
#define ncclTuningPatModelInit CostModelMicroInitStub
#define ncclTuningCollnetModelSim CostModelMicroSimStub
#define ncclTuningNvlsModelSim CostModelMicroSimStub
#define ncclTuningPatModelSim CostModelMicroSimStub
#define ncclTuningSymkModelSim CostModelMicroSimStub
#define ncclTuningCeModelSim CostModelMicroSimStub
#define ncclFuncStr CostModelMicroFuncStr
#define ncclSymkKernelIdToString CostModelMicroSymkKernelIdToString

#include TUNING_COST_MODEL_CC_PATH
#include TUNING_RING_CC_PATH
#include TUNING_TREE_CC_PATH

#undef ncclSymkKernelIdToString
#undef ncclFuncStr
#undef ncclTuningCeModelSim
#undef ncclTuningSymkModelSim
#undef ncclTuningPatModelSim
#undef ncclTuningNvlsModelSim
#undef ncclTuningCollnetModelSim
#undef ncclTuningPatModelInit
#undef ncclTuningNvlsModelInit
#undef ncclTuningCollnetModelInit

namespace {

class TuningCostModelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 8;
    comm_->nNodes = 1;
    comm_->maxTreePattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    // Links fast enough that the per-arch LL cap, not the graph, bounds LL bandwidth.
    for (int algo : {NCCL_ALGO_TREE, NCCL_ALGO_RING}) {
      comm_->graphs[algo].nChannels = 16;
      comm_->graphs[algo].bwIntra = comm_->graphs[algo].bwInter = 100.0f;
    }
  }

  // The per-comm cost model over the shipped defaults, which comm init installs before it runs.
  void InitCostModel(int compCap) {
    comm_->minCompCap = comm_->maxCompCap = compCap;
    comm_->tuningContext.tuningConstants = ncclTunerConstantsDefaults;
    ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  }

  float LlBandwidth(ncclFunc_t func, int algo) {
    return comm_->tuningContext.generalBandwidths[func][algo][NCCL_PROTO_LL];
  }

  std::unique_ptr<ncclComm> comm_;
};

// HIP reports compCap as gfx major*10+minor, so a gfx12 GPU (120) takes the Blackwell row.
TEST_F(TuningCostModelMicrotest, Gfx12SingleNodeLlUsesBlackwellCap) {
  InitCostModel(120);
  // Ring/LL bus bandwidth is the cap; Broadcast runs nRanks steps over nRanks ranks.
  EXPECT_FLOAT_EQ(2 * 141.2f, LlBandwidth(ncclFuncBroadcast, NCCL_ALGO_RING));
  // Tree/LL AllReduce reports half its capped bus bandwidth.
  EXPECT_FLOAT_EQ(141.2f, LlBandwidth(ncclFuncAllReduce, NCCL_ALGO_TREE));
}

// gfx942 (94) takes the Hopper row.
TEST_F(TuningCostModelMicrotest, Gfx942SingleNodeLlUsesHopperCap) {
  InitCostModel(94);
  EXPECT_FLOAT_EQ(141.0f, LlBandwidth(ncclFuncBroadcast, NCCL_ALGO_RING));
  EXPECT_FLOAT_EQ(70.5f, LlBandwidth(ncclFuncAllReduce, NCCL_ALGO_TREE));
}

}  // namespace
