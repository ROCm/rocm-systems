/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the Tree model in src/tuning/tree.cc.

#include <gtest/gtest.h>

#include <memory>

#include "comm.h"
#include "fakes/nccl_fakes.h"

#include TREE_CC_PATH

// cost_model.cc owns the real table. Every entry here halves the bandwidth, except Simple's bucket 10, which
// quarters it, so a test sees whether a correction applied and from which bucket.
float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24] = {
  {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  0.5f,
   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  0.5f},
  {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  0.5f,
   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  0.5f},
  {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.5f,
   0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,  0.5f}};

namespace {

class TreeMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    comm_ = std::make_unique<ncclComm>();
    // One node, four ranks, Hopper; the Tree graph has 2 channels over PATH_NVL.
    comm_->nNodes = 1;
    comm_->nRanks = 4;
    comm_->minCompCap = 94;
    comm_->maxTreePattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    ncclTopoGraph& graph = comm_->graphs[NCCL_ALGO_TREE];
    graph.nChannels = 2;
    graph.bwIntra = 10.0f;
    graph.bwInter = 1.0f;
    graph.typeIntra = PATH_NVL;
    graph.latencyInter = 0.5f;
    // Caps high enough not to bind unless a test lowers them.
    for (int cc = 0; cc < NCCL_NUM_COMPCAPS; cc++) {
      for (int scale = 0; scale < NCCL_NUM_TUNING_SCALES; scale++) {
        k().llMaxBws[cc][scale] = 1000.0;
        k().perChMaxTreeBws[cc][scale] = 1000.0;
        k().perChMaxTreeLL128Bws[cc][scale] = 1000.0;
      }
    }
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) enabled_[f] = 1;
  }

  void TearDown() override { ResetNcclFakes(); }

  ncclTunerConstants_t& k() { return comm_->tuningContext.tuningConstants; }

  ncclResult_t Init(int proto) {
    return ncclTuningTreeModelInit(comm_.get(), NCCL_ALGO_TREE * NCCL_NUM_PROTOCOLS + proto, enabled_);
  }
  float Bw(int func, int proto) { return comm_->tuningContext.generalBandwidths[func][NCCL_ALGO_TREE][proto]; }
  float Lat(int func, int proto) { return comm_->tuningContext.generalLatencies[func][NCCL_ALGO_TREE][proto]; }

  ncclTuningResult_t Sim(ncclFunc_t func, size_t bytes) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = func;
    input.nBytes = bytes;
    input.numPipeOps = 1;
    input.datatype = ncclFloat32;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.algo = NCCL_ALGO_TREE;
    result.proto = NCCL_PROTO_SIMPLE;
    result.valid = 1;
    result.timeUs = 0.0f;
    EXPECT_EQ(ncclSuccess, ncclTuningTreeModelSim(&input, &result));
    return result;
  }

  std::unique_ptr<ncclComm> comm_;
  int enabled_[NCCL_NUM_FUNCTIONS];
};

TEST_F(TreeMicrotest, InitRejectsNonAlgoProtoIds) {
  EXPECT_EQ(ncclInvalidUsage, ncclTuningTreeModelInit(comm_.get(), -1, enabled_));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningTreeModelInit(comm_.get(), NCCL_TUNING_COUNT, enabled_));
  // The first symmetric-kernel id.
  EXPECT_EQ(ncclInvalidUsage, ncclTuningTreeModelInit(comm_.get(), NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, enabled_));
}

TEST_F(TreeMicrotest, InitDisablesEveryCollectiveButAllReduce) {
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    if (f == ncclFuncAllReduce) {
      EXPECT_EQ(1, enabled_[f]);
      EXPECT_GT(Bw(f, NCCL_PROTO_SIMPLE), 0.0f);
      continue;
    }
    EXPECT_EQ(0, enabled_[f]) << f;
    EXPECT_EQ(-1.0f, Bw(f, NCCL_PROTO_SIMPLE)) << f;
    EXPECT_EQ(-1.0f, Lat(f, NCCL_PROTO_SIMPLE)) << f;
  }
}

TEST_F(TreeMicrotest, InitSingleNodeSimpleBandwidthAndLatency) {
  k().baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 5.0;
  k().hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 1.0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  // busBw = 92% of bwIntra 10 * 2 channels, halved. Latency = base + 2 * (3 local hops * 1 us).
  EXPECT_NEAR(9.2f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(11.0f, Lat(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TreeMicrotest, InitCapsSimpleAtPerChannelMax) {
  int cc = ncclTuningGetCompCapIndex(comm_.get());
  k().perChMaxTreeBws[cc][0] = 5.0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(5.0f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TreeMicrotest, InitScalesAndCapsLlAndLl128) {
  int cc = ncclTuningGetCompCapIndex(comm_.get());
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL));
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL128));
  // LL divides busBw 18.4 by 3.8; single-node LL128 keeps 7/9 of it.
  EXPECT_NEAR(18.4f / 3.8f / 2, Bw(ncclFuncAllReduce, NCCL_PROTO_LL), 1e-4);
  EXPECT_NEAR(18.4f * 7 / 9 / 2, Bw(ncclFuncAllReduce, NCCL_PROTO_LL128), 1e-4);

  k().llMaxBws[cc][0] = 2.0;
  k().perChMaxTreeLL128Bws[cc][0] = 3.0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL));
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL128));
  EXPECT_NEAR(1.0f, Bw(ncclFuncAllReduce, NCCL_PROTO_LL), 1e-4);
  EXPECT_NEAR(3.0f, Bw(ncclFuncAllReduce, NCCL_PROTO_LL128), 1e-4);  // 2 channels * 3, halved
}

TEST_F(TreeMicrotest, InitMultiNodeLl128KeepsMoreBandwidthThanSingleNode) {
  comm_->nNodes = 2;
  comm_->nRanks = 8;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL128));
  EXPECT_NEAR(18.4f * 120 / 128 / 2, Bw(ncclFuncAllReduce, NCCL_PROTO_LL128), 1e-4);
}

TEST_F(TreeMicrotest, InitTreePatternLosesFifteenPercent) {
  comm_->maxTreePattern = NCCL_TOPO_PATTERN_TREE;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(9.2f * 0.85f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TreeMicrotest, InitBandwidthSourceDependsOnNodesAndComputeCapability) {
  // Below compute capability 100, up to two nodes use bwIntra and more use bwInter.
  comm_->nNodes = 2;
  comm_->nRanks = 8;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(9.2f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  comm_->nNodes = 4;
  comm_->nRanks = 16;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(0.92f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  // From 100 on, the slower of the two.
  comm_->nNodes = 2;
  comm_->nRanks = 8;
  comm_->minCompCap = 100;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(0.92f, Bw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TreeMicrotest, InitMultiNodeLatencyAddsLogNetworkHops) {
  comm_->nNodes = 4;
  comm_->nRanks = 16;
  k().baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 7.0;
  k().hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 1.0;
  k().hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 4.0;
  k().hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_LL] = 1.0;
  k().hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_LL] = 2.0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL));
  // Simple pays latencyInter twice per network hop: 4 + 2 * 0.5 = 5. Base 7 + 2 * (3 * 1 + log2(4) * 5) = 33.
  EXPECT_NEAR(33.0f, Lat(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  // LL pays it once: 2 + 0.5 = 2.5. 0 + 2 * (3 * 1 + 2 * 2.5) = 16.
  EXPECT_NEAR(16.0f, Lat(ncclFuncAllReduce, NCCL_PROTO_LL), 1e-4);
}

TEST_F(TreeMicrotest, SimIsInvalidForCollectivesOtherThanAllReduce) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllGather][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 10.0f;
  ncclTuningResult_t result = Sim(ncclFuncAllGather, 1 << 20);
  EXPECT_EQ(0, result.valid);
  EXPECT_EQ(-1.0f, result.timeUs);
}

TEST_F(TreeMicrotest, SimIsInvalidWithoutBandwidth) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = -1.0f;
  ncclTuningResult_t result = Sim(ncclFuncAllReduce, 1 << 20);
  EXPECT_EQ(0, result.valid);
  EXPECT_EQ(-1.0f, result.timeUs);
}

TEST_F(TreeMicrotest, SimScalesBandwidthByTheSizeCorrection) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 2.0f;
  comm_->tuningContext.generalLatencies[ncclFuncAllReduce][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 3.0f;
  // 64 << 10 bytes is log-size bucket 10.
  ncclTuningResult_t result = Sim(ncclFuncAllReduce, size_t(64) << 10);
  EXPECT_EQ(1, result.valid);
  EXPECT_NEAR(3.0f + 65536.0f / (1000 * 2.0f * 0.25f), result.timeUs, 1e-3);
}

TEST_F(TreeMicrotest, SimCorrectsOnlyBucketsBelow23) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 2.0f;
  comm_->tuningContext.generalLatencies[ncclFuncAllReduce][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 3.0f;
  // Bucket 23 has a table entry, but the model corrects only buckets 0-22.
  size_t bytes = size_t(64) << 23;
  ncclTuningResult_t result = Sim(ncclFuncAllReduce, bytes);
  EXPECT_NEAR(3.0f + bytes / (1000 * 2.0f), result.timeUs, 1e-1);
}

}  // namespace
