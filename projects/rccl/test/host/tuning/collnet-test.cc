/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the CollNet model in src/tuning/collnet.cc.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "comm.h"
#include "fakes/nccl_fakes.h"
#include "fakes/topo_stubs.h"
#include "ScopedHook.h"

#include COLLNET_CC_PATH

namespace {

class CollnetMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetTopoStubs();
    comm_ = std::make_unique<ncclComm>();
    // The model only checks these two for null.
    collNet_.iallgather = [](auto...) { return ncclInternalError; };
    collNet_.ireducescatter = [](auto...) { return ncclInternalError; };
    comm_->ncclCollNet = &collNet_;
    comm_->config.collnetEnable = 1;
    comm_->collNetChainSupport = 1;
    // One node of four ranks, Ampere. Both CollNet graphs have 2 channels over PATH_NVL.
    comm_->nNodes = 1;
    comm_->nRanks = 4;
    comm_->maxLocalRanks = 4;
    comm_->minCompCap = 80;
    for (int a : {NCCL_ALGO_COLLNET_DIRECT, NCCL_ALGO_COLLNET_CHAIN}) {
      ncclTopoGraph& graph = comm_->graphs[a];
      graph.nChannels = 2;
      graph.bwIntra = 10.0f;
      graph.bwInter = 4.0f;
      graph.typeIntra = PATH_NVL;
      graph.latencyInter = 0.5f;
      k().baseLatencies[a][NCCL_PROTO_SIMPLE] = 2.0;
      k().hwLatencies[NCCL_HW_NVLINK][a][NCCL_PROTO_SIMPLE] = 1.0;
    }
    ResetEnabled();
  }

  void TearDown() override {
    ResetTopoStubs();
    ResetNcclFakes();
  }

  ncclTunerConstants_t& k() { return comm_->tuningContext.tuningConstants; }
  void ResetEnabled() {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) enabled_[f] = 1;
  }
  bool NoneEnabled() {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      if (enabled_[f]) return false;
    }
    return true;
  }

  // Two nodes of four ranks, with `heads` CollNet heads.
  void SetTwoNodes(int heads) {
    comm_->nNodes = 2;
    comm_->nRanks = 8;
    comm_->collNetHeadsNum = heads;
    rankToNode_ = {0, 0, 0, 0, 1, 1, 1, 1};
    nodeRanks_.assign(2, ncclNodeRanks{4, nullptr});
    comm_->rankToNode = rankToNode_.data();
    comm_->nodeRanks = nodeRanks_.data();
  }

  ncclResult_t Init(int algo) {
    return ncclTuningCollnetModelInit(comm_.get(), algo * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE, enabled_);
  }
  float Bw(int func, int algo) { return comm_->tuningContext.generalBandwidths[func][algo][NCCL_PROTO_SIMPLE]; }
  float Lat(int func, int algo) { return comm_->tuningContext.generalLatencies[func][algo][NCCL_PROTO_SIMPLE]; }

  ncclTuningResult_t Sim(int collNetSupport) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = ncclFuncAllReduce;
    input.nBytes = 1 << 20;
    input.numPipeOps = 1;
    input.collNetSupport = collNetSupport;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.algo = NCCL_ALGO_COLLNET_DIRECT;
    result.proto = NCCL_PROTO_SIMPLE;
    result.valid = 1;
    result.timeUs = 0.0f;
    EXPECT_EQ(ncclSuccess, ncclTuningCollnetModelSim(&input, &result));
    return result;
  }

  ncclCollNet_t collNet_{};
  std::unique_ptr<ncclComm> comm_;
  std::vector<int> rankToNode_;
  std::vector<ncclNodeRanks> nodeRanks_;
  int enabled_[NCCL_NUM_FUNCTIONS];
};

TEST_F(CollnetMicrotest, InitDisablesEverythingWithoutACollNetPlugin) {
  comm_->ncclCollNet = nullptr;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_TRUE(NoneEnabled());
}

TEST_F(CollnetMicrotest, InitRejectsNonAlgoProtoIds) {
  EXPECT_EQ(ncclInvalidUsage, ncclTuningCollnetModelInit(comm_.get(), -1, enabled_));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningCollnetModelInit(comm_.get(), NCCL_TUNING_COUNT, enabled_));
  // The first symmetric-kernel id.
  EXPECT_EQ(ncclInvalidUsage,
            ncclTuningCollnetModelInit(comm_.get(), NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, enabled_));
}

TEST_F(CollnetMicrotest, InitWithoutNvSwitchDisablesChainAndMultiRankDirect) {
  comm_->config.collnetEnable = 0;
  int nvs = 0;
  ScopedHook nvsCount(g_ncclTopoGetNvsCount, [&nvs](struct ncclTopoSystem*, int* count) {
    *count = nvs;
    return ncclSuccess;
  });
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_CHAIN));
  EXPECT_TRUE(NoneEnabled());
  ResetEnabled();
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_TRUE(NoneEnabled());
  // Direct with one local rank, or either algorithm with an NVSwitch, is still modelled.
  ResetEnabled();
  comm_->maxLocalRanks = 1;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_EQ(1, enabled_[ncclFuncAllReduce]);
  ResetEnabled();
  nvs = 1;
  comm_->maxLocalRanks = 4;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_CHAIN));
  EXPECT_EQ(1, enabled_[ncclFuncAllReduce]);
}

TEST_F(CollnetMicrotest, InitDisablesChainWithoutChainSupport) {
  comm_->collNetChainSupport = 0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_CHAIN));
  EXPECT_TRUE(NoneEnabled());
}

TEST_F(CollnetMicrotest, InitModelsOnlyAllGatherReduceScatterAndAllReduce) {
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  for (int f : {ncclFuncBroadcast, ncclFuncReduce}) {
    EXPECT_EQ(0, enabled_[f]) << f;
    EXPECT_EQ(-1.0f, Bw(f, NCCL_ALGO_COLLNET_DIRECT)) << f;
    EXPECT_EQ(-1.0f, Lat(f, NCCL_ALGO_COLLNET_DIRECT)) << f;
  }
  for (int f : {ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce}) EXPECT_EQ(1, enabled_[f]) << f;
}

TEST_F(CollnetMicrotest, InitSingleNodeDirectBandwidthAndLatency) {
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  // AllGather / ReduceScatter: 4 ranks per node * min(bwIntra 10, 90% of bwInter 4).
  EXPECT_NEAR(14.4f, Bw(ncclFuncAllGather, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
  EXPECT_NEAR(14.4f, Bw(ncclFuncReduceScatter, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
  // AllReduce: 2 channels * 10, divided by the GPU/NIC factor 2 - (2 - 1) / 2 = 1.5, then halved.
  EXPECT_NEAR(20.0f / 1.5f / 2, Bw(ncclFuncAllReduce, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
  // base 2 + 2 * (1 us + 3 * 0.4 us arity) + inter (1 + 2 * latencyInter 0.5).
  EXPECT_NEAR(8.4f, Lat(ncclFuncAllReduce, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
}

TEST_F(CollnetMicrotest, InitDirectAllReduceLosesFifteenPercentFromHopper) {
  comm_->minCompCap = 90;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_NEAR(20.0f / 1.5f * 0.85f / 2, Bw(ncclFuncAllReduce, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
}

TEST_F(CollnetMicrotest, InitSingleNodeChainBandwidthAndLatency) {
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_CHAIN));
  EXPECT_NEAR(10.0f, Bw(ncclFuncAllReduce, NCCL_ALGO_COLLNET_CHAIN), 1e-4);
  // base 2 + 2 * 3 local hops * 1 us + inter 2.
  EXPECT_NEAR(10.0f, Lat(ncclFuncAllReduce, NCCL_ALGO_COLLNET_CHAIN), 1e-4);
}

TEST_F(CollnetMicrotest, InitMultiNodeNeedsEnoughHeads) {
  SetTwoNodes(4);
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_EQ(1, enabled_[ncclFuncAllReduce]);
  EXPECT_NEAR(20.0f / 1.5f / 2, Bw(ncclFuncAllReduce, NCCL_ALGO_COLLNET_DIRECT), 1e-4);
  ResetEnabled();
  comm_->collNetHeadsNum = 3;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  for (int f : {ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce}) EXPECT_EQ(0, enabled_[f]) << f;
}

TEST_F(CollnetMicrotest, InitMultiNodeNeedsThePluginCollective) {
  SetTwoNodes(4);
  collNet_.iallgather = nullptr;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_EQ(0, enabled_[ncclFuncAllGather]);
  EXPECT_EQ(-1.0f, Bw(ncclFuncAllGather, NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_EQ(1, enabled_[ncclFuncReduceScatter]);
  ResetEnabled();
  collNet_.ireducescatter = nullptr;
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_EQ(0, enabled_[ncclFuncReduceScatter]);
  EXPECT_EQ(-1.0f, Bw(ncclFuncReduceScatter, NCCL_ALGO_COLLNET_DIRECT));
}

TEST_F(CollnetMicrotest, InitMultiNodeNeedsCollnetEnabled) {
  SetTwoNodes(4);
  comm_->config.collnetEnable = 0;
  comm_->maxLocalRanks = 1;
  ScopedHook nvsCount(g_ncclTopoGetNvsCount, [](struct ncclTopoSystem*, int* count) {
    *count = 0;
    return ncclSuccess;
  });
  ASSERT_EQ(ncclSuccess, Init(NCCL_ALGO_COLLNET_DIRECT));
  for (int f : {ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce}) {
    EXPECT_EQ(0, enabled_[f]) << f;
    EXPECT_EQ(-1.0f, Bw(f, NCCL_ALGO_COLLNET_DIRECT)) << f;
  }
}

TEST_F(CollnetMicrotest, SimIsInvalidWithoutCollNetSupport) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 2.0f;
  ncclTuningResult_t result = Sim(0);
  EXPECT_EQ(0, result.valid);
  EXPECT_EQ(-1.0f, result.timeUs);
}

TEST_F(CollnetMicrotest, SimIsInvalidWithoutBandwidth) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = -1.0f;
  ncclTuningResult_t result = Sim(1);
  EXPECT_EQ(0, result.valid);
  EXPECT_EQ(-1.0f, result.timeUs);
}

TEST_F(CollnetMicrotest, SimSupportsAtMostEightLocalRanks) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 2.0f;
  comm_->tuningContext.generalLatencies[ncclFuncAllReduce][NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 3.0f;
  comm_->maxLocalRanks = NCCL_MAX_DIRECT_ARITY + 1;
  ncclTuningResult_t result = Sim(1);
  EXPECT_EQ(1, result.valid);
  EXPECT_NEAR(3.0f + (1 << 20) / 2000.0f, result.timeUs, 1e-3);
  comm_->maxLocalRanks = NCCL_MAX_DIRECT_ARITY + 2;
  result = Sim(1);
  EXPECT_EQ(0, result.valid);
  EXPECT_EQ(-1.0f, result.timeUs);
}

}  // namespace
