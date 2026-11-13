/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for ncclPatEnable and the PAT model in src/tuning/pat.cc.

#include <gtest/gtest.h>

#include <cfloat>
#include <memory>
#include <vector>

#include "comm.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include PAT_CC_PATH

namespace {

class PatMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    comm_ = std::make_unique<ncclComm>();
    // Four nodes, one rank each; the PAT graph has 2 channels.
    comm_->nNodes = 4;
    comm_->nRanks = 4;
    comm_->isOneRPN = 1;
    comm_->netDeviceType = NCCL_NET_DEVICE_HOST;
    ncclTopoGraph& graph = comm_->graphs[NCCL_ALGO_PAT];
    graph.nChannels = 2;
    graph.bwIntra = 10.0f;
    graph.bwInter = 3.0f;
    graph.latencyInter = 2.0f;
    ResetEnabled();
  }

  void TearDown() override { ResetNcclFakes(); }

  void ResetEnabled() {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) enabled_[f] = 1;
  }

  void SetPatEnable(int64_t value) { g_loadParam = ParamOverride({{"PAT_ENABLE", value}}); }

  // Two nodes of two ranks each, with NVLS and one head per local rank.
  void SetMultiRpn() {
    comm_->isOneRPN = 0;
    comm_->nNodes = 2;
    comm_->nRanks = 4;
    comm_->nvlsSupport = 1;
    comm_->channels[0].nvls.nHeads = 2;
    rankToNode_ = {0, 0, 1, 1};
    nodeRanks_.assign(2, ncclNodeRanks{2, nullptr});
    comm_->rankToNode = rankToNode_.data();
    comm_->nodeRanks = nodeRanks_.data();
  }

  ncclResult_t Init(int proto) {
    return ncclTuningPatModelInit(comm_.get(), NCCL_ALGO_PAT * NCCL_NUM_PROTOCOLS + proto, enabled_);
  }
  float Bw(int func) { return comm_->tuningContext.generalBandwidths[func][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE]; }
  float Lat(int func) { return comm_->tuningContext.generalLatencies[func][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE]; }

  ncclTuningResult_t Sim(ncclFunc_t func, ncclDevRedOp_t op = ncclDevSum, ncclDataType_t type = ncclFloat32) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = func;
    input.nBytes = 1 << 20;
    input.numPipeOps = 1;
    input.devRedOp = op;
    input.datatype = type;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.algo = NCCL_ALGO_PAT;
    result.proto = NCCL_PROTO_SIMPLE;
    result.valid = 1;
    EXPECT_EQ(ncclSuccess, ncclTuningPatModelSim(&input, &result));
    return result;
  }

  std::unique_ptr<ncclComm> comm_;
  std::vector<int> rankToNode_;
  std::vector<ncclNodeRanks> nodeRanks_;
  int enabled_[NCCL_NUM_FUNCTIONS];
};

TEST_F(PatMicrotest, EnableIsOffByDefault) {
  EXPECT_EQ(0, ncclPatEnable(comm_.get()));
}

TEST_F(PatMicrotest, EnableHonorsForcedPat) {
  comm_->forcePatEnable = true;
  EXPECT_EQ(1, ncclPatEnable(comm_.get()));
}

TEST_F(PatMicrotest, EnableOneIsUnconditional) {
  SetPatEnable(1);
  comm_->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  EXPECT_EQ(1, ncclPatEnable(comm_.get()));
}

TEST_F(PatMicrotest, EnableTwoNeedsHostNetDevice) {
  SetPatEnable(2);
  EXPECT_EQ(1, ncclPatEnable(comm_.get()));
  comm_->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  EXPECT_EQ(0, ncclPatEnable(comm_.get()));
}

TEST_F(PatMicrotest, EnableTwoMultiRpnNeedsSeveralNodesAndNvls) {
  SetPatEnable(2);
  SetMultiRpn();
  EXPECT_EQ(1, ncclPatEnable(comm_.get()));
  comm_->config.nvlsHostMode = ncclNvlsHostModeDisableTransport;
  EXPECT_EQ(0, ncclPatEnable(comm_.get()));
  comm_->config.nvlsHostMode = 0;
  comm_->nNodes = 1;
  EXPECT_EQ(0, ncclPatEnable(comm_.get()));
}

TEST_F(PatMicrotest, InitDisablesEverythingWhenPatIsOff) {
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) EXPECT_EQ(0, enabled_[f]) << f;
}

TEST_F(PatMicrotest, InitDisablesEverythingOffSimple) {
  SetPatEnable(1);
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_LL));
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) EXPECT_EQ(0, enabled_[f]) << f;
}

TEST_F(PatMicrotest, InitRejectsNonAlgoProtoIds) {
  SetPatEnable(1);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningPatModelInit(comm_.get(), -1, enabled_));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningPatModelInit(comm_.get(), NCCL_TUNING_COUNT, enabled_));
  // The first symmetric-kernel id.
  EXPECT_EQ(ncclInvalidUsage, ncclTuningPatModelInit(comm_.get(), NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, enabled_));
}

TEST_F(PatMicrotest, InitOneRpnModelsOnlyAllGatherAndReduceScatter) {
  SetPatEnable(1);
  auto& k = comm_->tuningContext.tuningConstants;
  k.baseLatencies[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 1.0;
  k.hwLatencies[NCCL_HW_NET][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 5.0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  for (int f : {ncclFuncAllGather, ncclFuncReduceScatter}) {
    EXPECT_EQ(1, enabled_[f]) << f;
    // 75% of min(bwInter 3, bwIntra 10) * 2 channels.
    EXPECT_NEAR(4.5f, Bw(f), 1e-4) << f;
    // base 1 + log2(4) * (5 + latencyInter 2) / 3.5 + 4 ranks * 2.8.
    EXPECT_NEAR(1.0f + 2 * 7.0f / 3.5f + 11.2f, Lat(f), 1e-4) << f;
  }
  for (int f : {ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllReduce}) {
    EXPECT_EQ(0, enabled_[f]) << f;
    EXPECT_EQ(-1.0f, Bw(f)) << f;
    EXPECT_EQ(-1.0f, Lat(f)) << f;
  }
}

TEST_F(PatMicrotest, InitDisablesAGraphWithoutBandwidth) {
  SetPatEnable(1);
  comm_->graphs[NCCL_ALGO_PAT].nChannels = 0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_EQ(0, enabled_[ncclFuncAllGather]);
  EXPECT_EQ(-1.0f, Bw(ncclFuncAllGather));
}

TEST_F(PatMicrotest, InitMultiRpnNeedsNvlsAndOneHeadPerLocalRank) {
  SetPatEnable(1);
  SetMultiRpn();
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_EQ(1, enabled_[ncclFuncAllGather]);
  EXPECT_NEAR(4.5f, Bw(ncclFuncAllGather), 1e-4);

  ResetEnabled();
  comm_->channels[0].nvls.nHeads = 1;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_EQ(0, enabled_[ncclFuncAllGather]);
  EXPECT_EQ(-1.0f, Bw(ncclFuncAllGather));

  ResetEnabled();
  comm_->channels[0].nvls.nHeads = 2;
  comm_->nvlsSupport = 0;
  ASSERT_EQ(ncclSuccess, Init(NCCL_PROTO_SIMPLE));
  EXPECT_EQ(0, enabled_[ncclFuncAllGather]);
  EXPECT_EQ(-1.0f, Bw(ncclFuncAllGather));
}

TEST_F(PatMicrotest, SimIsInvalidWithoutBandwidth) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllGather][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = -1.0f;
  EXPECT_EQ(0, Sim(ncclFuncAllGather).valid);
}

TEST_F(PatMicrotest, SimOneRpnTimesLatencyPlusTransfer) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllGather][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 2.0f;
  comm_->tuningContext.generalLatencies[ncclFuncAllGather][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 3.0f;
  ncclTuningResult_t result = Sim(ncclFuncAllGather);
  EXPECT_EQ(1, result.valid);
  EXPECT_NEAR(3.0f + (1 << 20) / 2000.0f, result.timeUs, 1e-3);
}

TEST_F(PatMicrotest, SimMultiRpnIsValidButNeverPreferred) {
  SetMultiRpn();
  comm_->tuningContext.generalBandwidths[ncclFuncAllGather][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 2.0f;
  ncclTuningResult_t result = Sim(ncclFuncAllGather);
  EXPECT_EQ(1, result.valid);
  EXPECT_EQ(FLT_MAX / 2.0f, result.timeUs);
}

TEST_F(PatMicrotest, SimMultiRpnReduceScatterNeedsAnNvlsReduction) {
  SetMultiRpn();
  comm_->tuningContext.generalBandwidths[ncclFuncReduceScatter][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 2.0f;
  EXPECT_EQ(0, Sim(ncclFuncReduceScatter, ncclDevProd).valid);
  EXPECT_EQ(1, Sim(ncclFuncReduceScatter, ncclDevSum).valid);
}

}  // namespace
