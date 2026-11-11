/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/tuning_general.cc and the Ring model in src/tuning/ring.cc.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "ScopedHook.h"

#include TUNING_GENERAL_CC_PATH
#include RING_CC_PATH

namespace {

class TuningGeneralMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    comm_ = std::make_unique<ncclComm>();
  }

  void TearDown() override {
    ResetEnvFakes();
    ResetNcclFakes();
  }

  void SetChannelDefaults(int channels = 8, int threads = 256, int threshold = 64) {
    comm_->nChannels = channels;
    comm_->nvlsChannels = channels / 2;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
        comm_->tuningContext.maxThreads[a][p] = threads;
        comm_->tuningContext.threadThresholds[a][p] = threshold;
      }
    }
  }

  ncclTuningResult_t GetChannels(int algo, int proto, size_t bytes, int maxChannels = -1) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.nBytes = bytes;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.algo = algo;
    result.proto = proto;
    result.maxChannels = maxChannels;
    EXPECT_EQ(ncclSuccess, ncclTuningGetChannels(&input, &result));
    return result;
  }

  ncclTuningInput_t MakeTimeInput(size_t bytes, int pipeOps, ncclDataType_t datatype) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.nBytes = bytes;
    input.numPipeOps = pipeOps;
    input.datatype = datatype;
    return input;
  }

  // Ring/Simple time from ring.cc for a 10 us base latency and a bandwidth so high that the size term
  // vanishes: 10 us times the Ring plateau factor.
  float RingSimpleTimeUs(ncclFunc_t func, int nRanks, size_t bytesPerRankPerChannel) {
    comm_->nRanks = nRanks;
    comm_->nChannels = 4;
    comm_->tuningContext.generalLatencies[func][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 10.0f;
    comm_->tuningContext.generalBandwidths[func][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 1e12f;
    ncclTuningInput_t input = MakeTimeInput(bytesPerRankPerChannel * comm_->nChannels * nRanks, 1, ncclFloat32);
    input.func = func;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.algo = NCCL_ALGO_RING;
    result.proto = NCCL_PROTO_SIMPLE;
    EXPECT_EQ(ncclSuccess, ncclTuningRingModelSim(&input, &result));
    return result.timeUs;
  }

  // A Ring graph with 2 channels over PATH_NVL, bwIntra 10 and bwInter 1, for ncclTuningRingModelInit.
  void SetRingInitComm(int nNodes, int nRanks, int minCompCap) {
    comm_->nNodes = nNodes;
    comm_->nRanks = nRanks;
    comm_->minCompCap = minCompCap;
    ncclTopoGraph& graph = comm_->graphs[NCCL_ALGO_RING];
    graph.nChannels = 2;
    graph.bwIntra = 10.0f;
    graph.bwInter = 1.0f;
    graph.typeIntra = PATH_NVL;
    graph.sameChannels = 1;
  }

  ncclResult_t RingInit(int proto) {
    return ncclTuningRingModelInit(comm_.get(), NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + proto, nullptr);
  }
  float RingBw(ncclFunc_t func, int proto) {
    return comm_->tuningContext.generalBandwidths[func][NCCL_ALGO_RING][proto];
  }
  float RingLat(ncclFunc_t func, int proto) {
    return comm_->tuningContext.generalLatencies[func][NCCL_ALGO_RING][proto];
  }

  std::unique_ptr<ncclComm> comm_;
};

TEST_F(TuningGeneralMicrotest, StepCountsMatchCollectiveShape) {
  EXPECT_EQ(14, ncclTuningGetNsteps(ncclFuncAllReduce, 8));
  EXPECT_EQ(7, ncclTuningGetNsteps(ncclFuncReduceScatter, 8));
  EXPECT_EQ(7, ncclTuningGetNsteps(ncclFuncAllGather, 8));
  EXPECT_EQ(8, ncclTuningGetNsteps(ncclFuncBroadcast, 8));

  EXPECT_EQ(6, ncclTuningGetNsteps(ncclFuncAllReduce, 4));
  EXPECT_EQ(3, ncclTuningGetNsteps(ncclFuncReduceScatter, 4));
  EXPECT_EQ(3, ncclTuningGetNsteps(ncclFuncAllGather, 4));
  EXPECT_EQ(4, ncclTuningGetNsteps(ncclFuncBroadcast, 4));
}

TEST_F(TuningGeneralMicrotest, ComputeCapabilityBoundariesSelectExpectedTable) {
  for (const auto& entry : {std::pair{70, NCCL_VOLTA_COMPCAP_IDX}, std::pair{79, NCCL_VOLTA_COMPCAP_IDX},
                            std::pair{80, NCCL_AMPERE_COMPCAP_IDX}, std::pair{89, NCCL_AMPERE_COMPCAP_IDX},
                            std::pair{90, NCCL_HOPPER_COMPCAP_IDX}, std::pair{99, NCCL_HOPPER_COMPCAP_IDX},
                            std::pair{100, NCCL_BLACKWELL_COMPCAP_IDX}}) {
    comm_->minCompCap = entry.first;
    EXPECT_EQ(entry.second, ncclTuningGetCompCapIndex(comm_.get()));
  }
}

TEST_F(TuningGeneralMicrotest, ConstantsIndexesSeparateGpuCpuAndNodeCount) {
  int index1 = -1, index2 = -1;
  comm_->nNodes = 1;
  comm_->minCompCap = 90;
  ncclTuningGetConstantsIndexes(comm_.get(), &index1, &index2);
  EXPECT_EQ(NCCL_HOPPER_COMPCAP_IDX, index1);
  EXPECT_EQ(0, index2);

  comm_->nNodes = 2;
  comm_->cpuVendor = NCCL_TOPO_CPU_VENDOR_AMD;
  ncclTuningGetConstantsIndexes(comm_.get(), &index1, &index2);
  EXPECT_EQ(1, index1);
  EXPECT_EQ(1, index2);

  comm_->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
  ncclTuningGetConstantsIndexes(comm_.get(), &index1, &index2);
  EXPECT_EQ(1, index1);

  comm_->nNodes = 4;
  comm_->cpuVendor = NCCL_TOPO_CPU_VENDOR_INTEL;
  ncclTuningGetConstantsIndexes(comm_.get(), &index1, &index2);
  EXPECT_EQ(0, index1);
  EXPECT_EQ(2, index2);
}

TEST_F(TuningGeneralMicrotest, HardwareIndexesDistinguishLocalLinksAndNetwork) {
  int intra = -1, inter = -1;
  comm_->nNodes = 1;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, &intra, &inter);
  EXPECT_EQ(NCCL_HW_NVLINK, intra);
  EXPECT_EQ(NCCL_HW_NVLINK, inter);

  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_PIX;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, &intra, &inter);
  EXPECT_EQ(NCCL_HW_PCI, intra);
  EXPECT_EQ(NCCL_HW_PCI, inter);

  comm_->nNodes = 2;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, &intra, &inter);
  EXPECT_EQ(NCCL_HW_PCI, intra);
  EXPECT_EQ(NCCL_HW_NET, inter);

  intra = -1;
  inter = -1;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, &intra, nullptr);
  EXPECT_EQ(NCCL_HW_PCI, intra);
  EXPECT_EQ(-1, inter);

  intra = -1;
  inter = -1;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, nullptr, &inter);
  EXPECT_EQ(-1, intra);
  EXPECT_EQ(NCCL_HW_NET, inter);

  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, nullptr, nullptr);
}

TEST_F(TuningGeneralMicrotest, TimeEstimateCombinesLatencyAndBandwidth) {
  ncclTuningInput_t input = MakeTimeInput(8000, 3, ncclFloat32);
  float latency = 2.0f, bandwidth = 4.0f;
  EXPECT_FLOAT_EQ(8.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));

  input.numPipeOps = NCCL_MAX_DEV_WORK_BATCH_COLLS + 1;
  EXPECT_FLOAT_EQ(6.0f, ncclTuningGetTime(&input, NCCL_ALGO_TREE, &latency, &bandwidth));
}

TEST_F(TuningGeneralMicrotest, TimeEstimatePenalizesDeepFp8RingOnly) {
  ncclTuningInput_t input = MakeTimeInput(1000, 1, ncclFloat8e4m3);
  comm_->nRanks = 9;
  float latency = 1.0f, bandwidth = 1.0f;
  EXPECT_FLOAT_EQ(2048.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));

  comm_->nRanks = 8;
  EXPECT_FLOAT_EQ(2.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));
  comm_->nRanks = 9;
  EXPECT_FLOAT_EQ(2.0f, ncclTuningGetTime(&input, NCCL_ALGO_TREE, &latency, &bandwidth));
  input.datatype = ncclFloat32;
  EXPECT_FLOAT_EQ(2.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));
  input.datatype = ncclFloat8e5m2;
  EXPECT_FLOAT_EQ(2048.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));
}

// ncclCudaCompCap() (src/misc/utils.cc) gives gfx1250 125. ncclTopoAddXGMI (src/graph/topo.cc) links XGMI
// peers as LINK_NVL and the Ring search starts typeIntra at the minimum GPU path type (src/graph/search.cc),
// so a gfx1250 Ring within an XGMI tray is PATH_NVL and takes ring.cc's single-node Blackwell NVLink branch.
// A Ring spanning trays is PATH_SYS and does not.
TEST_F(TuningGeneralMicrotest, RingSimplePlateauScalesWithStepsAndWorkOnSingleNodeNvl) {
  comm_->nNodes = 1;
  comm_->minCompCap = 125;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  // Two ranks (two AllReduce steps) sit below saturation; eight (fourteen steps) saturate at 1.4.
  EXPECT_NEAR(13.0, RingSimpleTimeUs(ncclFuncAllReduce, 2, 1024), 1e-3);
  EXPECT_NEAR(14.0, RingSimpleTimeUs(ncclFuncAllReduce, 8, 1024), 1e-3);
  // A one-step Ring gets no plateau.
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllGather, 2, 1024), 1e-3);
  // Work past 256 KiB per rank/channel trims the delta linearly, by at most 0.1 from 512 KiB: three
  // steps (0.9375) give 1 + 0.9375 * 0.35 at 384 KiB and 1 + 0.9375 * 0.3 at 1 MiB.
  EXPECT_NEAR(13.28125, RingSimpleTimeUs(ncclFuncReduceScatter, 4, 384 << 10), 1e-3);
  EXPECT_NEAR(12.8125, RingSimpleTimeUs(ncclFuncAllGather, 4, 1 << 20), 1e-3);
}

// Multi-node, only AllReduce takes the plateau, at a fixed factor whatever the intra-node link.
TEST_F(TuningGeneralMicrotest, RingSimplePlateauIsFixedMultiNode) {
  comm_->nNodes = 2;
  comm_->minCompCap = 125;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  EXPECT_NEAR(14.0, RingSimpleTimeUs(ncclFuncAllReduce, 2, 1024), 1e-3);
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllGather, 8, 1024), 1e-3);
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_PIX;
  EXPECT_NEAR(14.0, RingSimpleTimeUs(ncclFuncAllReduce, 2, 1024), 1e-3);
}

// Compute capability below 80 is NVIDIA-only; every AMD target reports 90 or more.
TEST_F(TuningGeneralMicrotest, RingSimplePlateauIsLargerMultiNodeBeforeAmpere) {
  comm_->nNodes = 2;
  comm_->minCompCap = 70;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  EXPECT_NEAR(19.0, RingSimpleTimeUs(ncclFuncAllReduce, 2, 1024), 1e-3);
}

// gfx942 reports compute capability 94, below the Blackwell gate.
TEST_F(TuningGeneralMicrotest, RingSimplePlateauIsAbsentOnNonBlackwell) {
  comm_->nNodes = 1;
  comm_->minCompCap = 94;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllReduce, 8, 1024), 1e-3);
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllGather, 8, 1024), 1e-3);
}

TEST_F(TuningGeneralMicrotest, RingSimplePlateauIsAbsentOverPci) {
  comm_->nNodes = 1;
  comm_->minCompCap = 125;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_PIX;
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllGather, 8, 1024), 1e-3);
}

// A seven-step Ring isolates the 64-byte threshold from step scaling.
TEST_F(TuningGeneralMicrotest, RingSimplePlateauIsAbsentBelow64Bytes) {
  comm_->nNodes = 1;
  comm_->minCompCap = 125;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_NVL;
  EXPECT_NEAR(10.0, RingSimpleTimeUs(ncclFuncAllGather, 8, 63), 1e-3);
  EXPECT_NEAR(14.0, RingSimpleTimeUs(ncclFuncAllGather, 8, 64), 1e-3);
}

TEST_F(TuningGeneralMicrotest, RingModelSimIsInvalidWithoutBandwidth) {
  comm_->tuningContext.generalBandwidths[ncclFuncAllGather][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = -1.0f;
  ncclTuningInput_t input = MakeTimeInput(1024, 1, ncclFloat32);
  input.func = ncclFuncAllGather;
  ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
  result.algo = NCCL_ALGO_RING;
  result.proto = NCCL_PROTO_SIMPLE;
  result.valid = 1;
  EXPECT_EQ(ncclSuccess, ncclTuningRingModelSim(&input, &result));
  EXPECT_EQ(0, result.valid);
}

TEST_F(TuningGeneralMicrotest, RingModelInitRejectsNonAlgoProtoIds) {
  EXPECT_EQ(ncclInvalidUsage, ncclTuningRingModelInit(comm_.get(), -1, nullptr));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningRingModelInit(comm_.get(), NCCL_TUNING_COUNT, nullptr));
  // The first symmetric-kernel id.
  EXPECT_EQ(ncclInvalidUsage,
            ncclTuningRingModelInit(comm_.get(), NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, nullptr));
}

TEST_F(TuningGeneralMicrotest, RingModelInitSingleNodeSimpleLatencyAndBandwidth) {
  SetRingInitComm(1, 4, 94);
  auto& k = comm_->tuningContext.tuningConstants;
  k.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 5.0;
  k.hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 1.0;
  k.hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 3.0;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_SIMPLE));
  // busBw = bwIntra 10 * 2 channels; bandwidth = busBw * nRanks / nSteps; latency = base + nSteps * hw.
  EXPECT_NEAR(80.0f / 6, RingBw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(11.0f, RingLat(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(80.0f / 3, RingBw(ncclFuncAllGather, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(8.0f, RingLat(ncclFuncAllGather, NCCL_PROTO_SIMPLE), 1e-4);
  // Broadcast on shared channels pays one hop; otherwise nSteps hops at the Tree Simple latency.
  EXPECT_NEAR(20.0f, RingBw(ncclFuncBroadcast, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(6.0f, RingLat(ncclFuncBroadcast, NCCL_PROTO_SIMPLE), 1e-4);
  comm_->graphs[NCCL_ALGO_RING].sameChannels = 0;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(17.0f, RingLat(ncclFuncBroadcast, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TuningGeneralMicrotest, RingModelInitCapsLlAndLl128Bandwidth) {
  SetRingInitComm(1, 4, 94);
  int cc = ncclTuningGetCompCapIndex(comm_.get());
  auto& k = comm_->tuningContext.tuningConstants;
  // LL halves busBw 20 and caps it at llMaxBw; LL128 keeps 92% and caps it at nChannels * per-channel max.
  k.llMaxBws[cc][0] = 4.0;
  k.perChMaxRingLL128Bws[cc][0] = 3.0;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_LL));
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_LL128));
  EXPECT_NEAR(4.0f * 4 / 3, RingBw(ncclFuncAllGather, NCCL_PROTO_LL), 1e-4);
  EXPECT_NEAR(6.0f * 4 / 3, RingBw(ncclFuncAllGather, NCCL_PROTO_LL128), 1e-4);
  k.llMaxBws[cc][0] = 100.0;
  k.perChMaxRingLL128Bws[cc][0] = 100.0;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_LL));
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_LL128));
  EXPECT_NEAR(10.0f * 4 / 3, RingBw(ncclFuncAllGather, NCCL_PROTO_LL), 1e-4);
  EXPECT_NEAR(18.4f * 4 / 3, RingBw(ncclFuncAllGather, NCCL_PROTO_LL128), 1e-4);
}

TEST_F(TuningGeneralMicrotest, RingModelInitMultiNodeNetOverheadAndInterLatency) {
  SetRingInitComm(2, 8, 100);
  comm_->maxLocalRanks = 4;
  comm_->cpuArch = NCCL_TOPO_CPU_ARCH_X86;
  comm_->cpuVendor = NCCL_TOPO_CPU_VENDOR_AMD;
  comm_->graphs[NCCL_ALGO_RING].latencyInter = 0.5f;
  auto& k = comm_->tuningContext.tuningConstants;
  k.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 5.0;
  k.hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 1.0;
  k.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 4.0;
  k.hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 7.0;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_SIMPLE));
  // Compute capability 100 on two nodes takes bwInter 1. Intra steps cost max(1, 3 * 2 us AMD x86 net
  // overhead) = 6; inter steps cost 4 + 2 * latencyInter = 5. AllReduce: 12 intra + 2 inter steps.
  EXPECT_NEAR(16.0f / 14, RingBw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(87.0f, RingLat(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  EXPECT_NEAR(46.0f, RingLat(ncclFuncAllGather, NCCL_PROTO_SIMPLE), 1e-4);
  // One rank per node: inter steps use the Tree NET latency, 7 + 1 = 8.
  comm_->maxLocalRanks = 1;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(93.0f, RingLat(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
  // Below compute capability 100, up to two nodes keep bwIntra 10.
  comm_->minCompCap = 94;
  ASSERT_EQ(ncclSuccess, RingInit(NCCL_PROTO_SIMPLE));
  EXPECT_NEAR(160.0f / 14, RingBw(ncclFuncAllReduce, NCCL_PROTO_SIMPLE), 1e-4);
}

TEST_F(TuningGeneralMicrotest, TimeEstimateAllowsValidatedGfx1250Fp8AllReduceRing) {
  rcclArchThresholds table{};
  table.ddaLL128Max[ncclFuncAllReduce] = 256ULL << 20;
  comm_->archThresholds = &table;
  comm_->nNodes = 1;
  comm_->archName = const_cast<char*>("gfx1250");
  const size_t ceiling = rcclDdaLL128Threshold(comm_.get(), ncclFuncAllReduce);
  ASSERT_EQ(256ULL << 20, ceiling);

  float latency = 1.0f, bandwidth = 1.0f;
  for (int nRanks : {kFp8RingPenaltyRanks + 1, 16, 32}) {
    comm_->nRanks = nRanks;
    ncclTuningInput_t atCeiling = MakeTimeInput(ceiling, 1, ncclFloat8e4m3);
    atCeiling.func = ncclFuncAllReduce;
    const float base = latency + static_cast<float>(atCeiling.nBytes) / (1000.0f * bandwidth);
    EXPECT_FLOAT_EQ(base, ncclTuningGetTime(&atCeiling, NCCL_ALGO_RING, &latency, &bandwidth)) << "nRanks=" << nRanks;

    ncclTuningInput_t below = MakeTimeInput(ceiling - 1, 1, ncclFloat8e4m3);
    below.func = ncclFuncAllReduce;
    const float belowBase = latency + static_cast<float>(below.nBytes) / (1000.0f * bandwidth);
    EXPECT_FLOAT_EQ(belowBase * 1024.0f, ncclTuningGetTime(&below, NCCL_ALGO_RING, &latency, &bandwidth))
        << "nRanks=" << nRanks;
  }

  comm_->nRanks = 32;
  comm_->archName = const_cast<char*>("gfx950");
  ncclTuningInput_t otherArch = MakeTimeInput(ceiling, 1, ncclFloat8e4m3);
  otherArch.func = ncclFuncAllReduce;
  const float otherBase = latency + static_cast<float>(otherArch.nBytes) / (1000.0f * bandwidth);
  EXPECT_FLOAT_EQ(otherBase * 1024.0f, ncclTuningGetTime(&otherArch, NCCL_ALGO_RING, &latency, &bandwidth));

  comm_->archName = const_cast<char*>("gfx1250");
  comm_->nRanks = 16;
  comm_->nNodes = 2;
  EXPECT_FLOAT_EQ(otherBase * 1024.0f, ncclTuningGetTime(&otherArch, NCCL_ALGO_RING, &latency, &bandwidth));

  comm_->nNodes = 1;
  comm_->nRanks = 12;
  EXPECT_FLOAT_EQ(otherBase, ncclTuningGetTime(&otherArch, NCCL_ALGO_RING, &latency, &bandwidth));

  table.ddaLL128Max[ncclFuncAllReduce] = 0;
  ASSERT_EQ(0u, rcclDdaLL128Threshold(comm_.get(), ncclFuncAllReduce));
  comm_->nRanks = 16;
  EXPECT_FLOAT_EQ(otherBase * 1024.0f, ncclTuningGetTime(&otherArch, NCCL_ALGO_RING, &latency, &bandwidth));
}

TEST_F(TuningGeneralMicrotest, ThreadCountValidationUsesDefaultAndBounds) {
  EXPECT_EQ(128, ncclTuningGetNthreads("threads", -2, 64, 256, 128));
  EXPECT_EQ(128, ncclTuningGetNthreads("threads", 0, 64, 256, 128));
  EXPECT_EQ(64, ncclTuningGetNthreads("threads", 64, 64, 256, 96));
  EXPECT_EQ(96, ncclTuningGetNthreads("threads", 96, 64, 256, 128));
  EXPECT_EQ(128, ncclTuningGetNthreads("threads", 128, 64, 256, 96));
  EXPECT_EQ(256, ncclTuningGetNthreads("threads", 256, 64, 256, 96));
  EXPECT_EQ(256, ncclTuningGetNthreads("threads", 65, 64, 256, 96));
  EXPECT_EQ(256, ncclTuningGetNthreads("threads", 288, 64, 256, 96));
  EXPECT_EQ(64, ncclTuningGetNthreads("threads", 32, 64, 256, 96));
}

TEST_F(TuningGeneralMicrotest, ThreadThresholdDefaultsAndEnvironmentOverride) {
  comm_->nRanks = 4;
  SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(NCCL_SIMPLE_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_SIMPLE_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_MAX_NTHREADS,
            comm_->tuningContext.maxThreads[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_MAX_NTHREADS,
            comm_->tuningContext.maxThreads[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(NCCL_LL_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL128_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
  EXPECT_EQ(NCCL_LL128_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
  EXPECT_EQ(4 * NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_PAT][NCCL_PROTO_LL]);
  EXPECT_EQ(512, comm_->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(512, comm_->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE]);

  {
    ScopedHook loadParam(g_loadParam, [](const char* name, int64_t defaultValue) {
      if (std::strcmp(name, "NTHREADS") == 0) return int64_t{128};
      if (std::strcmp(name, "LL128_NTHREADS") == 0) return int64_t{32};
      return defaultValue;
    });
    ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
    EXPECT_EQ(128, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
    EXPECT_EQ(128, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
    EXPECT_EQ(128, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL]);
    EXPECT_EQ(128, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
    EXPECT_EQ(NCCL_LL128_MAX_NTHREADS / 4,
              comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
    EXPECT_EQ(NCCL_LL128_MAX_NTHREADS / 4,
              comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
  }

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "1 2 3 4 5 6");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(1, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(3, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(4, comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(5, comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
  EXPECT_EQ(6, comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "7 -2 9 -2 11 -2");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(7, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL128_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
  EXPECT_EQ(9, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "13 14 15");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(13, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(14, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
  EXPECT_EQ(15, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(4 * NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL128_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128]);
  EXPECT_EQ(NCCL_SIMPLE_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "0 -2 -2 -2 -2 -2");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(0, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL128_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
}

TEST_F(TuningGeneralMicrotest, ThreadThresholdParameterValuesClampToMinimum) {
  comm_->nRanks = 4;
  SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
  ScopedHook loadParam(g_loadParam, [](const char* name, int64_t defaultValue) {
    if (std::strcmp(name, "NTHREADS") == 0) return int64_t{32};
    return defaultValue;
  });

  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(2 * WARP_SIZE, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(2 * WARP_SIZE, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(2 * WARP_SIZE, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(2 * WARP_SIZE, comm_->tuningContext.maxThreads[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
}

TEST_F(TuningGeneralMicrotest, ChannelSelectionScalesRingAndHonorsCap) {
  const int threads = 256;
  const int threshold = 64;
  SetChannelDefaults(8, threads, threshold);
  ncclTuningResult_t large = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, large.nChannels);
  EXPECT_EQ(9, large.nWarps);

  ncclTuningResult_t small = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(1, small.nChannels);
  EXPECT_EQ(3, small.nWarps);

  const size_t oneHalvingBytes = static_cast<size_t>(threads / 2) * threshold + 1;
  ncclTuningResult_t halved = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, oneHalvingBytes);
  EXPECT_EQ(1, halved.nChannels);
  EXPECT_EQ(4, halved.nWarps);

  const size_t exactThresholdBytes = 2 * static_cast<size_t>(threads) * threshold;
  ncclTuningResult_t exactThreshold = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, exactThresholdBytes);
  EXPECT_EQ(2, exactThreshold.nChannels);
  EXPECT_EQ(8, exactThreshold.nWarps);

  ncclTuningResult_t capped = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, 1ull << 30, 3);
  EXPECT_EQ(3, capped.nChannels);
  EXPECT_EQ(3, capped.maxChannels);

  ncclTuningResult_t uncapped = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1ull << 30, 0);
  EXPECT_EQ(8, uncapped.nChannels);
  EXPECT_EQ(8, uncapped.maxChannels);

  SetChannelDefaults(8, 64, 64);
  ncclTuningResult_t minimumWarps = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, 1ull << 30, 20);
  EXPECT_EQ(3, minimumWarps.nWarps);
  EXPECT_EQ(8, minimumWarps.maxChannels);

  SetChannelDefaults(8, 192, 64);
  ncclTuningResult_t nonDivisibleThreads = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, 1);
  EXPECT_EQ(1, nonDivisibleThreads.nChannels);
  EXPECT_EQ(6, nonDivisibleThreads.nWarps);
}

TEST_F(TuningGeneralMicrotest, ChannelSelectionHandlesTreeNvlsPatAndCollnet) {
  SetChannelDefaults();
  comm_->nNodes = 2;
  comm_->nvlsChannels = 12;
  ncclTuningResult_t tree = GetChannels(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(1, tree.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, tree.nWarps);

  ncclTuningResult_t nvls = GetChannels(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, nvls.nChannels);
  comm_->nNodes = 1;
  ncclTuningResult_t singleNodeNvls = GetChannels(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(12, singleNodeNvls.nChannels);
  ncclTuningResult_t smallNvls = GetChannels(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(12, smallNvls.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, smallNvls.nWarps);
  comm_->nNodes = 2;
  ncclTuningResult_t nvlsTree = GetChannels(NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(12, nvlsTree.nChannels);
  ncclTuningResult_t smallNvlsTree = GetChannels(NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(12, smallNvlsTree.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, smallNvlsTree.nWarps);

  comm_->nvlsChannels = 4;
  comm_->isOneRPN = false;
  ncclTuningResult_t pat = GetChannels(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(4, pat.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, pat.nWarps);
  comm_->isOneRPN = true;
  ncclTuningResult_t oneRpnPat = GetChannels(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, oneRpnPat.nChannels);
  comm_->tuningContext.maxThreads[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 64;
  ncclTuningResult_t forcedPatThreads = GetChannels(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, forcedPatThreads.nWarps);

  SetChannelDefaults(24);
  comm_->channels[0].collnetDirect.nHeads = 1;
  ncclTuningResult_t highChannelCollnet =
      GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 150000);
  EXPECT_EQ(18, highChannelCollnet.nChannels);

  SetChannelDefaults();
  comm_->channels[0].collnetDirect.nHeads = 2;
  ncclTuningResult_t collnet = GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, collnet.nChannels);
  ncclTuningResult_t smallCollnet = GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(1, smallCollnet.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, smallCollnet.nWarps);
  ncclTuningResult_t midCollnet = GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 50000);
  EXPECT_EQ(3, midCollnet.nChannels);
}

TEST_F(TuningGeneralMicrotest, ExpandIdDecodesEveryRangeAndRejectsBadInputs) {
  int algo = -9, proto = -9, sym = -9, ce = -9;
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(-1, &algo, &proto, &sym, &ce));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(NCCL_TUNING_COUNT, &algo, &proto, &sym, &ce));

  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(0, &algo, &proto, &sym, &ce));
  EXPECT_EQ(NCCL_ALGO_TREE, algo);
  EXPECT_EQ(NCCL_PROTO_LL, proto);

  int generalId = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL128;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(generalId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(NCCL_ALGO_RING, algo);
  EXPECT_EQ(NCCL_PROTO_LL128, proto);
  EXPECT_EQ(ncclSymkKernelId_Count, sym);
  EXPECT_EQ(ncclCeMethodId_Count, ce);

  int distinctGeneralId = NCCL_ALGO_PAT * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
  ASSERT_EQ(ncclSuccess,
            ncclTuningExpandId(distinctGeneralId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(NCCL_ALGO_PAT, algo);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, proto);

  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(generalId, nullptr, &proto, &sym, &ce));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(generalId, &algo, nullptr, &sym, &ce));

  int symId = NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(symId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(0, sym);
  EXPECT_EQ(NCCL_ALGO_UNDEF, algo);
  EXPECT_EQ(NCCL_PROTO_UNDEF, proto);
  EXPECT_EQ(ncclCeMethodId_Count, ce);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(symId, &algo, &proto, nullptr, &ce));
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(symId + 1, &algo, &proto, &sym, &ce));
  EXPECT_EQ(1, sym);

  int ceId = NCCL_TUNING_CE_METHOD_ID_OFFSET;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(ceId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(0, ce);
  EXPECT_EQ(NCCL_ALGO_UNDEF, algo);
  EXPECT_EQ(NCCL_PROTO_UNDEF, proto);
  EXPECT_EQ(ncclSymkKernelId_Count, sym);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(ceId, &algo, &proto, &sym, nullptr));
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(ceId + 1, &algo, &proto, &sym, &ce));
  EXPECT_EQ(1, ce);
}

}  // namespace
