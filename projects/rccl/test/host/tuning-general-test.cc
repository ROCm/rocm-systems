/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/tuning_general.cc.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include TUNING_GENERAL_CC_PATH

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

  std::unique_ptr<ncclComm> comm_;
};

TEST_F(TuningGeneralMicrotest, StepCountsMatchCollectiveShape) {
  EXPECT_EQ(14, ncclTuningGetNsteps(ncclFuncAllReduce, 8));
  EXPECT_EQ(7, ncclTuningGetNsteps(ncclFuncReduceScatter, 8));
  EXPECT_EQ(7, ncclTuningGetNsteps(ncclFuncAllGather, 8));
  EXPECT_EQ(8, ncclTuningGetNsteps(ncclFuncBroadcast, 8));
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

  comm_->nNodes = 2;
  comm_->graphs[NCCL_ALGO_RING].typeIntra = PATH_PIX;
  ncclTuningGetHwIndexes(comm_.get(), NCCL_ALGO_RING, &intra, &inter);
  EXPECT_EQ(NCCL_HW_PCI, intra);
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
  input.datatype = ncclFloat8e5m2;
  EXPECT_FLOAT_EQ(2048.0f, ncclTuningGetTime(&input, NCCL_ALGO_RING, &latency, &bandwidth));
}

TEST_F(TuningGeneralMicrotest, ThreadCountValidationUsesDefaultAndBounds) {
  EXPECT_EQ(128, ncclTuningGetNthreads("threads", -2, 64, 256, 128));
  EXPECT_EQ(128, ncclTuningGetNthreads("threads", 128, 64, 256, 96));
  EXPECT_EQ(256, ncclTuningGetNthreads("threads", 65, 64, 256, 96));
  EXPECT_EQ(256, ncclTuningGetNthreads("threads", 288, 64, 256, 96));
  EXPECT_EQ(64, ncclTuningGetNthreads("threads", 32, 64, 256, 96));
}

TEST_F(TuningGeneralMicrotest, ThreadThresholdDefaultsAndEnvironmentOverride) {
  comm_->nRanks = 4;
  SetMicroEnvAbsent("NCCL_THREAD_THRESHOLDS");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(NCCL_SIMPLE_MAX_NTHREADS, comm_->tuningContext.maxThreads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(4 * NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(512, comm_->tuningContext.threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE]);

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "1 2 3 4 5 6");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(1, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(3, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(4, comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_LL]);
  EXPECT_EQ(6, comm_->tuningContext.threadThresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);

  SetMicroEnv("NCCL_THREAD_THRESHOLDS", "7 -2 9 -2 11 -2");
  ASSERT_EQ(ncclSuccess, ncclTuningSetThreadThresholds(comm_.get()));
  EXPECT_EQ(7, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
  EXPECT_EQ(NCCL_LL128_THREAD_THRESHOLD,
            comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL128]);
  EXPECT_EQ(9, comm_->tuningContext.threadThresholds[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
}

TEST_F(TuningGeneralMicrotest, ChannelSelectionScalesRingAndHonorsCap) {
  SetChannelDefaults();
  ncclTuningResult_t large = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, large.nChannels);
  EXPECT_EQ(9, large.nWarps);

  ncclTuningResult_t small = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(1, small.nChannels);
  EXPECT_EQ(3, small.nWarps);

  ncclTuningResult_t capped = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, 1ull << 30, 3);
  EXPECT_EQ(3, capped.nChannels);
  EXPECT_EQ(3, capped.maxChannels);

  SetChannelDefaults(8, 64, 64);
  ncclTuningResult_t minimumWarps = GetChannels(NCCL_ALGO_RING, NCCL_PROTO_LL, 1ull << 30, 20);
  EXPECT_EQ(3, minimumWarps.nWarps);
  EXPECT_EQ(8, minimumWarps.maxChannels);
}

TEST_F(TuningGeneralMicrotest, ChannelSelectionHandlesTreeNvlsPatAndCollnet) {
  SetChannelDefaults();
  comm_->nNodes = 2;
  comm_->nvlsChannels = 12;
  ncclTuningResult_t tree = GetChannels(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, tree.nWarps);

  ncclTuningResult_t nvls = GetChannels(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, nvls.nChannels);
  comm_->nNodes = 1;
  ncclTuningResult_t singleNodeNvls = GetChannels(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(12, singleNodeNvls.nChannels);
  comm_->nNodes = 2;
  ncclTuningResult_t nvlsTree = GetChannels(NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(12, nvlsTree.nChannels);

  comm_->isOneRPN = false;
  ncclTuningResult_t pat = GetChannels(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, pat.nChannels);
  EXPECT_EQ(NCCL_MAX_NTHREADS / WARP_SIZE, pat.nWarps);
  comm_->isOneRPN = true;
  ncclTuningResult_t oneRpnPat = GetChannels(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, oneRpnPat.nChannels);

  comm_->channels[0].collnetDirect.nHeads = 1;
  ncclTuningResult_t collnet = GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 1ull << 30);
  EXPECT_EQ(8, collnet.nChannels);
  ncclTuningResult_t smallCollnet = GetChannels(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, 1);
  EXPECT_EQ(1, smallCollnet.nChannels);
}

TEST_F(TuningGeneralMicrotest, ExpandIdDecodesEveryRangeAndRejectsBadInputs) {
  int algo = -9, proto = -9, sym = -9, ce = -9;
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(-1, &algo, &proto, &sym, &ce));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(NCCL_TUNING_COUNT, &algo, &proto, &sym, &ce));

  int generalId = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(generalId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(NCCL_ALGO_RING, algo);
  EXPECT_EQ(NCCL_PROTO_LL, proto);
  EXPECT_EQ(ncclSymkKernelId_Count, sym);
  EXPECT_EQ(ncclCeMethodId_Count, ce);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(generalId, nullptr, &proto, &sym, &ce));
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(generalId, &algo, nullptr, &sym, &ce));

  int symId = NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(symId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(0, sym);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(symId, &algo, &proto, nullptr, &ce));

  int ceId = NCCL_TUNING_CE_METHOD_ID_OFFSET;
  ASSERT_EQ(ncclSuccess, ncclTuningExpandId(ceId, &algo, &proto, &sym, &ce));
  EXPECT_EQ(0, ce);
  EXPECT_EQ(ncclInvalidUsage, ncclTuningExpandId(ceId, &algo, &proto, &sym, nullptr));
}

}  // namespace
