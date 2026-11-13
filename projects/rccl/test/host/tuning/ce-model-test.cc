/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the copy-engine AllGather model in src/tuning/ce_model.cc.

#include <gtest/gtest.h>

#include <memory>

#include "comm.h"
#include "fakes/nccl_fakes.h"
#include "fakes/nccl_stubs.h"

#include CE_MODEL_CC_PATH

namespace {

constexpr size_t kPerRankBytes = 1000000;
constexpr int kFixedDriver = CE_BLACKWELL_FIX_DRIVER;
constexpr int kPreFixDriver = CE_BLACKWELL_FIX_DRIVER - 10;

class CeModelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->minCompCap = 90;
    comm_->cpuArch = NCCL_TOPO_CPU_ARCH_X86;
    comm_->devrState.lsaSize = 4;
    ncclCudaDriverVersionCache = kFixedDriver;
  }

  void TearDown() override {
    ResetNcclStubs();
    ResetNcclFakes();
  }

  float TimeUs(int method, int inPlace, int captured = 0) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = ncclFuncAllGather;
    input.nBytes = kPerRankBytes;
    input.inPlace = inPlace;
    input.captured = captured;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.id = NCCL_TUNING_CE_METHOD_ID_OFFSET + method;
    EXPECT_EQ(ncclSuccess, ncclTuningCeModelSim(&input, &result));
    return result.timeUs;
  }

  std::unique_ptr<ncclComm> comm_;
};

// Hopper unicast, not captured: floor 17.50 + 1.957 us per rank; 3 peers' data over 392 GB/s NVLink.
constexpr float kSm90UcFloor4 = 17.50f + 1.957f * 4;
constexpr float kSm90UcNvlinkUs4 = 3.0f * kPerRankBytes / 392 / 1e3f;
// Hopper multicast, not captured, 2 ranks.
constexpr float kSm90McFloor2 = 14.90f + 0.726f * 2;

TEST_F(CeModelMicrotest, UnicastInPlaceIsFloorPlusNvlinkTransfer) {
  EXPECT_NEAR(kSm90UcFloor4 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 1), 1e-3);
}

TEST_F(CeModelMicrotest, UnicastOutOfPlaceIsBoundedByTheD2hSlice) {
  // One rank's slice over 2 D2H engines * 53 GB/s is slower than the NVLink term here.
  EXPECT_NEAR(kSm90UcFloor4 + kPerRankBytes / 106.0f / 1e3f, TimeUs(ncclCeMethodId_AllGather_UC, 0), 1e-3);
}

TEST_F(CeModelMicrotest, CoherentCpuSkipsTheD2hSlice) {
  comm_->cpuArch = NCCL_TOPO_CPU_ARCH_ARM;
  EXPECT_NEAR(kSm90UcFloor4 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 0), 1e-3);
}

TEST_F(CeModelMicrotest, BlackwellUsesItsOwnFloorAndNvlinkBandwidth) {
  comm_->minCompCap = 100;
  EXPECT_NEAR(32.04f + 0.130f * 4 + 3.0f * kPerRankBytes / 779 / 1e3f, TimeUs(ncclCeMethodId_AllGather_UC, 1),
              1e-3);
}

TEST_F(CeModelMicrotest, MulticastBeforeTheDriverFixIsCappedByD2h) {
  comm_->devrState.lsaSize = 2;
  ncclCudaDriverVersionCache = kPreFixDriver;
  // min(106 GB/s D2H, 392 / 2 per link) * 2 ranks = 212 GB/s for 2 ranks' data.
  EXPECT_NEAR(kSm90McFloor2 + 2.0f * kPerRankBytes / 212 / 1e3f, TimeUs(ncclCeMethodId_AllGather_MC, 1), 1e-3);
}

TEST_F(CeModelMicrotest, MulticastFromTheDriverFixOrOnCoherentCpuUsesFullNvlink) {
  comm_->devrState.lsaSize = 2;
  float expected = kSm90McFloor2 + 2.0f * kPerRankBytes / 392 / 1e3f;
  EXPECT_NEAR(expected, TimeUs(ncclCeMethodId_AllGather_MC, 1), 1e-3);
  ncclCudaDriverVersionCache = kPreFixDriver;
  comm_->cpuArch = NCCL_TOPO_CPU_ARCH_ARM;
  EXPECT_NEAR(expected, TimeUs(ncclCeMethodId_AllGather_MC, 1), 1e-3);
}

// Hopper unicast, captured: floor 9.98 + 6.160 us per rank.
constexpr float kSm90UcGraphFloor4 = 9.98f + 6.160f * 4;

TEST_F(CeModelMicrotest, CapturedUnicastAddsOneFloorPerIntraBatchSync) {
  comm_->ceColl.intraBatchSyncFreq = 2;
  comm_->ceColl.intraBatchSyncMsgThreshold = 0;
  // 3 in-place ops at frequency 2: 2 blocks, one sync between them plus one for the even block count.
  EXPECT_NEAR(kSm90UcGraphFloor4 * 3 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 1, 1), 1e-3);
  // 4 out-of-place ops at frequency 3: the same 2 blocks. Captured runs skip the D2H slice.
  comm_->ceColl.intraBatchSyncFreq = 3;
  EXPECT_NEAR(kSm90UcGraphFloor4 * 3 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 0, 1), 1e-3);
}

TEST_F(CeModelMicrotest, CapturedUnicastSkipsSyncsBelowThresholdWithoutCeInitOrWithinOneBlock) {
  comm_->ceColl.intraBatchSyncFreq = 2;
  comm_->ceColl.intraBatchSyncMsgThreshold = 3 * kPerRankBytes + 1;
  EXPECT_NEAR(kSm90UcGraphFloor4 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 1, 1), 1e-3);
  // A frequency of 0 means CE was never initialized.
  comm_->ceColl.intraBatchSyncFreq = 0;
  comm_->ceColl.intraBatchSyncMsgThreshold = 0;
  EXPECT_NEAR(kSm90UcGraphFloor4 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 1, 1), 1e-3);
  // Not more ops than the frequency: no sync.
  comm_->ceColl.intraBatchSyncFreq = 3;
  EXPECT_NEAR(kSm90UcGraphFloor4 + kSm90UcNvlinkUs4, TimeUs(ncclCeMethodId_AllGather_UC, 1, 1), 1e-3);
}

}  // namespace
