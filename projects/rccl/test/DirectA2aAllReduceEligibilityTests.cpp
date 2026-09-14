/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "algorithms/direct_a2a/all_reduce/direct_a2a_all_reduce.h"
#include "gtest/gtest.h"

#include <cstring>

namespace RcclUnitTesting {

class DirectA2aAllReduceEligibilityTest : public ::testing::Test {
protected:
  DirectA2aAllReduceEligibilityTest() { reset(); }

  void reset() {
    std::memset(static_cast<void*>(&comm_), 0, sizeof(comm_));
    std::strcpy(archName_, "gfx1151");
    comm_.archName = archName_;
    comm_.bootstrap = &bootstrapPlaceholder_;
    comm_.minLocalRanks = 1;
    comm_.maxLocalRanks = 1;
    comm_.config.blocking = 1;
    comm_.directA2aScratch = reinterpret_cast<void*>(0x1000);
    setRanks(RCCL_DIRECT_A2A_MAX_RANKS);
  }

  void setRanks(int nRanks) {
    comm_.nRanks = nRanks;
    comm_.nNodes = nRanks;
    comm_.directA2aScratchBytes = rcclDirectA2aAllReduceScratchBytes(nRanks);
  }

  ncclComm comm_{};
  char archName_[16]{};
  char bootstrapPlaceholder_{};
};

TEST_F(DirectA2aAllReduceEligibilityTest, EligibleSupportedTypes) {
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat16, ncclSum));
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclBfloat16, ncclSum));
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, EligibleRankAndSizeBoundaries) {
  setRanks(RCCL_DIRECT_A2A_MIN_RANKS);
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(
    &comm_, RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES / sizeof(float), ncclFloat32, ncclSum));
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(
    &comm_, RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES / sizeof(float) + 1, ncclFloat32, ncclSum));
  setRanks(3);
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(
    &comm_, RCCL_DIRECT_A2A_MAX_BYTES / sizeof(float), ncclFloat32, ncclSum));
  setRanks(RCCL_DIRECT_A2A_MAX_RANKS);
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(
    &comm_, RCCL_DIRECT_A2A_MAX_BYTES / sizeof(float), ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, SupportsOneShotAndUnevenTwoShotCounts) {
  const size_t oneShotCount = RCCL_DIRECT_A2A_DEFAULT_ONESHOT_THRESHOLD_BYTES / sizeof(float);
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, oneShotCount, ncclFloat32, ncclSum));
  // oneShotCount+1 = 16385 is not divisible by nRanks=4, so this is an uneven two-shot.
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, oneShotCount + 1, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, TwoRanksUseOneShotThroughFourMiB) {
  setRanks(2);
  const size_t count = RCCL_DIRECT_A2A_TWO_RANK_MAX_BYTES / sizeof(float);
  comm_.directA2aScratchBytes = (size_t)comm_.nRanks * count * sizeof(float) - 1;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, count, ncclFloat32, ncclSum));

  comm_.directA2aScratchBytes++;
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, count, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsNullCommAndMissingResources) {
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(nullptr, 256, ncclFloat32, ncclSum));
  comm_.bootstrap = nullptr;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
  reset();
  comm_.directA2aScratch = nullptr;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsWrongArchitecture) {
  std::strcpy(archName_, "gfx1250");
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsNonBlockingCommunicator) {
  comm_.config.blocking = 0;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, EnforcesRankRange) {
  setRanks(1);
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
  reset();
  setRanks(RCCL_DIRECT_A2A_MAX_RANKS + 1);
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RequiresOneGpuPerNode) {
  comm_.nNodes = 2;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
  reset();
  comm_.maxLocalRanks = 2;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
  reset();
  comm_.minLocalRanks = 2;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsZeroAndOversizedMessages) {
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 0, ncclFloat32, ncclSum));
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(
    &comm_, RCCL_DIRECT_A2A_MAX_BYTES / sizeof(float) + 1, ncclFloat32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsUnsupportedOperationAndDatatype) {
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclProd));
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclInt32, ncclSum));
}

TEST_F(DirectA2aAllReduceEligibilityTest, RejectsInsufficientScratch) {
  comm_.directA2aScratchBytes = comm_.nRanks * 256 * sizeof(float) - 1;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));
  comm_.directA2aScratchBytes++;
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, 256, ncclFloat32, ncclSum));

  reset();
  const size_t count = RCCL_DIRECT_A2A_DEFAULT_ONESHOT_THRESHOLD_BYTES / sizeof(float) + 1;
  const size_t maxChunkCount = (count + (size_t)comm_.nRanks - 1) / (size_t)comm_.nRanks;
  comm_.directA2aScratchBytes = ((size_t)comm_.nRanks + 1) * maxChunkCount * sizeof(float) - 1;
  EXPECT_FALSE(rcclDirectA2aAllReduceEligible(&comm_, count, ncclFloat32, ncclSum));
  comm_.directA2aScratchBytes++;
  EXPECT_TRUE(rcclDirectA2aAllReduceEligible(&comm_, count, ncclFloat32, ncclSum));
}

} // namespace RcclUnitTesting
