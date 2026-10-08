/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for GIN-SDMA AllReduce size policy in
// src/algorithms/gin/gin_all_reduce_policy.h. No GPU, no librccl: the header
// is the same code ncclAllReduceGinSdmaEligible() uses after comm/arch gates.

#include <gtest/gtest.h>

#include <cstddef>

#include "algorithms/gin/gin_all_reduce_policy.h"

namespace {

constexpr size_t kKiB = 1024ull;
constexpr size_t kMiB = 1024ull * kKiB;
constexpr size_t kFloat = 4;
constexpr int kRanks = 8;

size_t countForBytes(size_t bytes, size_t typeSize = kFloat) { return bytes / typeSize; }

TEST(GinAllReducePolicy, ThresholdsMatchDocumentedBands) {
  EXPECT_EQ(kGinAllReduceMinBytes, static_cast<int>(512 * kKiB));
  EXPECT_EQ(kGinAllReduceLsaOneShotMaxBytes, 4ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinTwoShotMinBytes, 256ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinScatterMinBytes, 64ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinScatterPipelineMinBytes, 512ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinScatterMaxBytes, 4ull * 1024ull * kMiB);
  EXPECT_EQ(kGinAllReduceMaxRanks, 8);
  EXPECT_EQ(kGinAllReduceMinPutBytes, 128u);
  EXPECT_EQ(kGinAllReduceTwoShotScratchBytes, 512ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinScatterChunkBytes, 16ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinScatterNumSlots, 2);
}

TEST(GinAllReducePolicy, DefaultRejectsBelow256MiB) {
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(4 * kMiB), kFloat, kRanks, false));
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(8 * kMiB), kFloat, kRanks, false));
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(128 * kMiB), kFloat, kRanks, false));
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(256 * kMiB - 16), kFloat, kRanks, false));
}

TEST(GinAllReducePolicy, DefaultAcceptsAligned256MiB) {
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(256 * kMiB), kFloat, kRanks, false));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(512 * kMiB), kFloat, kRanks, false));
}

TEST(GinAllReducePolicy, DefaultRejectsUnaligned256MiB) {
  // count not divisible by nRanks
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(256 * kMiB) + 1, kFloat, kRanks, false));
}

TEST(GinAllReducePolicy, ForceRejectsBelowMinBytes) {
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(256 * kKiB), kFloat, kRanks, true));
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(512 * kKiB - 4), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceAcceptsOneShotBand) {
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(512 * kKiB), kFloat, kRanks, true));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(2 * kMiB), kFloat, kRanks, true));
  // Inclusive 4 MiB: one-shot, so no two-shot alignment check.
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(4 * kMiB), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceJustAbove4MiBRequiresTwoShotAlignment) {
  // 4 MiB + 16 of float32 is two-shot; per-rank slice is not 16-byte aligned.
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(4 * kMiB + 16), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceAcceptsAlignedTwoShotBand) {
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(8 * kMiB), kFloat, kRanks, true));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(16 * kMiB), kFloat, kRanks, true));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(128 * kMiB), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceRejectsUnalignedTwoShotBand) {
  // > 4 MiB so LSA two-shot; per-rank slice is not 16-byte aligned.
  constexpr size_t countPerRank = 262145; // 262145 * 4 % 16 != 0
  const size_t count = static_cast<size_t>(kRanks) * countPerRank;
  ASSERT_GT(count * kFloat, kGinAllReduceLsaOneShotMaxBytes);
  ASSERT_LT(count * kFloat, kGinAllReduceGinTwoShotMinBytes);
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(count, kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceAcceptsAlignedGinTwoShot) {
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(256 * kMiB), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, HalfAndBf16Default256MiB) {
  constexpr size_t kHalf = 2;
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(128 * kMiB, kHalf), kHalf, kRanks, false));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(256 * kMiB, kHalf), kHalf, kRanks, false));
}

TEST(GinAllReducePolicy, YieldToDdaBySize) {
  EXPECT_TRUE(ginAllReduceYieldToDdaBySize(countForBytes(4 * kMiB), kFloat, false));
  EXPECT_TRUE(ginAllReduceYieldToDdaBySize(countForBytes(255 * kMiB), kFloat, false));
  EXPECT_FALSE(ginAllReduceYieldToDdaBySize(countForBytes(256 * kMiB), kFloat, false));
  EXPECT_FALSE(ginAllReduceYieldToDdaBySize(countForBytes(4 * kMiB), kFloat, true));
  EXPECT_FALSE(ginAllReduceYieldToDdaBySize(countForBytes(256 * kMiB), kFloat, true));
}

TEST(GinAllReducePolicy, TwoShotRejectsZeroRanks) {
  EXPECT_FALSE(ginAllReduceTwoShotEligible(1024, kFloat, 0));
  EXPECT_FALSE(ginAllReduceGinTwoShotEligible(countForBytes(256 * kMiB), kFloat, 0));
}

TEST(GinAllReducePolicy, TwoShotRejectsCountNotDivisibleByRanks) {
  EXPECT_FALSE(ginAllReduceTwoShotEligible(7, kFloat, kRanks));
}

TEST(GinAllReducePolicy, Gfx1250ScatterMinBytesFrom64MiB) {
  EXPECT_FALSE(
    ginAllReduceSizePolicyEligible(countForBytes(64 * kMiB - 16), kFloat, kRanks, false, kGinAllReduceGinScatterMinBytes));
  EXPECT_TRUE(
    ginAllReduceSizePolicyEligible(countForBytes(64 * kMiB), kFloat, kRanks, false, kGinAllReduceGinScatterMinBytes));
  EXPECT_TRUE(
    ginAllReduceSizePolicyEligible(countForBytes(128 * kMiB), kFloat, kRanks, false, kGinAllReduceGinScatterMinBytes));
  // gfx950 default floor stays 256 MiB.
  EXPECT_FALSE(ginAllReduceSizePolicyEligible(countForBytes(64 * kMiB), kFloat, kRanks, false));
}

TEST(GinAllReducePolicy, YieldToDdaBySizeGfx1250) {
  EXPECT_TRUE(ginAllReduceYieldToDdaBySize(countForBytes(63 * kMiB), kFloat, false, kGinAllReduceGinScatterMinBytes));
  EXPECT_FALSE(ginAllReduceYieldToDdaBySize(countForBytes(64 * kMiB), kFloat, false, kGinAllReduceGinScatterMinBytes));
}

TEST(GinAllReducePolicy, GinScatterDispatchGfx1250From64MiB) {
  EXPECT_FALSE(ginAllReduceGinScatterDispatch(true, 64ull * kMiB - 16));
  EXPECT_TRUE(ginAllReduceGinScatterDispatch(true, 64ull * kMiB));
  EXPECT_TRUE(ginAllReduceGinScatterDispatch(true, 256ull * kMiB));
  EXPECT_FALSE(ginAllReduceGinScatterDispatch(false, 64ull * kMiB));
  EXPECT_FALSE(ginAllReduceGinScatterDispatch(false, 256ull * kMiB));
}

TEST(GinAllReducePolicy, GinScatterLaunchFrom64MiB) {
  const size_t scratch = kGinAllReduceTwoShotScratchBytes;
  const size_t chunk64 = (64ull * kMiB) / kRanks;
  const size_t chunk256 = (256ull * kMiB) / kRanks;
  const size_t chunk512 = (512ull * kMiB) / kRanks;
  EXPECT_FALSE(ginAllReduceGinScatterLaunch(64ull * kMiB - 16, chunk64, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(64ull * kMiB, chunk64, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(256ull * kMiB, chunk256, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(512ull * kMiB, chunk512, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(2ull * 1024ull * kMiB, 256ull * kMiB, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(kGinAllReduceGinScatterMaxBytes, 512ull * kMiB, kRanks, scratch));
}

TEST(GinAllReducePolicy, GinScatterLaunchWhenMessageExceedsScratch) {
  const size_t chunk4g = kGinAllReduceGinScatterMaxBytes / kRanks;
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(kGinAllReduceGinScatterMaxBytes, chunk4g, kRanks, 512ull * kMiB));
  const size_t minScratch = ginAllReduceGinScatterStagedScratchBytes(kGinAllReduceGinScatterChunkAlign, kRanks, 1);
  EXPECT_TRUE(ginAllReduceGinScatterLaunch(64ull * kMiB, (64ull * kMiB) / kRanks, kRanks, minScratch));
  EXPECT_FALSE(ginAllReduceGinScatterLaunch(64ull * kMiB, (64ull * kMiB) / kRanks, kRanks, minScratch - 1));
}

TEST(GinAllReducePolicy, GinScatterDefaultScratchStagesAbove512MiB) {
  const size_t scratch = kGinAllReduceTwoShotScratchBytes;
  EXPECT_FALSE(ginAllReduceGinScatterStaged((64ull * kMiB) / kRanks, kRanks, scratch));
  EXPECT_FALSE(ginAllReduceGinScatterStaged((256ull * kMiB) / kRanks, kRanks, scratch));
  EXPECT_FALSE(ginAllReduceGinScatterStaged((512ull * kMiB) / kRanks, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterStaged((1024ull * kMiB) / kRanks, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterStaged((2ull * 1024ull * kMiB) / kRanks, kRanks, scratch));
  EXPECT_TRUE(ginAllReduceGinScatterStaged(kGinAllReduceGinScatterMaxBytes / kRanks, kRanks, scratch));
}

TEST(GinAllReducePolicy, GinScatterStagesWhenMessageExceedsScratch) {
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  const size_t smallSeq = ginAllReduceGinScatterStagedScratchBytes(kGinAllReduceGinScatterChunkBytes, kRanks, 1);
  ginAllReduceGinScatterChunkPlan(32ull * kMiB, kRanks, smallSeq, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 2);
  EXPECT_EQ(nPhases, 1);
  EXPECT_EQ(uniformBytes, kGinAllReduceGinScatterChunkBytes);
  const size_t smallPipe = ginAllReduceGinScatterStagedScratchBytes(kGinAllReduceGinScatterChunkBytes, kRanks, 2);
  ginAllReduceGinScatterChunkPlan(512ull * kMiB, kRanks, smallPipe, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 32);
  EXPECT_EQ(nPhases, 2);
  EXPECT_EQ(uniformBytes, kGinAllReduceGinScatterChunkBytes);
}

TEST(GinAllReducePolicy, GinScatterNoPipelineBelow512MiB) {
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  const size_t scratch = kGinAllReduceTwoShotScratchBytes;
  ginAllReduceGinScatterChunkPlan((64ull * kMiB) / kRanks, kRanks, scratch, &nChunks, &uniformBytes, &lastBytes,
                                  &nPhases);
  EXPECT_EQ(nChunks, 1);
  EXPECT_EQ(nPhases, 1);
  EXPECT_EQ(uniformBytes, 8ull * kMiB);
  ginAllReduceGinScatterChunkPlan(32ull * kMiB, kRanks, scratch, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 1);
  EXPECT_EQ(nPhases, 1);
  EXPECT_EQ(uniformBytes, 32ull * kMiB);
  ginAllReduceGinScatterChunkPlan(63ull * kMiB, kRanks, scratch, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 1);
  EXPECT_EQ(nPhases, 1);
}

TEST(GinAllReducePolicy, GinScatterChunksOverlapAt512MiB) {
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  ginAllReduceGinScatterChunkPlan(64ull * kMiB, kRanks, kGinAllReduceTwoShotScratchBytes, &nChunks, &uniformBytes,
                                  &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 4);
  EXPECT_EQ(nPhases, 1);
  EXPECT_EQ(uniformBytes, kGinAllReduceGinScatterChunkBytes);
  EXPECT_EQ(lastBytes, kGinAllReduceGinScatterChunkBytes);
}

TEST(GinAllReducePolicy, GinScatterStagedPingPongAt1GiB) {
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  ginAllReduceGinScatterChunkPlan(128ull * kMiB, kRanks, kGinAllReduceTwoShotScratchBytes, &nChunks, &uniformBytes,
                                  &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 4);
  EXPECT_EQ(nPhases, 2);
  EXPECT_EQ(uniformBytes, 32ull * kMiB);
  EXPECT_EQ(lastBytes, 32ull * kMiB);
}

TEST(GinAllReducePolicy, GinScatterSingleChunkWhenAtMostChunkSize) {
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  ginAllReduceGinScatterChunkPlan(kGinAllReduceGinScatterChunkBytes, kRanks, kGinAllReduceTwoShotScratchBytes, &nChunks,
                                  &uniformBytes, &lastBytes, &nPhases);
  EXPECT_EQ(nChunks, 1);
  EXPECT_EQ(nPhases, 1);
  EXPECT_EQ(uniformBytes, kGinAllReduceGinScatterChunkBytes);
  ginAllReduceGinScatterChunkPlan(0, kRanks, kGinAllReduceTwoShotScratchBytes, &nChunks, &uniformBytes, &lastBytes,
                                  &nPhases);
  EXPECT_EQ(nChunks, 1);
  EXPECT_EQ(uniformBytes, 0u);
}

TEST(GinAllReducePolicy, GinScatterChunksTilePerRankBytes) {
  const size_t perRank = 132ull * kMiB;
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  ginAllReduceGinScatterChunkPlan(perRank, kRanks, kGinAllReduceTwoShotScratchBytes, &nChunks, &uniformBytes, &lastBytes,
                                  &nPhases);
  EXPECT_GT(nChunks, 1);
  EXPECT_EQ(nChunks, 5);
  EXPECT_EQ(nPhases, 2);
  EXPECT_EQ(uniformBytes, 32ull * kMiB);
  EXPECT_EQ(lastBytes, 4ull * kMiB);
  EXPECT_EQ(uniformBytes * static_cast<size_t>(nChunks - 1) + lastBytes, perRank);
  EXPECT_EQ(ginAllReduceGinScatterChunkOff(nChunks - 1, uniformBytes),
            uniformBytes * static_cast<size_t>(nChunks - 1));
  EXPECT_EQ(ginAllReduceGinScatterChunkSize(0, nChunks, uniformBytes, lastBytes), uniformBytes);
  EXPECT_EQ(ginAllReduceGinScatterChunkSize(nChunks - 1, nChunks, uniformBytes, lastBytes), lastBytes);
}

TEST(GinAllReducePolicy, GinScatterScratchIsIncomingSlots) {
  EXPECT_EQ(ginAllReduceGinScatterScratchBytes(32ull * kMiB, kRanks), 8ull * 32ull * kMiB);
  EXPECT_EQ(ginAllReduceGinScatterStagedScratchBytes(8ull * kMiB, kRanks, 2), 16ull * 8ull * kMiB);
  EXPECT_EQ(ginAllReduceGinScatterScratchBytes(16, 1), 16u);
  EXPECT_EQ(ginAllReduceGinScatterScratchBytes(16, 0), 0u);
  EXPECT_EQ(ginAllReduceGinScatterFitSlot(kGinAllReduceTwoShotScratchBytes, kRanks, 1), 64ull * kMiB);
  EXPECT_EQ(ginAllReduceGinScatterFitSlot(kGinAllReduceTwoShotScratchBytes, kRanks, 2), 32ull * kMiB);
}

} // namespace
