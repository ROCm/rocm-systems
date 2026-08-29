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
  EXPECT_EQ(kGinAllReduceLsaOneShotMaxBytes, 8ull * kMiB);
  EXPECT_EQ(kGinAllReduceGinTwoShotMinBytes, 256ull * kMiB);
  EXPECT_EQ(kGinAllReduceMaxRanks, 8);
  EXPECT_EQ(kGinAllReduceMinPutBytes, 128u);
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
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(4 * kMiB), kFloat, kRanks, true));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(8 * kMiB), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceAcceptsAlignedTwoShotBand) {
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(16 * kMiB), kFloat, kRanks, true));
  EXPECT_TRUE(ginAllReduceSizePolicyEligible(countForBytes(128 * kMiB), kFloat, kRanks, true));
}

TEST(GinAllReducePolicy, ForceRejectsUnalignedTwoShotBand) {
  // > 8 MiB so LSA two-shot; per-rank slice is not 16-byte aligned.
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

} // namespace
