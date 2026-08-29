/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for the GIN Anvil-SDMA ReduceScatter policy helpers
// (gin_sdma_reducescatter_policy.h). No GPU required: GIN_SDMA_HOST_ONLY drops
// the HIP attributes so the header compiles as plain C++. These validate the
// exact per-rank slice sizing, size-adaptive CTA schedule, devComm requirement
// shape, threshold precedence and bandwidth math that reduce_scatter.cu
// (GinReduceScatterKernel) relies on. Companion to gin_sdma_allgather_policy_test.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#define GIN_SDMA_HOST_ONLY 1
#include "gin_sdma_reducescatter_policy.h"

using namespace gin_sdma_reducescatter;

namespace {

// ---- sliceBaseCount: floor(count/nranks) aligned down to 16 bytes ------------

TEST(ReduceScatterPolicySliceBase, AlignsDownTo16Bytes) {
  // eltSize=4 (int/float): 16 B == 4 elements, so base is a multiple of 4.
  EXPECT_EQ(sliceBaseCount(4096, 4, 8), 512u);      // 4096/8=512, already 4-aligned
  EXPECT_EQ(sliceBaseCount(4095, 4, 8), 508u);      // 4095/8=511 -> down to 508
  EXPECT_EQ(sliceBaseCount(100, 4, 8), 12u);        // 100/8=12 (mult of 4)
  EXPECT_EQ(sliceBaseCount(104, 4, 8), 12u);        // 104/8=13 -> down to 12
}

TEST(ReduceScatterPolicySliceBase, EltSizeControlsAlignment) {
  // eltSize=1 (int8): 16 B == 16 elements, so base is a multiple of 16.
  EXPECT_EQ(sliceBaseCount(1024, 1, 8), 128u);      // 1024/8=128 (mult of 16)
  EXPECT_EQ(sliceBaseCount(1000, 1, 8), 112u);      // 1000/8=125 -> down to 112
  // eltSize=8 (double/int64): 16 B == 2 elements.
  EXPECT_EQ(sliceBaseCount(1000, 8, 8), 124u);      // 1000/8=125 -> down to 124
  // eltSize=16: 16 B == 1 element, no alignment loss.
  EXPECT_EQ(sliceBaseCount(1000, 16, 8), 125u);
}

TEST(ReduceScatterPolicySliceBase, DegenerateInputsReturnZero) {
  EXPECT_EQ(sliceBaseCount(4096, 4, 0), 0u);        // nranks == 0
  EXPECT_EQ(sliceBaseCount(4096, 4, -1), 0u);       // nranks < 0
  EXPECT_EQ(sliceBaseCount(4096, 0, 8), 0u);        // eltSize == 0
  EXPECT_EQ(sliceBaseCount(4096, 32, 8), 0u);       // eltSize > 16 (unsupported)
}

// ---- sliceBytes --------------------------------------------------------------

TEST(ReduceScatterPolicySliceBytes, ElementsTimesSize) {
  EXPECT_EQ(sliceBytes(512, 4), 2048u);
  EXPECT_EQ(sliceBytes(0, 4), 0u);
  EXPECT_EQ(sliceBytes(128, 1), 128u);
}

// ---- reduceScatterKernelTier: slice <= threshold is LSA (reserved) -----------

TEST(ReduceScatterPolicyTier, BelowOrEqualThresholdIsLsa) {
  EXPECT_EQ(reduceScatterKernelTier(0, 262144), RSTier::LSA);
  EXPECT_EQ(reduceScatterKernelTier(262144, 262144), RSTier::LSA);   // boundary
}

TEST(ReduceScatterPolicyTier, AboveThresholdIsGin) {
  EXPECT_EQ(reduceScatterKernelTier(262145, 262144), RSTier::Gin);
  EXPECT_EQ(reduceScatterKernelTier(1, 0), RSTier::Gin);             // threshold 0
  EXPECT_EQ(reduceScatterKernelTier(0, 0), RSTier::LSA);             // empty stays LSA
}

// ---- pickSdmaThreshold: per-collective > global env > compiled default -------

TEST(ReduceScatterPolicyThreshold, PerCollectiveWins) {
  EXPECT_EQ(pickSdmaThreshold(/*perCollSet=*/true, /*perCollVal=*/4096,
                              /*globalSet=*/true, /*globalVal=*/65536,
                              /*compiledDefault=*/kReduceScatterSdmaThresholdDefault),
            4096u);
}

TEST(ReduceScatterPolicyThreshold, GlobalUsedWhenNoPerCollective) {
  EXPECT_EQ(pickSdmaThreshold(false, 0, true, 65536, kReduceScatterSdmaThresholdDefault), 65536u);
}

TEST(ReduceScatterPolicyThreshold, CompiledDefaultWhenNothingSet) {
  EXPECT_EQ(pickSdmaThreshold(false, 0, false, 0, kReduceScatterSdmaThresholdDefault),
            kReduceScatterSdmaThresholdDefault);
  EXPECT_EQ(kReduceScatterSdmaThresholdDefault, 262144u);  // 256 KiB/rank slice
}

TEST(ReduceScatterPolicyThreshold, ExplicitZeroIsHonored) {
  EXPECT_EQ(pickSdmaThreshold(true, 0, true, 65536, kReduceScatterSdmaThresholdDefault), 0u);
  EXPECT_EQ(pickSdmaThreshold(false, 0, true, 0, kReduceScatterSdmaThresholdDefault), 0u);
}

TEST(ReduceScatterPolicyThreshold, LargeValueDoesNotWrap) {
  const unsigned long long big = 4ull * 1024 * 1024 * 1024;  // 4 GiB
  EXPECT_EQ(pickSdmaThreshold(true, big, false, 0, kReduceScatterSdmaThresholdDefault),
            (size_t)big);
}

// ---- reduceScatterCtas: size-adaptive ladder + env override ------------------

TEST(ReduceScatterPolicyCtas, MidBandUses48) {
  // [8 MiB, 48 MiB) -> the grid-stride mid band (48 CTAs).
  EXPECT_EQ(reduceScatterCtas(8ull * 1024 * 1024, kThresholdUnset), 48);
  EXPECT_EQ(reduceScatterCtas(33ull * 1024 * 1024, kThresholdUnset), 48);
  EXPECT_EQ(reduceScatterCtas(48ull * 1024 * 1024 - 1, kThresholdUnset), 48);
}

TEST(ReduceScatterPolicyCtas, SmallAndLargeUse32) {
  EXPECT_EQ(reduceScatterCtas(0, kThresholdUnset), 32);
  EXPECT_EQ(reduceScatterCtas(1ull * 1024 * 1024, kThresholdUnset), 32);      // < 8 MiB
  EXPECT_EQ(reduceScatterCtas(8ull * 1024 * 1024 - 1, kThresholdUnset), 32);  // just below mid
  EXPECT_EQ(reduceScatterCtas(48ull * 1024 * 1024, kThresholdUnset), 32);     // >= mid hi (large)
  EXPECT_EQ(reduceScatterCtas(2ull * 1024 * 1024 * 1024, kThresholdUnset), 32);
}

TEST(ReduceScatterPolicyCtas, EnvOverridePinsAllSizesClampedTo128) {
  EXPECT_EQ(reduceScatterCtas(33ull * 1024 * 1024, 16), 16);   // pin below the ladder
  EXPECT_EQ(reduceScatterCtas(1024, 64), 64);
  EXPECT_EQ(reduceScatterCtas(1024, 256), 128);                // clamp to 128
  // unset/zero env falls back to the size-adaptive ladder.
  EXPECT_EQ(reduceScatterCtas(33ull * 1024 * 1024, 0), 48);
}

TEST(ReduceScatterPolicyCtas, MaxCtasIsTheMidBandValue) {
  EXPECT_EQ(reduceScatterMaxCtas(), 48);
}

// ---- reduceScatterDevReqs: one barrier/lsaBarrier/signal per CTA, needs GIN --

TEST(ReduceScatterPolicyDevReqs, PerCtaBarriersNeedGin) {
  DevReqs r = reduceScatterDevReqs(32);
  EXPECT_EQ(r.barrierCount, 32);
  EXPECT_EQ(r.lsaBarrierCount, 32);
  EXPECT_EQ(r.ginSignalCount, 32);
  EXPECT_TRUE(r.needsGin);
  EXPECT_TRUE(r.supported);
}

// ---- reduceScatterScratchBytes: 128 B-rounded per-rank sendbuf (reserved) ----

TEST(ReduceScatterPolicyScratch, RoundsUpTo128AndZeroStaysZero) {
  EXPECT_EQ(reduceScatterScratchBytes(0), 0u);
  EXPECT_EQ(reduceScatterScratchBytes(1), 128u);
  EXPECT_EQ(reduceScatterScratchBytes(128), 128u);
  EXPECT_EQ(reduceScatterScratchBytes(129), 256u);
}

// ---- bandwidthGBps: algBw counts all ranks, busBw applies (n-1)/n ------------

TEST(ReduceScatterPolicyBandwidth, AlgAndBusBandwidth) {
  double alg = -1.0, bus = -1.0;
  // perRankCount=1e9 elts, 1 B each, 8 ranks, 1 s -> base = 8 GB/s.
  bandwidthGBps(1000000000ull, 1, 1.0, 8, &alg, &bus);
  EXPECT_DOUBLE_EQ(alg, 8.0);
  EXPECT_DOUBLE_EQ(bus, 8.0 * 7.0 / 8.0);
}

TEST(ReduceScatterPolicyBandwidth, NullPointersAreSafe) {
  bandwidthGBps(1000, 4, 0.5, 4, nullptr, nullptr);
  double bus = -1.0;
  bandwidthGBps(1000, 4, 0.5, 4, nullptr, &bus);
  EXPECT_GT(bus, 0.0);
  double alg = -1.0;
  bandwidthGBps(1000, 4, 0.5, 4, &alg, nullptr);
  EXPECT_GT(alg, 0.0);
}

}  // namespace
