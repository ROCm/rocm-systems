/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for the GIN Anvil-SDMA hybrid AllGather policy helpers
// (gin_sdma_allgather_policy.h). No GPU required: GIN_SDMA_HOST_ONLY drops the
// HIP attributes so the header compiles as plain C++. These validate the exact
// message sizing, LSA<->SDMA tier crossover, context-threshold fallback and
// bandwidth math that all_gather.cu (GinHybridAllGatherKernel) relies on.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#define GIN_SDMA_HOST_ONLY 1
#include "gin_sdma_allgather_policy.h"

using namespace gin_sdma_allgather;

namespace {

// ---- chunkBaseCount: floor(count/nranks) aligned down to 16 bytes ------------

TEST(AllGatherPolicyChunkBase, AlignsDownTo16Bytes) {
  // eltSize=4 (int/float): 16 B == 4 elements, so base is a multiple of 4.
  EXPECT_EQ(chunkBaseCount(4096, 4, 8), 512u);      // 4096/8=512, already 4-aligned
  EXPECT_EQ(chunkBaseCount(4095, 4, 8), 508u);      // 4095/8=511 -> down to 508
  EXPECT_EQ(chunkBaseCount(100, 4, 8), 12u);        // 100/8=12 (mult of 4)
  EXPECT_EQ(chunkBaseCount(104, 4, 8), 12u);        // 104/8=13 -> down to 12
}

TEST(AllGatherPolicyChunkBase, EltSizeControlsAlignment) {
  // eltSize=1 (int8): 16 B == 16 elements, so base is a multiple of 16.
  EXPECT_EQ(chunkBaseCount(1024, 1, 8), 128u);      // 1024/8=128 (mult of 16)
  EXPECT_EQ(chunkBaseCount(1000, 1, 8), 112u);      // 1000/8=125 -> down to 112
  // eltSize=8 (double/int64): 16 B == 2 elements.
  EXPECT_EQ(chunkBaseCount(1000, 8, 8), 124u);      // 1000/8=125 -> down to 124
  // eltSize=16: 16 B == 1 element, no alignment loss.
  EXPECT_EQ(chunkBaseCount(1000, 16, 8), 125u);
}

TEST(AllGatherPolicyChunkBase, DegenerateInputsReturnZero) {
  EXPECT_EQ(chunkBaseCount(4096, 4, 0), 0u);        // nranks == 0
  EXPECT_EQ(chunkBaseCount(4096, 4, -1), 0u);       // nranks < 0
  EXPECT_EQ(chunkBaseCount(4096, 0, 8), 0u);        // eltSize == 0
  EXPECT_EQ(chunkBaseCount(4096, 32, 8), 0u);       // eltSize > 16 (unsupported)
}

// ---- chunkBytes --------------------------------------------------------------

TEST(AllGatherPolicyChunkBytes, ElementsTimesSize) {
  EXPECT_EQ(chunkBytes(512, 4), 2048u);
  EXPECT_EQ(chunkBytes(0, 4), 0u);
  EXPECT_EQ(chunkBytes(128, 1), 128u);
}

// ---- chunkUsesLsaTier: chunk <= threshold takes LSA, above takes SDMA --------

TEST(AllGatherPolicyTier, BelowOrEqualThresholdIsLsa) {
  EXPECT_TRUE(chunkUsesLsaTier(0, 2097152));
  EXPECT_TRUE(chunkUsesLsaTier(128, 128));          // boundary: equal -> LSA
  EXPECT_TRUE(chunkUsesLsaTier(2097152, 2097152));
}

TEST(AllGatherPolicyTier, AboveThresholdIsSdma) {
  EXPECT_FALSE(chunkUsesLsaTier(129, 128));
  EXPECT_FALSE(chunkUsesLsaTier(2097153, 2097152));
  // threshold 0 (all-SDMA gate): every non-empty chunk goes to SDMA, and even
  // an empty chunk stays LSA (0 <= 0) which is harmless (no bytes to move).
  EXPECT_FALSE(chunkUsesLsaTier(1, 0));
  EXPECT_TRUE(chunkUsesLsaTier(0, 0));
}

// ---- resolveSdmaThreshold: context value only when present + magic valid -----

TEST(AllGatherPolicyThreshold, UsesContextWhenPresentAndValid) {
  EXPECT_EQ(resolveSdmaThreshold(/*ctxPresent=*/true, /*magicValid=*/true,
                                 /*ctxThreshold=*/4096, /*default=*/128),
            4096u);
}

TEST(AllGatherPolicyThreshold, FallsBackWhenContextAbsent) {
  EXPECT_EQ(resolveSdmaThreshold(false, false, 4096, 128), 128u);
  EXPECT_EQ(resolveSdmaThreshold(false, true, 4096, 128), 128u);  // absent dominates
}

TEST(AllGatherPolicyThreshold, FallsBackWhenMagicInvalid) {
  EXPECT_EQ(resolveSdmaThreshold(true, false, 4096, 128), 128u);
}

TEST(AllGatherPolicyThreshold, ContextZeroIsHonoredWhenValid) {
  // An explicit context threshold of 0 (all-SDMA) is honored, not overridden.
  EXPECT_EQ(resolveSdmaThreshold(true, true, 0, 128), 0u);
}

// ---- bandwidthGBps: algBw counts all ranks, busBw applies (n-1)/n ------------

TEST(AllGatherPolicyBandwidth, AlgAndBusBandwidth) {
  double alg = -1.0, bus = -1.0;
  // perRankCount=1e9 elts, 1 B each, 8 ranks, 1 s -> base = 8 GB/s.
  bandwidthGBps(1000000000ull, 1, 1.0, 8, &alg, &bus);
  EXPECT_DOUBLE_EQ(alg, 8.0);
  EXPECT_DOUBLE_EQ(bus, 8.0 * 7.0 / 8.0);
}

TEST(AllGatherPolicyBandwidth, NullPointersAreSafe) {
  // Exercise both null-guard branches without crashing.
  bandwidthGBps(1000, 4, 0.5, 4, nullptr, nullptr);
  double bus = -1.0;
  bandwidthGBps(1000, 4, 0.5, 4, nullptr, &bus);
  EXPECT_GT(bus, 0.0);
  double alg = -1.0;
  bandwidthGBps(1000, 4, 0.5, 4, &alg, nullptr);
  EXPECT_GT(alg, 0.0);
}

}  // namespace
