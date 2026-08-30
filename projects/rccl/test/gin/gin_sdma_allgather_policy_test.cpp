/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdlib>
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

// ---- pickSdmaThreshold: per-collective > global env > compiled default -------

TEST(AllGatherPolicyThreshold, PerCollectiveWins) {
  // Per-collective override dominates the global env and the compiled default.
  EXPECT_EQ(pickSdmaThreshold(/*perCollSet=*/true, /*perCollVal=*/4096,
                              /*globalSet=*/true, /*globalVal=*/65536,
                              /*compiledDefault=*/kAllGatherSdmaThresholdDefault),
            4096u);
}

TEST(AllGatherPolicyThreshold, GlobalUsedWhenNoPerCollective) {
  EXPECT_EQ(pickSdmaThreshold(false, 0, true, 65536, kAllGatherSdmaThresholdDefault), 65536u);
}

TEST(AllGatherPolicyThreshold, DefaultCrossoverAt32KiB) {
  EXPECT_TRUE(chunkUsesLsaTier(32768, kAllGatherSdmaThresholdDefault));
  EXPECT_FALSE(chunkUsesLsaTier(32769, kAllGatherSdmaThresholdDefault));
}

TEST(AllGatherPolicyThreshold, CompiledDefaultWhenNothingSet) {
  // Behavioral: with nothing set, resolution returns the compiled default (rather
  // than mirroring the constant's value, which only trips on a deliberate renumber).
  EXPECT_EQ(pickSdmaThreshold(false, 0, false, 0, kAllGatherSdmaThresholdDefault),
            kAllGatherSdmaThresholdDefault);
}

TEST(AllGatherPolicyThreshold, ExplicitZeroIsHonored) {
  // An explicit 0 (all-SDMA tier) is honored at either precedence level.
  EXPECT_EQ(pickSdmaThreshold(true, 0, true, 65536, kAllGatherSdmaThresholdDefault), 0u);
  EXPECT_EQ(pickSdmaThreshold(false, 0, true, 0, kAllGatherSdmaThresholdDefault), 0u);
}

TEST(AllGatherPolicyThreshold, LargeValueDoesNotWrap) {
  // Host resolution is 64-bit, so a >2 GiB threshold survives (unlike the
  // backend's int-typed inline-put threshold).
  const unsigned long long big = 4ull * 1024 * 1024 * 1024;  // 4 GiB
  EXPECT_EQ(pickSdmaThreshold(true, big, false, 0, kAllGatherSdmaThresholdDefault),
            (size_t)big);
}

// ---- allGatherCtas: size-adaptive ladder, clamped to the launched pool --------
// The launched grid must never exceed the barrier/lsaBarrier/signal pool the kernel
// indexes by blockIdx.x. allGatherPoolCtas() == that pool (max(-V, ladder peak));
// each allGatherCtas() result is a grid that must fit inside it.

TEST(AllGatherPolicyCtas, LsaTierUsesLsaCtaCount) {
  const int pool = allGatherPoolCtas(16);  // pool covers the ladder peak
  // chunk <= threshold -> LSA-direct tier -> kAllGatherCtasLsa.
  EXPECT_EQ(allGatherCtas(0, 32768, kAllGatherCtasUnset, pool), kAllGatherCtasLsa);
  EXPECT_EQ(allGatherCtas(32768, 32768, kAllGatherCtasUnset, pool), kAllGatherCtasLsa);  // boundary
  EXPECT_EQ(allGatherCtas(1024, 32768, kAllGatherCtasUnset, pool), kAllGatherCtasLsa);
}

TEST(AllGatherPolicyCtas, SdmaTierUsesSdmaCtaCount) {
  const int pool = allGatherPoolCtas(16);
  // chunk > threshold -> GIN-put/SDMA tier -> kAllGatherCtasSdma (few CTAs).
  EXPECT_EQ(allGatherCtas(32769, 32768, kAllGatherCtasUnset, pool), kAllGatherCtasSdma);
  EXPECT_EQ(allGatherCtas(64ull * 1024 * 1024, 32768, kAllGatherCtasUnset, pool), kAllGatherCtasSdma);
}

TEST(AllGatherPolicyCtas, TracksThresholdOverride) {
  const int pool = allGatherPoolCtas(16);
  // The ladder keys off the SAME predicate as the kernel, so a moved crossover
  // moves the CTA choice: at threshold 0 (all-SDMA) any non-empty chunk is SDMA;
  // at a huge threshold (all-LSA) even a large chunk is LSA.
  EXPECT_EQ(allGatherCtas(1, 0, kAllGatherCtasUnset, pool), kAllGatherCtasSdma);
  EXPECT_EQ(allGatherCtas(64ull * 1024 * 1024, 1ull << 40, kAllGatherCtasUnset, pool), kAllGatherCtasLsa);
}

TEST(AllGatherPolicyCtas, EnvPinHonoredWithinPool) {
  const int pool = allGatherPoolCtas(/*deviceCtaCount=*/64);  // 64 slots
  // A set env value pins a fixed count for both tiers, as long as it fits the pool.
  EXPECT_EQ(allGatherCtas(1024, 32768, 8, pool), 8);            // LSA size, pinned 8
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, 8, pool), 8);     // SDMA size, pinned 8
  // An env of 0 is "not pinned" -> fall back to the size-adaptive ladder.
  EXPECT_EQ(allGatherCtas(1024, 32768, 0, pool), kAllGatherCtasLsa);
}

TEST(AllGatherPolicyCtas, EnvPinClampedToPool) {
  // A pin larger than the pool is clamped to the pool (not the old fixed 128), so
  // -V governs the real ceiling and the pin can never index past the pool.
  EXPECT_EQ(allGatherCtas(1024, 32768, /*pin=*/1000, allGatherPoolCtas(16)), 16);
  EXPECT_EQ(allGatherCtas(1024, 32768, /*pin=*/1000, allGatherPoolCtas(64)), 64);
}

TEST(AllGatherPolicyCtas, GridNeverExceedsPool) {
  // Regression for the OOB the env pin used to allow: for ANY chunk, tier, -V, and
  // env pin (including absurd ones), the launched grid stays within [1, pool].
  const size_t chunks[] = {0, 1024, 32768, 32769, 64ull << 20};
  const int deviceVs[] = {1, 4, 16, 32, 128};
  const size_t pins[]  = {kAllGatherCtasUnset, 0, 1, 8, 200, 100000};
  for (int v : deviceVs) {
    const int pool = allGatherPoolCtas(v);
    for (size_t c : chunks) {
      for (size_t pin : pins) {
        const int g = allGatherCtas(c, 32768, pin, pool);
        EXPECT_GE(g, 1);
        EXPECT_LE(g, pool) << "chunk=" << c << " V=" << v << " pin=" << pin;
      }
    }
  }
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

// ---- host env resolution (resolveSdmaThresholdFromEnv, parseAllGatherCtasEnv) --

TEST(AllGatherPolicyEnv, ResolveThresholdFromEnv) {
  unsetenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER");
  unsetenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD");
  EXPECT_EQ(resolveSdmaThresholdFromEnv(), kAllGatherSdmaThresholdDefault);

  setenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD", "4096", 1);
  unsetenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER");
  EXPECT_EQ(resolveSdmaThresholdFromEnv(), 4096u);

  setenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER", "8192", 1);
  EXPECT_EQ(resolveSdmaThresholdFromEnv(), 8192u);

  unsetenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD");
  unsetenv("NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER");
}

TEST(AllGatherPolicyEnv, ParseAllGatherCtasEnv) {
  unsetenv("NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS");
  unsetenv("NCCL_GIN_ANVIL_AG_CTAS");
  EXPECT_EQ(parseAllGatherCtasEnv(), kAllGatherCtasUnset);

  setenv("NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS", "8", 1);
  EXPECT_EQ(parseAllGatherCtasEnv(), 8u);

  unsetenv("NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS");
  setenv("NCCL_GIN_ANVIL_AG_CTAS", "4", 1);
  EXPECT_EQ(parseAllGatherCtasEnv(), 4u);

  unsetenv("NCCL_GIN_ANVIL_AG_CTAS");
}

}  // namespace
