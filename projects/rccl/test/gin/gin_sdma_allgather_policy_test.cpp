/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for the GIN Anvil-SDMA hybrid AllGather policy helpers
// (gin_sdma_allgather_policy.h). No GPU required: GIN_SDMA_HOST_ONLY drops the
// HIP attributes so the header compiles as plain C++. These validate the exact
// message sizing, LSA<->SDMA tier crossover, context-threshold fallback and
// bandwidth math that all_gather.cu (GinHybridAllGatherKernel) relies on, plus
// the grid and in-place signal accounting gather.cu (GinHybridGatherKernel) reuses.

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

// ---- archSdmaThreshold: per-architecture compiled default ----------------------

constexpr ArchSdmaThreshold kArchTable[] = {{"gfx942", 1u << 25}, {"gfx1250", 1u << 17}};
constexpr size_t kArchTableSize = sizeof(kArchTable) / sizeof(kArchTable[0]);
constexpr size_t kArchFallback = 1u << 20;

TEST(AllGatherPolicyArchThreshold, ExactNameMatches) {
  EXPECT_EQ(archSdmaThreshold("gfx942", kArchTable, kArchTableSize, kArchFallback), 1u << 25);
  EXPECT_EQ(archSdmaThreshold("gfx1250", kArchTable, kArchTableSize, kArchFallback), 1u << 17);
}

TEST(AllGatherPolicyArchThreshold, FeatureAndVariantSuffixesMatch) {
  EXPECT_EQ(archSdmaThreshold("gfx942:sramecc+:xnack-", kArchTable, kArchTableSize, kArchFallback),
            1u << 25);
  EXPECT_EQ(archSdmaThreshold("gfx1250-strict", kArchTable, kArchTableSize, kArchFallback), 1u << 17);
}

TEST(AllGatherPolicyArchThreshold, PrefixOfALongerNameDoesNotMatch) {
  EXPECT_EQ(archSdmaThreshold("gfx9420", kArchTable, kArchTableSize, kArchFallback), kArchFallback);
  EXPECT_EQ(archSdmaThreshold("gfx94", kArchTable, kArchTableSize, kArchFallback), kArchFallback);
  EXPECT_EQ(archSdmaThreshold("gfx12", kArchTable, kArchTableSize, kArchFallback), kArchFallback);
}

TEST(AllGatherPolicyArchThreshold, UnknownOrMissingNameFallsBack) {
  EXPECT_EQ(archSdmaThreshold("gfx950", kArchTable, kArchTableSize, kArchFallback), kArchFallback);
  EXPECT_EQ(archSdmaThreshold("", kArchTable, kArchTableSize, kArchFallback), kArchFallback);
  EXPECT_EQ(archSdmaThreshold(nullptr, kArchTable, kArchTableSize, kArchFallback), kArchFallback);
  EXPECT_EQ(archSdmaThreshold("gfx942", nullptr, 0, kArchFallback), kArchFallback);
}

TEST(AllGatherPolicyArchThreshold, EnvOverridesStillWinOverArchDefault) {
  const size_t arch = archSdmaThreshold("gfx1250-strict", kArchTable, kArchTableSize, kArchFallback);
  EXPECT_EQ(pickSdmaThreshold(true, 0, false, 0, arch), 0u);
  EXPECT_EQ(pickSdmaThreshold(false, 0, true, 4096, arch), 4096u);
  EXPECT_EQ(pickSdmaThreshold(false, 0, false, 0, arch), 1u << 17);
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

// ---- Gather -D 3: one SDMA CTA, same LSA grid and pool clamp ------------------

TEST(GatherPolicyCtas, SdmaTierUsesOneCta) {
  const int pool = allGatherPoolCtas(16);
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, kAllGatherCtasUnset, pool, 1), 1);
  EXPECT_EQ(allGatherCtas(1024, 32768, kAllGatherCtasUnset, pool, 1), kAllGatherCtasLsa);
}

TEST(GatherPolicyCtas, EnvPinStillWinsAndIsClamped) {
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, 8, allGatherPoolCtas(64), 1), 8);
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, 1000, allGatherPoolCtas(16), 1), 16);
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, 0, allGatherPoolCtas(16), 1), 1);
}

TEST(GatherPolicyCtas, LsaTierUsesPassedCtaCount) {
  const int pool = allGatherPoolCtas(16);
  EXPECT_EQ(allGatherCtas(1024, 32768, kAllGatherCtasUnset, pool, 1, 2), 2);
  EXPECT_EQ(allGatherCtas(64ull << 20, 32768, kAllGatherCtasUnset, pool, 1, 2), 1);  // SDMA tier
  EXPECT_EQ(allGatherCtas(1024, 32768, 4, pool, 1, 2), 4);                           // pin wins
  EXPECT_EQ(allGatherCtas(1024, 32768, kAllGatherCtasUnset, pool, 1, 1000), pool);   // clamped
  EXPECT_EQ(allGatherCtas(1024, 32768, kAllGatherCtasUnset, pool, 1, 0), 1);         // floor of 1
}

// ---- lsaCtasFromLadder: per-size LSA-tier CTA count --------------------------

constexpr LsaCtaStep kLadder[] = {{8u << 10, 1}, {16u << 10, 2}};
constexpr size_t kLadderSize = sizeof(kLadder) / sizeof(kLadder[0]);

TEST(GatherPolicyLsaLadder, StepBoundariesAreInclusive) {
  EXPECT_EQ(lsaCtasFromLadder(0, kLadder, kLadderSize, 8), 1);
  EXPECT_EQ(lsaCtasFromLadder(8u << 10, kLadder, kLadderSize, 8), 1);
  EXPECT_EQ(lsaCtasFromLadder((8u << 10) + 1, kLadder, kLadderSize, 8), 2);
  EXPECT_EQ(lsaCtasFromLadder(16u << 10, kLadder, kLadderSize, 8), 2);
  EXPECT_EQ(lsaCtasFromLadder((16u << 10) + 1, kLadder, kLadderSize, 8), 8);
  EXPECT_EQ(lsaCtasFromLadder(512u << 10, kLadder, kLadderSize, 8), 8);
}

TEST(GatherPolicyLsaLadder, EmptyLadderUsesAbove) {
  EXPECT_EQ(lsaCtasFromLadder(1024, nullptr, 0, kAllGatherCtasLsa), kAllGatherCtasLsa);
  EXPECT_EQ(lsaCtasFromLadder(1024, kLadder, 0, kAllGatherCtasLsa), kAllGatherCtasLsa);
}

TEST(GatherPolicyLsaLadder, LadderFitsDefaultPool) {
  const int pool = allGatherPoolCtas(16);
  for (size_t c : {size_t{0}, size_t{4096}, size_t{12288}, size_t{1} << 19}) {
    const int g = allGatherCtas(c, 512u << 10, kAllGatherCtasUnset, pool, 1,
                                lsaCtasFromLadder(c, kLadder, kLadderSize, 8));
    EXPECT_GE(g, 1);
    EXPECT_LE(g, pool);
  }
}

TEST(GatherPolicyArch, GcnArchMatchesBaseNameOnly) {
  EXPECT_TRUE(gcnArchMatches("gfx1250", "gfx1250"));
  EXPECT_TRUE(gcnArchMatches("gfx1250-strict", "gfx1250"));
  EXPECT_TRUE(gcnArchMatches("gfx942:sramecc+:xnack-", "gfx942"));
  EXPECT_FALSE(gcnArchMatches("gfx12500", "gfx1250"));
  EXPECT_FALSE(gcnArchMatches("gfx125", "gfx1250"));
  EXPECT_FALSE(gcnArchMatches(nullptr, "gfx1250"));
  EXPECT_FALSE(gcnArchMatches("gfx1250", nullptr));
}

// ---- Gather -D 3 in-place: root skips its self-put and waits for one fewer ----

TEST(GatherPolicyInPlace, RootSkipsSelfPutOnlyInPlace) {
  const size_t chunk = 4096;
  const int root = 2;
  // In-place root: same window, send slice == root's recv slice (with a shift).
  EXPECT_TRUE(gatherRootSkipsSelfPut(true, true, root * chunk, 0, root, chunk));
  EXPECT_TRUE(gatherRootSkipsSelfPut(true, true, 512 + root * chunk, 512, root, chunk));
  // Out-of-place root: different windows, or the same window at another offset.
  EXPECT_FALSE(gatherRootSkipsSelfPut(true, false, root * chunk, 0, root, chunk));
  EXPECT_FALSE(gatherRootSkipsSelfPut(true, true, 0, 0, root, chunk));
  // Non-root ranks always put, even at the offset that would make root skip.
  EXPECT_FALSE(gatherRootSkipsSelfPut(false, true, root * chunk, 0, root, chunk));
  EXPECT_FALSE(gatherRootSkipsSelfPut(false, true, 512 + root * chunk, 512, root, chunk));
}

TEST(GatherPolicyInPlace, RootWaitsForOneSignalPerPut) {
  EXPECT_EQ(gatherRootExpectedSignals(8, false), 8u);
  EXPECT_EQ(gatherRootExpectedSignals(8, true), 7u);
  EXPECT_EQ(gatherRootExpectedSignals(1, false), 1u);
  EXPECT_EQ(gatherRootExpectedSignals(1, true), 0u);
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

  setenv("NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS", "8foo", 1);
  EXPECT_EQ(parseAllGatherCtasEnv(), kAllGatherCtasUnset);
  unsetenv("NCCL_GIN_ANVIL_SDMA_ALLGATHER_CTAS");
}

TEST(GatherPolicyEnv, ResolveThresholdReadsGatherEnv) {
  EXPECT_STREQ(kGatherSdmaThresholdEnv, "NCCL_GIN_ANVIL_SDMA_THRESHOLD_GATHER");
  const size_t archDefault = 512u << 10;
  unsetenv(kGatherSdmaThresholdEnv);
  unsetenv(kAllGatherSdmaThresholdEnv);
  unsetenv(kSdmaThresholdEnv);
  EXPECT_EQ(resolveSdmaThresholdFromEnv(kGatherSdmaThresholdEnv, archDefault), archDefault);

  setenv(kSdmaThresholdEnv, "4096", 1);
  EXPECT_EQ(resolveSdmaThresholdFromEnv(kGatherSdmaThresholdEnv, archDefault), 4096u);

  // The AllGather variable must not leak into Gather.
  setenv(kAllGatherSdmaThresholdEnv, "8192", 1);
  EXPECT_EQ(resolveSdmaThresholdFromEnv(kGatherSdmaThresholdEnv, archDefault), 4096u);

  setenv(kGatherSdmaThresholdEnv, "0", 1);
  EXPECT_EQ(resolveSdmaThresholdFromEnv(kGatherSdmaThresholdEnv, archDefault), 0u);

  unsetenv(kGatherSdmaThresholdEnv);
  unsetenv(kAllGatherSdmaThresholdEnv);
  unsetenv(kSdmaThresholdEnv);
}

TEST(GatherPolicyEnv, ParseGatherCtasEnvHasNoAlias) {
  const char* kName = "NCCL_GIN_ANVIL_SDMA_GATHER_CTAS";
  unsetenv(kName);
  setenv("NCCL_GIN_ANVIL_AG_CTAS", "4", 1);
  EXPECT_EQ(parseCtasEnv(kName), kAllGatherCtasUnset);
  unsetenv("NCCL_GIN_ANVIL_AG_CTAS");

  setenv(kName, "1", 1);
  EXPECT_EQ(parseCtasEnv(kName), 1u);
  setenv(kName, "-2", 1);
  EXPECT_EQ(parseCtasEnv(kName), kAllGatherCtasUnset);
  setenv(kName, "2x", 1);
  EXPECT_EQ(parseCtasEnv(kName), kAllGatherCtasUnset);
  unsetenv(kName);
}

}  // namespace
