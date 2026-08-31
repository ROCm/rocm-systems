/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for the pure GIN Anvil-SDMA hybrid Broadcast policy
// helpers (gin_sdma_broadcast_policy.h). No GPU required: GIN_SDMA_HOST_ONLY
// drops the HIP attributes so the header compiles as plain host C++, and the
// same functions are called by broadcast.cu's __global__ kernels on device, so
// exercising them here exercises the real decision code. These cover the
// threshold/env resolution, LL/LSA/flat/scatter+allgather/ring tier selection,
// scatter/ring chunk + alignment math, and devComm signal-count decisions to
// high line+branch coverage. GPU data-movement correctness (the branch bodies)
// is covered separately by the broadcast_perf functional sweep.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#define GIN_SDMA_HOST_ONLY 1
#include "gin_sdma_broadcast_policy.h"

using namespace gin_sdma;

namespace {

constexpr size_t kUnset = kThresholdUnset;

// --------------------------------- parseSize ---------------------------------

TEST(ParseSize, NullAndEmptyAreUnset) {
  EXPECT_EQ(parseSize(nullptr), kUnset);
  EXPECT_EQ(parseSize(""), kUnset);
}

TEST(ParseSize, NonNumericIsUnset) {
  EXPECT_EQ(parseSize("abc"), kUnset);  // no digits consumed
  EXPECT_EQ(parseSize("K"), kUnset);    // suffix only, no digits
}

TEST(ParseSize, NegativeIsUnset) {
  // Without an explicit sign check strtoull() wraps these into near-SIZE_MAX
  // thresholds, which read as "always LSA" instead of "unset".
  EXPECT_EQ(parseSize("-1"), kUnset);
  EXPECT_EQ(parseSize("-4K"), kUnset);
  EXPECT_EQ(parseSize("-262144"), kUnset);
}

TEST(ParseSize, PlainDecimal) {
  EXPECT_EQ(parseSize("0"), 0u);
  EXPECT_EQ(parseSize("1024"), 1024u);
  EXPECT_EQ(parseSize("262144"), 262144u);
}

TEST(ParseSize, BinarySuffixesBothCases) {
  EXPECT_EQ(parseSize("4K"), 4u * 1024);
  EXPECT_EQ(parseSize("4k"), 4u * 1024);
  EXPECT_EQ(parseSize("2M"), 2u * 1024 * 1024);
  EXPECT_EQ(parseSize("2m"), 2u * 1024 * 1024);
  EXPECT_EQ(parseSize("1G"), 1024u * 1024 * 1024);
  EXPECT_EQ(parseSize("1g"), 1024u * 1024 * 1024);
}

TEST(ParseSize, UnknownSuffixIgnored) {
  EXPECT_EQ(parseSize("512B"), 512u);
  EXPECT_EQ(parseSize("10X"), 10u);
}

// ------------------------------ resolveThreshold -----------------------------

TEST(ResolveThreshold, CollectiveValueWins) {
  EXPECT_EQ(resolveThreshold(4096, 8192, 262144), 4096u);
}

TEST(ResolveThreshold, SharedUsedWhenCollUnset) {
  EXPECT_EQ(resolveThreshold(kUnset, 8192, 262144), 8192u);
}

TEST(ResolveThreshold, DefaultWhenBothUnset) {
  EXPECT_EQ(resolveThreshold(kUnset, kUnset, 262144), 262144u);
}

TEST(ResolveThreshold, ExplicitZeroIsHonored) {
  // Zero is a valid "force GIN/SDMA always" value, distinct from unset.
  EXPECT_EQ(resolveThreshold(0, kUnset, 262144), 0u);
  EXPECT_EQ(resolveThreshold(kUnset, 0, 262144), 0u);
}

// -------------------------------- resolveLLCap -------------------------------

TEST(ResolveLLCap, UnsetUsesDefaultThenAligns) {
  EXPECT_EQ(resolveLLCap(kUnset, kBroadcastLLDefaultMaxBytes, kBroadcastLLMaxBytes),
            2048u);
}

TEST(ResolveLLCap, ClampsToMax) {
  EXPECT_EQ(resolveLLCap(1u << 20, kBroadcastLLDefaultMaxBytes, kBroadcastLLMaxBytes),
            kBroadcastLLMaxBytes);
}

TEST(ResolveLLCap, RoundsDownToEightBytes) {
  EXPECT_EQ(resolveLLCap(1001, 0, 65536), 1000u);  // 1001 -> 1000
  EXPECT_EQ(resolveLLCap(1000, 0, 65536), 1000u);  // already aligned
  EXPECT_EQ(resolveLLCap(7, 0, 65536), 0u);        // below one slot
}

// ------------------------------ alignChunkCount ------------------------------

TEST(AlignChunkCount, GuardsInvalidInputs) {
  EXPECT_EQ(alignChunkCount(1000, 0, 4), 0u);
  EXPECT_EQ(alignChunkCount(1000, -1, 4), 0u);
  EXPECT_EQ(alignChunkCount(1000, 8, 0), 0u);
}

TEST(AlignChunkCount, MasksPerElementSize) {
  // eltSize 4 -> mask ~3: 1000/8=125 -> 124.
  EXPECT_EQ(alignChunkCount(1000, 8, 4), 124u);
  // eltSize 8 -> mask ~1: 125 -> 124.
  EXPECT_EQ(alignChunkCount(1000, 8, 8), 124u);
  // eltSize 1 -> mask ~15: 125 -> 112.
  EXPECT_EQ(alignChunkCount(1000, 8, 1), 112u);
  // eltSize 16 -> mask all ones: unchanged.
  EXPECT_EQ(alignChunkCount(1000, 8, 16), 125u);
  // eltSize 2 -> mask ~7: 125 -> 120.
  EXPECT_EQ(alignChunkCount(1000, 8, 2), 120u);
}

// -------------------------- bcastUseScatterAllgather -------------------------

TEST(BcastScatterAllgather, DisabledWhenSagMinZero) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 1u << 20, 8, 0));
}

TEST(BcastScatterAllgather, RequiresAtLeastTwoRanks) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 1u << 20, 1, 2u << 20));
}

TEST(BcastScatterAllgather, RequiresMessageAtOrAboveMin) {
  const size_t sagMin = 2u << 20;
  EXPECT_FALSE(bcastUseScatterAllgather(sagMin - 1, 1u << 20, 8, sagMin));
  EXPECT_TRUE(bcastUseScatterAllgather(sagMin, 1u << 20, 8, sagMin));
}

TEST(BcastScatterAllgather, RequiresCountAtLeastNRanks) {
  EXPECT_FALSE(bcastUseScatterAllgather(8u << 20, 7, 8, 1024));
  EXPECT_TRUE(bcastUseScatterAllgather(8u << 20, 8, 8, 1024));
}

// -------------------------------- bcastUseRing -------------------------------

TEST(BcastUseRing, DisabledWhenRingMinZero) {
  EXPECT_FALSE(bcastUseRing(64u << 20, 16u << 20, 8, 0));
}

TEST(BcastUseRing, RequiresAtLeastTwoRanks) {
  EXPECT_FALSE(bcastUseRing(64u << 20, 16u << 20, 1, kBroadcastRingMinDefault));
}

TEST(BcastUseRing, EngagesAtOrAboveMin) {
  const size_t ringMin = kBroadcastRingMinDefault;  // 32 MiB
  EXPECT_FALSE(bcastUseRing(ringMin - 1, 16u << 20, 8, ringMin));
  EXPECT_TRUE(bcastUseRing(ringMin, 16u << 20, 8, ringMin));
}

TEST(BcastUseRing, RequiresCountAtLeastNRanks) {
  EXPECT_FALSE(bcastUseRing(64u << 20, 7, 8, kBroadcastRingMinDefault));
  EXPECT_TRUE(bcastUseRing(64u << 20, 8, 8, kBroadcastRingMinDefault));
}

// ------------------------------- bcastRingChunks -----------------------------

TEST(BcastRingChunks, EnvOverrideWinsClampedToMax) {
  // Env pin honored, clamped to the compile ceiling.
  EXPECT_EQ(bcastRingChunks(2ull << 30, 128, 32), 32);
  EXPECT_EQ(bcastRingChunks(2ull << 30, 128, 100000),
            kBroadcastRingMaxChunks);
}

TEST(BcastRingChunks, SizeAdaptivePerCtaClampedToRange) {
  // Small stripe -> floored at the minimum depth.
  EXPECT_EQ(bcastRingChunks(1u << 20, 128, kUnset), kBroadcastRingMinChunks);
  // Very large message over few CTAs -> capped at the maximum depth.
  EXPECT_EQ(bcastRingChunks(2ull << 30, 1, kUnset), kBroadcastRingMaxChunks);
  // Mid message: ~64 KiB per chunk of this CTA's stripe.
  // 128 MiB / 8 CTAs = 16 MiB stripe; 16 MiB / 64 KiB = 256 -> capped to max.
  EXPECT_EQ(bcastRingChunks(128u << 20, 8, kUnset), kBroadcastRingMaxChunks);
}

TEST(BcastRingChunks, GuardsZeroCtas) {
  // ctas < 1 is clamped to 1 internally (no divide-by-zero).
  EXPECT_GE(bcastRingChunks(2ull << 30, 0, kUnset), kBroadcastRingMinChunks);
  EXPECT_LE(bcastRingChunks(2ull << 30, 0, kUnset), kBroadcastRingMaxChunks);
}

// ------------------------------ bcast LL / tier ------------------------------

TEST(BcastLLEligible, AllGates) {
  const size_t cap = kBroadcastLLMaxBytes;
  EXPECT_FALSE(bcastLLEligible(64, 0, cap));            // no slots
  EXPECT_FALSE(bcastLLEligible(60, 1000, cap));         // not 8-aligned
  EXPECT_FALSE(bcastLLEligible(cap + 8, 100000, cap));  // above ceiling
  EXPECT_FALSE(bcastLLEligible(800, 99, cap));          // 800/8=100 slots > 99
  EXPECT_TRUE(bcastLLEligible(800, 100, cap));          // exactly fits
  EXPECT_TRUE(bcastLLEligible(8, 1, cap));
}

TEST(BcastKernelTier, LadderSelection) {
  const size_t thr = 262144;
  const size_t cap = kBroadcastLLMaxBytes;
  EXPECT_EQ(bcastKernelTier(thr + 1, thr, 100000, cap), BcastTier::Flat);
  EXPECT_EQ(bcastKernelTier(800, thr, 100000, cap), BcastTier::LL);
  // <=thr but LL not configured -> LSA direct.
  EXPECT_EQ(bcastKernelTier(800, thr, 0, cap), BcastTier::LSADirect);
  // <=thr, LL configured but message above LL ceiling -> LSA direct.
  EXPECT_EQ(bcastKernelTier(cap + 8, thr, 1u << 20, cap), BcastTier::LSADirect);
}

TEST(BcastSignalCount, FloorOfTwo) {
  EXPECT_EQ(bcastSignalCount(0), 2);
  EXPECT_EQ(bcastSignalCount(1), 2);
  EXPECT_EQ(bcastSignalCount(2), 2);
  EXPECT_EQ(bcastSignalCount(8), 8);
}

// ---- bcastHybridCtas / bcastSagCtas: size-adaptive ladder, clamped to pool ----

TEST(BcastPolicyCtas, HybridLsaTierUsesLsaCtaCount) {
  const int pool = bcastHybridPoolCtas(16);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  EXPECT_EQ(bcastHybridCtas(thr, thr, 100000, kBroadcastLLMaxBytes, kBroadcastCtasUnset, pool),
            kBroadcastCtasLsa);
  EXPECT_EQ(bcastHybridCtas(1024, thr, 0, kBroadcastLLMaxBytes, kBroadcastCtasUnset, pool),
            kBroadcastCtasLsa);
}

TEST(BcastPolicyCtas, HybridLLTierUsesOneCta) {
  const int pool = bcastHybridPoolCtas(16);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  EXPECT_EQ(bcastHybridCtas(800, thr, 100, kBroadcastLLMaxBytes, kBroadcastCtasUnset, pool),
            kBroadcastCtasLL);
}

TEST(BcastPolicyCtas, HybridSdmaTierUsesSdmaCtaCount) {
  const int pool = bcastHybridPoolCtas(16);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  EXPECT_EQ(bcastHybridCtas(thr + 1, thr, 0, kBroadcastLLMaxBytes, kBroadcastCtasUnset, pool),
            kBroadcastCtasSdma);
  EXPECT_EQ(bcastHybridCtas(64ull << 20, thr, 0, kBroadcastLLMaxBytes, kBroadcastCtasUnset, pool),
            kBroadcastCtasSdma);
}

TEST(BcastPolicyCtas, SagKeysOffPerRankSlice) {
  const int pool = bcastHybridPoolCtas(16);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  // 128 MiB / 8 ranks = 16 MiB slice -> SDMA tier -> 4 CTAs.
  EXPECT_EQ(bcastSagCtas(128ull << 20, 8, thr, kBroadcastCtasUnset, pool), kBroadcastCtasSdma);
  // 2 MiB / 8 = 256 KiB slice -> at threshold -> LSA tier -> 16 CTAs.
  EXPECT_EQ(bcastSagCtas(2ull << 20, 8, thr, kBroadcastCtasUnset, pool), kBroadcastCtasLsa);
}

TEST(BcastPolicyCtas, EnvPinHonoredWithinPool) {
  const int pool = bcastHybridPoolCtas(64);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  EXPECT_EQ(bcastHybridCtas(1024, thr, 0, kBroadcastLLMaxBytes, 8, pool), 8);
  EXPECT_EQ(bcastHybridCtas(64ull << 20, thr, 0, kBroadcastLLMaxBytes, 8, pool), 8);
  EXPECT_EQ(bcastHybridCtas(1024, thr, 0, kBroadcastLLMaxBytes, 0, pool), kBroadcastCtasLsa);
}

TEST(BcastPolicyCtas, GridNeverExceedsPool) {
  const size_t msgs[] = {128, 800, 262144, 262145, 64ull << 20};
  const int deviceVs[] = {1, 4, 16, 32, 128};
  const size_t pins[] = {kBroadcastCtasUnset, 0, 1, 8, 200, 100000};
  for (int v : deviceVs) {
    const int pool = bcastHybridPoolCtas(v);
    for (size_t m : msgs) {
      for (size_t pin : pins) {
        const int g = bcastHybridCtas(m, kBroadcastSdmaThresholdDefault, 0,
                                      kBroadcastLLMaxBytes, pin, pool);
        EXPECT_GE(g, 1);
        EXPECT_LE(g, pool) << "msg=" << m << " V=" << v << " pin=" << pin;
      }
    }
  }
}

// --------------------------------- sagChunk ----------------------------------

TEST(SagChunk, GuardsNonPositiveRanks) {
  Chunk c = sagChunk(800, 0, 0);
  EXPECT_EQ(c.count, 0u);
  EXPECT_EQ(c.eltOffset, 0u);
}

TEST(SagChunk, EvenSplit) {
  Chunk r0 = sagChunk(800, 8, 0);
  EXPECT_EQ(r0.count, 100u);
  EXPECT_EQ(r0.eltOffset, 0u);
  Chunk r3 = sagChunk(800, 8, 3);
  EXPECT_EQ(r3.count, 100u);
  EXPECT_EQ(r3.eltOffset, 300u);
  Chunk r7 = sagChunk(800, 8, 7);
  EXPECT_EQ(r7.count, 100u);
  EXPECT_EQ(r7.eltOffset, 700u);
}

TEST(SagChunk, RemainderFoldedIntoLast) {
  // 803 / 8 = base 100, last gets 803 - 700 = 103.
  Chunk r0 = sagChunk(803, 8, 0);
  EXPECT_EQ(r0.count, 100u);
  Chunk r7 = sagChunk(803, 8, 7);
  EXPECT_EQ(r7.count, 103u);
  EXPECT_EQ(r7.eltOffset, 700u);
  // Slices must exactly tile the whole message.
  size_t total = 0;
  for (int r = 0; r < 8; ++r) total += sagChunk(803, 8, r).count;
  EXPECT_EQ(total, 803u);
}

}  // namespace
