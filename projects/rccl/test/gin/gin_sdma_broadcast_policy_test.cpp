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
  EXPECT_EQ(parseSize(" -4K"), kUnset);
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

TEST(ParseSize, OverflowIsUnsetNotASmallThreshold) {
  // strtoull saturates at ULLONG_MAX rather than failing, and the suffix multiply
  // can then wrap a huge value back down to a plausible-looking small threshold --
  // the same silent mis-tiering shape as a negative value. Both must read as unset
  // so the caller falls back to the tested default.
  EXPECT_EQ(parseSize("99999999999999999999"), kUnset);
  EXPECT_EQ(parseSize("99999999999999999999G"), kUnset);
  // Not saturated on its own, but wraps when scaled by the suffix.
  EXPECT_EQ(parseSize("17179869184G"), kUnset);  // 2^34 GiB = 2^64 bytes
  // The largest value that still scales cleanly is accepted.
  EXPECT_EQ(parseSize("1024G"), 1024ull * 1024 * 1024 * 1024);
}

TEST(ParseSize, HexIsNotAcceptedAsHex) {
  // Base 10 by contract: "0x10" consumes the leading 0 and stops at 'x', which
  // yields 0 -- and 0 is meaningful (it force-enables GIN / disables a tier), so
  // this documents that a hex-looking value is NOT silently read as 16.
  EXPECT_EQ(parseSize("0x10"), 0u);
  EXPECT_NE(parseSize("0x10"), 16u);
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
  EXPECT_FALSE(bcastUseRing(64u << 20, 16u << 20, 8, 8, 0));
}

TEST(BcastUseRing, RequiresAtLeastTwoRanks) {
  EXPECT_FALSE(bcastUseRing(64u << 20, 16u << 20, 1, 1, kBroadcastRingMinDefault));
}

TEST(BcastUseRing, EngagesAtOrAboveMin) {
  const size_t ringMin = kBroadcastRingMinDefault;  // 32 MiB
  EXPECT_FALSE(bcastUseRing(ringMin - 1, 16u << 20, 8, 8, ringMin));
  EXPECT_TRUE(bcastUseRing(ringMin, 16u << 20, 8, 8, ringMin));
}

TEST(BcastUseRing, RequiresCountAtLeastNRanks) {
  EXPECT_FALSE(bcastUseRing(64u << 20, 7, 8, 8, kBroadcastRingMinDefault));
  EXPECT_TRUE(bcastUseRing(64u << 20, 8, 8, 8, kBroadcastRingMinDefault));
}

// The ring forwards with direct peer stores addressed by WORLD rank index
// (ncclGetLsaPointer(recvwin, off, nextRank)) and syncs on an LSA-team barrier,
// so it is only correct when the LSA team is the whole world. Anything smaller
// -- multi-node being the ordinary case -- means the index handed to the LSA API
// is not an LSA peer index at all.
TEST(BcastUseRing, RequiresLsaTeamToCoverTheWorld) {
  const size_t ringMin = kBroadcastRingMinDefault;
  const size_t bytes = 64u << 20;
  const size_t count = 16u << 20;
  // Single node: LSA team == world, so world and LSA indices coincide.
  EXPECT_TRUE(bcastUseRing(bytes, count, 8, 8, ringMin));
  // Two nodes of 8: every size/shape condition still holds, and the ring is
  // still refused, because nextRank would address outside the LSA team.
  EXPECT_FALSE(bcastUseRing(bytes, count, 16, 8, ringMin));
  // A single off-team rank is enough to break the index correspondence.
  EXPECT_FALSE(bcastUseRing(bytes, count, 9, 8, ringMin));
  // An lsaSize larger than the world is not a topology we know how to reason
  // about; require exact agreement rather than lsaSize >= nRanks.
  EXPECT_FALSE(bcastUseRing(bytes, count, 8, 16, ringMin));
}

// ------------------------------ bcastLargeTier -------------------------------
// The predicates above are each true in isolation for large messages, so testing
// them one at a time cannot pin down which tier actually runs. These assert the
// resolved decision across the three boundaries that separate the tiers.

TEST(BcastLargeTier, RingWinsWhereBothPredicatesHold) {
  const size_t ringMin = kBroadcastRingMinDefault;             // 32 MiB
  const size_t sagMin = kBroadcastScatterAgMinDefault;         // 2 MiB
  const size_t bytes = 64u << 20;
  const size_t count = bytes / 4;
  // Both gates are open here; precedence -- not the predicates -- picks the ring.
  ASSERT_TRUE(bcastUseRing(bytes, count, 8, 8, ringMin));
  ASSERT_TRUE(bcastUseScatterAllgather(bytes, count, 8, sagMin));
  EXPECT_EQ(bcastLargeTier(bytes, count, 8, 8, ringMin, sagMin),
            BcastLargeTier::Ring);
}

TEST(BcastLargeTier, BoundariesSelectEachTier) {
  const size_t ringMin = kBroadcastRingMinDefault;             // 32 MiB
  const size_t sagMin = kBroadcastScatterAgMinDefault;         // 2 MiB
  auto tier = [&](size_t bytes) {
    return bcastLargeTier(bytes, bytes / 4, 8, 8, ringMin, sagMin);
  };
  // Below the SAG cutover: neither large tier, so the flat/LSA/LL kernel runs.
  EXPECT_EQ(tier(1u << 20), BcastLargeTier::None);
  // At the SAG cutover but below the ring cutover.
  EXPECT_EQ(tier(sagMin), BcastLargeTier::ScatterAG);
  EXPECT_EQ(tier(ringMin - 1), BcastLargeTier::ScatterAG);
  // At and above the ring cutover.
  EXPECT_EQ(tier(ringMin), BcastLargeTier::Ring);
  EXPECT_EQ(tier(4ull << 30), BcastLargeTier::Ring);
}

TEST(BcastLargeTier, RingOptOutFallsBackToScatterAllgather) {
  // NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0 disables the ring; SAG must take over
  // rather than the message silently dropping to the flat tier.
  const size_t sagMin = kBroadcastScatterAgMinDefault;
  EXPECT_EQ(bcastLargeTier(64u << 20, 16u << 20, 8, 8, 0, sagMin),
            BcastLargeTier::ScatterAG);
  // Both disabled -> flat.
  EXPECT_EQ(bcastLargeTier(64u << 20, 16u << 20, 8, 8, 0, 0),
            BcastLargeTier::None);
}

// Ring is checked first, so without the LSA condition every message at or above
// the ring cutover would take a tier that cannot reach an off-team peer. SAG
// moves data with gin.put over the world team, so it is the correct fallback on
// a world the LSA team does not cover, and it sits directly behind the ring.
TEST(BcastLargeTier, MultiNodeFallsBackToScatterAllgatherNotRing) {
  const size_t ringMin = kBroadcastRingMinDefault;      // 32 MiB
  const size_t sagMin = kBroadcastScatterAgMinDefault;  // 2 MiB
  const size_t count = 16u << 20;
  auto tier = [&](size_t bytes, int nRanks, int lsaSize) {
    return bcastLargeTier(bytes, count, nRanks, lsaSize, ringMin, sagMin);
  };
  // 2x8 ranks, well above the ring cutover: SAG, not Ring.
  EXPECT_EQ(tier(64u << 20, 16, 8), BcastLargeTier::ScatterAG);
  EXPECT_EQ(tier(4ull << 30, 16, 8), BcastLargeTier::ScatterAG);
  // The same sizes on one node still choose the ring, so the gate costs the
  // single-node path nothing.
  EXPECT_EQ(tier(64u << 20, 8, 8), BcastLargeTier::Ring);
  EXPECT_EQ(tier(4ull << 30, 8, 8), BcastLargeTier::Ring);
  // Below the ring cutover the LSA team is irrelevant: SAG either way.
  EXPECT_EQ(tier(sagMin, 16, 8), BcastLargeTier::ScatterAG);
  EXPECT_EQ(tier(sagMin, 8, 8), BcastLargeTier::ScatterAG);
}

// With the ring disabled by env AND an LSA team that does not cover the world,
// the outcome must still be SAG rather than a silent drop to the flat tier.
TEST(BcastLargeTier, MultiNodeWithRingDisabledStillSelectsScatterAllgather) {
  EXPECT_EQ(bcastLargeTier(64u << 20, 16u << 20, 16, 8, 0,
                           kBroadcastScatterAgMinDefault),
            BcastLargeTier::ScatterAG);
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
  // Interior case: 64 MiB / 8 CTAs = 8 MiB stripe; 8 MiB / 64 KiB = 128 chunks.
  EXPECT_EQ(bcastRingChunks(64ull << 20, 8, kUnset), 128);
  // Mid message at max depth: 128 MiB / 8 CTAs = 16 MiB stripe -> 256 chunks.
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

TEST(BcastPolicyCtas, LadderIsMonotonicAcrossTiers) {
  // The per-tier assertions above pin which branch was taken, but each compares a
  // constant against itself, so the constants' relative ordering -- the property
  // the ladder exists for -- is not pinned by any of them. Assert the shape: a
  // single-CTA LL tier, a wider LSA tier, and a narrow SDMA tier (only nRanks
  // threads issue puts), with every grid inside [1, pool].
  const int pool = bcastHybridPoolCtas(16);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  const int llGrid = bcastHybridCtas(1024, thr, 4096, kBroadcastLLDefaultMaxBytes,
                                     kBroadcastCtasUnset, pool);
  const int lsaGrid = bcastHybridCtas(thr, thr, 0, kBroadcastLLMaxBytes,
                                      kBroadcastCtasUnset, pool);
  const int sdmaGrid = bcastHybridCtas(thr + 1, thr, 0, kBroadcastLLMaxBytes,
                                       kBroadcastCtasUnset, pool);
  EXPECT_LT(llGrid, lsaGrid) << "LL is a single-CTA tier; LSA must be wider";
  EXPECT_LT(sdmaGrid, lsaGrid) << "GIN-put tier must be narrower than LSA";
  for (int g : {llGrid, lsaGrid, sdmaGrid}) {
    EXPECT_GE(g, 1);
    EXPECT_LE(g, pool);
  }
}

TEST(BcastPolicyCtas, SagEnvPinHonoredWithinPool) {
  const int pool = bcastHybridPoolCtas(64);
  const size_t thr = kBroadcastSdmaThresholdDefault;
  EXPECT_EQ(bcastSagCtas(128ull << 20, 8, thr, 8, pool), 8);
  EXPECT_EQ(bcastSagCtas(2ull << 20, 8, thr, 0, pool),
            bcastSagCtas(2ull << 20, 8, thr, kBroadcastCtasUnset, pool));
}

TEST(BcastPoolCtas, HybridPoolAndRingPeak) {
  EXPECT_EQ(bcastPoolCtas(16, 8), 16);
  EXPECT_EQ(bcastPoolCtas(16, 128), 128);
  EXPECT_GE(bcastPoolCtas(4, 64), 64);
  EXPECT_GE(bcastPoolCtas(16, 128), bcastHybridPoolCtas(16));
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

// ---------------------------- ring stripe / chunk ----------------------------
// The default large tier walks these two splits, so they carry the exactness
// requirement that sagChunk carries for the scatter+allgather tier: every element
// covered exactly once, by exactly one CTA, in exactly one pipeline chunk.

TEST(RingStripe, GuardsOutOfRangeBlocks) {
  EXPECT_EQ(ringStripe(1000, 0, 0).count, 0u);
  EXPECT_EQ(ringStripe(1000, -1, 8).count, 0u);
  EXPECT_EQ(ringStripe(1000, 8, 8).count, 0u);
}

TEST(RingStripe, PartitionsBufferExactlyAndContiguously) {
  // Deliberately indivisible: 1000 elements over 128 CTAs, and a prime count.
  for (size_t count : {size_t(1000), size_t(1048576), size_t(9973)}) {
    for (int nBlocks : {1, 7, 8, 128}) {
      size_t expectedOffset = 0;
      for (int b = 0; b < nBlocks; ++b) {
        Chunk s = ringStripe(count, b, nBlocks);
        EXPECT_EQ(s.eltOffset, expectedOffset)
            << "count=" << count << " nBlocks=" << nBlocks << " b=" << b;
        expectedOffset += s.count;
      }
      // Contiguous from 0 and covering everything => an exact partition.
      EXPECT_EQ(expectedOffset, count)
          << "count=" << count << " nBlocks=" << nBlocks;
    }
  }
}

TEST(RingStripe, BalancedToWithinOneElement) {
  const size_t count = 1000;
  const int nBlocks = 128;
  size_t lo = count, hi = 0;
  for (int b = 0; b < nBlocks; ++b) {
    size_t c = ringStripe(count, b, nBlocks).count;
    lo = (c < lo) ? c : lo;
    hi = (c > hi) ? c : hi;
  }
  EXPECT_LE(hi - lo, 1u);
}

TEST(RingChunk, GuardsOutOfRangeChunks) {
  EXPECT_EQ(ringChunk(0, 1000, 0, 0).count, 0u);
  EXPECT_EQ(ringChunk(0, 1000, -1, 16).count, 0u);
  EXPECT_EQ(ringChunk(0, 1000, 16, 16).count, 0u);
}

TEST(RingChunk, PartitionsStripeFromItsBase) {
  const size_t stripeBase = 4096;
  const size_t stripeCount = 803;   // indivisible by C below
  const int C = 16;
  size_t expectedOffset = stripeBase;
  for (int c = 0; c < C; ++c) {
    Chunk k = ringChunk(stripeBase, stripeCount, c, C);
    EXPECT_EQ(k.eltOffset, expectedOffset) << "c=" << c;
    expectedOffset += k.count;
  }
  EXPECT_EQ(expectedOffset, stripeBase + stripeCount);
}

TEST(RingChunk, ComposesWithRingStripeToCoverTheBuffer) {
  // The two splits together are what the kernel walks: every element of the
  // message must land in exactly one (block, chunk) pair.
  const size_t count = 9973;   // prime
  const int nBlocks = 8;
  const int C = 7;
  size_t total = 0;
  size_t expectedOffset = 0;
  for (int b = 0; b < nBlocks; ++b) {
    Chunk s = ringStripe(count, b, nBlocks);
    for (int c = 0; c < C; ++c) {
      Chunk k = ringChunk(s.eltOffset, s.count, c, C);
      EXPECT_EQ(k.eltOffset, expectedOffset) << "b=" << b << " c=" << c;
      expectedOffset += k.count;
      total += k.count;
    }
  }
  EXPECT_EQ(total, count);
  EXPECT_EQ(expectedOffset, count);
}

}  // namespace
