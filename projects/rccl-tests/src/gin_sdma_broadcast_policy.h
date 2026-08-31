/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Pure (no-GPU, no-RCCL) policy logic for the GIN-SDMA hybrid Broadcast design
// (broadcast.cu, deviceImpl == 3). Single source of truth for:
//   - the LSA<->GIN threshold + LL-cap env resolution,
//   - the tier-selection predicates the device kernels branch on (LL / LSA /
//     flat GIN / scatter+allgather / pipelined ring),
//   - the scatter/ring chunk + alignment math, and
//   - the devComm signal-count computation.
//
// Every function here is free of `ncclDevComm`, GPU intrinsics and getenv() so
// it can be (a) unit-tested on the host with GoogleTest and (b) called from the
// __global__ kernels so the SAME decision code runs on device. Attributes are
// guarded (GIN_SDMA_HOST_ONLY) so the header also compiles as plain host C++ in
// the test binary.
//
// This header intentionally lives in `namespace gin_sdma` next to the backend
// put-segmentation policy (nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h,
// also `namespace gin_sdma`). Namespaces are open, so the two coexist as long as
// no symbol is defined twice: the shared kGinPutMaxBytes / kGinSdmaSafeCopyBytes /
// ginPutSegment* live in the put-policy header and are NOT redefined here.
//
// See ddai-artifacts/docs/gin-anvil-sdma-broadcast-design-plan.md for the
// measured basis of each default.

#ifndef GIN_SDMA_BROADCAST_POLICY_H_
#define GIN_SDMA_BROADCAST_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The device kernels get __host__ __device__; plain host builds (and the host
// unit test, which defines GIN_SDMA_HOST_ONLY) get no attribute so the header
// compiles as ordinary C++ with no device codegen. Guarded with #ifndef so it
// does not clash with the identical definition in gin_anvil_sdma_put_policy.h
// when both headers are pulled into the same device TU (broadcast.cu).
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIN_SDMA_HOST_ONLY)
#ifndef GIN_SDMA_HD
#define GIN_SDMA_HD __host__ __device__
#endif
#else
#ifndef GIN_SDMA_HD
#define GIN_SDMA_HD
#endif
#endif

namespace gin_sdma {

// Sentinel meaning "env var unset/empty/unparseable" (mirrors
// TEST_SDMA_THRESHOLD_UNSET in common.h so the two agree).
#ifndef GIN_SDMA_KTHRESHOLDUNSET_DEFINED
#define GIN_SDMA_KTHRESHOLDUNSET_DEFINED 1
static constexpr size_t kThresholdUnset = (size_t)-1;
#endif

// ---- Design constants (single source of truth; broadcast.cu aliases these) ----

// Broadcast LSA<->GIN default; compared against the full message bytes.
static constexpr size_t kBroadcastSdmaThresholdDefault = 262144;        // 256 KiB
// Broadcast scatter+allgather (large tier) default; 0 disables the tier.
static constexpr size_t kBroadcastScatterAgMinDefault  = 2ull * 1024 * 1024;  // 2 MiB
// Broadcast LL fast path: compile ceiling + default runtime cutover.
static constexpr size_t kBroadcastLLMaxBytes           = 65536;         // 64 KiB
static constexpr size_t kBroadcastLLDefaultMaxBytes    = 2048;          // 2 KiB
// Broadcast pipelined-ring (large tier) minimum; 0 disables. Enabled by default
// at the SAG crossover (~32 MiB): below it the (N-1)-hop pipeline over tiny
// per-CTA slices collapses (ring 16M 98 << SAG 145), above it the ring is the
// best GIN option and beats host from 256 MiB up. Checked before SAG; SAG stays
// the opt-out fallback (NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0).
static constexpr size_t kBroadcastRingMinDefault       = 32ull * 1024 * 1024;  // 32 MiB
// Target bytes per ring pipeline chunk, sized PER CTA (see bcastRingChunks): the
// ring is an (N-1)-hop pipeline whose fill/drain efficiency is C/(C+N-1), so the
// depth must be large. A chunk-depth sweep (8x MI355X, 2G, 128 CTAs) rises
// monotonically past the old 64 cap (32ch 322, 64ch 350), so the cap is raised
// to 256 and the count targets ~64 KiB per chunk per CTA.
static constexpr size_t kBroadcastRingTargetChunk      = 64ull * 1024;  // 64 KiB per CTA
static constexpr int    kBroadcastRingMaxChunks        = 256;
static constexpr int    kBroadcastRingMinChunks        = 16;            // fill the pipeline past the N-1 cost

// ---------------------------- env / threshold ----------------------------

// Parse a size string ("123", "4K", "2M", "1G"; decimal only, optional single
// binary suffix). Returns kThresholdUnset for null/empty/negative/non-numeric
// input.
// Pure: takes the raw string so it is testable without touching the process
// environment. (common.h::testParseSdmaThresholdEnv = this on getenv(name).)
inline size_t parseSize(const char* v) {
  // strtoull() accepts a leading '-' and wraps it into a huge unsigned value,
  // which would read as a ~16 EiB threshold and silently pin every message to
  // the LSA tier. Reject the sign before parsing.
  if (v == nullptr || v[0] == '\0' || v[0] == '-') return kThresholdUnset;
  char* end = nullptr;
  unsigned long long val = strtoull(v, &end, 10);
  if (end == v) return kThresholdUnset;  // no digits consumed
  if (*end) {  // trailing suffix char (end is always set by strtoull)
    switch (*end) {
      case 'k': case 'K': val *= 1024ULL; break;
      case 'm': case 'M': val *= 1024ULL * 1024ULL; break;
      case 'g': case 'G': val *= 1024ULL * 1024ULL * 1024ULL; break;
      default: break;  // unknown suffix ignored (matches common.h)
    }
  }
  return (size_t)val;
}

// Resolve a collective LSA<->GIN threshold from already-parsed candidates:
//   1. collective-specific value, if set;
//   2. shared NCCL_GIN_ANVIL_SDMA_THRESHOLD value, if set (global force knob);
//   3. the collective default.
GIN_SDMA_HD inline size_t resolveThreshold(size_t collVal, size_t sharedVal,
                                           size_t collDefault) {
  if (collVal != kThresholdUnset) return collVal;
  if (sharedVal != kThresholdUnset) return sharedVal;
  return collDefault;
}

// Resolve an LL cutover cap from an already-parsed env value:
//   unset -> unsetDefault (Broadcast: 2 KiB),
//   then clamp to the compile ceiling, then round down to 8 bytes.
GIN_SDMA_HD inline size_t resolveLLCap(size_t envVal, size_t unsetDefault,
                                       size_t maxCap) {
  size_t cap = (envVal == kThresholdUnset) ? unsetDefault : envVal;
  if (cap > maxCap) cap = maxCap;
  return (cap / 8) * 8;
}

// ------------------------------- alignment -------------------------------

// The rccl-tests per-rank chunk count, aligned so a chunk is a whole number of
// 16-byte lines: (count/nRanks) & -(16/eltSize). eltSize must divide 16.
GIN_SDMA_HD inline size_t alignChunkCount(size_t count, int nRanks, size_t eltSize) {
  if (nRanks <= 0 || eltSize == 0) return 0;
  size_t mask = (size_t)0 - (16 / eltSize);   // e.g. eltSize 4 -> mask ~3
  return (count / (size_t)nRanks) & mask;
}

// ------------------------------- Broadcast -------------------------------

// Host gate: use the scatter+allgather large tier instead of the flat/LSA/LL
// hybrid. Requires SAG enabled, >=2 ranks, message at/above the cutover, and at
// least one element per rank (the scatter needs a non-empty slice each).
GIN_SDMA_HD inline bool bcastUseScatterAllgather(size_t msgBytes, size_t count,
                                                 int nRanks, size_t sagMin) {
  return sagMin != 0 && nRanks >= 2 && msgBytes >= sagMin &&
         count >= (size_t)nRanks;
}

// Host gate: use the pipelined-ring large tier (highest priority when enabled).
// Opt-in via ringMin (NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES); needs >=2 ranks,
// message at/above the cutover, and count >= nRanks so every ring chunk is
// non-empty on every rank.
GIN_SDMA_HD inline bool bcastUseRing(size_t msgBytes, size_t count,
                                     int nRanks, size_t ringMin) {
  return ringMin != 0 && nRanks >= 2 && msgBytes >= ringMin &&
         count >= (size_t)nRanks;
}

// Ring pipeline depth (chunks per CTA stripe). Env override (envChunks) wins when
// set (clamped to <= max); otherwise size-adaptive: target ~64 KiB per chunk of
// THIS CTA's stripe (msgBytes/ctas), clamped to [min,max]. Per-CTA sizing keeps
// the chunk bytes constant across sizes so small messages are not over-chunked.
// The kernel further clamps to <= sCount so none is empty.
GIN_SDMA_HD inline int bcastRingChunks(size_t msgBytes, int ctas, size_t envChunks) {
  if (envChunks != kThresholdUnset && envChunks > 0) {
    return (envChunks > (size_t)kBroadcastRingMaxChunks)
               ? kBroadcastRingMaxChunks : (int)envChunks;
  }
  if (ctas < 1) ctas = 1;
  size_t stripeBytes = msgBytes / (size_t)ctas;
  size_t c = stripeBytes / kBroadcastRingTargetChunk;
  if (c < (size_t)kBroadcastRingMinChunks) c = (size_t)kBroadcastRingMinChunks;
  if (c > (size_t)kBroadcastRingMaxChunks) c = (size_t)kBroadcastRingMaxChunks;
  return (int)c;
}

// In-kernel LL eligibility (inside the msgBytes<=sdmaThreshold branch):
// LL configured, 8-byte aligned, within the compile ceiling, and it fits the
// pre-sized slot count (a broadcast receiver needs msgBytes/8 u64 slots).
GIN_SDMA_HD inline bool bcastLLEligible(size_t msgBytes, int llSlots,
                                        size_t llMaxBytes) {
  return llSlots != 0 && (msgBytes % 8 == 0) && msgBytes <= llMaxBytes &&
         (msgBytes / 8) <= (size_t)llSlots;
}

enum class BcastTier { LL, LSADirect, Flat };

// Tier chosen by GinHybridBroadcastKernel (assumes the host already decided this
// is NOT the scatter+allgather / ring tier). Uses bcastLLEligible so the kernel
// and the tests share the decision.
GIN_SDMA_HD inline BcastTier bcastKernelTier(size_t msgBytes, size_t sdmaThreshold,
                                             int llSlots, size_t llMaxBytes) {
  if (msgBytes <= sdmaThreshold) {
    if (bcastLLEligible(msgBytes, llSlots, llMaxBytes)) return BcastTier::LL;
    return BcastTier::LSADirect;
  }
  return BcastTier::Flat;
}

// Sentinel meaning "CTA-count env var unset/empty/unparseable"; bcastHybridCtas() /
// bcastSagCtas() fall back to the size-adaptive ladder when the env value is this.
static constexpr size_t kBroadcastCtasUnset = (size_t)-1;

// Broadcast -D 3 size-adaptive CTA count (decoupled from -V, like AllGather). Keyed
// off the SAME tier predicates the kernels evaluate:
//   * LL tier (tiny, eligible): 1 CTA (kernel returns early for blockIdx.x != 0).
//   * LSA-direct tier (msg <= threshold, or SAG slice <= threshold): ~16 CTAs.
//   * GIN-put / Anvil-SDMA tier: few (4) CTAs -- only nRanks threads issue puts.
// NCCL_GIN_ANVIL_BCAST_CTAS pins a fixed count (diagnostic), clamped to the pool.
static constexpr int kBroadcastCtasLsa  = 16;
static constexpr int kBroadcastCtasSdma = 4;
static constexpr int kBroadcastCtasLL   = 1;

GIN_SDMA_HD inline int bcastHybridMaxCtas() {
  return (kBroadcastCtasLsa > kBroadcastCtasSdma) ? kBroadcastCtasLsa : kBroadcastCtasSdma;
}

// Barrier/lsaBarrier/signal pool for the hybrid + SAG kernels: max(-V, ladder peak).
GIN_SDMA_HD inline int bcastHybridPoolCtas(int deviceCtaCount) {
  const int maxTier = bcastHybridMaxCtas();
  return (deviceCtaCount > maxTier) ? deviceCtaCount : maxTier;
}

// Full pool covering hybrid/SAG/P2P grids AND the ring's independent CTA count.
GIN_SDMA_HD inline int bcastPoolCtas(int deviceCtaCount, int ringCtasPeak) {
  int pool = bcastHybridPoolCtas(deviceCtaCount);
  return (ringCtasPeak > pool) ? ringCtasPeak : pool;
}

// Tier predicate for the flat hybrid kernel and SAG slice sizing.
GIN_SDMA_HD inline bool bcastChunkUsesLsaTier(size_t msgBytes, size_t sdmaThreshold) {
  return msgBytes <= sdmaThreshold;
}

// Launched grid for GinHybridBroadcastKernel (-D 3 flat/LL/LSA path).
GIN_SDMA_HD inline int bcastHybridCtas(size_t msgBytes, size_t sdmaThreshold,
                                       int llSlots, size_t llMaxBytes,
                                       size_t envCtas, int poolCtas) {
  if (poolCtas < 1) poolCtas = 1;
  if (envCtas != kBroadcastCtasUnset && envCtas > 0)
    return (envCtas > (size_t)poolCtas) ? poolCtas : (int)envCtas;
  if (msgBytes <= sdmaThreshold) {
    if (bcastLLEligible(msgBytes, llSlots, llMaxBytes))
      return kBroadcastCtasLL;
    const int adaptive = kBroadcastCtasLsa;
    return (adaptive > poolCtas) ? poolCtas : adaptive;
  }
  const int adaptive = kBroadcastCtasSdma;
  return (adaptive > poolCtas) ? poolCtas : adaptive;
}

// Launched grid for GinScatterAllgatherBroadcastKernel. Keys off the per-rank
// allgather slice (msgBytes / nRanks), matching the AllGather adaptive ladder.
GIN_SDMA_HD inline int bcastSagCtas(size_t msgBytes, int nRanks, size_t sdmaThreshold,
                                    size_t envCtas, int poolCtas) {
  if (poolCtas < 1) poolCtas = 1;
  if (envCtas != kBroadcastCtasUnset && envCtas > 0)
    return (envCtas > (size_t)poolCtas) ? poolCtas : (int)envCtas;
  size_t sliceBytes = (nRanks > 0) ? (msgBytes / (size_t)nRanks) : msgBytes;
  const int adaptive = bcastChunkUsesLsaTier(sliceBytes, sdmaThreshold)
                           ? kBroadcastCtasLsa : kBroadcastCtasSdma;
  return (adaptive > poolCtas) ? poolCtas : adaptive;
}

// ginSignalCount for Broadcast case 3: >=2 so the SAG two-signal scheme works.
GIN_SDMA_HD inline int bcastSignalCount(int deviceCtaCount) {
  return (deviceCtaCount < 2) ? 2 : deviceCtaCount;
}

// Even split with the remainder folded into the last rank's slice (SAG / ring).
struct Chunk {
  size_t count;      // elements in this rank's slice
  size_t eltOffset;  // element offset of this rank's slice
};
GIN_SDMA_HD inline Chunk sagChunk(size_t totalCount, int nRanks, int rank) {
  Chunk c{0, 0};
  if (nRanks <= 0) return c;
  size_t base = totalCount / (size_t)nRanks;
  size_t tail = totalCount - base * (size_t)(nRanks - 1);
  c.count = (rank == nRanks - 1) ? tail : base;
  c.eltOffset = (size_t)rank * base;
  return c;
}

}  // namespace gin_sdma

#endif  // GIN_SDMA_BROADCAST_POLICY_H_
