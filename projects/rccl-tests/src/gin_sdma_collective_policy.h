/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared, pure (no-GPU, no-RCCL) policy logic for the GIN-SDMA collective
// designs (Broadcast / AllGather / AllToAll, deviceImpl == 3). This is the
// single source of truth for:
//   - per-collective LSA<->GIN threshold + LL-cap env resolution,
//   - the tier-selection predicates the device kernels branch on,
//   - the chunk/offset and alignment math, and
//   - the devComm resource-requirement computation.
//
// Every function here is deliberately free of `ncclDevComm`, GPU intrinsics and
// getenv() so it can be (a) unit-tested on the host with GoogleTest to high
// line+branch coverage, and (b) called from the __global__ kernels so the SAME
// decision code is exercised on device. Attributes are guarded so the header
// also compiles as plain host C++ in the test binary.
//
// See ddai-artifacts/docs/gin-anvil-sdma-broadcast-design-plan.md (and the
// AllGather/AllToAll notes) for the measured basis of each default.

#ifndef GIN_SDMA_COLLECTIVE_POLICY_H_
#define GIN_SDMA_COLLECTIVE_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The device kernels get __host__ __device__; plain host builds (and the host
// unit test, which defines GIN_SDMA_HOST_ONLY) get no attribute so the header
// compiles as ordinary C++ with no device codegen.
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIN_SDMA_HOST_ONLY)
#define GIN_SDMA_HD __host__ __device__
#else
#define GIN_SDMA_HD
#endif

namespace gin_sdma {

// Sentinel meaning "env var unset/empty/unparseable" (mirrors
// TEST_SDMA_THRESHOLD_UNSET in common.h so the two agree).
static constexpr size_t kThresholdUnset = (size_t)-1;

// ---- Design constants (single source of truth; the .cu files alias these) ----

// Broadcast LSA<->GIN default; compared against the full message bytes.
static constexpr size_t kBroadcastSdmaThresholdDefault = 262144;        // 256 KiB
// Broadcast scatter+allgather (large tier) default; 0 disables the tier.
static constexpr size_t kBroadcastScatterAgMinDefault  = 2ull * 1024 * 1024;  // 2 MiB
// Broadcast LL fast path: compile ceiling + default runtime cutover.
static constexpr size_t kBroadcastLLMaxBytes           = 65536;         // 64 KiB
static constexpr size_t kBroadcastLLDefaultMaxBytes    = 2048;          // 2 KiB
// Broadcast pipelined-ring (large tier) minimum; 0 disables. Enabled by default at
// the SAG crossover. With the deep per-CTA-adaptive pipeline (see bcastRingChunks)
// a warm A/B on 8x MI355X (128 CTAs, -c 0, float) shows the ring is the best GIN
// option from ~64 MiB up and BEATS HOST from 256 MiB up:
//   size   host   SAG(-V32)  ring        ring/host
//   64M    251    196        245         0.98
//   128M   271    213        268         0.99
//   256M   300    222        313         1.04
//   512M   308    226        342         1.11
//   1G     321    229        356         1.11
//   2G     322    230        364         1.13
// Below the crossover the (N-1)-hop pipeline over tiny per-CTA slices collapses
// (ring 8M 49, 16M 98 << SAG 104/145), so SAG owns the small/mid fill regime. A
// warm mid-band A/B (8x MI355X, 2026-08-04, float, no -V; §9.7) pinned the actual
// ring<->SAG crossover at ~32 MiB, LOWER than the previous 64 MiB gate: at 33 MiB
// the ring (182) already edges SAG (177), and at 16 MiB it still collapses (98).
// So the gate is lowered 64->32 MiB to capture 33 MiB (+3%, 90->93% of host) with
// no regression (16 MiB stays SAG; 64 MiB+ unchanged). The 8-16 MiB residual dip
// (~87-90% of host) has no lever -- SAG is CTA-saturated at the bare -V=16 default
// (more CTAs monotonically HURT, the opposite of ReduceScatter) and the ring
// fill-stalls there -- so it stays a structural small-message latency class (§9.3).
// The ring gate is checked before SAG; SAG stays the opt-out fallback
// (NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0).
static constexpr size_t kBroadcastRingMinDefault       = 32ull * 1024 * 1024;  // 32 MiB
// Target bytes per ring pipeline chunk, sized PER CTA (see bcastRingChunks): the
// ring is an (N-1)-hop pipeline whose fill/drain efficiency is C/(C+N-1), so the
// depth must be large. A chunk-depth sweep (8x MI355X, 2G, 128 CTAs) rises
// monotonically past the old 64 cap -- 8ch 217, 16ch 280, 32ch 322, 64ch 350 --
// so the cap is raised to 256 and the count targets ~64 KiB per chunk per CTA
// (the same sweet spot the Reduce multi-ring found; deeper starves small sizes).
static constexpr size_t kBroadcastRingTargetChunk      = 64ull * 1024;  // 64 KiB per CTA
static constexpr int    kBroadcastRingMaxChunks        = 256;           // was 64 (the ring was still climbing there)
static constexpr int    kBroadcastRingMinChunks        = 16;            // fill the pipeline past the N-1 cost

// AllGather LSA<->GIN default; compared against the per-rank chunk bytes.
static constexpr size_t kAllGatherSdmaThresholdDefault = 262144;        // 256 KiB/rank
static constexpr size_t kAllGatherLLMaxBytes           = 4096;          // 4 KiB/rank
// Per-rank chunk at/below which the LSA branch collapses to a single CTA.
static constexpr size_t kAllGatherLsaSingleCtaMax      = 8192;          // 8 KiB/rank

// AllToAll LSA<->GIN default; compared against the per-peer chunk bytes. The
// value is the *largest per-peer chunk the LSA tier still wins* (not the
// crossover), matching the Scatter/Broadcast convention. Measured on 8x MI355X
// (2026-07-31, F1 adaptive-CTA + F2 rotated schedule + F3 per-CTA peer phase)
// via all-LSA vs all-SDMA sweeps: with F3 running all 7 xGMI egress links
// concurrently the LSA tier leads through 2 MiB/peer (16 MiB total, 251 vs 244
// GB/s) and SDMA takes over at 4 MiB/peer (32 MiB total, 311 vs 265), scaling
// to ~426 at 2 GiB. So <=2 MiB/peer -> LSA, >2 MiB/peer -> GIN-SDMA. Raised
// from the pre-F3 1 MiB/peer default (F3 roughly doubled the large-chunk LSA
// busbw, moving the crossover out one f2 step).
static constexpr size_t kAllToAllSdmaThresholdDefault  = 2097152;       // 2 MiB/peer
static constexpr size_t kAllToAllLLMaxBytes            = 65536;         // 64 KiB/peer
// AllToAll LL is OFF by default (unset env -> 0 cap).
static constexpr size_t kAllToAllLLDefaultMaxBytes     = 0;
// AllToAll LL-tier CTA count (multi-CTA barrier-free LL prototype for the small-
// message tail). The single-CTA LL scatter/gather collapses above ~32 KiB total
// because one CTA can't drive the xGMI inbox writes; splitting the per-peer chunk
// across this many CTAs (each owning its own scratch block, indexed by blockIdx)
// restores parallelism while keeping the barrier-free epoch/flag protocol. The LL
// scratch is allocated for this many blocks; the tier launches this many CTAs.
// Override at runtime (<= this max) with NCCL_GIN_ANVIL_A2A_LL_CTAS for sweeps.
static constexpr int    kA2aLLCtas                     = 16;
// AllToAll LSA-tier barrier-free completion (Option A): number of double-buffered
// flag slots per (CTA, source rank). The direct 1-pass LSA copy beats host RING
// but the two collective LSA barriers are the whole small-message gap; a per-call
// point-to-point done-flag exchange (each CTA signals every peer once its slice
// is written+fenced, then waits for all sources) replaces them. Epoch-tagged and
// double-buffered so consecutive calls reuse distinct slots; AllToAll's mutual
// coupling bounds cross-rank skew well below this depth.
static constexpr int    kA2aFlagSlots                  = 4;

// AllToAll LSA-tier CTA parallelism (F1: size-adaptive CTA count). The direct
// LSA scatter is a pure SM-copy tier whose xGMI-egress throughput scales with
// the number of in-flight remote stores, i.e. the launched CTA count -- exactly
// the "channel parallelism" the host RING gets from its ~32 channels. The
// original fixed grid (-V 8) under-parallelizes: a -V sweep on 8x MI355X showed
// a 2.85x busbw gain at a 4 MiB total message (44.6 -> 127 GB/s) going V8->V32,
// with the optimal count growing with the per-peer chunk. a2aLsaCtaCount() maps
// the per-peer chunk to that ladder; kA2aLsaMaxCtas caps it (and sizes the
// lsaBarrier allocation so any adaptive grid is always <= the allocated
// barriers, regardless of -V). The SDMA (large) tier only needs nRanks threads
// to issue one put per peer, so it launches a small fixed grid (kA2aSdmaCtas):
// more CTAs there only inflate the world-barrier cost without adding puts.
static constexpr int kA2aLsaMaxCtas = 64;
static constexpr int kA2aSdmaCtas   = 8;

// Size-adaptive CTA count for the LSA (SM-copy) tier. Tuned from CTA sweeps on
// 8x MI355X (2026-07-31, all-LSA) at fixed message size. With F3 driving all 7
// xGMI links concurrently the optimum shifted up vs the pre-F3 ladder: 32 CTAs
// wins through 512 KiB/peer (<=4 MiB total: 32 beats 48 by ~15% at 4 MiB), and
// 48 wins for the larger 1-2 MiB/peer chunks (8 MiB: 241 vs 234 @32; 16 MiB: 284
// vs 218 @32, 265 @64) -- more links in flight help until 64 overshoots into
// barrier/incast overhead. Truly tiny chunks stay latency-bound on a small grid.
// Chunks >2 MiB/peer route to GIN-SDMA by default, so the top rung only serves
// the 1-2 MiB/peer band (and caps any raised-threshold LSA run). Result is
// clamped to maxCtas (== the allocated lsaBarrier count).
GIN_SDMA_HD inline int a2aLsaCtaCount(size_t perPeerBytes, int maxCtas) {
  int n;
  if (perPeerBytes <= 32u * 1024)          n = 8;    // tiny: latency-bound
  else if (perPeerBytes <= 64u * 1024)     n = 16;
  else if (perPeerBytes <= 512u * 1024)    n = 32;   // <=512 KiB/peer (<=4 MiB total)
  else                                     n = 48;   // 1-2 MiB/peer: F3 sweep optimum
  return n < maxCtas ? n : maxCtas;
}

// SendRecv LL fast path: compile ceiling + default runtime cutover. Point-to-
// point carries a single message per receiver (like Broadcast, not AllGather),
// so a receiver needs just cap/8 u64 slots. On by default (2 KiB) since the ring
// is barrier-bound at tiny sizes and LL removes the exit barrier.
static constexpr size_t kSendRecvLLMaxBytes            = 65536;         // 64 KiB
static constexpr size_t kSendRecvLLDefaultMaxBytes     = 2048;          // 2 KiB

// Scatter LL fast path: compile ceiling + default runtime cutover. Scatter is
// one-to-all with a DISTINCT per-rank chunk, but each receiver still takes a
// single incoming message (its own chunk from the root), so a receiver needs
// just chunk/8 u64 slots -- the same single-message sizing as Broadcast/SendRecv
// (not the nRanks*... AllGather/AllToAll form). The LL cap is compared against
// the per-rank chunk bytes (matching the Scatter LSA<->GIN threshold). On by
// default (2 KiB) since tiny scatters are barrier-bound and LL drops the exit
// barrier; tune/disable via NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES (0 = disable).
static constexpr size_t kScatterLLMaxBytes             = 65536;         // 64 KiB/chunk
static constexpr size_t kScatterLLDefaultMaxBytes      = 2048;          // 2 KiB/chunk

// Gather LL fast path: compile ceiling + default runtime cutover. Gather is the
// all-to-one inverse of Scatter: every rank packs its chunk into the ROOT's LL
// scratch and the root alone polls all N chunks out. So -- unlike Scatter's
// single-message sizing -- the root holds nRanks chunks and the slot demand is
// nRanks*(chunk/8) (the AllGather form). The cap is compared against the per-rank
// chunk bytes (matching the Gather LSA<->GIN threshold). Provisionally on at
// 2 KiB pending the LL-on/off A/B (unlike Scatter, Gather concentrates all
// unpacking on the root, so the win is expected to be narrower); tune/disable via
// NCCL_GIN_ANVIL_GATHER_LL_MAX_BYTES (0 = disable).
static constexpr size_t kGatherLLMaxBytes              = 65536;         // 64 KiB/chunk
static constexpr size_t kGatherLLDefaultMaxBytes       = 2048;          // 2 KiB/chunk

// Max bytes per single gin.put() on the Anvil-SDMA backend. The SDMA linear-copy
// descriptor count field is 30 bits and 1-based (HW encodes count = bytes - 1,
// see rocr-runtime amd_blit_sdma.cpp / sdma_registers.h), so the largest single
// packet is (2^30 - 1) + 1 = 2^30 = exactly 1 GiB. A put of >1 GiB silently
// truncates: a 2 GiB put encodes count = (2^31-1) & 0x3FFFFFFF = 2^30-1 and the
// HW copies only 1 GiB, corrupting the transfer. Every GIN-tier put must be
// split into segments of at most this size (see common.h::ginPutChunked).
//
// Set to exactly 1 GiB: this is the hardware maximum, so the segmentation clamp
// (seg <= kGinPutMaxBytes) guarantees count = seg-1 <= 2^30-1 = 0x3FFFFFFF, which
// fills the 30-bit field exactly with no truncation. Zero margin by design; do
// NOT raise above 2^30. 1 GiB is a multiple of 32 B, satisfying the copy
// descriptor's 32 B length alignment.
static constexpr size_t kGinPutMaxBytes                = 1024ull * 1024 * 1024;  // 1 GiB (2^30, HW max)

// Max bytes per single gin.put() that the Anvil-SDMA backend copies *reliably*
// on MI355X + ROCm 7.13 (NCCL_GIN_TYPE=5). This is SMALLER than kGinPutMaxBytes:
// the 30-bit count field bounds correctness at 1 GiB, but a single copy
// descriptor at/above 256 MiB (2^28) on the fused COPY_LINEAR_WAIT_SIGNAL_MI4
// path stalls the SDMA engine, so the fused copy never lands AND its SignalInc
// never fires -> every rank spins forever in waitSignal (a HANG, not a data
// miscompare). Measured on 8x MI355X (2026-08-07, alltoall_perf -D 3): a single
// 256 MiB/peer put (AllToAll @ 2 GiB total) HANGS; capping each put to 128 MiB
// (2 puts/peer) completes with identical bandwidth (busbw ~423 GB/s, unchanged
// vs the 1 GiB total case). 128 MiB is proven safe with zero measured perf loss;
// do not raise without re-measuring. ginPutChunked segments every GIN-tier put
// to this size, so it protects ALL GIN-SDMA collectives (A2A, A2Av, AllGather,
// Broadcast, ReduceScatter, AllReduce) that route through it.
static constexpr size_t kGinSdmaSafeCopyBytes          = 128ull * 1024 * 1024;   // 128 MiB (MI355X reliable single-copy max)

// ---------------------------- env / threshold ----------------------------

// Parse a size string ("123", "4K", "2M", "1G"; decimal only, optional single
// binary suffix). Returns kThresholdUnset for null/empty/non-numeric input.
// Pure: takes the raw string so it is testable without touching the process
// environment. (common.h::testParseSdmaThresholdEnv = this on getenv(name).)
// Host-only: it calls strtoull(), and only host dispatch/requirements code uses
// it, so it is not marked for device to keep the kernels device-clean.
inline size_t parseSize(const char* v) {
  if (v == nullptr || v[0] == '\0') return kThresholdUnset;
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
//   unset -> unsetDefault (Broadcast: 2 KiB; AllToAll: 0 = off),
//   then clamp to the compile ceiling, then round down to 8 bytes.
GIN_SDMA_HD inline size_t resolveLLCap(size_t envVal, size_t unsetDefault,
                                       size_t maxCap) {
  size_t cap = (envVal == kThresholdUnset) ? unsetDefault : envVal;
  if (cap > maxCap) cap = maxCap;
  return (cap / 8) * 8;
}

// ------------------------------- alignment -------------------------------

// The rccl-tests per-rank/per-peer chunk count, aligned so a chunk is a whole
// number of 16-byte lines: (count/nRanks) & -(16/eltSize). Mirrors
// AllGather/AllToAll GetCollByteCount. eltSize must divide 16 (1,2,4,8,16).
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
// set (clamped to <= max); otherwise size-adaptive like the Reduce multi-ring:
// target ~64 KiB per chunk of THIS CTA's stripe (msgBytes/ctas), clamped to
// [min,max]. Per-CTA sizing (vs the old global msgBytes/target) keeps the chunk
// bytes constant across sizes so small messages are not over-chunked into tiny
// (barrier-bound) chunks. The kernel further clamps to <= sCount so none is empty.
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
// is NOT the scatter+allgather tier). Uses bcastLLEligible so the kernel and the
// tests share the decision.
GIN_SDMA_HD inline BcastTier bcastKernelTier(size_t msgBytes, size_t sdmaThreshold,
                                             int llSlots, size_t llMaxBytes) {
  if (msgBytes <= sdmaThreshold) {
    if (bcastLLEligible(msgBytes, llSlots, llMaxBytes)) return BcastTier::LL;
    return BcastTier::LSADirect;
  }
  return BcastTier::Flat;
}

// ginSignalCount for Broadcast case 3: >=2 so the SAG two-signal scheme works.
GIN_SDMA_HD inline int bcastSignalCount(int deviceCtaCount) {
  return (deviceCtaCount < 2) ? 2 : deviceCtaCount;
}

// Even split with the remainder folded into the last rank's slice (SAG).
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

// ------------------------------- AllGather -------------------------------

GIN_SDMA_HD inline bool agLLEligible(size_t chunkBytes, int llSlots, int nRanks,
                                     size_t llMaxBytes) {
  return llSlots != 0 && (chunkBytes % 8 == 0) && chunkBytes <= llMaxBytes &&
         (size_t)nRanks * (chunkBytes / 8) <= (size_t)llSlots;
}

enum class AGTier { LL, LSASingleCta, LSAMultiCta, Gin };

GIN_SDMA_HD inline AGTier agKernelTier(size_t chunkBytes, size_t sdmaThreshold,
                                       int llSlots, int nRanks, size_t llMaxBytes,
                                       size_t singleCtaMax) {
  if (chunkBytes <= sdmaThreshold) {
    if (agLLEligible(chunkBytes, llSlots, nRanks, llMaxBytes)) return AGTier::LL;
    if (chunkBytes <= singleCtaMax) return AGTier::LSASingleCta;
    return AGTier::LSAMultiCta;
  }
  return AGTier::Gin;
}

// ------------------------------- AllToAll -------------------------------

GIN_SDMA_HD inline bool a2aLLEligible(size_t size, int llSlots, int nRanks,
                                      size_t llMaxBytes) {
  return llSlots != 0 && (size % 8 == 0) && size <= llMaxBytes &&
         (size_t)nRanks * (size / 8) <= (size_t)llSlots;
}

enum class A2ATier { LL, LSA, Gin };

GIN_SDMA_HD inline A2ATier a2aKernelTier(size_t size, size_t sdmaThreshold,
                                         int llSlots, int nRanks, size_t llMaxBytes) {
  if (size <= sdmaThreshold) {
    if (a2aLLEligible(size, llSlots, nRanks, llMaxBytes)) return A2ATier::LL;
    return A2ATier::LSA;
  }
  return A2ATier::Gin;
}

// devComm resource requirements per AllToAll deviceImpl (host-side). `supported`
// is false for unknown deviceImpls; `needsGin` marks the impls that fail when
// the communicator has no GIN backend.
struct DevReqs {
  int barrierCount;
  int lsaBarrierCount;
  int ginSignalCount;
  bool needsGin;
  bool supported;
};
GIN_SDMA_HD inline DevReqs a2aDevReqs(int deviceImpl, int deviceCtaCount) {
  DevReqs r{0, 0, 0, false, true};
  switch (deviceImpl) {
    case 1:  // NvlAlltoAllKernel
    case 2:  // NvlAlltoAllKernelOptimized
      r.lsaBarrierCount = deviceCtaCount;
      return r;
    case 3: {  // GinHybridAlltoAllKernel
      // Size the barrier/signal pools for the largest grid the size-adaptive
      // launch (F1) can request (kA2aLsaMaxCtas), so any per-call adaptive grid
      // uses only a subset of the allocated per-CTA-index barriers -- launching
      // *fewer* CTAs than allocated is always safe (each CTA index syncs its
      // cross-rank counterpart independently); launching more would overflow.
      int lsaMax = deviceCtaCount > kA2aLsaMaxCtas ? deviceCtaCount : kA2aLsaMaxCtas;
      r.barrierCount = lsaMax;
      r.lsaBarrierCount = lsaMax;
      r.ginSignalCount = lsaMax;
      r.needsGin = true;
      return r;
    }
    case 4:  // HybridAlltoAllKernel: CTA 0 = GIN, CTAs 1..N = LSA
      r.barrierCount = 1;
      r.lsaBarrierCount = deviceCtaCount - 1;
      r.ginSignalCount = 1;
      r.needsGin = true;
      return r;
    default:
      r.supported = false;
      return r;
  }
}

// ---------------- Scatter / Gather / SendRecv (pure movement) ----------------
//
// Phase-1 single-phase movement collectives. They share one tier predicate and
// one devComm-requirement shape: a size-hybrid LSA(small) / GIN-SDMA(large) split
// at a per-collective LSA<->GIN threshold (default 256 KiB, retuned by
// measurement per the design plan), and barrier = lsaBarrier = ginSignal =
// deviceCtaCount with needsGin. No scratch. SendRecv additionally has an
// opt-out LL tiny-message tier (below); Scatter/Gather stay LSA/GIN for now.

// Compared against the transfer bytes (Scatter/Gather: per-rank chunk; SendRecv:
// full message). Tuned on 8x MI355X (gfx950, NCCL_GIN_TYPE=6, -V 32, 2026-07-27;
// LSA-forced vs GIN-forced sweeps, gin-sdma-phase1-tune.bash):
//
//   * Scatter LSA is root-egress-bound (only the root SM-stores all N-1 peer
//     chunks, ~64 GB/s ceiling), so GIN/SDMA wins decisively once the per-rank
//     chunk is large: crossover at a 256 KiB chunk (2 MiB total), and by 512 MiB
//     GIN is 390 vs 64 GB/s. Default 128 KiB puts the cutover just below the
//     measured crossover (<=128 KiB chunk -> LSA, >=256 KiB -> GIN).
//   * Gather and SendRecv distribute the writes across all ranks (every rank
//     stores its own slice), so direct LSA beats GIN/SDMA at *every* measured
//     size up to 512 MiB (Gather 432 vs 419, SendRecv 62.4 vs 61.1 GB/s at
//     512 MiB). Their default is set high so LSA is used across all practical
//     sizes; the GIN tier remains available as an env-tunable escape hatch
//     (NCCL_GIN_ANVIL_SDMA_THRESHOLD_{GATHER,SENDRECV}=0 forces it).
static constexpr size_t kScatterSdmaThresholdDefault  = 131072;      // 128 KiB/rank chunk
static constexpr size_t kGatherSdmaThresholdDefault   = 1073741824;  // 1 GiB: LSA-always
static constexpr size_t kSendRecvSdmaThresholdDefault = 1073741824;  // 1 GiB: LSA-always

// AllToAllv LSA<->GIN default; compared against the NOMINAL per-peer chunk
// (count*eltSize, the AllToAll-style average slice). AllToAllv is AllToAll with
// per-peer variable counts (a sparse, per-(sender,receiver) size matrix), so
// there is no single per-peer chunk to key the tier on; the kernel picks ONE
// tier for the whole collective from that nominal slice, mirroring how AllToAll
// keys on its per-peer chunk. Like the AllToAll default this is the *largest
// per-peer chunk the LSA tier still wins* (not the crossover). Measured on 8x
// MI355X (2026-08-05, float, -V 32, -G 4 graph replay for GIN vs launch-inclusive
// LSA; grid-wide LSA scatter) via all-LSA vs all-GIN/SDMA sweeps: LSA leads at
// 256 KiB/peer (2 MiB total, 39.6 vs 36.3 GB/s) and SDMA takes over from
// 384 KiB/peer (3 MiB total, 47.5 vs 41.4; 512 KiB 50.7 vs 43.4; 768 KiB 61.2 vs
// 56.9). So <=256 KiB/peer -> LSA, >256 KiB/peer -> GIN-SDMA. The grid-wide LSA
// rewrite lifted the 128-512 KiB/peer band ~2.7-4.8x (it beats host RING through
// 256 KiB/peer), which is what pulled the crossover down to this clean 256 KiB
// step. Tune with NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALLV or the shared
// NCCL_GIN_ANVIL_SDMA_THRESHOLD.
static constexpr size_t kAllToAllvSdmaThresholdDefault = 262144;     // 256 KiB nominal/peer (measured)

enum class MoveTier { LSA, Gin };

// Tier chosen by the single-phase movement kernels: LSA below/at the threshold,
// GIN/SDMA above it.
GIN_SDMA_HD inline MoveTier moveKernelTier(size_t bytes, size_t sdmaThreshold) {
  return (bytes <= sdmaThreshold) ? MoveTier::LSA : MoveTier::Gin;
}

// AllToAllv tier selector. Same LSA(small)/GIN(large) split as moveKernelTier,
// but named for AllToAllv and keyed on the nominal per-peer chunk (count*eltSize)
// since the actual per-peer sizes vary. Both tiers are PUSH (each rank writes its
// per-peer chunk into that peer's recvbuf), so the shape matches AllToAll case 3
// (barrier = lsaBarrier = ginSignal = deviceCtaCount, needsGin: see moveDevReqs).
GIN_SDMA_HD inline MoveTier a2avKernelTier(size_t nominalPeerBytes, size_t sdmaThreshold) {
  return moveKernelTier(nominalPeerBytes, sdmaThreshold);
}

// devComm requirements for the -D 3 movement kernels (scatter/gather/sendrecv):
// one barrier + one lsaBarrier + one signal per CTA, GIN required.
GIN_SDMA_HD inline DevReqs moveDevReqs(int deviceCtaCount) {
  DevReqs r{deviceCtaCount, deviceCtaCount, deviceCtaCount, true, true};
  return r;
}

// SendRecv LL eligibility (inside the msgBytes<=sdmaThreshold branch): LL
// configured (nSlots>0), 8-byte aligned, within the compile ceiling, and it
// fits the pre-sized slot count. Single incoming message per receiver, so the
// slot demand is msgBytes/8 (mirrors bcastLLEligible, not the nRanks*... forms).
GIN_SDMA_HD inline bool sendRecvLLEligible(size_t msgBytes, int llSlots,
                                           size_t llMaxBytes) {
  return llSlots != 0 && (msgBytes % 8 == 0) && msgBytes <= llMaxBytes &&
         (msgBytes / 8) <= (size_t)llSlots;
}

enum class SendRecvTier { LL, LSA, Gin };

// Tier chosen by GinSendRecvKernel: LL (tiny, one barrier removed) below the LL
// cap, else LSA below/at the SDMA threshold, else GIN/SDMA. Shared by the kernel
// and the host unit tests.
GIN_SDMA_HD inline SendRecvTier sendRecvKernelTier(size_t msgBytes,
                                                   size_t sdmaThreshold,
                                                   int llSlots,
                                                   size_t llMaxBytes) {
  if (msgBytes <= sdmaThreshold) {
    if (sendRecvLLEligible(msgBytes, llSlots, llMaxBytes)) return SendRecvTier::LL;
    return SendRecvTier::LSA;
  }
  return SendRecvTier::Gin;
}

// Scatter LL eligibility (inside the chunkBytes<=sdmaThreshold branch): LL
// configured (nSlots>0), 8-byte aligned, within the compile ceiling, and it
// fits the pre-sized slot count. Each receiver takes ONE chunk from the root, so
// the slot demand is chunkBytes/8 (mirrors bcast/sendRecvLLEligible, not the
// nRanks*... forms). Compared against the per-rank chunk bytes.
GIN_SDMA_HD inline bool scatterLLEligible(size_t chunkBytes, int llSlots,
                                          size_t llMaxBytes) {
  return llSlots != 0 && (chunkBytes % 8 == 0) && chunkBytes <= llMaxBytes &&
         (chunkBytes / 8) <= (size_t)llSlots;
}

enum class ScatterTier { LL, LSA, Gin };

// Tier chosen by GinScatterKernel: LL (tiny, exit barrier removed) below the LL
// cap, else LSA below/at the SDMA threshold, else GIN/SDMA. Compared against the
// per-rank chunk bytes. Shared by the kernel and the host unit tests.
GIN_SDMA_HD inline ScatterTier scatterKernelTier(size_t chunkBytes,
                                                 size_t sdmaThreshold,
                                                 int llSlots,
                                                 size_t llMaxBytes) {
  if (chunkBytes <= sdmaThreshold) {
    if (scatterLLEligible(chunkBytes, llSlots, llMaxBytes)) return ScatterTier::LL;
    return ScatterTier::LSA;
  }
  return ScatterTier::Gin;
}

// Gather LL eligibility (inside the chunkBytes<=sdmaThreshold branch): LL
// configured (nSlots>0), 8-byte aligned, within the compile ceiling, and it fits
// the pre-sized slot count. All-to-one fan-in: the ROOT holds nRanks chunks in
// its scratch, so the slot demand is nRanks*(chunk/8) -- the AllGather form, not
// Scatter's single-message form. Compared against the per-rank chunk bytes.
GIN_SDMA_HD inline bool gatherLLEligible(size_t chunkBytes, int llSlots,
                                         int nRanks, size_t llMaxBytes) {
  return llSlots != 0 && (chunkBytes % 8 == 0) && chunkBytes <= llMaxBytes &&
         (size_t)nRanks * (chunkBytes / 8) <= (size_t)llSlots;
}

enum class GatherTier { LL, LSA, Gin };

// Tier chosen by GinGatherKernel: LL (tiny, exit barrier removed) below the LL
// cap, else LSA below/at the SDMA threshold (default 1 GiB = LSA-always), else
// GIN/SDMA. Compared against the per-rank chunk bytes. Shared by the kernel and
// the host unit tests.
GIN_SDMA_HD inline GatherTier gatherKernelTier(size_t chunkBytes,
                                               size_t sdmaThreshold,
                                               int llSlots, int nRanks,
                                               size_t llMaxBytes) {
  if (chunkBytes <= sdmaThreshold) {
    if (gatherLLEligible(chunkBytes, llSlots, nRanks, llMaxBytes)) return GatherTier::LL;
    return GatherTier::LSA;
  }
  return GatherTier::Gin;
}

// ---------------------- ReduceScatter (Phase-2, reduction) ----------------------
//
// First reduction collective: adds an SM-side reduce (SDMA/LSA only move bytes).
// Two-tier ladder keyed on the per-rank output-slice bytes (chunk = recvcount *
// eltSize = sendBytes/N):
//   * Small/med -- LSA read-reduce: every rank reads its owned slice [rank*chunk]
//     from EVERY peer's sendbuff via ncclGetLsaPointer, folds with Apply<op,T>,
//     writes recvbuff. Egress is balanced (each rank pulls N-1 remote slices), so
//     no scratch and no signals -- just an entry+exit LSA barrier. This is the
//     latency-optimal small path.
//   * Large -- put-partials + SM reduce: each rank gin.puts its partial for peer
//     p (its slice p) into p's scratch window at slot [srcRank*chunk]
//     (SignalInc{0}); after waitSignal(base + N-1) each rank SM-reduces the N
//     contributions in its scratch into recvbuff; flush. Egress balanced across
//     all N ranks (the ReduceScatter roofline).
//
// Threshold default 256 KiB/rank slice (provisional; retune by measurement per
// the design plan). Compared against the per-rank slice bytes.
static constexpr size_t kReduceScatterSdmaThresholdDefault = 262144;  // 256 KiB/rank slice

enum class RSTier { LSA, Gin };

// Tier chosen by GinReduceScatterKernel: LSA read-reduce below/at the threshold,
// else put-partials GIN/SDMA + SM reduce. Compared against the per-rank slice
// bytes. Shared by the kernel and the host unit tests.
GIN_SDMA_HD inline RSTier reduceScatterKernelTier(size_t chunkBytes,
                                                  size_t sdmaThreshold) {
  return (chunkBytes <= sdmaThreshold) ? RSTier::LSA : RSTier::Gin;
}

// devComm requirements for the -D 3 ReduceScatter kernel: one barrier +
// lsaBarrier + signal per CTA, GIN required. The large tier additionally needs a
// scratch window (see reduceScatterScratchBytes); that requirement is added
// separately in the .cu (mirroring how the LL scratch is wired), so DevReqs here
// carries only the barrier/signal shape.
GIN_SDMA_HD inline DevReqs reduceScatterDevReqs(int deviceCtaCount) {
  DevReqs r{deviceCtaCount, deviceCtaCount, deviceCtaCount, true, true};
  return r;
}

// ReduceScatter -D 3 size-adaptive CTA count (decoupled from -V, mirrors the
// broadcast/reduce rings). The LSA read-reduce is occupancy-bound in the
// grid-stride mid-band [kReduceScatterCtaMidLo, RS_UNROLL_MIN): ~48 CTAs peaks
// there (33 MiB 88->~100% of host, 16 MiB ->86%), while the small tier and the
// warp-unrolled large tier (>= RS_UNROLL_MIN) peak at 32 -- more CTAs crater the
// unroll path (67 MiB 249->153 busbw at 64 CTAs). The bare -V default (16) badly
// under-launches the mid-band (16 MiB ~46%, 33 MiB ~43% of host); self-selecting
// repairs that for callers that don't pass -V. NCCL_GIN_ANVIL_RS_CTAS pins a
// fixed count for all sizes (diagnostic).
static constexpr int    kReduceScatterCtasMid   = 48;                    // grid-stride mid-band
static constexpr int    kReduceScatterCtasOther = 32;                    // small + warp-unroll large
static constexpr size_t kReduceScatterCtaMidLo  = 8ull  * 1024 * 1024;   // >= -> mid band
static constexpr size_t kReduceScatterCtaMidHi  = 48ull * 1024 * 1024;   // <  -> mid band (== RS_UNROLL_MIN)
GIN_SDMA_HD inline int reduceScatterCtas(size_t totalBytes, size_t envCtas) {
  if (envCtas != kThresholdUnset && envCtas > 0)
    return (envCtas > 128) ? 128 : (int)envCtas;
  if (totalBytes >= kReduceScatterCtaMidLo && totalBytes < kReduceScatterCtaMidHi)
    return kReduceScatterCtasMid;
  return kReduceScatterCtasOther;
}
GIN_SDMA_HD inline int reduceScatterMaxCtas() {
  return (kReduceScatterCtasMid > kReduceScatterCtasOther) ? kReduceScatterCtasMid
                                                           : kReduceScatterCtasOther;
}

// Bytes of scratch-window the large tier needs per rank: it stages N incoming
// per-source partials, each up to the largest per-rank slice, so it needs the
// full per-rank send-buffer worth (N * maxChunkBytes == maxSendBytesPerRank).
// Sized once at the worst-case (max) message; rounded up to the 128 B
// resource-buffer granularity. Zero maxSendBytes -> zero (no scratch).
GIN_SDMA_HD inline size_t reduceScatterScratchBytes(size_t maxSendBytesPerRank) {
  if (maxSendBytesPerRank == 0) return 0;
  return (maxSendBytesPerRank + 127) & ~(size_t)127;
}

// ------------------------------- AllReduce -------------------------------
//
// GIN-SDMA AllReduce is deviceImpl 5 (single-launch) / 6 (two-launch); the upstream
// LSA/multimem demo kernels keep -D 1..4. Both realizations run ReduceScatter (LSA
// read-reduce) then an in-place AllGather, and this threshold selects the AllGather TIER
// by TOTAL message bytes (AllReduce operates on the whole buffer):
//   * small (< threshold): LSA AllGather -- each rank STORES its reduced slice directly
//     into every peer's recvbuf slot over xGMI (F2+F3 link spreading), released by a
//     single GIN world barrier. Drops the SDMA put + waitSignal round-trip, cutting
//     small/mid-message latency, while keeping exactly one GIN barrier per launch (so the
//     GIN completion cadence stays uniform -- see the hang NOTE in all_reduce.cu).
//   * large (>= threshold): SDMA AllGather -- one put of the reduced slice to each peer on
//     a single recycled GIN signal + waitSignal. Bandwidth-optimal; LSA store-gather
//     collapses at large sizes (128M: ~218 vs ~366 GB/s), so SDMA is kept for the tail.
// Default measured on 8x MI355X (2026-08-02, float sum): LSA AllGather wins out-of-place
// up to ~8 MiB (4M ~108 vs 101 GB/s, 8M ~158 vs 153) and in-place a bit higher, while SDMA
// wins from 16 MiB up (16M ~209 vs 198). 16 MiB keeps LSA for the sizes it wins.
static constexpr size_t kAllReduceSdmaThresholdDefault = 16777216;  // 16 MiB total (measured)

// TINY out-of-place AllReduce one-shot cutoff (total bytes). Below this, -D 5 uses a
// single-barrier LSA read-reduce (arAllReduceOneShotLsa) that mirrors the host-initiated
// single-round design -- one GIN world barrier then read all peers' sendbuf and reduce
// locally -- to cut the fixed latency that dominates tiny messages. It reads N*msgBytes, so
// it only wins while latency-bound; above this the RS+AG tier (moves ~2*msgBytes) takes over.
// Override with NCCL_GIN_ANVIL_ONESHOT_THRESHOLD_ALLREDUCE. Measured crossover on 8x MI355X.
static constexpr size_t kAllReduceOneShotThresholdDefault = 262144;  // 256 KiB total (measured)

// Two-shot (large/in-place) CTA cap. The two-shot path issues a PER-CTA world GIN
// barrier + AllGather puts; on 8x MI355X with NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 a
// dense sweep at -V 32 deadlocks (cumulative GIN/SDMA resource pressure from 32
// concurrent per-CTA barriers/puts), while -V 8 and -V 16 sweep cleanly to 128 MiB.
// The two-shot grid is therefore capped to this many CTAs (extra CTAs return
// early); the one-shot / LSA tiers are unaffected and keep the full launch grid.
static constexpr int kAllReduceTwoShotMaxCtas = 16;

enum class ARTier { LSA, Gin };

// Shared by GinAllReduceKernel and the host unit tests. Compared against total
// message bytes.
GIN_SDMA_HD inline ARTier allReduceKernelTier(size_t msgBytes, size_t sdmaThreshold) {
  return (msgBytes <= sdmaThreshold) ? ARTier::LSA : ARTier::Gin;
}

// GIN signal count for the -D 5 two-shot scheme. AG (and variant B's RS) use a
// PER-CTA signal (index = blockIdx.x) so each CTA's completion count is
// independent, so one signal per CTA is needed. Floored at 2 for the degenerate
// single-CTA case.
GIN_SDMA_HD inline int allReduceSignalCount(int deviceCtaCount) {
  return (deviceCtaCount < 2) ? 2 : deviceCtaCount;
}

// devComm requirements for the -D 5 GIN-SDMA AllReduce kernel: one barrier +
// lsaBarrier per CTA (small tier needs only the lsaBarrier; the large tier adds
// the world barrier) and allReduceSignalCount GIN signals for the two-shot
// phases. GIN required. The scratch window (variant B) is added separately in
// the .cu (mirroring ReduceScatter), so DevReqs carries only the barrier/signal
// shape.
GIN_SDMA_HD inline DevReqs allReduceDevReqs(int deviceCtaCount) {
  DevReqs r{deviceCtaCount, deviceCtaCount, allReduceSignalCount(deviceCtaCount), true, true};
  return r;
}

// Per-slot byte stride for the two-shot slice split: ceil(msgBytes/nRanks)
// rounded up to 16 B so every rank's slice starts 128-bit aligned (packed
// read-reduce). Shared by the kernel (slice math) and the scratch sizing so the
// writer's slot offset (source s -> slot s*stride) matches the allocation.
GIN_SDMA_HD inline size_t allReduceSliceStride(size_t msgBytes, int nRanks) {
  if (nRanks <= 0) return 0;
  size_t per = (msgBytes + (size_t)nRanks - 1) / (size_t)nRanks;
  return (per + 15) & ~(size_t)15;
}

// Bytes of scratch the large put-partials RS variant (B) needs per rank: it
// stages N incoming per-source partials, each one slice-slot wide
// (allReduceSliceStride), so N * stride. Sized once at the worst-case (max)
// message and rounded up to the 128 B resource-buffer granularity. Because the
// stride is monotonic in msgBytes, sizing at maxMsgBytes covers every smaller
// message. Zero -> no scratch. Variant A needs no scratch.
GIN_SDMA_HD inline size_t allReduceScratchBytes(size_t maxMsgBytes, int nRanks) {
  if (maxMsgBytes == 0 || nRanks <= 0) return 0;
  size_t total = allReduceSliceStride(maxMsgBytes, nRanks) * (size_t)nRanks;
  return (total + 127) & ~(size_t)127;
}

}  // namespace gin_sdma

#endif  // GIN_SDMA_COLLECTIVE_POLICY_H_
