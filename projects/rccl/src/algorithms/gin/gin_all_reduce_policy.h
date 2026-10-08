/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Size and alignment policy for GIN-SDMA AllReduce. Included by gin_all_reduce.h
 * so unit tests exercise the same gates ncclAllReduceGinSdmaEligible() uses after
 * the comm / buffer / gfx950 / gfx1250 checks pass. Chunk-plan helpers are
 * host+device so the GinScatter kernel splits the same way the tests do.
 *
 * gfx950 default: GIN two-shot from 256 MiB (LSA RS + GIN AG, Anvil SDMA).
 * gfx1250 + Anvil SDMA: GinScatter from 64 MiB.
 * GinScatter scratch follows CE AllReduce (ce_reduce_impl.h / ce_coll.h):
 * 512 MiB of incoming ping-pong slots, reduced output in the user recvbuff.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_POLICY_H_
#define GIN_ALL_REDUCE_POLICY_H_

#include <cstddef>

constexpr int kGinAllReduceLsaCtas = 56;
constexpr int kGinAllReduceLsaTwoShotCtasPerPeer = 8;
constexpr int kGinAllReduceLsaTwoShotMaxCtas = kGinAllReduceLsaTwoShotCtasPerPeer * 16;
constexpr int kGinAllReduceMaxRanks = 8;

constexpr int kGinAllReduceMinBytes = 512ULL * 1024;
constexpr int kGinAllReduceLsaThreadsPerCta = 512;
constexpr size_t kGinAllReduceLsaOneShotMaxBytes = 4ULL * 1024 * 1024;
constexpr size_t kGinAllReduceLsaTwoShotMidBytes = 32ULL * 1024 * 1024;
constexpr size_t kGinAllReduceGinTwoShotMinBytes = 256ULL * 1024 * 1024;
// GinScatter (CE-style local reduce) from this size inclusive on gfx1250.
constexpr size_t kGinAllReduceGinScatterMinBytes = 64ULL * 1024 * 1024;
// Put/reduce overlap starts here. Below this, ginScatter is one full scatter then
// one full reduce (or sequential slot reuse if scratch cannot hold the shard).
constexpr size_t kGinAllReduceGinScatterPipelineMinBytes = 512ULL * 1024 * 1024;
// Largest AllReduce ginScatter is sized for (2 GiB and 4 GiB included).
constexpr size_t kGinAllReduceGinScatterMaxBytes = 4ULL * 1024 * 1024 * 1024;
// CE ceARTmpBuf analogue: 512 MiB of incoming ping-pong. Reduced shard is recvbuff.
constexpr size_t kGinAllReduceTwoShotScratchBytes = 512ULL * 1024 * 1024;

constexpr size_t kGinAllReduceMinPutBytes = 128;

// Pipeline granule when the AllReduce is >= 512 MiB and scratch holds the shard.
// With the default 512 MiB scratch, messages above 512 MiB are staged and the
// slot (scratch / (nSlots * nRanks)) is the granule instead.
constexpr size_t kGinAllReduceGinScatterChunkBytes = 64ULL * 1024 * 1024;
constexpr size_t kGinAllReduceGinScatterChunkAlign = 16;
// Matches NCCL_CE_NUM_SLOTS: ping-pong scatter staging when the shard does not
// fit in one incoming column.
constexpr int kGinAllReduceGinScatterNumSlots = 2;

#if defined(__HIPCC__) || defined(__CUDACC__)
#define GIN_AR_HD __host__ __device__ __forceinline__
#else
#define GIN_AR_HD inline
#endif

GIN_AR_HD size_t ginAllReduceGinScatterAlignDown(size_t n, size_t align) {
  if (align == 0) return n;
  return (n / align) * align;
}

// CE ceARTmpBuf: nSlots * nRanks * slotBytes. Reduced output is the user recvbuff.
GIN_AR_HD size_t ginAllReduceGinScatterStagedScratchBytes(size_t slotBytes, int nRanks, int nSlots) {
  if (nRanks <= 0 || nSlots <= 0) return 0;
  return static_cast<size_t>(nSlots) * static_cast<size_t>(nRanks) * slotBytes;
}

// Largest 16-byte-aligned per-rank slot that fits in scratch with nSlots windows.
GIN_AR_HD size_t ginAllReduceGinScatterFitSlot(size_t scratchBytes, int nRanks, int nSlots) {
  if (nRanks <= 0 || nSlots <= 0) return 0;
  const size_t denom = static_cast<size_t>(nSlots) * static_cast<size_t>(nRanks);
  return ginAllReduceGinScatterAlignDown(scratchBytes / denom, kGinAllReduceGinScatterChunkAlign);
}

// True when scratch cannot hold nRanks full incoming columns (CE pipelined path).
GIN_AR_HD bool ginAllReduceGinScatterStaged(size_t perRankBytes, int nRanks, size_t scratchBytes) {
  if (nRanks <= 0) return false;
  return scratchBytes < static_cast<size_t>(nRanks) * perRankBytes;
}

// Split one per-rank slice. Full-slot path: one chunk below 512 MiB, 16 MiB
// stages at 512 MiB+. Staged path (message > scratch): CE ping-pong — two slots
// and overlap at >= 512 MiB; one slot and sequential reuse below 512 MiB.
GIN_AR_HD void ginAllReduceGinScatterChunkPlan(size_t perRankBytes, int nRanks, size_t scratchBytes, int* nChunks,
                                               size_t* uniformBytes, size_t* lastBytes, int* nPhases) {
  const size_t maxChunk = kGinAllReduceGinScatterChunkBytes;
  const size_t align = kGinAllReduceGinScatterChunkAlign;
  const size_t totalBytes = nRanks > 0 ? perRankBytes * static_cast<size_t>(nRanks) : 0;
  if (perRankBytes == 0) {
    *nChunks = 1;
    *uniformBytes = 0;
    *lastBytes = 0;
    *nPhases = 1;
    return;
  }
  if (!ginAllReduceGinScatterStaged(perRankBytes, nRanks, scratchBytes)) {
    *nPhases = 1;
    if (totalBytes < kGinAllReduceGinScatterPipelineMinBytes || perRankBytes <= maxChunk) {
      *nChunks = 1;
      *uniformBytes = perRankBytes;
      *lastBytes = perRankBytes;
      return;
    }
    int n = static_cast<int>((perRankBytes + maxChunk - 1) / maxChunk);
    size_t uniform = (perRankBytes / static_cast<size_t>(n) / align) * align;
    if (uniform < align) uniform = align;
    *nChunks = n;
    *uniformBytes = uniform;
    *lastBytes = perRankBytes - uniform * static_cast<size_t>(n - 1);
    return;
  }

  int phases = (totalBytes >= kGinAllReduceGinScatterPipelineMinBytes) ? kGinAllReduceGinScatterNumSlots : 1;
  size_t slot = ginAllReduceGinScatterFitSlot(scratchBytes, nRanks, phases);
  if (slot < align && phases > 1) {
    phases = 1;
    slot = ginAllReduceGinScatterFitSlot(scratchBytes, nRanks, 1);
  }
  if (slot < align) {
    *nChunks = 1;
    *uniformBytes = perRankBytes;
    *lastBytes = perRankBytes;
    *nPhases = 1;
    return;
  }
  if (perRankBytes <= slot) {
    *nChunks = 1;
    *uniformBytes = perRankBytes;
    *lastBytes = perRankBytes;
    *nPhases = 1;
    return;
  }
  int n = static_cast<int>((perRankBytes + slot - 1) / slot);
  *nChunks = n;
  *uniformBytes = slot;
  *lastBytes = perRankBytes - slot * static_cast<size_t>(n - 1);
  *nPhases = phases;
}

GIN_AR_HD size_t ginAllReduceGinScatterChunkOff(int chunk, size_t uniformBytes) {
  return static_cast<size_t>(chunk) * uniformBytes;
}

GIN_AR_HD size_t ginAllReduceGinScatterChunkSize(int chunk, int nChunks, size_t uniformBytes, size_t lastBytes) {
  return (chunk == nChunks - 1) ? lastBytes : uniformBytes;
}

#undef GIN_AR_HD

// Full-slot incoming columns (no reduced scratch column; CE writes the reduced
// shard to recvbuff). Used to decide whether the message fits without staging.
inline size_t ginAllReduceGinScatterScratchBytes(size_t chunkBytes, int nRanks) {
  if (nRanks <= 0) return 0;
  return static_cast<size_t>(nRanks) * chunkBytes;
}

// Two-shot kernels require a whole number of elements per rank and a 16-byte
// per-rank slice so vector loads stay aligned.
inline bool ginAllReduceTwoShotEligible(size_t count, size_t typeSize, int nRanks) {
  if (nRanks <= 0 || typeSize == 0) return false;
  if (count % static_cast<size_t>(nRanks) != 0) return false;
  const size_t countPerRank = count / static_cast<size_t>(nRanks);
  return (countPerRank * typeSize) % 16 == 0;
}

inline bool ginAllReduceGinTwoShotEligible(size_t count, size_t typeSize, int nRanks) {
  if (!ginAllReduceTwoShotEligible(count, typeSize, nRanks)) return false;
  const size_t chunkBytes = (count / static_cast<size_t>(nRanks)) * typeSize;
  return chunkBytes >= kGinAllReduceMinPutBytes;
}

// ginScatter from 64 MiB when scratch holds either nRanks full incoming columns
// or at least one 16-byte-aligned staging slot per rank.
inline bool ginAllReduceGinScatterLaunch(size_t bytes, size_t chunkBytes, int nRanks, size_t scratchBytes) {
  if (bytes < kGinAllReduceGinScatterMinBytes) return false;
  if (nRanks <= 0) return false;
  if (scratchBytes >= ginAllReduceGinScatterScratchBytes(chunkBytes, nRanks)) return true;
  return ginAllReduceGinScatterFitSlot(scratchBytes, nRanks, 1) >= kGinAllReduceGinScatterChunkAlign;
}

// Size-only GIN AllReduce policy. forceEnable is RCCL_GIN_ALLREDUCE_FORCE_ENABLE==1.
// ginMinBytes is the default-path floor (256 MiB two-shot on gfx950, 64 MiB
// GinScatter on gfx1250). Force: LSA one-shot <= 4 MiB (and >= 512 KiB), LSA
// two-shot in between, GIN path at ginMinBytes+.
inline bool ginAllReduceSizePolicyEligible(size_t count, size_t typeSize, int nRanks, bool forceEnable,
                                           size_t ginMinBytes = kGinAllReduceGinTwoShotMinBytes) {
  const size_t bytes = count * typeSize;
  if (bytes < ginMinBytes && !forceEnable) return false;
  if (bytes < static_cast<size_t>(kGinAllReduceMinBytes)) return false;
  // Inclusive 4 MiB: one-shot has no per-rank alignment requirement. Two-shot
  // starts strictly above this, matching ncclAllReduceGinSdmaTyped().
  if (bytes <= kGinAllReduceLsaOneShotMaxBytes) return true;
  if (bytes >= ginMinBytes) {
    return ginAllReduceGinTwoShotEligible(count, typeSize, nRanks);
  }
  return ginAllReduceTwoShotEligible(count, typeSize, nRanks);
}

// Smaller-than-ginMinBytes GIN candidates yield to DDA unless FORCE_ENABLE=1.
inline bool ginAllReduceYieldToDdaBySize(size_t count, size_t typeSize, bool forceEnable,
                                         size_t ginMinBytes = kGinAllReduceGinTwoShotMinBytes) {
  if (forceEnable) return false;
  return count * typeSize < ginMinBytes;
}

// Host dispatch for ginAllReduceTwoShotGinScatterKernel: gfx1250 and >= 64 MiB.
// Eligibility still requires the Anvil SDMA GIN backend before launch.
inline bool ginAllReduceGinScatterDispatch(bool isGfx1250, size_t bytes) {
  return isGfx1250 && bytes >= kGinAllReduceGinScatterMinBytes;
}

#endif
