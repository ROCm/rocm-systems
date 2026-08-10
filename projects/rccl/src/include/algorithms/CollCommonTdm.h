/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Tensor-data-mover (TDM) implementations of the DDA Simple-protocol
 * primitives in CollCommon.h.
 *
 * The Simple protocol moves bulk data with per-lane uint4 loads/stores. On
 * gfx1250 the same traffic can be driven by the tensor data mover, which DMAs
 * global <-> LDS without occupying the vector memory pipe. These helpers are
 * shape-for-shape replacements for copyFromSrcToDest / allGather /
 * reduceScatter, so the fabric kernels can pick an implementation without
 * changing their phase structure or their barrier placement.
 *
 * Two properties of TDM drive the differences from CollCommon.h:
 *
 *   1. Descriptors live in SGPRs, so every address and length must be
 *      wave-uniform. The CollCommon helpers hand each lane its own offset
 *      (gtIdx * countPerThread); these helpers instead give each warp a
 *      contiguous kTdmWindowBytes tile and grid-stride over tiles.
 *
 *   2. A transfer is staged through an LDS window, so the round trip is
 *      global -> LDS -> global. Copies reuse one window per warp; the
 *      reduction double-buffers so a peer's fabric read overlaps folding the
 *      previous peer out of the other window.
 *
 * ALIGNMENT AND WHY IT IS CHECKED ON THE DEVICE
 * ---------------------------------------------
 * asyncLoadToLDS/asyncStoreFromLDS peel a head against the global pointer and
 * apply the same offset to the LDS side, so a load/store pair through one
 * window is only coherent when source and destination share their sub-128B
 * offset. Requiring both to be kTdmAlign-aligned satisfies that and lets the
 * peel compile out entirely.
 *
 * That check cannot live in the host eligibility predicate. FabricGpuBarrier
 * synchronises rank-to-rank on matching blockIdx, so every rank must launch an
 * identical grid; a user buffer that happens to be aligned on one rank and not
 * on another would produce different grids and hang the barrier. The existing
 * DDA eligibility predicates are careful to ignore the buffer pointers for the
 * same reason. So the grid is sized from counts alone and each range picks TDM
 * or the vector fallback at run time, on a condition that is uniform across the
 * block. Mixed choices across ranks are harmless: the barrier only cares about
 * the grid.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/CollCommon.h"
#include "tdm/tdmCopy.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace meta::comms {

// TDM bulk transfers are issued as rows of this width; a window is a whole
// number of rows and tile offsets are window multiples, so every TDM address
// stays row-aligned.
constexpr uint32_t kTdmRowBytes = 256;

// TDM is gfx1250-only and gfx1250 is wave32. Fixing the lane count lets the
// reduction hold its accumulator in a compile-time sized register array.
constexpr uint32_t kTdmLanes = 32;

// ---- tunables ------------------------------------------------------------
// Per-warp LDS staging window. Every path here is double-buffered, so a block
// reserves two of these per issuing warp.
constexpr uint32_t kTdmWindowBytes = 2048;

// Warps per block that issue TDM, and therefore the block size. TDM has one
// engine per SIMD pair, so a block needs several issuing warps to keep the
// engines busy; beyond that, bandwidth comes from having many blocks resident.
constexpr uint32_t kTdmIssueWarps = 8;
// --------------------------------------------------------------------------

constexpr uint32_t kTdmThreadsPerBlock = kTdmIssueWarps * kTdmLanes;

// Two windows per issuing warp: both the copy and the reduction software
// pipeline across a pair of windows (see tdmCopyRangeBytes / tdmReduceRangeBytes).
constexpr uint32_t kTdmLdsBytes = 2 * kTdmIssueWarps * kTdmWindowBytes;

// gfx1250 allows a workgroup to reserve up to 320 KiB of LDS, and the per-CU
// total is that same number -- so LDS per block sets residency directly:
//
//     blocksPerCu = 327680 / kTdmLdsBytes
//
// The grid is capped at DDA_FABRIC_MAXBLOCKS (256) over 32 CUs, i.e. 8 blocks
// per CU, so a footprint at or under 40 KiB keeps the whole grid resident. The
// defaults above sit at 32 KiB (10 blocks/CU) and therefore cost no occupancy.
// There is a lot of headroom left: raising kTdmWindowBytes amortises the
// per-descriptor cost over more bytes and raising kTdmIssueWarps engages more
// engines per block, but each halving of blocksPerCu gives up latency hiding
// across blocks, and a larger window also grows the reduction's register
// accumulator (kTdmVecPerLane below). Worth a sweep on hardware.
constexpr uint32_t kTdmMaxLdsPerBlockBytes = 320 * 1024;
static_assert(kTdmWindowBytes % kTdmRowBytes == 0, "window must be a whole number of TDM rows");
static_assert(kTdmLdsBytes <= kTdmMaxLdsPerBlockBytes, "TDM staging exceeds the per-workgroup LDS limit");
static_assert(kTdmThreadsPerBlock <= 1024, "block exceeds the hardware thread limit");

// uint4 slots per lane in one window, i.e. the reduction's accumulator depth.
constexpr uint32_t kTdmVecPerLane = kTdmWindowBytes / (kTdmLanes * sizeof(uint4));
static_assert(kTdmVecPerLane * kTdmLanes * sizeof(uint4) == kTdmWindowBytes, "window must divide evenly by lane");

// Sub-128B offset that a source/destination pair must share for a staged round
// trip to be coherent; kTdmRowBytes is a multiple of it and keeps bulk rows
// aligned too.
constexpr uintptr_t kTdmAlign = kTdmRowBytes;

// Fabric peers are read through the system scope; the staged data is streamed
// once and not revisited, so mark it non-temporal.
constexpr CachePolicy kTdmCachePolicy = createCachePolicy(TemporalHint::NT, MemScope::SYS);

// Block-uniform test that a staged round trip between these two ranges is
// representable in one LDS window. Uniform because the pointers are kernel
// arguments, so every thread in the block agrees.
__device__ __forceinline__ bool tdmPairAligned(const void* a, const void* b) {
  return (((uintptr_t)a | (uintptr_t)b) & (kTdmAlign - 1)) == 0;
}

__device__ __forceinline__ bool tdmPtrAligned(const void* a) {
  return ((uintptr_t)a & (kTdmAlign - 1)) == 0;
}

// Retire this wave's outstanding LDS reads. The tensor data mover writes LDS
// behind the compiler's back, so a window being refilled must be fenced against
// the ds_reads that folded its previous contents (WAR); s_wait_tensorcnt only
// covers the TDM side.
__device__ __forceinline__ void tdmWaitLdsReads() {
#if TDM_SUPPORTED
  asm volatile("s_wait_dscnt 0x0" ::: "memory");
#endif
}

// How a warp's tiles are laid out across the grid. Only the first
// kTdmIssueWarps warps of a block issue TDM; the rest sit out the transfer and
// rejoin at the caller's barrier.
struct TdmWarpTile {
  uint32_t lane;
  uint32_t gWarp;   // this warp's index among all issuing warps in the grid
  uint32_t nGWarps; // total issuing warps in the grid
  bool issuer;
};

// This warp's `which` (0 or 1) staging window inside the block-wide LDS buffer.
// Warps beyond kTdmIssueWarps do not issue, so folding their slot back into
// range just keeps the arithmetic in bounds; they never touch the memory.
__device__ inline uint8_t* tdmWindow(uint8_t* lds, uint32_t which) {
  const uint32_t slot = (threadIdx.x / kTdmLanes) % kTdmIssueWarps;
  return lds + (slot * 2 + which) * kTdmWindowBytes;
}

__device__ inline TdmWarpTile tdmWarpTile() {
  const uint32_t warpId = threadIdx.x / kTdmLanes;
  const uint32_t warpsInBlock = (blockDim.x + kTdmLanes - 1) / kTdmLanes;
  const uint32_t issuers = warpsInBlock < kTdmIssueWarps ? warpsInBlock : kTdmIssueWarps;

  TdmWarpTile t;
  t.lane = threadIdx.x % kTdmLanes;
  t.issuer = warpId < issuers;
  t.gWarp = blockIdx.x * issuers + warpId;
  t.nGWarps = gridDim.x * issuers;
  return t;
}

// ---------------------------------------------------------------------------
// Vector fallbacks
//
// Used when a range's pointers are not mutually aligned. These mirror the
// CollCommon loops but index in bytes and are driven by the whole block, since
// the alignment branch is block-uniform and the non-issuing warps are awake.
// ---------------------------------------------------------------------------

__device__ inline void vecCopyRangeBytes(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t nbytes) {
  const size_t nVec = nbytes / sizeof(uint4);
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nVec; i += (size_t)gridDim.x * blockDim.x) {
    d[i] = s[i];
  }
}

template <typename T, int NRANKS_CT, bool hasAcc>
__device__ inline void vecReduceRangeBytes(uint8_t* const* __restrict__ peers, uint8_t* __restrict__ dst,
                                           const uint8_t* __restrict__ acc, int nRanksRuntime, size_t srcOff,
                                           size_t dstOff, size_t nbytes) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t nVec = nbytes / sizeof(uint4);
  uint4* d = reinterpret_cast<uint4*>(dst + dstOff);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nVec; i += (size_t)gridDim.x * blockDim.x) {
    uint4 sum{0, 0, 0, 0};
    if constexpr (hasAcc) {
      sum = reinterpret_cast<const uint4*>(acc + srcOff)[i];
    }
#pragma unroll kUnroll
    for (int r = 0; r < nRanks; ++r) {
      sum = vecElementAdd<T>(sum, reinterpret_cast<const uint4*>(peers[r] + srcOff)[i]);
    }
    d[i] = sum;
  }
}

// ---------------------------------------------------------------------------
// TDM primitives
//
// The tdm:: entry points are `= delete` on targets without a tensor data mover,
// so the bodies below are compiled only into the gfx1250 device pass. The other
// passes of the fat binary keep the symbols (the host launcher references them)
// but fall through to the vector path, and are never dispatched because the
// launcher gates on tdmSimpleSupported().
// ---------------------------------------------------------------------------

// Staged copy of a byte range, replacing copyFromSrcToDest. Each issuing warp
// walks a grid-strided sequence of kTdmWindowBytes tiles.
//
// The two legs are software pipelined across a pair of windows: tile i's store
// is issued and then left in flight while tile i+1's load runs, so the fabric
// read and the writeback overlap. A single window cannot do this -- the store
// reads exactly what the load just wrote (RAW) and the next load overwrites
// what the store is still draining (WAR), so both edges need a wait and the
// warp ends up latency-bound issuing one transfer at a time. Alternating
// windows costs one wait per tile instead of two.
//
// A window-sized aligned tile is a single TDM descriptor, so the steady state
// is two ops in flight, inside the per-wave TENSORcnt limit of three. Only the
// ragged last tile can add a sub-row tail op and momentarily exceed it, which
// just throttles issue.
template <CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmCopyRangeBytes(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t nbytes,
                                         uint8_t* window0, uint8_t* window1, const TdmWarpTile& t) {
#if TDM_SUPPORTED
  if (!t.issuer) {
    return;
  }
  const size_t stride = (size_t)t.nGWarps * kTdmWindowBytes;
  size_t off = (size_t)t.gWarp * kTdmWindowBytes;
  if (off >= nbytes) {
    return;
  }

  uint32_t n = (nbytes - off) < kTdmWindowBytes ? (uint32_t)(nbytes - off) : kTdmWindowBytes;
  tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(src + off, window0, n);

  for (uint32_t parity = 0; off < nbytes; ++parity) {
    uint8_t* cur = (parity & 1) ? window1 : window0;
    uint8_t* nxt = (parity & 1) ? window0 : window1;
    const size_t nextOff = off + stride;

    // Retires this tile's load and the previous tile's store, which together
    // are the only ops this wave has outstanding.
    tdm::tdmWait();
    tdm::asyncStoreFromLDS<SyncPolicy::Async, cp, true>(cur, dst + off, n);

    if (nextOff < nbytes) {
      n = (nbytes - nextOff) < kTdmWindowBytes ? (uint32_t)(nbytes - nextOff) : kTdmWindowBytes;
      tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(src + nextOff, nxt, n);
    }
    off = nextOff;
  }
  tdm::tdmWait();
#else
  (void)src, (void)dst, (void)nbytes, (void)window0, (void)window1, (void)t;
#endif
}

// copyFromSrcToDest equivalent: TDM when the pair allows it, vector otherwise.
template <CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmCopyRange(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t nbytes,
                                    uint8_t* window0, uint8_t* window1, const TdmWarpTile& t) {
#if TDM_SUPPORTED
  if (tdmPairAligned(src, dst)) {
    tdmCopyRangeBytes<cp>(src, dst, nbytes, window0, window1, t);
    return;
  }
#else
  (void)window0, (void)window1, (void)t;
#endif
  vecCopyRangeBytes(src, dst, nbytes);
}

// allGather equivalent. perRankBytes is one rank's contribution; enableOffset
// selects whether a peer's shard sits at its own rank offset in the peer buffer
// (the tree all-reduce's gather phase) or at the base (the plain all-gather).
template <int NRANKS_CT, CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmAllGather(uint8_t* const* __restrict__ peers, uint8_t* __restrict__ dst, int selfRank,
                                    int nRanksRuntime, size_t perRankBytes, bool enableOffset, uint8_t* window0,
                                    uint8_t* window1, const TdmWarpTile& t) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  // Stagger the starting peer by rank so the ranks do not all pull from the
  // same peer at once, matching the CollCommon all-gather.
  for (int r = 0; r < nRanks; ++r) {
    const int srcRank = (selfRank + r) % nRanks;
    const size_t dstOff = (size_t)srcRank * perRankBytes;
    const size_t srcOff = enableOffset ? dstOff : 0;
    tdmCopyRange<cp>(peers[srcRank] + srcOff, dst + dstOff, perRankBytes, window0, window1, t);
  }
}

// alltoall's peer loop: this rank's shard is pulled out of every peer and laid
// down at that peer's slot in the destination.
template <int NRANKS_CT, CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmAllToAll(uint8_t* const* __restrict__ peers, uint8_t* __restrict__ dst, int selfRank,
                                   int nRanksRuntime, size_t perRankBytes, uint8_t* window0, uint8_t* window1,
                                   const TdmWarpTile& t) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  const size_t srcOff = (size_t)selfRank * perRankBytes;
  for (int r = 0; r < nRanks; ++r) {
    const int srcRank = (selfRank + r) % nRanks;
    tdmCopyRange<cp>(peers[srcRank] + srcOff, dst + (size_t)srcRank * perRankBytes, perRankBytes, window0, window1, t);
  }
}

// Whether every buffer the reduction touches can be staged. Checked once per
// kernel rather than per tile; the peer table is tiny and this keeps the hot
// loop branch-free.
template <int NRANKS_CT>
__device__ inline bool tdmReduceAligned(uint8_t* const* __restrict__ peers, const uint8_t* dst, const uint8_t* acc,
                                        int nRanksRuntime, size_t srcOff, size_t dstOff) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  uintptr_t bits = (uintptr_t)(dst + dstOff) | (uintptr_t)srcOff;
  for (int r = 0; r < nRanks; ++r) {
    bits |= (uintptr_t)(peers[r] + srcOff);
  }
  if (acc != nullptr) {
    bits |= (uintptr_t)(acc + srcOff);
  }
  return (bits & (kTdmAlign - 1)) == 0;
}

// reduceScatter equivalent, expressed as byte offsets so it covers all three
// CollCommon patterns:
//   all-reduce flat  : srcOff = 0,                dstOff = 0
//   all-reduce tree  : srcOff = selfRank * n,     dstOff = selfRank * n (into own scratch)
//   reduce-scatter   : srcOff = selfRank * n,     dstOff = 0
//
// Each issuing warp owns a tile and folds all peers for it before moving on, so
// the accumulator stays in registers and the destination is written once. Peer
// staging is double-buffered: peer r+1's fabric read is issued before peer r is
// folded out of the other window.
template <typename T, int NRANKS_CT, bool hasAcc, CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmReduceRangeBytes(uint8_t* const* __restrict__ peers, uint8_t* __restrict__ dst,
                                           const uint8_t* __restrict__ acc, int nRanksRuntime, size_t srcOff,
                                           size_t dstOff, size_t nbytes, uint8_t* window0, uint8_t* window1,
                                           const TdmWarpTile& t) {
#if TDM_SUPPORTED
  if (!t.issuer) {
    return;
  }
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t stride = (size_t)t.nGWarps * kTdmWindowBytes;

  for (size_t off = (size_t)t.gWarp * kTdmWindowBytes; off < nbytes; off += stride) {
    const size_t rem = nbytes - off;
    const uint32_t n = rem < kTdmWindowBytes ? (uint32_t)rem : kTdmWindowBytes;
    const uint32_t nVec = n / sizeof(uint4);

    uint4 sum[kTdmVecPerLane];
#pragma unroll
    for (uint32_t v = 0; v < kTdmVecPerLane; ++v) {
      sum[v] = uint4{0, 0, 0, 0};
    }
    if constexpr (hasAcc) {
      const uint4* a = reinterpret_cast<const uint4*>(acc + srcOff + off);
#pragma unroll
      for (uint32_t v = 0; v < kTdmVecPerLane; ++v) {
        const uint32_t i = t.lane + v * kTdmLanes;
        if (i < nVec) {
          sum[v] = a[i];
        }
      }
    }

    tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(peers[0] + srcOff + off, window0, n);
    tdm::tdmWait();

#pragma unroll kUnroll
    for (int r = 0; r < nRanks; ++r) {
      uint8_t* cur = (r & 1) ? window1 : window0;
      if (r + 1 < nRanks) {
        // The window about to be refilled was folded on the previous iteration.
        tdmWaitLdsReads();
        uint8_t* nxt = (r & 1) ? window0 : window1;
        tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(peers[r + 1] + srcOff + off, nxt, n);
      }
      const uint4* s = reinterpret_cast<const uint4*>(cur);
#pragma unroll
      for (uint32_t v = 0; v < kTdmVecPerLane; ++v) {
        const uint32_t i = t.lane + v * kTdmLanes;
        if (i < nVec) {
          sum[v] = vecElementAdd<T>(sum[v], s[i]);
        }
      }
      if (r + 1 < nRanks) {
        tdm::tdmWait();
      }
    }

    uint4* d = reinterpret_cast<uint4*>(dst + dstOff + off);
#pragma unroll
    for (uint32_t v = 0; v < kTdmVecPerLane; ++v) {
      const uint32_t i = t.lane + v * kTdmLanes;
      if (i < nVec) {
        d[i] = sum[v];
      }
    }
  }
#else
  (void)peers, (void)dst, (void)acc, (void)nRanksRuntime, (void)srcOff, (void)dstOff, (void)nbytes;
  (void)window0, (void)window1, (void)t;
#endif
}

template <typename T, int NRANKS_CT, bool hasAcc, CachePolicy cp = kTdmCachePolicy>
__device__ inline void tdmReduceRange(uint8_t* const* __restrict__ peers, uint8_t* __restrict__ dst,
                                      const uint8_t* __restrict__ acc, int nRanksRuntime, size_t srcOff, size_t dstOff,
                                      size_t nbytes, uint8_t* window0, uint8_t* window1, const TdmWarpTile& t) {
#if TDM_SUPPORTED
  if (tdmReduceAligned<NRANKS_CT>(peers, dst, acc, nRanksRuntime, srcOff, dstOff)) {
    tdmReduceRangeBytes<T, NRANKS_CT, hasAcc, cp>(peers, dst, acc, nRanksRuntime, srcOff, dstOff, nbytes, window0,
                                                  window1, t);
    return;
  }
#else
  (void)window0, (void)window1, (void)t;
#endif
  vecReduceRangeBytes<T, NRANKS_CT, hasAcc>(peers, dst, acc, nRanksRuntime, srcOff, dstOff, nbytes);
}

// ---------------------------------------------------------------------------
// Host-side launch sizing
// ---------------------------------------------------------------------------

// Grid for a TDM kernel covering nbytes of payload. Derived from the byte count
// alone so every rank in the clique computes the same grid, which
// FabricGpuBarrier requires.
inline std::pair<dim3, dim3> getTdmGridAndBlockDims(size_t nbytes, size_t maxBlocks) {
  const size_t tiles = (nbytes + kTdmWindowBytes - 1) / kTdmWindowBytes;
  size_t blocks = (tiles + kTdmIssueWarps - 1) / kTdmIssueWarps;
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > maxBlocks) {
    blocks = maxBlocks;
  }
  return std::make_pair(dim3((uint32_t)blocks, 1, 1), dim3(kTdmThreadsPerBlock, 1, 1));
}

// Whether the TDM Simple kernels can be built and run here. Host side this is a
// device-architecture query; device side it folds to a compile-time constant so
// the TDM bodies vanish from non-gfx1250 passes of the fat binary.
__host__ __device__ inline bool tdmSimpleSupported(int deviceId = 0) {
  return tdm::IsTdmCopySupported(deviceId);
}

} // namespace meta::comms
