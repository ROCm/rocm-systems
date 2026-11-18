/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TDM_REDUCE_H_
#define RCCL_TDM_REDUCE_H_

// The TDM reduction path of reduceCopy for the SIMPLE generic kernels. common_kernel.h includes it after BytePack, the
// reduction functors and kRcclTdmPolicy are defined, and before reduceCopy, which calls it.

#if TDM_SUPPORTED && ENABLE_TDM_SIMPLE
// The warp's staging window holds two tiles: tile 0 accumulates, tile 1 receives each further source in turn.
constexpr int kRcclTdmReduceTileBytes = RCCL_TDM_STAGE_BYTES_PER_WARP / 2;
constexpr int kRcclTdmReduceTilePacks = kRcclTdmReduceTileBytes / (WARP_SIZE * sizeof(BytePack<16>));
static_assert(kRcclTdmReduceTileBytes > 0 && kRcclTdmReduceTileBytes % RCCL_TDM_ALIGN == 0,
              "A reduce tile must be a non-empty multiple of the TDM direct-path alignment.");
static_assert(kRcclTdmReduceTilePacks * WARP_SIZE * sizeof(BytePack<16>) == kRcclTdmReduceTileBytes,
              "A reduce tile must split evenly into 16-byte packs across the warp.");

static __device__ __forceinline__ int laneId() {
  return (int)__lane_id();
}

// The mover's LDS traffic is invisible to the compiler, so no lane LDS access may move across an issue or a wait.
__device__ __forceinline__ void tdmReduceLoadTile(BytePack<16>* tile, uintptr_t global,
                                                  uint32_t bytes = kRcclTdmReduceTileBytes) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  tdm::asyncLoadToLDS<SyncPolicy::Async, kRcclTdmPolicy, /*Aligned=*/true>(
    reinterpret_cast<const uint8_t*>(global), reinterpret_cast<uint8_t*>(tile), bytes);
}

__device__ __forceinline__ void tdmReduceStoreTile(uintptr_t global, const BytePack<16>* tile,
                                                   uint32_t bytes = kRcclTdmReduceTileBytes) {
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
  tdm::asyncStoreFromLDS<SyncPolicy::Async, kRcclTdmPolicy, /*Aligned=*/true>(
    reinterpret_cast<const uint8_t*>(tile), reinterpret_cast<uint8_t*>(global), bytes);
}

__device__ __forceinline__ void tdmReduceWait() {
  tdm::tdmWait();
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// acc = (First ? preOp(acc) : acc) op in over this lane's packs, then postOp when asked; a null `in` leaves a single
// source. `in` holds source inSrc. As in reduceCopyPacks, the pre-op applies to every source below PreOpSrcs.
template <typename RedFn, int PreOpSrcs, bool First>
__device__ __forceinline__ void tdmReduceCombine(RedFn redFn, BytePack<16>* acc, const BytePack<16>* in, int inSrc,
                                                 bool doPostOp) {
  const int lane = laneId();
#pragma unroll 4
  for (int lanePack = 0; lanePack < kRcclTdmReduceTilePacks; lanePack++) {
    const int tilePack = lane + WARP_SIZE * lanePack;
    BytePack<16> x = acc[tilePack];
    if (First && 0 < PreOpSrcs) x = applyPreOp(redFn, x);
    if (in != nullptr) {
      BytePack<16> y = in[tilePack];
      if (inSrc < PreOpSrcs) y = applyPreOp(redFn, y);
      x = applyReduce(redFn, x, y);
    }
    if (doPostOp) x = applyPostOp(redFn, x);
    acc[tilePack] = x;
  }
}

// A copy-shaped slice: one source and nothing to apply to it.
template <typename RedFn, int PreOpSrcs>
__device__ __forceinline__ bool tdmReduceIsCopy(int nSrcs, bool postOp) {
  constexpr bool PreOpIsIdentity = PreOpSrcs == 0 || Apply_PreOp<RedFn, 1>::IsIdentity;
  constexpr bool PostOpIsIdentity = Apply_PostOp<RedFn, 1>::IsIdentity;
  return nSrcs == 1 && PreOpIsIdentity && (PostOpIsIdentity || !postOp);
}

// The whole of a copy-shaped slice through the TDM mover, ping-ponging the warp's two tiles: each warp loads its next
// tile while the current one drains to every destination. Tiles are sized so every warp gets the same number, at most
// one reduce tile each; only the last tile of the slice is shorter. Consumes the slice: nBytesAhead becomes 0.
template <int MaxDsts, typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void copyTdm(int nThreads, int thread, SrcPtrFn const& srcPtrFn, int nDsts,
                                        DstPtrFn const& dstPtrFn, IntBytes& nBytesBehind, IntBytes& nBytesAhead) {
  const IntBytes nWarps = nThreads / WARP_SIZE;
  const IntBytes rounds = (nBytesAhead + nWarps * kRcclTdmReduceTileBytes - 1) / (nWarps * kRcclTdmReduceTileBytes);
  IntBytes tileBytes = (nBytesAhead + rounds * nWarps - 1) / (rounds * nWarps);
  tileBytes = (tileBytes + RCCL_TDM_ALIGN - 1) / RCCL_TDM_ALIGN * RCCL_TDM_ALIGN;
  const IntBytes end = nBytesBehind + nBytesAhead;
  IntBytes cursor = nBytesBehind + IntBytes(thread / WARP_SIZE) * tileBytes;
  nBytesBehind = end;
  nBytesAhead = 0;
  if (end <= cursor) return;

  BytePack<16>* tile = reinterpret_cast<BytePack<16>*>(ncclTdmStageForWarp(threadIdx.x / WARP_SIZE));
  const uintptr_t src = cvta_to_global(srcPtrFn(0));
  int slot = 0;
  IntBytes prev = cursor;
  uint32_t prevBytes = uint32_t(end - cursor < tileBytes ? end - cursor : tileBytes);
  tdmReduceLoadTile(tile, src + cursor, prevBytes);
  for (;;) {
    cursor += nWarps * tileBytes;
    const uint32_t bytes = cursor < end ? uint32_t(end - cursor < tileBytes ? end - cursor : tileBytes) : 0;
    // Drains the load into `slot` and the stores out of the other tile, which the next load then refills.
    tdmReduceWait();
    if (bytes) tdmReduceLoadTile(tile + (slot ^ 1) * kRcclTdmReduceTilePacks * WARP_SIZE, src + cursor, bytes);
    for (int d = 0; d < MaxDsts && d < nDsts; d++)
      tdmReduceStoreTile(cvta_to_global(dstPtrFn(d)) + prev, tile + slot * kRcclTdmReduceTilePacks * WARP_SIZE,
                         prevBytes);
    if (!bytes) break;
    slot ^= 1;
    prev = cursor;
    prevBytes = bytes;
  }
  tdmReduceWait();
}

// Whole tiles of a slice through the TDM mover: load, reduce in LDS, store to every destination. A copy-shaped slice
// goes to copyTdm instead, which takes all of it.
// Updates the cursors and rotates `thread` the same way reduceCopyPacks does, so the remainder can be handed to it.
template <typename RedFn, typename T, int MaxSrcs, int MaxDsts, int PreOpSrcs, typename IntBytes, typename SrcPtrFn,
          typename DstPtrFn>
__device__ __forceinline__ void reduceCopyTdm(int nThreads, int& thread, uint64_t redArg, bool postOp, int nSrcs,
                                              SrcPtrFn const& srcPtrFn, int nDsts, DstPtrFn const& dstPtrFn,
                                              IntBytes& nBytesBehind, IntBytes& nBytesAhead) {
  if (tdmReduceIsCopy<RedFn, PreOpSrcs>(nSrcs, postOp)) {
    copyTdm<MaxDsts>(nThreads, thread, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
    return;
  }
  const int nWarps = nThreads / WARP_SIZE;
  const int warp = thread / WARP_SIZE;
  const int lane = laneId();

  IntBytes nTiles = nBytesAhead / kRcclTdmReduceTileBytes;
  IntBytes cursor = nBytesBehind + IntBytes(warp) * kRcclTdmReduceTileBytes;
  nBytesBehind += nTiles * kRcclTdmReduceTileBytes;
  nBytesAhead -= nTiles * kRcclTdmReduceTileBytes;
  IntBytes nTilesAhead = nTiles - warp;

  if (0 < nTilesAhead) {
    RedFn redFn(redArg);
    // Indexed by the hardware warp since several groups may run reduceCopy concurrently in one block.
    BytePack<16>* acc = reinterpret_cast<BytePack<16>*>(ncclTdmStageForWarp(threadIdx.x / WARP_SIZE));
    BytePack<16>* in = acc + kRcclTdmReduceTilePacks * WARP_SIZE;
    const bool twoSrcs = 1 < MaxSrcs && 1 < nSrcs;
    do {
      tdmReduceLoadTile(acc, cvta_to_global(srcPtrFn(0)) + cursor);
      if (twoSrcs) tdmReduceLoadTile(in, cvta_to_global(srcPtrFn(1)) + cursor);
      tdmReduceWait();
      tdmReduceCombine<RedFn, PreOpSrcs, /*First=*/true>(redFn, acc, twoSrcs ? in : nullptr, /*inSrc=*/1,
                                                         postOp && nSrcs <= 2);
      for (int s = 2; s < MaxSrcs && s < nSrcs; s++) {
        tdmReduceLoadTile(in, cvta_to_global(srcPtrFn(s)) + cursor);
        tdmReduceWait();
        tdmReduceCombine<RedFn, PreOpSrcs, /*First=*/false>(redFn, acc, in, s, postOp && s == nSrcs - 1);
      }
      // The lanes' LDS writes must be visible to the mover before it reads the tile back out.
      __threadfence_block();
      for (int d = 0; d < MaxDsts && d < nDsts; d++) tdmReduceStoreTile(cvta_to_global(dstPtrFn(d)) + cursor, acc);
      // Drain before the tile is refilled; postPeer's cheap fence does not wait on TENSORcnt either.
      tdmReduceWait();
      cursor += IntBytes(nWarps) * kRcclTdmReduceTileBytes;
      nTilesAhead -= nWarps;
    } while (0 < nTilesAhead);
  }

  // Rotate warps so the warp which got the least work here will be warp 0.
  thread = int(-nTilesAhead) * WARP_SIZE + lane;
}

// Does this slice qualify for reduceCopyTdm? Pure gating, no side effects.
template <typename RedFn, int PreOpSrcs, typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ bool canUseTdmReduce(int thread, int nThreads, bool postOp, int nSrcs,
                                                SrcPtrFn const& srcPtrFn, int nDsts, DstPtrFn const& dstPtrFn,
                                                IntBytes nBytes) {
  if (nSrcs < 1) return false;
  // Single-destination copies belong to tryTdmCopy. Multi-destination copies come here so each tile is read once.
  // copyTdm takes a copy of any size; a reduction needs at least one whole tile.
  if (tdmReduceIsCopy<RedFn, PreOpSrcs>(nSrcs, postOp)) {
    if (!ncclShmem.comm.tdmSimpleEnable || nDsts < 2 || nBytes <= 0) return false;
  } else if (!ncclShmem.comm.tdmSimpleReduceEnable || nBytes < kRcclTdmReduceTileBytes) {
    return false;
  }

  // The tensor op needs a converged wave and the staging window belongs to the hardware warp.
  const int base = threadIdx.x - thread;
  if (((base | nThreads) & (WARP_SIZE - 1)) != 0) return false;

  // Every global address must reach the mover's direct path. nSrcs/nDsts never exceed WARP_SIZE.
  const int lane = laneId();
  bool ok = true;
  if (lane < nSrcs) {
    uintptr_t src = cvta_to_global(srcPtrFn(lane));
    ok &= src != 0 && (src & (RCCL_TDM_ALIGN - 1)) == 0;
  }
  if (lane < nDsts) {
    uintptr_t dst = cvta_to_global(dstPtrFn(lane));
    ok &= dst != 0 && (dst & (RCCL_TDM_ALIGN - 1)) == 0;
  }
  return !__any(!ok);
}
#endif

#endif // RCCL_TDM_REDUCE_H_
