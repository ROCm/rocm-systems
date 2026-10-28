/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * DDA alltoall kernel for the fabric/VMM path, using FabricGpuBarrier.
 *
 * Templated on a compile-time rank count NRANKS_CT (matching the all-reduce
 * fabric design):
 *   - NRANKS_CT > 0  : specialized for that clique size (unrolled peer loop).
 *   - NRANKS_CT == 0 : runtime fallback driven by the nRanks argument, so a
 *                      single instantiation covers any other clique size up to
 *                      kDdaMaxNranks.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabric(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                    int selfRank, int nRanks, FabricGpuBarrier barrier) {
  // Release-acquire barrier ensures the stream-ordered scratch write is visible to peers.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  // use uint4 to do 16-byte loads to maximize memory efficiency. We assume
  // that count % countPerThread == 0, enforced before kernel launch.
  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  // Fully unroll when the clique size is a compile-time constant; otherwise
  // partially unroll 8-wide for the runtime fallback (Option A).
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      int srcRank = r;
      size_t srcIdx = idx + selfRank * idxEnd;
      size_t destIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[destIdx]) = reinterpret_cast<const uint4*>(&ipcbuffs[srcRank][srcIdx])[0];
    }
  }

  // barrier to ensure remote ranks won't free their buffers until I'm done
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

// Push alltoall: each rank writes its send chunk for peer r into peer r's scratch at this rank's slot,
// then drains its own scratch into recvbuff. The per-block barrier pairs those writes with the drain, so
// every rank must launch the same grid. Scratch for peer r, slot s, holds the chunk rank s sends to r.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPush(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                        const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                        FabricGpuBarrier barrier) {
  // The previous collective's readers must be done with every scratch slot this block is about to overwrite.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  const int nRanksEff = (NRANKS_CT > 0) ? NRANKS_CT : nRanks;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const size_t countPerRank = count;
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  const auto idxStart = gtIdx * countPerThread;
  const auto idxEnd = countPerRank;
  const auto idxStride = gridDim.x * blockDim.x * countPerThread;

  // sendbuff[r] is the chunk destined for rank r. Load a whole peer group into registers before issuing its
  // remote stores, so those loads can be in flight together. Block i on every rank owns the same idx set.
  const size_t destIdx = selfRank * idxEnd;
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    int r0 = 0;
    for (; r0 + kUnroll <= nRanksEff; r0 += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        v[u] = reinterpret_cast<const uint4*>(&sendbuff[idx + (r0 + u) * idxEnd])[0];
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        *reinterpret_cast<uint4*>(&ipcbuffs[r0 + u][destIdx + idx]) = v[u];
      }
    }
    for (int r = r0; r < nRanksEff; ++r) {
      *reinterpret_cast<uint4*>(&ipcbuffs[r][destIdx + idx]) =
        reinterpret_cast<const uint4*>(&sendbuff[idx + r * idxEnd])[0];
    }
  }

  // Publish the remote writes before any rank reads its scratch.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  T* selfScratch = ipcbuffs[selfRank];
  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
#pragma unroll kUnroll
    for (int r = 0; r < nRanksEff; ++r) {
      size_t slotIdx = idx + r * idxEnd;
      *reinterpret_cast<uint4*>(&recvbuff[slotIdx]) = reinterpret_cast<const uint4*>(&selfScratch[slotIdx])[0];
    }
  }
}

// Peer-major copy loops keep this many 16 B loads in flight per thread.
constexpr int kDdaA2APeerUnroll = 8;

// Copies 16 B lines i = begin, begin + stride, ... < end, loading each full group of kDdaA2APeerUnroll lines first.
// SysLoad / SysStore use system-scope loads / stores, so data shared with other GPUs needs no cache flush.
template <bool SysLoad, bool SysStore, typename T>
__device__ __forceinline__ void ddaA2APeerCopy(T* dst, const T* src, size_t begin, size_t end, size_t stride) {
  constexpr int kUnroll = kDdaA2APeerUnroll;
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);

  auto load = [&](size_t line, uint4& v) {
    if constexpr (SysLoad) {
      ddaLLLoadLineB128(reinterpret_cast<const uint32_t*>(s + line), v.x, v.y, v.z, v.w);
    } else {
      v = s[line];
    }
  };
  auto store = [&](size_t line, const uint4& v) {
    if constexpr (SysStore) {
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(d + line), v.x, v.y, v.z, v.w);
    } else {
      d[line] = v;
    }
  };
  size_t i = begin;
  for (; i + (kUnroll - 1) * stride < end; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) load(i + u * stride, v[u]);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) store(i + u * stride, v[u]);
  }
  for (; i < end; i += stride) {
    uint4 v;
    load(i, v);
    store(i, v);
  }
}

// Peer-major push alltoall, out of place: block b pushes its slice of the chunk to peer (self + k) % nRanks, with
// k = b / blocksPerPeer, and drains the slot of rank (self - k) % nRanks (written by that rank's block b).
// The share is cut into `pieces` runs of rows: step p pushes piece p, drains the source's piece p - 1, then posts p.
// The flags are of FabricGpuBarrier: epoch flags by default, CAS flags with RCCL_DDA_A2A_PUSH_PEER_EPOCH=0
// drain > 0: drain * (nRanks - 1) drain blocks follow the nRanks * blocksPerPeer push blocks, drain of them per source
// offset r = 1 .. nRanks - 1; drain block d of offset r drains slices d, d + drain, ... of every piece of rank
// (self - r) % nRanks, in piece order. The push blocks (k != 0) then only push and post.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPushPeer(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                            const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                            int blocksPerPeer, unsigned pieces, unsigned drain,
                                            FabricGpuBarrier barrier) {
  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  const size_t nLines = count / countPerThread;
  const size_t stride = static_cast<size_t>(blocksPerPeer) * blockDim.x;
  const unsigned pushBlocks = static_cast<unsigned>(nRanks * blocksPerPeer);

  // Every thread has at most nRows rows; pieces are whole unroll groups of rows. All ranks derive the same piece
  // count, since it depends only on count, blocksPerPeer and blockDim.
  const size_t nRows = (nLines + stride - 1) / stride;
  const size_t perPiece = ((nRows + pieces - 1) / pieces + kDdaA2APeerUnroll - 1) / kDdaA2APeerUnroll * kDdaA2APeerUnroll;
  const size_t nPieces = (nRows + perPiece - 1) / perPiece;
  // Line range [lo, hi) of piece p for the thread whose first line is `first`: rows [p * perPiece, (p + 1) * perPiece).
  auto pieceOf = [&](size_t first, size_t p) -> std::pair<size_t, size_t> {
    const size_t lo = first + p * perPiece * stride;
    const size_t hi = first + (p + 1) * perPiece * stride;
    return {lo, hi < nLines ? hi : nLines};
  };

  const uint32_t epoch = barrier.epochBegin();
  if (blockIdx.x >= pushBlocks) {
    // Drain block: waits for the push blocks it serves and copies their pieces out of this rank's scratch.
    const unsigned idx = blockIdx.x - pushBlocks;
    const int r = 1 + static_cast<int>(idx / drain);
    const int from = (selfRank - r + nRanks) % nRanks;
    const size_t fromSlot = static_cast<size_t>(from) * count;
    for (size_t p = 0; p < nPieces; ++p) {
      for (unsigned j = idx % drain; j < static_cast<unsigned>(blocksPerPeer); j += drain) {
        barrier.waitPeer(from, epoch, static_cast<unsigned>(p), r * blocksPerPeer + static_cast<int>(j));
        auto [lo, hi] = pieceOf(static_cast<size_t>(j) * blockDim.x + threadIdx.x, p);
        ddaA2APeerCopy<true, false>(recvbuff + fromSlot, ipcbuffs[selfRank] + fromSlot, lo, hi, stride);
      }
    }
    barrier.epochEnd(epoch, pieces);
    return;
  }

  const int k = blockIdx.x / blocksPerPeer;
  const size_t begin = (blockIdx.x % blocksPerPeer) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const int peer = (selfRank + k) % nRanks;
  const int srcRank = (selfRank - k + nRanks) % nRanks;

  T* const dst = ipcbuffs[peer] + static_cast<size_t>(selfRank) * count;
  const T* const src = sendbuff + static_cast<size_t>(peer) * count;
  const size_t slot = static_cast<size_t>(srcRank) * count;

  // Entry handshake instead of a full barrier: only peer reads the slot this block writes, so wait for peer alone.
  if (k != 0) barrier.signalWait(srcRank, peer, epoch);

  {
    auto piece = [&](size_t p) { return pieceOf(begin, p); };
    // With drain blocks the push blocks (k != 0) only push and post. Group 0 pushes its own chunk into its own scratch
    // and still drains it, since no drain block serves the own slot.
    const bool drainHere = drain == 0 || k == 0;
    for (size_t p = 0; p <= nPieces; ++p) {
      if (p < nPieces) {
         auto [lo, hi] = piece(p);
         ddaA2APeerCopy<false, true>(dst, src, lo, hi, stride);
      }
      if (p > 0 && drainHere) {
        barrier.waitPeer(srcRank, epoch, static_cast<unsigned>(p - 1));
        auto [lo, hi] = piece(p - 1);
        ddaA2APeerCopy<true, false>(recvbuff + slot, ipcbuffs[selfRank] + slot, lo, hi, stride);
      }
      if (p < nPieces) barrier.postPeer(peer, epoch, static_cast<unsigned>(p));
    }
  }
  barrier.epochEnd(epoch, pieces);
}
} // namespace dda::common
