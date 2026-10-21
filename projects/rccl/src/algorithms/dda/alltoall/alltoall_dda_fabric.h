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

// Copies 16 B lines i = begin, begin + stride, ... < end of src to dst. Full groups of kDdaA2APeerUnroll lines
// carry no per-line bounds check, so all their loads are issued before the first store; a predicated load per
// line stops the compiler from batching them and costs ~25% of the bandwidth. Leftover lines go one at a time.
template <typename T>
__device__ __forceinline__ void ddaA2APeerCopy(T* dst, const T* src, size_t begin, size_t end, size_t stride) {
  constexpr int kUnroll = kDdaA2APeerUnroll;
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);
  size_t i = begin;
  for (; i + (kUnroll - 1) * stride < end; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = s[i + u * stride];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) d[i + u * stride] = v[u];
  }
  for (; i < end; i += stride) d[i] = s[i];
}

// Peer-major push alltoall. The grid is nRanks * blocksPerPeer blocks; block b works only on peer
// k = b / blocksPerPeer (rank (selfRank + k) % nRanks) and on slice b % blocksPerPeer of that peer's chunk, so
// each block streams to one remote GPU instead of fanning out to all of them. The drain mirrors it: block b
// reads the slot of rank (selfRank - k) % nRanks. Rank s' wrote that slot from its block with the same k and
// slice, i.e. the same block index, so the sync pairs the write with the read. Every rank must launch the same
// grid. Out of place only: a block drains into recvbuff slots that other blocks of this rank may still be
// reading from sendbuff.
// The mid-kernel sync is a two-rank handshake (push destination, drain source) rather than an all-rank barrier,
// since a block only depends on those two ranks. Block group k == 0 pushes to this rank's own scratch like any
// other peer, so every block takes part in the handshake and the flag slots stay paired.
// pieces >= 1 (the host never passes 0): the block's share of the chunk is cut into `pieces` runs of rows (a row is the block's stride loop
// index, rounded to whole unroll groups). Step p pushes piece p, waits for the source's piece p - 1 and drains it
// while piece p's remote stores are still in flight, then waits for its own stores and posts piece p. The flag slot
// holds one token, so a post first waits for the destination to consume the previous piece, and every post is matched
// by exactly one wait. The pieces land at their place in the chunk, so the scratch needed is unchanged. One piece
// pushes the whole share, posts, waits and drains: no overlap.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllToAllFabricPushPeer(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t count,
                                            const T* __restrict__ sendbuff, int selfRank, int nRanks,
                                            int blocksPerPeer, unsigned pieces, FabricGpuBarrier barrier) {
  // A peer's scratch is free to overwrite once that peer's block has reached this op.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  const size_t nLines = count / countPerThread;
  const int k = blockIdx.x / blocksPerPeer;
  const size_t begin = (blockIdx.x % blocksPerPeer) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(blocksPerPeer) * blockDim.x;

  const int peer = (selfRank + k) % nRanks;
  const int srcRank = (selfRank - k + nRanks) % nRanks;

  T* const dst = ipcbuffs[peer] + static_cast<size_t>(selfRank) * count;
  const T* const src = sendbuff + static_cast<size_t>(peer) * count;
  const size_t slot = static_cast<size_t>(srcRank) * count;

  {
    // The host never passes pieces == 0, so the kernel does not guard it. Adding a check inside the kernel (for example
    // `pieces ? pieces : 1` in the divisions below) lowers the kernel from 80 to 64 VGPRs; neither version spills and
    // the speed is the same, so the simpler form is kept.
    // Every thread has at most nRows rows; pieces are whole unroll groups of rows. All ranks derive the same piece
    // count, since it depends only on count, blocksPerPeer and blockDim.
    const size_t nRows = (nLines + stride - 1) / stride;
    const size_t perPiece = ((nRows + pieces - 1) / pieces + kDdaA2APeerUnroll - 1) / kDdaA2APeerUnroll * kDdaA2APeerUnroll;
    const size_t nPieces = (nRows + perPiece - 1) / perPiece;
    // Copies this thread's rows [p * perPiece, (p + 1) * perPiece) of a chunk.
    auto piece = [&](T* d, const T* s, size_t p) {
      const size_t lo = begin + p * perPiece * stride;
      const size_t hi = begin + (p + 1) * perPiece * stride;
      ddaA2APeerCopy(d, s, lo, hi < nLines ? hi : nLines, stride);
    };
    for (size_t p = 0; p <= nPieces; ++p) {
      if (p < nPieces) piece(dst, src, p);
      if (p > 0) {
        barrier.waitPeer(srcRank);
        piece(recvbuff + slot, ipcbuffs[selfRank] + slot, p - 1);
      }
      if (p < nPieces) barrier.postPeer(peer, blockIdx.x);
    }
  }
}

} // namespace dda::common
