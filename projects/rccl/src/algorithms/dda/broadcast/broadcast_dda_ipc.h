/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-phase DDA Broadcast over the IPC scratch (on the DDA all-gather body), and its pipelined form.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_
#define RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_

#include <cstddef>
#include <cstdint>

#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"

namespace dda::common {

// Two-phase Broadcast over IPC scratch: a scatter of 1/NRANKS slices, then the DDA all-gather body (allGather).
// Slice k is bytes [k * sliceCount, (k + 1) * sliceCount) and sits at that offset in every scratch.
//   1. The root stages every slice in its own scratch.
//   2. Each non-root pulls its own slice from the root's scratch into its own scratch.
//   3. Each non-root gathers every slice from its owner (the root owns its own) into recvbuff.
// As in ddaAllGatherIpc (all_gather_dda.h) every write is local and every peer access a read, and block b only touches
// the indices block b wrote on its peers, so the same-block-index barrier orders each phase. kSendAligned and
// kRecvAligned are per rank: whether this rank's sendbuff (read on the root only) and recvbuff are 16-byte aligned;
// scratch always is.
template <typename T, int NRANKS, bool kSendAligned, bool kRecvAligned>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaBroadcastIpc2Phase(T* const* __restrict__ ipcbuffs, T* __restrict__ recvbuff, size_t sliceCount,
                                        const T* __restrict__ sendbuff, int selfRank, int root,
                                        const IpcGpuBarrier barrier) {
  bool barrierAborted = false; // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx

  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t idxStart = gtIdx * countPerThread;
  const size_t idxEnd = sliceCount;
  const size_t idxStride = gridDim.x * blockDim.x * countPerThread;
  const bool isRoot = selfRank == root;

  if (isRoot) {
#pragma unroll
    for (int k = 0; k < NRANKS; ++k) {
      const size_t off = static_cast<size_t>(k) * sliceCount;
      copyFromSrcToDest<T, kSendAligned, true>(sendbuff + off, ipcbuffs[selfRank] + off, idxStart, idxEnd, idxStride);
    }
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  if (!isRoot) {
    const size_t off = static_cast<size_t>(selfRank) * sliceCount;
    copyFromSrcToDest<T>(ipcbuffs[root] + off, ipcbuffs[selfRank] + off, idxStart, idxEnd, idxStride);
  } else if (sendbuff != recvbuff) {
    // The root's own copy waits for phase 2, where it is otherwise idle, so it stays off the scatter's critical path.
    const size_t total = static_cast<size_t>(NRANKS) * sliceCount;
    copyFromSrcToDest<T, kSendAligned, kRecvAligned>(sendbuff, recvbuff, idxStart, total, idxStride);
  }

  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>(barrierAborted);

  if (!isRoot) {
    allGather<T, NRANKS, kRecvAligned>(ipcbuffs, recvbuff, selfRank, NRANKS, idxStart, idxEnd, idxStride, true);
  }

  // Peers may still read this rank's scratch until they pass this barrier.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>(barrierAborted);
}

// Pipelined two-phase Broadcast. Step t: the root stages chunk t; non-root q pulls sub-slice q of chunk t - 1
// and gathers chunk t - 2.
template <int NRANKS, bool kSendAligned, bool kRecvAligned>
#if defined(USE_ROCM)
__launch_bounds__(kDdaBcIpcPipeThreads)
#endif
  __global__ void ddaBroadcastIpcPipe(int8_t* const* __restrict__ ipcbuffs, int8_t* __restrict__ recvbuff, size_t bytes,
                                      size_t subBytes, const int8_t* __restrict__ sendbuff, int selfRank, int root,
                                      const IpcGpuBarrier barrier) {
  constexpr int kOwners = NRANKS - 1;
  bool barrierAborted = false;  // sticky abort state, see IpcGpuBarrier::syncOnSameBlockIdx

  const size_t idxStart = (static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x) * sizeof(uint4);
  const size_t idxStride = static_cast<size_t>(gridDim.x) * blockDim.x * sizeof(uint4);
  const size_t chunkBytes = kOwners * subBytes;
  const int nChunks = static_cast<int>((bytes + chunkBytes - 1) / chunkBytes);
  const int nSteps = nChunks + 2;
  const bool isRoot = selfRank == root;
  const bool inPlace = sendbuff == recvbuff;
  const int selfOwner = (selfRank - root - 1 + NRANKS) % NRANKS;
  int8_t* const selfScratch = ipcbuffs[selfRank];
  // Bytes of the sub-slice at `start`, 0 past the end; whole 16 B units, as bytes is whole 128 B granules.
  auto subLen = [&](size_t start) -> size_t {
    if (start >= bytes) {
      return 0;
    }
    return bytes - start < subBytes ? bytes - start : subBytes;
  };
  // Two mailbox slots per block, so step t + 1 can signal before every peer has consumed step t's signal.
  auto slot = [&](int t) { return static_cast<int>(blockIdx.x) + (t & 1) * static_cast<int>(gridDim.x); };

  for (int t = 0; t < nSteps; ++t) {
    // Block b moves the same offsets on every rank, so it reads only what block b wrote one step earlier.
    if (t > 0) {
      barrier.waitOnSlot(slot(t - 1), barrierAborted);
    }
    if (isRoot) {
      if (t < nChunks) {
        const size_t chunkBase = static_cast<size_t>(t) * chunkBytes;
        for (size_t idx = idxStart; idx < subBytes; idx += idxStride) {
          uint4 v[kOwners]{};
#pragma unroll
          for (int q = 0; q < kOwners; ++q) {
            const size_t start = chunkBase + q * subBytes;
            if (idx < subLen(start)) {
              v[q] = ddaLoad16B<kSendAligned>(sendbuff + start + idx);
            }
          }
#pragma unroll
          for (int q = 0; q < kOwners; ++q) {
            const size_t start = chunkBase + q * subBytes;
            if (idx < subLen(start)) {
              ddaStore16B<true>(selfScratch + start + idx, v[q]);
              if (!inPlace) {
                ddaStore16B<kRecvAligned>(recvbuff + start + idx, v[q]);
              }
            }
          }
        }
      }
    } else {
      const int scatterChunk = t - 1;
      const int gatherChunk = t - 2;
      const size_t ownStart =
        static_cast<size_t>(scatterChunk) * chunkBytes + static_cast<size_t>(selfOwner) * subBytes;
      const size_t ownLen = (scatterChunk >= 0 && scatterChunk < nChunks) ? subLen(ownStart) : 0;
      const size_t gatherBase = static_cast<size_t>(gatherChunk) * chunkBytes;
      const bool gather = gatherChunk >= 0;
      // The root load and the peer loads go out together, so the scatter and gather links carry bytes at once.
      for (size_t idx = idxStart; idx < subBytes; idx += idxStride) {
        uint4 own{};
        uint4 v[kOwners]{};
        if (idx < ownLen) {
          own = ddaLoad16B<true>(ipcbuffs[root] + ownStart + idx);
        }
        if (gather) {
#pragma unroll
          for (int q = 0; q < kOwners; ++q) {
            const size_t start = gatherBase + q * subBytes;
            if (q != selfOwner && idx < subLen(start)) {
              v[q] = ddaLoad16B<true>(ipcbuffs[(root + 1 + q) % NRANKS] + start + idx);
            }
          }
        }
        if (idx < ownLen) {
          ddaStore16B<true>(selfScratch + ownStart + idx, own);
          ddaStore16B<kRecvAligned>(recvbuff + ownStart + idx, own);
        }
        if (gather) {
#pragma unroll
          for (int q = 0; q < kOwners; ++q) {
            const size_t start = gatherBase + q * subBytes;
            if (q != selfOwner && idx < subLen(start)) {
              ddaStore16B<kRecvAligned>(recvbuff + start + idx, v[q]);
            }
          }
        }
      }
    }
    barrier.signalOnSlot(slot(t), barrierAborted);
  }

  // Peers signal after their last reads: once all blocks pass this wait, the next launch may reuse scratch and slots.
  barrier.waitOnSlot(slot(nSteps - 1), barrierAborted);
}

}  // namespace dda::common

#endif  // RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_
