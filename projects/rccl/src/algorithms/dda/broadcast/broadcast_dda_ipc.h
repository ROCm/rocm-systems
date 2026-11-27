/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Two-phase DDA Broadcast over the IPC scratch, built on the DDA all-gather body.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_
#define RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_

#include <cstddef>

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

}  // namespace dda::common

#endif  // RCCL_ALGORITHMS_DDA_BROADCAST_BROADCAST_DDA_IPC_H_
