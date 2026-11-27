/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the two-phase DDA Broadcast over IPC.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_ipc.h"
#include "algorithms/dda/dda_init_detail.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaAllGatherIpcGeom;
using nccl_dda_detail::kDdaNranks;

static_assert(kDdaBcIpcGranuleBytes == static_cast<size_t>(kDdaNranks) * sizeof(uint4), "one 16 B unit per rank");

}  // namespace

// ddaBroadcastIpc2Phase runs the IPC all-gather grid at one slice (bytes / nRanks) per rank. Every input is
// rank-uniform; per-rank buffer alignment only picks the kernel variant.
bool ncclBroadcastDdaIpcEligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1 || comm->nRanks != kDdaNranks || count == 0) {
    return false;
  }
  const size_t bytes = count * ncclTypeSize(datatype);
  return bytes % kDdaBcIpcGranuleBytes == 0 && bytes <= comm->ddaScratchBytes;
}

uint32_t ncclBroadcastDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  const auto grid = ddaAllGatherIpcGeom(count * ncclTypeSize(datatype) / kDdaNranks).first;
  return grid.x * grid.y;
}

ncclResult_t ncclBroadcastDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
                                 ncclComm* comm, cudaStream_t stream) {
  if (!ncclBroadcastDdaIpcEligible(comm, count, datatype) || root < 0 || root >= comm->nRanks) {
    return ncclInvalidUsage;
  }
  using T = int8_t;
  using Kernel = decltype(&dda::common::ddaBroadcastIpc2Phase<T, kDdaNranks, true, true>);
  // [sendbuff aligned][recvbuff aligned]; a non-root never reads sendbuff.
  static constexpr Kernel kKernels[2][2] = {
    {dda::common::ddaBroadcastIpc2Phase<T, kDdaNranks, false, false>,
     dda::common::ddaBroadcastIpc2Phase<T, kDdaNranks, false, true>},
    {dda::common::ddaBroadcastIpc2Phase<T, kDdaNranks, true, false>,
     dda::common::ddaBroadcastIpc2Phase<T, kDdaNranks, true, true>}};
  const size_t sliceBytes = count * ncclTypeSize(datatype) / kDdaNranks;
  const auto gridBlock = ddaAllGatherIpcGeom(sliceBytes);
  const bool sendAligned = comm->rank != root || reinterpret_cast<uintptr_t>(sendbuff) % sizeof(uint4) == 0;
  const bool recvAligned = reinterpret_cast<uintptr_t>(recvbuff) % sizeof(uint4) == 0;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  T** ipcbuffs = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);

  INFO(NCCL_COLL,
       "DDA Broadcast IPC 2-phase: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u sendAligned=%d recvAligned=%d",
       comm->nRanks, root, sliceBytes * kDdaNranks, gridBlock.first.x, gridBlock.first.y, gridBlock.second.x,
       static_cast<int>(sendAligned), static_cast<int>(recvAligned));

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL(kKernels[sendAligned][recvAligned], gridBlock.first, gridBlock.second, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, ipcbuffs, static_cast<T*>(recvbuff),
                        sliceBytes, static_cast<const T*>(sendbuff), comm->rank, root, barrierState->barrierHost);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
