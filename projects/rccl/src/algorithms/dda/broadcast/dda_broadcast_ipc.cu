/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the two-phase and pipelined DDA Broadcast over IPC.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_ipc.h"
#include "algorithms/dda/dda_init_detail.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaAllGatherIpcGeom;
using nccl_dda_detail::kDdaNranks;

static_assert(kDdaBcIpcGranuleBytes == static_cast<size_t>(kDdaNranks) * sizeof(uint4), "one 16 B unit per rank");

constexpr size_t kDdaBcIpcPipeSweepBytes = kDdaBcIpcPipeThreads * sizeof(uint4);
// Sub-slice: 1/8 of an owner's share per step, long enough to outlast a step's barrier, short enough to pipeline.
constexpr size_t kDdaBcIpcPipeStepsPerShare = 8;
constexpr size_t kDdaBcIpcPipeMinSubBytes = 384 * 1024;
constexpr size_t kDdaBcIpcPipeMaxSubBytes = 1280 * 1024;
constexpr int kDdaBcIpcPipeOwners = kDdaNranks - 1;

bool ddaBcUsesPipe(const ncclComm* comm, size_t bytes) {
  const size_t minBytes = rcclDdaBcIpcPipeMinBytes(comm);
  return minBytes != 0 && bytes >= minBytes;
}

// Rounded up to whole block sweeps.
size_t ddaBcIpcPipeSubBytes(size_t bytes) {
  const size_t perOwner = (bytes + kDdaBcIpcPipeOwners - 1) / kDdaBcIpcPipeOwners;
  const size_t sub = std::clamp<size_t>((perOwner + kDdaBcIpcPipeStepsPerShare - 1) / kDdaBcIpcPipeStepsPerShare,
                                        kDdaBcIpcPipeMinSubBytes, kDdaBcIpcPipeMaxSubBytes);
  return (sub + kDdaBcIpcPipeSweepBytes - 1) / kDdaBcIpcPipeSweepBytes * kDdaBcIpcPipeSweepBytes;
}

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
  const size_t bytes = count * ncclTypeSize(datatype);
  if (ddaBcUsesPipe(comm, bytes)) {
    return kDdaBcIpcPipeBlocks;
  }
  const auto grid = ddaAllGatherIpcGeom(bytes / kDdaNranks).first;
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
  const bool sendAligned = comm->rank != root || reinterpret_cast<uintptr_t>(sendbuff) % sizeof(uint4) == 0;
  const bool recvAligned = reinterpret_cast<uintptr_t>(recvbuff) % sizeof(uint4) == 0;
  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  T** ipcbuffs = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  const size_t bytes = count * ncclTypeSize(datatype);
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (ddaBcUsesPipe(comm, bytes)) {
    using PipeKernel = decltype(&dda::common::ddaBroadcastIpcPipe<kDdaNranks, true, true>);
    // [sendbuff aligned][recvbuff aligned], as kKernels.
    static constexpr PipeKernel kPipeKernels[2][2] = {
      {dda::common::ddaBroadcastIpcPipe<kDdaNranks, false, false>,
       dda::common::ddaBroadcastIpcPipe<kDdaNranks, false, true>},
      {dda::common::ddaBroadcastIpcPipe<kDdaNranks, true, false>,
       dda::common::ddaBroadcastIpcPipe<kDdaNranks, true, true>}};
    const size_t subBytes = ddaBcIpcPipeSubBytes(bytes);
    INFO(NCCL_COLL,
         "DDA Broadcast IPC pipelined: nRanks=%d root=%d bytes=%zu grid=%u block=%u subBytes=%zu sendAligned=%d "
         "recvAligned=%d",
         comm->nRanks, root, bytes, kDdaBcIpcPipeBlocks, kDdaBcIpcPipeThreads, subBytes, static_cast<int>(sendAligned),
         static_cast<int>(recvAligned));
    hipExtLaunchKernelGGL(kPipeKernels[sendAligned][recvAligned], dim3(kDdaBcIpcPipeBlocks),
                          dim3(kDdaBcIpcPipeThreads), 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
                          ipcbuffs, static_cast<T*>(recvbuff), bytes, subBytes, static_cast<const T*>(sendbuff),
                          comm->rank, root, barrierState->barrierHost);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  const size_t sliceBytes = bytes / kDdaNranks;
  const auto gridBlock = ddaAllGatherIpcGeom(sliceBytes);

  INFO(NCCL_COLL,
       "DDA Broadcast IPC 2-phase: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u sendAligned=%d recvAligned=%d",
       comm->nRanks, root, bytes, gridBlock.first.x, gridBlock.first.y, gridBlock.second.x,
       static_cast<int>(sendAligned), static_cast<int>(recvAligned));

  hipExtLaunchKernelGGL(kKernels[sendAligned][recvAligned], gridBlock.first, gridBlock.second, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, ipcbuffs, static_cast<T*>(recvbuff),
                        sliceBytes, static_cast<const T*>(sendbuff), comm->rank, root, barrierState->barrierHost);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
