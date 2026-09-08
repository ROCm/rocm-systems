/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_gather/all_gather_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/dda_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllGatherIpcGeom(size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, ddaMaxNBlocksForScratch());
}

template <typename T>
static ncclResult_t ncclAllGatherDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                             cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = sendcount * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC allgather: send element count %zu needs %zu bytes; comm scratch is %zu bytes", sendcount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // sendcount is already the byte count (kernel instantiated for int8_t).
  auto gridBlock = ddaAllGatherIpcGeom(sendcount);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const int nRanks = comm->nRanks;

  INFO(NCCL_COLL, "DDA IPC AllGather: launching kernel: nRanks=%d sendcount=%zu grid=%u block=%u%s", nRanks, sendcount,
       grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  // Specialize the kernel for common clique sizes (compile-time NRANKS_CT -> the
  // unrolled CollCommon allGather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllGatherIpc<T, 4, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), sendcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  case 8:
    dda::common::ddaAllGatherIpc<T, 8, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), sendcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  default:
    dda::common::ddaAllGatherIpc<T, 0, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), sendcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  size_t need = sendcount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check for data size divisible by 16
  if ((sendcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaIpcBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  (void)comm;
  const auto grid = ddaAllGatherIpcGeom(sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaIpc(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaIpcTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
