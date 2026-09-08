/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda.h"
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

template <typename T>
static ncclResult_t ncclReduceScatterDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t recvcount, ncclComm* comm,
                                                 cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = recvcount * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC reduce-scatter: total element count %zu needs %zu bytes; comm scratch is %zu bytes", totalCount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  // For reduce-scatter, we use recvcount for grid calculation since each rank processes its portion
  auto gridBlock = dda::common::getGridAndBlockDims(recvcount, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const int nRanks = comm->nRanks;

  INFO(NCCL_COLL, "DDA IPC ReduceScatter: launching kernel: nRanks=%d recvcount=%zu grid=%u block=%u%s", nRanks,
       recvcount, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));

  // Specialize the kernel for common clique sizes (compile-time NRANKS_CT ->
  // the unrolled CollCommon reduceScatter), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  switch (nRanks) {
  case 4:
    dda::common::ddaReduceScatterIpc<T, 4, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  case 8:
    dda::common::ddaReduceScatterIpc<T, 8, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  default:
    dda::common::ddaReduceScatterIpc<T, 0, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, nRanks,
      barrierHost);
    break;
  }
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  size_t totalCount = recvcount * comm->nRanks;
  size_t need = totalCount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check 16-byte alignment for total data
  if ((totalCount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  // Check per-rank byte alignment
  if ((recvcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaIpc(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                     ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaIpcTyped<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaIpcTyped<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaIpcTyped<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
