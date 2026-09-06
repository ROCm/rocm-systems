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
#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

template <typename T, int NRANKS>
static ncclResult_t ncclReduceScatterDdaIpcLaunch(const void* sendbuff, void* recvbuff, size_t recvcount, ncclComm* comm,
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

  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  dda::common::ddaReduceScatterIpc<T, NRANKS, false><<<grid, block, 0, stream>>>(
    d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff), comm->rank, barrierHost);
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Dispatch to the template instantiation for the active participant count (2..kDdaNranks).
template <typename T>
static ncclResult_t ncclReduceScatterDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t recvcount, ncclComm* comm,
                                                 cudaStream_t stream) {
  switch (comm->nRanks) {
  case 8: return ncclReduceScatterDdaIpcLaunch<T, 8>(sendbuff, recvbuff, recvcount, comm, stream);
  case 7: return ncclReduceScatterDdaIpcLaunch<T, 7>(sendbuff, recvbuff, recvcount, comm, stream);
  case 6: return ncclReduceScatterDdaIpcLaunch<T, 6>(sendbuff, recvbuff, recvcount, comm, stream);
  case 5: return ncclReduceScatterDdaIpcLaunch<T, 5>(sendbuff, recvbuff, recvcount, comm, stream);
  case 4: return ncclReduceScatterDdaIpcLaunch<T, 4>(sendbuff, recvbuff, recvcount, comm, stream);
  case 3: return ncclReduceScatterDdaIpcLaunch<T, 3>(sendbuff, recvbuff, recvcount, comm, stream);
  case 2: return ncclReduceScatterDdaIpcLaunch<T, 2>(sendbuff, recvbuff, recvcount, comm, stream);
  default: WARN("DDA IPC reduce-scatter: unsupported nRanks %d", comm->nRanks); return ncclInvalidUsage;
  }
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
  if (!ncclDdaNranksSupported(comm->nRanks)) {
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
