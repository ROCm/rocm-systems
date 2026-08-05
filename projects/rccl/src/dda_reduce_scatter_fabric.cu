/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_reduce_scatter.h"

#include "algorithms/CollCommon.h"
#include "algorithms/reduce_scatter/reduce_scatter_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dda_init_detail.h"
#include "fabric_gpu_barrier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                    ncclComm* comm, cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t totalCount = recvcount * nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA fabric reduce-scatter: total element count %zu needs %zu bytes; comm scratch is %zu bytes", totalCount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // Use the block cap chosen at init (barrier flag buffer is sized for it).
  const int nBlocksMax = comm->ddaFabricMaxBlocks;
  // For reduce-scatter we use recvcount for grid calculation since each rank
  // processes its own shard.
  auto gridBlock = meta::comms::getGridAndBlockDims(recvcount, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  meta::comms::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));

  INFO(NCCL_COLL, "DDA fabric ReduceScatter: launching kernel: nRanks=%d recvcount=%zu grid=%u block=%u%s", nRanks,
       recvcount, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  switch (nRanks) {
  case 4:
    meta::comms::ddaReduceScatterFabric<T, 4, false>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff),
                                   comm->rank, nRanks, barrierHost, nullptr);
    break;
  case 8:
    meta::comms::ddaReduceScatterFabric<T, 8, false>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff),
                                   comm->rank, nRanks, barrierHost, nullptr);
    break;
  default:
    meta::comms::ddaReduceScatterFabric<T, 0, false>
      <<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), recvcount, static_cast<const T*>(sendbuff),
                                   comm->rank, nRanks, barrierHost, nullptr);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                        ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires its own handler + barrier state. Fabric handle
  // exchange works across nodes within an MNNVL clique.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaFabricBarrierState == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
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

ncclResult_t ncclReduceScatterDdaFabric(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
                                        ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricTyped<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricTyped<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricTyped<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
