/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_alltoall.h"

#include "algorithms/CollCommon.h"
#include "algorithms/alltoall/alltoall_dda_fabric.h"
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
static ncclResult_t ncclAllToAllDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                               cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t totalCount = count * nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA fabric alltoall: total element count %zu needs %zu bytes; comm scratch is %zu bytes", totalCount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // Use the block cap chosen at init (barrier flag buffer is sized for it).
  const int nBlocksMax = comm->ddaFabricMaxBlocks;
  auto gridBlock = meta::comms::getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  meta::comms::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  INFO(NCCL_COLL, "DDA fabric AllToAll: launching kernel: nRanks=%d count=%zu grid=%u block=%u%s", nRanks, count,
       grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));

  switch (nRanks) {
  case 4:
    meta::comms::ddaAllToAllFabric<T, 4><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  case 8:
    meta::comms::ddaAllToAllFabric<T, 8><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  default:
    meta::comms::ddaAllToAllFabric<T, 0><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                   ncclDataType_t datatype) {
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
  if (count == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  size_t totalCount = count * comm->nRanks;
  size_t need = totalCount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check for data size divisible by 16 (kernel does 16-byte vectorized loads)
  if ((count * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllToAllDdaFabric(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
