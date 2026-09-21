/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_gather/all_gather_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using nccl_dda_detail::DdaFabricBarrierState;

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllGatherFabricGeom(ncclComm* comm, size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, comm->ddaFabricMaxBlocks);
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricTyped(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                                cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const int nRanks = comm->nRanks;
  const size_t totalCount = sendcount * nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA fabric allgather: send element count %zu needs %zu bytes; comm scratch is %zu bytes", sendcount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  // Block cap chosen at init (barrier flag buffer is sized for it); sendcount is
  // already the byte count (kernel instantiated for int8_t).
  auto gridBlock = ddaAllGatherFabricGeom(comm, sendcount);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  dda::common::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  INFO(NCCL_COLL, "DDA fabric AllGather: launching kernel: nRanks=%d sendcount=%zu grid=%u block=%u%s", nRanks,
       sendcount, grid.x, block.x, (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  // Specialize the kernel for common clique sizes (compile-time NRANKS -> the
  // unrolled CollCommon allGather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllGatherFabric<T, 4><<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                                                                      static_cast<const T*>(sendbuff), comm->rank,
                                                                      nRanks, barrierHost);
    break;
  case 8:
    dda::common::ddaAllGatherFabric<T, 8><<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                                                                      static_cast<const T*>(sendbuff), comm->rank,
                                                                      nRanks, barrierHost);
    break;
  default:
    dda::common::ddaAllGatherFabric<T, 0><<<grid, block, 0, stream>>>(d_ipcbuffs, static_cast<T*>(recvbuff), sendcount,
                                                                      static_cast<const T*>(sendbuff), comm->rank,
                                                                      nRanks, barrierHost);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
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
  if (sendcount == 0) {
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

  // Check for data size divisible by 16 (kernel does 16-byte vectorized loads)
  if ((sendcount * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaFabricBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricGeom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabric(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                    ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
