/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/alltoall/alltoall_dda.h"
#include "algorithms/dda/dda_nan.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/dda_init_detail.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// Single source of the launch geometry: grid/block for a byte payload (the
// kernel is instantiated for int8_t). The grid is sized from the per-rank-pair
// chunk, not the whole message.
static inline std::pair<dim3, dim3> ddaAllToAllIpcGeom(size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, ddaMaxNBlocksForScratch());
}

template <typename T>
static ncclResult_t ncclAllToAllDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                            cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalBytes = count * comm->nRanks;
  if (totalBytes > comm->ddaScratchBytes) {
    WARN("DDA IPC alltoall: total %zu bytes exceeds comm scratch %zu bytes", totalBytes,
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  auto gridBlock = ddaAllToAllIpcGeom(count);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (dda::common::ddaAlltoAllSingleBlockGrid(count, sizeof(T))) {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllIpc<T, kDdaNranks, false, true>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count,
                          static_cast<const T*>(sendbuff), comm->rank, barrierHost);
  } else {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalBytes, cudaMemcpyDeviceToDevice, stream));
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllIpc<T, kDdaNranks, false, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count,
                          static_cast<const T*>(sendbuff), comm->rank, barrierHost);
  }
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// The registered kernel fills the recv buffer before it has sent everything, so
// it needs send and recv to be disjoint.
static bool ddaNanAllToAllRecvAddrs(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t bytesPerPeer,
                                    dda::nan::RegAddrs* mine) {
  const size_t total = bytesPerPeer * comm->nRanks;
  const uintptr_t s = (uintptr_t)sendbuff, r = (uintptr_t)recvbuff;
  if (s < r + total && r < s + total) return false;
  return rcclDdaNanRecvAddrs(comm, recvbuff, total, bytesPerPeer, mine);
}

static ncclResult_t ncclAllToAllDdaNanBytes(const void* sendbuff, void* recvbuff, size_t bytesPerPeer,
                                            ncclComm* comm, cudaStream_t stream) {
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  dda::nan::RegAddrs mine;
  const bool reg = ddaNanAllToAllRecvAddrs(comm, sendbuff, recvbuff, bytesPerPeer, &mine);
  const bool perPeer = !reg && rcclDdaNanPerPeer(bytesPerPeer);
  const auto gridBlock = rcclDdaNanCopyGeometry(comm, bytesPerPeer / 16, perPeer);
  if (reg) {
    hipExtLaunchKernelGGL((dda::nan::ddaNanAllToAllReg<uint32_t>), gridBlock.first, gridBlock.second, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, rcclDdaNanPeers(comm), comm->ddaNanEpochDev,
                          static_cast<const v4u*>(sendbuff), static_cast<v4u*>(recvbuff), mine, bytesPerPeer / 16,
                          comm->rank);
  } else {
    hipExtLaunchKernelGGL((perPeer ? dda::nan::ddaNanAllToAllPeer<uint32_t> : dda::nan::ddaNanAllToAll<uint32_t>),
                          gridBlock.first, gridBlock.second, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, rcclDdaNanPeers(comm), comm->ddaNanEpochDev,
                          static_cast<const v4u*>(sendbuff), static_cast<v4u*>(recvbuff), bytesPerPeer / 16,
                          comm->rank);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaNanEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype) {
  if (comm == nullptr || count == 0) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  if (!rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return false;
  }
  dda::nan::RegAddrs unused;
  return rcclDdaNanFits(comm, count * ncclTypeSize(datatype)) ||
         ddaNanAllToAllRecvAddrs(comm, sendbuff, recvbuff, count * ncclTypeSize(datatype), &unused);
}

uint32_t ncclAllToAllDdaNanBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t bytesPerPeer = count * ncclTypeSize(datatype);
  const dim3 grid = rcclDdaNanCopyGeometry(comm, bytesPerPeer / 16, rcclDdaNanPerPeer(bytesPerPeer)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaNan(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream) {
  return ncclAllToAllDdaNanBytes(sendbuff, recvbuff, count * ncclTypeSize(datatype), comm, stream);
}

bool ncclAllToAllDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != nccl_dda_detail::kDdaNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return ncclAllToAllDdaNanEligible(comm, sendbuff, recvbuff, count, datatype);
  }

  size_t totalCount = count * comm->nRanks;
  size_t need = totalCount * ncclTypeSize(datatype);
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  // Check for data size divisible by 16
  if ((count * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

uint32_t ncclAllToAllDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return ncclAllToAllDdaNanBlocks(comm, count, datatype);
  }
  const auto grid = ddaAllToAllIpcGeom(count * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return ncclAllToAllDdaNan(sendbuff, recvbuff, count, datatype, comm, stream);
  }
  return ncclAllToAllDdaIpcTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
