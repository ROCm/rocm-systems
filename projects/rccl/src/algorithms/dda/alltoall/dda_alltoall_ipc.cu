/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/alltoall/alltoall_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/ipc/ipc_gpu_barrier.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/ipc/ipc_init.h"

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
static ncclResult_t ncclAllToAllDdaIpcLaunch(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                             cudaStream_t stream) {
  // Defends the dispatcher's invariant: NRANKS is either 0 (use comm->nRanks at
  // runtime) or exactly comm->nRanks (the compile-time kDdaNranks
  // specialisation). A dispatch bug that picked the wrong instantiation for
  // the live rank count would otherwise silently read peer slots the comm was
  // never told about -- ipc_init.cu pre-zeroes the unused tail of the peer
  // table rather than leaving it as garbage, so this would produce a wrong
  // numeric result on hardware, not a crash.
  if (NRANKS != 0 && NRANKS != comm->nRanks) {
    WARN("DDA IPC alltoall: dispatch bug, instantiated for %d ranks but comm has %d", NRANKS, comm->nRanks);
    return ncclInternalError;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }

  const size_t totalCount = count * comm->nRanks;
  if (totalCount * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC alltoall: total element count %zu needs %zu bytes; comm scratch is %zu bytes", totalCount,
         totalCount * sizeof(T), comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  // For alltoall, we use count for grid calculation (data per rank pair)
  auto gridBlock = dda::common::getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  if (dda::common::ddaAlltoAllSingleBlockGrid(count, sizeof(T))) {
    dda::common::ddaAllToAllIpc<T, NRANKS, false, true><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, comm->nRanks,
      barrierHost);
  } else {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, totalCount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    dda::common::ddaAllToAllIpc<T, NRANKS, false, false><<<grid, block, 0, stream>>>(
      d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, comm->nRanks,
      barrierHost);
  }
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Dispatch to the template instantiation for the active participant count.
// Only the default kDdaNranks clique gets a compile-time specialisation, keeping
// that path bit- and perf-identical to baseline; every other supported count uses
// the NRANKS_CT == 0 runtime kernel. Specialising all seven would compile one
// kernel per count per type for each DEFAULT_GPUS target, for a path that is
// default-off and confined to gfx942/gfx950.
template <typename T>
static ncclResult_t ncclAllToAllDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                            cudaStream_t stream) {
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
    WARN("DDA IPC alltoall: unsupported nRanks %d", comm->nRanks);
    return ncclInvalidUsage;
  }
  if (comm->nRanks == kDdaNranks) {
    return ncclAllToAllDdaIpcLaunch<T, kDdaNranks>(sendbuff, recvbuff, count, comm, stream);
  }
  return ncclAllToAllDdaIpcLaunch<T, 0>(sendbuff, recvbuff, count, comm, stream);
}

} // namespace

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
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
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

  // Check for data size divisible by 16
  if ((count * ncclTypeSize(datatype)) % 16) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllToAllDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaIpcTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
