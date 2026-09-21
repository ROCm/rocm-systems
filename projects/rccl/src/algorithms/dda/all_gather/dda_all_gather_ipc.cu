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

// Single source of the launch geometry: grid/block for a byte payload. The
// kernel is instantiated for int8_t, so `bytes` is the per-block element count.
static inline std::pair<dim3, dim3> ddaAllGatherIpcGeom(size_t bytes) {
  return dda::common::getGridAndBlockDims(bytes, 1, ddaMaxNBlocksForScratch());
}

template <typename T, int NRANKS>
static ncclResult_t ncclAllGatherDdaIpcLaunch(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                              cudaStream_t stream) {
  // Defends the dispatcher's invariant: NRANKS is either 0 (use comm->nRanks at
  // runtime) or exactly comm->nRanks (the compile-time kDdaNranks
  // specialisation). A dispatch bug that picked the wrong instantiation for
  // the live rank count would otherwise silently read peer slots the comm was
  // never told about -- ipc_init.cu pre-zeroes the unused tail of the peer
  // table rather than leaving it as garbage, so this would produce a wrong
  // numeric result on hardware, not a crash.
  if (NRANKS != 0 && NRANKS != comm->nRanks) {
    WARN("DDA IPC allgather: dispatch bug, instantiated for %d ranks but comm has %d", NRANKS, comm->nRanks);
    return ncclInternalError;
  }
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

  dda::common::ddaAllGatherIpc<T, NRANKS, false><<<grid, block, 0, stream>>>(
    d_ipcbuffs, static_cast<T*>(recvbuff), sendcount, static_cast<const T*>(sendbuff), comm->rank, comm->nRanks,
    barrierHost);

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
static ncclResult_t ncclAllGatherDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t sendcount, ncclComm* comm,
                                             cudaStream_t stream) {
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
    WARN("DDA IPC allgather: unsupported nRanks %d", comm->nRanks);
    return ncclInvalidUsage;
  }
  if (comm->nRanks == kDdaNranks) {
    return ncclAllGatherDdaIpcLaunch<T, kDdaNranks>(sendbuff, recvbuff, sendcount, comm, stream);
  }
  return ncclAllGatherDdaIpcLaunch<T, 0>(sendbuff, recvbuff, sendcount, comm, stream);
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
  if (!ncclDdaIpcNranksSupported(comm->nRanks)) {
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
