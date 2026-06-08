/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/CollCommon.h"
#include "algorithms/all_reduce/all_reduce_dda.h"
#include "checks.h"
#include "comm.h"
#include "dda_all_reduce_common.h"
#include "debug.h"
#include "ipc_gpu_barrier.h"
#include "dda_init_detail.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaAllReduceCommonEligible;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::getGridAndBlockDims;
using nccl_dda_detail::kDdaFlatTreeThresholdBytes;
using nccl_dda_detail::kDdaNranks;

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaScratchBytes) {
    WARN(
        "DDA IPC allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes",
        count,
        count * sizeof(T),
        comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const size_t sizeBytes = count * sizeof(T);
  const unsigned threads = 512;
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk =
      wantTree && (count % static_cast<size_t>(kDdaNranks) == 0);

  if (wantTree && !treeOk) {
    INFO(
        NCCL_ALL,
        "DDA IPC: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel",
        sizeBytes,
        count,
        kDdaNranks);
  }

 
  const int nBlocksMax = ddaMaxNBlocksForScratch(); 
  auto gridBlock = getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(
        comm->ddaScratch,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));
    meta::comms::ddaAllReduceTreeIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  } else {
    meta::comms::ddaAllReduceFlatIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr) {
    return false;
  }
  // IPC path: requires its own handler + barrier state, a single node, and
  // exactly kDdaNranks ranks (the IPC kernels fix the rank count at compile
  // time).
  if (comm->ddaIpcMemHandler == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  return ddaAllReduceCommonEligible(comm, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
