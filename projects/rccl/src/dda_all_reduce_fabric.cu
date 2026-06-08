/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_fabric.h"
#include "checks.h"
#include "comm.h"
#include "dda_all_reduce_common.h"
#include "debug.h"
#include "dda_init_detail.h"
#include "fabric_gpu_barrier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using nccl_dda_detail::ddaAllReduceCommonEligible;
using nccl_dda_detail::DdaFabricBarrierState;
using nccl_dda_detail::getGridAndBlockDims;
using nccl_dda_detail::kDdaFlatTreeThresholdBytes;

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaScratchBytes) {
    WARN(
        "DDA fabric allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes",
        count,
        count * sizeof(T),
        comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nRanks = comm->nRanks;
  const size_t sizeBytes = count * sizeof(T);
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk =
      wantTree && (count % static_cast<size_t>(nRanks) == 0);

  if (wantTree && !treeOk) {
    INFO(
        NCCL_ALL,
        "DDA fabric: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel",
        sizeBytes,
        count,
        nRanks);
  }

  // Use the block cap chosen at init (barrier flag buffer is sized for it).
  const int nBlocksMax = comm->ddaFabricMaxBlocks;
  auto gridBlock = getGridAndBlockDims(count, sizeof(T), nBlocksMax);
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState =
      static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
  meta::comms::FabricGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  // Specialize the kernel for common clique sizes (compile-time NRANKS -> the
  // unrolled CollCommon reduce/gather), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  INFO(NCCL_COLL,
       "DDA fabric AllReduce: launching %s kernel: nRanks=%d count=%zu sizeBytes=%zu grid=%u block=%u%s",
       treeOk ? "tree (two-shot)" : "flat (one-shot)",
       nRanks, count, sizeBytes, grid.x, block.x,
       (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(
        comm->ddaScratch,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));
    // NRANKS_CT 4/8: unrolled CollCommon reduce; 0: runtime fallback.
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=4 (unrolled)");
      meta::comms::ddaAllReduceTreeFabric<T, 4, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: tree path, NRANKS_CT=8 (unrolled)");
      meta::comms::ddaAllReduceTreeFabric<T, 8, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    default:
      INFO(NCCL_COLL,
           "DDA fabric AllReduce: tree path, NRANKS_CT=0 (runtime, nRanks=%d)",
           nRanks);
      meta::comms::ddaAllReduceTreeFabric<T, 0, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    }
  } else {
    switch (nRanks) {
    case 4:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=4 (unrolled)");
      meta::comms::ddaAllReduceFlatFabric<T, 4, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    case 8:
      INFO(NCCL_COLL, "DDA fabric AllReduce: flat path, NRANKS_CT=8 (unrolled)");
      meta::comms::ddaAllReduceFlatFabric<T, 8, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    default:
      INFO(NCCL_COLL,
           "DDA fabric AllReduce: flat path, NRANKS_CT=0 (runtime, nRanks=%d)",
           nRanks);
      meta::comms::ddaAllReduceFlatFabric<T, 0, false><<<grid, block, 0, stream>>>(
          d_ipcbuffs, static_cast<T*>(recvbuff), count,
          static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
          nullptr);
      break;
    }
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricEligible(
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
  // Fabric path: requires its own handler + barrier state. Fabric handle
  // exchange works across nodes within an MNNVL clique
  if (comm->ddaFabricMemHandler == nullptr ||
      comm->ddaFabricBarrierState == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  return ddaAllReduceCommonEligible(comm, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaFabric(
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
    return ncclAllReduceDdaFabricTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
