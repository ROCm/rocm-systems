/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda.h"
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
#include <utility>

namespace {

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;

/** Flat below this size; tree above (see ddaAllReduceFlatIpc / ddaAllReduceTreeIpc). */
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

// Single source of the launch geometry: grid/block for `count` elements of
// `typeSize` bytes, capped by the scratch-derived block count.
static inline std::pair<dim3, dim3> ddaAllReduceIpcGeom(size_t count, int typeSize) {
  return dda::common::getGridAndBlockDims(count, typeSize, ddaMaxNBlocksForScratch());
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                             cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr ||
      comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaScratchBytes) {
    WARN("DDA IPC allreduce: element count %zu needs %zu bytes; comm scratch is %zu bytes", count, count * sizeof(T),
         comm->ddaScratchBytes);
    return ncclInvalidArgument;
  }

  const int nRanks = comm->nRanks;
  const size_t sizeBytes = count * sizeof(T);
  const unsigned threads = 512;
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk = wantTree && (count % static_cast<size_t>(nRanks) == 0);

  if (wantTree && !treeOk) {
    INFO(NCCL_ALL, "DDA IPC: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel", sizeBytes, count,
         nRanks);
  }

  auto gridBlock = ddaAllReduceIpcGeom(count, sizeof(T));
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  INFO(NCCL_COLL, "DDA IPC AllReduce: launching %s kernel: nRanks=%d count=%zu grid=%u block=%u%s",
       treeOk ? "tree" : "flat", nRanks, count, grid.x, block.x,
       (nRanks == 4 || nRanks == 8) ? " (unrolled)" : " (runtime)");

  // Specialize the kernels for common clique sizes (compile-time NRANKS_CT ->
  // the unrolled CollCommon helpers), and fall back to the runtime kernel
  // (NRANKS_CT == 0) for any other size.
  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    switch (nRanks) {
    case 4:
      dda::common::ddaAllReduceTreeIpc<T, 4, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    case 8:
      dda::common::ddaAllReduceTreeIpc<T, 8, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    default:
      dda::common::ddaAllReduceTreeIpc<T, 0, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    }
  } else {
    switch (nRanks) {
    case 4:
      dda::common::ddaAllReduceFlatIpc<T, 4, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    case 8:
      dda::common::ddaAllReduceFlatIpc<T, 8, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    default:
      dda::common::ddaAllReduceFlatIpc<T, 0, false><<<grid, block, 0, stream>>>(
        d_ipcbuffs, static_cast<T*>(recvbuff), count, static_cast<const T*>(sendbuff), comm->rank, nRanks, barrierHost,
        nullptr);
      break;
    }
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaIpcEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr) {
    return false;
  }
  // IPC path: requires its own handler + barrier state, a single node, and a
  // clique no larger than kDdaMaxNranks. Sizes other than the specialized 4/8
  // dispatch to the runtime kernel.
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  // Checks shared by both DDA all-reduce backends.
  if (comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes > comm->ddaScratchBytes) {
    return false;
  }
  if (bytes % 16) {
    // 16-byte alignment: the DDA kernels do 16-byte vectorized loads.
    return false;
  }
  if (bytes > kDdaFlatTreeThresholdBytes) {
    if (count % comm->nRanks || ((count / comm->nRanks) * ncclTypeSize(datatype)) % 16) {
      // Two-shot/tree path: each rank reduces count/nRanks elements, so that
      // per-rank slice must also be 16-byte aligned.
      return false;
    }
  }
  return true;
}

uint32_t ncclAllReduceDdaIpcBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  (void)comm;
  const auto grid = ddaAllReduceIpcGeom(count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcTyped<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcTyped<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcTyped<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
