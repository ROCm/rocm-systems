/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/all_reduce/all_reduce_dda.h"
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

  const size_t sizeBytes = count * sizeof(T);
  const unsigned threads = 512;
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk = wantTree && (count % static_cast<size_t>(kDdaNranks) == 0);

  if (wantTree && !treeOk) {
    INFO(NCCL_ALL, "DDA IPC: size %zu B > 256KB but count %zu not divisible by %d; using flat kernel", sizeBytes, count,
         kDdaNranks);
  }

  auto gridBlock = ddaAllReduceIpcGeom(count, sizeof(T));
  const auto& grid = gridBlock.first;
  const auto& block = gridBlock.second;

  auto* barrierState = static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  dda::common::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(comm->ddaScratch, sendbuff, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTreeIpc<T, kDdaNranks, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count,
                          static_cast<const T*>(sendbuff), comm->rank, barrierHost, nullptr);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatIpc<T, kDdaNranks, false>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, d_ipcbuffs, static_cast<T*>(recvbuff), count,
                          static_cast<const T*>(sendbuff), comm->rank, barrierHost, nullptr);
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// RCCL_DDA_NAN_AR_ONESHOT_MAX: largest NaN-flag AllReduce, in bytes, that pushes
// the whole input to every peer instead of going through two shots.
static size_t ddaNanOneShotMax() {
  static const size_t v = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_AR_ONESHOT_MAX");
    return s ? (size_t)strtoull(s, nullptr, 0) : (size_t)64 * 1024;
  }();
  return v;
}

enum class DdaNanArShape { None, OneShot, TwoShot };

static DdaNanArShape ddaNanArShape(const ncclComm* comm, size_t count, size_t typeSize) {
  const size_t bytes = count * typeSize;
  const bool twoShot = count % kDdaNranks == 0 && rcclDdaNanFits(comm, bytes / kDdaNranks);
  if (bytes <= ddaNanOneShotMax() || !twoShot) {
    return rcclDdaNanFits(comm, bytes) ? DdaNanArShape::OneShot : DdaNanArShape::None;
  }
  return DdaNanArShape::TwoShot;
}

static std::pair<dim3, dim3> ddaNanAllReduceGeom(const ncclComm* comm, size_t count, size_t typeSize) {
  const size_t bytes = count * typeSize;
  const size_t units = ddaNanArShape(comm, count, typeSize) == DdaNanArShape::TwoShot ? bytes / kDdaNranks / 16 : bytes / 16;
  return rcclDdaNanGeometry(units, 128);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaNan(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                        cudaStream_t stream) {
  const bool twoShot = ddaNanArShape(comm, count, sizeof(T)) == DdaNanArShape::TwoShot;
  const size_t units = count * sizeof(T) / 16 / (twoShot ? kDdaNranks : 1);
  const auto gridBlock = ddaNanAllReduceGeom(comm, count, sizeof(T));
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (twoShot) {
    hipExtLaunchKernelGGL((dda::nan::ddaNanAllReduceTwoShot<T>), gridBlock.first, gridBlock.second, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, rcclDdaNanPeers(comm), comm->ddaNanEpochDev,
                          static_cast<const v4u*>(sendbuff), static_cast<v4u*>(recvbuff), units, comm->rank);
  } else {
    hipExtLaunchKernelGGL((dda::nan::ddaNanAllReduceOneShot<T>), gridBlock.first, gridBlock.second, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, rcclDdaNanPeers(comm), comm->ddaNanEpochDev,
                          static_cast<const v4u*>(sendbuff), static_cast<v4u*>(recvbuff), units, comm->rank);
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
  // IPC path: requires its own handler + barrier state, a single node, and
  // exactly kDdaNranks ranks (the IPC kernels fix the rank count at compile
  // time).
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
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
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return ddaNanArShape(comm, count, ncclTypeSize(datatype)) != DdaNanArShape::None;
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
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    return ddaNanAllReduceGeom(comm, count, ncclTypeSize(datatype)).first.x;
  }
  const auto grid = ddaAllReduceIpcGeom(count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaIpc(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  if (rcclNanProtoForcedFor(comm->nNodes, comm->nRanks, datatype)) {
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaNan<float>(sendbuff, recvbuff, count, comm, stream);
    case ncclFloat16:
      return ncclAllReduceDdaNan<half>(sendbuff, recvbuff, count, comm, stream);
    case ncclBfloat16:
      return ncclAllReduceDdaNan<bf16>(sendbuff, recvbuff, count, comm, stream);
    default:
      return ncclInvalidArgument;
    }
  }
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
