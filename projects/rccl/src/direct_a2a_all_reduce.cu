/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "direct_a2a_all_reduce.h"

#include "algorithms/CollCommon.h"
#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

RCCL_PARAM(DirectA2aEnable, "DIRECT_A2A_ENABLE", 0);

namespace {

constexpr int kDirectA2aThreads = 256;
constexpr int kDirectA2aMaxBlocks = 256;

template <typename T>
__global__ void directA2aSumKernel(const T* __restrict__ scratch, T* __restrict__ recvbuff, size_t count, int nRanks) {
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count;
       idx += (size_t)gridDim.x * blockDim.x) {
    T sum = scratch[idx];
    for (int rank = 1; rank < nRanks; ++rank) {
      sum = sum + scratch[(size_t)rank * count + idx];
    }
    recvbuff[idx] = sum;
  }
}

template <typename T>
ncclResult_t launchDirectA2aSum(void* scratch, void* recvbuff, size_t count, int nRanks, cudaStream_t stream) {
  int blocks = (int)std::min<size_t>((count + kDirectA2aThreads - 1) / kDirectA2aThreads, kDirectA2aMaxBlocks);
  directA2aSumKernel<T><<<blocks, kDirectA2aThreads, 0, stream>>>(
    static_cast<const T*>(scratch), static_cast<T*>(recvbuff), count, nRanks);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

ncclResult_t rcclDirectA2aAllReduceCommInit(ncclComm* comm) {
  if (comm == nullptr || !rcclParamDirectA2aEnable()) return ncclSuccess;
  if (!IsArchMatch(comm->archName, "gfx1151") || comm->nRanks < RCCL_DIRECT_A2A_MIN_RANKS ||
      comm->nRanks > RCCL_DIRECT_A2A_MAX_RANKS || comm->nNodes != comm->nRanks || comm->minLocalRanks != 1 ||
      comm->maxLocalRanks != 1) {
    return ncclSuccess;
  }

  const size_t scratchBytes = (size_t)comm->nRanks * RCCL_DIRECT_A2A_MAX_BYTES;
  ncclResult_t res = ncclCudaMalloc(&comm->directA2aScratch, scratchBytes, comm->memManager);
  if (res != ncclSuccess) {
    comm->directA2aScratch = nullptr;
    comm->directA2aScratchBytes = 0;
    INFO(NCCL_INIT, "DIRECT_A2A standalone scratch allocation failed; falling back to standard AllReduce");
    return ncclSuccess;
  }
  comm->directA2aScratchBytes = scratchBytes;
  INFO(NCCL_INIT, "DIRECT_A2A standalone path initialized: rank=%d/%d scratchBytes=%zu", comm->rank, comm->nRanks,
       scratchBytes);
  return ncclSuccess;
}

ncclResult_t rcclDirectA2aAllReduceCommFini(ncclComm* comm) {
  if (comm != nullptr && comm->directA2aScratch != nullptr) {
    NCCLCHECK(ncclCudaFree(comm->directA2aScratch, comm->memManager));
    comm->directA2aScratch = nullptr;
    comm->directA2aScratchBytes = 0;
  }
  return ncclSuccess;
}

bool rcclDirectA2aAllReduceEligible(ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr || comm->directA2aScratch == nullptr) return false;
  // The standalone path launches its local reduction immediately after the
  // internal P2P group completes, which requires synchronous group completion.
  if (!comm->config.blocking) return false;
  if (!IsArchMatch(comm->archName, "gfx1151")) return false;
  if (comm->nRanks < RCCL_DIRECT_A2A_MIN_RANKS || comm->nRanks > RCCL_DIRECT_A2A_MAX_RANKS ||
      comm->nNodes != comm->nRanks || comm->minLocalRanks != 1 || comm->maxLocalRanks != 1) {
    return false;
  }
  if (count == 0 || op != ncclSum) return false;
  if (datatype != ncclFloat16 && datatype != ncclBfloat16 && datatype != ncclFloat32) return false;

  const size_t typeSize = ncclTypeSize(datatype);
  if (count > RCCL_DIRECT_A2A_MAX_BYTES / typeSize) return false;
  const size_t bytes = count * typeSize;
  return (size_t)comm->nRanks * bytes <= comm->directA2aScratchBytes;
}

ncclResult_t rcclDirectA2aAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  if (!rcclDirectA2aAllReduceEligible(comm, count, datatype, op)) return ncclInvalidUsage;

  const size_t bytes = count * ncclTypeSize(datatype);
  ncclResult_t ret = ncclSuccess;

  // Stage the local contribution directly. Only remote peers participate in
  // the P2P batch, avoiding a self Send/Recv task and its special-case batching.
  void* selfSlot = static_cast<char*>(comm->directA2aScratch) + (size_t)comm->rank * bytes;
  CUDACHECK(cudaMemcpyAsync(selfSlot, sendbuff, bytes, cudaMemcpyDeviceToDevice, stream));

  rccl::Recorder::instance().skip(true);
  ret = ncclGroupStart();
  if (ret == ncclSuccess) {
    for (int peer = 0; peer < comm->nRanks && ret == ncclSuccess; ++peer) {
      if (peer == comm->rank) continue;
      ret = ncclSend(sendbuff, count, datatype, peer, comm, stream);
      if (ret == ncclSuccess) {
        void* peerSlot = static_cast<char*>(comm->directA2aScratch) + (size_t)peer * bytes;
        ret = ncclRecv(peerSlot, count, datatype, peer, comm, stream);
      }
    }
    ncclResult_t groupEndResult = ncclGroupEnd();
    if (ret == ncclSuccess) ret = groupEndResult;
  }
  rccl::Recorder::instance().skip(false);
  NCCLCHECK(ret);

  INFO(NCCL_COLL, "AllReduce: taking standalone DIRECT_A2A path: rank=%d/%d count=%zu datatype=%d bytes=%zu",
       comm->rank, comm->nRanks, count, (int)datatype, bytes);

  switch (datatype) {
  case ncclFloat16:
    return launchDirectA2aSum<half>(comm->directA2aScratch, recvbuff, count, comm->nRanks, stream);
  case ncclBfloat16:
    return launchDirectA2aSum<bf16>(comm->directA2aScratch, recvbuff, count, comm->nRanks, stream);
  case ncclFloat32:
    return launchDirectA2aSum<float>(comm->directA2aScratch, recvbuff, count, comm->nRanks, stream);
  default:
    return ncclInvalidArgument;
  }
}
