/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the "simple_warpsync" DDA fabric all-reduce.
 * Copy of dda_all_reduce_fabric_ll128_warpsync.cu that drives the
 * ddaAllReduceFlatSimpleWarpsync kernel; selectable at runtime against the LL128
 * warpsync launcher via RCCL_DDA_AR_SIMPLE_WARPSYNC (see collectives.cc). Kept as
 * a separate translation unit so the two can diverge independently.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_fabric_simple_warpsync.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using meta::comms::kDdaSimpleWarpsyncArReserveBytes;
using meta::comms::TT;

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricSimpleWarpsyncTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes / sizeof(TT);

  // 1D grid over line-groups;
  const unsigned threads = 1024; // multiple of 16
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>(
      (nWords + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    assert (blocks > 0);
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(
      NCCL_COLL,
      "DDA fabric AllReduce simple warpsync: nRanks=%d bytes=%zu nWords=%zu grid=%u block=%u",
      nRanks, bytes, nWords, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatSimpleWarpsync<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatSimpleWarpsync<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllReduceFlatSimpleWarpsync<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricSimpleWarpsyncEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes % sizeof(TT) != 0) {
    return false;
  }

  size_t need = (size_t)2 * comm->nRanks * count * ncclTypeSize(datatype) + kDdaSimpleWarpsyncArReserveBytes;
  if (need > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaFabricSimpleWarpsync(
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
    return ncclAllReduceDdaFabricSimpleWarpsyncTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricSimpleWarpsyncTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricSimpleWarpsyncTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
