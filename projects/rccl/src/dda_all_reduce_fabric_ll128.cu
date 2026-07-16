/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll128.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL128 kernel from all_reduce_dda_fabric_ll128.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Runtime-adjustable LL128 AllReduce block size (threads/block). Must be a
// multiple of 16 (lanes per 128B line) in [16, 1024]; invalid values fall back
// to the tuned default. Env: RCCL_DDA_LL128_AR_THREADS.
RCCL_PARAM(DdaLL128ArThreads, "DDA_LL128_AR_THREADS", 1024);

namespace {

using meta::comms::kDdaLL128ArMaxBytes;
using meta::comms::kDdaLL128ArSlotStrideLines;
using meta::comms::kDdaLL128DataElems;
using meta::comms::kDdaLL128Lanes;
using meta::comms::LLLine128;

// LL128 scratch: 2 banks * nRanks slots * kDdaLL128ArSlotStrideLines * 128B.
static inline size_t ddaLL128ArScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128ArSlotStrideLines * sizeof(LLLine128);
}

// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128ArThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128ArThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128Typed(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3;
  const size_t numLines =
      (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;

  // 1D grid over line-groups; each block has threads/16 groups.
  const unsigned threads = ddaLL128ArThreads(1024); // multiple of 16 (lanes/line)
  const size_t groups = threads / (unsigned)kDdaLL128Lanes;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>(
      (numLines + groups - 1) / groups, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(
      NCCL_COLL,
      "DDA fabric AllReduce LL128: nRanks=%d bytes=%zu numLines=%zu grid=%u block=%u",
      nRanks, bytes, numLines, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL128<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL128<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL128<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricLL128Eligible(
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
  // Payload is staged as 8-byte words, so it must be a whole number of words.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLL128ArMaxBytes) {
    return false;
  }
  if (ddaLL128ArScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaFabricLL128(
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
    return ncclAllReduceDdaFabricLL128Typed<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricLL128Typed<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricLL128Typed<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
