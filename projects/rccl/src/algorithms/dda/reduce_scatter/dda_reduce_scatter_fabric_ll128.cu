/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of dda_all_reduce_fabric_ll128.cu; reuses the codepath-
 * agnostic ddaReduceScatterFabricLL128 kernel from reduce_scatter_dda_fabric_ll128.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Runtime-adjustable LL128 ReduceScatter block size (threads/block). Must be a
// multiple of 16 (lanes per 128B line) in [16, 1024]; invalid values fall back
// to the tuned default. Env: RCCL_DDA_LL128_RS_THREADS.
RCCL_PARAM(DdaLL128RsThreads, "DDA_LL128_RS_THREADS", 1024);

namespace {

using dda::common::kDdaLL128DataElems;
using dda::common::kDdaLL128Lanes;
using dda::common::kDdaLL128RsMaxBytes;
using dda::common::kDdaLL128RsSlotStrideLines;
using dda::common::LLLine128;

// LL128 scratch: 2 banks * nRanks slots * kDdaLL128RsSlotStrideLines * 128B.
static inline size_t ddaLL128RsScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128RsSlotStrideLines * sizeof(LLLine128);
}

// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128RsThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128RsThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLL128Typed(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = recvcount * sizeof(T); // per-rank shard bytes
  const size_t nWords = bytes >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;

  // 1D grid over line-groups; each block has threads/16 groups.
  const unsigned threads = ddaLL128RsThreads(1024); // multiple of 16 (lanes/line)
  const size_t groups = threads / (unsigned)kDdaLL128Lanes;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((numLines + groups - 1) / groups, (size_t)nBlocksMax);
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

  INFO(NCCL_COLL, "DDA fabric ReduceScatter LL128: nRanks=%d shardBytes=%zu numLines=%zu grid=%u block=%u", nRanks,
       bytes, numLines, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaReduceScatterFabricLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), recvcount,
                                   comm->rank, nRanks, epochDev, epochLen, comm->abortFlagDev);
    break;
  case 8:
    dda::common::ddaReduceScatterFabricLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), recvcount,
                                   comm->rank, nRanks, epochDev, epochLen, comm->abortFlagDev);
    break;
  default:
    dda::common::ddaReduceScatterFabricLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), recvcount,
                                   comm->rank, nRanks, epochDev, epochLen, comm->abortFlagDev);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = recvcount * ncclTypeSize(datatype); // per-rank shard
  // Payload is staged as 8-byte words, so the shard must be a whole number of
  // words.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLL128RsMaxBytes) {
    return false;
  }
  if (ddaLL128RsScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricLL128Typed<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricLL128Typed<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricLL128Typed<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
