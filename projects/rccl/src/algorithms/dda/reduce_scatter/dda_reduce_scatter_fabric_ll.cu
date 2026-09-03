/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of dda_all_reduce_fabric_ll.cu; reuses the codepath-
 * agnostic ddaReduceScatterFabricLL kernel from reduce_scatter_dda_fabric_ll.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

using dda::common::kDdaLLMaxBytes;
using dda::common::kDdaLLRsSlotStridePkts;
using dda::common::LLPacket16;

// LL scratch footprint: 2 banks * nRanks slots * slotStride packets * 16B.
static inline size_t ddaLLRsScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLRsSlotStridePkts * sizeof(LLPacket16);
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                      ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = recvcount * sizeof(T); // per-rank shard bytes
  const size_t nPk = bytes >> 3;              // 8 payload bytes per packet

  const unsigned threads = 256;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((nPk + threads - 1) / threads, (size_t)nBlocksMax);
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

  INFO(NCCL_COLL, "DDA fabric ReduceScatter LL: nRanks=%d shardBytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk,
       grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaReduceScatterFabricLL<T, 4><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                            static_cast<const T*>(sendbuff), recvcount,
                                                                            comm->rank, nRanks, epochDev, epochLen,
                                                                            comm->abortFlagDev);
    break;
  case 8:
    dda::common::ddaReduceScatterFabricLL<T, 8><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                            static_cast<const T*>(sendbuff), recvcount,
                                                                            comm->rank, nRanks, epochDev, epochLen,
                                                                            comm->abortFlagDev);
    break;
  default:
    dda::common::ddaReduceScatterFabricLL<T, 0><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                            static_cast<const T*>(sendbuff), recvcount,
                                                                            comm->rank, nRanks, epochDev, epochLen,
                                                                            comm->abortFlagDev);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
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

  if (bytes % 16 != 0) {
    return false;
  }
  // expand from 8B to 16B
  if (bytes * 2 > kDdaLLMaxBytes) {
    return false;
  }
  if (ddaLLRsScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaFabricLL(const void* sendbuff, void* recvbuff, size_t recvcount,
                                          ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                          cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricLLTyped<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricLLTyped<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricLLTyped<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
