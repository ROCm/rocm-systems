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
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                      ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = recvcount * sizeof(T); // per-rank shard bytes
  const size_t nPk = bytes >> 3;              // 8 payload bytes per packet
  const int peerLanes = dda::common::ddaLLRsPeerLanes(nRanks);

  const unsigned threads = 256;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  // One thread per packet and peer lane.
  const size_t laneThreads = nPk * (size_t)peerLanes;
  unsigned blocks = (unsigned)std::min<size_t>((laneThreads + threads - 1) / threads, (size_t)nBlocksMax);
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
  // Bank 1 starts at ddaBankSize, matching the other LL tiers on this scratch.
  const size_t bankSize = dda::common::ddaBankSize(comm->ddaScratchBytes);

  INFO(NCCL_COLL,
       "DDA fabric ReduceScatter LL: nRanks=%d shardBytes=%zu nPk=%zu peerLanes=%d grid=%u block=%u bankSize=%zu",
       nRanks, bytes, nPk, peerLanes, grid.x, block.x, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  auto launch = [&](auto kernel) {
    hipExtLaunchKernelGGL(kernel, grid, block, 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers,
                          static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), recvcount, comm->rank, nRanks,
                          epochDev, epochLen, bankSize, params);
  };
  // NRANKS_CT 4/8 unroll the reduce loop. 0 is the runtime fallback, which splits each packet's peers across peerLanes.
  switch (nRanks) {
  case 4:
    launch(dda::common::ddaReduceScatterFabricLL<T, 4>);
    break;
  case 8:
    launch(dda::common::ddaReduceScatterFabricLL<T, 8>);
    break;
  default:
    if (peerLanes == 1) {
      launch(dda::common::ddaReduceScatterFabricLL<T, 0, 1>);
    } else if (peerLanes == 2) {
      launch(dda::common::ddaReduceScatterFabricLL<T, 0, 2>);
    } else {
      launch(dda::common::ddaReduceScatterFabricLL<T, 0, dda::common::kDdaLLRsMaxPeerLanes>);
    }
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = recvcount * ncclTypeSize(datatype); // per-rank shard

  if (bytes % 16 != 0) {
    return false;
  }
  // Slot holds kDdaLLRsMaxBytes of payload (stride = cap/8 packets).
  if (bytes > dda::common::kDdaLLRsMaxBytes) {
    return false;
  }
  // Bank 0 is [0, bankBytes). Bank 1 is [bankSize, bankSize + bankBytes).
  const size_t bankBytes = dda::common::ddaLLRsBankBytes(comm->nRanks);
  const size_t bankSize = dda::common::ddaBankSize(comm->ddaScratchBytes);
  if (bankBytes > bankSize || bankSize + bankBytes > comm->ddaScratchBytes) {
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
  case ncclFloat8e4m3:
    return ncclReduceScatterDdaFabricLLTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e5m2:
    return ncclReduceScatterDdaFabricLLTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
