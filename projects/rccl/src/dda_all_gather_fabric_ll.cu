/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_gather.h"

#include "algorithms/all_gather/all_gather_dda_fabric_ll.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using meta::comms::kDdaLLAgMaxPerRankBytes;
using meta::comms::kDdaLLAgSlotStridePkts;
using meta::comms::LLPacket16;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// LL scratch: 2 banks * nRanks slots * kDdaLLAgSlotStridePkts * 16B.
static inline size_t ddaLLAgScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLAgSlotStridePkts * sizeof(LLPacket16);
}

// Adaptive block-per-peer fan-out. One block per peer for small messages
// larger ones split a peer's packet range across blocksPerPeer blocks
//
//   blocksPerPeer = clamp(ceil(nPk / kDdaLLAgPktsPerBlock), 1, cap)
//
// 256 pkts/block is one packet per thread at 256 threads.
constexpr size_t kDdaLLAgPktsPerBlock = 256;

// Blocks per peer for a given per-rank payload.
static inline int ddaLLAgBlocksPerPeer(size_t perRankBytes) {
  const size_t nPk = perRankBytes >> 3; // 8 payload bytes per packet
  if (nPk <= kDdaLLAgPktsPerBlock) {
    return 1;
  }
  size_t bpp = (nPk + kDdaLLAgPktsPerBlock - 1) / kDdaLLAgPktsPerBlock;
  if (bpp > (size_t)kDdaLLAgMaxBlocksPerPeer) {
    bpp = (size_t)kDdaLLAgMaxBlocksPerPeer;
  }
  return (int)bpp;
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLLTyped(
  const void* sendbuff, void* recvbuff,
  size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);

  // grid.x == nRanks (peer), grid.y == blocksPerPeer (packet split, 1 for small
  // messages).
  const unsigned threads = 256;
  const int blocksPerPeer = ddaLLAgBlocksPerPeer(perRankBytes);
  dim3 block(threads);
  dim3 grid((unsigned)nRanks, (unsigned)blocksPerPeer);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = std::min(comm->ddaLLEpochLen, nRanks*blocksPerPeer);  ;

  INFO(NCCL_COLL, "DDA fabric AllGather LL: nRanks=%d perRankBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perRankBytes, grid.x, grid.y, block.x, blocksPerPeer);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllGatherFabricLL<T, 4><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), perRankBytes,
                                                                        comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllGatherFabricLL<T, 8><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), perRankBytes,
                                                                        comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllGatherFabricLL<T, 0><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), perRankBytes,
                                                                        comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                      ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if (perRankBytes > kDdaLLAgMaxPerRankBytes) {
    return false;
  }
  if (ddaLLAgScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllGatherDdaFabricLL(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
                                      ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabric.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLLTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
