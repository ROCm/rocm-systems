/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll.h"
#include "checks.h"
#include "comm.h"
#include "algorithms/dda/dda_init_detail.h" // nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using dda::common::kDdaLLAgSlotStridePkts;
using dda::common::kDdaLLMaxBytes;
using dda::common::LLPacket16;
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

// Single source of the launch geometry: grid.x = peer (nRanks), grid.y = the
// per-peer packet split; 256 threads/block.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLLGeom(ncclComm* comm, size_t perRankBytes) {
  const unsigned threads = 256;
  const int blocksPerPeer = ddaLLAgBlocksPerPeer(perRankBytes);
  return std::make_pair(dim3((unsigned)comm->nRanks, (unsigned)blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLLTyped(
  const void* sendbuff, void* recvbuff,
  size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);

  auto gridBlock = ddaAllGatherFabricLLGeom(comm, perRankBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const uint32_t blocksPerPeer = grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllGather LL: nRanks=%d perRankBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%u)",
       nRanks, perRankBytes, grid.x, grid.y, block.x, blocksPerPeer);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllGatherFabricLL<T, 4><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), perRankBytes,
                                                                        comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllGatherFabricLL<T, 8><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                        static_cast<const T*>(sendbuff), perRankBytes,
                                                                        comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllGatherFabricLL<T, 0><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
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
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  // expand from 8B to 16B
  if (perRankBytes * 2 > kDdaLLMaxBytes) {
    return false;
  }
  if (ddaLLAgScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaFabricLLBlocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricLLGeom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
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
