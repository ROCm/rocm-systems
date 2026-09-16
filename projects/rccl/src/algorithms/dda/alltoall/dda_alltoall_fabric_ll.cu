/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-to-all.
 * Personalized analogue of dda_all_gather_fabric_ll.cu; reuses the codepath-
 * agnostic ddaAllToAllFabricLL kernel from alltoall_dda_fabric_ll.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/alltoall/alltoall_dda_fabric_ll.h"
#include "checks.h"
#include "comm.h"
#include "algorithms/dda/dda_init_detail.h" // nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

using dda::common::kDdaLLA2ASlotStridePkts;
using dda::common::kDdaLLMaxBytes;
using dda::common::LLPacket16;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// LL scratch: 2 banks * nRanks slots * kDdaLLA2ASlotStridePkts * 16B.
static inline size_t ddaLLA2AScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLLA2ASlotStridePkts * sizeof(LLPacket16);
}

// Adaptive block-per-peer fan-out. One block per peer for small chunks; larger
// ones split a peer's packet range across blocksPerPeer blocks. 256 pkts/block
// is one packet per thread at 256 threads.
constexpr size_t kDdaLLA2APktsPerBlock = 256;

static inline int ddaLLA2ABlocksPerPeer(size_t perChunkBytes) {
  const size_t nPk = perChunkBytes >> 3; // 8 payload bytes per packet
  if (nPk <= kDdaLLA2APktsPerBlock) {
    return 1;
  }
  size_t bpp = (nPk + kDdaLLA2APktsPerBlock - 1) / kDdaLLA2APktsPerBlock;
  if (bpp > (size_t)kDdaLLAgMaxBlocksPerPeer) {
    bpp = (size_t)kDdaLLAgMaxBlocksPerPeer;
  }
  return (int)bpp;
}

template <typename T>
static ncclResult_t ncclAllToAllDdaFabricLLTyped(
  const void* sendbuff, void* recvbuff,
  size_t count, // per-peer element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perChunkBytes = count * sizeof(T);

  const unsigned threads = 256;
  const int blocksPerPeer = ddaLLA2ABlocksPerPeer(perChunkBytes);
  dim3 block(threads);
  dim3 grid((unsigned)nRanks, (unsigned)blocksPerPeer);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllToAll LL: nRanks=%d perChunkBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perChunkBytes, grid.x, grid.y, block.x, blocksPerPeer);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllToAllFabricLL<T, 4><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                       static_cast<const T*>(sendbuff), perChunkBytes,
                                                                       comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllToAllFabricLL<T, 8><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                       static_cast<const T*>(sendbuff), perChunkBytes,
                                                                       comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllToAllFabricLL<T, 0><<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff),
                                                                       static_cast<const T*>(sendbuff), perChunkBytes,
                                                                       comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t perChunkBytes = count * ncclTypeSize(datatype);
  // Payload is staged as 8-byte LL packets; each 16B store covers 2 packets.
  if (perChunkBytes % 16 != 0) {
    return false;
  }
  // expand from 8B to 16B
  if (perChunkBytes * 2 > kDdaLLMaxBytes) {
    return false;
  }
  if (ddaLLA2AScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllToAllDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                     ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // All-to-all moves raw bytes, so instantiate once for int8_t and scale the
  // per-peer count, like ncclAllToAllDdaFabric.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricLLTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
