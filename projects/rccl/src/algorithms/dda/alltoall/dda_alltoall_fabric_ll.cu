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
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLLBlocksPerPeer;
using dda::common::ddaLLPeerSlotPkts;
using dda::common::kDdaLLMaxBytes;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// Single source of the launch geometry: grid.x = peer (nRanks), grid.y = the
// per-peer packet split; 256 threads/block, so 256 pkts/block is one packet per thread.
static inline std::pair<dim3, dim3> ddaAllToAllFabricLLGeom(ncclComm* comm, size_t perChunkBytes) {
  const unsigned threads = 256;
  // A gather block polls a slot written by a peer block of another index, so the whole grid must be
  // co-resident: blocks per peer are capped at the comm-wide CU budget's share. A peer always keeps
  // one block, so past nRanks > ddaFabricMaxBlocks the grid is nRanks blocks, above the budget.
  const unsigned blocksPerPeer =
    ddaLLBlocksPerPeer(perChunkBytes, comm->nRanks, (size_t)kDdaLLAgMaxBlocksPerPeer,
                       (size_t)std::max(comm->ddaFabricMaxBlocks, 0));
  return std::make_pair(dim3((unsigned)comm->nRanks, blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllToAllDdaFabricLLTyped(
  const void* sendbuff, void* recvbuff,
  size_t count, // per-peer element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perChunkBytes = count * sizeof(T);

  auto gridBlock = ddaAllToAllFabricLLGeom(comm, perChunkBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const int blocksPerPeer = (int)grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaRegionLL.peersDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = ddaBankSize(comm->ddaRegionLL.bytes);

  INFO(NCCL_COLL, "DDA fabric AllToAll LL: nRanks=%d perChunkBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perChunkBytes, grid.x, grid.y, block.x, blocksPerPeer);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
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
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaRegionLL.self == nullptr ||
      comm->ddaRegionLL.peersDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
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
  // One packet carries 8B of payload, so each peer's chunk has to fit the slot the bank gives it.
  if ((perChunkBytes >> 3) > ddaLLPeerSlotPkts(ddaBankSize(comm->ddaRegionLL.bytes), comm->nRanks)) {
    return false;
  }

  return true;
}

uint32_t ncclAllToAllDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllToAllFabricLLGeom(comm, count * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                     ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  // All-to-all moves raw bytes, so instantiate once for int8_t and scale the
  // per-peer count, like ncclAllToAllDdaFabric.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricLLTyped<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
