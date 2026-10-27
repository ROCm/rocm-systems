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
#include "nccl_device/gin/anvil_sdma/gin_fabric_ll_policy.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using dda::common::kDdaLLMaxBytes;

// The host DDA launcher below and the GIN device-API launcher
// (GinAlltoAllKernel in rccl-tests) drive the same ddaAllToAllFabricLL kernel
// over LL packet cells of the same geometry. Every constant that geometry
// depends on therefore exists in two headers, so pin all six pairs here: this
// is the one translation unit that sees both sides.
static_assert(gin::fabric::kGinFabricLlMaxNranks == dda::common::kDdaMaxNranks,
              "GIN device-API and host DDA max-rank limits must match");
static_assert(gin::fabric::kGinFabricLlMaxBytes == dda::common::kDdaLLMaxBytes,
              "GIN device-API and host DDA LL size limits must match");
static_assert(gin::fabric::kGinFabricLlAgMaxBlocksPerPeer == nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer,
              "GIN device-API and host DDA block limits must match");
static_assert(gin::fabric::kGinFabricLlPacketBytes == sizeof(dda::common::LLPacket16),
              "GIN device-API and host DDA LL packet size must match");
static_assert(gin::fabric::kGinFabricLlA2ASlotStridePkts == dda::common::kDdaLLA2ASlotStridePkts,
              "GIN device-API and host DDA LL slot stride must match");

// Scratch sizing and the block-per-peer fan-out are taken from the GIN policy
// header rather than recomputed here: a local copy would size the same arena
// and pick grid.y from a second body that no static_assert can pin.
// ginFabricLlA2AScratchBytes: 2 banks * nRanks slots * slot stride * 16B.
// ginFabricLlAlltoAllBlocksPerPeer: one block per peer for small chunks, else
// split a peer's packet range over up to kGinFabricLlAgMaxBlocksPerPeer blocks.
using gin::fabric::ginFabricLlA2AScratchBytes;
using gin::fabric::ginFabricLlAlltoAllBlocksPerPeer;

// Single source of the launch geometry: grid.x = peer (nRanks), grid.y = the
// per-peer packet split; 256 threads/block.
static inline std::pair<dim3, dim3> ddaAllToAllFabricLLGeom(ncclComm* comm, size_t perChunkBytes) {
  const unsigned threads = 256;
  const int blocksPerPeer = ginFabricLlAlltoAllBlocksPerPeer(perChunkBytes);
  return std::make_pair(dim3((unsigned)comm->nRanks, (unsigned)blocksPerPeer), dim3(threads));
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

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllToAll LL: nRanks=%d perChunkBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perChunkBytes, grid.x, grid.y, block.x, blocksPerPeer);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perChunkBytes, comm->rank, nRanks, epochDev, epochLen, params);
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
  if (ginFabricLlA2AScratchBytes(comm->nRanks) > comm->ddaScratchBytes) {
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
