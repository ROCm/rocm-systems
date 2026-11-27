/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll.h"
#include "algorithms/dda/dda_init_detail.h" // nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer
#include "algorithms/dda/dda_ll_region.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

namespace {

using dda::common::ddaLLAgSlotPkts;
using dda::common::LLPacket16;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

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
// per-peer packet split; 256 threads/block. IPC: kDdaSimpleThreads-wide blocks, rows within the wide window's blocks
// (nccl_dda_detail::ddaIpcLLPeerRows).
static inline std::pair<dim3, dim3> ddaAllGatherFabricLLGeom(ncclComm* comm, size_t perRankBytes) {
  if (nccl_dda_detail::ddaLLOnIpc(comm)) {
    const unsigned threads = dda::common::kDdaSimpleThreads;
    const uint32_t wide = rcclDdaIpcLLWideBlocks(comm, ncclFuncAllGather, perRankBytes * (size_t)comm->nRanks);
    const unsigned rows = nccl_dda_detail::ddaIpcLLPeerRows(comm, perRankBytes, threads, wide);
    return std::make_pair(dim3((unsigned)comm->nRanks, rows), dim3(threads));
  }
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

  // The LL region of either handler, under the epoch cells every LL tier shares.
  const nccl_dda_detail::DdaLLRegion region = nccl_dda_detail::ddaLLRegion(comm);
  T** peers = reinterpret_cast<T**>(region.peers);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = region.bank;

  INFO(NCCL_COLL, "DDA fabric AllGather LL%s: nRanks=%d perRankBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%u)",
       nccl_dda_detail::ddaLLOnIpc(comm) ? " (IPC)" : "", nRanks, perRankBytes, grid.x, grid.y, block.x,
       blocksPerPeer);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
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
  // The fabric or the IPC LL region. Nothing below reads the buffers, so every rank decides alike.
  if (!nccl_dda_detail::ddaLLRegionReady(comm)) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }

  // Fabric: up to the LL cap. IPC: only in the wide window, the sizes its grid is set for.
  const size_t totalBytes = perRankBytes * (size_t)comm->nRanks;
  const bool inWindow = nccl_dda_detail::ddaLLOnIpc(comm) ?
                          rcclDdaIpcLLWideBlocks(comm, ncclFuncAllGather, totalBytes) > 0 :
                          totalBytes <= rcclDdaLLThreshold(comm, ncclFuncAllGather);
  if (!inWindow) {
    return false;
  }
  // One packet carries 8B of payload, so this rank's contribution has to fit the
  // slot the bank gives each rank. A bank too small for one packet per rank takes
  // the tier out entirely.
  if ((perRankBytes >> 3) > ddaLLAgSlotPkts(nccl_dda_detail::ddaLLRegion(comm).bank, comm->nRanks)) {
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabric.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLLTyped<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
