/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of dda_all_reduce_fabric_ll.cu. Launches the shard
 * kernel ddaReduceScatterFabricLL below kDdaLLRsSpreadMinRanks ranks and the
 * spread kernel ddaReduceScatterFabricSpreadLL from there.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/dda_ll_region.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll_spread.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

// Launch shape of whichever kernel the message takes.
struct DdaLLRsGeom {
  dim3 grid;
  dim3 block;
  bool spread;
};

// The grid ncclReduceScatterDdaFabricLLBlocks reports: a thread per packet of a shard, or of the message when spread.
// IPC, never spread: ddaIpcLLBlocks, on the table's wide grid inside its window.
static inline DdaLLRsGeom ddaReduceScatterFabricLLGeom(const ncclComm* comm, size_t shardBytes) {
  if (nccl_dda_detail::ddaLLOnIpc(comm)) {
    const unsigned threads = dda::common::kDdaSimpleThreads;
    const uint32_t wide = rcclDdaIpcLLWideBlocks(comm, ncclFuncReduceScatter, shardBytes * (size_t)comm->nRanks);
    return {dim3(nccl_dda_detail::ddaIpcLLBlocks(comm, shardBytes, threads, /*capBlocks=*/0, wide)), dim3(threads),
            false};
  }
  const bool spread = comm->nRanks >= dda::common::kDdaLLRsSpreadMinRanks;
  const size_t nPk = shardBytes >> 3; // 8 payload bytes per packet
  const unsigned threads = (unsigned)dda::common::kDdaLLRsSpreadThreads;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  const size_t gridPk = spread ? nPk * (size_t)comm->nRanks : nPk;
  unsigned blocks = (unsigned)std::min<size_t>((gridPk + threads - 1) / threads, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) blocks = 1;
  }
  // Spread slices round up to whole lines, which can leave the last blocks without packets, so those are not launched.
  if (spread && nPk > 0) {
    const size_t slice = dda::common::ddaLLRsSpreadSlicePkts(nPk, blocks);
    blocks = (unsigned)((nPk + slice - 1) / slice);
  }
  return {dim3(blocks), dim3(threads), spread};
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                      ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = recvcount * sizeof(T); // per-rank shard bytes
  const size_t nPk = bytes >> 3;              // 8 payload bytes per packet

  const DdaLLRsGeom geom = ddaReduceScatterFabricLLGeom(comm, bytes);
  const dim3 grid = geom.grid;
  const dim3 block = geom.block;

  // The LL region of either handler; bank 1 starts at its bank size, matching the other LL tiers there.
  const nccl_dda_detail::DdaLLRegion region = nccl_dda_detail::ddaLLRegion(comm);
  T** peers = reinterpret_cast<T**>(region.peers);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = region.bank;

  INFO(NCCL_COLL, "DDA fabric ReduceScatter LL: nRanks=%d shardBytes=%zu nPk=%zu grid=%u block=%u bankSize=%zu%s%s",
       nRanks, bytes, nPk, grid.x, block.x, bankSize, geom.spread ? " (spread)" : "",
       nccl_dda_detail::ddaLLOnIpc(comm) ? " (IPC)" : "");

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (geom.spread) {
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricSpreadLL<T, 0>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), recvcount, comm->rank, nRanks, epochDev, epochLen,
                          bankSize, params);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          recvcount, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          recvcount, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          recvcount, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
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
  // The fabric or the IPC LL region. Nothing below reads the buffers, so every rank decides alike.
  if (!nccl_dda_detail::ddaLLRegionReady(comm)) {
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
  // IPC: the shard fits the per-rank slot of the 4 MiB LL bank (dda::common::ddaLLRsSlotPkts).
  if (nccl_dda_detail::ddaLLOnIpc(comm)) {
    return (bytes >> 3) <= dda::common::ddaLLRsSlotPkts(nccl_dda_detail::ddaLLRegion(comm).bank, comm->nRanks);
  }
  // Bank 0 is [0, bankBytes). Bank 1 is [bankSize, bankSize + bankBytes).
  const size_t bankBytes = dda::common::ddaLLRsBankBytes(comm->nRanks);
  const size_t bankSize = dda::common::ddaBankSize(comm->ddaScratchBytes);
  if (bankBytes > bankSize || bankSize + bankBytes > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

uint32_t ncclReduceScatterDdaFabricLLBlocks(ncclComm* comm, size_t recvcount, ncclDataType_t datatype) {
  return ddaReduceScatterFabricLLGeom(comm, recvcount * ncclTypeSize(datatype)).grid.x;
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
