/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Tuned block size, and the fallback when the env override is out of range.
constexpr unsigned kDdaLL128AGDefaultThreads = 512;

RCCL_PARAM(DdaLL128AGThreads, "DDA_LL128_AG_THREADS", kDdaLL128AGDefaultThreads);

namespace {

using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireBytesPerSlice;
using dda::common::kDdaLL128WireWordsPerSlice;

// Slot geometry is derived from the scratch allocation.
// Scratch holds 2 banks of nRanks slots
constexpr size_t ddaLL128AGSlotSlices(int nRanks, size_t scratchBytes) {
  return scratchBytes / ((size_t)2 * (size_t)nRanks * (size_t)kDdaLL128WireBytesPerSlice);
}

// Payload the slot carries.
constexpr size_t ddaLL128AGMaxPerRankBytes(int nRanks, size_t scratchBytes) {
  return ddaLL128AGSlotSlices(nRanks, scratchBytes) * (size_t)kDdaLL128DataBytesPerSlice;
}

unsigned ddaLL128AGThreads(unsigned blockSize) {
  const int64_t UserInput = rcclParamDdaLL128AGThreads();
  if (UserInput >= kDdaLL128Warp && UserInput <= 1024 && (UserInput % kDdaLL128Warp) == 0) {
    return (unsigned)UserInput;
  }
  return blockSize;
}

// Blocks in one peer column: a warp per slice, capped by the column's share of
// the grid budget.
unsigned ddaLL128AGBlocksPerPeer(size_t slices, size_t warps, int nPeers, size_t totalBlockCap) {
  const size_t cap = totalBlockCap / (size_t)nPeers;
  size_t blocksPerPeer = (slices + warps - 1) / warps;
  if (blocksPerPeer > cap) {
    blocksPerPeer = cap;
  }
  return blocksPerPeer < 1 ? 1u : (unsigned)blocksPerPeer;
}

// Single source of the launch geometry: grid.x = one column per remote peer,
// grid.y = the per-peer slice split.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes) {
  const unsigned threads = ddaLL128AGThreads(kDdaLL128AGDefaultThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  const unsigned blocksPerPeer =
    ddaLL128AGBlocksPerPeer(ddaLL128Slices(perRankBytes), warps, nPeers, (size_t)nBlocksMax);
  return std::make_pair(dim3((unsigned)nPeers, blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLL128Typed(
  const void* sendbuff, void* recvbuff,
  size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);
  const size_t slices = ddaLL128Slices(perRankBytes);
  // Slot stride in 8B words.
  const size_t slotWords =
    ddaLL128AGSlotSlices(nRanks, comm->ddaScratchBytes) * (size_t)kDdaLL128WireWordsPerSlice;

  auto gridBlock = ddaAllGatherFabricLL128Geom(comm, perRankBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const unsigned blocksPerPeer = grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-slice, bpp=%u, slotWords=%zu)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, blocksPerPeer, slotWords);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllGatherFabricLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  case 8:
    dda::common::ddaAllGatherFabricLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  default:
    dda::common::ddaAllGatherFabricLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen, slices, slotWords);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (!rcclParamDdaLL()) {
    return false;
  }
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
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
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  if (perRankBytes * (size_t)comm->nRanks > (size_t)rcclParamDdaLL128Threshold()) {
    return false;
  }
  // Derived from the scratch allocation
  if (perRankBytes > ddaLL128AGMaxPerRankBytes(comm->nRanks, comm->ddaScratchBytes)) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaFabricLL128Blocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricLL128Geom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabricLL.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
