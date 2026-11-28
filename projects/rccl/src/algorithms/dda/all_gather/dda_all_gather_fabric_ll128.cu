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
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Block size when neither a valid RCCL_DDA_LL128_AG_THREADS nor the arch table sets one.
constexpr unsigned kDdaLL128AGDefaultThreads = 512;

RCCL_PARAM(DdaLL128AGThreads, "DDA_LL128_AG_THREADS", 0);
// Stages phase 1 from this many pushed payload bytes a block; <= 0 is off. Needs gfx1250 wave32 and
// kDdaLL128AGStageThreads.
RCCL_PARAM(DdaLL128AGAsyncWireMinBytes, "DDA_LL128_AG_ASYNC_WIRE_MIN_BYTES", 80 * 1024);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128AgSlotWords;
using dda::common::ddaLL128BlocksPerPeer;
using dda::common::ddaLL128Slices;
using dda::common::ddaLL128StageBytes;
using dda::common::kDdaLL128AGStageThreads;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// Slices one per-rank slot holds, derived from the scratch bank the host picked.
constexpr size_t ddaLL128AGSlotSlices(int nRanks, size_t scratchBytes) {
  return ddaLL128AgSlotWords(ddaBankSize(scratchBytes), nRanks) / (size_t)kDdaLL128WireWordsPerSlice;
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


// LDS a block gets without raising its limit; the staged windows must fit it.
constexpr size_t kDdaDefaultLdsBytes = 64 * 1024;
static_assert(ddaLL128StageBytes(kDdaLL128AGStageThreads) <= kDdaDefaultLdsBytes, "staged windows exceed the LDS");

// Payload bytes one block of `grid` pushes in phase 1: perRankBytes to each of nPeers peers, over every block.
size_t ddaLL128AGBytesPerBlock(size_t perRankBytes, int nPeers, const dim3& grid) {
  return perRankBytes * (size_t)nPeers / ((size_t)grid.x * grid.y);
}

// Whether a launch takes the staged phase 1. Every input matches on all ranks of a DDA comm, so every rank agrees.
bool ddaLL128AGUseAsyncWire(const ncclComm* comm, size_t bytesPerBlock, unsigned threads) {
  const int64_t minBytes = rcclParamDdaLL128AGAsyncWireMinBytes();
  if (minBytes <= 0 || bytesPerBlock < (size_t)minBytes) {
    return false;
  }
  return dda::common::ddaLL128StagedWireOk(comm->archName, comm->WarpSize) && threads == kDdaLL128AGStageThreads;
}

// Single source of the launch geometry: grid.x = one column per remote peer,
// grid.y = the per-peer slice split.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes) {
  const size_t totalBytes = perRankBytes * (size_t)comm->nRanks;
  const int tuned = rcclDdaFabricAgLL128Threads(comm, totalBytes);
  const unsigned threads = ddaLL128AGThreads(tuned > 0 ? (unsigned)tuned : kDdaLL128AGDefaultThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  int nBlocksMax = rcclDdaFabricAgLL128Blocks(comm, totalBytes);
  if (nBlocksMax < 1) nBlocksMax = 1;
  if (nBlocksMax < nPeers) nBlocksMax = nPeers;
  const unsigned blocksPerPeer =
    ddaLL128BlocksPerPeer(ddaLL128Slices(perRankBytes), warps, nPeers, (size_t)nBlocksMax);
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
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  auto gridBlock = ddaAllGatherFabricLL128Geom(comm, perRankBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const unsigned blocksPerPeer = grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  const size_t bytesPerBlock = ddaLL128AGBytesPerBlock(perRankBytes, nRanks - 1, grid);
  const bool asyncWire = ddaLL128AGUseAsyncWire(comm, bytesPerBlock, block.x);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (asyncWire) {
    // Same grid, block and arguments; the staged phase 1 adds the windows' LDS and the peers a fill goes to.
    const int pushPeers = dda::common::kDdaLL128AGStagePushPeers;
    const size_t lds = ddaLL128StageBytes(block.x);
    INFO(NCCL_COLL,
         "DDA fabric AllGather LL128 async wire: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
         "bytesPerBlock=%zu pushPeers=%d groups=%d lds=%zu",
         nRanks, perRankBytes, slices, grid.x, grid.y, block.x, bytesPerBlock, pushPeers,
         dda::common::ddaLL128AGStageGroups(nRanks - 1, pushPeers), lds);
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128Staged<T>), grid, block, lds, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), perRankBytes, comm->rank, nRanks, epochDev, epochLen,
                          slices, bankSize, params, pushPeers);
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
  }
  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-slice, bpp=%u, slotWords=%zu)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, blocksPerPeer, ddaLL128AgSlotWords(bankSize, nRanks));
  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }
  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  if (perRankBytes * (size_t)comm->nRanks > rcclDdaLL128Threshold(comm, ncclFuncAllGather)) {
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
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabricLL.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
