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

// Tuned block sizes by total message size: 512 threads up to
// kDdaLL128AGWideAboveBytes, 1024 above it. A valid RCCL_DDA_LL128_AG_THREADS
// overrides both.
constexpr unsigned kDdaLL128AGThreads = 512;
constexpr unsigned kDdaLL128AGWideThreads = 1024;
constexpr size_t kDdaLL128AGWideAboveBytes = (size_t)4 * 1024 * 1024;
// Slices each warp keeps in flight; the kernel is only instantiated for 1, 2 and 4.
constexpr int kDdaLL128AGDefaultUnroll = 4;
// Grid budget for this tier, applied under RCCL_DDA_FABRIC_MAXBLOCKS to messages
// above the size below; 0 or less leaves only the fabric-wide cap.
constexpr int kDdaLL128AGDefaultMaxBlocks = 64;
// Total message size above which the budget applies. Smaller messages leave each
// warp a slice or two, and the extra slice the budget adds costs more time than
// the CUs it saves; across hosts that holds up to a larger size.
constexpr size_t kDdaLL128AGMaxBlocksAboveBytes = (size_t)4 * 1024 * 1024;
constexpr size_t kDdaLL128AGMaxBlocksAboveBytesMultiHost = (size_t)8 * 1024 * 1024;

RCCL_PARAM(DdaLL128AGThreads, "DDA_LL128_AG_THREADS", 0);
RCCL_PARAM(DdaLL128AGUnroll, "DDA_LL128_AG_UNROLL", kDdaLL128AGDefaultUnroll);
RCCL_PARAM(DdaLL128AGMaxBlocks, "DDA_LL128_AG_MAXBLOCKS", kDdaLL128AGDefaultMaxBlocks);
// -1 picks the size above by whether the ranks span hosts.
RCCL_PARAM(DdaLL128AGMaxBlocksAboveBytes, "DDA_LL128_AG_MAXBLOCKS_ABOVE_BYTES", -1);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128AgSlotWords;
using dda::common::ddaLL128Slices;
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

int ddaLL128AGUnroll() {
  const int64_t UserInput = rcclParamDdaLL128AGUnroll();
  if (UserInput == 1 || UserInput == 2 || UserInput == 4) {
    return (int)UserInput;
  }
  return kDdaLL128AGDefaultUnroll;
}

// MNNVL folds a multi-host gfx1250 clique into one node (comm->nNodes == 1), so
// hosts are told apart by host hash.
bool ddaLL128AGSpansHosts(const ncclComm* comm) {
  const uint64_t selfHost = comm->peerInfo[comm->rank].hostHash;
  for (int r = 0; r < comm->nRanks; ++r) {
    if (comm->peerInfo[r].hostHash != selfHost) return true;
  }
  return false;
}

size_t ddaLL128AGMaxBlocksAboveBytes(const ncclComm* comm) {
  const int64_t UserInput = rcclParamDdaLL128AGMaxBlocksAboveBytes();
  if (UserInput >= 0) {
    return (size_t)UserInput;
  }
  return ddaLL128AGSpansHosts(comm) ? kDdaLL128AGMaxBlocksAboveBytesMultiHost : kDdaLL128AGMaxBlocksAboveBytes;
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
// grid.y = the per-peer slice split, capped by the tier's grid budget.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes) {
  const size_t totalBytes = perRankBytes * (size_t)comm->nRanks;
  const unsigned threads =
    ddaLL128AGThreads(totalBytes > kDdaLL128AGWideAboveBytes ? kDdaLL128AGWideThreads : kDdaLL128AGThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  const int64_t agMaxBlocks = rcclParamDdaLL128AGMaxBlocks();
  if (totalBytes > ddaLL128AGMaxBlocksAboveBytes(comm) && agMaxBlocks > 0 && agMaxBlocks < nBlocksMax) {
    nBlocksMax = (int)agMaxBlocks;
  }
  if (nBlocksMax < 1) nBlocksMax = 1;
  if (nBlocksMax < nPeers) nBlocksMax = nPeers;
  const unsigned blocksPerPeer =
    ddaLL128AGBlocksPerPeer(ddaLL128Slices(perRankBytes), warps, nPeers, (size_t)nBlocksMax);
  return std::make_pair(dim3((unsigned)nPeers, blocksPerPeer), dim3(threads));
}

template <typename T, int UNROLL>
static void ddaAllGatherFabricLL128Launch(ncclComm* comm, dim3 grid, dim3 block, cudaStream_t stream,
                                          hipEvent_t stopEvent, T* recvbuff, const T* sendbuff, size_t perRankBytes,
                                          size_t slices, size_t bankSize) {
  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Always the runtime rank count: nothing in the kernel loops over ranks, and the
  // 4/8-rank variants need more registers than a 1024-thread block allows, so they
  // spill at UNROLL 4.
  hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<T, 0, UNROLL>), grid, block, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, recvbuff, sendbuff, perRankBytes,
                        comm->rank, comm->nRanks, comm->ddaLLEpochDev, comm->ddaLLEpochLen, slices, bankSize);
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
  const int unroll = ddaLL128AGUnroll();

  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-slice, bpp=%u, unroll=%d, slotWords=%zu)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, grid.y, unroll,
       ddaLL128AgSlotWords(bankSize, nRanks));

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  T* dst = static_cast<T*>(recvbuff);
  const T* src = static_cast<const T*>(sendbuff);
  switch (unroll) {
  case 4:
    ddaAllGatherFabricLL128Launch<T, 4>(comm, grid, block, stream, stopEvent, dst, src, perRankBytes, slices, bankSize);
    break;
  case 2:
    ddaAllGatherFabricLL128Launch<T, 2>(comm, grid, block, stream, stopEvent, dst, src, perRankBytes, slices, bankSize);
    break;
  default:
    ddaAllGatherFabricLL128Launch<T, 1>(comm, grid, block, stream, stopEvent, dst, src, perRankBytes, slices, bankSize);
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabricLL.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
