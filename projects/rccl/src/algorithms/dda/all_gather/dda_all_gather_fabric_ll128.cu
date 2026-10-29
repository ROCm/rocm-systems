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

// Tuned block size, and the fallback when the env override is out of range.
constexpr unsigned kDdaLL128AGDefaultThreads = 512;
// Grid budget for this tier, applied under RCCL_DDA_FABRIC_MAXBLOCKS to messages
// above the size below; 0 or less leaves only the fabric-wide cap.
constexpr int kDdaLL128AGDefaultMaxBlocks = 75;
// Total message size above which the budget applies. At 2 MiB and below it only
// binds across hosts, where it costs more time than the CUs it saves.
constexpr size_t kDdaLL128AGMaxBlocksAboveBytes = (size_t)2 * 1024 * 1024;
// Peers one block serves. 0 or less, or the peer count and above, has every block
// serve every peer, so the grid size does not depend on the peer count; 1 gives every
// peer its own column of blocks.
constexpr int kDdaLL128AGDefaultPeersPerBlock = 0;

RCCL_PARAM(DdaLL128AGThreads, "DDA_LL128_AG_THREADS", kDdaLL128AGDefaultThreads);
RCCL_PARAM(DdaLL128AGMaxBlocks, "DDA_LL128_AG_MAXBLOCKS", kDdaLL128AGDefaultMaxBlocks);
RCCL_PARAM(DdaLL128AGMaxBlocksAboveBytes, "DDA_LL128_AG_MAXBLOCKS_ABOVE_BYTES",
           (int64_t)kDdaLL128AGMaxBlocksAboveBytes);
RCCL_PARAM(DdaLL128AGPeersPerBlock, "DDA_LL128_AG_PEERS_PER_BLOCK", kDdaLL128AGDefaultPeersPerBlock);

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

size_t ddaLL128AGMaxBlocksAboveBytes() {
  const int64_t UserInput = rcclParamDdaLL128AGMaxBlocksAboveBytes();
  return UserInput >= 0 ? (size_t)UserInput : kDdaLL128AGMaxBlocksAboveBytes;
}

int ddaLL128AGPeersPerBlock(int nPeers) {
  const int64_t UserInput = rcclParamDdaLL128AGPeersPerBlock();
  return (UserInput <= 0 || UserInput >= nPeers) ? nPeers : (int)UserInput;
}
// Blocks in one group column: a warp per (slice, peer) item, capped by the column's
// share of the grid budget.
unsigned ddaLL128AGBlocksPerGroup(size_t items, size_t warps, int nGroups, size_t totalBlockCap) {
  const size_t cap = totalBlockCap / (size_t)nGroups;
  size_t blocksPerGroup = (items + warps - 1) / warps;
  if (blocksPerGroup > cap) {
    blocksPerGroup = cap;
  }
  return blocksPerGroup < 1 ? 1u : (unsigned)blocksPerGroup;
}

// Single source of the launch geometry: grid.x = one column per group of
// peersPerBlock peers, grid.y = the group's item split, capped by the tier's grid
// budget.
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes,
                                                                int peersPerBlock) {
  const size_t totalBytes = perRankBytes * (size_t)comm->nRanks;
  const unsigned threads = ddaLL128AGThreads(kDdaLL128AGDefaultThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  const int nGroups = (nPeers + peersPerBlock - 1) / peersPerBlock;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  const int64_t agMaxBlocks = rcclParamDdaLL128AGMaxBlocks();
  if (totalBytes > ddaLL128AGMaxBlocksAboveBytes() && agMaxBlocks > 0 && agMaxBlocks < nBlocksMax) {
    nBlocksMax = (int)agMaxBlocks;
  }
  if (nBlocksMax < 1) nBlocksMax = 1;
  if (nBlocksMax < nGroups) nBlocksMax = nGroups;
  const size_t items = ddaLL128Slices(perRankBytes) * (size_t)peersPerBlock;
  const unsigned blocksPerGroup = ddaLL128AGBlocksPerGroup(items, warps, nGroups, (size_t)nBlocksMax);
  return std::make_pair(dim3((unsigned)nGroups, blocksPerGroup), dim3(threads));
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

  const int peersPerBlock = ddaLL128AGPeersPerBlock(nRanks - 1);
  auto gridBlock = ddaAllGatherFabricLL128Geom(comm, perRankBytes, peersPerBlock);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const unsigned blocksPerGroup = grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-item, peersPerBlock=%d, bpg=%u, slotWords=%zu)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, peersPerBlock, blocksPerGroup,
       ddaLL128AgSlotWords(bankSize, nRanks));

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL((dda::common::ddaAllGatherFabricLL128<T>), grid, block, 0, stream, /*startEvent=*/nullptr,
                        stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                        perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, peersPerBlock,
                        params);

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
  const auto grid =
    ddaAllGatherFabricLL128Geom(comm, sendcount * ncclTypeSize(datatype), ddaLL128AGPeersPerBlock(comm->nRanks - 1))
      .first;
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
