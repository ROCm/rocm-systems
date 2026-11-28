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

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128AgSlotWords;
using dda::common::ddaLL128BlocksPerPeer;
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

  // Alignment differs between ranks, so it only picks this rank's user-buffer access, never the wire.
  const bool aligned = dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff);
  INFO(NCCL_COLL,
       "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu grid=%ux%u block=%u "
       "(warp-per-slice, bpp=%u, slotWords=%zu, aligned=%d)",
       nRanks, perRankBytes, slices, grid.x, grid.y, block.x, blocksPerPeer, ddaLL128AgSlotWords(bankSize, nRanks),
       (int)aligned);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  const auto launch = [&](auto kernel) {
    hipExtLaunchKernelGGL(kernel, grid, block, 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers,
                          static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes, comm->rank, nRanks,
                          epochDev, epochLen, slices, bankSize, params);
  };
  dda::common::ddaPickAligned(aligned, [&](auto a) {
    constexpr bool kAligned = decltype(a)::value;
    // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
    switch (nRanks) {
    case 4:
      launch(dda::common::ddaAllGatherFabricLL128<T, 4, kAligned>);
      break;
    case 8:
      launch(dda::common::ddaAllGatherFabricLL128<T, 8, kAligned>);
      break;
    default:
      launch(dda::common::ddaAllGatherFabricLL128<T, 0, kAligned>);
      break;
    }
  });

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
  // Buffer alignment is not checked: it differs between ranks, and every rank has to pick the same tier.
  if (perRankBytes % 16 != 0) {
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
