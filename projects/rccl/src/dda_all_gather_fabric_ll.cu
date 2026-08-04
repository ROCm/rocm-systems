/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_gather.h"

#include "algorithms/all_gather/all_gather_dda_fabric_ll.h"
#include "algorithms/all_gather/all_gather_dda_ll128.h"
#include "checks.h"
#include "comm.h"
#include "dda_init_detail.h" // kDdaLLMaxBlocks
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

namespace ll128 = meta::comms::ll128;

using meta::comms::ddaLLAgScratchSize;
using meta::comms::kDdaLLAgMaxPerRankBytes;
using nccl_dda_detail::kDdaLLMaxBlocks;

// LL runs one packet per thread.
constexpr unsigned kDdaLLAgThreads = 256;

int ddaLLAgBlocksPerPeer(
    size_t perRankBytes, int nRanks, size_t totalBlockCap) {
  const size_t nPk = perRankBytes >> 3; // 8 payload bytes per packet
  const size_t cap = totalBlockCap / (size_t)(nRanks < 1 ? 1 : nRanks);
  size_t bpp = (nPk + kDdaLLAgThreads - 1) / kDdaLLAgThreads;
  if (bpp > cap) {
    bpp = cap;
  }
  return bpp < 1 ? 1 : (int)bpp;
}

RCCL_PARAM(DdaAllGatherLL128Threads, "DDA_ALLGATHER_LL128_THREADS", 512);
RCCL_PARAM(
    DdaAllGatherLL128MaxBlocks,
    "DDA_ALLGATHER_LL128_MAXBLOCKS",
    kDdaLLMaxBlocks);

// Total blocks the grid may use
size_t ddaLL128AgBlockCap() {
  int64_t cap = rcclParamDdaAllGatherLL128MaxBlocks();
  if (cap < 1) {
    cap = 1;
  }
  if (cap > kDdaLLMaxBlocks) {
    cap = kDdaLLMaxBlocks;
  }
  return (size_t)cap;
}

static inline size_t ddaLL128AgWarpsPerBlock(int warpSize) {
  int64_t t = rcclParamDdaAllGatherLL128Threads();
  if (t < warpSize) {
    t = warpSize;
  }
  if (t > 1024) {
    t = 1024;
  }
  return (size_t)(t / warpSize);
}

// Blocks in one peer column, the LL128 counterpart of ddaLLAgBlocksPerPeer above:
// a warp per slice rather than a thread per packet, capped the same way by the
// column's share of the grid budget. Warps past the slice count own no slice.
unsigned ddaLL128AgBlocksPerPeer(
    size_t slices, size_t warpsPerBlock, int nRanks, size_t totalBlockCap) {
  const size_t warps = warpsPerBlock < 1 ? 1 : warpsPerBlock;
  const size_t cap = totalBlockCap / (size_t)(nRanks < 1 ? 1 : nRanks);
  size_t bpp = (slices + warps - 1) / warps;
  if (bpp > cap) {
    bpp = cap;
  }
  return bpp < 1 ? 1 : (unsigned)bpp;
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLLTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);

  // grid.x == nRanks (peer), grid.y == blocksPerPeer (packet split).
  const unsigned threads = kDdaLLAgThreads;
  const int blocksPerPeer =
      ddaLLAgBlocksPerPeer(perRankBytes, nRanks, kDdaLLMaxBlocks);
  dim3 block(threads);
  dim3 grid((unsigned)nRanks, (unsigned)blocksPerPeer);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(
      NCCL_COLL,
      "DDA fabric AllGather LL: nRanks=%d perRankBytes=%zu grid=%ux%u block=%u "
      "(block-per-peer, bpp=%d)",
      nRanks, perRankBytes, grid.x, grid.y, block.x, blocksPerPeer);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllGatherFabricLL<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    meta::comms::ddaAllGatherFabricLL<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    meta::comms::ddaAllGatherFabricLL<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLL128Typed(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);
  const int wireWordPerSlice = comm->WarpSize * ll128::kWordsPerThread;
  const int dataBytesPerSlice =
      (wireWordPerSlice - wireWordPerSlice / comm->ll128LineElems) * 8;
  const size_t slices =
      (perRankBytes + dataBytesPerSlice - 1) / dataBytesPerSlice;

  // One warp per slice, in a 2D grid of one block column per peer.
  const size_t warps = ddaLL128AgWarpsPerBlock(comm->WarpSize);
  dim3 block((unsigned)(warps * (size_t)comm->WarpSize));
  dim3 grid(
      (unsigned)nRanks,
      ddaLL128AgBlocksPerPeer(slices, warps, nRanks, ddaLL128AgBlockCap()));

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(
      NCCL_COLL,
      "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu slices=%zu "
      "grid=%ux%u block=%u wave=%d lineElems=%d wire=%dB data=%dB",
      nRanks, perRankBytes, slices, grid.x, grid.y, block.x, comm->WarpSize,
      comm->ll128LineElems, wireWordPerSlice * 8, dataBytesPerSlice);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllGatherLL128<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices,
        wireWordPerSlice, dataBytesPerSlice);
    break;
  case 8:
    meta::comms::ddaAllGatherLL128<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices,
        wireWordPerSlice, dataBytesPerSlice);
    break;
  default:
    meta::comms::ddaAllGatherLL128<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        perRankBytes, comm->rank, nRanks, epochDev, epochLen, slices,
        wireWordPerSlice, dataBytesPerSlice);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricLLEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if (perRankBytes > kDdaLLAgMaxPerRankBytes) {
    return false;
  }
  if (ddaLLAgScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllGatherDdaFabricLL(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabric.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLLTyped<int8_t>(
      sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}

bool ncclAllGatherDdaFabricLL128Eligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }
  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  // The LL128 line format packs 16B-aligned data with no straddling line, so
  // require a 16B multiple and 16B-aligned user buffers (assert alignment
  // rather than assume it).
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 ||
      (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  // Runtime slot stride: scratch must hold 2 banks * nRanks slots of whole slices
  // at this arch's wire expansion (16/15 on gfx1250, 8/7 on gfx9). The geometry
  // must match the launcher's exactly or this admits a launch that overruns
  // scratch. Falling back (returns false) when it doesn't fit lets the threshold
  // be raised independently of the scratch allocation.
  const int wireWordPerSlice = comm->WarpSize * ll128::kWordsPerThread;
  const int dataBytesPerSlice =
      (wireWordPerSlice - wireWordPerSlice / comm->ll128LineElems) * 8;
  const size_t slices =
      (perRankBytes + dataBytesPerSlice - 1) / dataBytesPerSlice;
  if ((size_t)2 * comm->nRanks * slices * wireWordPerSlice * sizeof(uint64_t) >
      comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllGatherDdaFabricLL128(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm* comm,
    cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // Pure copy: move the payload as raw bytes (int8_t), like the LL path.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(
      sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
