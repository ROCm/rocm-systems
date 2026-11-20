/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-to-all
 * (register-packed kernel ddaAllToAllFabricLL128, alltoall_dda_fabric_ll128.h).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/alltoall/alltoall_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib> // getenv (grid-shape tuning overrides)
#include <utility>

constexpr unsigned kDdaLL128A2ADefaultThreads = 256;

// Threads per block, overriding the arch table: a multiple of 32 in [32, 1024], else ignored.
RCCL_PARAM(DdaLL128A2AThreads, "DDA_LL128_A2A_THREADS", 0);

namespace {

using dda::common::kDdaLL128Warp;
using dda::common::ddaBankSize;
using dda::common::ddaLL128A2ASlotWords;
using dda::common::ddaLL128Slices;
using dda::common::ddaLL128SlotPayloadBytes;
using dda::common::ddaLL128ValidThreads;
using dda::common::kDdaLL128DataElems;

static inline size_t ddaLL128A2ALinesPerBlockEnv(size_t dflt) {
  const char* s = getenv("RCCL_DDA_LL128_A2A_LPB");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (size_t)v;
  }
  return dflt;
}
static inline int ddaLL128A2AMaxBppEnv(int dflt) {
  const char* s = getenv("RCCL_DDA_LL128_A2A_MAXBPP");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (int)v;
  }
  return dflt;
}
static inline unsigned ddaLL128A2AThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128A2AThreads();
  return ddaLL128ValidThreads(v) ? (unsigned)v : dflt;
}

// Largest per-peer chunk the slot (carved from one scratch bank) can carry.
static inline size_t ddaLL128A2AMaxPerChunkBytes(ncclComm* comm) {
  return ddaLL128SlotPayloadBytes(ddaLL128A2ASlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks));
}

// Geometry: grid.x = peer columns, grid.y = blocks of a warp per slice, all within the block
// budget and the epoch array.
static inline std::pair<dim3, dim3> ddaAllToAllFabricLL128Geom(ncclComm* comm, size_t perChunkBytes) {
  const size_t totalBytes = perChunkBytes * (size_t)comm->nRanks;
  const int tuned = rcclDdaFabricA2aLL128Threads(comm, totalBytes);
  const unsigned threads = ddaLL128A2AThreads(tuned > 0 ? (unsigned)tuned : kDdaLL128A2ADefaultThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  const int cap = std::min(rcclDdaFabricA2aLL128Blocks(comm, totalBytes), comm->ddaLLEpochLen);
  size_t blocksWanted = (ddaLL128Slices(perChunkBytes) + warps - 1) / warps;
  // A warp moves a whole 16-line slice, so fewer than 16 lines per warp only adds idle blocks.
  const size_t linesPerBlock = ddaLL128A2ALinesPerBlockEnv(0);
  if (linesPerBlock > 0) {
    const size_t nWords = perChunkBytes >> 3;
    const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
    blocksWanted = (numLines + linesPerBlock - 1) / linesPerBlock;
  }
  const int maxBpp = ddaLL128A2AMaxBppEnv(0);
  if (maxBpp > 0) blocksWanted = std::min(blocksWanted, (size_t)maxBpp);
  const auto grid = dda::common::ddaPeerGrid(nPeers, blocksWanted, cap < 0 ? 0 : (size_t)cap);
  return std::make_pair(dim3(grid.cols, grid.blocksPerCol), dim3(threads));
}

static ncclResult_t ddaAllToAllFabricLL128Launch(const void* sendbuff, void* recvbuff, size_t perChunkBytes,
                                                 ncclComm* comm, cudaStream_t stream) {
  const size_t slices = ddaLL128Slices(perChunkBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
  const auto gb = ddaAllToAllFabricLL128Geom(comm, perChunkBytes);
  INFO(NCCL_COLL, "DDA fabric AllToAll LL128: nRanks=%d perChunkBytes=%zu slices=%zu grid=%ux%u block=%u", comm->nRanks,
       perChunkBytes, slices, gb.first.x, gb.first.y, gb.second.x);
  int8_t** peers = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL128<int8_t>), gb.first, gb.second, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<int8_t*>(recvbuff),
                        static_cast<const int8_t*>(sendbuff), perChunkBytes, comm->rank, comm->nRanks,
                        comm->ddaLLEpochDev, comm->ddaLLEpochLen, slices, bankSize, params);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype) {
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
  if (perChunkBytes % 16 != 0) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  // No pointer check: each rank passes its own buffers, so a rank-local refusal would leave its
  // peers polling for a kernel it never runs. gfx1250's 16B global accesses take any alignment.
  (void)sendbuff;
  (void)recvbuff;
  return perChunkBytes <= ddaLL128A2AMaxPerChunkBytes(comm);
}

uint32_t ncclAllToAllDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllToAllFabricLL128Geom(comm, count * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  return ddaAllToAllFabricLL128Launch(sendbuff, recvbuff, count * ncclTypeSize(datatype), comm, stream);
}
