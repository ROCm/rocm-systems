/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-to-all
 * (register-packed kernel ddaAllToAllFabricLL128V2, alltoall_dda_fabric_ll128.h).
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

#include <cstddef>
#include <cstdint>
#include <cstdlib> // getenv (grid-shape override)
#include <utility>

// Runtime-adjustable LL128 AllToAll block size (threads/block). Must be a
// multiple of 32 (one warp per 2 KiB slice) in [32, 1024]; invalid values fall
// back to the tuned default. Env: RCCL_DDA_LL128_A2A_THREADS.
RCCL_PARAM(DdaLL128A2AThreads, "DDA_LL128_A2A_THREADS", 256);

namespace {

using dda::common::kDdaLL128Warp;
using dda::common::ddaBankSize;
using dda::common::ddaLL128Slices;
using dda::common::ddaLL128A2ASlotWords;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128DataBytesPerSlice;

static inline int ddaLL128A2AMaxBppEnv(int dflt) {
  const char* s = getenv("RCCL_DDA_LL128_A2A_MAXBPP");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (int)v;
  }
  return dflt;
}
// Validated block size from the runtime flag; falls back to `dflt`.
static inline unsigned ddaLL128A2AThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128A2AThreads();
  if (v >= 32 && v <= 1024 && (v % 32) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

// Largest per-peer chunk the slot (carved from one scratch bank) can carry.
static inline size_t ddaLL128A2AV2MaxPerChunkBytes(ncclComm* comm) {
  const size_t slotWords = ddaLL128A2ASlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  return slotWords / (size_t)kDdaLL128WireWordsPerSlice * (size_t)kDdaLL128DataBytesPerSlice;
}

// Geometry: grid.x = remote peers, grid.y = warp-per-slice split capped by the
// fabric block budget and the epoch array; threads multiple of 32.
static inline std::pair<dim3, dim3> ddaAllToAllFabricLL128V2Geom(ncclComm* comm, size_t perChunkBytes) {
  // 256 (8 warps) x 13 blocks/peer measured best on gfx1250 DPX: 1024-thread blocks put
  // too much in flight per CU (8M: 38 vs 25 us kernel), 384 already loses at 4M/8M.
  const unsigned threads = ddaLL128A2AThreads(256);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  int cap = comm->ddaFabricMaxBlocks;
  if (comm->ddaLLEpochLen < cap) cap = comm->ddaLLEpochLen;
  cap /= nPeers;
  const int envCap = ddaLL128A2AMaxBppEnv(0);
  if (envCap > 0 && envCap < cap) cap = envCap;
  if (cap < 1) cap = 1;
  size_t bpp = (ddaLL128Slices(perChunkBytes) + warps - 1) / warps;
  if (bpp > (size_t)cap) bpp = (size_t)cap;
  if (bpp < 1) bpp = 1;
  return std::make_pair(dim3((unsigned)nPeers, (unsigned)bpp), dim3(threads));
}

template <int NR>
static void ddaLL128A2AV2Launch(dim3 grid, dim3 block, cudaStream_t stream, hipEvent_t stopEvent, int8_t** peers,
                                void* recvbuff, const void* sendbuff, size_t perChunkBytes, ncclComm* comm,
                                size_t slices, size_t bankSize) {
  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricLL128V2<int8_t, NR, 1>), grid, block, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<int8_t*>(recvbuff),
                        static_cast<const int8_t*>(sendbuff), perChunkBytes, comm->rank, comm->nRanks,
                        comm->ddaLLEpochDev, comm->ddaLLEpochLen, slices, bankSize, params);
}

static ncclResult_t ncclAllToAllDdaFabricLL128V2(const void* sendbuff, void* recvbuff, size_t perChunkBytes,
                                                 ncclComm* comm, cudaStream_t stream) {
  const size_t slices = ddaLL128Slices(perChunkBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
  const auto gb = ddaAllToAllFabricLL128V2Geom(comm, perChunkBytes);
  INFO(NCCL_COLL, "DDA fabric AllToAll LL128 v2: nRanks=%d perChunkBytes=%zu slices=%zu grid=%ux%u block=%u",
       comm->nRanks, perChunkBytes, slices, gb.first.x, gb.first.y, gb.second.x);
  int8_t** peers = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  switch (comm->nRanks) {
  case 4:
    ddaLL128A2AV2Launch<4>(gb.first, gb.second, stream, stopEvent, peers, recvbuff, sendbuff, perChunkBytes, comm,
                           slices, bankSize);
    break;
  case 8:
    ddaLL128A2AV2Launch<8>(gb.first, gb.second, stream, stopEvent, peers, recvbuff, sendbuff, perChunkBytes, comm,
                           slices, bankSize);
    break;
  default:
    ddaLL128A2AV2Launch<0>(gb.first, gb.second, stream, stopEvent, peers, recvbuff, sendbuff, perChunkBytes, comm,
                           slices, bankSize);
    break;
  }
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
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < comm->nRanks - 1) {
    return false;
  }
  if (!dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff)) {
    return false;
  }
  return perChunkBytes <= ddaLL128A2AV2MaxPerChunkBytes(comm);
}

uint32_t ncclAllToAllDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllToAllFabricLL128V2Geom(comm, count * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return ncclInvalidArgument;
  }
  return ncclAllToAllDdaFabricLL128V2(sendbuff, recvbuff, count * ncclTypeSize(datatype), comm, stream);
}
