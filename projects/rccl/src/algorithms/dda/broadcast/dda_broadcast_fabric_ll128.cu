/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA broadcast.
 * Only the two-shot (scatter-allgather) variant is implemented for now.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_twoshot.h"
#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_dpx_twoshot.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

constexpr unsigned kDdaLL128BcDefaultThreads = 512;

RCCL_PARAM(DdaLL128BcThreads, "DDA_LL128_BC_THREADS", kDdaLL128BcDefaultThreads);
RCCL_PARAM(DdaLL128BcFanOut, "DDA_LL128_BC_FANOUT", 0);
RCCL_PARAM(DdaDpxWarpsPerPeer, "DDA_DPX_WARPS_PER_PEER", 2);
RCCL_PARAM(DdaDpxScaleInPct, "DDA_DPX_SCALE_IN_PCT", 25);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128BcTwoShotChunkSlices;
using dda::common::ddaLL128BcTwoShotSlotSlices;
using dda::common::ddaLL128BcDpxChunkSlices;
using dda::common::ddaLL128BcDpxSlotSlices;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaMaxNranks;

unsigned ddaLL128BcThreads() {
  const int64_t u = rcclParamDdaLL128BcThreads();
  if (u >= kDdaLL128Warp && u <= kDdaBcMaxThreads && (u % kDdaLL128Warp) == 0)
    return (unsigned)u;
  return kDdaLL128BcDefaultThreads;
}

unsigned ddaLL128BcBlocksPerPeer(size_t chunkSlices, size_t warps, int nPeers, size_t totalBlockCap) {
  const size_t cap = totalBlockCap / (size_t)nPeers;
  size_t bpp = (chunkSlices + warps - 1) / warps;
  if (bpp > cap) bpp = cap;
  return bpp < 1 ? 1u : (unsigned)bpp;
}

using dda::common::DpxWsGeom;
using dda::common::ddaDpxWsGeom;

struct DpxLaunchGeom {
  dim3 grid;
  dim3 block;
  DpxWsGeom ws;
};

static inline DpxLaunchGeom ddaBroadcastFabricLL128DpxGeom(ncclComm* comm, size_t totalBytes) {
  const int archThreadsRaw = rcclDdaFabricBcLL128Threads(comm);
  const unsigned archThreads = archThreadsRaw > 0 ? (unsigned)archThreadsRaw : 0;
  const bool envSet = (getenv("RCCL_DDA_LL128_BC_THREADS") != nullptr);
  const unsigned threads = envSet ? ddaLL128BcThreads()
                         : (archThreads > 0 ? archThreads : kDdaLL128BcDefaultThreads);
  const int nwarps   = (int)(threads / (unsigned)kDdaLL128Warp);
  const int nNonRoot = comm->nRanks - 2;
  const int budget   = (int)ncclBroadcastDdaBlockBudget(comm);

  int wpp = (int)rcclParamDdaDpxWarpsPerPeer();
  if (wpp < 1) wpp = 1;
  if (wpp >= nwarps) wpp = nwarps - 1;
  int pct = (int)rcclParamDdaDpxScaleInPct();
  if (pct < 1) pct = 1;
  if (pct > 99) pct = 99;

  // Same-parity non-root peer count (from root's perspective; conservative estimate)
  const int nSameParityPeers = (nNonRoot + 1) / 2;

  DpxWsGeom ws = ddaDpxWsGeom(nwarps, wpp, pct, nSameParityPeers, budget);
  dim3 grid((unsigned)ws.gridWidth, (unsigned)ws.bpp);
  dim3 block(threads);
  return {grid, block, ws};
}

static inline std::pair<dim3, dim3> ddaBroadcastFabricLL128Geom(ncclComm* comm, size_t totalBytes) {
  // Two-shot uses the arch table thread count if set, else falls back to the env var / default.
  const int archThreadsRaw = rcclDdaFabricBcLL128Threads(comm);
  const unsigned archThreads = archThreadsRaw > 0 ? (unsigned)archThreadsRaw : 0;
  // Env var overrides arch table; arch table overrides default.
  const bool envSet = (getenv("RCCL_DDA_LL128_BC_THREADS") != nullptr);
  const unsigned threads = envSet ? ddaLL128BcThreads()
                         : (archThreads > 0 ? archThreads : kDdaLL128BcDefaultThreads);
  const size_t warps = threads / (unsigned)kDdaLL128Warp;
  const int nPeers = comm->nRanks - 1;
  int nBlocksMax = (int)ncclBroadcastDdaBlockBudget(comm);

  const size_t slices = ddaLL128Slices(totalBytes);
  const size_t cs = ddaLL128BcTwoShotChunkSlices(slices, comm->nRanks);
  const unsigned bpp = ddaLL128BcBlocksPerPeer(cs, warps, nPeers, (size_t)nBlocksMax);
  return std::make_pair(dim3((unsigned)nPeers, bpp), dim3(threads));
}

} // namespace

bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr)
    return false;
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr)
    return false;
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1)
    return false;
  if (count == 0)
    return false;
  if (comm->nRanks < 2 || comm->nRanks > (int)kDdaMaxNranks)
    return false;
  if (!rcclDdaDatatypeSupported(comm, datatype))
    return false;

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes % 16 != 0)
    return false;

  // LL128 cap check handled by the selector before calling us.
  // Scratch capacity: the bank must hold (nRanks + 1) chunk-sized slots.
  const size_t bank = ddaBankSize(comm->ddaScratchBytes);
  const size_t slices = ddaLL128Slices(bytes);
  if (ddaLL128BcTwoShotSlotSlices(bank, comm->nRanks, slices) == 0)
    return false;

  return true;
}

bool ncclBroadcastDdaFabricLL128DpxEligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (!ncclBroadcastDdaFabricLL128Eligible(comm, count, datatype))
    return false;
  if (comm->ddaFabricScaleInSize != 2)
    return false;
  if (comm->nRanks < 4 || comm->nRanks % 2 != 0)
    return false;

  const size_t bytes  = count * ncclTypeSize(datatype);
  const size_t bank   = ddaBankSize(comm->ddaScratchBytes);
  const size_t slices = ddaLL128Slices(bytes);
  if (ddaLL128BcDpxSlotSlices(bank, comm->nRanks, slices) == 0)
    return false;

  return true;
}

uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const auto grid = ddaBroadcastFabricLL128Geom(comm, bytes).first;
  return grid.x * grid.y;
}

static ncclResult_t ncclBroadcastDdaFabricLL128DpxLaunch(
    const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes  = count * ncclTypeSize(datatype);
  const size_t slices = ddaLL128Slices(bytes);
  const size_t cs     = ddaLL128BcDpxChunkSlices(slices, comm->nRanks);
  const size_t bank   = ddaBankSize(comm->ddaScratchBytes);

  auto geom = ddaBroadcastFabricLL128DpxGeom(comm, bytes);
  const auto& grid = geom.grid;
  const auto& block = geom.block;
  const auto& ws = geom.ws;

  int8_t** peers     = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const int nRanks   = comm->nRanks;

  int wpp = (int)rcclParamDdaDpxWarpsPerPeer();
  if (wpp < 1) wpp = 1;

  INFO(NCCL_COLL,
       "DDA fabric Broadcast LL128 DPX WS two-shot: nRanks=%d root=%d bytes=%zu slices=%zu "
       "chunkSlices=%zu grid=%ux%u block=%u bankSize=%zu scaleInSize=%d "
       "warpsPerPeer=%d suWarps=%d siWarps=%d peersPerBlock=%d",
       nRanks, root, bytes, slices, cs, grid.x, grid.y, block.x, bank,
       comm->ddaFabricScaleInSize, wpp, ws.suWarps, ws.siWarps, ws.peersPerBlock);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);

  hipExtLaunchKernelGGL(
    (dda::common::ddaBroadcastFabricLL128DpxTwoShot<0>),
    grid, block, 0, stream,
    /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
    peers,
    static_cast<int8_t*>(recvbuff),
    static_cast<const int8_t*>(sendbuff),
    bytes,
    comm->rank, nRanks, root,
    epochDev, epochLen,
    slices, cs, bank, params,
    wpp, ws.suWarps, ws.siWarps, ws.peersPerBlock);

  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream) {
  if (!rcclDdaDatatypeSupported(comm, datatype))
    return ncclInvalidArgument;

  // DPX (partition mode): use the 4-phase hierarchical kernel.
  if (ncclBroadcastDdaFabricLL128DpxEligible(comm, count, datatype))
    return ncclBroadcastDdaFabricLL128DpxLaunch(sendbuff, recvbuff, count, datatype, root, comm, stream);

  const size_t bytes = count * ncclTypeSize(datatype);
  const size_t slices = ddaLL128Slices(bytes);
  const size_t cs = ddaLL128BcTwoShotChunkSlices(slices, comm->nRanks);
  const size_t bank = ddaBankSize(comm->ddaScratchBytes);

  auto [grid, block] = ddaBroadcastFabricLL128Geom(comm, bytes);

  int8_t** peers = reinterpret_cast<int8_t**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const int nRanks = comm->nRanks;

  INFO(NCCL_COLL,
       "DDA fabric Broadcast LL128 two-shot: nRanks=%d root=%d bytes=%zu slices=%zu chunkSlices=%zu "
       "grid=%ux%u block=%u bankSize=%zu fanOut=%d",
       nRanks, root, bytes, slices, cs, grid.x, grid.y, block.x, bank, (int)rcclParamDdaLL128BcFanOut());

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);

  hipExtLaunchKernelGGL(
    (dda::common::ddaBroadcastFabricLL128TwoShot<0>),
    grid, block, 0, stream,
    /*startEvent=*/nullptr, stopEvent, /*flags=*/0,
    peers,
    static_cast<int8_t*>(recvbuff),
    static_cast<const int8_t*>(sendbuff),
    bytes,
    comm->rank, nRanks, root,
    epochDev, epochLen,
    slices, cs, bank, params, (int)rcclParamDdaLL128BcFanOut(), comm->ddaFabricScaleInSize);

  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
