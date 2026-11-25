/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA broadcast.
 * Both the one-shot (direct scatter) and two-shot (scatter-allgather) variants.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_twoshot.h"
#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128.h"
#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_dpx_twoshot.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// Tuned block size, and the fallback when the env override is out of range.
constexpr unsigned kDdaLL128BcDefaultThreads = 512;
constexpr size_t kDdaLL128BcTwoShotMin = 256ULL * 1024;

RCCL_PARAM(DdaLL128BcThreads, "DDA_LL128_BC_THREADS", kDdaLL128BcDefaultThreads);
RCCL_PARAM(DdaLL128BcFanOut, "DDA_LL128_BC_FANOUT", 0);
RCCL_PARAM(DdaDpxWarpsPerPeer, "DDA_DPX_WARPS_PER_PEER", 2);
RCCL_PARAM(DdaDpxScaleInPct, "DDA_DPX_SCALE_IN_PCT", 25);
RCCL_PARAM(DdaBcForceDpx, "DDA_BC_FORCE_DPX", 0);

namespace {

using dda::common::ddaBankSize;
// Two-shot helpers
using dda::common::ddaLL128BcTwoShotChunkSlices;
using dda::common::ddaLL128BcTwoShotSlotSlices;
using dda::common::ddaLL128BlocksPerPeer;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaMaxNranks;
// One-shot helpers
using dda::common::ddaLL128BcSlotSlices;
using dda::common::kDdaLL128DataBytesPerSliceL;
using dda::common::kDdaLL128WireWordsPerSliceW;
// DPX helpers
using dda::common::ddaLL128BcDpxChunkSlices;
using dda::common::ddaLL128BcDpxSlotSlices;
using dda::common::DpxWsGeom;
using dda::common::ddaDpxWsGeom;

// Host mirror of kDdaLL128ArchLineElems / kDdaLL128ArchWarp.
struct DdaLL128BcArch {
  int lineBytes;
  size_t warp;
  size_t sliceWords;
  size_t sliceBytes; // payload
};

DdaLL128BcArch ddaLL128BcArch(const ncclComm* comm) {
  if (IsArchMatch(comm->archName, "gfx1250")) {
    return {128, 32, kDdaLL128WireWordsPerSliceW<32>, kDdaLL128DataBytesPerSliceL<16, 32>};
  }
  return {64, 64, kDdaLL128WireWordsPerSliceW<64>, kDdaLL128DataBytesPerSliceL<8, 64>};
}

// One-shot: validated against arch warp size.
unsigned ddaLL128BcThreads(const ncclComm* comm) {
  const int64_t userInput = rcclParamDdaLL128BcThreads();
  const int64_t warp = (int64_t)ddaLL128BcArch(comm).warp;
  if (userInput >= warp && userInput <= kDdaBcMaxThreads && userInput % warp == 0) {
    return (unsigned)userInput;
  }
  return kDdaLL128BcDefaultThreads;
}

// Two-shot: validated against fixed kDdaLL128Warp.
unsigned ddaLL128BcThreads() {
  const int64_t u = rcclParamDdaLL128BcThreads();
  if (u >= kDdaLL128Warp && u <= kDdaBcMaxThreads && (u % kDdaLL128Warp) == 0)
    return (unsigned)u;
  return kDdaLL128BcDefaultThreads;
}

size_t ddaLL128BcSlices(const ncclComm* comm, size_t bytes) {
  const size_t per = ddaLL128BcArch(comm).sliceBytes;
  return (bytes + per - 1) / per;
}

// Root: one column per peer (at least one block each, as AllGather LL128), the
// rest of the budget split across slice groups. Non-root: groups within the budget.
dim3 ddaLL128BcGrid(const ncclComm* comm, bool isRoot, size_t bytes) {
  const size_t warps = ddaLL128BcThreads(comm) / ddaLL128BcArch(comm).warp;
  const size_t groups = (ddaLL128BcSlices(comm, bytes) + warps - 1) / warps;
  const size_t budget = ncclBroadcastDdaBlockBudget(comm);
  if (isRoot) {
    const size_t peers = (size_t)(comm->nRanks - 1);
    const size_t perPeer = budget / peers < 1 ? 1 : budget / peers;
    return dim3((unsigned)peers, (unsigned)(groups < perPeer ? groups : perPeer));
  }
  return dim3((unsigned)(groups < budget ? groups : budget));
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
  const unsigned bpp = ddaLL128BlocksPerPeer(cs, warps, nPeers, (size_t)nBlocksMax);
  return std::make_pair(dim3((unsigned)nPeers, bpp), dim3(threads));
}


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

  const int nSameParityPeers = (nNonRoot + 1) / 2;

  DpxWsGeom ws = ddaDpxWsGeom(nwarps, wpp, pct, nSameParityPeers, budget);
  dim3 grid((unsigned)ws.gridWidth, (unsigned)ws.bpp);
  dim3 block(threads);
  return {grid, block, ws};
}

} // namespace

bool ncclBroadcastDdaFabricLL128DpxEligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (!ncclBroadcastDdaFabricLL128Eligible(comm, count, datatype))
    return false;
  if (!comm->hasMloPart)
    return false;
  if (comm->nRanks < 4 || comm->nRanks % 2 != 0)
    return false;

  const size_t bytes  = count * ncclTypeSize(datatype);
  if (bytes < kDdaLL128BcTwoShotMin)
    return false;
  const size_t bank   = ddaBankSize(comm->ddaScratchBytes);
  const size_t slices = ddaLL128Slices(bytes);
  if (ddaLL128BcDpxSlotSlices(bank, comm->nRanks, slices) == 0)
    return false;

  return true;
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
       comm->hasMloPart ? 2 : 1, wpp, ws.suWarps, ws.siWarps, ws.peersPerBlock);

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

bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->bootstrap == nullptr)
    return false;
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr)
    return false;
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1)
    return false;
  if (comm->nRanks < 2 || comm->nRanks > (int)kDdaMaxNranks)
    return false;
  // gfx1250 only until a cross-GPU gfx9 data test exists.
  if (!IsArchMatch(comm->archName, "gfx1250")) return false;
  if (count == 0)
    return false;

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes % 16 != 0)
    return false;

  const size_t bank = ddaBankSize(comm->ddaScratchBytes);

  if (bytes < kDdaLL128BcTwoShotMin) {
    // One-shot path checks
    if (bytes > rcclDdaLL128Threshold(comm, ncclFuncBroadcast)) return false;
    if (ddaLL128BcSlices(comm, bytes) >
        ddaLL128BcSlotSlices(bank, comm->nRanks, ddaLL128BcArch(comm).sliceWords)) {
      return false;
    }
    // Both the root and the non-root grid must fit the epoch cells.
    const dim3 r = ddaLL128BcGrid(comm, /*isRoot=*/true, bytes);
    const dim3 n = ddaLL128BcGrid(comm, /*isRoot=*/false, bytes);
    const long most = (long)r.x * r.y > (long)n.x ? (long)r.x * r.y : (long)n.x;
    return most <= (long)comm->ddaLLEpochLen;
  } else {
    // Two-shot path checks
    if (!rcclDdaDatatypeSupported(comm, datatype))
      return false;

    // LL128 cap check handled by the selector before calling us.
    // Scratch capacity: the bank must hold (nRanks + 1) chunk-sized slots.
    const size_t slices = ddaLL128Slices(bytes);
    if (ddaLL128BcTwoShotSlotSlices(bank, comm->nRanks, slices) == 0)
      return false;

    return true;
  }
}

uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes < kDdaLL128BcTwoShotMin) {
    const dim3 g = ddaLL128BcGrid(comm, /*isRoot=*/true, bytes);
    return g.x * g.y;
  } else {
    const auto grid = ddaBroadcastFabricLL128Geom(comm, bytes).first;
    return grid.x * grid.y;
  }
}

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes < kDdaLL128BcTwoShotMin) {
    // One-shot path
    const bool isRoot = comm->rank == root;
    const dim3 grid = ddaLL128BcGrid(comm, isRoot, bytes);
    const uintptr_t addrs =
      reinterpret_cast<uintptr_t>(recvbuff) | (isRoot ? reinterpret_cast<uintptr_t>(sendbuff) : 0);
    const bool aligned = addrs % 16 == 0;

    char* const* peers = reinterpret_cast<char* const*>(comm->ddaPeerPtrsDev);
    const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
    const dda::common::DdaKernelParams params{comm->abortFlagDev};

    const DdaLL128BcArch arch = ddaLL128BcArch(comm);
    const unsigned threads = ddaLL128BcThreads(comm);
    INFO(NCCL_COLL, "DDA Broadcast LL128 one-shot: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u line=%dB warp=%zu aligned=%d",
         comm->nRanks, root, bytes, grid.x, grid.y, threads, arch.lineBytes, arch.warp, (int)aligned);

    const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
    if (aligned) {
      hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL128<true>), grid, dim3(threads), 0,
                            stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                            static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                            comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
    } else {
      hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL128<false>), grid, dim3(threads), 0,
                            stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                            static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                            comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
    }
  } else {
    // DPX (partition mode): use the warp-specialized hierarchical kernel.
    if (ncclBroadcastDdaFabricLL128DpxEligible(comm, count, datatype) ||
        (rcclParamDdaBcForceDpx() > 0 && comm->nRanks >= 4 && comm->nRanks % 2 == 0))
      return ncclBroadcastDdaFabricLL128DpxLaunch(sendbuff, recvbuff, count, datatype, root, comm, stream);

    // Two-shot path
    if (!rcclDdaDatatypeSupported(comm, datatype))
      return ncclInvalidArgument;

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
      slices, cs, bank, params, (int)rcclParamDdaLL128BcFanOut());
  }

  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
