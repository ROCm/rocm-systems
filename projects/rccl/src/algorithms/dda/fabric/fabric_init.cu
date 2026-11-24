/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/fabric/fabric_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "algorithms/dda/fabric/fabric_mem_handler.h"
#include "bootstrap.h"
#include "rccl_common.h"
#include "param.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

using dda::common::kDdaMaxNranks;
using nccl_dda_detail::ddaFabricMaxNBlocksForScratch;
using nccl_dda_detail::ddaFabricRegionLayout;
using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::DdaFabricRegionLayout;
using nccl_dda_detail::kDdaRegionCe;
using nccl_dda_detail::kDdaRegionCount;
using nccl_dda_detail::kDdaRegionLL;
using nccl_dda_detail::kDdaRegionLL128;
using nccl_dda_detail::kDdaRegionSimple;
using nccl_dda_detail::DdaFabricBarrierState;
using nccl_dda_detail::DdaFabricMaxBlocksOverride;

// Fixed per-protocol scratch region sizes (see nccl_dda_detail::DdaFabricRegion). 0 disables a region and every tier
// that stages through it; a message that does not fit its region falls back to the next tier. Must match on every rank.
RCCL_PARAM(DdaFabricSimpleBufferSize, "DDA_FABRIC_SIMPLE_BUFFER_SIZE", 256LL * 1024 * 1024);
RCCL_PARAM(DdaFabricCeBufferSize, "DDA_FABRIC_CE_BUFFER_SIZE", 256LL * 1024 * 1024);
RCCL_PARAM(DdaFabricLLBufferSize, "DDA_FABRIC_LL_BUFFER_SIZE", 8LL * 1024 * 1024);
RCCL_PARAM(DdaFabricLL128BufferSize, "DDA_FABRIC_LL128_BUFFER_SIZE", 64LL * 1024 * 1024);
// RCCL_DDA_A2A_PUSH_PEER_EPOCH: 1 (default) uses epoch flags for the peer-major vector and TDM alltoall pushes, 0 the
// one-token CAS flags (see FabricGpuBarrier::signalWaitPeer / postPeer / waitPeer). Must be the same on every rank.
// With 0 the pushes run without drain blocks (RCCL_DDA_A2A_PUSH_PEER_DRAIN is ignored).
RCCL_PARAM(DdaA2APushPeerEpoch, "DDA_A2A_PUSH_PEER_EPOCH", 1);

namespace {

const char* const kDdaRegionNames[kDdaRegionCount] = {"Simple", "CE", "LL", "LL128"};

// Points `out` at the region [offset, offset + bytes) of every rank's scratch. Leaves it empty when bytes == 0.
ncclResult_t ddaFabricBuildRegion(int nRanks, int rank, const std::vector<void*>& peerBases, size_t offset,
                                  size_t bytes, ncclDdaScratchRegion* out) {
  *out = {};
  if (bytes == 0) {
    return ncclSuccess;
  }
  std::vector<void*> ptrs(nRanks, nullptr);
  for (int i = 0; i < nRanks; ++i) {
    ptrs[i] = static_cast<char*>(peerBases[i]) + offset;
  }
  ncclResult_t res = ncclSuccess;
  CUDACHECKGOTO(cudaMalloc(&out->peersDev, nRanks * sizeof(void*)), res, fail);
  CUDACHECKGOTO(cudaMemcpy(out->peersDev, ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToDevice), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&out->peersHost, nRanks), res, fail);
  CUDACHECKGOTO(cudaMemcpy(out->peersHost, ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToHost), res, fail);
  out->self = ptrs[rank];
  out->bytes = bytes;
  return ncclSuccess;
fail:
  CUDACHECKIGNORE(cudaFree(out->peersDev));
  free(out->peersHost);
  *out = {};
  return res;
}

void ddaFabricFreeRegion(ncclDdaScratchRegion* region) {
  CUDACHECKIGNORE(cudaFree(region->peersDev));
  free(region->peersHost);
  *region = {};
}

} // namespace

bool ncclDdaUseFabricPath(ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  return comm->MNNVL == 1 && IsArchMatch(comm->archName, "gfx1250");
}

ncclResult_t ncclDdaFabricCommInit(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  if (comm->nRanks < 2 || comm->nRanks > kDdaMaxNranks || comm->bootstrap == nullptr) {
    return ncclSuccess;
  }

  // Fabric DDA assumes every rank shares one UALink clique; skip (fall back to
  // normal RCCL) if this comm spans multiple cliques -- e.g. multiple racks.
  if (comm->clique.size != comm->nRanks) {
    INFO(
      NCCL_INIT,
      "ncclDdaFabricCommInit: comm spans multiple fabric cliques (nRanks %d, clique.size %d); skipping fabric DDA path",
      comm->nRanks, comm->clique.size);
    return ncclSuccess;
  }

  const int nRanks = comm->nRanks;
  if (!rcclParamDdaEnable()) {
    return ncclSuccess;
  }
  if (getenv("RCCL_DDA_FABRIC_BUFFER_SIZE") != nullptr) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_BUFFER_SIZE is ignored; the fabric DDA scratch is sized per protocol "
         "by RCCL_DDA_FABRIC_SIMPLE_BUFFER_SIZE, RCCL_DDA_FABRIC_CE_BUFFER_SIZE, RCCL_DDA_FABRIC_LL_BUFFER_SIZE and "
         "RCCL_DDA_FABRIC_LL128_BUFFER_SIZE");
  }

  // One allocation, one region per protocol, fixed by CVARs: nothing here depends on nRanks or the tuning tables.
  const int64_t requested[kDdaRegionCount] = {rcclParamDdaFabricSimpleBufferSize(), rcclParamDdaFabricCeBufferSize(),
                                              rcclParamDdaFabricLLBufferSize(), rcclParamDdaFabricLL128BufferSize()};
  const DdaFabricRegionLayout layout = ddaFabricRegionLayout(requested);
  const size_t bytes = layout.total;
  if (bytes == 0) {
    return ncclSuccess;
  }

  // Scratch (temp) buffer via VMM: the fabric path requires a fabric-capable
  // (cuMem) allocation so the handle can be exported across the clique. If VMM
  // is unavailable the fabric path is skipped (DDA disabled, normal RCCL path
  // used). ncclCuMemAlloc rounds size up to the allocation granularity.
  if (!ncclCuMemEnable()) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM unavailable; skipping fabric DDA path");
    return ncclSuccess;
  }

  {
    // Every rank's regions must sit at the same offsets: a peer addresses this rank's scratch with its own layout.
    std::vector<uint64_t> allSizes((size_t)nRanks * kDdaRegionCount, 0);
    for (int i = 0; i < kDdaRegionCount; ++i) {
      allSizes[(size_t)comm->rank * kDdaRegionCount + i] = layout.bytes[i];
    }
    NCCLCHECK(bootstrapAllGather(comm->bootstrap, allSizes.data(), kDdaRegionCount * sizeof(uint64_t)));
    for (int r = 0; r < nRanks; ++r) {
      for (int i = 0; i < kDdaRegionCount; ++i) {
        if (allSizes[(size_t)r * kDdaRegionCount + i] != layout.bytes[i]) {
          WARN("ncclDdaFabricCommInit: rank %d has a %llu-byte %s scratch region, this rank %zu; the "
               "RCCL_DDA_FABRIC_*_BUFFER_SIZE settings must match on every rank. Skipping fabric DDA path",
               r, (unsigned long long)allSizes[(size_t)r * kDdaRegionCount + i], kDdaRegionNames[i], layout.bytes[i]);
          return ncclSuccess;
        }
      }
    }
  }

  const char* fabricMaxBlocksOverride = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
  DdaFabricMaxBlocksOverride parsedOverride;
  const int localBlocksMax =
    ddaFabricMaxNBlocksForScratch(comm->cuCount, fabricMaxBlocksOverride, &parsedOverride);

  if (parsedOverride.specified && !parsedOverride.valid) {
    WARN("ncclDdaFabricCommInit: Ignoring invalid RCCL_DDA_FABRIC_MAXBLOCKS='%s'; using CU-derived cap %d.",
         fabricMaxBlocksOverride, localBlocksMax);
  } else if (parsedOverride.valid && parsedOverride.requested < 1) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld is below the minimum; using 1.",
         parsedOverride.requested);
  } else if (parsedOverride.valid && parsedOverride.requested > localBlocksMax) {
    WARN("ncclDdaFabricCommInit: RCCL_DDA_FABRIC_MAXBLOCKS=%ld exceeds CU-derived cap (%d); using %d. "
         "The override can only lower the block count, not raise it.",
         parsedOverride.requested, localBlocksMax, localBlocksMax);
  }

  std::vector<int> blockCaps(nRanks, 0);
  blockCaps[comm->rank] = localBlocksMax;
  // Fabric cliques are normally homogeneous, but harvested or disabled CUs
  // can produce different local caps. Every rank must launch the same number
  // of blocks because the barrier pairs participants by blockIdx.
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, blockCaps.data(), sizeof(int)));

  int nBlocksMax = localBlocksMax;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (blockCaps[rank] < nBlocksMax) {
      nBlocksMax = blockCaps[rank];
    }
  }

  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: allocating %zu-byte scratch: Simple %zu @0x%zx, CE %zu @0x%zx, LL %zu @0x%zx, "
       "LL128 %zu @0x%zx",
       bytes, layout.bytes[kDdaRegionSimple], layout.offset[kDdaRegionSimple], layout.bytes[kDdaRegionCe],
       layout.offset[kDdaRegionCe], layout.bytes[kDdaRegionLL], layout.offset[kDdaRegionLL],
       layout.bytes[kDdaRegionLL128], layout.offset[kDdaRegionLL128]);

  // Owned resources: handed to comm on success, freed at `fail` otherwise.
  // All declared before any goto so the cleanup label is reachable without
  // jumping over an initialization.
  void* scratch = nullptr;
  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  ncclDdaScratchRegion regions[kDdaRegionCount] = {};
  DdaFabricBarrierState* barrierState = nullptr;
  uint32_t* epochDev = nullptr;
  std::vector<void*> h_ptrs(nRanks, nullptr);
  const size_t epochLen = ddaLLEpochCount(nRanks, nBlocksMax);
  ncclResult_t res = ncclSuccess;

  res = ncclCuMemAlloc(&scratch, &scratchHandle, ncclCuMemHandleType, bytes, comm->memManager);
  if (res != ncclSuccess || scratch == nullptr) {
    INFO(NCCL_INIT, "ncclDdaFabricCommInit: VMM scratch alloc failed; skipping fabric DDA path");
    scratch = nullptr;
    goto fail;
  }

  handler = new (std::nothrow) ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  if (handler == nullptr) {
    WARN("ncclDdaFabricCommInit: OOM allocating ncclFabricMemHandler");
    goto fail;
  }

  NCCLCHECKGOTO(handler->addSelfDeviceMem(scratch, scratchHandle, bytes), res, fail);
  NCCLCHECKGOTO(handler->exchangeMemPtrs(), res, fail);

  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, fail);
  }
  for (int i = 0; i < kDdaRegionCount; ++i) {
    NCCLCHECKGOTO(ddaFabricBuildRegion(nRanks, comm->rank, h_ptrs, layout.offset[i], layout.bytes[i], &regions[i]),
                  res, fail);
  }

  {
    auto cudaOk = [](cudaError_t err, const char* what) {
      if (CUDACLEARERROR(err) != cudaSuccess) {
        WARN("ncclDdaFabricCommInit: %s failed: %s", what, cudaGetErrorString(err));
        return false;
      }
      return true;
    };
    // Past the barrier's pointer exchange, failures set !ready so every rank still reaches the allgather.
    bool ready = true;
    auto barrierPair =
      dda::common::FabricGpuBarrier::mallocAndInit(nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager,
                                                   dda::common::DdaKernelParams{comm->abortFlagDev});
    if (!barrierPair.first) {
      WARN("ncclDdaFabricCommInit: FabricGpuBarrier malloc/init failed");
      ready = false;
    } else if ((barrierState = new (std::nothrow) DdaFabricBarrierState()) == nullptr) {
      WARN("ncclDdaFabricCommInit: OOM allocating DdaFabricBarrierState");
      ready = false;
    } else {
      barrierState->resources = std::move(barrierPair.first);
      barrierState->barrierHost = barrierPair.second;
      barrierState->barrierHost.setEpochFlags(rcclParamDdaA2APushPeerEpoch() != 0);
    }

    // Own stream: syncing the null stream would also consume async errors from the app's blocking streams.
    cudaStream_t zeroStream = nullptr;
    ready = ready && cudaOk(cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking), "zero-fill stream create");
    // Zeroed once (every region, so the LL and LL128 flag words start at 0); later LL calls rely on monotonic
    // epochs (>= 1) and two banks, not re-zeroing.
    ready = ready && cudaOk(cudaMemsetAsync(scratch, 0, bytes, zeroStream), "scratch memset");
    // Zeroed epoch cells make the first device-derived LL flag 1.
    ready = ready && cudaOk(cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)), "epoch alloc") &&
            cudaOk(cudaMemsetAsync(epochDev, 0, epochLen * sizeof(uint32_t), zeroStream), "epoch memset");
    // Fills must land before any rank's first LL kernel (and before `fail` frees scratch); the allgather orders ranks.
    if (zeroStream != nullptr) {
      ready = cudaOk(cudaStreamSynchronize(zeroStream), "zero-fill sync") && ready;
      CUDACHECKIGNORE(cudaStreamDestroy(zeroStream));
    }

    std::vector<int> allReady(nRanks, 0);
    allReady[comm->rank] = ready ? 1 : 0;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allReady.data(), sizeof(int)), res, fail);
    for (int r = 0; r < nRanks; ++r) {
      if (!allReady[r]) {
        INFO(NCCL_INIT, "ncclDdaFabricCommInit: rank %d failed to set up its LL buffers; skipping fabric DDA path", r);
        goto fail;
      }
    }
  }

  // Success: hand ownership of every resource to comm. The Simple region keeps the legacy ddaScratch fields.
  comm->ddaFabricMemHandler = handler;
  comm->ddaFabricScratchBase = scratch;
  comm->ddaFabricScratchTotalBytes = bytes;
  comm->ddaScratch = regions[kDdaRegionSimple].self;
  comm->ddaScratchBytes = regions[kDdaRegionSimple].bytes;
  comm->ddaScratchIsVmm = true;
  comm->ddaPeerPtrsDev = regions[kDdaRegionSimple].peersDev;
  comm->ddaPeerPtrsHost = regions[kDdaRegionSimple].peersHost;
  comm->ddaRegionCe = regions[kDdaRegionCe];
  comm->ddaRegionLL = regions[kDdaRegionLL];
  comm->ddaRegionLL128 = regions[kDdaRegionLL128];
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = (int)epochLen;
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: nRanks %d, scratch %zu bytes (vmm, gfx1250 fabric path; Simple %zu, CE %zu, LL %zu, "
       "LL128 %zu), FabricGpuBarrier nBlocks=%d, peer tables on device",
       nRanks, bytes, regions[kDdaRegionSimple].bytes, regions[kDdaRegionCe].bytes, regions[kDdaRegionLL].bytes,
       regions[kDdaRegionLL128].bytes, nBlocksMax);
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: CU count=%d, local max blocks=%d, communicator max blocks=%d",
       comm->cuCount, localBlocksMax, nBlocksMax);
  return ncclSuccess;

fail:
  if (barrierState != nullptr) {
    delete barrierState;
  }
  if (epochDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(epochDev));
  }
  for (int i = 0; i < kDdaRegionCount; ++i) {
    ddaFabricFreeRegion(&regions[i]);
  }
  if (handler != nullptr) {
    delete handler;
  }
  if (scratch != nullptr) {
    (void)ncclCuMemFree(scratch, comm->memManager);
  }
  return ncclSuccess;
}

ncclResult_t ncclDdaFabricCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->ddaFabricBarrierState != nullptr) {
    delete static_cast<DdaFabricBarrierState*>(comm->ddaFabricBarrierState);
    comm->ddaFabricBarrierState = nullptr;
  }
  // The Simple region's tables live in the legacy fields.
  ncclDdaScratchRegion simple = {comm->ddaScratch, comm->ddaScratchBytes, comm->ddaPeerPtrsDev, comm->ddaPeerPtrsHost};
  ddaFabricFreeRegion(&simple);
  comm->ddaPeerPtrsDev = nullptr;
  comm->ddaPeerPtrsHost = nullptr;
  ddaFabricFreeRegion(&comm->ddaRegionCe);
  ddaFabricFreeRegion(&comm->ddaRegionLL);
  ddaFabricFreeRegion(&comm->ddaRegionLL128);
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  // Destroying the fabric handler unmaps/frees the imported peer scratch buffers.
  if (comm->ddaFabricMemHandler != nullptr) {
    delete static_cast<ncclFabricMemHandler*>(comm->ddaFabricMemHandler);
    comm->ddaFabricMemHandler = nullptr;
  }
  // Free this rank's scratch allocation (every region) with the allocator that produced it.
  if (comm->ddaFabricScratchBase != nullptr) {
    (void)ncclCuMemFree(comm->ddaFabricScratchBase, comm->memManager);
  } else if (comm->ddaScratch != nullptr) {
    if (comm->ddaScratchIsVmm) {
      (void)ncclCuMemFree(comm->ddaScratch, comm->memManager);
    } else {
      CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
    }
  }
  comm->ddaFabricScratchBase = nullptr;
  comm->ddaFabricScratchTotalBytes = 0;
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  comm->ddaScratchIsVmm = false;
  return ncclSuccess;
}
