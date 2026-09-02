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
using nccl_dda_detail::ddaFabricScratchSizing;
using nccl_dda_detail::ddaLLEpochCount;
using nccl_dda_detail::DdaFabricBarrierState;

RCCL_PARAM(DdaFabricBufferSizeForScratch, "DDA_FABRIC_BUFFER_SIZE", -1);

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
  const int64_t llEnabled = rcclParamDdaLL();
  const int64_t llThresh = rcclParamDdaLLThreshold();
  const int64_t ll128Enabled = rcclParamDdaLL128();
  const int64_t ll128Thresh = rcclParamDdaLL128Threshold();
  const int64_t simpleThresh = rcclParamDdaThreshold();
  const int64_t fabricScratchOverride = rcclParamDdaFabricBufferSizeForScratch();

  // Right-sized from the DDA thresholds and nRanks (env-overridable) instead of
  // a fixed 10 GiB. RCCL_DDA_FABRIC_BUFFER_SIZE=0 disables the fabric DDA path.
  size_t bytes =
    ddaFabricScratchSizing(nRanks, fabricScratchOverride, rcclParamDdaEnable(), simpleThresh, llEnabled, ll128Enabled);
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

  const char* fabricMaxBlocksOverride = getenv("RCCL_DDA_FABRIC_MAXBLOCKS");
  const int localBlocksMax = ddaFabricMaxNBlocksForScratch(comm->cuCount, fabricMaxBlocksOverride);
  std::vector<int> blockCaps(nRanks, 0);
  blockCaps[comm->rank] = localBlocksMax;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, blockCaps.data(), sizeof(int)));

  int nBlocksMax = localBlocksMax;
  for (int rank = 0; rank < nRanks; ++rank) {
    if (blockCaps[rank] < nBlocksMax) {
      nBlocksMax = blockCaps[rank];
    }
  }

  // Owned resources: handed to comm on success, freed at `fail` otherwise.
  // All declared before any goto so the cleanup label is reachable without
  // jumping over an initialization.
  void* scratch = nullptr;
  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  void* peerDev = nullptr;
  void** peerHost = nullptr;
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

  CUDACHECKGOTO(cudaMalloc(&peerDev, nRanks * sizeof(void*)), res, fail);

  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, fail);
  }

  CUDACHECKGOTO(cudaMemcpy(peerDev, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToDevice), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&peerHost, nRanks), res, fail);
  CUDACHECKGOTO(cudaMemcpy(peerHost, h_ptrs.data(), nRanks * sizeof(void*), cudaMemcpyHostToHost), res, fail);

  {
    auto barrierPair =
      dda::common::FabricGpuBarrier::mallocAndInit(nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager);
    if (!barrierPair.first) {
      WARN("ncclDdaFabricCommInit: FabricGpuBarrier malloc/init failed");
      goto fail;
    }
    barrierState = new (std::nothrow) DdaFabricBarrierState();
    if (barrierState == nullptr) {
      WARN("ncclDdaFabricCommInit: OOM allocating DdaFabricBarrierState");
      goto fail;
    }
    barrierState->resources = std::move(barrierPair.first);
    barrierState->barrierHost = barrierPair.second;
  }

  // Zero the scratch once so the first epoch (>= 1) across all LL operations
  // never false-matches leftover flag words. Subsequent LL calls rely on
  // monotonic epochs + the 2-bank layout rather than re-zeroing.
  CUDACHECKGOTO(cudaMemset(scratch, 0, bytes), res, fail);

  // Device epoch cells for the LL collectives: zero-initialised
  // so the first device-derived flag is 1. Bumped on the device every LL launch.
  CUDACHECKGOTO(cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)), res, fail);
  CUDACHECKGOTO(cudaMemset(epochDev, 0, epochLen * sizeof(uint32_t)), res, fail);

  // Success: hand ownership of every resource to comm.
  comm->ddaFabricMemHandler = handler;
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaScratchIsVmm = true;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaPeerPtrsHost = peerHost;
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = (int)epochLen;
  INFO(NCCL_INIT,
       "ncclDdaFabricCommInit: nRanks %d, scratch %zu bytes (vmm, gfx1250 fabric path; derived from RCCL DDA params; "
       "RCCL_DDA_FABRIC_BUFFER_SIZE=%lld), LL enabled=%lld threshold=%lld, "
       "LL128 enabled=%lld threshold=%lld, Simple threshold=%lld, FabricGpuBarrier nBlocks=%d, peer table on device",
       nRanks, bytes, (long long)fabricScratchOverride, (long long)llEnabled, (long long)llThresh,
       (long long)ll128Enabled, (long long)ll128Thresh, (long long)simpleThresh, nBlocksMax);
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
  if (peerHost != nullptr) {
    free(peerHost);
  }
  if (peerDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(peerDev));
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
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  // Destroying the fabric handler unmaps/frees the imported peer scratch buffers.
  if (comm->ddaFabricMemHandler != nullptr) {
    delete static_cast<ncclFabricMemHandler*>(comm->ddaFabricMemHandler);
    comm->ddaFabricMemHandler = nullptr;
  }
  // Free this rank's scratch buffer with the allocator that produced it.
  if (comm->ddaScratch != nullptr) {
    if (comm->ddaScratchIsVmm) {
      (void)ncclCuMemFree(comm->ddaScratch, comm->memManager);
    } else {
      CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
    }
  }
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  comm->ddaScratchIsVmm = false;
  return ncclSuccess;
}
