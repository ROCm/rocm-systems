/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "fabric_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dda_init_detail.h"
#include "fabric_gpu_barrier.h"
#include "fabric_mem_handler.h"
#include "ll_fabric.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

using meta::comms::kDdaMaxNranks;
using meta::comms::kLLDataBytesPerPacket;
using meta::comms::LLPacket16;
using nccl_dda_detail::ddaFabricLLMaxBytes;
using nccl_dda_detail::ddaFabricMaxNBlocksForScratch;
using nccl_dda_detail::DdaFabricBarrierState;

namespace {

// Optional dedicated LL (low-latency, remote-write) recv buffer for the DDA
// fabric all-reduce. Separate from ddaScratch so the Simple remote-read path
// never clobbers the persistent LL flags. Non-fatal: on any failure the LL
// fields are left null and the Simple fabric path remains usable.
void ncclDdaFabricLLInit(ncclComm* comm) {
  comm->ddaFabricLLMemHandler = nullptr;
  comm->ddaFabricLLRecv = nullptr;
  comm->ddaFabricLLBytes = 0;
  comm->ddaFabricLLPeerPtrsDev = nullptr;
  comm->ddaFabricLLEpoch = 0;

  const int nRanks = comm->nRanks;
  const size_t llMaxBytes = ddaFabricLLMaxBytes();
  const size_t slotStridePkts = llMaxBytes / kLLDataBytesPerPacket;
  const size_t bankStridePkts = static_cast<size_t>(nRanks) * slotStridePkts;
  // 2 banks (double-buffered) x nRanks slots x slotStridePkts packets.
  const size_t llBytes = 2 * bankStridePkts * sizeof(LLPacket16);

  void* llRecv = nullptr;
  CUmemGenericAllocationHandle llHandle{};
  ncclFabricMemHandler* llMemHandler = nullptr;
  void* llPeerDev = nullptr;
  std::vector<void*> h_ptrs(nRanks, nullptr);
  ncclResult_t res = ncclSuccess;

  res = ncclCuMemAlloc(
      &llRecv, &llHandle, ncclCuMemHandleType, llBytes, comm->memManager);
  if (res != ncclSuccess || llRecv == nullptr) {
    INFO(NCCL_INIT,
         "ncclDdaFabricLLInit: VMM LL buffer alloc failed; LL path disabled");
    llRecv = nullptr;
    goto ll_fail;
  }

  // Clear once: flag 0 is never a valid epoch (epochs start at 1), so readers
  // for epoch >= 1 correctly block on not-yet-written slots.
  CUDACHECKGOTO(cudaMemset(llRecv, 0, llBytes), res, ll_fail);

  llMemHandler = new (std::nothrow)
      ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  if (llMemHandler == nullptr) {
    WARN("ncclDdaFabricLLInit: OOM allocating ncclFabricMemHandler");
    goto ll_fail;
  }

  NCCLCHECKGOTO(
      llMemHandler->addSelfDeviceMem(llRecv, llHandle, llBytes), res, ll_fail);
  NCCLCHECKGOTO(llMemHandler->exchangeMemPtrs(), res, ll_fail);

  CUDACHECKGOTO(cudaMalloc(&llPeerDev, nRanks * sizeof(void*)), res, ll_fail);
  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(llMemHandler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, ll_fail);
  }
  CUDACHECKGOTO(
      cudaMemcpy(llPeerDev, h_ptrs.data(), nRanks * sizeof(void*),
                 cudaMemcpyHostToDevice),
      res, ll_fail);

  // Host epoch counter starts at 0; the first op increments it to 1 (flag 0 is
  // never valid, matching the cleared-once recv buffer).
  comm->ddaFabricLLEpoch = 0;

  comm->ddaFabricLLMemHandler = llMemHandler;
  comm->ddaFabricLLRecv = llRecv;
  comm->ddaFabricLLBytes = llBytes;
  comm->ddaFabricLLPeerPtrsDev = llPeerDev;
  INFO(NCCL_INIT,
       "ncclDdaFabricLLInit: nRanks %d, LL recv %zu bytes (vmm), maxBytes=%zu, slotStridePkts=%zu",
       nRanks, llBytes, llMaxBytes, slotStridePkts);
  return;

ll_fail:
  if (llPeerDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(llPeerDev));
  }
  if (llMemHandler != nullptr) {
    delete llMemHandler;
  }
  if (llRecv != nullptr) {
    (void)ncclCuMemFree(llRecv, comm->memManager);
  }
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

  if (comm->nRanks < 2 || comm->nRanks > kDdaMaxNranks ||
      comm->bootstrap == nullptr) {
    return ncclSuccess;
  }

  const int nRanks = comm->nRanks;

  size_t bytes = DDA_FABRIC_BUFFER_SIZE;
  if (bytes == 0) {
    return ncclSuccess;
  }

  // Scratch (temp) buffer via VMM: the fabric path requires a fabric-capable
  // (cuMem) allocation so the handle can be exported across the clique. If VMM
  // is unavailable the fabric path is skipped (DDA disabled, normal RCCL path
  // used). ncclCuMemAlloc rounds size up to the allocation granularity.
  if (!ncclCuMemEnable()) {
    INFO(
        NCCL_INIT,
        "ncclDdaFabricCommInit: VMM unavailable; skipping fabric DDA path");
    return ncclSuccess;
  }

  // Owned resources: handed to comm on success, freed at `fail` otherwise.
  // All declared before any goto so the cleanup label is reachable without
  // jumping over an initialization.
  void* scratch = nullptr;
  CUmemGenericAllocationHandle scratchHandle{};
  ncclFabricMemHandler* handler = nullptr;
  void* peerDev = nullptr;
  DdaFabricBarrierState* barrierState = nullptr;
  std::vector<void*> h_ptrs(nRanks, nullptr);
  const int nBlocksMax = ddaFabricMaxNBlocksForScratch();
  ncclResult_t res = ncclSuccess;

  res = ncclCuMemAlloc(
      &scratch, &scratchHandle, ncclCuMemHandleType, bytes, comm->memManager);
  if (res != ncclSuccess || scratch == nullptr) {
    INFO(
        NCCL_INIT,
        "ncclDdaFabricCommInit: VMM scratch alloc failed; skipping fabric DDA path");
    scratch = nullptr;
    goto fail;
  }

  handler = new (std::nothrow)
      ncclFabricMemHandler(comm->bootstrap, comm->rank, nRanks, comm->memManager);
  if (handler == nullptr) {
    WARN("ncclDdaFabricCommInit: OOM allocating ncclFabricMemHandler");
    goto fail;
  }

  NCCLCHECKGOTO(
      handler->addSelfDeviceMem(scratch, scratchHandle, bytes), res, fail);
  NCCLCHECKGOTO(handler->exchangeMemPtrs(), res, fail);

  CUDACHECKGOTO(cudaMalloc(&peerDev, nRanks * sizeof(void*)), res, fail);

  for (int i = 0; i < nRanks; ++i) {
    NCCLCHECKGOTO(handler->getPeerDeviceMemPtr(i, &h_ptrs[i]), res, fail);
  }

  CUDACHECKGOTO(
      cudaMemcpy(
          peerDev, h_ptrs.data(), nRanks * sizeof(void*),
          cudaMemcpyHostToDevice),
      res, fail);

  {
    auto barrierPair = meta::comms::FabricGpuBarrier::mallocAndInit(
        nRanks, nBlocksMax, comm->rank, comm->bootstrap, comm->memManager);
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

  // Success: hand ownership of every resource to comm.
  comm->ddaFabricMemHandler = handler;
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaScratchIsVmm = true;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaFabricBarrierState = barrierState;
  comm->ddaFabricMaxBlocks = nBlocksMax;
  INFO(
      NCCL_INIT,
      "ncclDdaFabricCommInit: nRanks %d, scratch %zu bytes (vmm), FabricGpuBarrier nBlocks=%d, peer table on device",
      nRanks,
      bytes,
      nBlocksMax);

  // Optional dedicated LL remote-write recv buffer (small-size fast path).
  ncclDdaFabricLLInit(comm);
  return ncclSuccess;

fail:
  if (barrierState != nullptr) {
    delete barrierState;
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
  // LL remote-write path resources (optional; may be null).
  comm->ddaFabricLLEpoch = 0;
  if (comm->ddaFabricLLPeerPtrsDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(comm->ddaFabricLLPeerPtrsDev));
    comm->ddaFabricLLPeerPtrsDev = nullptr;
  }
  if (comm->ddaFabricLLMemHandler != nullptr) {
    delete static_cast<ncclFabricMemHandler*>(comm->ddaFabricLLMemHandler);
    comm->ddaFabricLLMemHandler = nullptr;
  }
  if (comm->ddaFabricLLRecv != nullptr) {
    (void)ncclCuMemFree(comm->ddaFabricLLRecv, comm->memManager);
    comm->ddaFabricLLRecv = nullptr;
  }
  comm->ddaFabricLLBytes = 0;
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
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
