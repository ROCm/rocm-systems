/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/ipc/ipc_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/ipc/ipc_mem_handler.h"
#include "bootstrap.h"
#include "param.h"
#include "rccl_arch_thresholds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

// Banked DDA IPC scratch: unset (-1) follows the arch table (ddaIpcBankedScratch), 0 off, any other non-zero value on.
RCCL_PARAM(DdaIpcBankedScratch, "DDA_IPC_BANKED_SCRATCH", -1);

#define HIP_CALL(cmd) \
  do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " << __LINE__ << " in file " \
                << __FILE__ << "\n"; \
    } \
  } while (0)

namespace {

// The scratch comes from hipExtMallocWithFlags, so allocTracker is updated here, not by an allocator helper.
void freeDdaIpcScratch(ncclComm* comm, void* scratch, size_t bytes) {
  if (scratch == nullptr || comm == nullptr) return;
  ncclAllocUntrack(comm->cudaDev, bytes, "ncclDdaIpcCommInit");
  CUDACHECKIGNORE(cudaFree(scratch));
}

bool ddaIpcCommEligible(const ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  const bool ddaArchSupported =
    comm->archName != nullptr && (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"));
  return comm->nRanks == kDdaNranks && comm->nNodes == 1 && comm->bootstrap != nullptr && !comm->directMode &&
         !comm->MNNVL && ddaArchSupported;
}

} // namespace

ncclResult_t ncclDdaIpcCommInit(ncclComm* comm) {
  // Skip DDA if:
  // - nRanks is not exactly kDdaNranks (currently hardcoded to 8)
  // - multi-node runs
  // - not using 1 process per GPU
  // - MNNVL (fabric-based P2P)
  // - the arch is not one the DDA algorithm actually runs on. The dispatch path
  //   (rcclDdaEnabled() in collectives.cc) only enables DDA on gfx942/gfx950; on
  //   every other arch the algorithm is never selected, so allocating the IPC
  //   scratch/barrier here is not necessary. On gfx12xx (RDNA4) the
  //   uncached-memory IPC export fails (hipIpcGetMemHandle -> hipErrorInvalidValue),
  //   which aborts comm init entirely. Gate init to match dispatch.
  if (!ddaIpcCommEligible(comm)) {
    return ncclSuccess;
  }

  // DDA IPC requires cross-GPU IPC memory mapping (hipIpcOpenMemHandle).
  // comm->isAllCudaP2p is set via ncclTopoCheckP2p which on AMD/HIP returns true
  // whenever ranks share a hostHash, regardless of actual P2P support (paths.cc).
  // Use hipDeviceCanAccessPeer directly — the authoritative runtime check for IPC
  // capability, and the same check used in init.cc for hasPeerAccess.
  for (int i = 0; i < comm->nRanks; i++) {
    for (int j = i + 1; j < comm->nRanks; j++) {
      int canAccess = 0;
      hipError_t err = hipDeviceCanAccessPeer(&canAccess, comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
      if (err != hipSuccess || !canAccess) {
        INFO(NCCL_INIT, "ncclDdaIpcCommInit: no P2P between GPU %d and GPU %d, skipping DDA IPC",
             comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
        return ncclSuccess;
      }
    }
  }

  size_t bytes = DDA_IPC_BUFFER_SIZE;
  if (bytes == 0) {
    return ncclSuccess;
  }

  void* scratch = nullptr;
#if defined(HIP_UNCACHED_MEMORY)
  HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocUncached));
#else
  HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocFinegrained));
#endif
  if (scratch != nullptr) ncclAllocTrack(comm->cudaDev, bytes, "ncclDdaIpcCommInit");

  auto* handler = new (std::nothrow) ncclIpcMemHandler(comm->bootstrap, comm->rank, comm->nRanks);
  if (handler == nullptr) {
    freeDdaIpcScratch(comm, scratch, bytes);
    WARN("ncclDdaIpcCommInit: OOM allocating ncclIpcMemHandler");
    return ncclSuccess;
  }

  ncclResult_t res = handler->addSelfDeviceMemPtr(scratch);
  if (res != ncclSuccess) {
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    WARN("ncclDdaIpcCommInit: addSelfDeviceMemPtr failed");
    return ncclSuccess;
  }
  res = handler->exchangeMemPtrs();
  if (res != ncclSuccess) {
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    WARN("ncclDdaIpcCommInit: exchangeMemPtrs failed");
    return ncclSuccess;
  }

  auto cudaOk = [](cudaError_t err, const char* what) {
    if (CUDACLEARERROR(err) != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: %s failed: %s", what, cudaGetErrorString(err));
      return false;
    }
    return true;
  };

  // Device peer table, kDdaNranks entries each: scratch bases, LL region bases (ddaLLRegion), staging banks 0 and 1.
  void* peerDev = nullptr;
  void* h_ptrs[nccl_dda_detail::kDdaIpcPeerTableEntries] = {};
  // From here a local failure still joins the barrier exchange and the ready allgather, as its peers do.
  bool tableOk = cudaOk(cudaMalloc(&peerDev, sizeof(h_ptrs)), "cudaMalloc(peer table)");
  for (int i = 0; tableOk && i < kDdaNranks; ++i) {
    if (handler->getPeerDeviceMemPtr(i, &h_ptrs[i]) != ncclSuccess) {
      WARN("ncclDdaIpcCommInit: getPeerDeviceMemPtr failed");
      tableOk = false;
    } else {
      h_ptrs[kDdaNranks + i] = static_cast<char*>(h_ptrs[i]) + (bytes - dda::common::kDdaIpcLLRegionBytes);
      for (int bank = 0; bank < 2; ++bank) {
        h_ptrs[nccl_dda_detail::kDdaIpcStagingBankPeerIndex + bank * kDdaNranks + i] =
          static_cast<char*>(h_ptrs[i]) + nccl_dda_detail::ddaIpcStagingBankOffset(bytes, bank);
      }
    }
  }
  tableOk = tableOk && cudaOk(cudaMemcpy(peerDev, h_ptrs, sizeof(h_ptrs), cudaMemcpyHostToDevice),
                              "cudaMemcpy(peer table)");
  if (tableOk && ncclCalloc(&comm->ddaPeerPtrsHost, kDdaNranks) != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: OOM allocating host peer table");
    tableOk = false;
  }
  tableOk = tableOk && cudaOk(cudaMemcpy(comm->ddaPeerPtrsHost, h_ptrs, kDdaNranks * sizeof(void*),
                                         cudaMemcpyHostToHost),
                              "cudaMemcpy(host peer table)");

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto barrierPair = dda::common::IpcGpuBarrier::mallocAndInit(kDdaNranks, nBlocksMax, comm->rank, comm->bootstrap,
                                                               dda::common::DdaKernelParams{comm->abortFlagDev});
  DdaIpcBarrierState* barrierState = nullptr;
  if (!barrierPair.first) {
    WARN("ncclDdaIpcCommInit: IpcGpuBarrier::mallocAndInit failed");
  } else if ((barrierState = new (std::nothrow) DdaIpcBarrierState()) == nullptr) {
    WARN("ncclDdaIpcCommInit: OOM allocating DdaIpcBarrierState");
  } else {
    barrierState->resources = std::move(barrierPair.first);
    barrierState->barrierHost = barrierPair.second;
  }

  // Zero fills run on their own non-blocking stream and are synced before the ready gather, so they land before any
  // rank's first DDA kernel on any stream. Syncing the null stream would also consume the app's async errors.
  cudaStream_t zeroStream = nullptr;
  bool zeroed =
    tableOk && cudaOk(cudaStreamCreateWithFlags(&zeroStream, cudaStreamNonBlocking), "zero-fill stream create");
  // Zeroed once, so an LL line never false-matches a first epoch (>= 1); the copy-based paths overwrite per op.
  zeroed = zeroed && cudaOk(cudaMemsetAsync(scratch, 0, bytes, zeroStream), "scratch memset");
  // LL epoch cells (Broadcast, AR/RS LL); optional, a failure only disables those tiers. Zeroed cells make the first
  // flag 1. Sized for the CU-derived cap so the Broadcast grid is not limited by the cells.
  const size_t epochLen = nccl_dda_detail::ddaLLEpochCount(
    kDdaNranks, nccl_dda_detail::ddaFabricMaxNBlocksForScratch(comm->cuCount, nullptr));
  uint32_t* epochDev = nullptr;
  const bool epochOk = zeroed && barrierState != nullptr &&
                       cudaOk(cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)), "LL epoch alloc") &&
                       cudaOk(cudaMemsetAsync(epochDev, 0, epochLen * sizeof(uint32_t), zeroStream), "LL epoch memset");
  if (zeroStream != nullptr) {
    zeroed = cudaOk(cudaStreamSynchronize(zeroStream), "zero-fill sync") && zeroed;
    CUDACHECKIGNORE(cudaStreamDestroy(zeroStream));
  }
  if (!zeroed || !epochOk) {
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
  }

  const bool archBanked = comm->archThresholds != nullptr && comm->archThresholds->ddaIpcBankedScratch;
  const bool wantBanked = nccl_dda_detail::ddaIpcBankedScratchWanted(rcclParamDdaIpcBankedScratch(), archBanked);

  // Rank-uniform outcome, else ranks pick different backends and hang. 0: no DDA IPC, 1: no LL epoch cells, 2: all.
  int ready = (barrierState == nullptr || !zeroed) ? 0 : (epochDev == nullptr ? 1 : 2);
  bool banked = false;
  std::vector<int> allReady(kDdaNranks, 0);
  allReady[comm->rank] = nccl_dda_detail::ddaIpcReadyWord(ready, wantBanked);  // banks only if every rank wants them
  if (bootstrapAllGather(comm->bootstrap, allReady.data(), sizeof(int)) != ncclSuccess) {
    ready = 0;
  } else {
    nccl_dda_detail::ddaIpcAgreeReadyWords(allReady.data(), kDdaNranks, &ready, &banked);
    if (wantBanked && !banked) {
      WARN("ncclDdaIpcCommInit: a rank turned banked scratch off; DDA IPC runs without banked scratch");
    }
  }
  if (ready < 2 && epochDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
  }
  if (ready == 0) {
    delete barrierState;
    barrierPair.first.reset();
    free(comm->ddaPeerPtrsHost);
    comm->ddaPeerPtrsHost = nullptr;
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    freeDdaIpcScratch(comm, scratch, bytes);
    ATTN("DDA IPC setup failed on a rank; skipping the DDA IPC path");
    return ncclSuccess;
  }
  if (ready == 1) {
    ATTN("LL epoch cells unavailable on a rank; DDA IPC LL tiers (Broadcast, AR/RS one-shots) disabled");
  }

  barrierState->scratchAllocBytes = bytes;
  void** const peerTable = static_cast<void**>(peerDev);
  barrierState->stagingBankPeerPtrsDev = banked ? peerTable + nccl_dda_detail::kDdaIpcStagingBankPeerIndex : nullptr;
  barrierState->stagingBankBytes = banked ? nccl_dda_detail::kDdaIpcStagingBankBytes : 0;

  comm->ddaIpcMemHandler = handler;
  comm->ddaScratch = scratch;
  // Simple staging and CE copies stop below the staging banks and the LL region (ddaIpcPayloadBytes).
  comm->ddaScratchBytes = nccl_dda_detail::ddaIpcPayloadBytes(bytes, banked);
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaIpcBarrierState = barrierState;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = epochDev != nullptr ? (int)epochLen : 0;
  INFO(NCCL_INIT, "ncclDdaIpcCommInit: scratch %zu bytes, IpcGpuBarrier nBlocks=%d banked=%d, peer table on device",
       bytes, nBlocksMax, banked);
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommClosePeerMappings(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  auto* barrierState = comm->ddaIpcBarrierState;
  const bool hasBarrierMappings = barrierState != nullptr && barrierState->resources != nullptr &&
                                  barrierState->resources->ipcMemHandler != nullptr;
  if ((hasBarrierMappings || comm->ddaIpcMemHandler != nullptr) && comm->doneEvent != nullptr) {
    // DDA kernels and CE-scratch copies run on the user stream, which the
    // hostStream/deviceStream syncs in commDestroySync do not cover. doneEvent
    // is recorded after every kernel and CE launch, so waiting on it keeps
    // in-flight work off unmapped peers.
    cudaError_t syncErr = cudaEventSynchronize(comm->doneEvent);
    if (syncErr != cudaSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: doneEvent sync failed (%s); continuing cleanup",
           cudaGetErrorString(syncErr));
    }
  }
  if (hasBarrierMappings) {
    ncclResult_t closeResult = barrierState->resources->ipcMemHandler->closePeerMappings();
    if (closeResult != ncclSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: barrier peer mapping close failed (%d); continuing cleanup",
           closeResult);
    }
  }
  if (comm->ddaIpcMemHandler != nullptr) {
    ncclResult_t closeResult = comm->ddaIpcMemHandler->closePeerMappings();
    if (closeResult != ncclSuccess) {
      WARN("ncclDdaIpcCommClosePeerMappings: scratch peer mapping close failed (%d); continuing cleanup",
           closeResult);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }

  // commFree must remain non-collective. Destroy and abort close mappings in
  // commDestroySync and revoke closes them in commReclaim, before the serial
  // commFree loop; this idempotent call is the local backstop for init-failure
  // and other direct commFree paths.
  ncclResult_t closeResult = ncclSuccess;
  NCCLCHECKIGNORE(ncclDdaIpcCommClosePeerMappings(comm), closeResult);

  size_t scratchAllocBytes = comm->ddaScratchBytes + dda::common::kDdaIpcLLRegionBytes;
  if (comm->ddaIpcBarrierState != nullptr) {
    if (comm->ddaIpcBarrierState->scratchAllocBytes != 0) {
      scratchAllocBytes = comm->ddaIpcBarrierState->scratchAllocBytes;
    }
    delete comm->ddaIpcBarrierState;
    comm->ddaIpcBarrierState = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  CUDACHECKIGNORE(cudaFree(comm->ddaLLEpochDev));
  comm->ddaLLEpochDev = nullptr;
  comm->ddaLLEpochLen = 0;
  if (comm->ddaIpcMemHandler != nullptr) {
    delete comm->ddaIpcMemHandler;
    comm->ddaIpcMemHandler = nullptr;
  }
  freeDdaIpcScratch(comm, comm->ddaScratch, scratchAllocBytes);
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  return ncclSuccess;
}
