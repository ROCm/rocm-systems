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
#include "rccl_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

using nccl_dda_detail::DdaIpcBarrierState;
using nccl_dda_detail::ddaMaxNBlocksForScratch;
using nccl_dda_detail::kDdaNranks;

#define HIP_CALL(cmd) \
  do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " << __LINE__ << " in file " \
                << __FILE__ << "\n"; \
    } \
  } while (0)

namespace {

bool ddaIpcCommEligible(const ncclComm* comm) {
  if (comm == nullptr) {
    return false;
  }
  const bool ddaArchSupported =
    comm->archName != nullptr && (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"));
  return comm->nRanks == kDdaNranks && comm->nNodes == 1 && comm->bootstrap != nullptr && !comm->directMode &&
         !comm->MNNVL && ddaArchSupported;
}

// Allocates and zeroes the scratch. Returns null, with the HIP error cleared, on any failure.
void* ddaIpcAllocScratch(size_t bytes) {
  void* scratch = nullptr;
#if defined(HIP_UNCACHED_MEMORY)
  HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocUncached));
#else
  HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocFinegrained));
#endif
  if (scratch == nullptr) {
    (void)cudaGetLastError();
    return nullptr;
  }

  // Zero the scratch once so the LL all-gather's first epoch (>= 1) never
  // false-matches leftover flag words (mirrors the fabric path). Harmless for
  // the copy-based DDA collectives, which overwrite their staging area per op.
  hipError_t memsetErr = hipMemset(scratch, 0, bytes);
  if (memsetErr != hipSuccess) {
    WARN("ncclDdaIpcCommInit: hipMemset of the scratch failed (%s)", hipGetErrorString(memsetErr));
    (void)cudaGetLastError();
    CUDACHECKIGNORE(cudaFree(scratch));
    return nullptr;
  }
  return scratch;
}

// What each rank allocates on its own before the ranks vote. All set, or all null.
struct DdaIpcLocal {
  void* scratch = nullptr;
  ncclIpcMemHandler* handler = nullptr;
  void* peerDev = nullptr;
  void** peerHost = nullptr;
};

void ddaIpcFreeLocal(DdaIpcLocal* local) {
  delete local->handler;
  CUDACHECKIGNORE(cudaFree(local->peerDev));
  free(local->peerHost);
  CUDACHECKIGNORE(cudaFree(local->scratch));
  *local = DdaIpcLocal{};
}

bool ddaIpcAllocLocal(ncclComm* comm, size_t bytes, DdaIpcLocal* local) {
  local->scratch = ddaIpcAllocScratch(bytes);
  if (local->scratch == nullptr) return false;
  local->handler = new (std::nothrow) ncclIpcMemHandler(comm->bootstrap, comm->rank, comm->nRanks);
  if (local->handler == nullptr) {
    WARN("ncclDdaIpcCommInit: OOM allocating ncclIpcMemHandler");
    ddaIpcFreeLocal(local);
    return false;
  }
  cudaError_t ce = cudaMalloc(&local->peerDev, kDdaNranks * sizeof(void*));
  if (ce != cudaSuccess) {
    WARN("ncclDdaIpcCommInit: cudaMalloc(peer table) failed (%s)", cudaGetErrorString(ce));
    (void)cudaGetLastError();
    local->peerDev = nullptr;
    ddaIpcFreeLocal(local);
    return false;
  }
  if (ncclCalloc(&local->peerHost, kDdaNranks) != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: OOM allocating host peer table");
    local->peerHost = nullptr;
    ddaIpcFreeLocal(local);
    return false;
  }
  return true;
}

// Takes ownership of local. Collective: every rank runs both pointer exchanges and the ready
// allgather whatever fails on it, so one rank's failure leaves every rank without DDA IPC
// instead of blocking its peers in an exchange or splitting the backend choice.
ncclResult_t ddaIpcSetup(ncclComm* comm, DdaIpcLocal* local, size_t bytes) {
  const ncclResult_t addRes = local->handler->addSelfDeviceMemPtr(local->scratch);
  const ncclResult_t exchangeRes = local->handler->exchangeMemPtrs();
  bool localOk = addRes == ncclSuccess && exchangeRes == ncclSuccess;
  if (!localOk) WARN("ncclDdaIpcCommInit: scratch pointer exchange failed");

  void* h_ptrs[kDdaNranks] = {};
  for (int i = 0; localOk && i < kDdaNranks; ++i) {
    if (local->handler->getPeerDeviceMemPtr(i, &h_ptrs[i]) != ncclSuccess) {
      WARN("ncclDdaIpcCommInit: getPeerDeviceMemPtr failed");
      localOk = false;
    }
  }
  if (localOk) {
    cudaError_t ce = cudaMemcpy(local->peerDev, h_ptrs, kDdaNranks * sizeof(void*), cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMemcpy(peer table) failed (%s)", cudaGetErrorString(ce));
      (void)cudaGetLastError();
      localOk = false;
    }
  }
  if (localOk) {
    memcpy(local->peerHost, h_ptrs, kDdaNranks * sizeof(void*));
  }

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

  // LL epoch cells (DDA broadcast LL); optional, a failure only disables that tier.
  // Sized for the CU-derived cap so the Broadcast grid is not limited by the cells.
  const size_t epochLen = nccl_dda_detail::ddaLLEpochCount(
    kDdaNranks, nccl_dda_detail::ddaFabricMaxNBlocksForScratch(comm->cuCount, nullptr));
  uint32_t* epochDev = nullptr;
  if (localOk && barrierState != nullptr &&
      (cudaMalloc(&epochDev, epochLen * sizeof(uint32_t)) != cudaSuccess ||
       cudaMemset(epochDev, 0, epochLen * sizeof(uint32_t)) != cudaSuccess)) {
    (void)cudaGetLastError(); // clear the sticky error
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
  }

  // Rank-uniform outcome, else ranks pick different backends and hang. 0: no DDA IPC, 1: no LL epoch cells, 2: all.
  int ready = !localOk || barrierState == nullptr ? 0 : (epochDev == nullptr ? 1 : 2);
  std::vector<int> allReady(kDdaNranks, 0);
  allReady[comm->rank] = ready;
  if (bootstrapAllGather(comm->bootstrap, allReady.data(), sizeof(int)) != ncclSuccess) {
    ready = 0;
  } else {
    ready = *std::min_element(allReady.begin(), allReady.end());
  }
  if (ready < 2 && epochDev != nullptr) {
    CUDACHECKIGNORE(cudaFree(epochDev));
    epochDev = nullptr;
  }
  if (ready == 0) {
    delete barrierState;
    barrierPair.first.reset();
    ddaIpcFreeLocal(local);
    ATTN("DDA IPC setup failed on a rank; skipping the DDA IPC path");
    return ncclSuccess;
  }
  if (ready == 1) {
    ATTN("LL epoch cells unavailable on a rank; DDA Broadcast disabled");
  }

  comm->ddaIpcMemHandler = local->handler;
  comm->ddaScratch = local->scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaPeerPtrsDev = local->peerDev;
  comm->ddaPeerPtrsHost = local->peerHost;
  comm->ddaIpcBarrierState = barrierState;
  comm->ddaLLEpochDev = epochDev;
  comm->ddaLLEpochLen = epochDev != nullptr ? (int)epochLen : 0;
  *local = DdaIpcLocal{};
  INFO(NCCL_INIT, "ncclDdaIpcCommInit: scratch %zu bytes, IpcGpuBarrier nBlocks=%d, peer IPC table on device", bytes,
       nBlocksMax);
  return ncclSuccess;
}

// Collective: each rank votes on holding its local resources, so one unready rank leaves every rank without DDA IPC.
// Capturing ranks may not allocate: the setup's synchronous copies are illegal in capture.
ncclResult_t ddaIpcVoteAndSetup(ncclComm* comm, size_t bytes, bool mayAllocate, bool* agreed) {
  *agreed = false;
  DdaIpcLocal local;
  const bool allocated = mayAllocate && ddaIpcAllocLocal(comm, bytes, &local);
  std::vector<uint8_t> ready(comm->nRanks);
  ready[comm->rank] = allocated;
  ncclResult_t res = bootstrapAllGather(comm->bootstrap, ready.data(), sizeof(uint8_t));
  if (res != ncclSuccess || std::find(ready.begin(), ready.end(), 0) != ready.end()) {
    ddaIpcFreeLocal(&local);
    return res;
  }
  *agreed = true;
  return ddaIpcSetup(comm, &local, bytes);
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

  if (rcclParamDdaIpcLazyInit() != 0) {
    comm->ddaIpcLazyScratchBytes = bytes;
    INFO(NCCL_INIT, "ncclDdaIpcCommInit: deferring the %zu-byte scratch to the first DDA IPC collective", bytes);
    return ncclSuccess;
  }
  bool agreed = false;
  return ddaIpcVoteAndSetup(comm, bytes, /*mayAllocate=*/true, &agreed);
}

ncclResult_t ncclDdaIpcEnsureInit(ncclComm* comm, bool capturing) {
  if (comm == nullptr || comm->ddaIpcLazyScratchBytes == 0) {
    return ncclSuccess;
  }
  // Votes run on the 1st, 2nd, 4th, 8th... trigger: an unready rank costs its peers O(log n) AllGathers.
  const uint64_t call = ++comm->ddaIpcLazyCalls;
  if ((call & (call - 1)) != 0) {
    return ncclSuccess;
  }

  // A rank that cannot select its device still votes, not ready, so its peers do not hang.
  int savedDev = -1;
  bool onDevice = cudaGetDevice(&savedDev) == cudaSuccess;
  const bool switchDevice = onDevice && savedDev != comm->cudaDev;
  if (switchDevice) onDevice = cudaSetDevice(comm->cudaDev) == cudaSuccess;
  if (!onDevice) (void)cudaGetLastError();
  bool agreed = false;
  ncclResult_t res = ddaIpcVoteAndSetup(comm, comm->ddaIpcLazyScratchBytes, onDevice && !capturing, &agreed);
  if (agreed) {
    comm->ddaIpcLazyScratchBytes = 0;
  } else if (res == ncclSuccess) {
    INFO(NCCL_INIT, "ncclDdaIpcEnsureInit: not every rank is ready (call %lu, capturing %d); DDA IPC stays deferred",
         (unsigned long)call, (int)capturing);
  }
  if (switchDevice) CUDACHECKIGNORE(cudaSetDevice(savedDev));
  return res;
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

  if (comm->ddaIpcBarrierState != nullptr) {
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
  CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  comm->ddaIpcLazyScratchBytes = 0;
  return ncclSuccess;
}
