/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/ipc/ipc_init.h"

#include "alloc.h"
#include "archinfo.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/ipc/ipc_mem_handler.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
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

// Every rank contributes localReady; allReady is true only if all ranks reported ready.
static ncclResult_t ddaIpcAllRanksReady(ncclComm* comm, bool localReady, bool* allReady) {
  std::vector<uint8_t> ready(static_cast<size_t>(comm->nRanks), 0);
  ready[static_cast<size_t>(comm->rank)] = localReady ? 1 : 0;
  *allReady = false;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, ready.data(), static_cast<int>(sizeof(uint8_t))));
  for (int i = 0; i < comm->nRanks; ++i) {
    if (ready[static_cast<size_t>(i)] == 0) {
      INFO(NCCL_INIT, "ncclDdaIpcCommInit: rank %d could not set up DDA IPC; disabling it on all %d ranks", i,
           comm->nRanks);
      return ncclSuccess;
    }
  }
  *allReady = true;
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommInit(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
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
  const bool ddaArchSupported =
    comm->archName != nullptr && (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950"));
  if (comm->nRanks != kDdaNranks || comm->nNodes != 1 || comm->bootstrap == nullptr || comm->directMode ||
      comm->MNNVL || !ddaArchSupported) {
    return ncclSuccess;
  }

  // Every rank must reach each bootstrap collective below, so a local failure only clears localReady until the vote.
  bool localReady = true;

  // DDA IPC requires cross-GPU IPC memory mapping (hipIpcOpenMemHandle).
  // comm->isAllCudaP2p is set via ncclTopoCheckP2p which on AMD/HIP returns true
  // whenever ranks share a hostHash, regardless of actual P2P support (paths.cc).
  // Use hipDeviceCanAccessPeer directly — the authoritative runtime check for IPC
  // capability, and the same check used in init.cc for hasPeerAccess.
  for (int i = 0; i < comm->nRanks && localReady; i++) {
    for (int j = i + 1; j < comm->nRanks; j++) {
      int canAccess = 0;
      hipError_t err = hipDeviceCanAccessPeer(&canAccess, comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
      if (err != hipSuccess || !canAccess) {
        INFO(NCCL_INIT, "ncclDdaIpcCommInit: no P2P between GPU %d and GPU %d, skipping DDA IPC",
             comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
        localReady = false;
        break;
      }
    }
  }

  size_t bytes = DDA_IPC_BUFFER_SIZE;
  if (bytes == 0) {
    return ncclSuccess;
  }

  // Zero the scratch once so the LL all-gather's first epoch (>= 1) never
  // false-matches leftover flag words (mirrors the fabric path). Harmless for
  // the copy-based DDA collectives, which overwrite their staging area per op.
  void* scratch = nullptr;
  if (localReady) {
#if defined(HIP_UNCACHED_MEMORY)
    HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocUncached));
#else
    HIP_CALL(hipExtMallocWithFlags((void**)&scratch, bytes, hipDeviceMallocFinegrained));
#endif
    if (scratch == nullptr) {
      WARN("ncclDdaIpcCommInit: could not allocate the %zu-byte scratch buffer", bytes);
      localReady = false;
    } else if (hipMemset(scratch, 0, bytes) != hipSuccess) {
      WARN("ncclDdaIpcCommInit: hipMemset(scratch) failed");
      localReady = false;
    }
  }

  // A null self pointer makes the exchange fail on every rank rather than leave the peers waiting in it.
  auto handler = std::make_unique<ncclIpcMemHandler>(comm->bootstrap, comm->rank, comm->nRanks);
  (void)handler->addSelfDeviceMemPtr(localReady ? scratch : nullptr);
  if (handler->exchangeMemPtrs() != ncclSuccess) {
    WARN("ncclDdaIpcCommInit: exchangeMemPtrs failed");
    localReady = false;
  }

  void* h_ptrs[kDdaNranks] = {};
  for (int i = 0; i < kDdaNranks && localReady; ++i) {
    if (handler->getPeerDeviceMemPtr(i, &h_ptrs[i]) != ncclSuccess) {
      WARN("ncclDdaIpcCommInit: getPeerDeviceMemPtr failed");
      localReady = false;
    }
  }

  void* peerDev = nullptr;
  if (localReady) {
    cudaError_t ce = cudaMalloc(&peerDev, kDdaNranks * sizeof(void*));
    if (ce != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMalloc(peer table) failed (%s)", cudaGetErrorString(ce));
      peerDev = nullptr;
      localReady = false;
    }
  }
  if (localReady) {
    cudaError_t ce = cudaMemcpy(peerDev, h_ptrs, kDdaNranks * sizeof(void*), cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) {
      WARN("ncclDdaIpcCommInit: cudaMemcpy(peer table) failed (%s)", cudaGetErrorString(ce));
      localReady = false;
    }
  }
  if (localReady) {
    if (ncclCalloc(&comm->ddaPeerPtrsHost, kDdaNranks) != ncclSuccess) {
      WARN("ncclDdaIpcCommInit: OOM allocating host peer table");
      localReady = false;
    } else {
      cudaError_t ce = cudaMemcpy(comm->ddaPeerPtrsHost, h_ptrs, kDdaNranks * sizeof(void*), cudaMemcpyHostToHost);
      if (ce != cudaSuccess) {
        WARN("ncclDdaIpcCommInit: cudaMemcpy(host peer table) failed (%s)", cudaGetErrorString(ce));
        localReady = false;
      }
    }
  }

  // Entered on every rank: IpcGpuBarrier::mallocAndInit runs its own handle exchange.
  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto barrierPair = dda::common::IpcGpuBarrier::mallocAndInit(kDdaNranks, nBlocksMax, comm->rank, comm->bootstrap);
  if (!barrierPair.first) {
    WARN("ncclDdaIpcCommInit: IpcGpuBarrier::mallocAndInit failed");
    localReady = false;
  }

  bool allReady = false;
  ncclResult_t res = ddaIpcAllRanksReady(comm, localReady, &allReady);
  if (res != ncclSuccess || !allReady) {
    barrierPair.first.reset();
    free(comm->ddaPeerPtrsHost);
    comm->ddaPeerPtrsHost = nullptr;
    CUDACHECKIGNORE(cudaFree(peerDev));
    handler.reset();
    CUDACHECKIGNORE(cudaFree(scratch));
    return res;
  }

  auto barrierState = std::make_unique<DdaIpcBarrierState>();
  barrierState->resources = std::move(barrierPair.first);
  barrierState->barrierHost = barrierPair.second;

  comm->ddaIpcMemHandler = handler.release();
  comm->ddaScratch = scratch;
  comm->ddaScratchBytes = bytes;
  comm->ddaPeerPtrsDev = peerDev;
  comm->ddaIpcBarrierState = barrierState.release();
  INFO(NCCL_INIT, "ncclDdaIpcCommInit: scratch %zu bytes, IpcGpuBarrier nBlocks=%d, peer IPC table on device", bytes,
       nBlocksMax);
  return ncclSuccess;
}

ncclResult_t ncclDdaIpcCommFini(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->ddaIpcBarrierState != nullptr) {
    delete static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
    comm->ddaIpcBarrierState = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaPeerPtrsDev));
  comm->ddaPeerPtrsDev = nullptr;
  free(comm->ddaPeerPtrsHost);
  comm->ddaPeerPtrsHost = nullptr;
  if (comm->ddaIpcMemHandler != nullptr) {
    delete comm->ddaIpcMemHandler;
    comm->ddaIpcMemHandler = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaScratch));
  comm->ddaScratch = nullptr;
  comm->ddaScratchBytes = 0;
  return ncclSuccess;
}
