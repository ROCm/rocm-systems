/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "ipc_init.h"

#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_init_detail.h"
#include "ipc_mem_handler.h"

#include <cuda_runtime.h>

using nccl_dda_ipc_detail::DdaIpcBarrierState;
using nccl_dda_ipc_detail::ddaMaxNBlocksForScratch;
using nccl_dda_ipc_detail::kDdaNranks;


#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
        }                                                                               \
    } while (0)

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
      comm->archName != nullptr &&
      (IsArchMatch(comm->archName, "gfx942") ||
       IsArchMatch(comm->archName, "gfx950"));
  if (comm->nRanks != kDdaNranks || comm->nNodes != 1 ||
      comm->bootstrap == nullptr || comm->directMode || comm->MNNVL ||
      !ddaArchSupported) {
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
      hipError_t err = hipDeviceCanAccessPeer(
          &canAccess, comm->peerInfo[i].cudaDev, comm->peerInfo[j].cudaDev);
      if (err != hipSuccess || !canAccess) {
        INFO(NCCL_INIT,
             "ncclDdaIpcCommInit: no P2P between GPU %d and GPU %d, skipping DDA IPC",
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

  auto* handler = new (std::nothrow) ncclIpcMemHandler(
      comm->bootstrap, comm->rank, comm->nRanks);
  if (handler == nullptr) {
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN("ncclDdaIpcCommInit: OOM allocating ncclIpcMemHandler");
    return ncclSuccess;
  }

  ncclResult_t res = handler->addSelfDeviceMemPtr(scratch);
  if (res != ncclSuccess) {
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN("ncclDdaIpcCommInit: addSelfDeviceMemPtr failed");
    return ncclSuccess;
  }
  res = handler->exchangeMemPtrs();
  if (res != ncclSuccess) {
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN("ncclDdaIpcCommInit: exchangeMemPtrs failed");
    return ncclSuccess;
  }

  void* peerDev = nullptr;
  cudaError_t ce = cudaMalloc(&peerDev, kDdaNranks * sizeof(void*));
  if (ce != cudaSuccess) {
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN(
        "ncclDdaIpcCommInit: cudaMalloc(peer table) failed (%s)",
        cudaGetErrorString(ce));
    return ncclSuccess;
  }

  void* h_ptrs[kDdaNranks];
  for (int i = 0; i < kDdaNranks; ++i) {
    void* p = nullptr;
    res = handler->getPeerDeviceMemPtr(i, &p);
    if (res != ncclSuccess) {
      CUDACHECKIGNORE(cudaFree(peerDev));
      delete handler;
      CUDACHECKIGNORE(cudaFree(scratch));
      WARN("ncclDdaIpcCommInit: getPeerDeviceMemPtr failed");
      return ncclSuccess;
    }
    h_ptrs[i] = p;
  }

  ce = cudaMemcpy(
      peerDev,
      h_ptrs,
      kDdaNranks * sizeof(void*),
      cudaMemcpyHostToDevice);
  if (ce != cudaSuccess) {
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN(
        "ncclDdaIpcCommInit: cudaMemcpy(peer table) failed (%s)",
        cudaGetErrorString(ce));
    return ncclSuccess;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch();
  auto barrierPair = meta::comms::IpcGpuBarrier::mallocAndInit(
      kDdaNranks, nBlocksMax, comm->rank, comm->bootstrap);
  if (!barrierPair.first) {
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN("ncclDdaIpcCommInit: IpcGpuBarrier::mallocAndInit failed");
    return ncclSuccess;
  }

  auto* barrierState = new (std::nothrow) DdaIpcBarrierState();
  if (barrierState == nullptr) {
    barrierPair.first.reset();
    CUDACHECKIGNORE(cudaFree(peerDev));
    delete handler;
    CUDACHECKIGNORE(cudaFree(scratch));
    WARN("ncclDdaIpcCommInit: OOM allocating DdaIpcBarrierState");
    return ncclSuccess;
  }
  barrierState->resources = std::move(barrierPair.first);
  barrierState->barrierHost = barrierPair.second;

  comm->ddaIpcMemHandler = handler;
  comm->ddaIpcScratch = scratch;
  comm->ddaIpcScratchBytes = bytes;
  comm->ddaIpcPeerPtrsDev = peerDev;
  comm->ddaIpcBarrierState = barrierState;
  INFO(
      NCCL_INIT,
      "ncclDdaIpcCommInit: scratch %zu bytes, IpcGpuBarrier nBlocks=%d, peer IPC table on device",
      bytes,
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
  CUDACHECKIGNORE(cudaFree(comm->ddaIpcPeerPtrsDev));
  comm->ddaIpcPeerPtrsDev = nullptr;
  if (comm->ddaIpcMemHandler != nullptr) {
    delete comm->ddaIpcMemHandler;
    comm->ddaIpcMemHandler = nullptr;
  }
  CUDACHECKIGNORE(cudaFree(comm->ddaIpcScratch));
  comm->ddaIpcScratch = nullptr;
  comm->ddaIpcScratchBytes = 0;
  return ncclSuccess;
}
