/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce_ipc.h"

#include "algorithms/CollCommon.h"
#include "algorithms/all_reduce/all_reduce_dda.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "ipc_mem_handler.h"
#include "ipc_gpu_barrier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

namespace {

constexpr int kDdaNranks = 8;
/** Flat below this size; tree above (see ddaAllReduceFlatIpc / ddaAllReduceTreeIpc). */
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 20;


struct DdaIpcBarrierState {
  std::unique_ptr<meta::comms::IpcGpuBarrierResources> resources;
  meta::comms::IpcGpuBarrier barrierHost;
};

/** Max grid.x for any element count that fits in scratch (covers float / half / bf16). */
static int ddaMaxNBlocksForScratch(size_t scratchBytes) {
  constexpr unsigned threads = 512;
  unsigned maxBlocks = 1;
  {
    size_t maxCount = scratchBytes / sizeof(float);
    size_t denom = (size_t)threads * (sizeof(uint4) / sizeof(float));
    unsigned nb = (unsigned)((maxCount + denom - 1) / denom);
    if (nb > maxBlocks) {
      maxBlocks = nb;
    }
  }
  {
    size_t maxCount = scratchBytes / sizeof(half);
    size_t denom = (size_t)threads * (sizeof(uint4) / sizeof(half));
    unsigned nb = (unsigned)((maxCount + denom - 1) / denom);
    if (nb > maxBlocks) {
      maxBlocks = nb;
    }
  }
  return static_cast<int>(maxBlocks);
}

static size_t ddaIpcScratchBytesFromEnv() {
  const char* e = getenv("RCCL_DDA_IPC_BYTES");
  if (e == nullptr || e[0] == '\0') {
    return 64ULL * 1024 * 1024;
  }
  char* end = nullptr;
  unsigned long long v = strtoull(e, &end, 0);
  if (end == e) {
    return 64ULL * 1024 * 1024;
  }
  return static_cast<size_t>(v);
}

template <typename T>
static ncclResult_t ncclAllReduceDdaIpcTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return ncclInvalidUsage;
  }
  if (count * sizeof(T) > comm->ddaIpcScratchBytes) {
    WARN(
        "DDA IPC allreduce: element count %zu needs %zu bytes; comm scratch is %zu (set RCCL_DDA_IPC_BYTES)",
        count,
        count * sizeof(T),
        comm->ddaIpcScratchBytes);
    return ncclInvalidArgument;
  }

  const size_t sizeBytes = count * sizeof(T);
  const size_t countPerThread = sizeof(uint4) / sizeof(T);
  const unsigned threads = 512;
  const size_t denom = (size_t)threads * countPerThread;
  //unsigned nblocks = (unsigned)((count + denom - 1) / denom);
  const bool wantTree = sizeBytes > kDdaFlatTreeThresholdBytes;
  const bool treeOk =
      wantTree && (count % static_cast<size_t>(kDdaNranks) == 0);
  
  if (wantTree && !treeOk) {
    INFO(
        NCCL_ALL,
        "DDA IPC: size %zu B > 1 MiB but count %zu not divisible by %d; using flat kernel",
        sizeBytes,
        count,
        kDdaNranks);
  }

  unsigned nblocks;
  if (treeOk) {
    const size_t countPerRank = count / static_cast<size_t>(kDdaNranks);
    nblocks = (unsigned)((countPerRank + denom - 1) / denom);
  } else {
    nblocks = (unsigned)((count + denom - 1) / denom);
  }

  if (nblocks < 1) {
    nblocks = 1;
  }

  const int nBlocksMax = ddaMaxNBlocksForScratch(comm->ddaIpcScratchBytes);
  if (static_cast<int>(nblocks) > nBlocksMax) {
    WARN(
        "DDA IPC allreduce: grid %u exceeds init max %d (scratch %zu bytes)",
        nblocks,
        nBlocksMax,
        comm->ddaIpcScratchBytes);
    return ncclInternalError;
  }

  auto* barrierState =
      static_cast<DdaIpcBarrierState*>(comm->ddaIpcBarrierState);
  meta::comms::IpcGpuBarrier barrierHost = barrierState->barrierHost;

  void* peerPtrsDev = comm->ddaIpcPeerPtrsDev;
  T** d_ipcbuffs = reinterpret_cast<T**>(peerPtrsDev);

  dim3 grid(nblocks);
  dim3 block(threads);

  if (treeOk) {
    CUDACHECK(cudaMemcpyAsync(
        comm->ddaIpcScratch,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream));	  
    meta::comms::ddaAllReduceTreeIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  } else {
    meta::comms::ddaAllReduceFlatIpc<T, kDdaNranks, false>
        <<<grid, block, 0, stream>>>(
            d_ipcbuffs,
            static_cast<T*>(recvbuff),
            count,
            static_cast<const T*>(sendbuff),
            comm->rank,
            barrierHost,
            nullptr);
  }

  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaStreamSynchronize(stream));

  return ncclSuccess;
}

} // namespace

ncclResult_t ncclDdaIpcCommInit(ncclComm* comm) {
  if (comm == nullptr) {
    return ncclSuccess;
  }
  if (comm->nRanks != kDdaNranks || comm->nNodes != 1 ||
      comm->bootstrap == nullptr) {
    return ncclSuccess;
  }

  size_t bytes = ddaIpcScratchBytesFromEnv();
  if (bytes == 0) {
    return ncclSuccess;
  }

  void* scratch = nullptr;
  cudaError_t ce = cudaMalloc(&scratch, bytes);
  if (ce != cudaSuccess) {
    WARN(
        "ncclDdaIpcCommInit: cudaMalloc(%zu) failed (%s)",
        bytes,
        cudaGetErrorString(ce));
    return ncclSuccess;
  }

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
  ce = cudaMalloc(&peerDev, kDdaNranks * sizeof(void*));
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

  const int nBlocksMax = ddaMaxNBlocksForScratch(bytes);
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

bool ncclAllReduceDdaIpcEligible(
    ncclComm* comm,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaIpcMemHandler == nullptr || comm->ddaIpcScratch == nullptr ||
      comm->ddaIpcPeerPtrsDev == nullptr || comm->ddaIpcBarrierState == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nNodes != 1) {
    return false;
  }
  if (comm->nRanks != kDdaNranks) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }
  size_t need = count * 4;
  if (datatype == ncclFloat16 || datatype == ncclBfloat16) {
    need = count * 2;
  }
  if (need > comm->ddaIpcScratchBytes) {
    return false;
  }
  return true;
}

ncclResult_t ncclAllReduceDdaIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaIpcTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaIpcTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaIpcTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
