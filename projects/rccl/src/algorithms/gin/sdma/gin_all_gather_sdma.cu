/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA all-gather for single-node (scaleup-only) communicators.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/gin/gin_all_gather.h"

#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstdint>

NCCL_PARAM(GinAllGatherEnable, "GIN_ALLGATHER_ENABLE", 1)
NCCL_PARAM(GinAllGatherMinBytes, "GIN_ALLGATHER_MIN_BYTES", 0)
// Per-rank bytes at or above which SDMA beats direct LSA stores.
NCCL_PARAM(GinAllGatherSdmaMinBytes, "GIN_ALLGATHER_SDMA_MIN_BYTES", 8 * 1024 * 1024)

namespace {

constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);
// markSdmaDirty packs (peer, channel) into a 64-bit mask.
constexpr int kGinAllGatherMaxRanks = 16;
constexpr int kGinAllGatherSdmaCtas = 1;
constexpr int kGinAllGatherSdmaMinThreads = 64;
constexpr int kGinAllGatherSdmaMaxThreads = 256;

inline int ginAllGatherSdmaThreads(int nRanks) {
  int threads = nRanks < kGinAllGatherSdmaMinThreads ? kGinAllGatherSdmaMinThreads : nRanks;
  return threads > kGinAllGatherSdmaMaxThreads ? kGinAllGatherSdmaMaxThreads : threads;
}

struct GinAllGatherLsaBand {
  size_t maxBytes;
  int chunks;
  int threads;
};

constexpr GinAllGatherLsaBand kGinAllGatherLsaBands[] = {
  {16 * 1024, 2, 256}, {64 * 1024, 4, 256}, {512 * 1024, 8, 256}, {SIZE_MAX, 8, 512},
};

inline size_t ginAllGatherDivUp(size_t a, size_t b) { return (a + b - 1) / b; }
inline size_t ginAllGatherAlignUp(size_t a, size_t b) { return ginAllGatherDivUp(a, b) * b; }

__device__ __forceinline__ void ginAllGatherCopyCta(char* dst, const char* src, size_t bytes) {
  const size_t nVec = bytes / sizeof(uint4);
  for (size_t i = threadIdx.x; i < nVec; i += blockDim.x) ((uint4*)dst)[i] = ((const uint4*)src)[i];
  for (size_t i = nVec * sizeof(uint4) + threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
}

static_assert(kGinAllGatherMaxCtas >= kGinAllGatherMaxRanks, "LSA grid needs a chunk per peer");

void ginAllGatherLsaLaunchConfig(int nRanks, size_t bytes, int* chunks, int* threads, size_t* chunkBytes) {
  const GinAllGatherLsaBand* band = kGinAllGatherLsaBands;
  while (bytes >= band->maxBytes) band++;
  const int maxChunks = kGinAllGatherMaxCtas / nRanks;
  *chunks = band->chunks > maxChunks ? maxChunks : band->chunks;
  *threads = band->threads;
  *chunkBytes = ginAllGatherAlignUp(ginAllGatherDivUp(bytes, *chunks), sizeof(uint4));
}

// Every rank writes its local send slice to rank-indexed receive storage on every peer.
// sendOff is therefore invariant across peers, unlike all-to-all.
template <bool UseSdma>
__global__ void ncclGinAllGatherKernel(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin, size_t recvOff,
                                       size_t bytes, size_t chunkBytes, struct ncclDevComm devComm) {
  if constexpr (UseSdma) {
    if (blockIdx.x != 0) return;
    constexpr unsigned int slot = 0;
    ncclGin gin{devComm, 0};
    const uint64_t signalValue = gin.readSignal(slot);
    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, slot};
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);
    for (int peer = threadIdx.x; peer < devComm.nRanks; peer += blockDim.x) {
      gin.put(ncclTeamWorld(devComm), peer, recvWin, recvOff + devComm.rank * bytes, sendWin, sendOff, bytes,
              ncclGin_WeakSignalInc{slot});
    }
    gin.waitSignal(ncclCoopCta(), slot, signalValue + devComm.nRanks);
    gin.flush(ncclCoopCta());
    bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
  } else {
    const unsigned int ctaIndex = blockIdx.y * gridDim.x + blockIdx.x;
    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamTagLsa(), ctaIndex};
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire);
    const int peer = blockIdx.x;
    const size_t off = chunkBytes * blockIdx.y;
    if (off < bytes) {
      const size_t chunk = bytes - off > chunkBytes ? chunkBytes : bytes - off;
      char* dst = (char*)ncclGetLsaPointer(recvWin, recvOff + devComm.rank * bytes + off, peer);
      const char* src = (const char*)ncclGetLocalPointer(sendWin, sendOff + off);
      ginAllGatherCopyCta(dst, src, chunk);
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

ncclResult_t ncclGinAllGatherInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  ncclGinAllGatherState* state = &comm->ginAllGatherState;
  if (!state->initialized) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kGinAllGatherMaxCtas;
    reqs.ginSignalCount = kGinAllGatherMaxCtas;
    reqs.barrierCount = kGinAllGatherSdmaCtas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, true, NCCL_VERSION_CODE));
    state->initialized = true;
  }
  return ncclSuccess;
}

} // namespace

bool ncclAllGatherGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype) {
  if (!kSdmaDeviceBackendCompiled || !ncclParamGinAllGatherEnable()) return false;
  if (comm == nullptr || comm->bootstrap == nullptr || sendbuff == nullptr || recvbuff == nullptr || count == 0)
    return false;
  if (comm->nNodes != 1 || comm->nRanks <= 0 || comm->nRanks > kGinAllGatherMaxRanks || !comm->symmetricSupport ||
      comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL)
    return false;
  ncclGinType_t ginType = NCCL_GIN_TYPE_NONE;
  if (ncclGetGinType(comm, &ginType) != ncclSuccess ||
      ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA ||
      ncclTeamLsa(comm).nRanks != comm->nRanks)
    return false;

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes < (size_t)ncclParamGinAllGatherMinBytes()) return false;
  ncclDevrWindow* sendWin = nullptr;
  ncclDevrWindow* recvWin = nullptr;
  if (ncclDevrFindWindow(comm, sendbuff, &sendWin) != ncclSuccess ||
      ncclDevrFindWindow(comm, recvbuff, &recvWin) != ncclSuccess || sendWin == nullptr || recvWin == nullptr)
    return false;
  const size_t sendOff = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
  const size_t recvOff = (const uint8_t*)recvbuff - (const uint8_t*)recvWin->userPtr;
  // Allgather receive storage includes a slice for every rank.
  const size_t sendSafeSize = ncclDevrWindowPeerSafeSize(sendWin);
  const size_t recvSafeSize = ncclDevrWindowPeerSafeSize(recvWin);
  if (sendOff > sendSafeSize || bytes > sendSafeSize - sendOff || recvOff > recvSafeSize ||
      bytes > (recvSafeSize - recvOff) / (size_t)comm->nRanks)
    return false;
  if (bytes < (size_t)ncclParamGinAllGatherSdmaMinBytes() && bytes >= sizeof(uint4) &&
      ((sendOff | recvOff | bytes) & (sizeof(uint4) - 1)) != 0)
    return false;
  return true;
}

ncclResult_t ncclAllGatherGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                  ncclComm* comm, cudaStream_t stream) {
  ncclDevrWindow* sendWin = nullptr;
  ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) return ncclInvalidUsage;
  NCCLCHECK(ncclGinAllGatherInitOnce(comm));
  const size_t sendOff = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
  const size_t recvOff = (uint8_t*)recvbuff - (uint8_t*)recvWin->userPtr;
  const size_t bytes = count * ncclTypeSize(datatype);
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (bytes >= (size_t)ncclParamGinAllGatherSdmaMinBytes()) {
    const int threads = ginAllGatherSdmaThreads(comm->nRanks);
    INFO(NCCL_COLL, "AllGather GIN: transport=sdma bytes=%zu threads=%d", bytes, threads);
    hipExtLaunchKernelGGL((ncclGinAllGatherKernel<true>), kGinAllGatherSdmaCtas, threads, 0, stream, nullptr,
                          stopEvent, 0, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, bytes, 0,
                          comm->ginAllGatherState.devComm);
  } else {
    int chunks, threads;
    size_t chunkBytes;
    ginAllGatherLsaLaunchConfig(comm->nRanks, bytes, &chunks, &threads, &chunkBytes);
    INFO(NCCL_COLL, "AllGather GIN: transport=lsa bytes=%zu chunks=%d ctas=%d threads=%d", bytes, chunks,
         comm->nRanks * chunks, threads);
    hipExtLaunchKernelGGL((ncclGinAllGatherKernel<false>), dim3(comm->nRanks, chunks), threads, 0, stream, nullptr,
                          stopEvent, 0, sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, bytes, chunkBytes,
                          comm->ginAllGatherState.devComm);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclGinAllGatherFinalize(ncclComm* comm) {
  ncclGinAllGatherState* state = &comm->ginAllGatherState;
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  return ncclSuccess;
}
