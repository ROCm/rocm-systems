/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA all-gather for single-node communicators. gfx950 is on by default.
 * gfx1250, including DPX, stays opt-in (NCCL_GIN_ALLGATHER_GFX1250_ENABLE=1).
 * The kernel runs only for a single-segment NCCL_WIN_COLL_SYMMETRIC device
 * window; multi-segment and host-backed registrations stay on the non-GIN path.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/gin/gin_all_gather.h"

#include "archinfo.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

// group.h is host-only. The selector and this gate both read the same thread-local.
extern thread_local int ncclGroupDepth;

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstdint>

NCCL_PARAM(GinAllGatherEnable, "GIN_ALLGATHER_ENABLE", 1)
NCCL_PARAM(GinAllGatherMinBytes, "GIN_ALLGATHER_MIN_BYTES", 0)
// gfx950 at a full node is the measured case. gfx1250, including DPX, stays
// on the pre-existing AllGather selectors until it has data of its own.
NCCL_PARAM(GinAllGatherGfx1250Enable, "GIN_ALLGATHER_GFX1250_ENABLE", 0)
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

__device__ __forceinline__ void ginAllGatherCopyBytes(char* dst, const char* src, size_t begin, size_t end) {
  for (size_t i = begin + threadIdx.x; i < end; i += blockDim.x) dst[i] = src[i];
}

// Same tiers as bcastLsa: a byte head to 16-byte alignment, then 16-byte or
// 4-byte packs chosen by the source/destination offset difference, then a byte
// tail. A rank whose local slice is not 16-byte aligned still stays on this
// kernel instead of falling back to a byte-at-a-time copy of the whole slice.
__device__ __forceinline__ void ginAllGatherCopyCta(char* dst, const char* src, size_t bytes) {
  const uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
  const uint32_t alignDiff = static_cast<uint32_t>(srcAddr - dstAddr);
  size_t nPre = static_cast<size_t>((16u - (dstAddr & 15u)) & 15u);
  if (nPre > bytes) nPre = bytes;
  ginAllGatherCopyBytes(dst, src, 0, nPre);
  size_t cursor = nPre;
  if ((alignDiff % 16u) == 0) {
    const size_t nVec = (bytes - cursor) / sizeof(uint4);
    const uint4* in = reinterpret_cast<const uint4*>(src + cursor);
    uint4* out = reinterpret_cast<uint4*>(dst + cursor);
    for (size_t i = threadIdx.x; i < nVec; i += blockDim.x) out[i] = in[i];
    cursor += nVec * sizeof(uint4);
  } else if ((alignDiff % 4u) == 0) {
    const size_t nVec = (bytes - cursor) / sizeof(uint32_t);
    const uint32_t* in = reinterpret_cast<const uint32_t*>(src + cursor);
    uint32_t* out = reinterpret_cast<uint32_t*>(dst + cursor);
    for (size_t i = threadIdx.x; i < nVec; i += blockDim.x) out[i] = in[i];
    cursor += nVec * sizeof(uint32_t);
  }
  ginAllGatherCopyBytes(dst, src, cursor, bytes);
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
    // In-place allgather already holds this rank's slice in the receive window.
    const bool skipSelf = sendWin == recvWin && sendOff == recvOff + static_cast<size_t>(devComm.rank) * bytes;
    for (int peer = threadIdx.x; peer < devComm.nRanks; peer += blockDim.x) {
      if (skipSelf && peer == devComm.rank) continue;
      gin.put(ncclTeamWorld(devComm), peer, recvWin, recvOff + devComm.rank * bytes, sendWin, sendOff, bytes,
              ncclGin_WeakSignalInc{slot});
    }
    const uint64_t nSignals = static_cast<uint64_t>(devComm.nRanks) - (skipSelf ? 1 : 0);
    gin.waitSignal(ncclCoopCta(), slot, signalValue + nSignals);
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
      // In-place: this rank's slice already lives at dst.
      if (dst != src) ginAllGatherCopyCta(dst, src, chunk);
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
    // This TU is compiled for SDMA only, so a fallback backend's handles will not do.
    reqs.ginType = NCCL_GIN_TYPE_ANVIL_SDMA;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, true, NCCL_VERSION_CODE));
    state->initialized = true;
  }
  return ncclSuccess;
}

} // namespace

namespace {

bool ginAllGatherRangeFits(size_t off, size_t bytes, size_t limit) {
  return off <= limit && bytes <= limit - off;
}

// HIP GIN registration is single-segment device memory. Host-backed and
// multi-segment windows stay on the non-GIN allgather path.
bool ginAllGatherWindowOk(struct ncclDevrWindow* win) {
  return win != nullptr && (win->winFlags & NCCL_WIN_COLL_SYMMETRIC) != 0 && !ncclDevrWindowIsMultiSegment(win) &&
         !ncclDevrWindowHasSysmemSegment(win);
}

} // namespace

bool ncclAllGatherGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                  ncclDataType_t datatype) {
  if (!kSdmaDeviceBackendCompiled || !ncclParamGinAllGatherEnable()) return false;
  if (comm == nullptr || comm->bootstrap == nullptr || sendbuff == nullptr || recvbuff == nullptr || count == 0) {
    return false;
  }
  // gfx950 is the measured path. gfx1250, including DPX, stays opt-in.
  // Grouped and implicit-order calls stay out, matching DDA: the first grouped
  // call would block in device-comm bootstrap before the other ranks submit.
  const bool gfx950 = IsArchMatch(comm->archName, "gfx950");
  const bool gfx1250 = IsArchMatch(comm->archName, "gfx1250");
  if (!gfx950 && !gfx1250) return false;
  if (gfx1250 && ncclParamGinAllGatherGfx1250Enable() == 0) return false;
  if (ncclGroupDepth != 0 || comm->config.launchOrderImplicit == 1) return false;
  if (comm->nNodes != 1 || comm->nRanks <= 0 || comm->nRanks > kGinAllGatherMaxRanks || !comm->symmetricSupport ||
      comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) {
    return false;
  }
  ncclGinType_t ginType = NCCL_GIN_TYPE_NONE;
  if (ncclGetGinType(comm, &ginType) != ncclSuccess || ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA ||
      ncclTeamLsa(comm).nRanks != comm->nRanks) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes < (size_t)ncclParamGinAllGatherMinBytes()) return false;
  ncclDevrWindow* sendWin = nullptr;
  ncclDevrWindow* recvWin = nullptr;
  if (ncclDevrFindWindow(comm, sendbuff, &sendWin) != ncclSuccess ||
      ncclDevrFindWindow(comm, recvbuff, &recvWin) != ncclSuccess || !ginAllGatherWindowOk(sendWin) ||
      !ginAllGatherWindowOk(recvWin)) {
    return false;
  }
  const size_t sendOff = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
  const size_t recvOff = (const uint8_t*)recvbuff - (const uint8_t*)recvWin->userPtr;
  // The send slice is read locally, so a smaller peer allocation must not reject it.
  // Receive storage is written on every peer and has to fit in the smallest one.
  // Alignment is not a gate: ranks would otherwise disagree on in-place offsets.
  if (bytes > SIZE_MAX / (size_t)comm->nRanks) return false;
  const size_t recvBytes = bytes * (size_t)comm->nRanks;
  if (!ginAllGatherRangeFits(sendOff, bytes, sendWin->size) ||
      !ginAllGatherRangeFits(recvOff, recvBytes, ncclDevrWindowPeerSafeSize(recvWin))) {
    return false;
  }
  return true;
}

int ncclAllGatherGinSdmaCtaCount(int nRanks, size_t bytes) {
  if (nRanks <= 0) return 0;
  if (bytes >= (size_t)ncclParamGinAllGatherSdmaMinBytes()) return kGinAllGatherSdmaCtas;
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  ginAllGatherLsaLaunchConfig(nRanks, bytes, &chunks, &threads, &chunkBytes);
  return nRanks * chunks;
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
