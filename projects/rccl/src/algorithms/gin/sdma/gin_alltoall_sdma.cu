/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * GIN-SDMA all-to-all for single-node (scaleup-only) communicators.
 *
 * This translation unit is compiled with NCCL_GIN_ANVIL_SDMA_ENABLE=1 and
 * NCCL_GIN_PROXY_ENABLE=0 so that NCCL_GIN_BACKEND_MASK_ALL has a single bit
 * set and ncclGinCallImpl resolves the backend at compile time.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/gin/gin_alltoall.h"

#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "param.h"

#include <cuda_runtime.h>

#include <cstdint>

NCCL_PARAM(GinA2AEnable, "GIN_A2A_ENABLE", 1)
NCCL_PARAM(GinA2AMinBytes, "GIN_A2A_MIN_BYTES", 0)

// Per-peer bytes at or above which SDMA beats direct LSA stores.
// On 8 ranks that is a 64 MiB message, so LSA covers everything up to 32 MiB.
NCCL_PARAM(GinA2ASdmaMinBytes, "GIN_A2A_SDMA_MIN_BYTES", 8 * 1024 * 1024)

namespace {

// True only when the SDMA device templates were compiled into this TU.
constexpr bool kSdmaDeviceBackendCompiled = (NCCL_GIN_ANVIL_SDMA_ENABLE != 0);

// markSdmaDirty packs (peer, channel) into a 64-bit mask and drops anything past
// bit 64. The scaleup path uses one channel, so this bound keeps us in range.
// SDMA puts one peer per thread and strides if there are more peers than threads.
constexpr int kGinA2AMaxRanks = 16;

// This path only queues one put per peer and the SDMA engines do the copying.
// That is little enough work for a single CTA, and more would not copy faster.
constexpr int kGinA2ASdmaCtas = 1;

// One put per thread, so the block only has to cover the peers. Rounded up to a
// full wavefront instead of leaving a partial one.
constexpr int kGinA2ASdmaMinThreads = 64;

inline int ginA2ASdmaThreads(int nRanks) {
  int threads = nRanks < kGinA2ASdmaMinThreads ? kGinA2ASdmaMinThreads : nRanks;
  return threads > kGinA2AThreadsPerCta ? kGinA2AThreadsPerCta : threads;
}

// Best measured LSA geometry per size range.
struct GinA2ALsaBand {
  size_t maxBytesPerPeer;
  int chunksPerPeer;
  int threadsPerCta;
};

// Totals are for 8 ranks, the lookup itself keys on the per-peer bound.
constexpr GinA2ALsaBand kGinA2ALsaBands[] = {
  {16 * 1024, 2, 256},    // up to 16 KB/peer, 64 KB total: 2 chunks, 256 threads
  {64 * 1024, 4, 256},    // 16 to 64 KB/peer, 128 KB to 256 KB total: 4 chunks, 256 threads
  {512 * 1024, 8, 256},   // 64 to 512 KB/peer, 512 KB to 2 MB total: 8 chunks, 256 threads
  {SIZE_MAX, 8, 512},     // 512 KB/peer and up, 4 MB total and up: 8 chunks, 512 threads
};

inline size_t ginA2ADivUp(size_t a, size_t b) { return (a + b - 1) / b; }
inline size_t ginA2AAlignUp(size_t a, size_t b) { return ginA2ADivUp(a, b) * b; }

// Whole CTA cooperates on one chunk, 16B vectors with a scalar tail. Eligibility
// rejects anything that would leave these two pointers unaligned.
__device__ __forceinline__ void ginA2ACopyCta(char* dst, const char* src, size_t bytes) {
  size_t nVec = bytes / sizeof(uint4);
  for (size_t i = threadIdx.x; i < nVec; i += blockDim.x) {
    ((uint4*)dst)[i] = ((const uint4*)src)[i];
  }
  for (size_t i = nVec * sizeof(uint4) + threadIdx.x; i < bytes; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Keeps the maxChunks division below from rounding down to an empty grid.
static_assert(kGinA2AMaxCtas >= kGinA2AMaxRanks, "LSA grid needs a chunk per peer");

void ginA2ALsaLaunchConfig(int nRanks, size_t bytesPerPeer, int* chunksPerPeer, int* threads, size_t* chunkBytes) {
  const GinA2ALsaBand* band = kGinA2ALsaBands;
  while (bytesPerPeer >= band->maxBytesPerPeer) band++;

  // The grid is nRanks * chunks and every CTA takes a barrier and a signal slot.
  int maxChunks = kGinA2AMaxCtas / nRanks;
  int chunks = band->chunksPerPeer > maxChunks ? maxChunks : band->chunksPerPeer;

  *chunksPerPeer = chunks;
  *threads = band->threadsPerCta;
  *chunkBytes = ginA2AAlignUp(ginA2ADivUp(bytesPerPeer, chunks), sizeof(uint4));
}

// Acquire keeps a rank from writing before every peer is ready, release publishes those writes before any peer reads.
template <bool UseSdma>
__global__ void ncclGinA2AKernel(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin, size_t recvOff,
                                 size_t bytesPerPeer, size_t chunkBytes, struct ncclDevComm devComm) {
  if constexpr (UseSdma) {
    if (blockIdx.x != 0) return;

    constexpr int ginContext = 0;
    constexpr unsigned int slot = 0;
    ncclGin gin{devComm, ginContext};
    uint64_t signalValue = gin.readSignal(slot);

    // The puts span the world team, so the barrier has to as well.
    ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, slot};
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire, ncclGinFenceLevel::None);

    for (int peer = threadIdx.x; peer < devComm.nRanks; peer += blockDim.x) {
      gin.put(ncclTeamWorld(devComm), peer, recvWin, recvOff + devComm.rank * bytesPerPeer, sendWin,
              sendOff + peer * bytesPerPeer, bytesPerPeer, ncclGin_SignalInc{slot});
    }

    gin.waitSignal(ncclCoopCta(), slot, signalValue + devComm.nRanks);
    gin.flush(ncclCoopCta());

    bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::None);
  } else {
    // LSA runs a 2D (peer, chunk) grid, so flatten it for the per-CTA slots.
    unsigned int ctaIndex = blockIdx.y * gridDim.x + blockIdx.x;

    ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamTagLsa(), ctaIndex};
    bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

    // blockIdx.x picks the peer, blockIdx.y the chunk of that peer's slice.
    int r = blockIdx.x;
    size_t off = chunkBytes * blockIdx.y;

    if (off < bytesPerPeer) {
      size_t bytes = bytesPerPeer - off;
      if (bytes > chunkBytes) bytes = chunkBytes;

      char* dst = (char*)ncclGetLsaPointer(recvWin, recvOff + devComm.rank * bytesPerPeer + off, r);
      const char* src = (const char*)ncclGetLocalPointer(sendWin, sendOff + r * bytesPerPeer + off);
      ginA2ACopyCta(dst, src, bytes);
    }

    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }
}

ncclResult_t ncclGinA2AInitOnce(ncclComm* comm) {
  NCCLCHECK(ncclDevrInitOnce(comm));
  struct ncclGinA2AState* state = &comm->ginA2AState;
  if (!state->initialized) {
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = kGinA2AMaxCtas;
    reqs.ginSignalCount = kGinA2AMaxCtas;
    reqs.barrierCount = kGinA2ASdmaCtas;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &state->devComm, /*isInternal=*/true));
    state->initialized = true;
  }
  return ncclSuccess;
}

} // namespace

bool ncclAllToAllGinSdmaEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype) {
  if (!kSdmaDeviceBackendCompiled) return false;
  if (!ncclParamGinA2AEnable()) return false;
  if (comm == nullptr || comm->bootstrap == nullptr) return false;
  if (count == 0) return false;

  // Scaleup only.
  if (comm->nNodes != 1) return false;
  if (comm->nRanks > kGinA2AMaxRanks) return false;

  if (!comm->symmetricSupport) return false;
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;

  // Every rank must be reachable over LSA.
  if (ncclTeamLsa(comm).nRanks != comm->nRanks) return false;

  // This path runs on the comm's shared GIN backend, so it has to be SDMA.
  if (comm->sharedRes->ginState.ginType != (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA) return false;

  // sendbuff == recvbuff is accepted. ncclAlltoAll defines no in-place semantics
  // (unlike ncclAlltoAllv), and the reference GIN kernel likewise runs the same
  // puts with the send and receive windows aliased.
  size_t bytesPerPeer = count * ncclTypeSize(datatype);
  if (bytesPerPeer < (size_t)ncclParamGinA2AMinBytes()) return false;

  // Both buffers must already be symmetrically registered.
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  if (ncclDevrFindWindow(comm, sendbuff, &sendWin) != ncclSuccess || sendWin == nullptr) return false;
  if (ncclDevrFindWindow(comm, recvbuff, &recvWin) != ncclSuccess || recvWin == nullptr) return false;

  // The LSA copy stores 16B vectors, so the window offsets and the per-peer stride
  // have to be aligned. Below one vector the loop never runs, and SDMA is exempt.
  if (bytesPerPeer < (size_t)ncclParamGinA2ASdmaMinBytes() && bytesPerPeer >= sizeof(uint4)) {
    size_t sendOff = (uint8_t*)sendbuff - (uint8_t*)sendWin->userPtr;
    size_t recvOff = (uint8_t*)recvbuff - (uint8_t*)recvWin->userPtr;
    if (((sendOff | recvOff | bytesPerPeer) & (sizeof(uint4) - 1)) != 0) return false;
  }

  return true;
}

ncclResult_t ncclAllToAllGinSdma(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                 ncclComm* comm, cudaStream_t stream) {
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  NCCLCHECK(ncclDevrFindWindow(comm, sendbuff, &sendWin));
  NCCLCHECK(ncclDevrFindWindow(comm, recvbuff, &recvWin));
  if (sendWin == nullptr || recvWin == nullptr) return ncclInvalidUsage;

  NCCLCHECK(ncclGinA2AInitOnce(comm));

  size_t sendOff = (uint8_t*)sendbuff - (uint8_t*)sendWin->userPtr;
  size_t recvOff = (uint8_t*)recvbuff - (uint8_t*)recvWin->userPtr;
  size_t bytesPerPeer = count * ncclTypeSize(datatype);

  bool useSdma = bytesPerPeer >= (size_t)ncclParamGinA2ASdmaMinBytes();

  if (useSdma) {
    int sdmaThreads = ginA2ASdmaThreads(comm->nRanks);
    INFO(NCCL_COLL, "AllToAll GIN: transport=sdma bytesPerPeer=%zu threads=%d", bytesPerPeer, sdmaThreads);
    ncclGinA2AKernel<true><<<kGinA2ASdmaCtas, sdmaThreads, 0, stream>>>(
      sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, bytesPerPeer, 0, comm->ginA2AState.devComm);
  } else {
    // Under LSA the CTAs do the copying, so the grid scales with the message.
    int lsaChunks, lsaThreads;
    size_t lsaChunkBytes;
    ginA2ALsaLaunchConfig(comm->nRanks, bytesPerPeer, &lsaChunks, &lsaThreads, &lsaChunkBytes);

    INFO(NCCL_COLL, "AllToAll GIN: transport=lsa bytesPerPeer=%zu chunks=%d ctas=%d threads=%d", bytesPerPeer,
         lsaChunks, comm->nRanks * lsaChunks, lsaThreads);
    ncclGinA2AKernel<false><<<dim3(comm->nRanks, lsaChunks), lsaThreads, 0, stream>>>(
      sendWin->vidmem, sendOff, recvWin->vidmem, recvOff, bytesPerPeer, lsaChunkBytes, comm->ginA2AState.devComm);
  }
  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

ncclResult_t ncclGinA2AFinalize(ncclComm* comm) {
  struct ncclGinA2AState* state = &comm->ginA2AState;
  if (state->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &state->devComm));
    state->initialized = false;
  }
  return ncclSuccess;
}
