/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Direct LSA-flat-from-fabric AllToAll device body for GIN-SDMA Test#5.
 * Requires nccl_device.h (or equivalent) before this include.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_

#include <cstddef>
#include <cstdint>

#include "gin_fabric_lsa_policy.h"

namespace gin {
namespace fabric {

// 16-byte store into a peer's receive window. The system-scope store writes
// through to the peer, which is what lets ginFabricLsaA2APublish skip the
// release fence.
__device__ __forceinline__ void ginFabricLsaA2AStore16(uint4* dst, uint4 v) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    uint4 u;
    v4u v;
  } cv;
  cv.u = v;
  __builtin_amdgcn_global_store_b128((v4u_gptr)dst, cv.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  *dst = v;
#endif
}

// One CTA's copy. Issuing all kUnroll loads before their stores keeps several
// remote stores in flight per thread.
template <int kUnroll>
__device__ __forceinline__ void ginFabricLsaA2ACopy16(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n) {
  for (size_t i = threadIdx.x; i < n; i += kUnroll * blockDim.x) {
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * blockDim.x < n) v[k] = s[i + k * blockDim.x];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * blockDim.x < n) ginFabricLsaA2AStore16(&d[i + k * blockDim.x], v[k]);
    }
  }
}

// Window offsets can leave either pointer off a 16-byte boundary, and a uint4
// access there faults on gfx1250, so such a copy takes the byte loop. Returns
// false if it did: those plain stores need the release fence before a flag.
__device__ __forceinline__ bool ginFabricLsaA2ACopyCta(char* dst, const char* src, size_t bytes) {
  if ((((uintptr_t)dst | (uintptr_t)src | bytes) & (sizeof(uint4) - 1)) == 0) {
    // A destination that starts mid cache line makes every wavefront's stores
    // straddle lines; peel up to 112 bytes so the batched copy writes whole lines.
    const size_t head = min(bytes, (size_t)(-(uintptr_t)dst & 127));
    if (threadIdx.x < head / sizeof(uint4)) {
      ginFabricLsaA2AStore16(&((uint4*)dst)[threadIdx.x], ((const uint4*)src)[threadIdx.x]);
    }
    ginFabricLsaA2ACopy16<8>((const uint4*)(src + head), (uint4*)(dst + head), (bytes - head) / sizeof(uint4));
    return true;
  }
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
  return false;
}

// Makes this CTA's stores visible before the done flag that follows. The
// device-scope release fence writes back L2 in every CTA; after system-scope
// streaming stores, each wave only has to wait for its own stores.
__device__ __forceinline__ void ginFabricLsaA2APublish(bool streamed) {
  __syncthreads();
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  if (streamed) {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  } else
#endif
    cuda::atomic_thread_fence(cuda::memory_order_release);
  __syncthreads();
}

// 2D grid: blockIdx.x == peer, blockIdx.y == chunk. Extra CTAs from a padded
// host grid (gin.put / LL) return before touching the flags. CTA (peer, chunk)
// on every rank writes only to `peer`, so each CTA index is a Gather rooted at
// `peer`: the root's CTA flags its same-index CTA on every sender ready, each
// sender copies and answers with a done flag once its stores are visible, and
// only the root waits for all of them. `flags` comes from
// ncclLLA2ACreateRequirement(ginFabricLsaA2AFlagBlocks(nRanks), nRanks); its
// epoch-tagged slots keep a sender within one launch of the root. Past
// kGinFabricLsaA2AMaxCtas ranks the CTA indices would run past the flag blocks,
// so the body does nothing there; ginFabricLsaAlltoAllSizeOk keeps callers off
// that path.
__device__ __forceinline__ void ginFabricLsaAlltoAllBody(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin,
                                                         size_t recvOff, size_t bytesPerPeer,
                                                         struct ncclDevComm devComm, ncclLLA2AHandle flags) {
  if (devComm.nRanks > kGinFabricLsaA2AMaxCtas) return;
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  ginFabricLsaA2ALaunchConfig(devComm.nRanks, bytesPerPeer, &chunks, &threads, &chunkBytes);
  (void)threads;
  if ((int)blockIdx.x >= devComm.nRanks || (int)blockIdx.y >= chunks) return;

  const unsigned int ctaIndex = (unsigned int)blockIdx.y * (unsigned int)devComm.nRanks + (unsigned int)blockIdx.x;
  const int peer = (int)blockIdx.x;
  const int rank = devComm.rank;
  const size_t off = chunkBytes * (size_t)blockIdx.y;
  size_t bytes = off < bytesPerPeer ? bytesPerPeer - off : 0;
  if (bytes > chunkBytes) bytes = chunkBytes;
  char* dst = (char*)ncclGetLsaPointer(recvWin, recvOff + (size_t)rank * bytesPerPeer + off, peer);
  const char* src = (const char*)ncclGetLocalPointer(sendWin, sendOff + (size_t)peer * bytesPerPeer + off);

  ncclTeam lsa = ncclTeamLsa(devComm);
  ncclLLA2ASession<ncclCoopCta> done{ncclCoopCta(), devComm, lsa, flags, ctaIndex, devComm.nRanks};
  if (rank != peer) {
    if (threadIdx.x == 0) (void)done.recv<uint64_t>(peer);
    __syncthreads();
    const bool streamed = ginFabricLsaA2ACopyCta(dst, src, bytes);
    ginFabricLsaA2APublish(RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS && streamed);
    if (threadIdx.x == 0) done.send(lsa.rank + (peer - rank), rank, (uint64_t)1);
  } else {
    for (int p = threadIdx.x; p < devComm.nRanks; p += blockDim.x) {
      if (p != rank) done.send(lsa.rank + (p - rank), rank, (uint64_t)1);
    }
    ginFabricLsaA2ACopyCta(dst, src, bytes);
    for (int p = threadIdx.x; p < devComm.nRanks; p += blockDim.x) {
      if (p != rank) (void)done.recv<uint64_t>(p);
    }
  }
  done.endEpoch(ncclCoopCta());
}

}  // namespace fabric
}  // namespace gin

#endif  // _NCCL_DEVICE_GIN_ANVIL_SDMA_GIN_FABRIC_LSA_A2A_DEVICE_H_
