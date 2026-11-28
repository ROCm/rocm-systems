/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device body of the LSA AllToAll lane (alltoall.cu, -D 3). It uses only the
// public device API: LSA pointers into the peers' receive windows and LL A2A
// flags, so it runs wherever the LSA team covers every rank, whatever the GIN
// backend. Include after nccl_device.h and nccl_device/rccl_ptr.h.

#ifndef RCCL_TESTS_ALLTOALL_LSA_LANE_H_
#define RCCL_TESTS_ALLTOALL_LSA_LANE_H_

#include <cstddef>
#include <cstdint>

#include "alltoall_lsa_lane_policy.h"

namespace alltoall_lsa {

// 16-byte store into a peer's receive window. The system-scope store writes
// through to the peer, which is what lets publish() skip the release fence.
__device__ __forceinline__ void store16(uint4* dst, uint4 v) {
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
__device__ __forceinline__ void copy16(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n) {
  for (size_t i = threadIdx.x; i < n; i += kUnroll * blockDim.x) {
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * blockDim.x < n) v[k] = s[i + k * blockDim.x];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (i + k * blockDim.x < n) store16(&d[i + k * blockDim.x], v[k]);
    }
  }
}

// Window offsets can leave either pointer off a 16-byte boundary, and a uint4
// access there faults on gfx1250, so such a copy takes the byte loop. Returns
// false if it did: those plain stores need the release fence before a flag.
__device__ __forceinline__ bool copyCta(char* dst, const char* src, size_t bytes) {
  if ((((uintptr_t)dst | (uintptr_t)src | bytes) & (sizeof(uint4) - 1)) == 0) {
    // A destination that starts mid cache line makes every wavefront's stores
    // straddle lines; peel up to 112 bytes so the batched copy writes whole lines.
    const size_t head = min(bytes, (size_t)(-(uintptr_t)dst & 127));
    if (threadIdx.x < head / sizeof(uint4)) {
      store16(&((uint4*)dst)[threadIdx.x], ((const uint4*)src)[threadIdx.x]);
    }
    copy16<8>((const uint4*)(src + head), (uint4*)(dst + head), (bytes - head) / sizeof(uint4));
    return true;
  }
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
  return false;
}

// Makes this CTA's stores visible before the done flag that follows. The
// device-scope release fence writes back L2 in every CTA; after system-scope
// streaming stores, each wave only has to wait for its own stores.
__device__ __forceinline__ void publish(bool streamed) {
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

// 2D grid: blockIdx.x == peer, blockIdx.y == chunk. CTA (peer, chunk) on every
// rank writes only to `peer`, so each CTA index is a Gather rooted at `peer`:
// the root's CTA flags its same-index CTA on every sender ready, each sender
// copies and answers with a done flag once its stores are visible, and only
// the root waits for all of them. `flags` comes from
// ncclLLA2ACreateRequirement(flagBlocks(nRanks), nRanks); its epoch-tagged
// slots keep a sender within one launch of the root. Past maxCtas ranks the
// CTA indices would run past the flag blocks, so the body does nothing there;
// sizeOk keeps callers off that path.
__device__ __forceinline__ void alltoallBody(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin, size_t recvOff,
                                             size_t bytesPerPeer, struct ncclDevComm devComm, ncclLLA2AHandle flags) {
  if (devComm.nRanks > maxCtas) return;
  int chunks = 0;
  int threads = 0;
  size_t chunkBytes = 0;
  launchConfig(devComm.nRanks, bytesPerPeer, &chunks, &threads, &chunkBytes);
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
    const bool streamed = copyCta(dst, src, bytes);
    publish(RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS && streamed);
    if (threadIdx.x == 0) done.send(lsa.rank + (peer - rank), rank, (uint64_t)1);
  } else {
    for (int p = threadIdx.x; p < devComm.nRanks; p += blockDim.x) {
      if (p != rank) done.send(lsa.rank + (p - rank), rank, (uint64_t)1);
    }
    copyCta(dst, src, bytes);
    for (int p = threadIdx.x; p < devComm.nRanks; p += blockDim.x) {
      if (p != rank) (void)done.recv<uint64_t>(p);
    }
  }
  done.endEpoch(ncclCoopCta());
}

}  // namespace alltoall_lsa

#endif  // RCCL_TESTS_ALLTOALL_LSA_LANE_H_
