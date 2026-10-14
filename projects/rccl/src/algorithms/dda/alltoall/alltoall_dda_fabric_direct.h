/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA fabric AllToAll, ZERO-COPY one-shot push into the peers' recvbuff (gfx1250, out of place,
 * nRanks <= kDdaA2aDirectMaxRanks).
 *
 * The receive buffer lies in a symmetric window (ncclCommWindowRegister with
 * NCCL_WIN_COLL_SYMMETRIC, LSA-mapped): every peer's recvbuff is mapped in this rank's VA at the
 * same window offset, and the host passes those addresses in peerRecv[]. Each rank writes its
 * sendbuff[p*P] straight into peer p's recvbuff[rank*P]: no staging buffer, no copy-out.
 *
 * Grid (1D, dense): [self: S blocks][push rows 1..n-1: B blocks each].
 *   push block (row, j), peer p = rank + row:
 *     j == 0: ENTRY[rank] = E into p's flags (receiver authority: "I started op E, you may write
 *             my recvbuff"; until my kernel starts, earlier work on my stream may still use it)
 *     wait ENTRY[p] == E (local cell, written by p)
 *     copy its stride share of sendbuff[p*P] -> p's recvbuff[rank*P] (U loads held in flight)
 *     release (system), DONE[rank][j] = E into p's flags
 *     wait DONE[p][j] >= E (local; p's block j finished writing my recvbuff[p*P]), acquire
 *   self blocks: local copy sendbuff[rank*P] -> recvbuff[rank*P].
 * The op epoch E lives in this rank's flag buffer and is bumped by the last block of each launch
 * (no host state per call, so graph replays need nothing extra).
 * Flag reuse: p cannot post ENTRY(E+1) before finishing op E, which needs my DONE(E), which I post
 * only after my ENTRY(E) wait; likewise p's DONE(E+1) needs my ENTRY(E+1). So while this rank waits
 * in op E both cells hold E-1 or E.
 * Failure: a block that gives up its ENTRY wait (abort / opt-in timeout) writes nothing to p, and the
 * row's j == 0 block REVOKES its own ENTRY at p (kDdaA2aDirectRevoked, which equals no epoch). A p
 * that is behind (it skipped this call, or arrives after the timeout) would otherwise match the
 * stale ENTRY(E) with its own op E and write into this rank's recvbuff after this call has failed
 * and returned; the ENTRY compare is exact for that reason. What stays open is a p that passed its
 * ENTRY wait before the revoke and is still writing when this rank gives up its DONE wait (a peer
 * that dies mid-copy); the communicator is failed either way.
 *
 * Residency: every block waits only on flags written by blocks of the PEER (its partner row
 * (n - row, j) there), never on a block of its own rank; self blocks never wait. All of a rank's
 * blocks may wait at once, so the op completes only once every rank's whole grid has been resident.
 * The host caps the grid at the comm-wide CU count (comm->ddaFabricMaxBlocks; the launch bounds fit
 * at least one block per CU), so an otherwise idle device holds the whole grid. Kernels that end
 * on their own (compute, other collectives) delay residency but cannot prevent it. Another
 * SPINNING kernel that holds the device and waits on a peer that this kernel blocks is outside the
 * contract (two full-device AllToAlls of different communicators in flight at once, as in the push
 * kernel's header). Every wait here goes through the A2A watchdog (alltoall_dda_watchdog.h), so
 * ncclCommAbort or the opt-in RCCL_DDA_A2A_TIMEOUT ends such a hang. This kernel's own footprint
 * (at most comm-wide-CU-count blocks of 256 threads, no dynamic LDS) does not fill a device.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#include "algorithms/dda/alltoall/alltoall_dda_watchdog.h"

namespace dda::common {

constexpr int kDdaA2aDirectMaxBpp = 64;
constexpr int kDdaA2aDirectMaxRanks = 16;  // peer addresses travel in the kernel arguments
constexpr size_t kDdaA2aDirectLine = 8;     // uint64 cells per 64B line
constexpr size_t kDdaA2aDirectEntry = 0;    // [src], written by src
constexpr size_t kDdaA2aDirectDone = kDdaA2aDirectEntry + kDdaA2aDirectMaxRanks;  // [src][j], written by src
constexpr size_t kDdaA2aDirectEpoch = kDdaA2aDirectDone + (size_t)kDdaA2aDirectMaxRanks * kDdaA2aDirectMaxBpp;
constexpr size_t kDdaA2aDirectFin = kDdaA2aDirectEpoch + 1;
constexpr size_t kDdaA2aDirectNumCells = kDdaA2aDirectFin + 1;
constexpr size_t kDdaA2aDirectFlagBytes = 2u << 20;
constexpr uint64_t kDdaA2aDirectRevoked = ~(uint64_t)0;  // ENTRY value after an abandoned op (matches no epoch)
static_assert(kDdaA2aDirectNumCells * kDdaA2aDirectLine * sizeof(uint64_t) <= kDdaA2aDirectFlagBytes,
              "direct flag area overflows its buffer");

struct DdaA2aDirectArgs {
  const char* sendbuff;
  char* recvbuff;
  char* const* peerFlags;  // [nRanks] peer flag buffers in my VA (self included)
  char* selfFlags;
  // Next to the self pointer so thread 0's first kernarg line carries both.
  DdaA2aWdCtx wd;  // per-comm watchdog state: abort / opt-in timeout / error word (alltoall_dda_watchdog.h)
  uintptr_t peerRecv[kDdaA2aDirectMaxRanks];  // peer r's recvbuff in MY VA (symmetric window; 0 for self)
  size_t chunkBytes;  // P, multiple of 16
  int rank;
  int nRanks;
  int bpp;         // push blocks per peer row (B)
  int selfBlocks;  // S
};

namespace a2a_direct {

typedef __attribute__((address_space(1))) uint64_t* gp64;
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) v4u g4;

__device__ __forceinline__ uint64_t* cell(char* base, size_t idx) {
  return reinterpret_cast<uint64_t*>(base) + idx * kDdaA2aDirectLine;
}
__device__ __forceinline__ void storeSys(uint64_t* p, uint64_t v) {
  __scoped_atomic_store_n((gp64)p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t loadDev(uint64_t* p) {
  return __scoped_atomic_load_n((gp64)p, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}
// Poll at device scope (no invalidate inside the loop); the caller fences once. Abortable /
// opt-in bounded: false = the wait was abandoned (the comm's error word is set).
template <bool Exact>
__device__ __forceinline__ bool waitFor(uint64_t* p, uint64_t v, uint32_t site, DdaA2aWdCtx c) {
  DdaA2aWd wd(c);
  while (Exact ? loadDev(p) != v : loadDev(p) < v) {
    __builtin_amdgcn_s_sleep(1);
    if (wd.expired(site)) return false;
  }
  return true;
}

// 16B copy, U loads issued and held in registers before the U stores (the per-value asm keeps
// LLVM from sinking each load next to its store).
template <int U>
__device__ __forceinline__ void copyHold(uint4* dg, const uint4* sg, size_t begin, size_t end, size_t stride) {
  g4* __restrict__ d = (g4*)dg;
  const g4* __restrict__ s = (const g4*)sg;
  size_t i = begin;
  for (; i + (U - 1) * stride < end; i += (size_t)U * stride) {
    v4u t[U];
#pragma unroll
    for (int u = 0; u < U; ++u) t[u] = s[i + (size_t)u * stride];
#pragma unroll
    for (int u = 0; u < U; ++u) asm volatile("" : "+v"(t[u]));
#pragma unroll
    for (int u = 0; u < U; ++u) d[i + (size_t)u * stride] = t[u];
  }
  if (i < end) {
    v4u t[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (i + (size_t)u * stride < end) t[u] = s[i + (size_t)u * stride];
    }
#pragma unroll
    for (int u = 0; u < U; ++u) asm volatile("" : "+v"(t[u]));
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (i + (size_t)u * stride < end) d[i + (size_t)u * stride] = t[u];
    }
  }
}

} // namespace a2a_direct

template <int U>
__global__ __launch_bounds__(512) void ddaAllToAllFabricDirect(DdaA2aDirectArgs a) {
  using namespace a2a_direct;
  __shared__ uint64_t sEpoch;
  __shared__ int sRun, sGone;
  char* const self = a.selfFlags;
  if (threadIdx.x == 0) {
    // Poison / in-flight check: issued before the epoch load so the round trips overlap.
    const DdaA2aEntry en = ddaA2aEnterIssue(a.wd);
    sEpoch = loadDev(cell(self, kDdaA2aDirectEpoch)) + 1;
    sRun = ddaA2aEnterDecide(a.wd, en);
  }
  __syncthreads();
  if (!sRun) return;  // poisoned communicator / concurrent launch: move no data
  const uint64_t E = sEpoch;
  const int n = a.nRanks;
  const int me = a.rank;
  const int B = a.bpp;
  const int S = a.selfBlocks;
  const int lin = (int)blockIdx.x;
  const size_t P = a.chunkBytes;
  const size_t nvec = P / sizeof(uint4);
  const size_t tid = threadIdx.x;
  const size_t bs = blockDim.x;

  if (lin < S) {
    const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)me * P);
    uint4* dst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)me * P);
    copyHold<U>(dst, src, (size_t)lin * bs + tid, nvec, (size_t)S * bs);
  } else {
    // Block (row, j) pushes to p = me + row and receives from p (p's block (n - row, j) pushes to me).
    const int row = 1 + (lin - S) / B;
    const int j = (lin - S) % B;
    const int p = (me + row) % n;
    char* const peer = a.peerFlags[p];
    if (threadIdx.x == 0) {
      if (j == 0) storeSys(cell(peer, kDdaA2aDirectEntry + me), E);
      // Abandoned (abort / timeout): p may not have started op E, so its recvbuff may still be in
      // use -- write nothing to p, and revoke my ENTRY at p (header: "Failure").
      sGone = !waitFor<true>(cell(self, kDdaA2aDirectEntry + p), E, kDdaWdDirectEntry, a.wd);
      if (sGone && j == 0) storeSys(cell(peer, kDdaA2aDirectEntry + me), kDdaA2aDirectRevoked);
    }
    __syncthreads();
    if (!sGone) {
      const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)p * P);
      uint4* dst = reinterpret_cast<uint4*>(a.peerRecv[p] + (uintptr_t)me * P);
      copyHold<U>(dst, src, (size_t)j * bs + tid, nvec, (size_t)B * bs);
      __syncthreads();  // every wave waited storecnt 0
      if (threadIdx.x == 0) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        storeSys(cell(peer, kDdaA2aDirectDone + (size_t)me * kDdaA2aDirectMaxBpp + j), E);
        uint64_t* done = cell(self, kDdaA2aDirectDone + (size_t)p * kDdaA2aDirectMaxBpp + j);
        (void)waitFor<false>(done, E, kDdaWdDirectDone, a.wd);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
    }
  }

  // ---- epoch bump by the last block to finish ----
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t* fin = cell(self, kDdaA2aDirectFin);
    const uint64_t old = __scoped_atomic_fetch_add((gp64)fin, (uint64_t)1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (old == (uint64_t)gridDim.x - 1) {
      __scoped_atomic_store_n((gp64)fin, (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aDirectEpoch), E, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      ddaA2aLeave(a.wd);
    }
  }
}

} // namespace dda::common
