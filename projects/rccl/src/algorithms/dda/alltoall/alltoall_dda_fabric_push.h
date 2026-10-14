/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA fabric AllToAll, one-shot PUSH variant (mid/large messages, gfx1250).
 *
 * Each rank owns a fabric-exported buffer [flags | data]. For one op of
 * per-peer size P the 1D grid is [self: B*M][push: (n-1)*B][recv: (n-1)*B*M]:
 *   - push block (i, j), i = 1..n-1, peer p = rank+i: copies its share of
 *     sendbuff[p*P] (the B blocks of a row interleave with stride B*blockDim)
 *     straight into p's data[rank*P] (remote stores, U loads held in flight),
 *     in K pieces; after each piece one release + READY[rank][j] = E*16+k+1
 *     in p's flag area.
 *   - receive block (i, j'), sender s = rank-i: follows READY[s][j'%B] piece by
 *     piece and copies that share of data[s*P] (local HBM) to recvbuff[s*P],
 *     M receive blocks per sender block, so the copy-out overlaps the push.
 *   - self blocks copy sendbuff[rank*P] -> recvbuff[rank*P] locally.
 * Out of place only (sendbuff != recvbuff): a receive block writes
 * recvbuff[s*P] while push blocks may still be reading sendbuff, so the two
 * buffers must not alias. In place is not selected.
 * Slot reuse: ops that fit half the data region alternate halves by epoch and
 * need no handshake at all (see the comment at `half` in the kernel); larger
 * ops post an ENTRY flag and wait for the peer's before pushing.
 * Every flag carries the device-side op epoch E (bumped by the last block
 * of each launch), so the protocol needs no host state per call and replays
 * correctly under graph capture; the block count may change between ops.
 *
 * Residency: each kernel completes with any number >= 1 of its own blocks resident (see the
 * "Residency" comments in both kernels). It does NOT tolerate another spinning kernel holding the
 * rest of the device: two push AllToAlls of DIFFERENT communicators on one device, in flight at once
 * on different streams, can each fill their GPU and wait on a peer GPU filled by the other one, and
 * then neither progresses. That is outside the supported contract (as for any two full-device
 * collective kernels); ncclCommAbort, or the opt-in RCCL_DDA_A2A_TIMEOUT, ends such a hang because
 * every wait here goes through the A2A watchdog (alltoall_dda_watchdog.h).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#include "algorithms/dda/alltoall/alltoall_dda_watchdog.h"

namespace dda::common {

// Flag-area geometry (uint64 cells, one per 64B line to avoid sharing a line
// between a remotely-written flag and a locally-polled one).
constexpr int kDdaA2aPushMaxBpp = 64;     // max blocks per peer row
constexpr int kDdaA2aPushMaxRanks = 144;  // == kDdaMaxNranks
constexpr int kDdaA2aPushMaxPieces = 15;  // per-block pipelining pieces
constexpr int kDdaA2aPushUnroll = 8;      // 16B loads held in flight per thread (push and copy-out)
constexpr size_t kDdaA2aPushLine = 8;     // uint64 cells per 64B line
constexpr size_t kDdaA2aPushReady = 0;    // [MaxRanks][MaxBpp], written by senders
constexpr size_t kDdaA2aPushEntry = kDdaA2aPushReady + (size_t)kDdaA2aPushMaxRanks * kDdaA2aPushMaxBpp;  // [src], written by src
constexpr size_t kDdaA2aPushEpoch = kDdaA2aPushEntry + kDdaA2aPushMaxRanks;
constexpr size_t kDdaA2aPushFin = kDdaA2aPushEpoch + 1;
constexpr size_t kDdaA2aPushLastWhole = kDdaA2aPushFin + 1;  // epoch of the last op that used the whole data region
constexpr size_t kDdaA2aPushTileCtr = kDdaA2aPushLastWhole + 1;  // CU-split kernel: copy-tile work counter
// Block start ticket (role order = start order, see ddaAllToAllFabricPush / ...PushCu); reset per launch.
constexpr size_t kDdaA2aPushTicket = kDdaA2aPushTileCtr + 1;
// Non-CU kernel: receive-role pool counter (roles handed out only once every self/push block runs); reset per launch.
constexpr size_t kDdaA2aPushRecvTicket = kDdaA2aPushTicket + 1;
constexpr size_t kDdaA2aPushNumCells = kDdaA2aPushRecvTicket + 1;
constexpr size_t kDdaA2aPushFlagBytes = 2u << 20;  // data region starts here
static_assert(kDdaA2aPushNumCells * kDdaA2aPushLine * sizeof(uint64_t) <= kDdaA2aPushFlagBytes,
              "push flag area overflows its reserved region");

struct DdaA2aPushArgs {
  const char* sendbuff;
  char* recvbuff;
  char* const* peerBase;  // [nRanks] peer buffers (flags at +0, data at +kDdaA2aPushFlagBytes), self included
  char* selfBase;
  // Next to the self pointer so thread 0's first kernarg line carries both.
  DdaA2aWdCtx wd;  // per-comm watchdog state: abort / opt-in timeout / error word (alltoall_dda_watchdog.h)
  size_t chunkBytes;   // P: per-peer bytes moved by this launch, multiple of 16
  size_t strideBytes;  // per-peer stride of sendbuff/recvbuff (== P except for multi-round, where
                       // each launch moves one P-byte slice of every per-peer chunk)
  size_t halfBytes;    // data-region half; ops with nRanks*P <= halfBytes alternate halves by epoch
  int rank;
  int nRanks;
  int pieces;    // 1..kDdaA2aPushMaxPieces
  int recvMult;  // receive blocks per sender block (M; non-CU kernel)
  int bpp;       // sender blocks per peer row (B)
  // CU-split kernel only: the same-package sibling (from topology; the relation is symmetric) and
  // its row's block count. sibRank < 0: no sibling, every row has bpp blocks.
  int sibRank;
  int sibBpp;
  int nPush;    // push blocks of this rank (CU-split: tickets [0, nPush) push, the rest copy)
  int tileVec;  // CU-split copy tile, in 16B vectors
};

namespace a2a_push {

typedef __attribute__((address_space(1))) uint64_t* gp64;
typedef unsigned v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint64_t* cell(char* base, size_t idx) {
  return reinterpret_cast<uint64_t*>(base) + idx * kDdaA2aPushLine;
}
__device__ __forceinline__ void storeSys(uint64_t* p, uint64_t v) {
  __scoped_atomic_store_n((gp64)p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}
// Poll at device scope (no invalidate inside the loop); caller fences once.
// Abortable / opt-in bounded (alltoall_dda_watchdog.h): false = the wait was abandoned (abort or
// timeout; the communicator's error word is set and every other wait ends too).
__device__ __forceinline__ bool waitGe(uint64_t* p, uint64_t v, uint32_t site, DdaA2aWdCtx c) {
  DdaA2aWd wd(c);
  while (__scoped_atomic_load_n((gp64)p, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < v) {
    __builtin_amdgcn_s_sleep(1);
    if (wd.expired(site)) return false;
  }
  return true;
}

// 16B copy, U loads issued and held in registers before the U stores (the
// per-value asm keeps LLVM from sinking each load next to its store).
// Both pointers are global memory (user buffers or fabric-mapped peer
// buffers); the address-space cast keeps the stores global_ rather than flat_.
typedef __attribute__((address_space(1))) v4u g4;

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
  // Remainder (< U strides): one predicated batch, still all loads before the
  // stores -- a serial tail would pay one full memory latency per element.
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

// Boundary kk of K equal pieces over nvec vectors, rounded down to 4 KiB (256 vectors) so every
// wave keeps full 128B lines (a misaligned piece split halves remote-write efficiency).
__device__ __forceinline__ size_t pieceBound(int kk, int K, size_t nvec) {
  if (kk >= K) return nvec;
  return (nvec * (size_t)kk / (size_t)K) & ~(size_t)255;
}

} // namespace a2a_push

__global__ __launch_bounds__(1024) void ddaAllToAllFabricPush(DdaA2aPushArgs a) {
  using namespace a2a_push;
  constexpr int U = kDdaA2aPushUnroll;
  __shared__ uint64_t sEpoch, sLastWhole;
  __shared__ int sFirst, sRole, sRun, sGone;
  // Receive: piece k's READY wait was abandoned (watchdog). One slot per piece, set by thread 0
  // before that piece's barrier and never cleared, so every thread reading it after the barrier agrees.
  __shared__ int sFail[kDdaA2aPushMaxPieces];
  char* const self = a.selfBase;
  // Roles: [0, nFirst) self + push (never wait, except push on the peer's ENTRY), [nFirst, grid)
  // receive. See "Residency" below for why first-class roles stay at their blockIdx.
  const int nFirstAll = a.bpp * a.recvMult + a.nPush;
  const int nFirst = nFirstAll < (int)gridDim.x ? nFirstAll : (int)gridDim.x;
  const bool first = (int)blockIdx.x < nFirst;
  for (int i = threadIdx.x; i < kDdaA2aPushMaxPieces; i += blockDim.x) sFail[i] = 0;
  if (threadIdx.x == 0) {
    // Poison / in-flight check: issued first so its round trips overlap the loads below.
    const DdaA2aEntry en = ddaA2aEnterIssue(a.wd);
    sEpoch = __scoped_atomic_load_n((gp64)cell(self, kDdaA2aPushEpoch), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1;
    sLastWhole = __scoped_atomic_load_n((gp64)cell(self, kDdaA2aPushLastWhole), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    sFirst = 0;
    if (first) {
      // Start count. Its value is only needed when this op posts ENTRY (who posts it); otherwise a
      // non-returning add keeps the round trip off the push blocks' critical path.
      const uint64_t E1 = sEpoch;
      const bool half1 = (size_t)a.nRanks * a.chunkBytes <= a.halfBytes;
      if (!half1 || sLastWhole == E1 - 1) {
        sFirst = __scoped_atomic_fetch_add((gp64)cell(self, kDdaA2aPushTicket), (uint64_t)1, __ATOMIC_RELAXED,
                                           __MEMORY_SCOPE_DEVICE) == 0;
      } else {
        (void)__scoped_atomic_fetch_add((gp64)cell(self, kDdaA2aPushTicket), (uint64_t)1, __ATOMIC_RELAXED,
                                        __MEMORY_SCOPE_DEVICE);
      }
    }
    // A failed communicator's start counts are moot.
    sRun = ddaA2aEnterDecide(a.wd, en);
  }
  __syncthreads();
  if (!sRun) return;  // poisoned communicator / concurrent launch: move no data
  const uint64_t E = sEpoch;
  // Slot reuse without a per-op handshake. A "half" op (nRanks*P <= halfBytes)
  // uses data half (E & 1), last written by op E-2. Peer p finished reading
  // that op before it started op E-1, and op E-1 on this rank only completed
  // after observing p's E-1 READY flags -- so by the time op E starts, the
  // half is free, with no flag at all. A "whole" op (or the op right after
  // one) overlaps data p may still be reading from op E-1, so it posts
  // ENTRY[me] = E to every peer and pushes to p only after ENTRY[p] >= E
  // (p started op E => p finished op E-1).
  const bool half = (size_t)a.nRanks * a.chunkBytes <= a.halfBytes;
  const size_t dataOff = kDdaA2aPushFlagBytes + (half ? (size_t)(E & 1) * a.halfBytes : 0);
  const bool needEntry = !half || sLastWhole == E - 1;

  const int n = a.nRanks;
  const int me = a.rank;
  // 1D grid, no idle blocks (idle/interleaved placements measurably slow the
  // push): [self: B*M][push rows 1..n-1: B each][recv rows 1..n-1: B*M each].
  const int B = a.bpp;
  const int M = a.recvMult;
  // Residency (no dispatch-order assumption; the grid, e.g. 468 blocks at 8 ranks / 948 at 16, need
  // not be co-resident). Self/push roles ("first class") stay at their blockIdx:
  // their placement is what the push bandwidth depends on (a role-by-start-order ticket scattered
  // push and copy-out roles over the CUs: 32 MiB 8r 325 -> 193 GB/s). They never wait on this rank;
  // push blocks wait only for the peer's ENTRY, which the FIRST first-class block to start on each
  // rank posts before anything else (start ticket 0, not blockIdx 0). Receive roles are a pool,
  // claimed only after this rank's start ticket shows every first-class block running, so a
  // receive role never holds a slot a first-class block still needs; a receive-range block that
  // does not see that within a bounded wait exits without a role, and the roles left in the pool
  // are drained by the first-class blocks after their own role (the last one to start always sees
  // the full count). The kernel completes with any number >= 1 of its own blocks resident (other
  // kernels holding the device: see the header comment).
  if (needEntry && sFirst) {
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
      if (t != me) storeSys(cell(a.peerBase[t], kDdaA2aPushEntry + me), E);
    }
  }
  auto runRole = [&](const int lin) {
  int row, j;
  const int nSelfBlk = B * M;  // the local self copy gets receive-sized parallelism
  if (lin < nSelfBlk) {
    row = 0;
    j = lin;
  } else if (lin - nSelfBlk < (n - 1) * B) {
    const int rem = lin - nSelfBlk;
    row = 1 + rem / B;
    j = rem % B;
  } else {
    const int rem = lin - nSelfBlk - (n - 1) * B;
    row = n + rem / (B * M);
    j = rem % (B * M);
  }
  const size_t P = a.chunkBytes;
  const size_t nvec = P / sizeof(uint4);
  const int K = a.pieces;
  const size_t tid = threadIdx.x;
  const size_t bs = blockDim.x;
  // Element range of piece k handled by this block: the B blocks of a row interleave over the
  // whole piece (grid-stride within the row). A receive block (jb, sub) takes every M-th stride
  // step of sender block jb.
  auto range = [&](int k, int jb, int sub, int msub, size_t& beg, size_t& end, size_t& stride) {
    beg = pieceBound(k, K, nvec) + (size_t)jb * bs + tid + (size_t)sub * B * bs;
    end = pieceBound(k + 1, K, nvec);
    stride = (size_t)msub * B * bs;
  };
  const bool isPush = row >= 1 && row < n;
  const bool isRecv = row >= n;
  const int jb = j % B;  // receive role: sender block whose READY flag this block follows
  const int sub = j / B;
  const int ri = row - n + 1;  // receive distance

  if (row == 0) {
    const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)me * a.strideBytes);
    uint4* dst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)me * a.strideBytes);
    size_t b0, e0, st;
    for (int k = 0; k < K; ++k) {
      range(k, j % B, j / B, M, b0, e0, st);
      copyHold<U>(dst, src, b0, e0, st);
    }
  }

  if (isPush) {
    const int p = (me + row) % n;
    if (threadIdx.x == 0) {
      sGone = 0;
      if (needEntry) {
        sGone = !waitGe(cell(self, kDdaA2aPushEntry + p), E, kDdaWdPushEntry, a.wd);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
    }
    __syncthreads();
    // Abandoned ENTRY wait: p may still be reading (or have released) that slot -- write nothing to p.
    const bool gone = sGone != 0;
    char* peer = a.peerBase[p];
    const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)p * a.strideBytes);
    uint4* dst = reinterpret_cast<uint4*>(peer + dataOff + (size_t)me * P);
    uint64_t* ready = cell(peer, kDdaA2aPushReady + (size_t)me * kDdaA2aPushMaxBpp + j);
    for (int k = 0; k < (gone ? 0 : K); ++k) {
      size_t b0, e0, st;
      range(k, j, 0, 1, b0, e0, st);
      copyHold<U>(dst, src, b0, e0, st);
      // Release the remote stores (write-back only, no invalidate). __syncthreads
      // waits storecnt 0 in every wave, so one release from thread 0 covers the block.
      __syncthreads();
      if (threadIdx.x == 0) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        storeSys(ready, E * 16 + (uint64_t)(k + 1));
      }
    }
  }

  if (isRecv) {
    const int s = (me - ri + n) % n;
    const uint4* src = reinterpret_cast<const uint4*>(self + dataOff + (size_t)s * P);
    uint4* dst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)s * a.strideBytes);
    uint64_t* ready = cell(self, kDdaA2aPushReady + (size_t)s * kDdaA2aPushMaxBpp + jb);
    for (int k = 0; k < K; ++k) {
      if (threadIdx.x == 0) {
        if (!waitGe(ready, E * 16 + (uint64_t)(k + 1), kDdaWdPushReady, a.wd)) sFail[k] = 1;
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
      __syncthreads();
      // Abandoned wait (the communicator failed): write nothing more to recvbuff.
      if (sFail[k]) break;
      size_t b0, e0, st;
      range(k, jb, sub, M, b0, e0, st);
      copyHold<U>(dst, src, b0, e0, st);
    }
    __syncthreads();  // every thread's loads of slot [s] for this block have returned
  }
  };  // runRole

  if (first) runRole((int)blockIdx.x);
  {
    const int nRecvRoles = (int)gridDim.x - nFirst;
    bool waited = first;
    for (;;) {
      if (threadIdx.x == 0) {
        sRole = -1;
        gp64 rc = (gp64)cell(self, kDdaA2aPushRecvTicket);
        if (nRecvRoles > 0 &&
            (!first || __scoped_atomic_load_n(rc, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < (uint64_t)nRecvRoles)) {
          gp64 pc = (gp64)cell(self, kDdaA2aPushTicket);
          bool ok = __scoped_atomic_load_n(pc, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) >= (uint64_t)nFirst;
          if (!ok && !waited) {
            // Bounded, ~100 us: a tenth of the watchdog's ~1 ms abort gate (abortTicks, which fabric
            // init derives from the device wall-clock rate). Normally every first-class block is
            // already running; giving up only moves this block's role to the first-class blocks.
            const uint64_t lim = a.wd->abortTicks / 10;
            const uint64_t t0 = wall_clock64();
            while (!ok && wall_clock64() - t0 < lim) {
              __builtin_amdgcn_s_sleep(1);
              ok = __scoped_atomic_load_n(pc, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) >= (uint64_t)nFirst;
            }
          }
          if (ok) {
            const uint64_t r = __scoped_atomic_fetch_add(rc, (uint64_t)1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            if (r < (uint64_t)nRecvRoles) sRole = nFirst + (int)r;
          }
        }
      }
      waited = true;
      __syncthreads();
      const int role = sRole;
      __syncthreads();
      if (role < 0) break;
      runRole(role);
      // A receive-range block takes one role: roles left by blocks that gave up are drained by the
      // first-class blocks (the last one to start sees the full start count after its own role).
      if (!first) break;
    }
  }

  // ---- epoch bump by the last block to finish ----
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t* fin = cell(self, kDdaA2aPushFin);
    const uint64_t total = (uint64_t)gridDim.x;
    // Relaxed: the epoch is read by the NEXT launch (kernel boundary orders it).
    const uint64_t old = __scoped_atomic_fetch_add((gp64)fin, (uint64_t)1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (old == total - 1) {
      __scoped_atomic_store_n((gp64)fin, (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushTicket), (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushRecvTicket), (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if (!half) __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushLastWhole), E, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushEpoch), E, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      ddaA2aLeave(a.wd);
    }
  }
}


// ---------------------------------------------------------------------------
// CU-split variant. Measured on DPX gfx1250 (TransferBench): a local copy
// running on the SAME CU as remote-push blocks slows the push by up to 30%
// and itself crawls, while on separate CUs neither is affected. So: one block
// per CU (the host reserves > half the LDS per block), push blocks on their own
// CUs, and the copy-out (+ self copy) done by the remaining "copy" blocks from a
// shared tile queue; push blocks join the queue once their rows are done.
// Copy items in queue order: self, then piece k of every source (sibling
// first). A tile of item (s, k) is copied once every sender block of s has
// posted READY >= k+1.
// ---------------------------------------------------------------------------
namespace a2a_push {
// Item i of the copy queue: i == 0 self, else (k, source order o) with
// i - 1 = k * (n-1) + o. Source order: the same-package sibling (sib, from topology; < 0 = none)
// first, then increasing distance.
__device__ __forceinline__ int cuSrc(int o, int me, int n, int sib) {
  if (sib < 0) return (me - 1 - o + n) % n;
  if (o == 0) return sib;
  // o-th non-sibling source by distance 1..n-1
  int c = 0;
  for (int d = 1; d < n; ++d) {
    const int s = (me - d + n) % n;
    if (s == sib) continue;
    if (++c == o) return s;
  }
  return sib;
}

// Block copy of [b0, e0) with wave-contiguous runs: lane l of wave w loads
// vectors base + u*32 + l (u < U), base stepping by nWaves*32*U. All U loads
// are issued before the stores; the last partial run is predicated.
template <int U>
__device__ __forceinline__ void copyWave(uint4* dg, const uint4* sg, size_t b0, size_t e0) {
  g4* __restrict__ d = (g4*)dg;
  const g4* __restrict__ s = (const g4*)sg;
  const size_t l = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (size_t base = b0 + w * 32 * U; base < e0; base += nw * 32 * U) {
    v4u t[U];
    if (base + 32 * U <= e0) {
#pragma unroll
      for (int u = 0; u < U; ++u) t[u] = s[base + (size_t)u * 32 + l];
#pragma unroll
      for (int u = 0; u < U; ++u) asm volatile("" : "+v"(t[u]));
#pragma unroll
      for (int u = 0; u < U; ++u) d[base + (size_t)u * 32 + l] = t[u];
    } else {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (base + (size_t)u * 32 + l < e0) t[u] = s[base + (size_t)u * 32 + l];
      }
#pragma unroll
      for (int u = 0; u < U; ++u) asm volatile("" : "+v"(t[u]));
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (base + (size_t)u * 32 + l < e0) d[base + (size_t)u * 32 + l] = t[u];
      }
    }
  }
}
} // namespace a2a_push

__global__ __launch_bounds__(1024) void ddaAllToAllFabricPushCu(DdaA2aPushArgs a) {
  using namespace a2a_push;
  constexpr int U = kDdaA2aPushUnroll;
  extern __shared__ char ldsReserve[];  // never accessed: its launch size only keeps one block per CU
  __shared__ uint64_t sEpoch, sLastWhole, sTicket;
  __shared__ int sRun, sGone, sJoin;
  // Copy role: a READY wait was abandoned (watchdog). Never cleared; read after the barrier that
  // follows the waits, and the next write can only come two barriers later, so every thread agrees.
  __shared__ int sFail;
  __shared__ long long sTile;
  __shared__ int sKnown[kDdaA2aPushMaxRanks];  // copy role: highest piece known ready per source
  char* const self = a.selfBase;
  for (int i = threadIdx.x; i < kDdaA2aPushMaxRanks; i += blockDim.x) sKnown[i] = -1;
  if (threadIdx.x == 0) {
    // Poison / in-flight check: issued first so its round trips overlap the loads below.
    const DdaA2aEntry en = ddaA2aEnterIssue(a.wd);
    sFail = 0;
    sEpoch = __scoped_atomic_load_n((gp64)cell(self, kDdaA2aPushEpoch), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) + 1;
    sLastWhole = __scoped_atomic_load_n((gp64)cell(self, kDdaA2aPushLastWhole), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    sTicket = __scoped_atomic_fetch_add((gp64)cell(self, kDdaA2aPushTicket), (uint64_t)1, __ATOMIC_RELAXED,
                                        __MEMORY_SCOPE_DEVICE);
    sRun = ddaA2aEnterDecide(a.wd, en);
  }
  __syncthreads();
  if (!sRun) return;  // poisoned communicator / concurrent launch: move no data
  const uint64_t E = sEpoch;
  // Residency. Role index = START order (per-launch ticket, as in ddaAllToAllFabricPush), not
  // blockIdx: tickets [0, nPush) push (ticket 0 posts ENTRY before any wait), the rest copy. Push
  // blocks wait only for ENTRY; copy work waits only for READY of push blocks. A push block joins the
  // copy queue only once every push ticket of this rank is held by a running block (checked below),
  // and copy-only tickets are handed out after all push tickets, so no waiting block can hold a slot
  // a push block still needs: the kernel drains with any number >= 1 of its own blocks resident. The
  // host sizes the grid to one block per CU (comm-wide CU cap) for speed, not for correctness.
  // Reset by the last block, with the fin counter.
  const int lin = (int)(sTicket % (uint64_t)gridDim.x);
  const bool half = (size_t)a.nRanks * a.chunkBytes <= a.halfBytes;
  const size_t dataOff = kDdaA2aPushFlagBytes + (half ? (size_t)(E & 1) * a.halfBytes : 0);
  const bool needEntry = !half || sLastWhole == E - 1;
  const int n = a.nRanks;
  const int me = a.rank;
  const size_t P = a.chunkBytes;
  const size_t nvec = P / sizeof(uint4);
  const int K = a.pieces;
  const size_t tid = threadIdx.x;
  const size_t bs = blockDim.x;
  // Blocks of the row between this rank and p (symmetric, as the sibling relation is).
  auto rowBpp = [&](int p) -> int { return p == a.sibRank ? a.sibBpp : a.bpp; };

  if (needEntry && lin == 0) {
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
      if (t != me) storeSys(cell(a.peerBase[t], kDdaA2aPushEntry + me), E);
    }
  }

  // ---- push role ----
  bool copier = lin >= a.nPush;
  if (!copier) {
    int rem = lin, row = 1, j = 0;
    for (; row < n; ++row) {
      const int b = rowBpp((me + row) % n);
      if (rem < b) { j = rem; break; }
      rem -= b;
    }
    const int p = (me + row) % n;
    const int B = rowBpp(p);
    if (threadIdx.x == 0) {
      sGone = 0;
      if (needEntry) {
        sGone = !waitGe(cell(self, kDdaA2aPushEntry + p), E, kDdaWdPushCuEntry, a.wd);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
    }
    __syncthreads();
    // Abandoned ENTRY wait: write nothing to p (see ddaAllToAllFabricPush).
    const bool gone = sGone != 0;
    const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)p * a.strideBytes);
    uint4* dst = reinterpret_cast<uint4*>(a.peerBase[p] + dataOff + (size_t)me * P);
    uint64_t* ready = cell(a.peerBase[p], kDdaA2aPushReady + (size_t)me * kDdaA2aPushMaxBpp + j);
    for (int k = 0; k < (gone ? 0 : K); ++k) {
      copyHold<U>(dst, src, pieceBound(k, K, nvec) + (size_t)j * bs + tid, pieceBound(k + 1, K, nvec), (size_t)B * bs);
      __syncthreads();
      if (threadIdx.x == 0) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        storeSys(ready, E * 16 + (uint64_t)(k + 1));
      }
    }
    // Join the copy queue only if every push ticket of this rank is already taken (see the ticket comment).
    __syncthreads();
    if (threadIdx.x == 0) {
      sJoin = __scoped_atomic_load_n((gp64)cell(self, kDdaA2aPushTicket), __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) >=
              (uint64_t)a.nPush;
    }
    __syncthreads();
    copier = sJoin != 0;
  }

  // ---- copy role: ordered tile queue ----
  if (copier) {
    const size_t T = (size_t)a.tileVec;
    const size_t selfTiles = (nvec + T - 1) / T;
    uint64_t* ctr = cell(self, kDdaA2aPushTileCtr);
    while (true) {
      if (threadIdx.x == 0) sTile = (long long)__scoped_atomic_fetch_add((gp64)ctr, (uint64_t)1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __syncthreads();
      long long t = sTile;
      __syncthreads();
      const uint4* src;
      uint4* dst;
      size_t b0, e0;
      if (t < (long long)selfTiles) {
        src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)me * a.strideBytes);
        dst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)me * a.strideBytes);
        b0 = (size_t)t * T;
        e0 = b0 + T < nvec ? b0 + T : nvec;
      } else {
        t -= (long long)selfTiles;
        int k = 0, o = 0;
        bool found = false;
        size_t pb = 0, pe = 0;
        for (k = 0; k < K && !found; ++k) {
          pb = pieceBound(k, K, nvec);
          pe = pieceBound(k + 1, K, nvec);
          const long long nt = (long long)((pe - pb + T - 1) / T);
          if (t < nt * (n - 1)) {
            o = (int)(t / nt);
            t -= (long long)o * nt;
            found = true;
            break;
          }
          t -= nt * (n - 1);
        }
        if (!found) break;
        const int s = cuSrc(o, me, n, a.sibRank);
        // sKnown[s] is only written below, after a barrier, so every thread sees the same value here.
        if (sKnown[s] < k) {
          // One lane per sender block polls its flag (all in flight at once).
          const int Bs = rowBpp(s);
          if (threadIdx.x < (unsigned)Bs) {
            if (!waitGe(cell(self, kDdaA2aPushReady + (size_t)s * kDdaA2aPushMaxBpp + threadIdx.x),
                        E * 16 + (uint64_t)(k + 1), kDdaWdPushCuReady, a.wd)) {
              sFail = 1;
            }
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            sKnown[s] = k;
          }
        }
        __syncthreads();
        // Abandoned wait (the communicator failed): write nothing more to recvbuff.
        if (sFail) break;
        src = reinterpret_cast<const uint4*>(self + dataOff + (size_t)s * P);
        dst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)s * a.strideBytes);
        b0 = pb + (size_t)t * T;
        e0 = b0 + T < pe ? b0 + T : pe;
      }
      copyWave<U>(dst, src, b0, e0);
    }
  }

  // ---- epoch bump by the last block to finish (also resets the tile counter) ----
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t* fin = cell(self, kDdaA2aPushFin);
    const uint64_t total = (uint64_t)gridDim.x;
    const uint64_t old = __scoped_atomic_fetch_add((gp64)fin, (uint64_t)1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (old == total - 1) {
      __scoped_atomic_store_n((gp64)fin, (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushTileCtr), (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushTicket), (uint64_t)0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      if (!half) __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushLastWhole), E, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n((gp64)cell(self, kDdaA2aPushEpoch), E, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      ddaA2aLeave(a.wd);
    }
  }
}

} // namespace dda::common
