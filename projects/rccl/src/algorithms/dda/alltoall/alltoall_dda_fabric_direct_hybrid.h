/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * DDA fabric AllToAll, zero-copy with PER-PAIR STAGING (gfx1250, out of place, nRanks <=
 * kDdaA2aDirectMaxRanks): the Direct path for receive buffers that are NOT in a symmetric window
 * (implicit registration, RCCL_DDA_A2A_IMPLICIT; host: dda_alltoall_fabric_direct_host.h).
 *
 * Same grid, epoch, ENTRY cells, revoke and watchdog as the window kernel
 * (alltoall_dda_fabric_direct.h, which this kernel shares the flag buffer with; the window kernel is
 * unchanged). What differs: whether a pair can be written directly is known only to the RECEIVER
 * (its recvbuff may or may not be mapped in the sender's VA: registration cache state, failed or
 * refused imports, peers on another host), so the receiver decides per pair and tells the sender
 * at run time:
 *   j == 0 block of row (rank -> p): ADDR[rank] = my recvbuff[p*P] in p's VA, or kDdaA2aDirectAddrStaged,
 *            into p's flags; system release; ENTRY[rank] = E | kDdaA2aDirectHybridTag into p's flags.
 *   every block of the row: wait ENTRY[p] == E | tag (local, exact), system acquire, read ADDR[p].
 *   ADDR[p] != staged: copy sendbuff[p*P] -> that address (remote stores), as the window kernel.
 *   ADDR[p] == staged: copy into p's scratch slot of this rank (p's Direct-owned staging buffer,
 *            slotBytes per source) in rounds; p copies each round out into its recvbuff.
 * Pair completion (H cells, used by this kernel only, value (E << 16) | n):
 *   HDONE[src][j]  written by the sender into the receiver: n = rounds staged so far, or
 *                  kDdaA2aDirectAll once the direct copy is done (release before the store).
 *   HCONS[dst][j]  written by the receiver into the sender: rounds copied out (the sender reuses the
 *                  slot for round r only after HCONS >= r).
 * Round 0 of op E+1 cannot overwrite a slot that op E still copies out: the sender writes it only
 * after ENTRY(E+1), which the receiver posts when its op E+1 kernel starts (stream order: op E done).
 * ADDR reuse: p writes ADDR(E+1) only after its op E finished, which needs HDONE(E) from every block
 * of my row, each posted after that block read ADDR(E).
 * Failure: as the window kernel (abandoned ENTRY wait -> write nothing to p, revoke my ENTRY at p);
 * an abandoned HCONS wait stops writing into p's slot (p may still be copying it out), and an
 * abandoned staged HDONE wait stops copying out (the slot holds no complete round).
 * Residency: as the window kernel (every wait is on the peer's partner block; self blocks never wait).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/alltoall/alltoall_dda_fabric_direct.h"

namespace dda::common {

// Flag cells of this kernel, after the window kernel's (same flag buffer, same line stride).
constexpr size_t kDdaA2aDirectAddr = kDdaA2aDirectNumCells;  // [src], written by src
constexpr size_t kDdaA2aDirectHDone = kDdaA2aDirectAddr + kDdaA2aDirectMaxRanks;  // [src][j], by src
constexpr size_t kDdaA2aDirectHCons = kDdaA2aDirectHDone + (size_t)kDdaA2aDirectMaxRanks * kDdaA2aDirectMaxBpp;  // [dst][j], by dst
constexpr size_t kDdaA2aDirectHybridNumCells = kDdaA2aDirectHCons + (size_t)kDdaA2aDirectMaxRanks * kDdaA2aDirectMaxBpp;
static_assert(kDdaA2aDirectHybridNumCells * kDdaA2aDirectLine * sizeof(uint64_t) <= kDdaA2aDirectFlagBytes,
              "direct hybrid flag area overflows its buffer");
constexpr uint64_t kDdaA2aDirectAddrStaged = 0;  // ADDR value: "stage through my scratch, I copy out"
// This kernel posts ENTRY = E | kDdaA2aDirectHybridTag, the window kernel E: the two never match each
// other (a rank passing a window buffer while its peer does not -- a contract violation -- hangs
// detectably at ENTRY instead of reading a stale ADDR). Neither value equals kDdaA2aDirectRevoked.
constexpr uint64_t kDdaA2aDirectHybridTag = (uint64_t)1 << 63;
constexpr uint64_t kDdaA2aDirectAll = 0xFFFF;    // HDONE count of a pair written directly
constexpr int kDdaA2aDirectMaxRounds = 0xFFFE;   // round counts travel in 16 bits

struct DdaA2aDirectHybridArgs {
  const char* sendbuff;
  char* recvbuff;
  char* const* peerFlags;  // [nRanks] peer flag buffers in my VA (self included)
  char* selfFlags;
  DdaA2aWdCtx wd;
  uintptr_t myRecvInPeer[kDdaA2aDirectMaxRanks];  // my recvbuff in peer r's VA (0: r stages into my scratch)
  char* const* scratchPeer;  // [nRanks] peer staging buffers in my VA (self included)
  char* scratchSelf;
  size_t chunkBytes;  // P, multiple of 16
  size_t slotBytes;   // per-source staging slot, multiple of 4 KiB; ceil(P / slot) <= kDdaA2aDirectMaxRounds
  int rank;
  int nRanks;
  int bpp;
  int selfBlocks;
};

template <int U>
__global__ __launch_bounds__(512) void ddaAllToAllFabricDirectHybrid(DdaA2aDirectHybridArgs a) {
  using namespace a2a_direct;
  __shared__ uint64_t sEpoch;
  __shared__ uintptr_t sDst;
  __shared__ int sRun, sGone;
  char* const self = a.selfFlags;
  if (threadIdx.x == 0) {
    const DdaA2aEntry en = ddaA2aEnterIssue(a.wd);
    sEpoch = loadDev(cell(self, kDdaA2aDirectEpoch)) + 1;
    sRun = ddaA2aEnterDecide(a.wd, en);
  }
  __syncthreads();
  if (!sRun) return;
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
    const int row = 1 + (lin - S) / B;
    const int j = (lin - S) % B;
    const int p = (me + row) % n;
    char* const peer = a.peerFlags[p];
    const uint64_t E16 = E << 16;
    const bool recvDirect = a.myRecvInPeer[p] != 0;  // receiver authority: mine, for pair p -> me
    if (threadIdx.x == 0) {
      if (j == 0) {
        storeSys(cell(peer, kDdaA2aDirectAddr + me),
                 recvDirect ? (uint64_t)(a.myRecvInPeer[p] + (uintptr_t)p * P) : kDdaA2aDirectAddrStaged);
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        storeSys(cell(peer, kDdaA2aDirectEntry + me), E | kDdaA2aDirectHybridTag);
      }
      const bool gone =
        !waitFor<true>(cell(self, kDdaA2aDirectEntry + p), E | kDdaA2aDirectHybridTag, kDdaWdDirectEntry, a.wd);
      if (gone && j == 0) storeSys(cell(peer, kDdaA2aDirectEntry + me), kDdaA2aDirectRevoked);
      uintptr_t d = 0;
      if (!gone) {
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        d = (uintptr_t)loadDev(cell(self, kDdaA2aDirectAddr + p));
      }
      sGone = gone;
      sDst = d;
    }
    __syncthreads();
    const uintptr_t dstDirect = sDst;
    const uint4* src = reinterpret_cast<const uint4*>(a.sendbuff + (size_t)p * P);
    const size_t first = (size_t)j * bs + tid;
    const size_t step = (size_t)B * bs;
    uint64_t* doneOut = cell(peer, kDdaA2aDirectHDone + (size_t)me * kDdaA2aDirectMaxBpp + j);
    uint64_t* doneIn = cell(self, kDdaA2aDirectHDone + (size_t)p * kDdaA2aDirectMaxBpp + j);
    const bool sendDirect = dstDirect != kDdaA2aDirectAddrStaged;
    if (sGone) {
      // ENTRY wait abandoned: the pair moves nothing.
    } else if (sendDirect && recvDirect) {
      // Both directions direct: the window kernel's pair.
      copyHold<U>(reinterpret_cast<uint4*>(dstDirect), src, first, nvec, step);
      __syncthreads();  // every wave waited storecnt 0
      if (threadIdx.x == 0) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        storeSys(doneOut, E16 | kDdaA2aDirectAll);
        (void)waitFor<false>(doneIn, E16 | kDdaA2aDirectAll, kDdaWdDirectDone, a.wd);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
    } else {
      // At least one direction is staged. Round r moves vectors [r*slot, (r+1)*slot) of the chunk;
      // slot `src` of the receiver's staging buffer belongs to source src.
      const size_t slot = a.slotBytes / sizeof(uint4);
      const int R = (int)((nvec + slot - 1) / slot);
      // Round r: chunk vectors [s0, s1) <-> slot vectors [0, s1 - s0), same per-block partition at both ends.
      uint4* stage = reinterpret_cast<uint4*>(a.scratchPeer[p] + (size_t)me * a.slotBytes);
      const uint4* mine = reinterpret_cast<const uint4*>(a.scratchSelf + (size_t)p * a.slotBytes);
      uint4* rdst = reinterpret_cast<uint4*>(a.recvbuff + (size_t)p * P);
      uint64_t* consOut = cell(peer, kDdaA2aDirectHCons + (size_t)me * kDdaA2aDirectMaxBpp + j);
      uint64_t* consIn = cell(self, kDdaA2aDirectHCons + (size_t)p * kDdaA2aDirectMaxBpp + j);
      if (sendDirect) {
        copyHold<U>(reinterpret_cast<uint4*>(dstDirect), src, first, nvec, step);
        __syncthreads();
        if (threadIdx.x == 0) {
          __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
          storeSys(doneOut, E16 | kDdaA2aDirectAll);
        }
      }
      for (int r = 0; r < R; ++r) {
        const size_t s0 = (size_t)r * slot;
        const size_t s1 = s0 + slot < nvec ? s0 + slot : nvec;
        if (!sendDirect) {
          if (r > 0) {
            if (threadIdx.x == 0) {
              // The receiver may still be copying this slot out: on an abandoned wait stop writing to it.
              if (!waitFor<false>(consIn, E16 | (uint64_t)r, kDdaWdHybridConsumed, a.wd)) sGone = 1;
              __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            }
            __syncthreads();
            if (sGone) break;
          }
          copyHold<U>(stage, src + s0, first, s1 - s0, step);
          __syncthreads();
          if (threadIdx.x == 0) {
            __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
            storeSys(doneOut, E16 | (uint64_t)(r + 1));
          }
        }
        if (!recvDirect) {
          if (threadIdx.x == 0) {
            // Abandoned: the slot holds no complete round -- copy nothing, post no CONSUMED.
            if (!waitFor<false>(doneIn, E16 | (uint64_t)(r + 1), kDdaWdHybridDone, a.wd)) sGone = 1;
            __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
          }
          __syncthreads();
          if (sGone) break;
          copyHold<U>(rdst + s0, mine, first, s1 - s0, step);
          __syncthreads();  // every load of this round's share returned (its value was stored)
          if (threadIdx.x == 0 && r + 1 < R) {
            __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
            storeSys(consOut, E16 | (uint64_t)(r + 1));
          }
        }
      }
      if (recvDirect && threadIdx.x == 0) {
        (void)waitFor<false>(doneIn, E16 | kDdaA2aDirectAll, kDdaWdDirectDone, a.wd);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      }
    }
  }

  // ---- epoch bump by the last block to finish (the window kernel's) ----
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
