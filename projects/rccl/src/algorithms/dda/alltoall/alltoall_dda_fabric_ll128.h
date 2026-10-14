/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-to-all device kernel for the DDA fabric path (gfx1250).
 * Each 128B line holds 120B of payload (15 x uint64) + a trailing flag word;
 * 16 lanes cooperatively write one coalesced line, flag-last and unfenced (see
 * CollCommon_ll128.h). No GPU barrier; staging uses comm->ddaScratch reached via
 * comm->ddaPeerPtrsDev.
 *
 * All-to-all is the "personalized" analogue of all-gather: each rank sends a
 * distinct chunk (sendbuff[peer]) to each peer, and recvbuff[src] receives the
 * chunk sent by rank src. Only the scatter source and self-copy source differ
 * from all_gather_dda_fabric_ll128.h (per-peer offsets into sendbuff).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/alltoall/alltoall_dda_watchdog.h"

namespace dda::common {

// ---- register-packed (16B/lane, warp-per-2KiB-slice) A2A, 8-lane cooperative atomic
// 128B lines (same wire format and helpers as all_gather_dda_fabric_ll128.h). One
// atomic 16B load per line both polls the flag and carries the payload (no dependent
// second load); sendbuff loads for U slices are issued before any wire store and held
// in registers (no per-line load->store serialization); peer columns are rotated
// (peer = self+x+1) so the grid does not hammer one destination at a time; the self
// copy is spread over the whole grid; slots are carved from the shared bank layout
// (ddaBankSize), so the per-chunk cap is the slot size. (Replaces the original
// word-per-lane kernel, which this one matched or beat at every measured size.)

constexpr size_t ddaLL128A2ASlotWords(size_t bankSize, int nRanks) {
  return ddaLLSlotPkts(bankSize, sizeof(uint64_t) * (size_t)nRanks, (size_t)kDdaLL128WireWordsPerSlice);
}

// Plain (cacheable) 16B access for the local user buffers.
__device__ __forceinline__ void ddaA2aLd16(const int8_t* p, uint64_t& lo, uint64_t& hi) {
  const v4u v = *reinterpret_cast<const v4u*>(p);
  lo = ((uint64_t)v[1] << 32) | (uint64_t)v[0];
  hi = ((uint64_t)v[3] << 32) | (uint64_t)v[2];
}
__device__ __forceinline__ void ddaA2aSt16(int8_t* p, uint64_t lo, uint64_t hi) {
  v4u v;
  v[0] = (uint32_t)lo; v[1] = (uint32_t)(lo >> 32);
  v[2] = (uint32_t)hi; v[3] = (uint32_t)(hi >> 32);
  *reinterpret_cast<v4u*>(p) = v;
}

// Same lane/chunk mapping as ddaLL128LoadRegs, with plain loads.
__device__ __forceinline__ void ddaA2aLoadRegs(uint64_t (&regs)[kDdaLL128WordsPerThread], const int8_t* src,
                                               int eltN, int wid, bool flag) {
#pragma unroll
  for (int g = 0; g < kDdaLL128Pairs; g++) {
    if (!flag || g % 2 == 0) {
      const int ix = ddaLL128ChunkIx(g, wid);
      if (ix * 16 < eltN) ddaA2aLd16(src + ix * 16, regs[2 * g], regs[2 * g + 1]);
    }
  }
}
// Full-slice fast path: unconditional loads (no exec-masked branches, so the
// compiler keeps all four in flight). A flag lane's odd pair is not payload; it
// re-reads the even pair's address and is overwritten by ddaA2aShuffleIn.
__device__ __forceinline__ void ddaA2aLoadRegsFull(uint64_t (&regs)[kDdaLL128WordsPerThread], const int8_t* src,
                                                   int wid, bool flag) {
#pragma unroll
  for (int g = 0; g < kDdaLL128Pairs; g++) {
    const int ix = ddaLL128ChunkIx((flag && (g % 2)) ? g - 1 : g, wid);
    ddaA2aLd16(src + ix * 16, regs[2 * g], regs[2 * g + 1]);
  }
}
__device__ __forceinline__ void ddaA2aShuffleIn(uint64_t (&regs)[kDdaLL128WordsPerThread], bool flag) {
#pragma unroll
  for (int g = 1; g < kDdaLL128Pairs; g += 2)
    if (flag) regs[2 * g] = regs[2 * g - 1];
}
// Same mapping as ddaLL128StoreRegs, with plain stores.
__device__ __forceinline__ void ddaA2aStoreRegs(int8_t* dst, uint64_t (&regs)[kDdaLL128WordsPerThread], int eltN,
                                                int wid, bool flag) {
#pragma unroll
  for (int g = 1; g < kDdaLL128Pairs; g += 2)
    if (flag) regs[2 * g - 1] = regs[2 * g];
#pragma unroll
  for (int g = 0; g < kDdaLL128Pairs; g++) {
    if (!flag || g % 2 == 0) {
      const int ix = ddaLL128ChunkIx(g, wid);
      if (ix * 16 < eltN) ddaA2aSt16(dst + ix * 16, regs[2 * g], regs[2 * g + 1]);
    }
  }
}

// Pin a register so LLVM keeps every load of a batch issued before the first store
// (an empty asm memory clobber does not stop load sinking with __restrict__).
__device__ __forceinline__ void ddaA2aHold(uint64_t& v) { asm volatile("" : "+v"(v)); }

template <typename T, int NRANKS_CT, int U>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllToAllFabricLL128V2(T* const* __restrict__ peerScratch,
                                           T* __restrict__ recvbuff,
                                           const T* __restrict__ sendbuff,
                                           size_t perChunkBytes, // per-peer payload; multiple of 16
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, int epochLen,
                                           size_t slicesTotal, // slices per peer chunk
                                           size_t bankSize,
                                           DdaA2aWdCtx wdc) { // abort / timeout / error word
  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = (selfRank + (int)blockIdx.x + 1) % nRanks;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  // Epoch read WITHOUT the leading block barrier of ddaGetLLEpochInc: the barrier only
  // orders this read before ddaSetLLEpoch's writes, so it moves to the kernel end and the
  // epoch load overlaps the first sendbuff loads instead of delaying them.
  // Poison word and epoch loaded together. The empty asm makes both values ready before the poison
  // branch; without it LLVM branches on the poison word first and issues the epoch load only after
  // it returns (two dependent round trips at kernel entry).
  uint32_t site = wdc->site;
  uint32_t epoch = epochDev[flatBlockId];
  asm volatile("" : "+s"(site), "+s"(epoch));
  uint32_t flag32 = epoch + 1u;
  if (flag32 == 0u) flag32 = 2u; // skip 0 sentinel; keep bank parity (as ddaGetLLEpochInc)
  if (site != 0) return;         // failed communicator: move no data
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128A2ASlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff) + (size_t)peer * perChunkBytes;
  uint64_t* scatterSlot =
    reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perChunkBytes;

  // Phase 1: push U slices per warp per step; all loads of the step precede its stores.
  // Full slices (all but a trailing partial one) take branch-free loads under a
  // warp-uniform predicate; the partial slice, if any, is pushed after its batch.
  const size_t fullSlices = perChunkBytes / (size_t)kDdaLL128DataBytesPerSlice;
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += wstride * U) {
    uint64_t regs[U][kDdaLL128WordsPerThread];
    if (s0 + (size_t)(U - 1) * wstride < fullSlices) { // whole batch full: no predicates at all
#pragma unroll
      for (int u = 0; u < U; u++)
        ddaA2aLoadRegsFull(regs[u], srcBytes + (s0 + (size_t)u * wstride) * (size_t)kDdaLL128DataBytesPerSlice,
                           lane, flagLane);
#pragma unroll
      for (int u = 0; u < U; u++)
#pragma unroll
        for (int w = 0; w < kDdaLL128WordsPerThread; w++) ddaA2aHold(regs[u][w]);
#pragma unroll
      for (int u = 0; u < U; u++) {
        ddaA2aShuffleIn(regs[u], flagLane);
        ddaLL128StoreWire(scatterSlot + (s0 + (size_t)u * wstride) * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane,
                          regs[u], flag, flagLane);
      }
      continue;
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t s = s0 + (size_t)u * wstride;
      if (s < fullSlices) {
        ddaA2aLoadRegsFull(regs[u], srcBytes + s * (size_t)kDdaLL128DataBytesPerSlice, lane, flagLane);
      } else {
#pragma unroll
        for (int w = 0; w < kDdaLL128WordsPerThread; w++) regs[u][w] = 0;
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int w = 0; w < kDdaLL128WordsPerThread; w++) ddaA2aHold(regs[u][w]);
#pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t s = s0 + (size_t)u * wstride;
      if (s < fullSlices) {
        ddaA2aShuffleIn(regs[u], flagLane);
        ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs[u], flag, flagLane);
      } else if (s < slicesTotal) { // trailing partial slice
        const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
        ddaA2aLoadRegs(regs[u], srcBytes + dataByte, (int)(perChunkBytes - dataByte), lane, flagLane);
        ddaA2aShuffleIn(regs[u], flagLane);
        ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs[u], flag, flagLane);
      }
    }
  }

  // Self chunk: local copy spread over the whole grid.
  {
    const v4u* s4 = reinterpret_cast<const v4u*>(reinterpret_cast<const int8_t*>(sendbuff) +
                                                  (size_t)selfRank * perChunkBytes);
    v4u* d4 = reinterpret_cast<v4u*>(reinterpret_cast<int8_t*>(recvbuff) + (size_t)selfRank * perChunkBytes);
    const size_t nVec = perChunkBytes >> 4;
    const size_t gtid = (size_t)flatBlockId * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)total * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) d4[i] = s4[i];
  }

  // Phase 2: poll U slices at once; the atomic 128B line load that sees the flag
  // also carries the payload, so a matched load is consumed directly.
  for (size_t s0 = gwarp; s0 < slicesTotal; s0 += wstride * U) {
    uint64_t vr[U][kDdaLL128WordsPerThread];
    bool need;
    // Number of valid slices in this batch (warp-uniform).
    int nv = 0;
#pragma unroll
    for (int u = 0; u < U; u++) nv += (s0 + (size_t)u * wstride < slicesTotal) ? 1 : 0;
    const uint64_t* wire0 = gatherSlot + s0 * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
    const size_t wireStep = wstride * (size_t)kDdaLL128WireWordsPerSlice;
    DdaA2aWd wd(wdc);
    do {
#pragma unroll
      for (int u = 0; u < U; u++)
        if (u < nv)
#pragma unroll
          for (int w = 0; w < kDdaLL128WordsPerThread; w += 2)
            ddaLL128LoadAtomic16B(wire0 + u * wireStep + w * kDdaLL128Warp, vr[u][w], vr[u][w + 1]);
      need = false;
#pragma unroll
      for (int u = 0; u < U; u++)
        if (u < nv)
#pragma unroll
          for (int w = 0; w < kDdaLL128WordsPerThread; w += 2) need |= flagLane && (vr[u][w + 1] != flag);
    } while (__any(need) && !wd.expiredUniform(kDdaWdLL128V2));
#pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t s = s0 + (size_t)u * wstride;
      if (s < slicesTotal) {
        const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
        const size_t rem = perChunkBytes - dataByte;
        const int elt = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
        ddaA2aStoreRegs(dstBytes + dataByte, vr[u], elt, lane, flagLane);
      }
    }
  }
  __syncthreads(); // every thread of the block has read its epoch cell (see top)
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
