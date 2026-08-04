/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128 pack / poll / unpack helpers for the DDA path.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "nccl_device/rccl_ptr.h"

namespace meta::comms {
namespace ll128 {

// Lane and line geometry, mirroring device.h's WARP_SIZE and NCCL_LL128_LINESIZE
// but kept local so this header does not depend on the device-internal build.
// gfx1250 gets wave32 + the 128-byte line
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#if defined(__GFX9__)
constexpr int kWarp = 64;                                    // WARP_SIZE
#else
constexpr int kWarp = 32;
#endif
#if defined(__gfx1250__)
constexpr int kLineBytes = 128;                              // NCCL_LL128_LINESIZE
#else
constexpr int kLineBytes = 64;
#endif
#else
constexpr int kWarp = 32;
constexpr int kLineBytes = 64;
#endif

constexpr int kLineElems = kLineBytes / 8;                   // 16 (LL128) or 8 (LL64)
constexpr int kLineSkip = 2 * kWarp / kLineElems;            // 4 or 16
constexpr int kWordsPerThread = 8;
constexpr int kPairs = kWordsPerThread / 2;                  // register pairs / thread

// The last lane of each line's lane group owns that line's flag word: lanes
// 7,15,23,31 for a 128-byte line
__device__ __forceinline__ bool isFlagLane(int wid) {
  return (wid % (kLineElems / 2)) == (kLineElems / 2 - 1);
}

// Dense 16-byte-chunk index for register-pair g of lane wid (== the prims_ll128
// `ix` formula). Compensates for the flag holes so the packed payload in the
// user buffer stays gap-free and coalesced.
__device__ __forceinline__ int chunkIx(int g, int wid) {
  return g * kWarp - kLineSkip * (g / 2) + wid - (g % 2) * (wid / (kLineElems / 2));
}

// Offset, in u64 words, from lane wid's register-pair base to the flag word of
// the line that pair falls in. A pair sits at wire word p*kWarp + 2*wid and
// kWarp is a whole number of lines (64 % 8, 32 % 16), so the position within the
// line is 2*wid % kLineElems for every pair and one offset covers all of them.
// It cancels the 2*wid, mapping all kLineElems/2 lanes of a line to one address,
// which the hardware coalesces into a single 8-byte request -- see pollWire.
__device__ __forceinline__ int flagWordOffset(int wid) {
  return kLineElems - 1 - ((2 * wid) & (kLineElems - 1));
}

__device__ __forceinline__ void store128(uint64_t* dst, uint64_t lo, uint64_t hi) {
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint64_t w[2];
  } u;
  u.w[0] = lo;
  u.w[1] = hi;
  __builtin_amdgcn_global_store_b128((v4u_gptr)dst, u.v, RCCL_SYSTEM_SYNCSCOPE);
#else
  __builtin_nontemporal_store(lo, (u64_gptr)dst + 0);
  __builtin_nontemporal_store(hi, (u64_gptr)dst + 1);
#endif
  asm volatile("" ::: "memory");
}

__device__ __forceinline__ uint64_t loadWord(const uint64_t* src) {
  asm volatile("" ::: "memory");
  return __hip_atomic_load((u64_gptr)src, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void load128(const uint64_t* src, uint64_t& lo, uint64_t& hi) {
  asm volatile("" ::: "memory");
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  union {
    v4u v;
    uint64_t w[2];
  } u;
  u.v = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
  lo = u.w[0];
  hi = u.w[1];
#else
  lo = __builtin_nontemporal_load((u64_gptr)src + 0);
  hi = __builtin_nontemporal_load((u64_gptr)src + 1);
#endif
}

// (1) Dense load of a slice's payload from `src` into registers, then the
// flag-lane shuffle (== loadRegsBegin aligned-path + loadRegsFinish). `eltN` is
// the number of T elements in this slice (<= dataBytesPerSlice/sizeof(T)).
template <typename T>
__device__ __forceinline__ void loadRegs(
    uint64_t (&regs)[kWordsPerThread], const T* src, int eltN, int wid, bool flag) {
  constexpr int EltPer16B = 16 / sizeof(T);
#pragma unroll
  for (int g = 0; g < kPairs; g++) {
    if (!flag || g % 2 == 0) {
      int ix = chunkIx(g, wid);
      if (ix * EltPer16B < eltN)
        load128(reinterpret_cast<const uint64_t*>(src + ix * EltPer16B),
                regs[2 * g], regs[2 * g + 1]);
    }
  }
#pragma unroll
  for (int g = 1; g < kPairs; g += 2)  // move flag-lane data out of odd regs
    if (flag) regs[2 * g] = regs[2 * g - 1];
}

// (2) Store one slice to the wire with the flag word embedded on the flag lane
// (== recvReduceSendCopy SEND block). `wire` already includes the 2*wid lane
// offset; consecutive register pairs land kWarp u64 apart, so each pair covers a
// distinct set of lines and the whole warp covers kWordsPerThread * kWarp words.
__device__ __forceinline__ void storeWire(
    uint64_t* wire, const uint64_t (&regs)[kWordsPerThread], uint64_t flag, bool flagLane) {
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    store128(wire + u * kWarp, regs[u], flagLane ? flag : regs[u + 1]);
}

// (3) Poll until every line this lane reads has landed, then read the payload
// once.
//
// Unlike the recvReduceSendCopy RECV block, this spins on flag words alone rather
// than speculatively reloading the whole slice per attempt. Two things to keep:
// the kPairs flag loads are all issued before anything waits, so an attempt costs
// one round-trip rather than kPairs; and the per-lane exit is sound because pair g
// of lane wid lies inside the line whose flag it just saw and lines commit
// atomically, so no __any() is needed.
__device__ __forceinline__ void pollWire(
    const uint64_t* wire, uint64_t (&vr)[kWordsPerThread], uint64_t flag, int wid) {
  const int flagOff = flagWordOffset(wid);
  bool pending;
  do {
    uint64_t f[kPairs];
#pragma unroll
    for (int g = 0; g < kPairs; g++)  // all issued, then one wait
      f[g] = loadWord(wire + 2 * g * kWarp + flagOff);
    pending = false;
#pragma unroll
    for (int g = 0; g < kPairs; g++)
      pending |= (f[g] != flag);
  } while (pending);
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2)
    load128(wire + u * kWarp, vr[u], vr[u + 1]);
}

// (3) Poll until every line this lane reads has landed, then read the payload
// once. NCCL's recvReduceSendCopy RECV block: reload the whole slice each
// attempt, flag lanes test their high word, and the warp retries as a unit.
/*__device__ __forceinline__ void pollWire(
  const uint64_t* wire, uint64_t (&vr)[kWordsPerThread], uint64_t flag, int wid) {
const bool flagLane = isFlagLane(wid);
bool needReload;
do {
  needReload = false;
#pragma unroll
  for (int u = 0; u < kWordsPerThread; u += 2) {
    load128(wire + u * kWarp, vr[u], vr[u + 1]);
    needReload |= flagLane && (vr[u + 1] != flag);
  }
} while (__any(needReload));
#pragma unroll
for (int u = 0; u < kWordsPerThread; u += 2)
  load128(wire + u * kWarp, vr[u], vr[u + 1]);
}*/

// (4) Flag-lane un-shuffle then dense store of registers into `dst`
// (== storeRegs, shmem-free: out-of-range chunks are simply skipped).
template <typename T>
__device__ __forceinline__ void storeRegs(
    T* dst, uint64_t (&regs)[kWordsPerThread], int eltN, int wid, bool flag) {
  constexpr int EltPer16B = 16 / sizeof(T);
#pragma unroll
  for (int g = 1; g < kPairs; g += 2)  // reverse the load shuffle
    if (flag) regs[2 * g - 1] = regs[2 * g];
#pragma unroll
  for (int g = 0; g < kPairs; g++) {
    if (!flag || g % 2 == 0) {
      int ix = chunkIx(g, wid);
      if (ix * EltPer16B < eltN)
        store128(reinterpret_cast<uint64_t*>(dst + ix * EltPer16B),
                 regs[2 * g], regs[2 * g + 1]);
    }
  }
}

}  // namespace ll128
}  // namespace meta::comms
