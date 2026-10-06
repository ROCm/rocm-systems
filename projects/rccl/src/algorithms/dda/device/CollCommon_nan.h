/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

// NaN-flag DDA kernels (NCCL_PROTO=NaN). The payload doubles as the ready flag:
// every rank's scratch starts out all-ones, which no f16/bf16/f32 value the user
// may pass ever is (an all-ones dword is NaN), so a 16B unit has landed once none
// of its dwords reads back all-ones. Each rank pushes its data straight into
// every rank's scratch (its own included), polls its own, and puts the sentinel
// back after consuming a unit. That replaces the two cross-rank barriers of the copy-based DDA kernels.
//
// Reuse is safe without any acknowledgement: calls alternate between two banks,
// and every rank receives from every peer in every call, so by the time a rank
// writes a bank again it has seen data each peer sent after finishing (and
// restoring) that bank two calls back. The bank comes from device-side epoch
// cells, not a kernel argument, so captured graphs alternate too.

#pragma once

#include "algorithms/dda/device/CollCommon.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace dda::nan {

constexpr int kRanks = 8;
// One slot per (bank, stage, source rank). AllGather, ReduceScatter, AlltoAll and
// the one-shot AllReduce use stage 0; the two-shot AllReduce reduces through
// stage 0 and gathers through stage 1.
constexpr size_t kSlotBytes = (size_t)2 << 20;
constexpr size_t kStageBytes = kRanks * kSlotBytes;
constexpr size_t kBankBytes = 2 * kStageBytes;
constexpr size_t kScratchBytes = 2 * kBankBytes;
// Every launch bumps every cell, so all cells stay equal and the next launch can
// read its block's cell whatever its grid. Also the grid cap.
constexpr int kEpochCells = 64;
constexpr uint32_t kThreads = 512;

struct Peers {
  v4u* p[kRanks];
};

__device__ __forceinline__ v4u load(const v4u* p) {
  return *(v4u_gptr)p;
}
// Polls must not be served from a stale cached copy. The barrier keeps the
// compiler from hoisting a poll out of its wait loop.
__device__ __forceinline__ v4u poll(const v4u* p) {
  __asm__ volatile("" ::: "memory");
  return __builtin_nontemporal_load((v4u_gptr)p);
}
__device__ __forceinline__ void store(v4u* p, v4u v) {
  *(v4u_gptr)p = v;
}
__device__ __forceinline__ bool pending(v4u x) {
  return max(max(x[0], x[1]), max(x[2], x[3])) == 0xFFFFFFFFu;
}
__device__ __forceinline__ v4u sentinel() {
  const v4u s = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  return s;
}

// Slots are packed at this call's slot size so a small call touches few pages.
// Banks stay at fixed offsets; each call restores everything it consumed, so a
// bank is all sentinel whenever it is reused, whatever the previous packing.
__device__ __forceinline__ v4u* slot(v4u* base, uint32_t bank, int stage, int src, size_t units) {
  return base + bank * (kBankBytes / sizeof(v4u)) + (size_t)(stage * kRanks + src) * units;
}

template <typename T>
__device__ __forceinline__ v4u add(v4u a, v4u b) {
  v4u r;
  r[0] = dda::common::vecElementAdd<T>(a[0], b[0]);
  r[1] = dda::common::vecElementAdd<T>(a[1], b[1]);
  r[2] = dda::common::vecElementAdd<T>(a[2], b[2]);
  r[3] = dda::common::vecElementAdd<T>(a[3], b[3]);
  return r;
}

// Waits for unit u from every source in this rank's stage slots, then restores
// the sentinel. Every source, self included, has a slot so that pushes, polls and
// restores are straight-line: a per-peer branch makes the compiler wait out each
// store's acknowledgement (vmcnt counts loads and stores alike) before the next.
// Re-polling slots that already landed is harmless; only their owner rewrites them.
__device__ __forceinline__ void collect(v4u* mine, uint32_t bank, int stage, size_t u, size_t units, v4u (&v)[kRanks]) {
  bool more;
  do {
#pragma unroll
    for (int s = 0; s < kRanks; s++) v[s] = poll(slot(mine, bank, stage, s, units) + u);
    more = false;
#pragma unroll
    for (int s = 0; s < kRanks; s++) more |= pending(v[s]);
  } while (more);
#pragma unroll
  for (int s = 0; s < kRanks; s++) store(slot(mine, bank, stage, s, units) + u, sentinel());
}

template <typename T>
__device__ __forceinline__ v4u sumInRankOrder(const v4u (&v)[kRanks]) {
  v4u acc = v[0];
#pragma unroll
  for (int s = 1; s < kRanks; s++) acc = add<T>(acc, v[s]);
  return acc;
}

// The block's cell is read by every thread before any thread rewrites it.
__device__ __forceinline__ void epochAdvance(uint32_t* epoch, uint32_t e) {
  __syncthreads();
  for (int c = blockIdx.x + threadIdx.x * gridDim.x; c < kEpochCells; c += gridDim.x * blockDim.x) epoch[c] = e + 1;
}

// Grid for `units` 16B units per thread pass: at most maxBlocks blocks of at most
// maxThreads (a multiple of 64, at most kThreads) threads.
inline std::pair<dim3, dim3> geometry(size_t units, int maxBlocks, uint32_t maxThreads) {
  const size_t warps = (units + 63) / 64;
  const uint32_t threads = (uint32_t)std::min<size_t>(maxThreads, warps * 64);
  size_t blocks = (units + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > (size_t)maxBlocks) blocks = maxBlocks;
  return {dim3((uint32_t)blocks), dim3(threads)};
}

// The copy-only kernels are templates purely so several translation units can
// include this header; their parameter is unused.

// units: 16B units per source rank. Rank r's block lands in recv[r*units ...].
template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllGather(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    const v4u x = load(send + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 0, self, units) + u, x);
  }
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 0, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, e);
}

// Registered-buffer variants: pushes land straight in each peer's registered recv
// buffer, so there is no slot cap and no copy out of scratch. Each block fills the
// slices it is about to receive with the sentinel (its own slice with its own
// data), then posts to every peer's mailbox its recv address as that peer maps it.
// A sender pushes only once it has the post, so the fill cannot overwrite its data.
//
// Mailbox reuse follows the bank argument above: each cell is read and restored in
// the same call, and every call waits on a post from every peer.
struct RegAddrs {
  v4u* p[kRanks]; // this rank's recv buffer as rank p maps it (p[self] is local)
};

__device__ __forceinline__ v4u* mailbox(v4u* base, uint32_t bank, int src) {
  return base + (bank * kBankBytes + kStageBytes) / sizeof(v4u) + (size_t)src * kEpochCells + blockIdx.x;
}

// Posts {address lo, address hi, 0, 0} to every peer -- no dword is all-ones, the
// address being 16B aligned -- then waits for every peer's post and leaves its
// recv address in dst[].
__device__ __forceinline__ void exchange(const Peers& peers, uint32_t bank, int self, const RegAddrs& mine,
                                         v4u** dst) {
  if (threadIdx.x < kRanks) {
    const int r = threadIdx.x;
    const uint64_t a = (uint64_t)mine.p[r];
    const v4u post = {(uint32_t)a, (uint32_t)(a >> 32), 0u, 0u};
    store(mailbox(peers.p[r], bank, self), post);
    v4u* box = mailbox(peers.p[self], bank, r);
    v4u v;
    do {
      v = poll(box);
    } while (pending(v));
    store(box, sentinel());
    dst[r] = (v4u*)(((uint64_t)v[1] << 32) | v[0]);
  }
  __syncthreads();
}

// Pushes, and the fill, write through (sc0 sc1). A plain or nontemporal store to
// peer memory sits in this GPU's L2 until a release, so the peer would only see
// the data once the kernel had finished; a write-through fill needs no L2
// writeback before it is posted.
// `base` is block-uniform; readfirstlane keeps the descriptor in SGPRs when it
// comes back from LDS, which would otherwise cost a waterfall loop per store.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t pushTarget(v4u* base) {
  const uint64_t a = (uint64_t)base;
  const uint64_t s = ((uint64_t)__builtin_amdgcn_readfirstlane((uint32_t)(a >> 32)) << 32) |
                     (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)a);
  return __builtin_amdgcn_make_buffer_rsrc((void*)s, 0, 0x7fffffff, 0x00020000);
}
__device__ __forceinline__ void push(__amdgpu_buffer_rsrc_t dst, size_t u, v4u v) {
  __builtin_amdgcn_raw_buffer_store_b128(v, dst, (int)(u * sizeof(v4u)), 0, /*sc0 sc1*/ 17);
}

// Waits for unit u of every source in the recv buffer. Nothing to restore.
__device__ __forceinline__ void collectRecv(const v4u* recv, size_t u, size_t units) {
  bool more;
  do {
    v4u v[kRanks];
#pragma unroll
    for (int s = 0; s < kRanks; s++) v[s] = poll(recv + s * units + u);
    more = false;
#pragma unroll
    for (int s = 0; s < kRanks; s++) more |= pending(v[s]);
  } while (more);
}

// Shared body. units: 16B units per source block.
template <bool kAllToAll>
__device__ __forceinline__ void regCopy(const Peers& peers, uint32_t* epoch, const v4u* send, v4u* recv,
                                        const RegAddrs& mine, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  const v4u* own = kAllToAll ? send + self * units : send;
  __amdgpu_buffer_rsrc_t to[kRanks];
#pragma unroll
  for (int s = 0; s < kRanks; s++) to[s] = pushTarget(recv + s * units);
  for (size_t u = g0; u < units; u += G) {
    const v4u x = load(own + u);
#pragma unroll
    for (int s = 0; s < kRanks; s++) push(to[s], u, s == self ? x : sentinel());
  }
  // The fill is in memory once its write-through stores are acknowledged.
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();

  __shared__ v4u* dst[kRanks];
  exchange(peers, bank, self, mine, dst);
#pragma unroll
  for (int p = 0; p < kRanks; p++) to[p] = pushTarget(dst[p] + self * units);
  constexpr int kLoads = kAllToAll ? kRanks : 1;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kLoads];
#pragma unroll
    for (int p = 0; p < kLoads; p++) x[p] = load(send + p * units + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) push(to[p], u, x[kAllToAll ? p : 0]);
  }
  for (size_t u = g0; u < units; u += G) collectRecv(recv, u, units);
  epochAdvance(epoch, e);
}

// AllGather. In place (send == recv + self*units) is fine: this rank's own slice
// is filled with its data, not the sentinel.
template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllGatherReg(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, RegAddrs mine, size_t units, int self) {
  regCopy<false>(peers, epoch, send, recv, mine, units, self);
}

// AlltoAll. Not in place: the fill would overwrite blocks not yet sent.
template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllToAllReg(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, RegAddrs mine, size_t units, int self) {
  regCopy<true>(peers, epoch, send, recv, mine, units, self);
}

// units: 16B units per peer block. Block p of send goes to rank p.
template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllToAll(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = load(send + p * units + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 0, self, units) + u, x[p]);
  }
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 0, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, e);
}

// units: 16B units per output block. Every rank sums the eight contributions in
// rank order.
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanReduceScatter(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = load(send + p * units + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 0, self, units) + u, x[p]);
  }
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 0, u, units, v);
    store(recv + u, sumInRankOrder<T>(v));
  }
  epochAdvance(epoch, e);
}

// One shot: every rank pushes its whole input to every rank and reduces all of it.
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllReduceOneShot(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    const v4u x = load(send + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 0, self, units) + u, x);
  }
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 0, u, units, v);
    store(recv + u, sumInRankOrder<T>(v));
  }
  epochAdvance(epoch, e);
}

// Two shot: a push reduce-scatter through stage 0, then a push all-gather of the
// reduced blocks through stage 1. units: 16B units per block (count / 8).
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllReduceTwoShot(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const uint32_t e = epoch[blockIdx.x];
  const uint32_t bank = e & 1;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = load(send + p * units + u);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 0, self, units) + u, x[p]);
  }
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 0, u, units, v);
    const v4u r = sumInRankOrder<T>(v);
#pragma unroll
    for (int p = 0; p < kRanks; p++) store(slot(peers.p[p], bank, 1, self, units) + u, r);
  }
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, bank, 1, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, e);
}

} // namespace dda::nan
