/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

// NaN-flag DDA kernels (NCCL_PROTO=NaN). The payload doubles as the ready flag:
// every rank's scratch starts out all-ones, and a 16B unit has landed once none
// of its dwords reads back all-ones. Under f16/bf16/f32 an all-ones dword is a
// NaN, and every push escapes it to another NaN first (see escape), so real data
// never reads back as the sentinel. Each rank pushes its data straight into
// every rank's scratch (its own included), polls its own, and puts the sentinel
// back after consuming a unit. That replaces the two cross-rank barriers of the copy-based DDA kernels.
//
// Reuse is safe without any acknowledgement: calls rotate through three banks,
// and every rank receives from every peer in every call, so by the time a rank
// writes a bank again it has seen data each peer sent in a later kernel than the
// one that restored that bank. A small call leaves its restore to the next call,
// which issues it right after its own pushes: the restore's acknowledgement then
// overlaps that call's wait for its peers instead of ending this one, and the
// third bank is what makes the deferral safe. The bank comes from device-side epoch cells, not
// a kernel argument, so captured graphs rotate too.

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
constexpr size_t kSlotBytes = (size_t)4 << 20;
constexpr size_t kStageBytes = kRanks * kSlotBytes;
constexpr size_t kBankBytes = 2 * kStageBytes;
constexpr uint32_t kBanks = 3;
constexpr size_t kScratchBytes = kBanks * kBankBytes;
// Every launch advances every cell, so all cells stay equal and the next launch
// can read its block's cell whatever its grid. Also the grid cap.
constexpr int kEpochCells = 512;
// After the cells, per bank: how many units, from the start of the bank, the call
// that used it left for the next call to restore.
constexpr int kEpochWords = kEpochCells + kBanks;
// A call defers its restore when it consumed at most this many units, so that the
// next call, launched with at least kMinGridThreads threads, has at most 32 per
// thread to restore.
constexpr uint32_t kDeferUnits = 16384;
constexpr uint32_t kMinGridThreads = kDeferUnits / 32;
constexpr uint32_t kThreads = 512;

struct Peers {
  v4u* p[kRanks];
};

__device__ __forceinline__ v4u load(const v4u* p) {
  return *(v4u_gptr)p;
}
// Polls must not be served from a stale cached copy. The barrier keeps the
// compiler from hoisting a poll out of its wait loop. On gfx1250 a nontemporal
// load is not coherent with a peer's store (see RCCL_LL_FIFO_SYS_SCOPE), so polls
// there are system-scope, as are put()s.
__device__ __forceinline__ v4u poll(const v4u* p) {
  __asm__ volatile("" ::: "memory");
#if RCCL_LL_FIFO_SYS_SCOPE
  return __builtin_amdgcn_global_load_b128((v4u_gptr)p, RCCL_SYSTEM_SYNCSCOPE);
#else
  return __builtin_nontemporal_load((v4u_gptr)p);
#endif
}
// Into a recv buffer that nobody polls.
__device__ __forceinline__ void store(v4u* p, v4u v) {
  *(v4u_gptr)p = v;
}
// Into a scratch slot or mailbox that a peer polls.
__device__ __forceinline__ void put(v4u* p, v4u v) {
#if RCCL_LL_FIFO_SYS_SCOPE
  __builtin_amdgcn_global_store_b128((v4u_gptr)p, v, RCCL_SYSTEM_SYNCSCOPE);
#else
  *(v4u_gptr)p = v;
#endif
}
__device__ __forceinline__ bool pending(v4u x) {
  return max(max(x[0], x[1]), max(x[2], x[3])) == 0xFFFFFFFFu;
}
__device__ __forceinline__ v4u sentinel() {
  const v4u s = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  return s;
}
// Puts the sentinel back into this rank's scratch once a unit is consumed. On
// gfx1250 it must not linger in this GPU's cache, where a later writeback could
// clobber a peer's push (a plain store hangs); writing the cache back at kernel
// exit instead costs far more at large sizes, since that also flushes the output.
__device__ __forceinline__ void restore(v4u* p) {
  put(p, sentinel());
}
// Clears bit 31 of any all-ones dword. That bit is the sign of the f32 or of the
// upper f16/bf16, so the value stays a NaN. Every pushed value goes through this,
// reduced ones included: a sum can carry NaN payloads into an all-ones dword.
__device__ __forceinline__ v4u escape(v4u x) {
#pragma unroll
  for (int d = 0; d < 4; d++) x[d] = x[d] == 0xFFFFFFFFu ? 0x7FFFFFFFu : x[d];
  return x;
}

// Slots are packed at this call's slot size, rounded up to a 128B line, so a
// small call touches few pages while no two sources share a line: eight peers'
// pushes into one line take turns. Banks stay at fixed offsets; each call
// restores everything it consumed, so a bank is all sentinel whenever it is
// reused, whatever the previous packing.
__host__ __device__ constexpr size_t slotStride(size_t units) {
  return (units + 7) & ~(size_t)7;
}
__device__ __forceinline__ v4u* slot(v4u* base, uint32_t bank, int stage, int src, size_t units) {
  return base + bank * (kBankBytes / sizeof(v4u)) + (size_t)(stage * kRanks + src) * slotStride(units);
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

// Kernels launch 1D grids, or 2D ones whose rows are peers; the epoch cells and
// the deferred restore go by the flattened block.
__device__ __forceinline__ uint32_t flatBlock() {
  return blockIdx.y * gridDim.x + blockIdx.x;
}
__device__ __forceinline__ uint32_t flatBlocks() {
  return gridDim.x * gridDim.y;
}

// One call's bank, and how much of it the call leaves for the next to restore
// (0: the call restores as it consumes).
struct Call {
  uint32_t e;
  uint32_t bank;
  uint32_t deferred;
  v4u* prev;     // the previous call's bank in this rank's scratch
  uint32_t left; // and how much of it that call left to restore
};

// Picks this call's bank. `stages` and `units` give the extent this call will
// consume.
__device__ __forceinline__ Call begin(v4u* mine, const uint32_t* epoch, int stages, size_t units) {
  Call c;
  c.e = epoch[flatBlock()];
  c.bank = c.e;
  const size_t extent = (size_t)stages * kRanks * slotStride(units);
  c.deferred = extent <= kDeferUnits ? (uint32_t)extent : 0;
  // Read every bank's extent alongside the cell rather than after it.
  uint32_t lefts[kBanks];
#pragma unroll
  for (uint32_t b = 0; b < kBanks; b++) lefts[b] = epoch[kEpochCells + b];
  // Every thread has read the block's cell before any thread rewrites it (in
  // epochAdvance). A barrier there instead would hold each block until all its
  // system-scope stores were acknowledged.
  __syncthreads();
  const uint32_t prev = (c.e + kBanks - 1) % kBanks;
  c.left = prev == 0 ? lefts[0] : prev == 1 ? lefts[1] : lefts[2];
  c.prev = mine + prev * (kBankBytes / sizeof(v4u));
  return c;
}

// Restores what the previous call left in its bank; nobody pushes into that bank
// before this kernel has finished. Kernels issue this after their first pushes,
// which would otherwise queue behind it.
__device__ __forceinline__ void catchUp(const Call& c) {
  for (size_t u = (size_t)flatBlock() * blockDim.x + threadIdx.x; u < c.left; u += (size_t)flatBlocks() * blockDim.x)
    restore(c.prev + u);
}

// Waits for one unit, then restores the sentinel unless the call deferred that.
__device__ __forceinline__ v4u collectOne(v4u* p, const Call& c) {
  v4u v;
  do {
    v = poll(p);
  } while (pending(v));
  if (!c.deferred) restore(p);
  return v;
}

// Waits for unit u from every source in this rank's stage slots, then restores
// the sentinel unless the call deferred that. Every source, self included, has a
// slot so that pushes, polls and restores are straight-line: a per-peer branch
// makes the compiler wait out each store's acknowledgement (vmcnt counts loads and
// stores alike) before the next. Re-polling slots that already landed is harmless;
// only their owner rewrites them.
__device__ __forceinline__ void collect(v4u* mine, const Call& c, int stage, size_t u, size_t units,
                                        v4u (&v)[kRanks]) {
  bool more;
  do {
#pragma unroll
    for (int s = 0; s < kRanks; s++) v[s] = poll(slot(mine, c.bank, stage, s, units) + u);
    more = false;
#pragma unroll
    for (int s = 0; s < kRanks; s++) more |= pending(v[s]);
  } while (more);
  if (c.deferred) return;
#pragma unroll
  for (int s = 0; s < kRanks; s++) restore(slot(mine, c.bank, stage, s, units) + u);
}

template <typename T>
__device__ __forceinline__ v4u sumInRankOrder(const v4u (&v)[kRanks]) {
  v4u acc = v[0];
#pragma unroll
  for (int s = 1; s < kRanks; s++) acc = add<T>(acc, v[s]);
  return acc;
}

// Cell i belongs to block i % flatBlocks(); no other block reads it this launch.
__device__ __forceinline__ void epochAdvance(uint32_t* epoch, const Call& c) {
  const uint32_t next = (c.e + 1) % kBanks;
  for (uint32_t i = flatBlock() + threadIdx.x * flatBlocks(); i < kEpochCells; i += flatBlocks() * blockDim.x)
    epoch[i] = next;
  if (flatBlock() == 0 && threadIdx.x == 0) epoch[kEpochCells + c.bank] = c.deferred;
}

// Grid for `units` 16B units per thread pass: at most maxBlocks blocks of at most
// maxThreads (a multiple of 64, at most kThreads) threads, and at least
// kMinGridThreads threads for the previous call's deferred restore.
inline std::pair<dim3, dim3> geometry(size_t units, int maxBlocks, uint32_t maxThreads) {
  const size_t warps = (units + 63) / 64;
  const uint32_t threads = (uint32_t)std::min<size_t>(maxThreads, warps * 64);
  size_t blocks = (std::max<size_t>(units, kMinGridThreads) + threads - 1) / threads;
  if (blocks > (size_t)maxBlocks) blocks = maxBlocks;
  return {dim3((uint32_t)blocks), dim3(threads)};
}

// The same for the per-peer kernels: grid.y is the peer, grid.x splits its units.
inline std::pair<dim3, dim3> peerGeometry(size_t units, int maxBlocks, uint32_t maxThreads) {
  const size_t warps = (units + 63) / 64;
  const uint32_t threads = (uint32_t)std::min<size_t>(maxThreads, warps * 64);
  size_t blocks = (std::max<size_t>(units, kMinGridThreads / kRanks) + threads - 1) / threads;
  blocks = std::min<size_t>(blocks, std::max(1, maxBlocks / kRanks));
  return {dim3((uint32_t)blocks, kRanks), dim3(threads)};
}

// The copy-only kernels are templates purely so several translation units can
// include this header; their parameter is unused.

// Per-peer copies, for small calls (launched with peerGeometry). Block row s
// pushes this rank's block for rank s into rank s's slot and moves rank s's block
// for this rank out of this rank's slot, one unit per thread; row self copies
// locally. Each thread has one push and one poll in flight, so a call's latency
// is one round trip; the all-peer kernels below serialize eight of each per
// thread but move large calls faster.
template <bool kAllToAll>
__device__ __forceinline__ void peerCopy(const Peers& peers, uint32_t* epoch, const v4u* send, v4u* recv,
                                         size_t units, int self) {
  const int s = blockIdx.y;
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  const v4u* src = kAllToAll ? send + s * units : send;
  v4u* out = recv + s * units;
  // The first pass's load needs no bank, so it overlaps the epoch read.
  const v4u x0 = g0 < units ? load(src + g0) : sentinel();
  const Call c = begin(peers.p[self], epoch, 1, units);
  if (s == self) {
    if (g0 < units) store(out + g0, escape(x0));
    for (size_t u = g0 + G; u < units; u += G) store(out + u, escape(load(src + u)));
    // The previous call's restore, all of it here: this row has no peer to wait
    // for, and the others' ends then wait on no restore. At most eight units a
    // thread, the row having about `units` threads.
    for (size_t u = g0; u < c.left; u += G) restore(c.prev + u);
  } else {
    v4u* to = slot(peers.p[s], c.bank, 0, self, units);
    if (g0 < units) put(to + g0, escape(x0));
    for (size_t u = g0 + G; u < units; u += G) put(to + u, escape(load(src + u)));
    v4u* from = slot(peers.p[self], c.bank, 0, s, units);
    for (size_t u = g0; u < units; u += G) store(out + u, collectOne(from + u, c));
  }
  epochAdvance(epoch, c);
}

// units: 16B units per source rank. Rank r's block lands in recv[r*units ...].
// In place (send == recv + self*units) is fine: escaping is idempotent, so rows
// that read send after row self rewrote it push the same values.
template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllGatherPeer(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  peerCopy<false>(peers, epoch, send, recv, units, self);
}

template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllGather(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const Call c = begin(peers.p[self], epoch, 1, units);
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    const v4u x = escape(load(send + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 0, self, units) + u, x);
  }
  catchUp(c);
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 0, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, c);
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
    put(mailbox(peers.p[r], bank, self), post);
    v4u* box = mailbox(peers.p[self], bank, r);
    v4u v;
    do {
      v = poll(box);
    } while (pending(v));
    restore(box);
    dst[r] = (v4u*)(((uint64_t)v[1] << 32) | v[0]);
  }
  __syncthreads();
}

// Pushes, and the fill, write through (sc0 sc1; system scope on gfx1250). A plain or nontemporal store to
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
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __builtin_amdgcn_raw_buffer_store_b128(v, dst, (int)(u * sizeof(v4u)), 0, /*scope:SCOPE_SYS*/ 24);
#else
  __builtin_amdgcn_raw_buffer_store_b128(v, dst, (int)(u * sizeof(v4u)), 0, /*sc0 sc1*/ 17);
#endif
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
  const Call c = begin(peers.p[self], epoch, 0, units);
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
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __asm__ volatile("s_wait_storecnt 0x0" ::: "memory");
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
  __syncthreads();

  __shared__ v4u* dst[kRanks];
  exchange(peers, c.bank, self, mine, dst);
#pragma unroll
  for (int p = 0; p < kRanks; p++) to[p] = pushTarget(dst[p] + self * units);
  constexpr int kLoads = kAllToAll ? kRanks : 1;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kLoads];
#pragma unroll
    for (int p = 0; p < kLoads; p++) x[p] = escape(load(send + p * units + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) push(to[p], u, x[kAllToAll ? p : 0]);
  }
  catchUp(c);
  for (size_t u = g0; u < units; u += G) collectRecv(recv, u, units);
  epochAdvance(epoch, c);
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
  ddaNanAllToAllPeer(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  peerCopy<true>(peers, epoch, send, recv, units, self);
}

template <typename Unused>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllToAll(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const Call c = begin(peers.p[self], epoch, 1, units);
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = escape(load(send + p * units + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 0, self, units) + u, x[p]);
  }
  catchUp(c);
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 0, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, c);
}

// units: 16B units per output block. Every rank sums the eight contributions in
// rank order.
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanReduceScatter(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const Call c = begin(peers.p[self], epoch, 1, units);
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = escape(load(send + p * units + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 0, self, units) + u, x[p]);
  }
  catchUp(c);
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 0, u, units, v);
    store(recv + u, sumInRankOrder<T>(v));
  }
  epochAdvance(epoch, c);
}

// One shot: every rank pushes its whole input to every rank and reduces all of it.
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllReduceOneShot(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const Call c = begin(peers.p[self], epoch, 1, units);
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    const v4u x = escape(load(send + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 0, self, units) + u, x);
  }
  catchUp(c);
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 0, u, units, v);
    store(recv + u, sumInRankOrder<T>(v));
  }
  epochAdvance(epoch, c);
}

// Two shot: a push reduce-scatter through stage 0, then a push all-gather of the
// reduced blocks through stage 1. units: 16B units per block (count / 8).
template <typename T>
__global__ void __launch_bounds__(kThreads)
  ddaNanAllReduceTwoShot(Peers peers, uint32_t* epoch, const v4u* send, v4u* recv, size_t units, int self) {
  const Call c = begin(peers.p[self], epoch, 2, units);
  const size_t g0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, G = (size_t)gridDim.x * blockDim.x;
  for (size_t u = g0; u < units; u += G) {
    v4u x[kRanks];
#pragma unroll
    for (int p = 0; p < kRanks; p++) x[p] = escape(load(send + p * units + u));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 0, self, units) + u, x[p]);
  }
  catchUp(c);
  v4u* mine = peers.p[self];
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 0, u, units, v);
    const v4u r = escape(sumInRankOrder<T>(v));
#pragma unroll
    for (int p = 0; p < kRanks; p++) put(slot(peers.p[p], c.bank, 1, self, units) + u, r);
  }
  for (size_t u = g0; u < units; u += G) {
    v4u v[kRanks];
    collect(mine, c, 1, u, units, v);
#pragma unroll
    for (int s = 0; s < kRanks; s++) store(recv + s * units + u, v[s]);
  }
  epochAdvance(epoch, c);
}

} // namespace dda::nan
