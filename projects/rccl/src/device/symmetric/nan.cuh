// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// NaN-flag symmetric kernels (NCCL_PROTO=NaN). As in the NaN-flag DDA kernels
// (algorithms/dda/device/CollCommon_nan.h), the payload doubles as the ready flag: the scratch
// starts out all-ones, a 16 B unit has landed once none of its dwords reads back all-ones, and
// every push escapes an all-ones dword to another NaN first. Under f32/f16/bf16 an all-ones dword
// is a NaN, so real data never reads back as the sentinel; f64 is not served. Each rank pushes
// straight into every rank's scratch (its own included), polls its own, and puts the sentinel
// back after consuming a unit, so no epoch needs a barrier across ranks.
//
// A block streams its part of the call through its scratch region an epoch at a time, rotating
// through the region's three banks. Before a peer pushes a unit into a bank, this rank must have
// restored that unit's slots there from the bank's previous use:
//  - The register path gives each unit of an epoch to the same thread in every epoch, on every
//    rank. A thread waits for its restores to be acknowledged before its next epoch's pushes, and a
//    peer's thread pushes into a bank only after collecting this thread's pushes of the epoch
//    before, which come after its restores of the epoch before that.
//  - The TDM path pushes a whole chunk per warp, so the units a warp pushes were restored by every
//    thread of the peer's block. A block barrier after the restores' acknowledgement covers that.
// A launch's first epoch takes the bank after its predecessor's last. Two banks would not do: a
// peer finishes a launch once the units of its partial last epoch have landed, and a unit idle in
// that epoch was last restored in the bank the next launch starts in, by a warp that may still be
// at it. A third bank puts the bank's previous use two epochs back, which every unit took part in
// (or a launch whose start already orders it, when a launch has one epoch). Both paths lay out an
// epoch the same way, so ranks may take different paths in one launch (each picks by its own
// buffers' alignment).

#ifndef NCCL_DEVICE_SYMMETRIC_NAN_H_
#define NCCL_DEVICE_SYMMETRIC_NAN_H_

#include "sym_kernels.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "symmetric/kernel.h"
#include "symmetric/primitives.h"
#else
#include "kernel.cuh"
#include "primitives.cuh"
#endif

namespace ncclSymkNan {
constexpr int kRanks = ncclSymkNanMaxRanks;
constexpr int kUnits = ncclSymkNanEpochUnits;
constexpr int kUnroll = ncclSymkNanUnroll;
constexpr uint32_t kChunkBytes = kUnits * 16;
// Two-shot parts of at most this many units split pushes and polls between warps; larger ones
// need every warp pushing.
constexpr int kTwoShotSplitUnits = 64;

__device__ __forceinline__ v4u load(const v4u* p) {
  return *(v4u_gptr)p;
}
// Polls must not be served from a stale cached copy, and the barrier keeps the compiler from
// hoisting one out of its wait loop. On gfx1250 a nontemporal load is not coherent with a peer's
// store (see RCCL_LL_FIFO_SYS_SCOPE), so polls there are system-scope, as are puts.
__device__ __forceinline__ v4u poll(const v4u* p) {
  __asm__ volatile("" ::: "memory");
#if RCCL_LL_FIFO_SYS_SCOPE
  return __builtin_amdgcn_global_load_b128((v4u_gptr)p, RCCL_SYSTEM_SYNCSCOPE);
#else
  return __builtin_nontemporal_load((v4u_gptr)p);
#endif
}
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
// A system-scope store on gfx1250: the sentinel must not linger in this GPU's cache, where a later
// writeback could clobber a peer's push.
__device__ __forceinline__ void restore(v4u* p) {
  put(p, sentinel());
}
// Clears bit 31 of any all-ones dword: the sign of the f32 or of the upper f16/bf16, so the value
// stays a NaN.
__device__ __forceinline__ v4u escape(v4u x) {
#pragma unroll
  for (int d = 0; d < 4; d++) x[d] = x[d] == 0xFFFFFFFFu ? 0x7FFFFFFFu : x[d];
  return x;
}
__device__ __forceinline__ void waitStores() {
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  __asm__ volatile("s_wait_storecnt 0x0" ::: "memory");
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
}

__device__ __forceinline__ BytePack<16> asPack(v4u v) {
  BytePack<16> p;
  p.v4u = v;
  return p;
}

// NRanks is kRanks when the team has exactly that many ranks, letting every per-rank loop run
// without a branch (a per-peer branch makes the compiler wait out each store's acknowledgement
// before the next), and 0 for a smaller team.
// Kept off the stack, which costs a launch tens of us: every member function is forceinline (an
// outlined one takes the session's address), and the arguments are held by value (an outlined
// one reached through a reference into them copies all 4 KiB of them to the stack).
template <int NRanks>
struct Session {
  ncclTeam team;
  char* region; // this block's region of this rank's scratch, through the flat window
  uint32_t stride4G;
  uint32_t* abortFlag;
  uint32_t* header;
  uint32_t bank;

  // hdr is the header's contents, read at the kernel's start so the read overlaps finding the
  // block's work. The scratch starts out all-ones, header lines included: % keeps the first bank
  // in range.
  __device__ Session(ncclSymkDevWorkArgs const* args, ncclTeam team, uint32_t hdr)
    : team(team), stride4G(args->kcomm.devComm.resourceWindow_inlined.stride4G),
      abortFlag(args->kcomm.devComm.abortFlag), header(headerOf(args)), bank(hdr % ncclSymkNanBanks) {
    region = (char*)ncclGetResourceBufferLocalPointer(args->kcomm.devComm, args->kcomm.nanBuf) +
             ncclSymkNanHeaderBytes + (size_t)blockIdx.x * ncclSymkNanBlockBytes;
  }
  __device__ static uint32_t* headerOf(ncclSymkDevWorkArgs const* args) {
    return (uint32_t*)ncclGetResourceBufferLocalPointer(args->kcomm.devComm, args->kcomm.nanBuf) + blockIdx.x * 4;
  }
  // Every thread has read the header before thread 0 rewrites it. A barrier at the start
  // instead holds back the first pushes, and the kernel's end waits out their
  // acknowledgements either way.
  __device__ __forceinline__ void close() {
    ncclCoopCta().sync();
    if (threadIdx.x == 0) *header = bank;
  }

  __device__ __forceinline__ bool active(int r) const {
    return NRanks == kRanks || r < team.nRanks;
  }
  // Unit u of source src's slot in team rank r's current bank.
  __device__ __forceinline__ v4u* slot(int r, int src, int u) const {
    char* peer = nccl::utility::add4G(region, (r - team.rank) * team.stride * (int)stride4G);
    return (v4u*)peer + (bank * kRanks + src) * kUnits + u;
  }

  // Waits for units u[k] (where valid[k]) of every source, then restores them. Re-polling a slot
  // that already landed is harmless: only this thread rewrites it.
  __device__ __forceinline__ void collect(const int (&u)[kUnroll], const bool (&valid)[kUnroll],
                                          v4u (&v)[kUnroll][kRanks]) {
    using nccl::utility::testAbort;
    v4u* mine[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; k++) mine[k] = slot(team.rank, 0, u[k]);
    uint32_t steps = 0;
    bool more;
    do {
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
#pragma unroll
        for (int s = 0; s < kRanks; s++) {
          if (valid[k] && active(s)) v[k][s] = poll(mine[k] + s * kUnits);
        }
      }
      more = false;
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
#pragma unroll
        for (int s = 0; s < kRanks; s++) more |= valid[k] && active(s) && pending(v[k][s]);
      }
    } while (more && !testAbort(abortFlag, steps));
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
#pragma unroll
      for (int s = 0; s < kRanks; s++) {
        if (valid[k] && active(s)) restore(mine[k] + s * kUnits);
      }
    }
  }


  __device__ __forceinline__ void nextBank() {
    bank = bank == ncclSymkNanBanks - 1 ? 0 : bank + 1;
  }
  __device__ __forceinline__ void endEpoch() {
    waitStores();
    nextBank();
  }
};

// The units of an epoch of n units this thread owns: the same ones in every epoch.
__device__ __forceinline__ void unitsOf(int n, int (&u)[kUnroll], bool (&valid)[kUnroll]) {
#pragma unroll
  for (int k = 0; k < kUnroll; k++) {
    u[k] = threadIdx.x + k * blockDim.x;
    valid[k] = u[k] < n;
  }
}

// Sums the sources in rank order, like the NaN DDA kernels.
template <int NRanks, typename T, typename Red>
__device__ __forceinline__ BytePack<16> reduceRanks(Session<NRanks> const& ses, Red red, v4u const (&v)[kRanks]) {
  using Acc = typename Red::EltType;
  using AccPack = BytePack<16 * sizeof(Acc) / sizeof(T)>;
  AccPack acc = applyCast<T, Acc>(asPack(v[0]));
#pragma unroll
  for (int s = 1; s < kRanks; s++) {
    if (ses.active(s)) acc = applyReduce(red, acc, applyCast<T, Acc>(asPack(v[s])));
  }
  return applyCast<Acc, T>(applyPostOp(red, acc));
}

enum class Kind { AllGather, AllReduce, ReduceScatter };

// The split path takes a launch whose block parts are a single epoch of at most splitUnits()
// units. Even warps push and odd warps collect, unit u[k] of either half's thread i being
// i + k * blockDim.x / 2. Polls queue behind pushes issued from the same SIMD, which cost a small
// call about 2 us when the same threads did both, and as much again when the collecting warps
// were those blockDim.x / 2 threads on. A launch of one epoch may map
// threads to units any way it likes: the previous use of its bank ended in an earlier launch,
// so a peer pushing into it has collected this rank's pushes of a launch that started after that
// use's restores were acknowledged.
__device__ __forceinline__ int splitUnits() {
  return (int)blockDim.x / 2 * kUnroll;
}

template <Kind K, int NRanks, typename T, typename Red, typename EltType>
__device__ __forceinline__ void splitBody(Session<NRanks>& ses, Red red, EltType* input, EltType* output, int nElts,
                                          size_t nStrideElts) {
  constexpr int EltPerUnit = 16 / sizeof(EltType);
  const int nUnits = divUp(nElts, EltPerUnit);
  const int half = blockDim.x / 2;
  const int w = threadIdx.x / WARP_SIZE;
  const bool pusher = w % 2 == 0;
  const int i = w / 2 * WARP_SIZE + threadIdx.x % WARP_SIZE;
  int u[kUnroll];
  bool valid[kUnroll];
#pragma unroll
  for (int k = 0; k < kUnroll; k++) {
    u[k] = i + k * half;
    valid[k] = u[k] < nUnits;
  }
  if (pusher) {
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (!valid[k]) continue;
      if constexpr (K == Kind::ReduceScatter) {
        v4u x[kRanks];
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r))
            x[r] = escape(loadPack<BytePack<16>>(input + r * nStrideElts, u[k] * EltPerUnit, nElts).v4u);
        }
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r)) put(ses.slot(r, ses.team.rank, u[k]), x[r]);
        }
      } else {
        const v4u x = escape(loadPack<BytePack<16>>(input, u[k] * EltPerUnit, nElts).v4u);
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r)) put(ses.slot(r, ses.team.rank, u[k]), x);
        }
      }
    }
  } else {
    v4u v[kUnroll][kRanks];
    ses.collect(u, valid, v);
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (!valid[k]) continue;
      const int ix = u[k] * EltPerUnit;
      if constexpr (K == Kind::AllGather) {
#pragma unroll
        for (int s = 0; s < kRanks; s++) {
          if (ses.active(s)) storePack(output + s * nStrideElts, ix, nElts, asPack(v[k][s]));
        }
      } else {
        storePack(output, ix, nElts, reduceRanks<NRanks, T>(ses, red, v[k]));
      }
    }
  }
  ses.endEpoch();
}

// Two-shot AllReduce: the block's units split into a part per rank, and each round of up to `per`
// units of every part takes a reduce-scatter epoch (part r to rank r, which reduces it) and an
// all-gather epoch (the reduced parts to every rank), pushing 2 / nRanks of the one-shot's bytes.
// Parts of at most kTwoShotSplitUnits go in one round in which, as in splitBody, pushes and polls never
// share a warp: even warps push the reduce-scatter and collect the all-gather, odd warps collect
// the reduce-scatter, reduce, and push the all-gather. A round's two epochs use different banks,
// and each round ends with every thread's restores acknowledged and a block barrier, so the bank
// a round reuses from the round before was restored, whichever thread did it, before any of this
// rank's pushes that a peer's push into it follows.
template <int NRanks, typename T, typename Red, typename EltType>
__device__ __forceinline__ void twoShotBody(Session<NRanks>& ses, Red red, EltType* input, EltType* output,
                                            int nElts) {
  constexpr int EltPerUnit = 16 / sizeof(EltType);
  const int rank = ses.team.rank;
  const int nRanks = NRanks == kRanks ? kRanks : ses.team.nRanks;
  const int nUnits = divUp(nElts, EltPerUnit);
  const int part = divUp(nUnits, nRanks);
  const bool split = part <= kTwoShotSplitUnits;
  const int w = threadIdx.x / WARP_SIZE;
  const bool rsPush = !split || w % 2 == 0;
  const bool rsCollect = !split || w % 2 == 1;
  const int i = split ? w / 2 * WARP_SIZE + threadIdx.x % WARP_SIZE : (int)threadIdx.x;
  const int step = split ? (int)blockDim.x / 2 : (int)blockDim.x;
  const int per = split ? part : min((int)blockDim.x * kUnroll, kUnits);
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int o = 0; o < part; o += per) {
    const int c = min(per, part - o);
    int u[kUnroll];
    bool valid[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      u[k] = i + k * step;
      valid[k] = u[k] < c;
    }
    if (rsPush) {
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (!valid[k]) continue;
        v4u x[kRanks];
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r))
            x[r] = escape(loadPack<BytePack<16>>(input, (r * part + o + u[k]) * EltPerUnit, nElts).v4u);
        }
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r)) put(ses.slot(r, rank, u[k]), x[r]);
        }
      }
    }
    v4u y[kUnroll];
    if (rsCollect) {
      v4u v[kUnroll][kRanks];
      ses.collect(u, valid, v);
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (valid[k]) y[k] = escape(reduceRanks<NRanks, T>(ses, red, v[k]).v4u);
      }
    }
    ses.nextBank();
    if (rsCollect) {
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (!valid[k]) continue;
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (ses.active(r)) put(ses.slot(r, rank, u[k]), y[k]);
        }
      }
    }
    if (rsPush) {
      v4u v[kUnroll][kRanks];
      ses.collect(u, valid, v);
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (!valid[k]) continue;
#pragma unroll
        for (int s = 0; s < kRanks; s++) {
          if (ses.active(s)) storePack(output, (s * part + o + u[k]) * EltPerUnit, nElts, asPack(v[k][s]));
        }
      }
    }
    ses.endEpoch();
    if (o + per < part) ncclCoopCta().sync();
  }
}

// The register path. EltType is BytePack<16> when the buffers and the per-rank stride are 16 B
// aligned, else the user type (char for AllGather), which loadPack/storePack bound at nElts.
// Units per epoch are min(blockDim.x * kUnroll, kUnits), which the host keeps at kUnits.
template <Kind K, int NRanks, typename T, typename Red, typename EltType>
__device__ __forceinline__ void registerBody(Session<NRanks>& ses, Red red, EltType* input, EltType* output,
                                             int nElts, size_t nStrideElts) {
  constexpr int EltPerUnit = 16 / sizeof(EltType);
  const int rank = ses.team.rank;
  const int nUnits = divUp(nElts, EltPerUnit);
  const int per = min((int)blockDim.x * kUnroll, kUnits);
  if (nUnits == 0) return;
  if (nUnits <= splitUnits()) {
    splitBody<K, NRanks, T>(ses, red, input, output, nElts, nStrideElts);
    return;
  }
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int u0 = 0; u0 < nUnits; u0 += per) {
    int u[kUnroll];
    bool valid[kUnroll];
    unitsOf(min(per, nUnits - u0), u, valid);

    if constexpr (K == Kind::ReduceScatter) {
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        v4u x[kRanks];
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (valid[k] && ses.active(r))
            x[r] = escape(loadPack<BytePack<16>>(input + r * nStrideElts, (u0 + u[k]) * EltPerUnit, nElts).v4u);
        }
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (valid[k] && ses.active(r)) put(ses.slot(r, rank, u[k]), x[r]);
        }
      }
    } else {
      v4u x[kUnroll];
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
        if (valid[k]) x[k] = escape(loadPack<BytePack<16>>(input, (u0 + u[k]) * EltPerUnit, nElts).v4u);
      }
#pragma unroll
      for (int k = 0; k < kUnroll; k++) {
#pragma unroll
        for (int r = 0; r < kRanks; r++) {
          if (valid[k] && ses.active(r)) put(ses.slot(r, rank, u[k]), x[k]);
        }
      }
    }
    v4u v[kUnroll][kRanks];
    ses.collect(u, valid, v);
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (!valid[k]) continue;
      const int ix = (u0 + u[k]) * EltPerUnit;
      if constexpr (K == Kind::AllGather) {
#pragma unroll
        for (int s = 0; s < kRanks; s++) {
          if (ses.active(s)) storePack(output + s * nStrideElts, ix, nElts, asPack(v[k][s]));
        }
      } else {
        storePack(output, ix, nElts, reduceRanks<NRanks, T>(ses, red, v[k]));
      }
    }
    ses.endEpoch();
  }
}

#if NCCL_SYMK_TILE_TDM
// Slots are shared with peers: system scope, kept out of this GPU's caches like put()/poll().
constexpr CachePolicy kTdmSlot = createCachePolicy(TemporalHint::NT, MemScope::SYS);
constexpr CachePolicy kTdmUser = createCachePolicy(TemporalHint::RT, MemScope::DEV);

// One tensor store of n bytes (a multiple of 4) into a slot. Dword elements, never bytes: a dword
// that had landed only in part would be neither the sentinel nor the data, and would read as
// landed.
__device__ __forceinline__ void tdmPush(v4u* slotPtr, char* lds, uint32_t n) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(2);
  g1.tileDim0(n / 4);
  g1.tensorDim0(n / 4);
  gfx1250_TDM_GROUP0 g0((uint32_t)(uintptr_t)lds, (uintptr_t)slotPtr);
  tdm::detail::store<kTdmSlot>(g0, g1);
}

// A window per source. Not in the template below: each instantiation would get its own array, and a
// kernel instantiates it for both team shapes.
__device__ __forceinline__ char* tdmLds() {
  __shared__ __attribute__((aligned(128))) char lds[kRanks * kChunkBytes];
  return lds;
}

// The TDM path, for 16 B aligned buffers and stride. Warp w pushes this rank's chunk for rank w
// with one tensor store instead of a b128 store per lane per unit; a wave has only a few tensor
// ops in flight, so one op per warp keeps the peers' transfers concurrent. A shared source is
// loaded by every warp: eight local reads cost less than the block barriers that sharing one copy
// would take. The block then collects with plain polls, which a tensor poll would queue behind the
// wave's earlier tensor ops.
template <Kind K, int NRanks, typename T, typename Red>
__device__ __forceinline__ void tdmBody(Session<NRanks>& ses, Red red, const char* input, char* output,
                                        size_t nBytes, size_t strideBytes) {
  const int rank = ses.team.rank;
  const int lane = threadIdx.x % WARP_SIZE;
  const int w = (int)__builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
  char* win = tdmLds() + w * kChunkBytes;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (size_t off = 0; off < nBytes; off += kChunkBytes) {
    const uint32_t n = __builtin_amdgcn_readfirstlane((uint32_t)min((size_t)kChunkBytes, nBytes - off));
    if (w < ses.team.nRanks) {
      const char* src = input + (K == Kind::ReduceScatter ? (size_t)w * strideBytes : 0) + off;
      tdm::asyncLoadToLDS<SyncPolicy::Async, kTdmUser>((const uint8_t*)src, (uint8_t*)win, n);
      tdm::tdmWait();
      for (uint32_t b = lane * 16; b < n; b += WARP_SIZE * 16) *(v4u*)(win + b) = escape(*(const v4u*)(win + b));
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
      tdmPush(ses.slot(w, rank, 0), win, n);
      tdm::tdmWait();
    }
    // Polling only once the whole block's pushes are acknowledged keeps the polls from competing
    // with the peers' pushes into this rank.
    ncclCoopCta().sync();

    int u[kUnroll];
    bool valid[kUnroll];
    unitsOf((int)(n / 16), u, valid);
    v4u v[kUnroll][kRanks];
    ses.collect(u, valid, v);
#pragma unroll
    for (int k = 0; k < kUnroll; k++) {
      if (!valid[k]) continue;
      const size_t b = off + (size_t)u[k] * 16;
      if constexpr (K == Kind::AllGather) {
#pragma unroll
        for (int s = 0; s < kRanks; s++) {
          if (ses.active(s)) *(v4u_gptr)(output + s * strideBytes + b) = v[k][s];
        }
      } else {
        *(v4u_gptr)(output + b) = reduceRanks<NRanks, T>(ses, red, v[k]).v4u;
      }
    }
    ses.endEpoch();
    ncclCoopCta().sync();
  }
}
#endif

template <Kind K, bool Tdm, int NRanks, typename T, typename Red>
__device__ __forceinline__ void run(ncclSymkDevWorkArgs const* args, ncclTeam team, uint32_t hdr, Red red, T* input,
                                    T* output, int nElts, size_t nAllElts) {
  Session<NRanks> ses(args, team, hdr);
  const bool aligned = ((uintptr_t)input | (uintptr_t)output | (nAllElts * sizeof(T))) % 16 == 0;
  // Every rank takes the same shot: the choice goes by the call's size alone.
  if constexpr (K == Kind::AllReduce) {
    if (nAllElts * sizeof(T) > ncclSymkNanTwoShotMinBytes) {
      if (__builtin_expect(aligned, true)) {
        twoShotBody<NRanks, T>(ses, red, (BytePack<16>*)input, (BytePack<16>*)output, (int)(nElts * sizeof(T) / 16));
      } else {
        twoShotBody<NRanks, T>(ses, red, input, output, nElts);
      }
      ses.close();
      return;
    }
  }
#if NCCL_SYMK_TILE_TDM
  // An epoch is a chunk for every thread to collect, which takes the full launch width. A part
  // the split path takes gains nothing from tensor stores; every rank takes that path for it,
  // whatever its alignment.
  if constexpr (Tdm) {
    if (aligned && blockDim.x * kUnroll >= kUnits && (size_t)nElts * sizeof(T) > (size_t)splitUnits() * 16) {
      tdmBody<K, NRanks, T>(ses, red, (const char*)input, (char*)output, (size_t)nElts * sizeof(T),
                            nAllElts * sizeof(T));
      ses.close();
      return;
    }
  }
#endif
  if (__builtin_expect(aligned, true)) {
    registerBody<K, NRanks, T>(ses, red, (BytePack<16>*)input, (BytePack<16>*)output,
                               (int)(nElts * sizeof(T) / 16), nAllElts * sizeof(T) / 16);
  } else {
    registerBody<K, NRanks, T>(ses, red, input, output, nElts, nAllElts);
  }
  ses.close();
}

template <Kind K, bool Tdm, bool EnableProfiler, typename T, typename Red>
__device__ __forceinline__ void kernel(ncclSymkDevWorkArgs const* args, Red red) {
  const uint32_t hdr = *Session<kRanks>::headerOf(args);
  ncclSymkArgsHandler handler{args};
  ncclTeam team = ncclTeamLsa(handler.comm);
  handler.singleWork<T>([&] __device__(int nElts, int nAllElts, ncclSymPtr<T> inputPtr, ncclSymPtr<T> outputPtr) {
    T* input = (T*)inputPtr.localPtr();
    T* output = (T*)outputPtr.localPtr();
    if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_AFTER_OPEN);
    if (team.nRanks == kRanks) {
      run<K, Tdm, kRanks>(args, team, hdr, red, input, output, nElts, (size_t)nAllElts);
    } else {
      run<K, Tdm, 0>(args, team, hdr, red, input, output, nElts, (size_t)nAllElts);
    }
    if NCCL_IF_CONSTEXPR (EnableProfiler) ncclSymkProfilerPhase(args, NCCL_KERNEL_PHASE_BEFORE_CLOSE);
  });
}

} // namespace ncclSymkNan

// AllGather moves bytes and never reduces: the escape is the same for every NaN dtype.
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_NaN(ncclSymkDevWorkArgs const* args) {
  ncclSymkNan::kernel<ncclSymkNan::Kind::AllGather, /*Tdm=*/false, EnableProfiler, char>(args, FuncSum<float>(0));
}
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaNaN(ncclSymkDevWorkArgs const* args) {
  ncclSymkNan::kernel<ncclSymkNan::Kind::AllGather, /*Tdm=*/true, EnableProfiler, char>(args, FuncSum<float>(0));
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxNaN_R(ncclSymkDevWorkArgs const* args) {
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(args->getWorks(args->nMaxChannels)->redOpArg);
  ncclSymkNan::kernel<ncclSymkNan::Kind::AllReduce, /*Tdm=*/false, EnableProfiler, T>(args, red);
}
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxTmaNaN_R(ncclSymkDevWorkArgs const* args) {
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(args->getWorks(args->nMaxChannels)->redOpArg);
  ncclSymkNan::kernel<ncclSymkNan::Kind::AllReduce, /*Tdm=*/true, EnableProfiler, T>(args, red);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_NaN(ncclSymkDevWorkArgs const* args) {
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(args->getWorks(args->nMaxChannels)->redOpArg);
  ncclSymkNan::kernel<ncclSymkNan::Kind::ReduceScatter, /*Tdm=*/false, EnableProfiler, T>(args, red);
}
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_TmaNaN(ncclSymkDevWorkArgs const* args) {
  Red<typename ncclSymkAccumType<Red, T, /*nvls=*/false>::Type> red(args->getWorks(args->nMaxChannels)->redOpArg);
  ncclSymkNan::kernel<ncclSymkNan::Kind::ReduceScatter, /*Tdm=*/true, EnableProfiler, T>(args, red);
}

#endif // NCCL_DEVICE_SYMMETRIC_NAN_H_
