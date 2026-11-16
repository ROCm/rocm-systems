/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol reduce-scatter device kernel for the DDA fabric path. Each 16-byte
 * line holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * Reduce-scatter combines the personalized scatter of all-to-all with the fold
 * of all-reduce: rank selfRank owns output shard selfRank, so recvbuff[i] =
 * sum over ranks r of sendbuff_r[selfRank*recvcount + i]. Phase 1 scatters this
 * rank's per-peer chunk (sendbuff[peer]) into peer's slot; phase 2 folds every
 * incoming chunk (plus this rank's own self-chunk) into recvbuff. Only the
 * per-peer publish source and the reduce seed differ from all_reduce_dda_ll.h.
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

namespace dda::common {

// Per-shard payload cap, enforced by eligibility. The slot stride is fixed so
// every rank addresses the same geometry. One bank is nRanks slots with the
// 8B payload expanded to 16B: 8 MiB at 1 MiB/rank and 4 ranks.
constexpr size_t kDdaLLRsMaxBytes = 1ULL*1024*1024;         // 1 MiB per-rank (= 4 MiB total at 4 ranks)
constexpr size_t kDdaLLRsSlotStridePkts = kDdaLLRsMaxBytes / 8;   // 131072

// Bytes in one bank: nRanks slots at the fixed stride.
constexpr size_t ddaLLRsBankBytes(int nRanks) {
  return (size_t)nRanks * kDdaLLRsSlotStridePkts * sizeof(LLPacket16);
}

// Packet offset of this epoch's bank. Odd epochs start at bankSize, where the
// other LL tiers sharing this scratch put bank 1.
constexpr size_t ddaLLRsBankOffsetPkts(uint32_t flag, size_t bankSize) {
  return (size_t)(flag & 1u) * (bankSize / sizeof(LLPacket16));
}

// Phase-2 reads in flight at once. Each peer keeps two data words and two
// flags, so 4 stays in registers; 8 matches the send batch but costs twice
// the temporary values.
constexpr int kDdaLLRsRecvBatch = 4;

// Consecutive threads that share packets between peer lanes. A span lies inside one wave for wave32 and wave64.
constexpr int kDdaLLRsSpanLanes = 32;

// Most lanes one packet's peers are split across. A lane then covers 32 / 4 consecutive packets of a peer's slot, one
// whole 128B line per store or read.
constexpr int kDdaLLRsMaxPeerLanes = 4;

// Lanes a packet's peers are split across on nRanks ranks: the fewest that fit every lane's peers in one batch, at most
// kDdaLLRsMaxPeerLanes.
constexpr int ddaLLRsPeerLanes(int nRanks) {
  return nRanks - 1 <= kDdaLLPeerBatch ? 1 : nRanks - 1 <= 2 * kDdaLLPeerBatch ? 2 : kDdaLLRsMaxPeerLanes;
}

// LL reduce-scatter kernel. 1D grid over packets (8B payload each) of the
// per-rank shard (recvcount elements), kPeerLanes threads per packet.
//
// Phase 1 (scatter): rank selfRank writes its chunk-for-peer (sendbuff[peer])
// into peer's scratch at slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank selfRank seeds the accumulator with its own self-chunk
// (sendbuff[selfRank]), polls its own scratch slot for each other rank (waiting
// on the flag), and sums those into recvbuff. Flag polling provides cross-rank
// ordering, so no GPU barrier is used. Scratch is double-buffered: bank = flag & 1.
//
// With kPeerLanes > 1 a packet's peers are split across lanes, subset s taking those s + 1, s + 1 + kPeerLanes, ...
// past selfRank, and the subsets' sums are added across lanes, so a lane walks (nRanks - 1) / kPeerLanes of them.
template <typename T, int NRANKS_CT, int kPeerLanes = 1>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaReduceScatterFabricLL(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                           T* __restrict__ recvbuff, // local user output (recvcount elems)
                                           const T* __restrict__ sendbuff, // local user input (recvcount*nRanks)
                                           size_t recvcount, // per-rank shard element count
                                           int selfRank, int nRanksRt,
                                           uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                           int epochLen, // number of cells in epochDev
                                           size_t bankSize, // scratch bank size (from host)
                                           DdaKernelParams params) {
  static_assert(kPeerLanes == 1 || kPeerLanes == 2 || kPeerLanes == kDdaLLRsMaxPeerLanes, "1, 2 or 4 peer lanes");
  static_assert(NRANKS_CT == 0 || kPeerLanes == ddaLLRsPeerLanes(NRANKS_CT), "lanes follow the rank count");
  constexpr bool kStaticRanks = NRANKS_CT != 0;
  constexpr int kSpanPkts = kDdaLLRsSpanLanes / kPeerLanes;

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = recvcount * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  const size_t slot = kDdaLLRsSlotStridePkts;
  const size_t chunkWords = nPk * 2; // uint32 words per shard chunk

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = (int)(threadIdx.x % kDdaLLRsSpanLanes);
  const int sub = kPeerLanes == 1 ? 0 : lane / kSpanPkts;
  const size_t pk0 = (gtid - lane) / kPeerLanes + lane % kSpanPkts;
  const size_t stride = (size_t)gridDim.x * blockDim.x / kPeerLanes;
  // Peers of this lane's subset. The j-th sits sub + 1 + j * kPeerLanes past selfRank.
  const int nSub = (nRanks - 1 - sub + kPeerLanes - 1) / kPeerLanes;
  auto peerAt = [&](int j) {
    const int p = selfRank + sub + 1 + j * kPeerLanes;
    return p < nRanks ? p : p - nRanks;
  };

#if (defined(__gfx1250__) || defined(__gfx1250_strict__)) && __has_builtin(__builtin_amdgcn_global_prefetch)
  // Prefetch the send chunks into L2 before the epoch barrier, one 128B line per 16 packets split across the subsets.
  // th 1 walks page tables.
  if ((pk0 & 15) == 0 && pk0 < nPk) {
    const char* base = reinterpret_cast<const char*>(sendbuff) + pk0 * 8;
    for (int r = sub; r < nRanks; r += kPeerLanes) {
      __builtin_amdgcn_global_prefetch((const __attribute__((address_space(1))) void*)(base + (size_t)r * bytes), 0x11);
    }
  }
#endif
  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = ddaLLRsBankOffsetPkts(flag, bankSize);

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);

  // Phase 1: scatter sendbuff[peer] into peer's slot[self]. A batch reads its payloads and destinations before its
  // first store: every store ends with a compiler memory barrier, so a destination read after one waits by itself.
  for (size_t pk = pk0; pk < nPk; pk += stride) {
    for (int base = 0; base < nSub; base += kDdaLLPeerBatch) {
      uint32_t d0[kDdaLLPeerBatch], d1[kDdaLLPeerBatch];
      LLPacket16* dst[kDdaLLPeerBatch];
#pragma unroll
      for (int i = 0; i < kDdaLLPeerBatch; ++i) {
        if (kStaticRanks && base + i >= nSub) {
          continue;
        }
        // A runtime count reads slots past the end from the batch's first peer, so no read sits under a branch.
        const int peer = peerAt(base + i < nSub ? base + i : base);
        const uint32_t* inPeer = in + (size_t)peer * chunkWords;
        d0[i] = inPeer[2 * pk];
        d1[i] = inPeer[2 * pk + 1];
        dst[i] = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      }
#pragma unroll
      for (int i = 0; i < kDdaLLPeerBatch; ++i) {
        if (base + i < nSub) {
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[i][pk]), d0[i], flag, d1[i], flag);
        }
      }
    }
  }

  // Phase 2: seed with my self-chunk, poll my slots for the others, reduce.
  const uint32_t* inSelf = in + (size_t)selfRank * chunkWords;
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = pk0; pk < nPk; pk += stride) {
    // Only subset 0 seeds with the self-chunk, so the sum across subsets counts it once.
    const uint32_t self0 = inSelf[2 * pk];
    const uint32_t self1 = inSelf[2 * pk + 1];
    uint32_t acc0 = sub == 0 ? self0 : 0u;
    uint32_t acc1 = sub == 0 ? self1 : 0u;
    if constexpr (kStaticRanks) {
      // Each pass issues every read of a peer group before any flag is tested, so
      // those remote reads stay in flight together. The group is read again until
      // every line is fresh.
      for (int base = 0; base < nSub; base += kDdaLLRsRecvBatch) {
        uint32_t d0[kDdaLLRsRecvBatch], f0[kDdaLLRsRecvBatch];
        uint32_t d1[kDdaLLRsRecvBatch], f1[kDdaLLRsRecvBatch];
        // Read the batch until every line is fresh; the first pass is the first read, each pass one abort spin tick.
        int spins = 0;
        while (true) {
          // Each pass reads every line, sharing one wait; a fresh line cannot change before this call ends.
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
          asm volatile("" ::: "memory");
          v4u got[kDdaLLRsRecvBatch];
#pragma unroll
          for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
            if (base + i < nSub) {
              const uint32_t* src = reinterpret_cast<const uint32_t*>(myBase + (size_t)peerAt(base + i) * slot + pk);
              got[i] = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
            }
          }
          // Keep every load above this point so the flag tests cannot sink one between them.
          asm volatile("" ::: "memory");
#pragma unroll
          for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
            if (base + i < nSub) {
              d0[i] = got[i][0];
              f0[i] = got[i][1];
              d1[i] = got[i][2];
              f1[i] = got[i][3];
            }
          }
#else
#pragma unroll
          for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
            if (base + i < nSub) {
              const uint32_t* src = reinterpret_cast<const uint32_t*>(myBase + (size_t)peerAt(base + i) * slot + pk);
              ddaLLLoadLineB128(src, d0[i], f0[i], d1[i], f1[i]);
            }
          }
          // Keep every load above this point so none sinks past the flag tests or out of the loop.
          asm volatile("" ::: "memory");
#endif
          bool fresh = true;
#pragma unroll
          for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
            if (base + i < nSub && (f0[i] != flag || f1[i] != flag)) {
              fresh = false;
            }
          }
          if (fresh) {
            break;
          }
          if (params.abortSpinTick(spins)) {
            ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
            return;
          }
        }
#pragma unroll
        for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
          if (base + i < nSub) {
            ddaLLAddPayload<T>(acc0, acc1, d0[i], d1[i]);
          }
        }
      }
    } else {
      // One sweep reads up to a send batch of this lane's lines, a slot past the end rereading the first, so no read
      // sits under a branch. Between sweeps a lane spins on one stale line.
      for (int base = 0; base < nSub; base += kDdaLLPeerBatch) {
        const int live = nSub - base < kDdaLLPeerBatch ? nSub - base : kDdaLLPeerBatch;
        const uint32_t* line[kDdaLLPeerBatch];
        uint32_t d0[kDdaLLPeerBatch], d1[kDdaLLPeerBatch];
#pragma unroll
        for (int i = 0; i < kDdaLLPeerBatch; ++i) {
          line[i] = reinterpret_cast<const uint32_t*>(myBase + (size_t)peerAt(i < live ? base + i : base) * slot + pk);
        }
        if (!ddaLLWaitLineFlagsBatched<kDdaLLPeerBatch>(line, (1u << live) - 1u, flag, params, d0, d1)) {
          ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
          return;
        }
#pragma unroll
        for (int i = 0; i < kDdaLLPeerBatch; ++i) {
          if (i < live) {
            ddaLLAddPayload<T>(acc0, acc1, d0[i], d1[i]);
          }
        }
      }
    }
    if constexpr (kPeerLanes > 1) {
      // A packet's subsets sit kSpanPkts lanes apart, so adding the partner's sum at each stride leaves subset 0 with
      // the packet's total, in a fixed order.
#pragma unroll
      for (int m = kSpanPkts; m < kDdaLLRsSpanLanes; m *= 2) {
        const uint32_t o0 = __shfl_xor(acc0, m);
        const uint32_t o1 = __shfl_xor(acc1, m);
        ddaLLAddPayload<T>(acc0, acc1, o0, o1);
      }
    }
    if (sub == 0) {
      out[2 * pk] = acc0;
      out[2 * pk + 1] = acc1;
    }
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
