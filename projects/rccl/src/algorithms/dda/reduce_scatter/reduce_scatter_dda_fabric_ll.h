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

// Per-shard staging slot capacity and hard per-shard cap (enforced in the
// eligibility check). The slot stride is fixed so every rank addresses the
// same geometry. Bank 1 starts at bankSize, the same byte offset as the other
// DDA LL tiers that share this scratch and epoch counter.
// One bank holds nRanks * (kDdaLLRsMaxBytes * 2) bytes for the 8B->16B
// expansion; 8 MiB at 1 MiB/rank * 4 ranks * 2 (LL overhead).
constexpr size_t kDdaLLRsMaxBytes = 1ULL*1024*1024;         // 1 MiB per-rank (= 4 MiB total at 4 ranks)
constexpr size_t kDdaLLRsSlotStridePkts = kDdaLLRsMaxBytes / 8;   // 131072

// Bytes one bank occupies: nRanks slots at the fixed stride. The host passes
// ddaBankSize() as bankSize; bank 1 starts there, not after this many bytes.
constexpr size_t ddaLLRsBankBytes(int nRanks) {
  return (size_t)nRanks * kDdaLLRsSlotStridePkts * sizeof(LLPacket16);
}

// Packet offset of the bank selected by this epoch. Odd epochs start at
// bankSize, the same byte as all-reduce and all-gather LL.
constexpr size_t ddaLLRsBankOffsetPkts(uint32_t flag, size_t bankSize) {
  return (size_t)(flag & 1u) * (bankSize / sizeof(LLPacket16));
}

// Fixed-width peer staging for phase 1, same as the two-shot all-reduce.
// A runtime-sized array, or a runtime index into one, is placed in scratch.
// 8 covers rank counts 4 and 8 in a single pass.
constexpr int kDdaLLRsPeerBatch = 8;
// Phase-2 reads in flight at once. Each peer keeps two data words and two
// flags, so 4 stays in registers; 8 matches the send batch but costs twice
// the temporary values.
constexpr int kDdaLLRsRecvBatch = 4;

// LL reduce-scatter kernel. 1D grid over packets (8B payload each) of the
// per-rank shard (recvcount elements).
//
// Phase 1 (scatter): rank selfRank writes its chunk-for-peer (sendbuff[peer])
// into peer's scratch at slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank selfRank seeds the accumulator with its own self-chunk
// (sendbuff[selfRank]), polls its own scratch slot for each other rank (waiting
// on the flag), and sums those into recvbuff. Flag polling provides cross-rank
// ordering, so no GPU barrier is used. Scratch is double-buffered: bank = flag & 1,
// and bank 1 starts at bankSize (half the scratch), matching the other LL tiers.
template <typename T, int NRANKS_CT>
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
                                           size_t bankSize) { // scratch bank size (from host)

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = recvcount * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet
  const size_t slot = kDdaLLRsSlotStridePkts;
  const size_t chunkWords = nPk * 2; // uint32 words per shard chunk

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  // ddaBankSize() floors the bank to a multiple of 16B, so this is a whole
  // number of LL packets and bank 1 starts at the same byte as all-reduce LL.
  const size_t bankOffsetPkts = ddaLLRsBankOffsetPkts(flag, bankSize);

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);

  // Phase 1: scatter my chunk-for-peer (sendbuff[peer]) into peer's slot[self].
  // Loads are gathered before the stores, matching the two-shot all-reduce.
  // Interleaved, each store waits on the next load and the per-peer streams
  // stop overlapping. The fixed trip count with the rank test as a predicate
  // unrolls both loops and keeps d0/d1 in registers.
  for (size_t pk = gtid; pk < nPk; pk += stride) {
    for (int base = 1; base < nRanks; base += kDdaLLRsPeerBatch) {
      uint32_t d0[kDdaLLRsPeerBatch], d1[kDdaLLRsPeerBatch];
#pragma unroll
      for (int i = 0; i < kDdaLLRsPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* inPeer = in + (size_t)peer * chunkWords;
          d0[i] = inPeer[2 * pk];
          d1[i] = inPeer[2 * pk + 1];
        }
      }
#pragma unroll
      for (int i = 0; i < kDdaLLRsPeerBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
          ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0[i], flag, d1[i], flag);
        }
      }
    }
  }

  // Phase 2: seed with my self-chunk, poll my slots for the others, reduce.
  // The first read of each peer group is issued before any flag is tested, so
  // those remote reads stay in flight together. A line whose flags already
  // match is kept; only a stale line is read again.
  const uint32_t* inSelf = in + (size_t)selfRank * chunkWords;
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < nPk; pk += stride) {
    uint32_t acc0 = inSelf[2 * pk];
    uint32_t acc1 = inSelf[2 * pk + 1];
    for (int base = 1; base < nRanks; base += kDdaLLRsRecvBatch) {
      uint32_t d0[kDdaLLRsRecvBatch], f0[kDdaLLRsRecvBatch];
      uint32_t d1[kDdaLLRsRecvBatch], f1[kDdaLLRsRecvBatch];
#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
      v4u got[kDdaLLRsRecvBatch];
#pragma unroll
      for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* src = reinterpret_cast<const uint32_t*>(myBase + (size_t)peer * slot + pk);
          got[i] = __builtin_amdgcn_global_load_b128((v4u_gptr)src, RCCL_SYSTEM_SYNCSCOPE);
        }
      }
      // Keep every peer load above this point so the flag tests cannot sink
      // the later loads back between them.
      asm volatile("" ::: "memory");
#pragma unroll
      for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          union { v4u v; uint32_t w[4]; } u;
          u.v = got[i];
          d0[i] = u.w[0];
          f0[i] = u.w[1];
          d1[i] = u.w[2];
          f1[i] = u.w[3];
        }
      }
#else
#pragma unroll
      for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* src = reinterpret_cast<const uint32_t*>(myBase + (size_t)peer * slot + pk);
          ddaLLLoadLineB128(src, d0[i], f0[i], d1[i], f1[i]);
        }
      }
#endif
#pragma unroll
      for (int i = 0; i < kDdaLLRsRecvBatch; ++i) {
        const int r = base + i;
        if (r < nRanks) {
          const int peer = (selfRank + r) % nRanks;
          const uint32_t* src = reinterpret_cast<const uint32_t*>(myBase + (size_t)peer * slot + pk);
          while (f0[i] != flag || f1[i] != flag) {
            ddaLLLoadLineB128(src, d0[i], f0[i], d1[i], f1[i]);
          }
          acc0 = vecElementAdd<T>(acc0, d0[i]);
          acc1 = vecElementAdd<T>(acc1, d1[i]);
        }
      }
    }
    out[2 * pk] = acc0;
    out[2 * pk + 1] = acc1;
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
