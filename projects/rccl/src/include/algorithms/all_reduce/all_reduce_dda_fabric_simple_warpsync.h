/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * "simple_warpsync" all-reduce kernel: a copy of the LL128 warpsync variant
 * (all_reduce_dda_fabric_ll128_warpsync.h) intended as a starting point for a
 * simple-protocol staging experiment. Selectable at runtime against the LL128
 * warpsync kernel via RCCL_DDA_AR_SIMPLE_WARPSYNC (see collectives.cc).
 *
 * Like the LL128 warpsync kernel it carries payload in ALL 16 words of each 128B
 * line (no in-line flag word) and synchronizes phases with a separate arrival
 * barrier instead of per-line flags:
 *
 *  - Data staging starts kDdaSimpleWarpsyncArReserveBytes into each peer's scratch
 *    (peerScratch + kDdaSimpleWarpsyncArReserveBytes). Lines are double-buffered
 *    (bank = flag & 1).
 *  - The first kDdaSimpleWarpsyncArReserveBytes is repurposed as the barrier flag region:
 *    each rank owns a per-rank sub-region, indexed by global warp id.
 *
 * Sync (between phase 1 and phase 2): every warp publishes the current epoch
 * flag to each peer's flag slot, then waits until every peer has published to
 * it (an all-to-all, per-global-warp arrival barrier). Because the line->warp
 * mapping is identical on every rank and in both phases, peer p's warp w wrote
 * exactly the lines self's warp w reads, so a per-warp barrier is sufficient.
 * Unlike the data region, the flag region is NOT double-buffered, so the wait
 * uses a monotonic ">= flag" compare to tolerate a peer racing one call ahead.
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

#include "algorithms/CollCommon.h"
#include "algorithms/CollCommon_ll128.h"
#include "algorithms/all_reduce/all_reduce_dda_fabric_ll128_warpsync.h" // kDdaLL128ArReserveBytes

namespace meta::comms {

//using TT = uint64_t;
using TT = uint4;

// Front region of every peer's scratch reserved for the barrier flag exchange.
// Data staging begins at peerScratch + kDdaSimpleWarpsyncArReserveBytes.
constexpr size_t kDdaSimpleWarpsyncArReserveBytes = 1048576;                 // 1 MiB

// simple_warpsync flat all-reduce kernel (full-line + barrier variant). 1D grid
// over 128B lines; within a block the threads split into 16-lane groups, each
// owning one line at a time. Behaviourally identical to the LL128 warpsync
// kernel (see all_reduce_dda_fabric_ll128_warpsync.h); kept as a separate
// symbol so the two can be swapped at runtime and diverge independently.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
__global__ void ddaAllReduceFlatSimpleWarpsync(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t count,                          // full-message element count
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen) {                        // number of cells in epochDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes / sizeof(TT);                 // 8B payload words
  const size_t kDdaSimpleWarpsyncArReserveWords = kDdaSimpleWarpsyncArReserveBytes / sizeof(TT);
  // make multiple of 16
  const size_t slot = ((nWords + 16 - 1) / 16) * 16;

  // Flat block id + total launched blocks. tid 0 reads our own epoch cell (all
  // cells hold the same value) and derives this launch's flag on the device, so
  // nothing is baked into a HIP graph capture. bank = flag & 1.
  const int flatBlockId = blockIdx.x;
  const int total = gridDim.x;
  __shared__ uint32_t s_flag;
  if (threadIdx.x == 0) {
    uint32_t f = epochDev[flatBlockId] + 1u;
    if (f == 0u) f = 2u;                   // skip 0 sentinel; keep bank parity
    s_flag = f;
  }
  __syncthreads();
  const uint32_t flag = s_flag;
  const size_t bankOffsetLines = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const TT* in = reinterpret_cast<const TT*>(sendbuff);
  TT* out = reinterpret_cast<TT*>(recvbuff);

  // Phase 1: publish my payload into every peer's slot[selfRank].
  // No in-line flag word: all lanes carry payload.
  // Staging base is offset by kDdaSimpleWarpsyncArReserveWords (first 1 MiB reserved).
  for (size_t tid = gtid; tid < nWords; tid += stride) {
    const TT v = in[tid];
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      TT* dst = reinterpret_cast<TT*>(peerScratch[peer]) + kDdaSimpleWarpsyncArReserveWords +
        bankOffsetLines + (size_t)selfRank * slot;
      //dst[tid] = v;
      ddaLL128StoreWord(dst + tid, v);
    }
  }

#if RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__gfx1250__)
  asm volatile("s_wait_loadcnt 0x0\n\ts_wait_storecnt 0x0");
#else
  asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
#endif
  __atomic_signal_fence(__ATOMIC_SEQ_CST);
#else
  __threadfence_system();
#endif

  // Arrival barrier (between phase 1 and phase 2). The reserved 1 MiB front
  // region is split into per-rank sub-regions (flag_perRankBytes each), indexed
  // by global warp id. Each warp's lanes [0, nRanks) publish this epoch's flag
  // to every peer (lane r -> peer r, into peer's sub-region[selfRank]), then the
  // same lanes wait until every peer has published to this warp's slot.
  size_t flag_perRankBytes = kDdaSimpleWarpsyncArReserveBytes / nRanks;
  size_t flag_selfRankOffset = flag_perRankBytes * selfRank;
  int gwarpid = gtid / (int)warpSize;
  int lwarpid = gtid % (int)warpSize;
  #pragma unroll
  for (int r = lwarpid; r < nRanks; r+=(int)warpSize) {
    uint64_t *dst_flag = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(peerScratch[r]) + flag_selfRankOffset);
    ddaLL128StoreWord(dst_flag + gwarpid, flag);
  }

  #pragma unroll
  for (int r = lwarpid; r < nRanks; r+=(int)warpSize) {
      const uint64_t *src_flag = reinterpret_cast<size_t*>(reinterpret_cast<char*>(peerScratch[selfRank]) + flag_perRankBytes * r);
      // Monotonic compare (not ==): epochs strictly increase, so ">= flag" means
      // "peer arrived this call or later". This tolerates a peer racing one call
      // ahead and overwriting its flag slot, which is otherwise unprotected
      // because the flag region (unlike the data region) is not double-buffered.
      while (ddaLL128LoadWord(src_flag + gwarpid) < (uint64_t)flag) {}
  }

  __syncwarp();

  // Phase 2: read my slots for the other ranks, fold with my own data.
  // No flag word to poll; all lanes carry payload.
  for (size_t tid = gtid; tid < nWords; tid += stride) {
    TT acc = in[tid];
    #pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      const TT* src = reinterpret_cast<const TT*>(peerScratch[selfRank]) +
        kDdaSimpleWarpsyncArReserveWords + bankOffsetLines + (size_t)peer * slot;
      //const TT d = src[tid];
      const TT d = ddaLL128LoadWord(src + tid);
      acc = ddaLL128AddWord<T>(acc, d);
    }
    out[tid] = acc;
  }

  if (threadIdx.x == 0) {
    for (int e = flatBlockId; e < epochLen; e += total) {
      epochDev[e] = flag;
    }
  }
}

} // namespace meta::comms
