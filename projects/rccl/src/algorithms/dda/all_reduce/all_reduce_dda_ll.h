/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL-protocol all-reduce device kernel for the DDA path. Each 16-byte line
 * holds 8 bytes of payload and two 4-byte flags; the flags carry cross-rank
 * sync, so no GPU barrier is needed. Staging uses the DDA scratch
 * (comm->ddaScratch, reachable via comm->ddaPeerPtrsDev).
 *
 * This is the all-reduce analogue of all_gather_dda_fabric_ll.h. The kernel is
 * codepath-agnostic (it only needs the peer scratch table), so the same kernel
 * backs the fabric launcher (dda_all_reduce_fabric_ll.cu) and can equally back
 * an IPC launcher.
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

// LL is for small-message, so the full payload is well under the staging cap.
// each packet takes 16 bytes.
constexpr size_t kDdaLLArSlotStridePkts = kDdaLLMaxBytes / sizeof(LLPacket16);

// LL flat all-reduce kernel. 1D grid over packets (8B payload each).
//
// Phase 1 (publish): rank selfRank writes its full sendbuff into every peer's
// scratch at slot selfRank, as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank selfRank polls its own scratch slot for each other
// rank (waiting on the flag), and sums those with its own sendbuff into
// recvbuff. The flag polling provides the cross-rank ordering, so no GPU
// barrier is used.
//
// Scratch is double-buffered: bank = epoch & 1, selected via bankOffsetPkts.
// Self does not round-trip through scratch; its contribution is read directly
// from sendbuff in the reduce.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceFlatLL(T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
                                     T* __restrict__ recvbuff,              // local user output
                                     const T* __restrict__ sendbuff,        // local user input
                                     size_t count,                          // full-message element count
                                     int selfRank, int nRanksRt,
                                     uint32_t* __restrict__ epochDev,       // per-block LL epoch cells (shared AG+AR)
                                     int epochLen,                          // number of cells in epochDev
                                     const uint32_t* __restrict__ abortFlag) { // comm->abortFlagDev

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3;           // 8 payload bytes per packet
  const size_t slot = kDdaLLArSlotStridePkts;

  const uint32_t flag = ddaGetLLEpochInc(epochDev, blockIdx.x, 1);
  const size_t bankOffsetPkts = (size_t)(flag & 1u) * (size_t)nRanks * slot;

  const size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;

  const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);
  uint32_t* out = reinterpret_cast<uint32_t*>(recvbuff);

  // Phase 1: publish my payload into every peer's slot[selfRank].
  for (size_t pk = gtid; pk < nPk; pk += stride) {
    const uint32_t d0 = in[2 * pk];
    const uint32_t d1 = in[2 * pk + 1];

#pragma unroll
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      LLPacket16* dst = reinterpret_cast<LLPacket16*>(peerScratch[peer]) + bankOffsetPkts + (size_t)selfRank * slot;
      ddaLLStoreLineB128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
    }
  }

  // Phase 2: poll my slots for the other ranks, reduce with my own data.
  LLPacket16* myBase = reinterpret_cast<LLPacket16*>(peerScratch[selfRank]) + bankOffsetPkts;
  for (size_t pk = gtid; pk < nPk; pk += stride) {
    uint32_t acc0 = in[2 * pk];
    uint32_t acc1 = in[2 * pk + 1];
    for (int r = 1; r < nRanks; ++r) {
      const int peer = (selfRank + r) % nRanks;
      volatile LLPacket16* src = myBase + (size_t)peer * slot;
      uint32_t d0, f0, d1, f1;
      if (!ddaLLWaitLineFlags(reinterpret_cast<const uint32_t*>(const_cast<LLPacket16*>(&src[pk])), flag, abortFlag, d0,
                              f0, d1, f1)) {
        ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
        return;
      }
      acc0 = vecElementAdd<T>(acc0, d0);
      acc1 = vecElementAdd<T>(acc1, d1);
    }
    out[2 * pk] = acc0;
    out[2 * pk + 1] = acc1;
  }

  ddaSetLLEpoch(epochDev, epochLen, blockIdx.x, gridDim.x, flag);
}

} // namespace dda::common
