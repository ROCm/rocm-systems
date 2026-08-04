/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA path following RCCL LL128 protocol.
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

#include "algorithms/ll128_pack.h"

namespace meta::comms {

template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
__global__ void ddaAllGatherLL128(
    T* const* __restrict__ peerScratch,    // ddaPeerPtrsDev: nRanks scratch bases
    T* __restrict__ recvbuff,              // local user output
    const T* __restrict__ sendbuff,        // local user input
    size_t perRankBytes,                   // per-rank payload; multiple of 16
    int selfRank,
    int nRanksRt,
    uint32_t* __restrict__ epochDev,       // per-block LL epoch cells
    int epochLen,                          // number of cells in epochDev
    size_t slicesTotal,                    // host-derived slice count
    int wireWordPerSlice,                  // u64 words on the wire per slice
    int dataBytesPerSlice) {               // payload bytes per slice

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;
  const int peer = (int)blockIdx.x;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % ll128::kWarp;
  const int warp = tid / ll128::kWarp;
  const int nwarps = nthreads / ll128::kWarp;
  const bool flagLane = ll128::isFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  uint32_t f = epochDev[flatBlockId] + 1u;
  if (f == 0u) f = 2u;                     // skip 0 sentinel; keep bank parity
  const uint32_t flag32 = f;
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const uint32_t bank = flag32 & 1u;

  const uint64_t slotWords = slicesTotal * (uint64_t)wireWordPerSlice;
  const uint64_t bankWords = (uint64_t)bank * (uint64_t)nRanks * slotWords;

  // Slices stride by warp within this peer's column only.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  if (peer == selfRank) {
    // Local copy sendbuff -> recvbuff[selfRank], split across this column.
    const uint4* s4 = reinterpret_cast<const uint4*>(sendbuff);
    uint4* d4 = reinterpret_cast<uint4*>(
        reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes);
    const size_t nVec = perRankBytes >> 4;  // 16B chunks
    const size_t gtid = (size_t)blockIdx.y * (size_t)nthreads + (size_t)tid;
    const size_t stride = (size_t)gridDim.y * (size_t)nthreads;
    for (size_t i = gtid; i < nVec; i += stride) {
      const uint4* p = &s4[i];
      uint4 v;
      v.x = __builtin_nontemporal_load(&p->x);
      v.y = __builtin_nontemporal_load(&p->y);
      v.z = __builtin_nontemporal_load(&p->z);
      v.w = __builtin_nontemporal_load(&p->w);
      uint4* q = &d4[i];
      __builtin_nontemporal_store(v.x, &q->x);
      __builtin_nontemporal_store(v.y, &q->y);
      __builtin_nontemporal_store(v.z, &q->z);
      __builtin_nontemporal_store(v.w, &q->w);
    }
  } else {
    const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
    uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) +
        bankWords + (uint64_t)selfRank * slotWords;
    const uint64_t* gatherSlot =
        reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
        (uint64_t)peer * slotWords;
    int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

    // Phase 1: pack and push this column's slices to the one peer it owns.
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)dataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
          rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
      uint64_t regs[ll128::kWordsPerThread];
      ll128::loadRegs<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
      ll128::storeWire(scatterSlot + s * (size_t)wireWordPerSlice + 2 * lane,
                       regs, flag, flagLane);
    }

    // Phase 2: poll the same slices in that peer's slot and unpack.
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)dataBytesPerSlice;
      const size_t rem = perRankBytes - dataByte;
      const int eltInSlice =
          rem < (size_t)dataBytesPerSlice ? (int)rem : dataBytesPerSlice;
      uint64_t vr[ll128::kWordsPerThread];
      ll128::pollWire(gatherSlot + s * (size_t)wireWordPerSlice + 2 * lane,
                      vr, flag, lane);
      ll128::storeRegs<int8_t>(dstBytes + dataByte, vr, eltInSlice, lane, flagLane);
    }
  }

  __syncthreads();
  for (int e = flatBlockId + tid * total; e < epochLen; e += total * nthreads) {
    epochDev[e] = flag32;
  }
}

} // namespace meta::comms
