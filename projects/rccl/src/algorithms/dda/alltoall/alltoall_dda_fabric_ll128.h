/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-to-all device kernel for the DDA fabric path (gfx1250), in the
 * register-packed wire format of all_gather_dda_fabric_ll128.h (see CollCommon_ll128.h).
 * No GPU barrier; staging uses comm->ddaScratch reached via comm->ddaPeerPtrsDev.
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

namespace dda::common {

// LL128 AllToAll: block column x serves peers self+x+1, self+x+1+gridDim.x, ..., one 2 KiB slice
// per warp at a time. The self copy uses the whole grid; per-peer slots share one scratch bank.

constexpr size_t ddaLL128A2ASlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllToAllFabricLL128(T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
                                         const T* __restrict__ sendbuff,
                                         size_t perChunkBytes, // per-peer payload; multiple of 16
                                         int selfRank, int nRanks, uint32_t* __restrict__ epochDev, int epochLen,
                                         size_t slicesTotal, // slices per peer chunk
                                         size_t bankSize, DdaKernelParams params) {
  const DdaPeerColumns peerCols{nRanks - 1};
  const auto peerOf = [=](int x) { return (selfRank + x + 1) % nRanks; };
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1u);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128A2ASlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  for (const int x : peerCols) {
    const int peer = peerOf(x);
    const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff) + (size_t)peer * perChunkBytes;
    uint64_t* scatterSlot =
      reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords + (uint64_t)selfRank * (uint64_t)slotWords;
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = perChunkBytes - dataByte;
      const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t regs[kDdaLL128WordsPerThread];
      ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, eltInSlice, lane, flagLane);
      ddaLL128StoreWire(scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
    }
  }

  ddaGridCopy16B(reinterpret_cast<const int8_t*>(sendbuff) + (size_t)selfRank * perChunkBytes,
                 reinterpret_cast<int8_t*>(recvbuff) + (size_t)selfRank * perChunkBytes, perChunkBytes,
                 (size_t)flatBlockId * (size_t)nthreads + (size_t)tid, (size_t)total * (size_t)nthreads);

  for (const int x : peerCols) {
    const int peer = peerOf(x);
    const uint64_t* gatherSlot =
      reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords + (uint64_t)peer * (uint64_t)slotWords;
    int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perChunkBytes;
    for (size_t s = gwarp; s < slicesTotal; s += wstride) {
      const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
      const size_t rem = perChunkBytes - dataByte;
      const int eltInSlice = rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
      uint64_t vr[kDdaLL128WordsPerThread];
      const uint64_t* wire = gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane;
#if RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS
      ddaLL128PollWireBatched(wire, vr, flag, lane, params);
#else
      // Plain 16B loads can see a line's flag before its payload; only this poll reads it again.
      ddaLL128PollWire(wire, vr, flag, lane, params);
#endif
      ddaLL128StoreRegs<int8_t>(dstBytes + dataByte, vr, eltInSlice, lane, flagLane);
    }
  }
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
