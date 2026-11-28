/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LL128-protocol all-gather device kernel for the DDA fabric path (gfx1250).
 * A warp owns one 2 KiB slice: eight uint64 registers per lane. 
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

// Slot geometry, derived from the scratch bank the host picked. The bank splits
// evenly across ranks and the per-rank stride is floored to a whole number of
// slices, so a slot always begins on a slice boundary and the 16B wire accesses
// stay aligned however nRanks divides the bank.
constexpr size_t ddaLL128AgSlotWords(size_t bankSize, int nRanks) {
  return ddaLL128SlotWords(bankSize, nRanks, 1);
}

// Payload bytes in slice s: a whole slice, or the partial last one.
__device__ __forceinline__ int ddaLL128AGSliceBytes(size_t perRankBytes, size_t s) {
  const size_t rem = perRankBytes - s * (size_t)kDdaLL128DataBytesPerSlice;
  return rem < (size_t)kDdaLL128DataBytesPerSlice ? (int)rem : kDdaLL128DataBytesPerSlice;
}

// Copies sendbuff to this rank's output slot, 16 B a thread over the whole grid. In place there is nothing to copy.
template <typename T>
__device__ __forceinline__ void ddaLL128AGLocalCopy(T* recvbuff, const T* sendbuff, size_t perRankBytes, int selfRank,
                                                    int flatBlockId, int total) {
  char* localRecv = reinterpret_cast<char*>(recvbuff) + (size_t)selfRank * perRankBytes;
  if (reinterpret_cast<const char*>(sendbuff) == localRecv) {
    return;
  }
  v4u_gptr s4 = (v4u_gptr)sendbuff;
  v4u_gptr d4 = (v4u_gptr)localRecv;
  const size_t nVec = perRankBytes >> 4; // 16B chunks
  const size_t gtid = (size_t)flatBlockId * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t stride = (size_t)total * (size_t)blockDim.x;
  for (size_t i = gtid; i < nVec; i += stride) {
    d4[i] = s4[i];
  }
}

// Phase 2: unpacks this column's peer's slices from my slot, a warp a slice. Stops on abort; stores done on return.
__device__ __forceinline__ void ddaLL128AGGather(const uint64_t* gatherSlot, int8_t* dstBytes, size_t perRankBytes,
                                                 size_t slicesTotal, uint64_t flag, int warp, int nwarps, int lane,
                                                 bool flagLane, DdaKernelParams params) {
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;
  for (size_t s = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp; s < slicesTotal; s += wstride) {
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!ddaLL128PollWireBatched(gatherSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, vr, flag, lane,
                                 params)) {
      break; // not return: every thread must reach the caller's __syncthreads
    }
    ddaLL128StoreRegs<int8_t>(dstBytes + s * (size_t)kDdaLL128DataBytesPerSlice, vr,
                              ddaLL128AGSliceBytes(perRankBytes, s), lane, flagLane);
  }
  ddaLL128WaitStores();
}

// LL128 all-gather. 2D grid: grid.x == nRanks - 1 places one column per remote
// peer; grid.y splits that peer's slices across blocks, one
// warp per slice. Each column packs this rank's payload into its peer's slot,
// then polls its own slot for that peer's payload and unpacks it.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(1024)
#endif
  __global__ void ddaAllGatherFabricLL128(T* const* __restrict__ peerScratch, // ddaPeerPtrsDev: nRanks scratch bases
                                          T* __restrict__ recvbuff, // local user output
                                          const T* __restrict__ sendbuff, // local user input
                                          size_t perRankBytes, // per-rank payload; multiple of 16
                                          int selfRank, int nRanksRt,
                                          uint32_t* __restrict__ epochDev, // per-block LL epoch cells
                                          int epochLen, // number of cells in epochDev
                                          size_t slicesTotal, // slices this call uses
                                          size_t bankSize, // scratch bank size (from host)
                                          DdaKernelParams params) {

  const int nRanks = NRANKS_CT ? NRANKS_CT : nRanksRt;

  const int column = (int)blockIdx.x;
  const int peer = (selfRank + column + 1) % nRanks;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  const int warp = tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128AgSlotWords(bankSize, nRanks);
  // Bank 1 sits at bankSize rather than right after the slots, so every DDA LL
  // tier sharing this scratch and epoch counter banks at the same byte offsets.
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));

  // Slices stride by warp within this peer's column only.
  const size_t gwarp = (size_t)blockIdx.y * (size_t)nwarps + (size_t)warp;
  const size_t wstride = (size_t)gridDim.y * (size_t)nwarps;

  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);
  uint64_t* scatterSlot = reinterpret_cast<uint64_t*>(peerScratch[peer]) + bankWords +
    (uint64_t)selfRank * (uint64_t)slotWords;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;

  // Phase 1: pack and push this column's slices to the one peer it owns.
  for (size_t s = gwarp; s < slicesTotal; s += wstride) {
    const size_t dataByte = s * (size_t)kDdaLL128DataBytesPerSlice;
    uint64_t regs[kDdaLL128WordsPerThread];
    ddaLL128LoadRegsBatched<int8_t>(regs, srcBytes + dataByte, ddaLL128AGSliceBytes(perRankBytes, s), lane, flagLane);
    ddaLL128StoreWire(
      scatterSlot + s * (size_t)kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
  }

  ddaLL128AGLocalCopy(recvbuff, sendbuff, perRankBytes, selfRank, flatBlockId, total);
  ddaLL128AGGather(gatherSlot, dstBytes, perRankBytes, slicesTotal, flag, warp, nwarps, lane, flagLane, params);
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

// Block size of the staged kernel: its two LDS windows per warp take 64 KiB.
constexpr unsigned kDdaLL128AGStageThreads = 512;

// Most peers a fill takes (the ladder's top rung). Above 7, two fills exceed the 6-bit async counter, so the wave
// stalls at issue: correct but slower.
constexpr int kDdaLL128AGStageMaxPushPeers = (1 << kDdaLL128StageWaitRungs) - 1;

// Peers a fill goes to. Off the ladder's rungs (3, 7), so a refill also waits on one push of the last fill.
constexpr int kDdaLL128AGStagePushPeers = 4;
static_assert(kDdaLL128AGStagePushPeers >= 1 && kDdaLL128AGStagePushPeers <= kDdaLL128AGStageMaxPushPeers,
              "a fill goes to 1..kDdaLL128AGStageMaxPushPeers peers");
#if ASYNC_COPY_SUPPORTED
// 63 is the 6-bit async counter's top.
static_assert(2 * kDdaLL128AGStagePushPeers * kDdaLL128StagePushOps <= 63, "two fills overflow the async counter");
#endif

// Staged peer groups: ceil(nPeers / pushPeers), group g holding peer indices g, g + groups, ...
// Peer index i is rank (self + 1 + i) % nRanks.
__host__ __device__ constexpr int ddaLL128AGStageGroups(int nPeers, int pushPeers) {
  return (nPeers + pushPeers - 1) / pushPeers;
}

// Peers in group g of `groups`.
__host__ __device__ constexpr int ddaLL128AGStageGroupPeers(int g, int groups, int nPeers) {
  return (nPeers - g + groups - 1) / groups;
}

// Slices that row `row` of a `rows`-tall grid polls in phase 2: one chunk of nwarps slices in every rows * nwarps.
__host__ __device__ constexpr uint32_t ddaLL128AGRowSlices(uint32_t slicesTotal, uint32_t row, uint32_t rows,
                                                           uint32_t nwarps) {
  const uint32_t rem = slicesTotal % (rows * nwarps);
  const uint32_t first = row * nwarps;
  const uint32_t tail = rem <= first ? 0 : (rem - first < nwarps ? rem - first : nwarps);
  return slicesTotal / (rows * nwarps) * nwarps + tail;
}

// Index of the k-th of those slices.
__host__ __device__ constexpr uint32_t ddaLL128AGRowSlice(uint32_t k, uint32_t row, uint32_t rows, uint32_t nwarps) {
  return k / nwarps * (rows * nwarps) + row * nwarps + k % nwarps;
}

// ddaAllGatherFabricLL128 with an LDS-staged phase 1: ddaLL128StageBytes(blockDim.x) of dynamic LDS,
// pushPeers 1..kDdaLL128AGStageMaxPushPeers.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(kDdaLL128AGStageThreads)
#endif
  __global__ void ddaAllGatherFabricLL128Staged(T* const* __restrict__ peerScratch, T* __restrict__ recvbuff,
                                                const T* __restrict__ sendbuff, size_t perRankBytes, int selfRank,
                                                int nRanks, uint32_t* __restrict__ epochDev, int epochLen,
                                                size_t slicesTotal, size_t bankSize, DdaKernelParams params,
                                                int pushPeers) {
  constexpr int kWire = kDdaLL128WireAsync;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kDdaLL128Warp;
  // Wave uniform where an LL128 warp is a wave (wave32), so the unit bookkeeping stays in scalar registers.
  const int warp = __builtin_amdgcn_wavefrontsize() == kDdaLL128Warp ?
                     __builtin_amdgcn_readfirstlane(tid / kDdaLL128Warp) :
                     tid / kDdaLL128Warp;
  const int nwarps = nthreads / kDdaLL128Warp;
  const bool flagLane = ddaLL128IsFlagLane(lane);

  extern __shared__ __attribute__((aligned(kDdaLL128LineBytes))) uint8_t ddaLL128AGStageLds[];
  DdaLL128Stage stage = ddaLL128StageInit(ddaLL128AGStageLds, warp);

  const int flatBlockId = (int)(blockIdx.x * gridDim.y + blockIdx.y);
  const int total = (int)(gridDim.x * gridDim.y);
  const uint32_t flag32 = ddaGetLLEpochInc(epochDev, flatBlockId, 1);
  const uint64_t flag = ((uint64_t)flag32 << 32) | (uint64_t)flag32;
  const size_t slotWords = ddaLL128AgSlotWords(bankSize, nRanks);
  const uint64_t bankWords = (uint64_t)(flag32 & 1u) * (uint64_t)(bankSize / sizeof(uint64_t));
  const int8_t* srcBytes = reinterpret_cast<const int8_t*>(sendbuff);

  // Phase 1: (slice, peer group) units over the slices this row polls, so a row waits only on its own row of the peers.
  const int nPeers = nRanks - 1;
  const uint32_t groups = (uint32_t)ddaLL128AGStageGroups(nPeers, pushPeers);
  const uint32_t units = ddaLL128AGRowSlices((uint32_t)slicesTotal, blockIdx.y, gridDim.y, (uint32_t)nwarps) * groups;
  const uint32_t unitStride = gridDim.x * (uint32_t)nwarps;
  auto unitSlice = [&](uint32_t unit) {
    return ddaLL128AGRowSlice(unit / groups, blockIdx.y, gridDim.y, (uint32_t)nwarps);
  };
  uint32_t unit = blockIdx.x * (uint32_t)nwarps + (uint32_t)warp;
  if (unit < units) {
    uint64_t cur[kDdaLL128WordsPerThread];
    const uint32_t firstSlice = unitSlice(unit);
    ddaLL128LoadRegsBatched<int8_t>(cur, srcBytes + (size_t)firstSlice * (size_t)kDdaLL128DataBytesPerSlice,
                                    ddaLL128AGSliceBytes(perRankBytes, firstSlice), lane, flagLane);
#pragma unroll 1
    for (; unit < units; unit += unitStride) {
      const uint32_t slice = unitSlice(unit);
      const int group = (int)(unit % groups);
      // Look-ahead: past the last unit, reload this one, so the loads stay in bounds and unconditional.
      const uint32_t nextSlice = unitSlice(unit + unitStride < units ? unit + unitStride : unit);
      uint64_t nxt[kDdaLL128WordsPerThread];
      ddaLL128LoadRegsBatched<int8_t>(nxt, srcBytes + (size_t)nextSlice * (size_t)kDdaLL128DataBytesPerSlice,
                                      ddaLL128AGSliceBytes(perRankBytes, nextSlice), lane, flagLane);
      if constexpr (ddaLL128Staged<kWire>()) {
        // Defensive: a VGPR count would make the ladder's rungs divergent and wait to 0.
        stage.lastOps = __builtin_amdgcn_readfirstlane(stage.lastOps);
      }
      ddaLL128StageFill<kWire>(stage, cur, flag, flagLane, lane);
      const int groupPeers = ddaLL128AGStageGroupPeers(group, (int)groups, nPeers);
      // Start at member slice % groupPeers, so the warps of a group spread their first pushes over its peers.
      int member = (int)(slice % (uint32_t)groupPeers);
#pragma unroll 1
      for (int push = 0; push < groupPeers; ++push) {
        const int peerRank = (selfRank + 1 + group + member * (int)groups) % nRanks;
        uint64_t* wireSlice = reinterpret_cast<uint64_t*>(peerScratch[peerRank]) + bankWords +
                              (uint64_t)selfRank * (uint64_t)slotWords + (size_t)slice * kDdaLL128WireWordsPerSlice;
        ddaLL128StagePush<kWire>(stage, wireSlice, cur, flag, flagLane, lane);
        member = member + 1 == groupPeers ? 0 : member + 1;
      }
#pragma unroll
      for (int w = 0; w < kDdaLL128WordsPerThread; ++w) {
        cur[w] = nxt[w];
      }
    }
  }

  ddaLL128AGLocalCopy(recvbuff, sendbuff, perRankBytes, selfRank, flatBlockId, total);
  const int peer = (selfRank + (int)blockIdx.x + 1) % nRanks;
  const uint64_t* gatherSlot = reinterpret_cast<const uint64_t*>(peerScratch[selfRank]) + bankWords +
    (uint64_t)peer * (uint64_t)slotWords;
  int8_t* dstBytes = reinterpret_cast<int8_t*>(recvbuff) + (size_t)peer * perRankBytes;
  ddaLL128AGGather(gatherSlot, dstBytes, perRankBytes, slicesTotal, flag, warp, nwarps, lane, flagLane, params);
  // The windows are LDS: every async push must be done before the block leaves.
  ddaLL128StageDrain<kWire>();
  __syncthreads();
  ddaSetLLEpoch(epochDev, epochLen, flatBlockId, total, flag32);
}

} // namespace dda::common
