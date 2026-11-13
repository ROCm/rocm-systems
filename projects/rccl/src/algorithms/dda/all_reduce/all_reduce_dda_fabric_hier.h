/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-hierarchical AllReduce in one kernel on the fabric/VMM path: reduce
 * within the host, reduce across hosts, gather within the host. Every phase
 * pushes into peer scratch and reduces from local scratch, and each block
 * synchronizes only with the ranks it pushed to, so blocks overlap phases.
 *
 * P_CT and N_CT fix ranks per host and hosts at compile time so the peer loops
 * unroll and the peer pointers stay in registers; 0 reads them from the layout.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"

namespace dda::common {

constexpr int kHierFusedMaxRanksPerHost = 16;
constexpr int kHierFusedMaxHosts = 16;

struct HierFusedLayout {
  int nHosts;
  int ranksPerHost;
  int hostIndex;
  int localRank;
  int intra[kHierFusedMaxRanksPerHost]; // Rank of each local rank on this host
  int inter[kHierFusedMaxHosts];        // Rank of this local rank on each host
};

// Elements of one scratch bank per element of a round: P slices received from the host, N from the other hosts, and P
// gathered back from the host.
__host__ __device__ constexpr size_t hierBankElems(size_t roundCount, int ranksPerHost, int nHosts) {
  return roundCount / ranksPerHost * (2 * ranksPerHost + nHosts);
}

template <typename T>
__device__ __forceinline__ uint4 hierLoad(const T* p) {
  return *reinterpret_cast<const uint4*>(p);
}

template <typename T>
__device__ __forceinline__ void hierStore(T* p, const uint4& v) {
  *reinterpret_cast<uint4*>(p) = v;
}

// Sums n slots `slotStride` apart, always in slot order, so every rank that sums the same slots gets the same bits.
template <typename T, int kMax>
__device__ __forceinline__ uint4 hierSumSlots(const T* first, size_t slotStride, int n) {
  uint4 vals[kMax];
#pragma unroll
  for (int i = 0; i < kMax; i++) {
    if (i < n) vals[i] = hierLoad(first + i * slotStride);
  }
  uint4 sum = vals[0];
#pragma unroll
  for (int i = 1; i < kMax; i++) {
    if (i < n) sum = vecElementAdd<T>(sum, vals[i]);
  }
  return sum;
}

// A bank holds [P host slots | N inter-host slots | P gathered slots], each `roundCount / P` elements even when the last
// round is shorter: barriers only order blocks with the same index, so every block must touch the same scratch
// addresses in every round. A round's bank is reused two rounds later, after the barriers of the round in between have
// ordered every reader.
template <typename T, int P_CT, int N_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ddaAllReduceHierFabric(T* const* __restrict__ bufs, T* __restrict__ recvbuff,
                                         const T* __restrict__ sendbuff, size_t count, size_t roundCount,
                                         HierFusedLayout lay, FabricGpuBarrier barrier) {
  constexpr size_t kVec = sizeof(uint4) / sizeof(T);
  constexpr int kMaxP = P_CT > 0 ? P_CT : kHierFusedMaxRanksPerHost;
  constexpr int kMaxN = N_CT > 0 ? N_CT : kHierFusedMaxHosts;
  const int P = P_CT > 0 ? P_CT : lay.ranksPerHost;
  const int N = N_CT > 0 ? N_CT : lay.nHosts;
  const int local = lay.localRank;

  T* intra[kMaxP];
#pragma unroll
  for (int p = 0; p < kMaxP; p++) intra[p] = p < P ? bufs[lay.intra[p]] : nullptr;
  T* inter[kMaxN];
#pragma unroll
  for (int h = 0; h < kMaxN; h++) inter[h] = h < N ? bufs[lay.inter[h]] : nullptr;
  T* const own = bufs[lay.intra[local]];

  const size_t slot = roundCount / P;
  const size_t bankStride = hierBankElems(roundCount, P, N);
  const size_t start = (static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x) * kVec;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * kVec;

  // Peers may still read their scratch in an earlier DDA kernel.
  barrier.syncOnSameBlockIdx<false /* hasPreviousMemAccess */, true /* hasSubsequentMemAccess */>();

  int bank = 0;
  for (size_t base = 0; base < count; base += roundCount, bank ^= 1) {
    const size_t slice = min(roundCount, count - base) / P;
    const size_t hostSlots = bank * bankStride;
    const size_t interSlots = hostSlots + P * slot;
    const size_t gatherSlots = interSlots + N * slot;

    for (size_t j = start; j < slice; j += stride) {
      uint4 vals[kMaxP];
#pragma unroll
      for (int s = 0; s < kMaxP; s++) {
        if (s < P) vals[s] = hierLoad(sendbuff + base + s * slice + j);
      }
#pragma unroll
      for (int s = 0; s < kMaxP; s++) {
        if (s < P) hierStore(intra[s] + hostSlots + local * slot + j, vals[s]);
      }
    }
    barrier.orderDestWrites();
    barrier.syncWithRanks(lay.intra, P);

    if (N > 1) {
      for (size_t j = start; j < slice; j += stride) {
        const uint4 sum = hierSumSlots<T, kMaxP>(own + hostSlots + j, slot, P);
#pragma unroll
        for (int h = 0; h < kMaxN; h++) {
          if (h < N) hierStore(inter[h] + interSlots + lay.hostIndex * slot + j, sum);
        }
      }
      barrier.orderDestWrites();
      barrier.syncWithRanks(lay.inter, N);
    }

    for (size_t j = start; j < slice; j += stride) {
      const uint4 sum = N > 1 ? hierSumSlots<T, kMaxN>(own + interSlots + j, slot, N)
                              : hierSumSlots<T, kMaxP>(own + hostSlots + j, slot, P);
#pragma unroll
      for (int s = 0; s < kMaxP; s++) {
        if (s < P) hierStore(intra[s] + gatherSlots + local * slot + j, sum);
      }
    }
    barrier.orderDestWrites();
    barrier.syncWithRanks(lay.intra, P);

    for (size_t j = start; j < slice; j += stride) {
      uint4 vals[kMaxP];
#pragma unroll
      for (int s = 0; s < kMaxP; s++) {
        if (s < P) vals[s] = hierLoad(own + gatherSlots + s * slot + j);
      }
#pragma unroll
      for (int s = 0; s < kMaxP; s++) {
        if (s < P) hierStore(recvbuff + base + s * slice + j, vals[s]);
      }
    }
  }
  // No rank may return while a peer still reads its scratch.
  barrier.syncOnSameBlockIdx<true /* hasPreviousMemAccess */, false /* hasSubsequentMemAccess */>();
}

} // namespace dda::common
