// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Slot and flag steps shared by the hierarchical ReduceScatter and AllReduce.

#ifndef NCCL_DEVICE_SYMMETRIC_HIER_H_
#define NCCL_DEVICE_SYMMETRIC_HIER_H_

#include "sym_kernels.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "symmetric/symk.h"
#include "symmetric/data_ops.h"
#else
#include "symk.cuh"
#include "data_ops.cuh"
#endif

// LSA rank of the package rank i places up from ours, so copy 0 is our own.
static __device__ __forceinline__ int hierCopyLsa(ncclDevComm const& comm, ncclTeam scaleIn, int i) {
  int r = scaleIn.rank + i;
  if (r >= scaleIn.nRanks) r -= scaleIn.nRanks;
  return ncclTeamRankToLsa(comm, scaleIn, r);
}

// Rank of scaleUp s + 1 places up from ours: where our s-th send goes, and whose credits we wait on.
static __device__ __forceinline__ int hierPeerUp(ncclTeam scaleUp, int s) {
  int r = scaleUp.rank + 1 + s;
  return r >= scaleUp.nRanks ? r - scaleUp.nRanks : r;
}

// Rank of scaleUp s + 1 places down from ours: whose send lands in our s-th slot.
static __device__ __forceinline__ int hierPeerDown(ncclTeam scaleUp, int s) {
  int r = scaleUp.rank - 1 - s;
  return r < 0 ? r + scaleUp.nRanks : r;
}

// This block's slots for a round on the given rank of scaleUp, one per other package. Rounds alternate
// between two sets so senders can store a round while the owners' finishers still read the one before.
static __device__ char* hierSlots(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle slotsBuf, int peer,
                                  uint64_t round, size_t chunkBytes) {
  size_t roundBytes = size_t(scaleUp.nRanks - 1) * chunkBytes;
  return (char*)ncclGetResourceBufferLsaPointer(comm, slotsBuf, ncclTeamRankToLsa(comm, scaleUp, peer)) +
         (2 * blockIdx.x + (round & 1)) * roundBytes;
}

// How the hierarchical kernels run reduceBatch: plain loads through the global aperture, the
// packages' slots written side by side, and each group of sources in flight together.
struct hierBatchMode : ReduceBatchMode {
  using SrcSpace = GMemApertureTag;
  static constexpr bool Interleave = true;
  static constexpr bool Overlap = true;
};

// A sender's part of a round: for each other package, sums our package's copies of the block its
// counterpart of us owns and stores that into our slot there.
template <typename T, typename Red>
static __device__ void hierReduceSend(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red,
                                      ncclCoopWarpSpan role, ncclSymPtr<T> input, size_t nAllElts,
                                      ncclDevResourceHandle slotsBuf, uint64_t round, int nElts, size_t chunkBytes,
                                      bool aligned) {
  // A pointer getter so the window is read before the loops: ncclSymPtr re-reads it after every store.
  ncclLsaPointerGetter<T> inputLsa(input);
  reduceBatch</*PostOp=*/false, hierBatchMode>(
    role, /*inPlace=*/false, /*nBatch=*/scaleUp.nRanks - 1, nElts, GMemApertureTag(), /*dstAlignMin=*/16,
    /*getDst=*/
    [&] __device__(int s) -> T* {
      return (T*)(hierSlots(comm, scaleUp, slotsBuf, hierPeerUp(scaleUp, s), round, chunkBytes) + s * chunkBytes);
    },
    red, /*nSrcs=*/scaleIn.nRanks,
    /*getSrc=*/
    [&] __device__(int s, int i) -> T* {
      return inputLsa(hierCopyLsa(comm, scaleIn, i)) +
             ncclTeamRankToWorld(comm, scaleUp, hierPeerUp(scaleUp, s)) * nAllElts;
    },
    /*srcPtrCommonMask=*/aligned ? 15u : 0u, /*getSrcPtrMasked=*/[&] __device__(int) -> unsigned { return 0u; });
}

// A finisher's part of a round: sums our package's copies of our own block with the sums the other
// packages stored into our slots, and writes that to output with the post-op applied.
template <typename T, typename Red>
static __device__ void hierReduceFinish(ncclDevComm const& comm, ncclTeam scaleIn, ncclTeam scaleUp, Red red,
                                        ncclCoopWarpSpan role, ncclSymPtr<T> input, size_t nAllElts, char const* slots,
                                        T* output, int nElts, size_t chunkBytes, bool aligned) {
  ncclLsaPointerGetter<T> blockLsa(input + comm.rank * nAllElts);
  size_t const slotElts = chunkBytes / sizeof(T);
  reduceBatch</*PostOp=*/true, hierBatchMode>(
    role, /*inPlace=*/false, /*nBatch=*/1, nElts, GMemApertureTag(), /*dstAlignMin=*/16,
    /*getDst=*/[&] __device__(int) -> T* { return output; }, red, /*nSrcs=*/scaleIn.nRanks + scaleUp.nRanks - 1,
    /*getSrc=*/
    [&] __device__(int, int i) -> T* {
      return i < scaleIn.nRanks ? blockLsa(hierCopyLsa(comm, scaleIn, i)) :
                                  (T*)slots + size_t(i - scaleIn.nRanks) * slotElts;
    },
    /*srcPtrCommonMask=*/aligned ? 15u : 0u, /*getSrcPtrMasked=*/[&] __device__(int) -> unsigned { return 0u; });
}

// This block's row of the flags buffer on the given rank of scaleUp.
static __device__ uint64_t* hierFlagRow(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle flagsBuf,
                                        int peer) {
  return (uint64_t*)ncclGetResourceBufferLsaPointer(comm, flagsBuf, ncclTeamRankToLsa(comm, scaleUp, peer)) +
         blockIdx.x * ncclSymkRsHierFlagsPerBlock(scaleUp.nRanks);
}

// Raises flag base + s on the rank of scaleUp s + 1 places up from ours, after a system-scope release.
static __device__ void hierRaiseUp(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle flagsBuf, int base,
                                   uint64_t value) {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int s = 0; s < scaleUp.nRanks - 1; s++) {
    cuda::atomic_ref<uint64_t>(hierFlagRow(comm, scaleUp, flagsBuf, hierPeerUp(scaleUp, s))[base + s])
      .store(value, cuda::memory_order_relaxed);
  }
}

// Raises flag base + s on the rank of scaleUp s + 1 places down from ours, after a system-scope release.
static __device__ void hierRaiseDown(ncclDevComm const& comm, ncclTeam scaleUp, ncclDevResourceHandle flagsBuf,
                                     int base, uint64_t value) {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int s = 0; s < scaleUp.nRanks - 1; s++) {
    cuda::atomic_ref<uint64_t>(hierFlagRow(comm, scaleUp, flagsBuf, hierPeerDown(scaleUp, s))[base + s])
      .store(value, cuda::memory_order_relaxed);
  }
}

// Spins until each of the n flags starting at flags reaches target, then orders the caller's later
// accesses after what their raisers did before raising them.
static __device__ void hierWaitFlags(ncclDevComm const& comm, uint64_t* flags, int n, uint64_t target) {
  uint32_t steps = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int s = 0; s < n; s++) {
    cuda::atomic_ref<uint64_t> flag(flags[s]);
    while (flag.load(cuda::memory_order_relaxed) < target) {
      if (nccl::utility::testAbort(comm.abortFlag, steps)) return;
    }
  }
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

#endif // NCCL_DEVICE_SYMMETRIC_HIER_H_
