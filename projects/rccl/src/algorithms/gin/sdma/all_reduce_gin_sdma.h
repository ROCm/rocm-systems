/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LSA AllReduce device kernels:
 *   allReduceLsaOneShotKernel — read all peers, sum, write all peers (small messages).
 *   lsaAllReduceTwoShotKernel — LSA reduce-scatter, barrier, LSA all-gather (remote reads only).
 *   ginAllReduceTwoShotKernel  — LSA reduce-scatter, intra-GPU CTA barrier, GIN all-gather from recv.
 *   ginAllReduceTwoShotGinScatterKernel — CTA 0 GIN-puts remote send slices
 *     (self from send, remotes from scratch). Scratch is 512 MiB of CE-style
 *     ping-pong incoming slots; reduced output is recvbuff. Local reduce matches
 *     CE (16B vectors, GpuUnroll=4, ranks 0..nRanks-1). Used on gfx1250 from 64 MiB
 *     when the Anvil SDMA GIN backend is enabled.
 *
 * One-shot follows projects/rccl-tests/src/all_reduce.cu allReduceLsaKernel.
 ******************************************************************************/

#pragma once

#include "nccl_device.h"
#include "nccl_device/ptr.h"
#include "algorithms/dda/device/CollCommon.h"
#include "algorithms/gin/gin_all_reduce_policy.h"

namespace gin::sdma {
using dda::common::vecElementAdd;

// Scalar sum for the one-shot kernel; covers float, half and bf16.
//
// bf16 must not go through __hadd. That is a half intrinsic whose bf16 overload only exists on
// CUDA for __CUDA_ARCH__ >= 800 (device/reduce_kernel.h gates it on exactly that), so on HIP it
// either fails to compile or silently converts bf16 -> half -> bf16, clipping bf16's 8-bit
// exponent to half's 5-bit one. operator+ is what the DDA bf16 path uses (vecElementAdd in
// dda/device/CollCommon.h) and is available wherever __hadd is on CUDA.
template <typename T>
__device__ __forceinline__ T allReduceLsaSumAdd(T a, T b) {
  return a + b;
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void allReduceLsaOneShotKernel(ncclWindow_t sendWin, size_t sendOff, ncclWindow_t recvWin,
                                            size_t recvOff, size_t count, struct ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int rank = devComm.rank;
  const int nRanks = devComm.nRanks;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;

  for (size_t offset = static_cast<size_t>(globalTid); offset < count;
       offset += static_cast<size_t>(globalNthreads)) {
    T v = T{0};
    for (int peer = 0; peer < nRanks; peer++) {
      T* sendPtr = reinterpret_cast<T*>(ncclGetLsaPointer(sendWin, sendOff, peer));
      v = allReduceLsaSumAdd(v, sendPtr[offset]);
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* recvPtr = reinterpret_cast<T*>(ncclGetLsaPointer(recvWin, recvOff, peer));
      recvPtr[offset] = v;
    }
  }

  bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

// 16B window accesses through the global aperture.
//
// ncclGetLsaPointer hands back a generic (flat) pointer built by bit-twiddling a base loaded from
// the window struct, so nothing downstream can prove it is device memory and the vector accesses
// compile to flat_load_dwordx4 / flat_store_dwordx4. Peer windows live in the global aperture, so
// casting to an address_space(1) pointer leaves the semantics untouched (plain, non-atomic, ordered
// by the surrounding LSA barriers) while emitting global_load_dwordx4 / global_store_dwordx4 —
// exactly what rccl_ptr.h prescribes for hot paths. If an ISA dump of this kernel mentions "flat",
// one of these casts was lost.
__device__ __forceinline__ uint4 lsaLoadVec(const char* p) {
  union {
    v4u vec;
    uint4 val;
  } u;
  u.vec = *(v4u_gptr)(const_cast<char*>(p));
  return u.val;
}

__device__ __forceinline__ void lsaStoreVec(char* p, uint4 val) {
  union {
    v4u vec;
    uint4 val;
  } u;
  u.val = val;
  *(v4u_gptr)(p) = u.vec;
}

// A rank's own column of an LSA window, resolved for every peer at once.
//
// ncclGetLsaPointer(win, off, peer) is lsaFlatBase + (peer * stride4G << 32) + off, i.e. linear in
// peer. Resolving base and stride once per kernel is what keeps the window-struct loads out of the
// hot loop: called inline, ncclGetLsaPointer reloads lsaFlatBase and stride4G for every peer of
// every vector, because the peer stores may alias the window and stop the compiler from hoisting
// them. That was ~2 dependent loads per peer per 16B vector ahead of every data access.
struct LsaColumn {
  char* base;        // peer 0, at this rank's column
  size_t peerStride; // bytes between consecutive peers
};

__device__ __forceinline__ LsaColumn lsaColumnResolve(ncclWindow_t win, size_t colByteOff, int nRanks) {
  char* peer0 = reinterpret_cast<char*>(ncclGetLsaPointer(win, colByteOff, 0));
  const size_t stride =
    nRanks > 1 ? static_cast<size_t>(reinterpret_cast<char*>(ncclGetLsaPointer(win, colByteOff, 1)) - peer0) : 0;
  return LsaColumn{peer0, stride};
}

// NRANKS_CT > 0 folds the clique size to a constant and fully unrolls the peer loop (as the DDA
// kernels do); NRANKS_CT == 0 is the runtime fallback with an 8-wide partial unroll.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ uint4 lsaReduceVec(const LsaColumn& sendCol, size_t elemOff, int nRanksRuntime) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 0) ? NRANKS_CT : 8;
  const char* src = sendCol.base + elemOff * sizeof(T);

  uint4 sum{0, 0, 0, 0};
  uint4 srcVals[2];
  srcVals[0] = lsaLoadVec(src);
#pragma unroll kUnroll
  for (int peer = 0; peer < nRanks - 1; ++peer) {
    srcVals[(peer + 1) & 1] = lsaLoadVec(src + (peer + 1) * sendCol.peerStride);
    sum = vecElementAdd<T>(sum, srcVals[peer & 1]);
  }
  return vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);
}

// Pull all-gather: read each peer's reduced column out of its own window into the local recv
// buffer. Column srcRank sits at (srcRank * peerStride) in the flat mapping and at
// (srcRank * colStride) inside the window, so one combined stride walks both.
//
// This rank's own column is skipped: the reduce-scatter wrote it straight into the recv window, so
// gathering it would be a local copy onto itself. (DDA cannot skip it — its reduce-scatter lands in
// the IPC scratch, so the self shard still has to be copied out to the user buffer.) Peers are
// visited starting at rank+1, as DDA's all-gather does, so the ranks are not all pulling from the
// same peer at the same instant.
template <typename T, int NRANKS_CT>
__device__ __forceinline__ void lsaGatherVec(const char* peer0Recv, char* localRecv, size_t peerStride,
                                             size_t colStride, size_t byteOff, int rank, int nRanksRuntime) {
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;
  constexpr int kUnroll = (NRANKS_CT > 1) ? NRANKS_CT - 1 : 8;

#pragma unroll kUnroll
  for (int r = 1; r < nRanks; ++r) {
    int srcRank = rank + r;
    if (srcRank >= nRanks) {
      srcRank -= nRanks;
    }
    lsaStoreVec(localRecv + srcRank * colStride + byteOff,
                lsaLoadVec(peer0Recv + srcRank * (peerStride + colStride) + byteOff));
  }
}

// Non-overlapped two-shot: LSA reduce-scatter into this rank's own recv column, one barrier, then
// pull every peer's column back. All stores are local; only the loads cross the fabric.
//
// Three barrier round trips, matching ddaAllReduceTreeIpc: acquire on entry (peers' send buffers
// must be complete), acq_rel in the middle, release on exit (a peer may still be reading our column
// when we return). The middle one used to be a release barrier followed by an acquire barrier,
// which paid two fabric round trips for ordering that one acq_rel barrier expresses: its arrive
// carries the release fence and its wait does the acquire loads.
template <typename T, int NRANKS_CT>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void lsaAllReduceTwoShotKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                            ncclWindow_t recvWin, size_t recvOff, size_t countPerRank,
                                            int nRanksRuntime) {
  ncclCoopCta cta;
  ncclLsaBarrierSession<ncclCoopCta> bar{cta, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};
  const int nRanks = (NRANKS_CT > 0) ? NRANKS_CT : nRanksRuntime;

  const size_t colStride = countPerRank * sizeof(T);
  const size_t colByteOff = static_cast<size_t>(devComm.rank) * colStride;
  const LsaColumn sendCol = lsaColumnResolve(sendWin, sendOff + colByteOff, nRanks);
  const LsaColumn recvCol = lsaColumnResolve(recvWin, recvOff + colByteOff, nRanks);

  // recvCol.base is peer 0's window at this rank's column; stepping one peer stride lands on this
  // rank's own copy of it, and backing out colByteOff gives the start of each recv buffer.
  char* localCol = recvCol.base + static_cast<size_t>(devComm.rank) * recvCol.peerStride;
  const char* peer0Recv = recvCol.base - colByteOff;
  char* localRecv = localCol - colByteOff;

  constexpr size_t countPerThread = sizeof(uint4) / sizeof(T);
  const size_t idxStart = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * countPerThread;
  const size_t idxStride = static_cast<size_t>(gridDim.x) * blockDim.x * countPerThread;

  bar.sync(cta, cuda::memory_order_acquire);

  for (size_t idx = idxStart; idx < countPerRank; idx += idxStride) {
    lsaStoreVec(localCol + idx * sizeof(T), lsaReduceVec<T, NRANKS_CT>(sendCol, idx, nRanks));
  }

  bar.sync(cta, cuda::memory_order_acq_rel);

  for (size_t idx = idxStart; idx < countPerRank; idx += idxStride) {
    lsaGatherVec<T, NRANKS_CT>(peer0Recv, localRecv, recvCol.peerStride, colStride, idx * sizeof(T), devComm.rank,
                               nRanks);
  }

  bar.sync(cta, cuda::memory_order_release);
}

// One-time init only, never part of a captured graph: replaying this would zero signals that
// peers are concurrently incrementing. ncclGinAllReduceInitOnce runs it on a private stream.
__global__ void ginAllReduceResetSignalsKernel(struct ncclDevComm devComm) {
  if (threadIdx.x != 0) {
    return;
  }
  ncclGin gin{devComm, /*ginContext=*/0};
  gin.resetSignal(static_cast<unsigned>(blockIdx.x));
}

// Intra-GPU barrier across every CTA on this device. LSA/world bar.sync only
// pair CTA i with CTA i on other ranks, so they do not wait for sibling CTAs here.
// Sense-reversing (same pattern as ncclCoopWarpSpan::sync / ncclCeGlobalBlockBarrier):
// last arriver clears the count and flips sense; others spin on the sense word.
// Arrival is a relaxed counter; an agent-scope acquire fence on every thread after
// the wait covers the last arriver (CTA 0 can be last and is the one that gin.put()s).
// The two device words are allocated once at comm init; sense carries across graph replays.
__device__ __forceinline__ void ginIntraGpuCtaBarrier(uint32_t* bar, unsigned nCtas) {
  uint32_t* arrived = bar;
  uint32_t* sense = bar + 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t s = __scoped_atomic_load_n(sense, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    const uint32_t prev = __scoped_atomic_fetch_add(arrived, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
    if (prev + 1u == nCtas) {
      __scoped_atomic_store_n(arrived, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
      __scoped_atomic_store_n(sense, 1u - s, __ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
    } else {
      while (__scoped_atomic_load_n(sense, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) == s) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
#else
    const uint32_t s = *static_cast<volatile uint32_t*>(sense);
    const uint32_t prev = atomicAdd(arrived, 1u);
    if (prev + 1u == nCtas) {
      (void)atomicExch(arrived, 0u);
      *static_cast<volatile uint32_t*>(sense) = 1u - s;
    } else {
      while (*static_cast<volatile uint32_t*>(sense) == s) {
      }
    }
#endif
  }
#if NCCL_DEVICE_COMPILE
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
#endif
  __syncthreads();
}

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTwoShotKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                            ncclWindow_t recvWin, size_t recvOff, size_t countPerRank, int nRanks,
                                            uint32_t* intraGpuCtaBar) {
  constexpr int ginContext = 0;

  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);
  ncclCoopCta cta;
  ncclLsaBarrierSession<ncclCoopCta> bar{cta, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         static_cast<uint32_t>(blockIdx.x)};

  const size_t rankChunkStride = countPerRank * sizeof(T);
  const size_t globalElemOff = static_cast<size_t>(devComm.rank) * countPerRank;
  const size_t sliceSendByteOff = sendOff + globalElemOff * sizeof(T);
  const size_t sliceRecvByteOff = recvOff + static_cast<size_t>(devComm.rank) * rankChunkStride;
  T* reducedOut = reinterpret_cast<T*>(ncclGetLocalPointer(recvWin, sliceRecvByteOff));
  const size_t chunkBytes = countPerRank * sizeof(T);

  const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int nthreads = static_cast<int>(blockDim.x * gridDim.x);
  constexpr auto countPerThread = sizeof(uint4) / sizeof(T);

  bar.sync(cta, cuda::memory_order_acquire);

  // --- Shot 1: multi-CTA LSA reduce-scatter → local recv column ---
  const size_t idxStart = static_cast<size_t>(tid) * countPerThread;
  const size_t idxEnd = countPerRank;
  const size_t idxStride = static_cast<size_t>(nthreads) * countPerThread;

  for (size_t idx = idxStart; idx < idxEnd; idx += idxStride) {
    uint4 sum{0, 0, 0, 0};
    uint4 srcVals[2];
    *reinterpret_cast<uint4*>(&srcVals[0]) = *reinterpret_cast<const uint4*>(
      reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, 0)) + idx);
#pragma unroll 8
    for (int peer = 0; peer < nRanks - 1; ++peer) {
      *reinterpret_cast<uint4*>(&srcVals[(peer + 1) & 1]) = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const T*>(ncclGetLsaPointer(sendWin, sliceSendByteOff, peer + 1)) + idx);
      sum = vecElementAdd<T>(sum, srcVals[peer & 1]);
    }
    sum = vecElementAdd<T>(sum, srcVals[(nRanks - 1) & 1]);
    *reinterpret_cast<uint4*>(reducedOut + idx) = sum;
  }

  bar.sync(cta, cuda::memory_order_release);
  // LSA/world barriers only pair CTA i across ranks. Wait for every local CTA's
  // RS stores before CTA 0 GIN-puts the full reduced column.
  ginIntraGpuCtaBarrier(intraGpuCtaBar, static_cast<unsigned>(gridDim.x));

  // --- Shot 2: CTA 0 GIN all-gather of the full reduced column ---
  // The wait target is relative to a baseline read on the device at launch time, so it stays
  // correct across graph replays: nothing about the expected signal value is fixed on the host.
  if (blockIdx.x == 0) {
    const unsigned int signalIndex = static_cast<unsigned int>(blockIdx.x);
    const uint64_t signalValue = gin.readSignal(signalIndex);
    ncclBarrierSession<ncclCoopCta> ginBar{cta, ncclTeamTagWorld(), gin, blockIdx.x};
    ginBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);

    for (int dst = static_cast<int>(threadIdx.x); dst < nRanks; dst += static_cast<int>(blockDim.x)) {
      gin.put(world, dst, recvWin, sliceRecvByteOff, recvWin, sliceRecvByteOff, chunkBytes,
              ncclGin_SignalInc{signalIndex});
    }
    gin.waitSignal(cta, signalIndex, signalValue + static_cast<uint64_t>(nRanks));
  
    gin.flush(cta);

    ginBar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::None);
  }

}

// GIN-put one pipeline chunk of send[dst] into dest dst's staging slot for every dst != rank.
__device__ __forceinline__ void ginScatterPutRemoteChunk(ncclGin& gin, ncclTeam world, int rank, int nRanks,
                                                         ncclWindow_t scratchWin, size_t destSlotOff,
                                                         ncclWindow_t sendWin, size_t sendOff, size_t perRankBytes,
                                                         size_t sendChunkOff, size_t chunkBytes, unsigned signalIndex) {
  for (int dst = static_cast<int>(threadIdx.x); dst < nRanks; dst += static_cast<int>(blockDim.x)) {
    if (dst == rank) {
      continue;
    }
    gin.put(world, dst, scratchWin, destSlotOff, sendWin,
            sendOff + static_cast<size_t>(dst) * perRankBytes + sendChunkOff, chunkBytes,
            ncclGin_WeakSignalInc{signalIndex});
  }
}

__device__ __forceinline__ size_t ginScatterPutDestOff(size_t scratchOff, int rank, int nRanks, size_t perRankBytes,
                                                       size_t slotBytes, int slot, size_t chunkOff, bool staged) {
  if (!staged) {
    return scratchOff + static_cast<size_t>(rank) * perRankBytes + chunkOff;
  }
  return scratchOff + static_cast<size_t>(slot) * static_cast<size_t>(nRanks) * slotBytes +
         static_cast<size_t>(rank) * slotBytes;
}

__device__ __forceinline__ const char* ginScatterIncomingBase(const char* scratchBase, int nRanks, size_t slotBytes,
                                                             int slot, bool staged) {
  if (!staged) {
    return scratchBase;
  }
  return scratchBase + static_cast<size_t>(slot) * static_cast<size_t>(nRanks) * slotBytes;
}

// CE ncclCeLocalReduceKernelVec: 16B vectors, GpuUnroll=4, rank loop 0..nRanks-1
// into recvbuff + rank*shard. Self is not written to scratch (GIN-put skips it),
// so rank == myRank loads from send instead of the incoming slot.
template <typename T>
__device__ __forceinline__ const char* ginScatterReduceSrc(int peer, int rank, const char* selfSend,
                                                          const char* incomingBase, size_t incomingStride, size_t elem,
                                                          size_t elemStart, bool staged) {
  if (peer == rank) {
    return selfSend + elem * sizeof(T);
  }
  const size_t srcOff = staged ? (elem - elemStart) * sizeof(T) : elem * sizeof(T);
  return incomingBase + static_cast<size_t>(peer) * incomingStride + srcOff;
}

template <typename T>
__device__ __forceinline__ void ginScatterReduceChunk(const char* selfSend, const char* incomingBase, char* reducedOut,
                                                     int rank, int nRanks, size_t incomingStride, size_t elemStart,
                                                     size_t elemEnd, bool staged) {
  constexpr int W = static_cast<int>(sizeof(uint4) / sizeof(T));
  constexpr int U = 4;
  const size_t nVec = (elemEnd - elemStart) / static_cast<size_t>(W);
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  size_t vi = tid;
  for (; vi + static_cast<size_t>(U - 1) * stride < nVec; vi += stride * static_cast<size_t>(U)) {
    uint4 acc[U];
    size_t elem[U];
#pragma unroll
    for (int i = 0; i < U; ++i) {
      elem[i] = elemStart + (vi + static_cast<size_t>(i) * stride) * static_cast<size_t>(W);
      acc[i] = lsaLoadVec(ginScatterReduceSrc<T>(0, rank, selfSend, incomingBase, incomingStride, elem[i], elemStart,
                                                staged));
    }
#pragma clang loop unroll(disable) vectorize(disable)
    for (int r = 1; r < nRanks; ++r) {
      uint4 tmp[U];
#pragma unroll
      for (int i = 0; i < U; ++i) {
        tmp[i] = lsaLoadVec(ginScatterReduceSrc<T>(r, rank, selfSend, incomingBase, incomingStride, elem[i], elemStart,
                                                  staged));
      }
#pragma unroll
      for (int i = 0; i < U; ++i) {
        acc[i] = vecElementAdd<T>(acc[i], tmp[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < U; ++i) {
      lsaStoreVec(reducedOut + elem[i] * sizeof(T), acc[i]);
    }
  }

  const size_t vectorTailStart = nVec - (nVec % (stride * static_cast<size_t>(U)));
  if (nVec > 0 && nVec % (stride * static_cast<size_t>(U)) != 0) {
    for (size_t vIdx = vectorTailStart + tid; vIdx < nVec; vIdx += stride) {
      const size_t e = elemStart + vIdx * static_cast<size_t>(W);
      uint4 acc =
        lsaLoadVec(ginScatterReduceSrc<T>(0, rank, selfSend, incomingBase, incomingStride, e, elemStart, staged));
#pragma clang loop unroll(disable) vectorize(disable)
      for (int r = 1; r < nRanks; ++r) {
        acc = vecElementAdd<T>(
          acc, lsaLoadVec(ginScatterReduceSrc<T>(r, rank, selfSend, incomingBase, incomingStride, e, elemStart,
                                                staged)));
      }
      lsaStoreVec(reducedOut + e * sizeof(T), acc);
    }
  }
}

__device__ __forceinline__ void ginScatterIssuePut(ncclGin& gin, ncclTeam world, int rank, int nRanks,
                                                  ncclWindow_t scratchWin, size_t scratchOff, ncclWindow_t sendWin,
                                                  size_t sendOff, size_t perRankBytes, size_t slotBytes,
                                                  size_t uniformBytes, size_t lastBytes, int nChunks, int nPhases,
                                                  bool staged, int chunk, unsigned signalIndex) {
  const size_t chunkOff = ginAllReduceGinScatterChunkOff(chunk, uniformBytes);
  const size_t chunkBytes = ginAllReduceGinScatterChunkSize(chunk, nChunks, uniformBytes, lastBytes);
  const int slot = staged ? ((nPhases <= 1) ? 0 : (chunk % nPhases)) : 0;
  const size_t destOff =
    ginScatterPutDestOff(scratchOff, rank, nRanks, perRankBytes, slotBytes, slot, staged ? 0 : chunkOff, staged);
  ginScatterPutRemoteChunk(gin, world, rank, nRanks, scratchWin, destOff, sendWin, sendOff, perRankBytes, chunkOff,
                           chunkBytes, signalIndex);
}

// Thread 0 polls; callers that need the rest of the CTA to observe completion
// follow with ginIntraGpuCtaBarrier or __syncthreads.
__device__ __forceinline__ void ginScatterWaitThread0(ncclGin& gin, unsigned signalIndex, uint64_t least) {
  if (threadIdx.x == 0) {
    gin.waitSignal(ncclCoopThread(), signalIndex, least);
  }
}

// One-way doorbell: this rank finished reading a reused scratch slot. Cheaper than
// a world barrier (CE uses per-slot signal==0 the same way).
__device__ __forceinline__ void ginScatterNotifySlotFree(ncclGin& gin, ncclTeam world, int rank, int nRanks,
                                                         unsigned doneSignal) {
  for (int dst = static_cast<int>(threadIdx.x); dst < nRanks; dst += static_cast<int>(blockDim.x)) {
    if (dst == rank) {
      continue;
    }
    gin.signal(world, dst, ncclGin_WeakSignalInc{doneSignal});
  }
}

// Two-shot AllReduce over GIN-SDMA. Grid is 56 CTAs x 512 threads.
//
// Scratch follows CE AllReduce (512 MiB, NCCL_CE_NUM_SLOTS ping-pong):
//   Fit:     [peer 0 .. nRanks) full incoming columns (reduced shard is recvbuff)
//   Staged:  [slot 0: nRanks slots][slot 1: nRanks slots], slot = step % nPhases
//            Default 512 MiB scratch stages ginScatter messages above 512 MiB.
//
// Shot 1: CTA 0 GIN-puts send[dst] into dest's staging slot (dst != myRank).
// Local reduce is CE: 16B vectors, unroll 4, rank 0..nRanks-1 into recv[myRank]
// (self from send). Below 512 MiB a step completes scatter before reduce. At
// 512 MiB+ put of chunk i+1 overlaps reduce of chunk i. CTA 0 GIN all-gathers.
//
// Sync: one world acquire so all ranks snapshot GIN signal baselines. Each pipeline
// step is waitSignal on fused WeakSignalInc (scatter/AG) or a one-way slot-free
// doorbell — not ginBar.sync or gin.flush. Intra-GPU CTA bars only join the 56 CTAs.
template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTwoShotGinScatterKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                                      ncclWindow_t recvWin, size_t recvOff, ncclWindow_t scratchWin,
                                                      size_t scratchOff, size_t countPerRank, int nRanks,
                                                      size_t scratchBytes, uint32_t* intraGpuCtaBar) {
  constexpr int ginContext = 0;
  constexpr unsigned scatterSignal = 0;
  constexpr unsigned agSignal = 1;
  constexpr unsigned doneSignal = 2;

  // gin/ginBar must outlive the first put; only CTA 0 snapshots signals and world-syncs.
  // Pipeline steps use waitSignal / WeakSignalInc (CE doorbells), not world barriers.
  ncclGin gin{devComm, ginContext};
  ncclTeam world = ncclTeamWorld(devComm);
  ncclCoopCta cta;
  ncclBarrierSession<ncclCoopCta> ginBar{cta, ncclTeamTagWorld(), gin, blockIdx.x};

  const size_t perRankBytes = countPerRank * sizeof(T);
  const size_t sliceRecvByteOff = recvOff + static_cast<size_t>(devComm.rank) * perRankBytes;
  char* scratchBase = reinterpret_cast<char*>(ncclGetLocalPointer(scratchWin, scratchOff));
  char* reducedOut = reinterpret_cast<char*>(ncclGetLocalPointer(recvWin, sliceRecvByteOff));
  const char* selfSend = reinterpret_cast<const char*>(ncclGetLocalPointer(sendWin, sendOff)) +
                         static_cast<size_t>(devComm.rank) * perRankBytes;

  int nChunks = 1;
  int nPhases = 1;
  size_t uniformBytes = perRankBytes;
  size_t lastBytes = perRankBytes;
  ginAllReduceGinScatterChunkPlan(perRankBytes, nRanks, scratchBytes, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  const bool staged = ginAllReduceGinScatterStaged(perRankBytes, nRanks, scratchBytes);
  const size_t slotBytes = staged ? ginAllReduceGinScatterFitSlot(scratchBytes, nRanks, nPhases) : perRankBytes;
  const size_t incomingStride = staged ? slotBytes : perRankBytes;
  const bool overlap = (nChunks > 1) && !(staged && nPhases <= 1);

  const uint64_t signalsPerChunk = static_cast<uint64_t>(nRanks - 1);
  uint64_t scatterValue = 0;
  uint64_t doneValue = 0;
  uint64_t agValue = 0;

  // One world acquire so every rank snapshots signal baselines before any put.
  // Later pipeline steps are waitSignal doorbells (CE), not ginBar.sync.
  if (blockIdx.x == 0) {
    scatterValue = gin.readSignal(scatterSignal);
    doneValue = gin.readSignal(doneSignal);
    agValue = gin.readSignal(agSignal);
    ginBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::None);
  }

  if (overlap) {
    if (blockIdx.x == 0) {
      ginScatterIssuePut(gin, world, devComm.rank, nRanks, scratchWin, scratchOff, sendWin, sendOff, perRankBytes,
                         slotBytes, uniformBytes, lastBytes, nChunks, nPhases, staged, 0, scatterSignal);
      ginScatterWaitThread0(gin, scatterSignal, scatterValue + signalsPerChunk);
    }
    ginIntraGpuCtaBarrier(intraGpuCtaBar, static_cast<unsigned>(gridDim.x));

    for (int c = 0; c < nChunks; ++c) {
      if (blockIdx.x == 0 && c + 1 < nChunks) {
        if (staged && c + 1 >= nPhases) {
          ginScatterWaitThread0(gin, doneSignal, doneValue + static_cast<uint64_t>(c + 2 - nPhases) * signalsPerChunk);
          __syncthreads();
        }
        ginScatterIssuePut(gin, world, devComm.rank, nRanks, scratchWin, scratchOff, sendWin, sendOff, perRankBytes,
                           slotBytes, uniformBytes, lastBytes, nChunks, nPhases, staged, c + 1, scatterSignal);
      }
      const size_t chunkOff = ginAllReduceGinScatterChunkOff(c, uniformBytes);
      const size_t chunkBytes = ginAllReduceGinScatterChunkSize(c, nChunks, uniformBytes, lastBytes);
      const size_t elemStart = chunkOff / sizeof(T);
      const size_t elemEnd = elemStart + chunkBytes / sizeof(T);
      const int slot = staged ? ((nPhases <= 1) ? 0 : (c % nPhases)) : 0;
      ginScatterReduceChunk<T>(selfSend, ginScatterIncomingBase(scratchBase, nRanks, slotBytes, slot, staged),
                               reducedOut, devComm.rank, nRanks, incomingStride, elemStart, elemEnd, staged);
      if (blockIdx.x == 0 && c + 1 < nChunks) {
        ginScatterWaitThread0(gin, scatterSignal, scatterValue + static_cast<uint64_t>(c + 2) * signalsPerChunk);
      }
      ginIntraGpuCtaBarrier(intraGpuCtaBar, static_cast<unsigned>(gridDim.x));
      if (blockIdx.x == 0 && staged && c + nPhases < nChunks) {
        ginScatterNotifySlotFree(gin, world, devComm.rank, nRanks, doneSignal);
        __syncthreads();
      }
    }
  } else {
    for (int c = 0; c < nChunks; ++c) {
      if (blockIdx.x == 0) {
        if (c > 0 && staged) {
          ginScatterWaitThread0(gin, doneSignal, doneValue + static_cast<uint64_t>(c) * signalsPerChunk);
          __syncthreads();
        }
        ginScatterIssuePut(gin, world, devComm.rank, nRanks, scratchWin, scratchOff, sendWin, sendOff, perRankBytes,
                           slotBytes, uniformBytes, lastBytes, nChunks, nPhases, staged, c, scatterSignal);
        ginScatterWaitThread0(gin, scatterSignal, scatterValue + static_cast<uint64_t>(c + 1) * signalsPerChunk);
      }
      ginIntraGpuCtaBarrier(intraGpuCtaBar, static_cast<unsigned>(gridDim.x));
      const size_t chunkOff = ginAllReduceGinScatterChunkOff(c, uniformBytes);
      const size_t chunkBytes = ginAllReduceGinScatterChunkSize(c, nChunks, uniformBytes, lastBytes);
      const size_t elemStart = chunkOff / sizeof(T);
      const size_t elemEnd = elemStart + chunkBytes / sizeof(T);
      const int slot = staged ? ((nPhases <= 1) ? 0 : (c % nPhases)) : 0;
      ginScatterReduceChunk<T>(selfSend, ginScatterIncomingBase(scratchBase, nRanks, slotBytes, slot, staged),
                               reducedOut, devComm.rank, nRanks, incomingStride, elemStart, elemEnd, staged);
      ginIntraGpuCtaBarrier(intraGpuCtaBar, static_cast<unsigned>(gridDim.x));
      if (blockIdx.x == 0 && staged && c + 1 < nChunks) {
        ginScatterNotifySlotFree(gin, world, devComm.rank, nRanks, doneSignal);
        __syncthreads();
      }
    }
  }

  if (blockIdx.x == 0) {
    // Shot 2: waitSignal on fused AG puts is the completion doorbell; no world barrier.
    for (int dst = static_cast<int>(threadIdx.x); dst < nRanks; dst += static_cast<int>(blockDim.x)) {
      if (dst == devComm.rank) {
        continue;
      }
      gin.put(world, dst, recvWin, sliceRecvByteOff, recvWin, sliceRecvByteOff, perRankBytes,
              ncclGin_WeakSignalInc{agSignal});
    }
    ginScatterWaitThread0(gin, agSignal, agValue + static_cast<uint64_t>(nRanks - 1));
    __syncthreads();
  }
}

} // namespace gin::sdma
