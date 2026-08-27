/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * LSA AllReduce device kernels:
 *   allReduceLsaOneShotKernel — read all peers, sum, write all peers (small messages).
 *   lsaAllReduceTwoShotKernel — LSA reduce-scatter, barrier, LSA all-gather (remote reads only).
 *   lsaAllReduceTwoShotOverlappedKernel — same two shots pipelined, pushing the reduced column to
 *                                        peers so vector i is all-gathered while i+1 is reduced.
 *   ginAllReduceTwoShotKernel  — LSA reduce-scatter + multi-CTA GIN all-gather from recv.
 *
 * One-shot follows projects/rccl-tests/src/all_reduce.cu allReduceLsaKernel.
 ******************************************************************************/

#pragma once

#include "nccl_device.h"
#include "nccl_device/ptr.h"
#include "algorithms/dda/device/CollCommon.h"

namespace gin::sdma{
using dda::common::vecElementAdd;

template <typename T>
__device__ __forceinline__ T allReduceLsaSumAdd(T a, T b) {
  if constexpr (std::is_same<T, bf16>::value) {
    return __hadd(a, b);
  } else {
    return a + b;
  }
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

template <typename T>
#if defined(USE_ROCM)
__launch_bounds__(512)
#endif
  __global__ void ginAllReduceTwoShotKernel(struct ncclDevComm devComm, ncclWindow_t sendWin, size_t sendOff,
                                            ncclWindow_t recvWin, size_t recvOff, size_t countPerRank, int nRanks) {
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

  // --- Shot 2: multi-CTA GIN all-gather (rccl-tests GinAlltoAllKernel) ---
  // The wait target is relative to a baseline read on the device at launch time, so it stays
  // correct across graph replays: nothing about the expected signal value is fixed on the host.
  const unsigned int signalIndex = static_cast<unsigned int>(blockIdx.x);
  const uint64_t signalValue = gin.readSignal(signalIndex);
  ncclBarrierSession<ncclCoopCta> ginBar{cta, ncclTeamTagWorld(), gin, blockIdx.x};
  ginBar.sync(cta, cuda::memory_order_acquire, ncclGinFenceLevel::Relaxed);

  for (int dst = tid; dst < nRanks; dst += nthreads) {
    gin.put(world, dst, recvWin, sliceRecvByteOff, recvWin, sliceRecvByteOff, chunkBytes,
            ncclGin_SignalInc{signalIndex});
  }

  const int receivingCta = (devComm.rank % nthreads) / blockDim.x;
  if (blockIdx.x == receivingCta) {
    gin.waitSignal(cta, signalIndex, signalValue + static_cast<uint64_t>(nRanks));
  }
  gin.flush(cta);

  ginBar.sync(cta, cuda::memory_order_release, ncclGinFenceLevel::Relaxed);

}

} // namespace gin::sdma
