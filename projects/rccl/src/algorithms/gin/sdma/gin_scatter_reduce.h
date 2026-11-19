/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * CE-style local reduce for GinScatter. Self is read from the send slice;
 * every other rank is read from that rank's incoming scratch column.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#pragma once

#include "algorithms/dda/device/CollCommon.h"

namespace gin::sdma {

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
  // Complete U-wide spans only. The tail used to restart at the last partial
  // span, so threads whose unrolled iteration already covered that span reduced
  // those vectors twice.
  const size_t span = stride * static_cast<size_t>(U);
  const size_t nComplete = span == 0 ? 0 : (nVec / span) * span;

  for (size_t vi = tid; vi + static_cast<size_t>(U - 1) * stride < nComplete; vi += span) {
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
        acc[i] = dda::common::vecElementAdd<T>(acc[i], tmp[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < U; ++i) {
      lsaStoreVec(reducedOut + elem[i] * sizeof(T), acc[i]);
    }
  }

  for (size_t vIdx = nComplete + tid; vIdx < nVec; vIdx += stride) {
    const size_t e = elemStart + vIdx * static_cast<size_t>(W);
    uint4 acc =
      lsaLoadVec(ginScatterReduceSrc<T>(0, rank, selfSend, incomingBase, incomingStride, e, elemStart, staged));
#pragma clang loop unroll(disable) vectorize(disable)
    for (int r = 1; r < nRanks; ++r) {
      acc = dda::common::vecElementAdd<T>(
        acc, lsaLoadVec(ginScatterReduceSrc<T>(r, rank, selfSend, incomingBase, incomingStride, e, elemStart, staged)));
    }
    lsaStoreVec(reducedOut + e * sizeof(T), acc);
  }
}

} // namespace gin::sdma
