/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_ROCSHMEM_CALC_HPP_
#define LIBRARY_SRC_ROCSHMEM_CALC_HPP_

namespace rocshmem {

// clang-format off
NOWARN(-Wunused-parameter,
template <ROCSHMEM_OP Op>
struct OpWrap {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    static_assert(true, "Unimplemented ipc collective.");
  }
};
)
// clang-format on

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <>
struct OpWrap<ROCSHMEM_SUM> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] += src[i];
  }
};

template <>
struct OpWrap<ROCSHMEM_MAX> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] = max(dst[i], src[i]);
  }
};

template <>
struct OpWrap<ROCSHMEM_MIN> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] = min(dst[i], src[i]);
  }
};

template <>
struct OpWrap<ROCSHMEM_PROD> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] *= src[i];
  }
};

template <>
struct OpWrap<ROCSHMEM_AND> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] &= src[i];
  }
};

template <>
struct OpWrap<ROCSHMEM_OR> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] |= src[i];
  }
};

template <>
struct OpWrap<ROCSHMEM_XOR> {
  template <typename T>
  __device__ static void Calc(T *src, T *dst, int i) {
    dst[i] ^= src[i];
  }
};


/******************************************************************************
 ************************** VECTORIZED REDUCE HELPERS *************************
 * Inspired by RCCL GIN AllReduce: process 16 bytes (uint4) per thread per
 * iteration to maximize memory throughput. Double-buffered loads overlap
 * the next peer's load with the current peer's reduction.
 *****************************************************************************/

// Number of T elements that fit in a uint4 (16 bytes).
template <typename T>
__device__ __forceinline__ constexpr int vecElems() {
  return static_cast<int>(sizeof(uint4) / sizeof(T));
}

// Apply Op element-wise across a uint4 interpreted as vecElems<T>() values.
template <typename T, ROCSHMEM_OP Op>
__device__ __forceinline__ uint4 vecReduce(uint4 acc, uint4 val) {
  T *a = reinterpret_cast<T *>(&acc);
  const T *v = reinterpret_cast<const T *>(&val);
  // OpWrap operates on arrays; use a temporary for the source.
  // Unroll for the small, fixed count (2, 4, 8, or 16 iterations).
#pragma unroll
  for (int k = 0; k < vecElems<T>(); k++) {
    T src_val = v[k];
    OpWrap<Op>::Calc(&src_val, &a[k], 0);
  }
  return acc;
}

// Vectorised ipc_compute_reduce: operates on uint4 chunks where possible,
// falls back to scalar for the tail.
template <typename T, ROCSHMEM_OP Op>
__device__ void ipc_compute_reduce_vec(T *src, T *dst, int size,
                                       int tid, int stride) {
  constexpr int VEC = vecElems<T>();
  const int vec_count = size / VEC;
  const int vec_tail  = vec_count * VEC;

  uint4 *src4 = reinterpret_cast<uint4 *>(src);
  uint4 *dst4 = reinterpret_cast<uint4 *>(dst);

  for (int i = tid; i < vec_count; i += stride) {
    dst4[i] = vecReduce<T, Op>(dst4[i], src4[i]);
  }
  // Scalar tail
  for (int i = vec_tail + tid; i < size; i += stride) {
    OpWrap<Op>::Calc(src, dst, i);
  }
}

// Reduce across nPeers peer buffers with double-buffered prefetching.
// peerPtrs[p] points to peer p's contribution (already resolved).
// Result is accumulated into dst. tid/stride are the thread's position.
template <typename T, ROCSHMEM_OP Op>
__device__ void reduce_peers_vec(T **peerPtrs, int nPeers,
                                 T *dst, int size,
                                 int tid, int stride) {
  constexpr int VEC = vecElems<T>();
  const int vec_count = size / VEC;
  const int vec_tail  = vec_count * VEC;

  uint4 *dst4 = reinterpret_cast<uint4 *>(dst);

  for (int i = tid; i < vec_count; i += stride) {
    uint4 acc = dst4[i];
    uint4 bufs[2];
    // Load first peer
    bufs[0] = reinterpret_cast<const uint4 *>(peerPtrs[0])[i];
#pragma unroll 8
    for (int p = 0; p < nPeers - 1; p++) {
      // Prefetch next peer while reducing current
      bufs[(p + 1) & 1] = reinterpret_cast<const uint4 *>(peerPtrs[p + 1])[i];
      acc = vecReduce<T, Op>(acc, bufs[p & 1]);
    }
    acc = vecReduce<T, Op>(acc, bufs[(nPeers - 1) & 1]);
    dst4[i] = acc;
  }

  // Scalar tail
  for (int i = vec_tail + tid; i < size; i += stride) {
    for (int p = 0; p < nPeers; p++) {
      OpWrap<Op>::Calc(peerPtrs[p], dst, i);
    }
  }
}

}
#endif // LIBRARY_SRC_ROCSHMEM_CALC_HPP_
