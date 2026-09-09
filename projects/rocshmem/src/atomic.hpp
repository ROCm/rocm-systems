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

#ifndef LIBRARY_SRC_ATOMIC_HPP
#define LIBRARY_SRC_ATOMIC_HPP

#include <hip/hip_runtime.h>

namespace rocshmem {
namespace detail {
namespace atomic {

typedef enum rocshmem_memory_scope {
  memory_scope_single    = __MEMORY_SCOPE_SINGLE,
  memory_scope_wavefront = __MEMORY_SCOPE_WVFRNT,
  memory_scope_workgroup = __MEMORY_SCOPE_WRKGRP,
  memory_scope_device    = __MEMORY_SCOPE_DEVICE,
  memory_scope_system    = __MEMORY_SCOPE_SYSTEM,
} rocshmem_memory_scope;

typedef enum rocshmem_memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} rocshmem_memory_order;

template <typename T, rocshmem_memory_scope s>
__host__ __device__
T load(const T* address, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_load_n(address, order, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
void store(T* address, const T value, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_store_n(address, value, order, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
T exchange(T* obj, T desired, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_exchange_n(obj, desired, order, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
bool compare_exchange_weak(T* obj, T& expected, T desired,
                           rocshmem_memory_order success = memory_order_seq_cst,
                           rocshmem_memory_order failure = memory_order_seq_cst) {
  return __scoped_atomic_compare_exchange_n(obj, &expected, desired, true, success, failure, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
bool compare_exchange_strong(T* obj, T& expected, T desired,
                             rocshmem_memory_order success = memory_order_seq_cst,
                             rocshmem_memory_order failure = memory_order_seq_cst) {
  return __scoped_atomic_compare_exchange_n(obj, &expected, desired, false, success, failure, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_add(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_add(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_sub(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_sub(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_and(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_and(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_or(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_or(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_xor(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_xor(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_max(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_max(obj, arg, order, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_min(T* obj, U arg, rocshmem_memory_order order = memory_order_seq_cst) {
  return __scoped_atomic_fetch_min(obj, arg, order, s);
}

#define ROCSHMEM_DISPATCH_FENCE_ORDER(SCOPE_STR)         \
  if constexpr (order == memory_order_acquire)           \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, SCOPE_STR); \
  else if constexpr (order == memory_order_release)      \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, SCOPE_STR); \
  else if constexpr (order == memory_order_acq_rel)      \
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, SCOPE_STR); \
  else                                                   \
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, SCOPE_STR)

template <rocshmem_memory_scope scope = memory_scope_system,
          rocshmem_memory_order order = memory_order_seq_cst>
__device__ __forceinline__ void threadfence() {
  if constexpr (scope == memory_scope_single ||
                scope == memory_scope_wavefront ||
                scope == memory_scope_workgroup) {
    ROCSHMEM_DISPATCH_FENCE_ORDER("workgroup");
  } else if constexpr (scope == memory_scope_device) {
    ROCSHMEM_DISPATCH_FENCE_ORDER("agent");
  } else {
    ROCSHMEM_DISPATCH_FENCE_ORDER("");  // system scope
  }
}

#undef ROCSHMEM_DISPATCH_FENCE_ORDER

} // namespace atomic
} // namespace detail
} // namespace rocshmem

#endif  // LIBRARY_SRC_ATOMIC_HPP_
