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
  memory_scope_thread = __HIP_MEMORY_SCOPE_SINGLETHREAD,
  memory_scope_wavefront = __HIP_MEMORY_SCOPE_WAVEFRONT,
  memory_scope_workgroup = __HIP_MEMORY_SCOPE_WORKGROUP,
  memory_scope_agent = __HIP_MEMORY_SCOPE_AGENT,
  memory_scope_system = __HIP_MEMORY_SCOPE_SYSTEM,
} rocshmem_memory_scope;

typedef enum rocshmem_memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_consume = __ATOMIC_CONSUME,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} rocshmem_memory_order;

struct rocshmem_memory_orders {
  rocshmem_memory_order load {memory_order_acquire};
  rocshmem_memory_order store {memory_order_release};
  rocshmem_memory_order atomic {memory_order_acq_rel};
  rocshmem_memory_order weak_cas_success {memory_order_acq_rel};
  rocshmem_memory_order weak_cas_failure {memory_order_acq_rel};
  rocshmem_memory_order strong_cas_success {memory_order_acq_rel};
  rocshmem_memory_order strong_cas_failure {memory_order_acq_rel};
};

template <typename T, rocshmem_memory_scope s>
__host__ __device__
T load(const T* address, rocshmem_memory_orders o) {
  return __hip_atomic_load(address, o.load, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
void store(T* address, const T value, rocshmem_memory_orders o) {
  return __hip_atomic_store(address, value, o.store, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
bool compare_exchange_weak(T& expected, T desired, rocshmem_memory_orders o) {
  return __hip_atomic_compare_exchange_weak(expected, desired, o.weak_cas_success, o.weak_cas_failure, s);
}

template <typename T, rocshmem_memory_scope s>
__host__ __device__
bool compare_exchange_strong(T& expected, T desired, rocshmem_memory_orders o) {
  return __hip_atomic_compare_exchange_strong(expected, desired, o.strong_cas_success, o.strong_cas_failure, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_add(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_add(obj, arg, o.atomic, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_sub(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_sub(obj, arg, o.atomic, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_and(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_and(obj, arg, o.atomic, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_or(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_or(obj, arg, o, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_xor(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_xor(obj, arg, o.atomic, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_max(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_max(obj, arg, o.atomic, s);
}

template <class T, class U, rocshmem_memory_scope s>
__host__ __device__
T fetch_min(T* obj, U arg, rocshmem_memory_orders o) {
  return __hip_atomic_fetch_min(obj, arg, o.atomic, s);
}

template <rocshmem_memory_scope s>
__device__
void threadfence() {
  if constexpr (s == memory_scope_system) {
    __threadfence_system();
  } else if constexpr (s == memory_scope_agent) {
    __threadfence();
  } else if constexpr (s == memory_scope_workgroup) {
    __threadfence_block();
  }
}

} // namespace atomic
} // namespace detail
} // namespace rocshmem

#endif  // LIBRARY_SRC_ATOMIC_HPP_
