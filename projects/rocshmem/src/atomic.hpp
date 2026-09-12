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
#include <type_traits>

namespace rocshmem {
namespace detail {
namespace atomic {

enum class memory_scope : int {
  single    = __MEMORY_SCOPE_SINGLE,
  wavefront = __MEMORY_SCOPE_WVFRNT,
  workgroup = __MEMORY_SCOPE_WRKGRP,
  device    = __MEMORY_SCOPE_DEVICE,
  system    = __MEMORY_SCOPE_SYSTEM,
#if defined(__MEMORY_SCOPE_CLUSTR)
  cluster   = __MEMORY_SCOPE_CLUSTR,
#else
  cluster   = __MEMORY_SCOPE_DEVICE,
#endif
};

enum class memory_order : int {
  relaxed = __ATOMIC_RELAXED,
  consume = __ATOMIC_CONSUME,
  acquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST,
};

// Suppresses template argument deduction for a function parameter so that the
// value type is fixed by the object pointer alone (C++17-safe stand-in for
// std::type_identity, which is C++20).
template <typename T>
struct type_identity { using type = T; };

template <typename T>
using type_identity_t = typename type_identity<T>::type;

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T load(const T* address) {
  return __scoped_atomic_load_n(address, static_cast<int>(order),
                                static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
void store(T* address, type_identity_t<T> value) {
  return __scoped_atomic_store_n(address, value, static_cast<int>(order),
                                 static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T exchange(T* obj, type_identity_t<T> desired) {
  return __scoped_atomic_exchange_n(obj, desired, static_cast<int>(order),
                                    static_cast<int>(scope));
}

template <memory_scope scope   = memory_scope::system,
          memory_order success = memory_order::seq_cst,
          memory_order failure = memory_order::seq_cst, typename T>
__host__ __device__
bool compare_exchange_weak(T* obj, T& expected, type_identity_t<T> desired) {
  return __scoped_atomic_compare_exchange_n(obj, &expected, desired, true,
                                            static_cast<int>(success),
                                            static_cast<int>(failure),
                                            static_cast<int>(scope));
}

template <memory_scope scope   = memory_scope::system,
          memory_order success = memory_order::seq_cst,
          memory_order failure = memory_order::seq_cst, typename T>
__host__ __device__
bool compare_exchange_strong(T* obj, T& expected, type_identity_t<T> desired) {
  return __scoped_atomic_compare_exchange_n(obj, &expected, desired, false,
                                            static_cast<int>(success),
                                            static_cast<int>(failure),
                                            static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_add(T* obj, type_identity_t<T> arg) {
  // __scoped_atomic_fetch_add does not scale the addend by sizeof(T) for
  // pointer types (unlike __hip_atomic_fetch_add), so pointer atomics would
  // silently change behavior. Reject them until a scaled path is needed.
  static_assert(!std::is_pointer_v<T>,
                "atomic::fetch_add does not support pointer element types");
  return __scoped_atomic_fetch_add(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_sub(T* obj, type_identity_t<T> arg) {
  static_assert(!std::is_pointer_v<T>,
                "atomic::fetch_sub does not support pointer element types");
  return __scoped_atomic_fetch_sub(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_and(T* obj, type_identity_t<T> arg) {
  return __scoped_atomic_fetch_and(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_or(T* obj, type_identity_t<T> arg) {
  return __scoped_atomic_fetch_or(obj, arg, static_cast<int>(order),
                                  static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_xor(T* obj, type_identity_t<T> arg) {
  return __scoped_atomic_fetch_xor(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_max(T* obj, type_identity_t<T> arg) {
  return __scoped_atomic_fetch_max(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst, typename T>
__host__ __device__
T fetch_min(T* obj, type_identity_t<T> arg) {
  return __scoped_atomic_fetch_min(obj, arg, static_cast<int>(order),
                                   static_cast<int>(scope));
}

#define ROCSHMEM_DISPATCH_FENCE_ORDER(SCOPE_STR)         \
  if constexpr (order == memory_order::acquire)          \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, SCOPE_STR); \
  else if constexpr (order == memory_order::release)     \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, SCOPE_STR); \
  else if constexpr (order == memory_order::acq_rel)     \
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, SCOPE_STR); \
  else                                                   \
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, SCOPE_STR)

template <memory_scope scope = memory_scope::system,
          memory_order order = memory_order::seq_cst>
__device__ __forceinline__ void threadfence() {
  if constexpr (scope == memory_scope::single ||
                scope == memory_scope::wavefront ||
                scope == memory_scope::workgroup) {
    ROCSHMEM_DISPATCH_FENCE_ORDER("workgroup");
  } else if constexpr (scope == memory_scope::device) {
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
