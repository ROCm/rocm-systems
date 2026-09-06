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

#ifndef LIBRARY_SRC_UTIL_HPP_
#define LIBRARY_SRC_UTIL_HPP_

#include <hip/hip_runtime.h>

#include "bit.hpp"

namespace rocshmem {

/* Helper Macros for handling dynamic libraries */
#define PPCAT_NX(prefix, func_name) prefix##func_name
#define PPCAT(prefix, func_name) PPCAT_NX(prefix, func_name)

#define STRINGIFY_NX(name) #name
#define STRINGIFY(name) STRINGIFY_NX(name)

#define DLSYM_OPT_HELPER(func_struct, prefix, handle, func_name)                            \
do {                                                                                        \
  *(void **) (&func_struct.func_name) = dlsym(handle, STRINGIFY(PPCAT(prefix, func_name))); \
} while (0)

#define DLSYM_HELPER(func_struct, prefix, handle, func_name)                                \
do {                                                                                        \
  *(void **) (&func_struct.func_name) = dlsym(handle, STRINGIFY(PPCAT(prefix, func_name))); \
  if (!func_struct.func_name) {                                                             \
    LOG_WARN("Failed to find function %s",  STRINGIFY(PPCAT(prefix, func_name)));           \
    dlclose(handle);                                                                        \
    handle = nullptr;                                                                       \
    return ROCSHMEM_ERROR;                                                                  \
  }                                                                                         \
} while (0)

#define DLSYM_VAR_HELPER(func_struct, handle, var_name)                     \
do {                                                                        \
  *(void **) (&func_struct.var_name) = dlsym(handle, STRINGIFY(var_name));  \
  if (!func_struct.var_name) {                                              \
    LOG_WARN("Failed to find function %s",  STRINGIFY(var_name));           \
    dlclose(handle);                                                        \
    handle = nullptr;                                                       \
    return ROCSHMEM_ERROR;                                                  \
  }                                                                         \
} while (0)



/* Device-side internal functions */
[[maybe_unused]] __device__ __forceinline__ uint64_t get_active_lane_mask() {
  return __ballot(true);
}

[[maybe_unused]] __device__ __forceinline__ int get_active_lane_count(uint64_t active_lane_mask) {
  return popcount(active_lane_mask);
}

[[maybe_unused]] __device__ __forceinline__ int get_active_lane_count() {
  return get_active_lane_count(get_active_lane_mask());
}

[[maybe_unused]] __device__ __forceinline__ int get_active_lane_num(uint64_t active_lane_mask) {
  return popcount(active_lane_mask & (__lanemask_eq() - 1));
}

[[maybe_unused]] __device__ __forceinline__ int get_active_lane_num() {
  return get_active_lane_num(get_active_lane_mask());
}

[[maybe_unused]] __device__ __forceinline__ int get_first_active_lane_id(uint64_t active_lane_mask) {
  return countr_zero(active_lane_mask);
}

[[maybe_unused]] __device__ __forceinline__ int get_first_active_lane_id() {
  return get_first_active_lane_id(get_active_lane_mask());
}

[[maybe_unused]] __device__ __forceinline__ bool is_first_active_lane(uint64_t active_lane_mask) {
  return get_active_lane_num(active_lane_mask) == 0;
}

[[maybe_unused]] __device__ __forceinline__ bool is_first_active_lane() {
  return is_first_active_lane(get_active_lane_mask());
}

[[maybe_unused]] __device__ __forceinline__ int get_last_active_lane_id(uint64_t active_lane_mask) {
  return bit_log2(active_lane_mask);
}

[[maybe_unused]] __device__ __forceinline__ int get_last_active_lane_id() {
  return get_last_active_lane_id(get_active_lane_mask());
}

[[maybe_unused]] __device__ __forceinline__ bool is_last_active_lane(uint64_t active_lane_mask) {
  return get_active_lane_num(active_lane_mask) == get_active_lane_count(active_lane_mask) - 1;
}

__device__ __forceinline__ bool is_last_active_lane() {
  return is_last_active_lane(get_active_lane_mask());
}

#define SPIN_LOCK_INVALID  0xdead
#define SPIN_LOCK_UNLOCKED 0x1234
#define SPIN_LOCK_LOCKED   0xabcd

/*
 * Each thread in wave tries to acquire a different lock.
 */
[[maybe_unused]] __device__ __forceinline__ bool spin_lock_try_acquire_unique(uint32_t *lock) {
  uint32_t lock_val = SPIN_LOCK_UNLOCKED;

  __hip_atomic_compare_exchange_strong(lock, &lock_val, SPIN_LOCK_LOCKED,
                                       __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                       __HIP_MEMORY_SCOPE_AGENT);

  return lock_val == SPIN_LOCK_UNLOCKED;
}

/*
 * Each thread in wave acquires a different lock.
 * (deadlock if locks are not different)
 */
[[maybe_unused]] __device__ __forceinline__ void spin_lock_acquire_unique(uint32_t *lock) {
  while (!spin_lock_try_acquire_unique(lock)) {
    // spin
  }
}

/*
 * Each thread in wave releases a different lock.
 */
[[maybe_unused]] __device__ __forceinline__ void spin_lock_release_unique(uint32_t *lock) {
  __hip_atomic_store(lock, SPIN_LOCK_UNLOCKED, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_AGENT);
}

/*
 * Threads in activemask together try to acquire the same lock.
 */
[[maybe_unused]] __device__ __forceinline__ bool spin_lock_try_acquire_shared(uint32_t *lock, uint64_t activemask) {
  uint32_t lock_val = SPIN_LOCK_INVALID;

  if (is_first_active_lane(activemask)) {
    lock_val = SPIN_LOCK_UNLOCKED;
    __hip_atomic_compare_exchange_strong(lock, &lock_val, SPIN_LOCK_LOCKED,
                                         __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_AGENT);
  }
  lock_val = __shfl(lock_val, get_first_active_lane_id(activemask));

  return lock_val == SPIN_LOCK_UNLOCKED;
}

/*
 * Threads in activemask together acquire the same lock.
 */
[[maybe_unused]] __device__ __forceinline__ void spin_lock_acquire_shared(uint32_t *lock, uint64_t activemask) {
  while (!spin_lock_try_acquire_shared(lock, activemask)) {
    // spin
  }
}

/*
 * Threads in activemask together release the same lock.
 */
[[maybe_unused]] __device__ __forceinline__ void spin_lock_release_shared(uint32_t *lock, uint64_t activemask) {
  if (is_first_active_lane(activemask)) {
    __hip_atomic_store(lock, SPIN_LOCK_UNLOCKED, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
}

/* Is ptr_b in range [ptr_a, ptr_a + len_a) */
[[maybe_unused]]
__host__ __device__ static bool
is_ptr_in_range(uintptr_t ptr_a, size_t len_a, uintptr_t ptr_b) {

  if ((len_a == 0) || (ptr_b < ptr_a)) {
    return false;
  }

  return static_cast<size_t>(ptr_b - ptr_a) < len_a;
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_UTIL_HPP_
