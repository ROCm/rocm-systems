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
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cassert>
#include <vector>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "assembly.hpp"
#include "atomic.hpp"
#include "bit.hpp"
#include "constants.hpp"
#include "log.hpp"

namespace rocshmem {

#define LIKELY(X)   __builtin_expect(X, 1)
#define UNLIKELY(X) __builtin_expect(X, 0)

/**
 * @name CHECK_NNULL
 * @brief Checks if value is NOT null. If it is null print errno and exit the program.
 *
 * @param[in] value    Value to check
 * @param[in] fn_str   String describing checked function
 *
 */
#define CHECK_NNULL(value, fn_str) do {                                        \
  if (UNLIKELY(nullptr == (value))) {                                          \
    LOG_ERROR_ABORT("%s: %s (%d)", fn_str, strerror(errno), errno);            \
  }                                                                            \
} while(0)

/**
 * @name CHECK_ZERO
 * @brief Checks if value is zero. If it is not zero print errno and exit the program.
 *
 * @param[in] value    Value to check
 * @param[in] fn_str   String describing checked function
 *
 */
#define CHECK_ZERO(value, fn_str) do {                                         \
  if (UNLIKELY(0 != (value))) {                                                \
    LOG_ERROR_ABORT("%s: %s (%d)", fn_str, strerror(errno), errno);            \
  }                                                                            \
} while(0)

/**
 * @name CHECK_HIP
 * @brief Checks if HIP command succeeded. If it is not success then it exits the program.
 *
 * @param[in] instr    HIP function to run and check
 *
 */
#define CHECK_HIP(instr) do {                                                  \
  hipError_t error = (instr);                                                  \
  if (error != hipSuccess) {                                                   \
    LOG_ERROR_ABORT(#instr ": %s (%d)", hipGetErrorString(error), error);       \
  }                                                                            \
} while(0)

/**
 * @name CHECK_HSA
 * @brief Checks if HSA command succeeded. If it is not success then it exits the program.
 *
 * @param[in] cmd HSA function to run and check
 *
 */
#define CHECK_HSA(cmd) do {                                                    \
  hsa_status_t error = cmd;                                                    \
  if (error != HSA_STATUS_SUCCESS) {                                           \
    LOG_ERROR_EXIT(#cmd ": %d", error);                                        \
  }                                                                            \
} while (0)

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
  if (!func_struct.var_name) {                                             \
    LOG_WARN("Failed to find function %s",  STRINGIFY(var_name));          \
    dlclose(handle);                                                        \
    handle = nullptr;                                                       \
    return ROCSHMEM_ERROR;                                                  \
  }                                                                         \
} while (0)

extern const int gpu_clock_freq_mhz;


typedef struct device_prop {
  int warpSize;
  int maxThreadsPerBlock;
  char gcnArchName[256];
} device_prop_t;

extern std::vector<device_prop_t> device_properties;

[[maybe_unused]] static int get_threads_per_block(int device_id) {
  assert(static_cast<int>(device_properties.size()) > device_id);
  return device_properties[device_id].maxThreadsPerBlock;
}

[[maybe_unused]] static int get_wf_size(int device_id) {
  assert(static_cast<int>(device_properties.size()) > device_id);
  return device_properties[device_id].warpSize;
}

[[maybe_unused]] static const char* get_arch_name(int device_id) {
  assert(static_cast<int>(device_properties.size()) > device_id);
  return device_properties[device_id].gcnArchName;
}

/* Device-side internal functions */
[[maybe_unused]] __device__ __forceinline__ uint32_t lowerID() {
  return __ffsll(__ballot(1)) - 1;
}

[[maybe_unused]] __device__ __forceinline__ int wave_SZ() { return __popcll(__ballot(1)); }

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
[[maybe_unused]] __device__ __forceinline__ bool is_thread_zero_in_block() {
  return hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0;
}

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
[[maybe_unused]] __device__ __forceinline__ bool is_block_zero_in_grid() {
  return hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0;
}

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
[[maybe_unused]] __device__ __forceinline__ int get_flat_block_size() {
  return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
}

/*
 * Returns the number of threads in the caller's flattened grid.
 */
[[maybe_unused]] __device__ __forceinline__ int get_flat_grid_size() {
  return get_flat_block_size() * hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
[[maybe_unused]] __device__ __forceinline__ int get_flat_block_id() {
  return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
         hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

/*
 * Returns the number of blocks in the caller's flattened grid.
 */
[[maybe_unused]] __device__ __forceinline__ int get_grid_num_blocks() {
  return hipGridDim_x * hipGridDim_y * hipGridDim_z;
}

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
[[maybe_unused]] __device__ __forceinline__ int get_flat_grid_id() {
  return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
         hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

/*
 * Returns the flattened thread index of the calling thread within the grid.
 */
[[maybe_unused]] __device__ __forceinline__ int get_flat_id() {
  return get_flat_grid_id() * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z) + get_flat_block_id();
}

/*
 * Returns true if the caller's thread flad_id is 0 in its wave.
 */
[[maybe_unused]] __device__ __forceinline__ bool is_thread_zero_in_wave() {
  return (get_flat_block_id() % WF_SIZE) == 0;
}

/*
 * Returns true if the caller's thread flat_id is in the zero'th wave.
 */
[[maybe_unused]] __device__ __forceinline__ bool is_wave_zero_in_block() {
  return (get_flat_block_id() / WF_SIZE) == 0;
}

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

/**
 * Grid barrier implementation using a global counter.
 * All the work-groups must be co-resident on the GPU for this to work
 * correctly.
 */
[[maybe_unused]] __forceinline__ __device__ void grid_barrier(int* global_counter, int num_blocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(&global_counter[0], 1,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(global_counter,
                             __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) != num_blocks);
  }
  __syncthreads();
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


#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

enum class MemcpyKind { Put, Get, PutBlocking, GetBlocking };

constexpr bool is_put(MemcpyKind k) {
  return k == MemcpyKind::Put || k == MemcpyKind::PutBlocking;
}

constexpr bool is_blocking(MemcpyKind k) {
  return k == MemcpyKind::PutBlocking || k == MemcpyKind::GetBlocking;
}

template <int ChunkSize, CachePolicy LoadPolicy, CachePolicy StorePolicy, int Unroll>
__device__ __forceinline__ void copy_bulk(void* dst, void* src,
                                          int n_chunks, int tid, int stride) {
  using Acc = AsmAccess<ChunkSize, LoadPolicy, StorePolicy>;
  using T = typename Acc::type;

  const uint32_t buf_bytes = static_cast<uint32_t>(n_chunks * ChunkSize);
  int chunk_batch = stride * Unroll;
  int offset = 0;
  T regs[Unroll] = {};

  // Unrolled block copy: issue all Unroll loads first to fill the memory pipeline,
  // then drain with stores. Hardware RAW tracking ensures load→store ordering per reg.
  for (; offset + chunk_batch <= n_chunks; offset += chunk_batch) {
    #pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      regs[u] = Acc::load_buffer(src, buf_bytes,
          static_cast<uint32_t>((offset + tid + u * stride) * ChunkSize));
    }
    #pragma unroll
    for (int u = 0; u < Unroll; ++u) {
      Acc::store_buffer(dst, buf_bytes,
          static_cast<uint32_t>((offset + tid + u * stride) * ChunkSize), regs[u]);
    }
  }

  // Tail: remaining chunks that don't fill a full unrolled batch
  for (int i = offset + tid; i < n_chunks; i += stride) {
    T val = Acc::load(static_cast<uint8_t*>(src) + i * ChunkSize);
    if constexpr (LoadPolicy != CachePolicy::Standard) {
      wait_on_vmem_and_lds(0);
    }
    Acc::store(static_cast<uint8_t*>(dst) + i * ChunkSize, val);
  }
}

// ==============================================================================
// REMAINDER COPY (< 16 Bytes Fast Path)
// ==============================================================================
template <CachePolicy LP, CachePolicy SP>
__device__ __forceinline__ void copy_remainder(uint8_t* dst,
                                               uint8_t* src,
                                               int remainder) {
  if (remainder == 0) return;

  if (remainder & 1) {
    auto val = AsmAccess<1, LP, SP>::load(src);
    if constexpr (LP != CachePolicy::Standard) {
      wait_on_vmem_and_lds(0);
    }
    AsmAccess<1, LP, SP>::store(dst, val);
    if (remainder == 1) {
      return;
    }
    dst += 1;
    src += 1;
  }
  if (remainder & 2) {
    auto val = AsmAccess<2, LP, SP>::load(src);
    if constexpr (LP != CachePolicy::Standard) {
      wait_on_vmem_and_lds(0);
    }
    AsmAccess<2, LP, SP>::store(dst, val);
    if (remainder == 2) {
      return;
    }
    dst += 2;
    src += 2;
  }
  if (remainder & 4) {
    auto val = AsmAccess<4, LP, SP>::load(src);
    if constexpr (LP != CachePolicy::Standard) {
      wait_on_vmem_and_lds(0);
    }
    AsmAccess<4, LP, SP>::store(dst, val);
    if (remainder == 4) {
      return;
    }
    dst += 4;
    src += 4;
  }
  if (remainder & 8) {
    auto val = AsmAccess<8, LP, SP>::load(src);
    if constexpr (LP != CachePolicy::Standard) {
      wait_on_vmem_and_lds(0);
    }
    AsmAccess<8, LP, SP>::store(dst, val);
  }
}

// ==============================================================================
// LANE, WAVE, AND WG IMPLEMENTATIONS
// ==============================================================================

// Blocking variants additionally drain all in-flight VMEM ops before returning.
template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_lane(void* dst, void* src,
                                                             size_t size) {
  if (size == 0) return;

  constexpr int ChunkSize = 16;
  constexpr int Unroll    = 16;
  // Compile-time bypass policy: cache-bypass in the direction of the remote side.
  constexpr CachePolicy LP = is_put(Kind) ? CachePolicy::Standard    : CachePolicy::SystemScope;
  constexpr CachePolicy SP = is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard;

  int n_chunks  = static_cast<int>(size / ChunkSize);
  int remainder = static_cast<int>(size % ChunkSize);

  if (size >= 16 && get_flat_block_size() > 4) {
    // Many threads, large transfer: use cached Standard policy.
    // Fences are direction-specific to maintain system-scope coherence.
    if constexpr (!is_put(Kind)) {
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                  detail::atomic::memory_order_acquire>();
    }

    if (n_chunks > 0) {
      copy_bulk<ChunkSize, CachePolicy::Standard, CachePolicy::Standard, 8>(
          dst, src, n_chunks, 0, 1);
    }
    copy_remainder<CachePolicy::Standard, CachePolicy::Standard>(
        static_cast<uint8_t*>(dst) + n_chunks * ChunkSize,
        static_cast<uint8_t*>(src) + n_chunks * ChunkSize, remainder);

    if constexpr (is_put(Kind)) {
      detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                  detail::atomic::memory_order_release>();
    }
  } else {
    // Small transfer or single-lane: cache-bypass policy provides direct
    // remote visibility without explicit fences.
    if (n_chunks > 0) {
      copy_bulk<ChunkSize, LP, SP, Unroll>(dst, src, n_chunks, 0, 1);
    }
    copy_remainder<LP, SP>(
        static_cast<uint8_t*>(dst) + n_chunks * ChunkSize,
        static_cast<uint8_t*>(src) + n_chunks * ChunkSize, remainder);
  }
}

template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wave(void* dst, void* src,
                                                             size_t size) {
  if (size == 0) return;

  constexpr int ChunkSize = 16;
  constexpr int Unroll    = 16;

  constexpr CachePolicy LP =
      is_put(Kind) ? CachePolicy::Standard : CachePolicy::SystemScope;
  constexpr CachePolicy SP =
      is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard;

  int wave_tid = get_flat_block_id() % WF_SIZE;
  int wave_size{wave_SZ()};
  int n_chunks = size / ChunkSize;
  int remainder = size % ChunkSize;

  if (n_chunks > 0) {
    copy_bulk<ChunkSize, LP, SP, Unroll>(dst, src, n_chunks, wave_tid, wave_size);
  }

  // Remainder handled uniquely by the first thread in the wave
  if (wave_tid == 0) {
    copy_remainder<LP, SP>(static_cast<uint8_t*>(dst) + n_chunks * ChunkSize,
                           static_cast<uint8_t*>(src) + n_chunks * ChunkSize, 
                           remainder);
  }
}

template <MemcpyKind Kind = MemcpyKind::Put>
[[maybe_unused]] __device__ __forceinline__ void memcpy_wg(void* dst, void* src,
                                                           size_t size) {
  if (size == 0) return;

  constexpr int ChunkSize = 16;
  constexpr int Unroll    = 16;

  constexpr CachePolicy LP =
      is_put(Kind) ? CachePolicy::Standard : CachePolicy::SystemScope;
  constexpr CachePolicy SP =
      is_put(Kind) ? CachePolicy::SystemScope : CachePolicy::Standard;

  int thread_id = get_flat_block_id();
  int block_size = get_flat_block_size();
  int n_chunks = size / ChunkSize;
  int remainder = size % ChunkSize;

  if (n_chunks > 0) {
    copy_bulk<ChunkSize, LP, SP, Unroll>(dst, src, n_chunks, thread_id, block_size);
  }

  // Remainder handled uniquely by the first thread in the workgroup
  if (thread_id == 0) {
    copy_remainder<LP, SP>(static_cast<uint8_t*>(dst) + n_chunks * ChunkSize,
                           static_cast<uint8_t*>(src) + n_chunks * ChunkSize,
                           remainder);
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

int rocm_init();

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr, int gpu_id);

}  // namespace rocshmem

#endif  // LIBRARY_SRC_UTIL_HPP_
