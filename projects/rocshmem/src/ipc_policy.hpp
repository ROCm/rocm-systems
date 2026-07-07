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

#ifndef LIBRARY_SRC_IPC_POLICY_HPP_
#define LIBRARY_SRC_IPC_POLICY_HPP_

#include <hip/hip_runtime.h>

#include <atomic>
#include <vector>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "constmem.hpp"
#include "mpi_instance.hpp"
#include "memory/std_allocator.hpp"
#include "util.hpp"
#include "bootstrap/bootstrap.hpp"
#include "atomic.hpp"
#include "sdma_policy.hpp"

namespace rocshmem {

class Backend;
class Context;

class IpcOnImpl {
 protected:
  using HEAP_BASES_T = std::vector<char *, StdAllocatorHIP<char *>>;

 public:
  int shm_rank{0};

  int shm_size{0};

  char **ipc_bases{nullptr};

  int *pes_with_ipc_avail{nullptr};

  /**
   * @brief Fast O(1) IPC availability check.
   *
   * IPC-available PEs are either consecutive (e.g., [0,1,2,...,7]) or
   * strided (e.g., [0,8,16,...]) in world rank. Detected at init time
   * and stored as first_pe + stride, enabling arithmetic membership
   * check with zero memory loads.
   */
  int ipc_first_pe{0};
  int ipc_stride{0};    // 0 = pattern invalid

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            MPI_Comm thread_comm);

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            TcpBootstrap *bootstrap);

  __host__ void ipcHostStop();

  /**
   * @brief Detect consecutive or strided pattern from a host-accessible rank array.
   * Must be called with host memory (not a hipMalloc'd device pointer).
   * Sets ipc_stride:
   *   > 0: consecutive (1) or strided pattern detected
   *    0: irregular pattern (use linear scan fallback)
   * IPC disabled is indicated by constmem.ipc_shm_size == 0.
   */
  __host__ void ipcDetectPattern(const int* host_ranks, int count) {
    ipc_stride = 0;
    if (host_ranks == nullptr || count <= 0) {
      return;
    }
    ipc_first_pe = host_ranks[0];
    int stride = (count > 1) ? (host_ranks[1] - host_ranks[0]) : 1;
    if (stride <= 0) {
      return;
    }
    for (int i = 0; i < count; i++) {
      if (host_ranks[i] != ipc_first_pe + stride * i) {
        return;
      }
    }
    ipc_stride = stride;
  }

  __host__ bool isIpcAvailable([[maybe_unused]] int my_pe, int target_pe, int *local_target_pe) {
    if (nullptr == pes_with_ipc_avail) { return false; }
    for (int i=0; i<shm_size; i++) {
      if (pes_with_ipc_avail[i] == target_pe) {
        *local_target_pe = i;
        return true;
      }
    }
    return false;
  }

  void initFrom(const IpcOnImpl &other) {
    ipc_bases = other.ipc_bases;
    shm_size = other.shm_size;
    shm_rank = other.shm_rank;
    pes_with_ipc_avail = other.pes_with_ipc_avail;
  }

  void assignSdmaChannel([[maybe_unused]] unsigned int ctx_id) {}


  __device__ __attribute__((noinline))
  bool isIpcAvailable_irregular(int target_pe, int *local_target_pe) {
    for (int i = 0; i < shm_size; i++) {
      if (pes_with_ipc_avail[i] == target_pe) {
        *local_target_pe = i;
        return true;
      }
    }
    return false;
  }

  __device__ __attribute__((noinline))
  bool isIpcAvailable_strided(unsigned offset, int stride, int *local_target_pe) {
    int idx = static_cast<int>(offset) / stride;
    if (static_cast<unsigned>(idx) < static_cast<unsigned>(constmem.ipc_shm_size) &&
        offset == static_cast<unsigned>(idx * stride)) {
      *local_target_pe = idx;
      return true;
    }
    return false;
  }

  __device__ bool isIpcAvailable([[maybe_unused]] int my_pe, int target_pe, int *local_target_pe) {
    // IPC disabled: shm_size == 0
    if (constmem.ipc_shm_size == 0) return false;

    unsigned offset = static_cast<unsigned>(target_pe - constmem.ipc_first_pe);
    int stride = constmem.ipc_stride;

    if (stride == 1) {
      // Consecutive PEs: branchless range check.
      // Negative offsets wrap to large unsigned, failing the compare.
      *local_target_pe = static_cast<int>(offset);
      return offset < static_cast<unsigned>(constmem.ipc_shm_size);
    }

    if (stride > 1) {
      return isIpcAvailable_strided(offset, stride, local_target_pe);
    }

    // stride == 0: irregular pattern (noinline linear scan)
    return isIpcAvailable_irregular(target_pe, local_target_pe);
  }


  __device__ void ipcGpuInit(Backend *gpu_backend, Context *ctx, int thread_id);

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy(void *dst, void *src, size_t size, [[maybe_unused]] int local_pe) {
    memcpy_lane<Kind>(dst, src, size);
  }

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wg(void *dst, void *src, size_t size, [[maybe_unused]] int local_pe) {
    memcpy_wg<Kind>(dst, src, size);
  }

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wave(void *dst, void *src, size_t size, [[maybe_unused]] int local_pe) {
    memcpy_wave<Kind>(dst, src, size);
  }

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence() {
    detail::atomic::threadfence<scope, order>();
  }

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence([[maybe_unused]] int local_pe) {
    detail::atomic::threadfence<scope, order>();
  }

  __device__ void ipcQuiet() {
    detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                detail::atomic::memory_order_release>();
  }

  __device__ void ipcQuiet([[maybe_unused]] int local_pe) {
    detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                detail::atomic::memory_order_release>();
  }

  template <typename T>
  __device__ void ipcAMOAdd(T *val, T value) {
    __hip_atomic_fetch_add(val, value, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchAdd(T *val, T value) {
    return __hip_atomic_fetch_add(val, value, __ATOMIC_SEQ_CST,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOCas(T *val, T cond, T value) {
    __hip_atomic_compare_exchange_strong(val, &cond, value, __ATOMIC_SEQ_CST,
                                         __ATOMIC_SEQ_CST,
                                         __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchCas(T *val, T cond, T value) {
    __hip_atomic_compare_exchange_strong(val, &cond, value, __ATOMIC_SEQ_CST,
                                         __ATOMIC_SEQ_CST,
                                         __HIP_MEMORY_SCOPE_SYSTEM);
    return cond;
  }

  template <typename T>
  __device__ void ipcAMOSet(T *val, T value) {
    __hip_atomic_store(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOSwap(T *val, T value) {
    return __hip_atomic_exchange(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOAnd(T *val, T value) {
    __hip_atomic_fetch_and(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchAnd(T *val, T value) {
    return __hip_atomic_fetch_and(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOOr(T *val, T value) {
    __hip_atomic_fetch_or(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchOr(T *val, T value) {
    return __hip_atomic_fetch_or(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOXor(T *val, T value) {
    __hip_atomic_fetch_xor(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchXor(T *val, T value) {
    return __hip_atomic_fetch_xor(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  __device__ void zero_byte_read(int pe) {
    int local_pe = pe % shm_size;
    uint32_t *pe_ipc_base = reinterpret_cast<uint32_t *>(ipc_bases[local_pe]);
    [[maybe_unused]] volatile uint32_t read_value = __hip_atomic_load(
        pe_ipc_base, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }
};

#if defined(USE_SDMA)
class IpcSdmaImpl : public IpcOnImpl {
 public:
  SdmaImpl sdmaImpl_;

  void initFrom(const IpcSdmaImpl &other) {
    IpcOnImpl::initFrom(other);
    sdmaImpl_ = other.sdmaImpl_;
  }

  void assignSdmaChannel(unsigned int ctx_id) {
    if (sdmaImpl_.numChannels > 0) {
      sdmaImpl_.sdmaChannel = ctx_id % sdmaImpl_.numChannels;
      // stride=1 (spread wf_id across channels) for the default context (ctx_id=0,
      // shared by all WGs) or when ROCSHMEM_SDMA_SPREAD_CHANNELS=1 forces it.
      // Per-WG contexts (ctx_id>=1) already distribute across channels via ctx_id;
      // the wf_id offset would only reshuffle contention without reducing it.
      sdmaImpl_.sdmaChannelStride =
          ((ctx_id == 0) || static_cast<bool>(envvar::sdma::spread_channels)) ? 1 : 0;
    }
  }

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            MPI_Comm thread_comm);

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            TcpBootstrap *bootstrap);

  __host__ void ipcHostStop();

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy(void *dst, void *src, size_t size, int local_pe) {
    if (sdmaImpl_.sdmaEnabled && size >= sdmaImpl_.sdmaThreshold) {
      auto* handle = sdmaImpl_.sdmaCopy<Kind>(dst, src, size, local_pe);
      assert(nullptr != handle /* Assuming sdma is available to all pes uniformly */);
      if constexpr (is_blocking(Kind)) handle->quietAll();
      return;
    }
    memcpy_lane<Kind>(dst, src, size);
  }

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wg(void *dst, void *src, size_t size, int local_pe) {
    if (sdmaImpl_.sdmaEnabled && size >= sdmaImpl_.sdmaThreshold) {
      anvil::SdmaQueueDeviceHandle* handle = nullptr;
      if (is_thread_zero_in_block()) {
        handle = sdmaImpl_.sdmaCopy<Kind>(dst, src, size, local_pe);
        assert(nullptr != handle /* Assuming sdma is available to all pes uniformly */);
        if constexpr (is_blocking(Kind)) handle->quietAll();
      }
      return;
    }
    memcpy_wg<Kind>(dst, src, size);
  }

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wave(void *dst, void *src, size_t size, int local_pe) {
    if (sdmaImpl_.sdmaEnabled && size >= sdmaImpl_.sdmaThreshold) {
      anvil::SdmaQueueDeviceHandle* handle = nullptr;
      if (is_thread_zero_in_wave()) {
        handle = sdmaImpl_.sdmaCopy<Kind>(dst, src, size, local_pe);
        assert(nullptr != handle /* Assuming sdma is available to all pes uniformly */);
        if constexpr (is_blocking(Kind)) handle->quietAll();
      }
      return;
    }
    memcpy_wave<Kind>(dst, src, size);
  }

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence() {
    if (sdmaImpl_.sdmaEnabled &&
        __hip_atomic_load(&sdmaImpl_.sdmaDirty, __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_AGENT) != 0)
      sdmaImpl_.sdmaQuietAll();
    detail::atomic::threadfence<scope, order>();
  }

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence(int local_pe) {
    if (sdmaImpl_.sdmaEnabled) {
      uint64_t pe_mask = ((1ULL << sdmaImpl_.numChannels) - 1) <<
                         (local_pe * sdmaImpl_.numChannels);
      if (__hip_atomic_load(&sdmaImpl_.sdmaDirty, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) & pe_mask)
        sdmaImpl_.sdmaQuiet(local_pe);
    }
    detail::atomic::threadfence<scope, order>();
  }

  __device__ void ipcQuiet() {
    if (sdmaImpl_.sdmaEnabled &&
        __hip_atomic_load(&sdmaImpl_.sdmaDirty, __ATOMIC_RELAXED,
                          __HIP_MEMORY_SCOPE_AGENT) != 0)
      sdmaImpl_.sdmaQuietAll();
    detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                detail::atomic::memory_order_acq_rel>();
  }

  __device__ void ipcQuiet(int local_pe) {
    if (sdmaImpl_.sdmaEnabled) {
      uint64_t pe_mask = ((1ULL << sdmaImpl_.numChannels) - 1) <<
                         (local_pe * sdmaImpl_.numChannels);
      if (__hip_atomic_load(&sdmaImpl_.sdmaDirty, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_AGENT) & pe_mask)
        sdmaImpl_.sdmaQuiet(local_pe);
    }
    detail::atomic::threadfence<detail::atomic::memory_scope_system,
                                detail::atomic::memory_order_acq_rel>();
  }
};
#endif  // USE_SDMA

// clang-format off
NOWARN(-Wunused-parameter,
class IpcOffImpl {
  using HEAP_BASES_T = std::vector<char *, StdAllocatorHIP<char *>>;

 public:
  int shm_rank{0};

  uint32_t shm_size{0};

  char **ipc_bases{nullptr};

  int *pes_with_ipc_avail{nullptr};

  int ipc_first_pe{0};
  int ipc_stride{0};

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            MPI_Comm thread_comm) {}

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            TcpBootstrap *bootstrap){}

  __host__ void ipcHostStop() {}

  __host__ bool isIpcAvailable([[maybe_unused]] int my_pe, int target_pe, int *local_target_pe) { return false; }
  __device__ bool isIpcAvailable([[maybe_unused]] int my_pe, int target_pe, int *local_target_pe) { return false; }

  void initFrom(const IpcOffImpl &) {}

  void assignSdmaChannel(unsigned int) {}

  __device__ void ipcGpuInit(Backend *rocshmem_handle, Context *ctx,
                             int thread_id) {}

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy([[maybe_unused]] void *dst, [[maybe_unused]] void *src,
                          [[maybe_unused]] size_t size, [[maybe_unused]] int local_pe) {}

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wg([[maybe_unused]] void *dst, [[maybe_unused]] void *src,
                             [[maybe_unused]] size_t size, [[maybe_unused]] int local_pe) {}

  template <MemcpyKind Kind = MemcpyKind::Put>
  __device__ void ipcCopy_wave([[maybe_unused]] void *dst, [[maybe_unused]] void *src,
                               [[maybe_unused]] size_t size, [[maybe_unused]] int local_pe) {}

  __device__ void ipcQuiet() {}

  __device__ void ipcQuiet(int) {}

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence() {}

  template <detail::atomic::rocshmem_memory_scope scope = detail::atomic::memory_scope_system,
            detail::atomic::rocshmem_memory_order order = detail::atomic::memory_order_release>
  __device__ __forceinline__ void ipcFence(int) {}

  template <typename T>
  __device__ T ipcAMOFetchAdd(T *val, T value) {
    return T();
  }

  template <typename T>
  __device__ T ipcAMOFetchCas(T *val, T cond, T value) {
    return T();
  }

  template <typename T>
  __device__ void ipcAMOAdd(T *val, T value) {}

  template <typename T>
  __device__ void ipcAMOSet(T *val, T value) {}

  template <typename T>
  __device__ void ipcAMOCas(T *val, T cond, T value) {}

  __device__ void zero_byte_read(int pe) {}
};
)
// clang-format on

/*
 * Select which one of our IPC policies to use at compile time.
 */
#if defined(USE_SDMA)
typedef IpcSdmaImpl IpcImpl;
#elif defined(USE_IPC)
typedef IpcOnImpl IpcImpl;
#else
typedef IpcOffImpl IpcImpl;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_POLICY_HPP_
