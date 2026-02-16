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
#include "mpi_instance.hpp"
#include "memory/hip_allocator.hpp"
#include "util.hpp"
#include "bootstrap/bootstrap.hpp"

namespace rocshmem {

class Backend;
class Context;

class IpcOnImpl {
  using HEAP_BASES_T = std::vector<char *, StdAllocatorHIP<char *>>;

 public:
  int shm_rank{0};

  int shm_size{0};

  char **ipc_bases{nullptr};

  int *pes_with_ipc_avail{nullptr};

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            MPI_Comm thread_comm);

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            TcpBootstrap *bootstrap);

  __host__ void ipcHostStop();

  __host__ __device__ bool isIpcAvailable(int my_pe, int target_pe, int *local_target_pe) {
    if (nullptr == pes_with_ipc_avail) { return false; }

    for (int i=0; i<shm_size; i++) {
      if (pes_with_ipc_avail[i] == target_pe) {
        *local_target_pe = i;
        return true;
      }
    }

    return false;
  }

  __device__ void ipcGpuInit(Backend *gpu_backend, Context *ctx, int thread_id);

  __device__ void ipcCopy(void *dst, void *src, size_t size);

  __device__ void ipcCopy_wg(void *dst, void *src, size_t size);

  __device__ void ipcCopy_wave(void *dst, void *src, size_t size);

  __device__ void ipcFence() { __threadfence_system(); }

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
    volatile uint32_t read_value = __hip_atomic_load(
        pe_ipc_base, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }
};

// clang-format off
NOWARN(-Wunused-parameter,
class IpcOffImpl {
  using HEAP_BASES_T = std::vector<char *, StdAllocatorHIP<char *>>;

 public:
  int shm_rank{0};

  uint32_t shm_size{0};

  char **ipc_bases{nullptr};

  int *pes_with_ipc_avail{nullptr};

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            MPI_Comm thread_comm) {}

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                            TcpBootstrap *bootstrap){}

  __host__ void ipcHostStop() {}

  __host__ __device__ bool isIpcAvailable(int my_pe, int target_pe, int *local_target_pe) { return false; }

  __device__ void ipcGpuInit(Backend *rocshmem_handle, Context *ctx,
                             int thread_id) {}

  __device__ void ipcCopy(void *dst, void *src, size_t size) {}

  __device__ void ipcCopy_wg(void *dst, void *src, size_t size) {}

  __device__ void ipcCopy_wave(void *dst, void *src, size_t size) {}

  __device__ void ipcFence() {}

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
#if defined(USE_IPC)
typedef IpcOnImpl IpcImpl;
#else
typedef IpcOffImpl IpcImpl;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_POLICY_HPP_
