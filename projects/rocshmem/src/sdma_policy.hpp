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

#ifndef LIBRARY_SRC_SDMA_POLICY_HPP_
#define LIBRARY_SRC_SDMA_POLICY_HPP_

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "mpi_instance.hpp"
#include "bootstrap/bootstrap.hpp"
#include "util.hpp"

#if defined(USE_SDMA)
#include "sdma/anvil.hpp"
#include "sdma/anvil_device.hpp"
#endif

namespace rocshmem {

class SdmaOnImpl {
 public:
  // Configuration (set from environment variables during init)
  size_t sdmaThreshold{8192};  // Use SDMA for transfers >= 8KB
  int numChannels{2};

  // Device resources
  anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  uint64_t* signalPtrs{nullptr};
  uint64_t* expectedSignals{nullptr};
  int shm_size{0};
  int my_pe{0};

  // Host initialization (called from IpcOnImpl::ipcHostInit)
  __host__ void sdmaHostInit(int pe, int num_pes, MPI_Comm comm);
  __host__ void sdmaHostInit(int pe, int num_pes, TcpBootstrap* bootstrap);
  __host__ void sdmaHostStop();

  // Check SDMA availability for target PE
  __host__ __device__ bool isSdmaAvailable(int src_pe, int target_pe) {
    // SDMA is only available for local (same-node) PEs
    // and when device handles have been initialized
    if (deviceHandles_d == nullptr) return false;
    // For now, assume all local PEs can use SDMA
    return true;
  }

#if defined(__HIPCC__) || defined(__CUDACC__)
  // Device-side copy (routes to SDMA queue)
  __device__ void sdmaCopy(void* dst, void* src, size_t size, int pe) {
    int local_pe = pe % shm_size;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[local_pe];
    if (handle != nullptr) {
      anvil::putWithSignal(*handle, dst, src, size, &signalPtrs[local_pe]);
      __hip_atomic_fetch_add(&expectedSignals[local_pe], 1,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  __device__ void sdmaCopy_wave(void* dst, void* src, size_t size, int pe) {
    // For wave-level, only first thread in wave submits
    if (is_thread_zero_in_wave()) {
      sdmaCopy(dst, src, size, pe);
    }
  }

  __device__ void sdmaCopy_wg(void* dst, void* src, size_t size, int pe) {
    // For workgroup-level, only first thread in block submits
    if (is_thread_zero_in_block()) {
      sdmaCopy(dst, src, size, pe);
    }
    __syncthreads();
  }

  // Wait for SDMA completions for a specific PE
  __device__ void sdmaQuiet(int pe) {
    int local_pe = pe % shm_size;
    uint64_t expected = __hip_atomic_load(&expectedSignals[local_pe],
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    anvil::waitForSignal(reinterpret_cast<HSAuint64*>(&signalPtrs[local_pe]), expected);
  }

  // Wait for all SDMA completions
  __device__ void sdmaQuietAll() {
    for (int i = 0; i < shm_size; i++) {
      sdmaQuiet(i);
    }
  }
#endif  // __HIPCC__ || __CUDACC__
};

// clang-format off
NOWARN(-Wunused-parameter,
class SdmaOffImpl {
 public:
  size_t sdmaThreshold{8192};
  int numChannels{2};
  void** deviceHandles_d{nullptr};
  uint64_t* signalPtrs{nullptr};
  uint64_t* expectedSignals{nullptr};
  int shm_size{0};
  int my_pe{0};

  __host__ void sdmaHostInit(int pe, int num_pes, MPI_Comm comm) {}
  __host__ void sdmaHostInit(int pe, int num_pes, TcpBootstrap* bootstrap) {}
  __host__ void sdmaHostStop() {}

  __host__ __device__ bool isSdmaAvailable(int src_pe, int target_pe) { return false; }

  __device__ void sdmaCopy(void* dst, void* src, size_t size, int pe) {}
  __device__ void sdmaCopy_wave(void* dst, void* src, size_t size, int pe) {}
  __device__ void sdmaCopy_wg(void* dst, void* src, size_t size, int pe) {}
  __device__ void sdmaQuiet(int pe) {}
  __device__ void sdmaQuietAll() {}
};
)
// clang-format on

/*
 * Select which one of our SDMA policies to use at compile time.
 */
#if defined(USE_SDMA)
typedef SdmaOnImpl SdmaImpl;
#else
typedef SdmaOffImpl SdmaImpl;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
