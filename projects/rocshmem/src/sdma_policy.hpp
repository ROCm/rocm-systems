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

#if defined(USE_SDMA)
class SdmaImpl {
 public:
  // Configuration (set from environment variables during init)
  bool sdmaEnabled{true};
  uint64_t sdmaDirtyPEs{0};  // Bitmask: bit i set = PE i has pending SDMA ops (atomic)
  size_t sdmaThreshold{256};  // Use SDMA for transfers >= 256B
  int numChannels{1};
  int sdmaChannel{0};  // Per-context channel index (assigned at ctx creation)

  // Device resources - 2D array: [shm_size * numChannels]
  // Index as: deviceHandles_d[local_pe * numChannels + sdmaChannel]
  anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  int shm_size{0};
  int my_pe{0};
  int local_rank{0};

  // Host initialization (called from IpcOnImpl::ipcHostInit)
  __host__ void sdmaHostInit(int pe, int num_pes, MPI_Comm comm);
  __host__ void sdmaHostInit(int pe, int num_pes, TcpBootstrap* bootstrap);
  __host__ void sdmaHostStop();

  // Device-side copy using a single channel (for single-thread operations)
  // Returns the handle used (for direct quietAll by caller).
  __device__ anvil::SdmaQueueDeviceHandle* sdmaCopy(void* dst, void* src,
                                                     size_t size, int local_pe) {
    int idx = local_pe * numChannels + sdmaChannel;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      // Flush GL0/GL1 → GL2 before submitting the SDMA descriptor.
      // Fine-grain memory on AMD CDNA is CC (cache-coherent, cached in GL2):
      // the SDMA engine reads from GL2, but __syncthreads() in the caller only
      // drains stores to GL0 without flushing to GL2.  Agent scope is sufficient
      // because SDMA probes GL2 via the coherence protocol on the same die.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      anvil::put(*handle, dst, src, size);
      __hip_atomic_fetch_or(&sdmaDirtyPEs, 1ULL << local_pe, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return handle;
  }

  // Wait for SDMA completions for a specific PE (this context's channel)
  // Atomically reads and clears the PE's dirty bit — no race with concurrent submits.
  __device__ void sdmaQuiet(int local_pe) {
    uint64_t mask = 1ULL << local_pe;
    if (!(__hip_atomic_fetch_and(&sdmaDirtyPEs, ~mask, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & mask))
      return;
    int idx = local_pe * numChannels + sdmaChannel;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      anvil::quiet(*handle);
    }
  }

  // Wait for all SDMA completions (only PEs with pending ops)
  __device__ void sdmaQuietAll() {
    uint64_t dirty = __hip_atomic_exchange(&sdmaDirtyPEs, 0ULL, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
    while (dirty) {
      int pe = __builtin_ffsll(dirty) - 1;
      int idx = pe * numChannels + sdmaChannel;
      anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
      if (handle != nullptr) {
        anvil::quiet(*handle);
      }
      dirty &= ~(1ULL << pe);
    }
  }
};
#endif  // USE_SDMA

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
