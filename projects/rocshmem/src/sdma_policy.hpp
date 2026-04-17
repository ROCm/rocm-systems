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
class SdmaOnImpl {
 public:
  // Configuration (set from environment variables during init)
  size_t sdmaThreshold{128};  // Use SDMA for transfers >= 128B
  size_t minChunkPerChannel{4096};  // Minimum bytes per channel to avoid over-parallelization
  int numChannels{1};

  // Device resources - 2D array: [shm_size * numChannels]
  // Index as: deviceHandles_d[local_pe * numChannels + channel_idx]
  anvil::SdmaQueueDeviceHandle** deviceHandles_d{nullptr};
  uint64_t* signalPtrs{nullptr};
  uint64_t* expectedSignals{nullptr};
  int shm_size{0};
  int my_pe{0};
  int local_rank{0};

  // Host initialization (called from IpcOnImpl::ipcHostInit)
  __host__ void sdmaHostInit(int pe, int num_pes, MPI_Comm comm);
  __host__ void sdmaHostInit(int pe, int num_pes, TcpBootstrap* bootstrap);
  __host__ void sdmaHostStop();

  // Check SDMA availability for target PE
  __host__ __device__ bool isSdmaAvailable(int src_pe, int target_pe) {
    // SDMA is only available for local (same-node) PEs
    // and when device handles have been initialized
    if (deviceHandles_d == nullptr) {
      // printf("[SDMA] isSdmaAvailable: src=%d target=%d -> false (no handles)\n",
      //        src_pe, target_pe);
      return false;
    }
    // printf("[SDMA] isSdmaAvailable: src=%d target=%d -> true\n", src_pe, target_pe);
    // For now, assume all local PEs can use SDMA
    return true;
  }

#if defined(__HIPCC__) || defined(__CUDACC__)
  // Device-side copy using a single channel (for single-thread operations)
  __device__ void sdmaCopy(void* dst, void* src, size_t size, int pe) {
    int local_pe = pe % shm_size;
    // Use channel 0 for single-thread operations
    int idx = local_pe * numChannels;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      // printf("[SDMA] sdmaCopy: dst=%p src=%p size=%zu pe=%d idx=%d\n",
      //        dst, src, size, pe, idx);
      anvil::putWithSignal(*handle, dst, src, size, &signalPtrs[idx]);
      __hip_atomic_fetch_add(&expectedSignals[idx], 1,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  // Wave-level copy: split transfer across multiple channels
  // Each participating lane handles a different channel concurrently
  __device__ void sdmaCopy_wave(void* dst, void* src, size_t size, int pe) {
    int local_pe = pe % shm_size;
    int lane_id = get_flat_block_id() % WF_SIZE;
    int num_lanes = wave_SZ();

    // Determine how many channels to use based on transfer size
    int channels_to_use = (size / minChunkPerChannel);
    if (channels_to_use < 1) channels_to_use = 1;
    if (channels_to_use > numChannels) channels_to_use = numChannels;
    if (channels_to_use > num_lanes) channels_to_use = num_lanes;

    if (lane_id < channels_to_use) {
      int channel_idx = lane_id;
      size_t chunk_size = size / channels_to_use;
      size_t offset = lane_id * chunk_size;

      // Last participating lane handles remainder
      if (lane_id == channels_to_use - 1) {
        chunk_size = size - offset;
      }

      int idx = local_pe * numChannels + channel_idx;
      anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
      if (handle != nullptr && chunk_size > 0) {
        char* dst_chunk = static_cast<char*>(dst) + offset;
        char* src_chunk = static_cast<char*>(src) + offset;
        // printf("[SDMA] sdmaCopy_wave: lane=%d dst=%p src=%p size=%zu pe=%d ch=%d idx=%d\n",
        //        lane_id, dst_chunk, src_chunk, chunk_size, pe, channel_idx, idx);
        anvil::putWithSignal(*handle, dst_chunk, src_chunk, chunk_size, &signalPtrs[idx]);
        __hip_atomic_fetch_add(&expectedSignals[idx], 1,
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }

  // Workgroup-level copy: split transfer across multiple channels
  // Multiple threads prepare and submit packets concurrently
  __device__ void sdmaCopy_wg(void* dst, void* src, size_t size, int pe) {
    int local_pe = pe % shm_size;
    int thread_id = get_flat_block_id();

    // Determine how many channels to use based on transfer size
    int channels_to_use = (size / minChunkPerChannel);
    if (channels_to_use < 1) channels_to_use = 1;
    if (channels_to_use > numChannels) channels_to_use = numChannels;

    if (thread_id < channels_to_use) {
      int channel_idx = thread_id;
      size_t chunk_size = size / channels_to_use;
      size_t offset = thread_id * chunk_size;

      // Last participating thread handles remainder
      if (thread_id == channels_to_use - 1) {
        chunk_size = size - offset;
      }

      int idx = local_pe * numChannels + channel_idx;
      anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
      if (handle != nullptr && chunk_size > 0) {
        char* dst_chunk = static_cast<char*>(dst) + offset;
        char* src_chunk = static_cast<char*>(src) + offset;
        // printf("[SDMA] sdmaCopy_wg: thread=%d dst=%p src=%p size=%zu pe=%d ch=%d idx=%d\n",
        //        thread_id, dst_chunk, src_chunk, chunk_size, pe, channel_idx, idx);
        anvil::putWithSignal(*handle, dst_chunk, src_chunk, chunk_size, &signalPtrs[idx]);
        __hip_atomic_fetch_add(&expectedSignals[idx], 1,
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
  }

  // Wait for SDMA completions for a specific PE (all channels)
  __device__ void sdmaQuiet(int pe) {
    int local_pe = pe % shm_size;
    // printf("[SDMA] sdmaQuiet: pe=%d\n", pe);
    for (int ch = 0; ch < numChannels; ch++) {
      int idx = local_pe * numChannels + ch;
      uint64_t expected = __hip_atomic_load(&expectedSignals[idx],
                                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      anvil::waitForSignal(reinterpret_cast<HSAuint64*>(&signalPtrs[idx]), expected);
    }
  }

  // Wait for all SDMA completions (all PEs, all channels)
  __device__ void sdmaQuietAll() {
    // printf("[SDMA] sdmaQuietAll: shm_size=%d\n", shm_size);
    for (int pe = 0; pe < shm_size; pe++) {
      for (int ch = 0; ch < numChannels; ch++) {
        int idx = pe * numChannels + ch;
        uint64_t expected = __hip_atomic_load(&expectedSignals[idx],
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        anvil::waitForSignal(reinterpret_cast<HSAuint64*>(&signalPtrs[idx]), expected);
      }
    }
  }
#endif  // __HIPCC__ || __CUDACC__
};
#endif  // USE_SDMA

// clang-format off
NOWARN(-Wunused-parameter,
class SdmaOffImpl {
 public:
  size_t sdmaThreshold{8192};
  size_t minChunkPerChannel{4096};
  int numChannels{2};
  void** deviceHandles_d{nullptr};
  uint64_t* signalPtrs{nullptr};
  uint64_t* expectedSignals{nullptr};
  int shm_size{0};
  int my_pe{0};
  int local_rank{0};

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
