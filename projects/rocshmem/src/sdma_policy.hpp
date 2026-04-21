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
  size_t sdmaThreshold{128};  // Use SDMA for transfers >= 128B
  size_t minChunkPerChannel{4096};  // Minimum bytes per channel to avoid over-parallelization
  int numChannels{1};

  // Device resources - 2D array: [shm_size * numChannels]
  // Index as: deviceHandles_d[local_pe * numChannels + channel_idx]
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
    int idx = local_pe * numChannels;
    anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
    if (handle != nullptr) {
      anvil::put(*handle, dst, src, size);
    }
    return handle;
  }

  // Wave-level copy: split transfer across multiple channels
  // Each participating lane handles a different channel concurrently.
  // Returns the handle used by this lane (nullptr if lane did not participate).
  __device__ anvil::SdmaQueueDeviceHandle* sdmaCopy_wave(void* dst, void* src,
                                                          size_t size, int local_pe) {
    int lane_id = get_flat_block_id() % WF_SIZE;
    int num_lanes = wave_SZ();
    anvil::SdmaQueueDeviceHandle* handle = nullptr;

    int channels_to_use = (size / minChunkPerChannel);
    if (channels_to_use < 1) channels_to_use = 1;
    if (channels_to_use > numChannels) channels_to_use = numChannels;
    if (channels_to_use > num_lanes) channels_to_use = num_lanes;

    if (lane_id < channels_to_use) {
      int channel_idx = lane_id;
      size_t chunk_size = size / channels_to_use;
      size_t offset = lane_id * chunk_size;

      if (lane_id == channels_to_use - 1) {
        chunk_size = size - offset;
      }

      int idx = local_pe * numChannels + channel_idx;
      handle = deviceHandles_d[idx];
      if (handle != nullptr && chunk_size > 0) {
        char* dst_chunk = static_cast<char*>(dst) + offset;
        char* src_chunk = static_cast<char*>(src) + offset;
        anvil::put(*handle, dst_chunk, src_chunk, chunk_size);
      }
    }
    return handle;
  }

  // Workgroup-level copy: split transfer across multiple channels
  // Multiple threads prepare and submit packets concurrently.
  // Returns the handle used by this thread (nullptr if thread did not participate).
  __device__ anvil::SdmaQueueDeviceHandle* sdmaCopy_wg(void* dst, void* src,
                                                        size_t size, int local_pe) {
    int thread_id = get_flat_block_id();
    anvil::SdmaQueueDeviceHandle* handle = nullptr;

    int channels_to_use = (size / minChunkPerChannel);
    if (channels_to_use < 1) channels_to_use = 1;
    if (channels_to_use > numChannels) channels_to_use = numChannels;

    if (thread_id < channels_to_use) {
      int channel_idx = thread_id;
      size_t chunk_size = size / channels_to_use;
      size_t offset = thread_id * chunk_size;

      if (thread_id == channels_to_use - 1) {
        chunk_size = size - offset;
      }

      int idx = local_pe * numChannels + channel_idx;
      handle = deviceHandles_d[idx];
      if (handle != nullptr && chunk_size > 0) {
        char* dst_chunk = static_cast<char*>(dst) + offset;
        char* src_chunk = static_cast<char*>(src) + offset;
        anvil::put(*handle, dst_chunk, src_chunk, chunk_size);
      }
    }
    __syncthreads();
    return handle;
  }

  // Wait for SDMA completions for a specific PE (all channels)
  // Uses rptr-based polling via anvil::quiet()
  __device__ void sdmaQuiet(int local_pe) {
    for (int ch = 0; ch < numChannels; ch++) {
      int idx = local_pe * numChannels + ch;
      anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
      if (handle != nullptr) {
        anvil::quiet(*handle);
      }
    }
  }

  // Wait for all SDMA completions (all PEs, all channels)
  // Uses rptr-based polling via anvil::quiet()
  __device__ void sdmaQuietAll() {
    for (int pe = 0; pe < shm_size; pe++) {
      for (int ch = 0; ch < numChannels; ch++) {
        int idx = pe * numChannels + ch;
        anvil::SdmaQueueDeviceHandle* handle = deviceHandles_d[idx];
        if (handle != nullptr) {
          anvil::quiet(*handle);
        }
      }
    }
  }
};
#endif  // USE_SDMA

}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_POLICY_HPP_
