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

#include "sdma_policy.hpp"

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "envvar.hpp"
#include "util.hpp"

#if defined(USE_SDMA)
#include "sdma/anvil.hpp"
#endif

namespace rocshmem {

#if defined(USE_SDMA)

__host__ void SdmaImpl::sdmaHostInit(int pe, [[maybe_unused]] int num_pes, MPI_Comm comm) {
  my_pe = pe;

  // Get local communicator for shared memory
  MPI_Comm shmcomm;
  mpilib_ftable_.Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);

  int local_size;
  mpilib_ftable_.Comm_size(shmcomm, &local_size);
  shm_size = local_size;

  mpilib_ftable_.Comm_rank(shmcomm, &local_rank);

  // Read configuration from environment variables
  sdmaThreshold = static_cast<size_t>(envvar::sdma::threshold);
  numChannels = static_cast<int>(envvar::sdma::num_channels);
  minChunkPerChannel = static_cast<size_t>(envvar::sdma::min_chunk_per_channel);

  LOG_INFO("SDMA init with threshold=%zu, channels=%d, "
           "min_chunk=%zu, local_size=%d",
           sdmaThreshold, numChannels, minChunkPerChannel, shm_size);

  // Initialize the Anvil library
  anvil::anvil.init();

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  // Create SDMA connections to all other local PEs (numChannels per destination)
  for (int i = 0; i < shm_size; i++) {
    if (i != local_rank) {
      anvil::EnablePeerAccess(deviceId, i);
      anvil::anvil.connect(deviceId, i, numChannels);
    }
  }

  // Total number of handles: shm_size * numChannels
  // Indexed as: deviceHandles_d[local_pe * numChannels + channel_idx]
  int total_handles = shm_size * numChannels;

  // Allocate device-side array to hold SDMA queue device handles
  CHECK_HIP(hipMalloc(&deviceHandles_d,
                      total_handles * sizeof(anvil::SdmaQueueDeviceHandle*)));

  // Copy device handles to device memory
  anvil::SdmaQueueDeviceHandle** handles_h =
      new anvil::SdmaQueueDeviceHandle*[total_handles];
  for (int i = 0; i < shm_size; i++) {
    for (int ch = 0; ch < numChannels; ch++) {
      int idx = i * numChannels + ch;
      if (i != local_rank) {
        anvil::SdmaQueue* queue = anvil::anvil.getSdmaQueue(deviceId, i, ch);
        handles_h[idx] = queue ? queue->deviceHandle() : nullptr;
      } else {
        handles_h[idx] = nullptr;
      }
    }
  }
  CHECK_HIP(hipMemcpy(deviceHandles_d, handles_h,
                      total_handles * sizeof(anvil::SdmaQueueDeviceHandle*),
                      hipMemcpyHostToDevice));
  delete[] handles_h;

}


__host__ void SdmaImpl::sdmaHostInit(int pe, [[maybe_unused]] int num_pes, TcpBootstrap* bootstrap) {
  my_pe = pe;
  shm_size = bootstrap->getNranksPerNode();
  auto local_ranks = bootstrap->getLocalRanks();
  local_rank = std::find(local_ranks.begin(), local_ranks.end(), pe) - local_ranks.begin();

  // Read configuration from environment variables
  sdmaThreshold = static_cast<size_t>(envvar::sdma::threshold);
  numChannels = static_cast<int>(envvar::sdma::num_channels);
  minChunkPerChannel = static_cast<size_t>(envvar::sdma::min_chunk_per_channel);

  LOG_INFO("SDMA init with threshold=%zu, channels=%d, "
           "min_chunk=%zu, local_size=%d",
           sdmaThreshold, numChannels, minChunkPerChannel, shm_size);

  // Initialize the Anvil library
  anvil::anvil.init();

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  // Create SDMA connections to all other local PEs (numChannels per destination)
  for (int i = 0; i < shm_size; i++) {
    if (i != local_rank) {
      anvil::EnablePeerAccess(deviceId, i);
      anvil::anvil.connect(deviceId, i, numChannels);
    }
  }

  // Total number of handles: shm_size * numChannels
  // Indexed as: deviceHandles_d[local_pe * numChannels + channel_idx]
  int total_handles = shm_size * numChannels;

  // Allocate device-side array to hold SDMA queue device handles
  CHECK_HIP(hipMalloc(&deviceHandles_d,
                      total_handles * sizeof(anvil::SdmaQueueDeviceHandle*)));

  // Copy device handles to device memory
  anvil::SdmaQueueDeviceHandle** handles_h =
      new anvil::SdmaQueueDeviceHandle*[total_handles];
  for (int i = 0; i < shm_size; i++) {
    for (int ch = 0; ch < numChannels; ch++) {
      int idx = i * numChannels + ch;
      if (i != local_rank) {
        anvil::SdmaQueue* queue = anvil::anvil.getSdmaQueue(deviceId, i, ch);
        handles_h[idx] = queue ? queue->deviceHandle() : nullptr;
      } else {
        handles_h[idx] = nullptr;
      }
    }
  }
  CHECK_HIP(hipMemcpy(deviceHandles_d, handles_h,
                      total_handles * sizeof(anvil::SdmaQueueDeviceHandle*),
                      hipMemcpyHostToDevice));
  delete[] handles_h;

}


__host__ void SdmaImpl::sdmaHostStop() {
  LOG_TRACE("SDMA stop");
  if (deviceHandles_d != nullptr) {
    CHECK_HIP(hipFree(deviceHandles_d));
    deviceHandles_d = nullptr;
  }
}

#endif  // USE_SDMA

}  // namespace rocshmem
