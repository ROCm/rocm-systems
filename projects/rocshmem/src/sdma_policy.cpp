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

__host__ void SdmaImpl::sdmaHostInit(int pe, int num_pes, int rank) {
  my_pe = pe;
  shm_size = num_pes;
  local_rank = rank;

  // Read configuration from environment variables
  sdmaEnabled = static_cast<bool>(envvar::sdma::enabled);
  sdmaThreshold = static_cast<size_t>(envvar::sdma::threshold);
  numChannels = static_cast<int>(envvar::sdma::num_channels);
  if (numChannels < 1 || numChannels > 8) {
    LOG_ERROR_ABORT("ROCSHMEM_SDMA_NUM_CHANNELS=%d is out of range [1, 8]", numChannels);
  }

  if (!sdmaEnabled) {
    LOG_INFO("SDMA disabled at runtime (ROCSHMEM_SDMA_ENABLED=0)");
    return;
  }

  LOG_INFO("SDMA init with threshold=%zu, channels=%d, local_size=%d",
           sdmaThreshold, numChannels, shm_size);

  // Initialize the Anvil library
  anvil::anvil.init();

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  // Create SDMA connections to all local PEs including self
  for (int i = 0; i < shm_size; i++) {
    if (i != deviceId) {
      anvil::EnablePeerAccess(deviceId, i);
    }
    anvil::anvil.connect(deviceId, i, numChannels);
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
      anvil::SdmaQueue* queue = anvil::anvil.getSdmaQueue(deviceId, i, ch);
      handles_h[idx] = queue ? queue->deviceHandle() : nullptr;
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
  anvil::anvil.disconnect();
}

#endif  // USE_SDMA

}  // namespace rocshmem
