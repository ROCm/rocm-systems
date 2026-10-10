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

#include <vector>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "envvar.hpp"
#include "log.hpp"
#include "sdma_agreement.hpp"
#include "util.hpp"

#if defined(USE_SDMA)
#include "sdma/anvil.hpp"
#endif

namespace rocshmem {

#if defined(USE_SDMA)

__host__ void SdmaImpl::sdmaHostInit(int pe, int num_pes, int rank,
                                     const LocalAllGather& allGather) {
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
  const bool forceFail =
      static_cast<int>(envvar::sdma::fail_connect_local_rank) == local_rank;

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  // Peers this init wired, so a fallback releases exactly those.
  std::vector<int> wired;
  uint32_t queuesBefore = 0;
  int status = kSdmaDisabledByEnv;
  if (sdmaEnabled) {
    LOG_INFO("SDMA init with threshold=%zu, channels=%d, local_size=%d",
             sdmaThreshold, numChannels, shm_size);

    // Initialize the Anvil library
    sdma_anvil::anvil.init();
    queuesBefore = sdma_anvil::anvil.queuesUsed();
    status = kSdmaReady;

    // Create SDMA connections to all local PEs including self. A failure is not acted on here:
    // sdmaEnabled also selects the alltoall algorithm, and ranks that disagree on it run
    // incompatible synchronization and hang, so the result is agreed across the group below.
    for (int i = 0; i < shm_size; i++) {
      if (i != deviceId) {
        sdma_anvil::EnablePeerAccess(deviceId, i);
      }
      // The forced failure hits the last peer, so the earlier ones are wired and the fallback's
      // release path runs.
      const bool forced = forceFail && i == shm_size - 1;
      if (forced || !sdma_anvil::anvil.connect(deviceId, i, numChannels)) {
        LOG_ERROR("SDMA: connect failed from device %d to %d with %d channel(s)%s", deviceId, i,
                  numChannels, forced ? " (forced by ROCSHMEM_SDMA_FAIL_CONNECT_LOCAL_RANK)" : "");
        status = kSdmaConnectFailed;
        break;
      }
      wired.push_back(i);
    }
  }

  std::vector<int> statuses(shm_size, kSdmaDisabledByEnv);
  statuses[local_rank] = status;
  allGather(statuses.data());
  const SdmaAgreement agreed = agreeSdmaStatus(statuses.data(), shm_size);

  switch (agreed.decision) {
    case SdmaDecision::kCommit:
      sdmaPublishHandles(deviceId, true);
      return;
    case SdmaDecision::kAllDisabled:
      LOG_INFO("SDMA disabled at runtime (ROCSHMEM_SDMA_ENABLED=0)");
      return;
    case SdmaDecision::kFallbackConnectFailed:
      LOG_ERROR("SDMA: connect failed on local rank(s) %s of %d; all of them fall back to IPC "
                "memcpy. Lower ROCSHMEM_SDMA_NUM_CHANNELS or use fewer ranks per node to keep "
                "SDMA", agreed.ranks.c_str(), shm_size);
      break;
    case SdmaDecision::kFallbackEnvMismatch:
      LOG_ERROR("SDMA: ROCSHMEM_SDMA_ENABLED=0 on local rank(s) %s of %d only; all of them fall "
                "back to IPC memcpy. Set it the same on every rank", agreed.ranks.c_str(),
                shm_size);
      break;
  }

  // Release only what this init wired: the channel map is process-global, so disconnect() would
  // also destroy queues another user still holds.
  for (int i : wired) sdma_anvil::anvil.disconnectDevice(i);
  if (status != kSdmaDisabledByEnv) {
    const uint32_t queuesAfter = sdma_anvil::anvil.queuesUsed();
    if (queuesAfter != queuesBefore) {
      LOG_WARN("SDMA: %u queue(s) held after the fallback, %u before init", queuesAfter,
               queuesBefore);
    } else {
      LOG_INFO("SDMA: fallback released %zu peer(s), %u queue(s) held", wired.size(), queuesAfter);
    }
  }
  sdmaEnabled = false;
  // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
  sdmaPublishHandles(deviceId, false);
}

__host__ void SdmaImpl::sdmaPublishHandles(int deviceId, bool useQueues) {
  // Total number of handles: shm_size * numChannels
  // Indexed as: deviceHandles_d[local_pe * numChannels + channel_idx]
  int total_handles = shm_size * numChannels;

  // Allocate device-side array to hold SDMA queue device handles
  CHECK_HIP(hipMalloc(&deviceHandles_d,
                      total_handles * sizeof(sdma_anvil::SdmaQueueDeviceHandle*)));

  // Copy device handles to device memory
  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> handles_h(total_handles, nullptr);
  if (useQueues) {
    for (int i = 0; i < shm_size; i++) {
      for (int ch = 0; ch < numChannels; ch++) {
        int idx = i * numChannels + ch;
        sdma_anvil::SdmaQueue* queue = sdma_anvil::anvil.getSdmaQueue(deviceId, i, ch);
        handles_h[idx] = queue ? queue->deviceHandle() : nullptr;
      }
    }
  }
  CHECK_HIP(hipMemcpy(deviceHandles_d, handles_h.data(),
                      total_handles * sizeof(sdma_anvil::SdmaQueueDeviceHandle*),
                      hipMemcpyHostToDevice));
}

__host__ void SdmaImpl::sdmaHostStop() {
  LOG_TRACE("SDMA stop");
  if (deviceHandles_d != nullptr) {
    CHECK_HIP(hipFree(deviceHandles_d));
    deviceHandles_d = nullptr;
  }
  // Release only the peers sdmaHostInit wired: the channel map is process-global, so disconnect()
  // would also destroy queues another user still holds. A disabled init wired none.
  if (sdmaEnabled) {
    for (int i = 0; i < shm_size; i++) sdma_anvil::anvil.disconnectDevice(i);
  }
}

#endif  // USE_SDMA

}  // namespace rocshmem
