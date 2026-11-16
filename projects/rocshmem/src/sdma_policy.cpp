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
  int status = initialSdmaStatus(sdmaEnabled, numChannels);
  if (status == kSdmaBadChannels) {
    LOG_ERROR("ROCSHMEM_SDMA_NUM_CHANNELS=%d is out of range [1, 8]", numChannels);
  }

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  wiredPeers = 0;
  uint32_t queuesBefore = 0;
  if (status == kSdmaReady) {
    LOG_INFO("SDMA init with threshold=%zu, channels=%d, local_size=%d",
             sdmaThreshold, numChannels, shm_size);

    // Initialize the Anvil library
    sdma_anvil::anvil.init();
    queuesBefore = sdma_anvil::anvil.queuesUsed();

    // Connect to every local PE. A failure is not acted on here; the group agrees on the result
    // below, because disagreeing on sdmaEnabled hangs alltoall.
    for (int i = 0; i < shm_size; i++) {
      if (i != deviceId) {
        sdma_anvil::EnablePeerAccess(deviceId, i);
      }
      if (!sdma_anvil::anvil.connect(deviceId, i, numChannels)) {
        LOG_ERROR("SDMA: connect failed from device %d to %d with %d channel(s)", deviceId, i,
                  numChannels);
        status = kSdmaConnectFailed;
        break;
      }
      wiredPeers = i + 1;
    }
  }

  std::vector<int> statuses(shm_size, kSdmaDisabledByEnv);
  statuses[local_rank] = applyForcedFailure(status, local_rank, forcedFailureLocalRank());
  if (statuses[local_rank] != status) {
    LOG_ERROR("SDMA: local rank %d reports a failed connect, forced by "
              "ROCSHMEM_SDMA_FAIL_CONNECT_LOCAL_RANK", local_rank);
  }
  allGather(statuses.data());
  const SdmaAgreement agreed = agreeSdmaStatus(statuses.data(), shm_size);

  switch (agreed.decision) {
    case SdmaDecision::kAbortBadChannels:
      LOG_ERROR_ABORT("SDMA: ROCSHMEM_SDMA_NUM_CHANNELS is out of range [1, 8] on local rank(s) "
                      "%s of %d", agreed.ranks.c_str(), shm_size);
      return;
    case SdmaDecision::kCommit:
      sdmaPublishHandles(deviceId, true);
      return;
    case SdmaDecision::kAllDisabled:
      LOG_INFO("SDMA disabled at runtime (ROCSHMEM_SDMA_ENABLED=0)");
      // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
      sdmaPublishHandles(deviceId, false);
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

  const int released = wiredPeers;
  sdmaReleasePeers();
  if (status != kSdmaDisabledByEnv) {
    const uint32_t queuesAfter = sdma_anvil::anvil.queuesUsed();
    if (queuesAfter != queuesBefore) {
      LOG_WARN("SDMA: %u queue(s) held after the fallback, %u before init", queuesAfter,
               queuesBefore);
    } else {
      LOG_INFO("SDMA: fallback released %d peer(s), %u queue(s) held", released, queuesAfter);
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

__host__ void SdmaImpl::sdmaReleasePeers() {
  // Release only the peers sdmaHostInit wired: the channel map is process-global, so disconnect()
  // would also destroy queues another user still holds.
  for (int i = 0; i < wiredPeers; i++) sdma_anvil::anvil.disconnectDevice(i);
  wiredPeers = 0;
}

__host__ void SdmaImpl::sdmaHostStop() {
  LOG_TRACE("SDMA stop");
  if (deviceHandles_d != nullptr) {
    CHECK_HIP(hipFree(deviceHandles_d));
    deviceHandles_d = nullptr;
  }
  sdmaReleasePeers();
}

#endif  // USE_SDMA

}  // namespace rocshmem
