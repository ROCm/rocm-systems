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

#include <algorithm>
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

__host__ void SdmaImpl::sdmaHostInit(int pe, int local_size, int lrank, int pod_sz, int prank,
                                     const std::vector<int>& local_pod_index,
                                     const PodAllGather& allGather) {
  my_pe = pe;
  shm_size = local_size;
  local_rank = lrank;
  pod_size = pod_sz;
  pod_rank = prank;

  // Read configuration from environment variables
  sdmaEnabled = static_cast<bool>(envvar::sdma::enabled);
  sdmaThreshold = static_cast<size_t>(envvar::sdma::threshold);
  numChannels = static_cast<int>(envvar::sdma::num_channels);
  if (numChannels < 1 || numChannels > 8) {
    LOG_ERROR_ABORT("ROCSHMEM_SDMA_NUM_CHANNELS=%d is out of range [1, 8]", numChannels);
  }

  // Get current device
  int deviceId;
  CHECK_HIP(hipGetDevice(&deviceId));

  uint32_t queuesBefore = 0;
  int status = kSdmaDisabledByEnv;
  if (sdmaEnabled) {
    LOG_INFO("SDMA init with threshold=%zu, channels=%d, local_size=%d, pod_size=%d",
             sdmaThreshold, numChannels, shm_size, pod_size);

    // Initialize the Anvil library
    sdma_anvil::anvil.init();
    queuesBefore = sdma_anvil::anvil.queuesUsed();
    status = kSdmaReady;

    // Peer access lets SDMA read and write the other local GPUs' heaps.
    int numDevices = 0;
    CHECK_HIP(hipGetDeviceCount(&numDevices));
    for (int i = 0; i < std::min(shm_size, numDevices); i++) {
      if (i != deviceId) sdma_anvil::EnablePeerAccess(deviceId, i);
    }
    // A failure is not acted on here; the pod agrees on the result below, because disagreeing on
    // sdmaEnabled hangs alltoall.
    if (!sdmaBuildPool(deviceId)) status = kSdmaConnectFailed;
  }

  std::vector<int> statuses(pod_size, kSdmaDisabledByEnv);
  statuses[pod_rank] = applyForcedFailure(status, local_rank, forcedFailureLocalRank());
  if (statuses[pod_rank] != status) {
    LOG_ERROR("SDMA: local rank %d reports a failed connect, forced by "
              "ROCSHMEM_SDMA_FAIL_CONNECT_LOCAL_RANK", local_rank);
  }
  allGather(statuses.data());
  const SdmaAgreement agreed = agreeSdmaStatus(statuses.data(), pod_size);

  switch (agreed.decision) {
    case SdmaDecision::kCommit:
      sdmaPublishHandles(deviceId, true, local_pod_index);
      return;
    case SdmaDecision::kAllDisabled:
      LOG_INFO("SDMA disabled at runtime (ROCSHMEM_SDMA_ENABLED=0)");
      // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
      sdmaPublishHandles(deviceId, false, local_pod_index);
      return;
    case SdmaDecision::kFallbackConnectFailed:
      LOG_ERROR("SDMA: connect failed on pod rank(s) %s of %d; all of them fall back to IPC "
                "memcpy. Lower ROCSHMEM_SDMA_NUM_CHANNELS or use fewer ranks per node to keep "
                "SDMA", agreed.ranks.c_str(), pod_size);
      break;
    case SdmaDecision::kFallbackEnvMismatch:
      LOG_ERROR("SDMA: ROCSHMEM_SDMA_ENABLED=0 on pod rank(s) %s of %d only; all of them fall "
                "back to IPC memcpy. Set it the same on every rank", agreed.ranks.c_str(),
                pod_size);
      break;
  }

  const int released = numPoolEngines * numPoolLayers;
  sdmaReleasePool();
  if (status != kSdmaDisabledByEnv) {
    const uint32_t queuesAfter = sdma_anvil::anvil.queuesUsed();
    if (queuesAfter != queuesBefore) {
      LOG_WARN("SDMA: %u queue(s) held after the fallback, %u before init", queuesAfter,
               queuesBefore);
    } else {
      LOG_INFO("SDMA: fallback released %d pool queue(s), %u queue(s) held", released,
               queuesAfter);
    }
  }
  sdmaEnabled = false;
  // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
  sdmaPublishHandles(deviceId, false, local_pod_index);
}

__host__ bool SdmaImpl::sdmaBuildPool(int deviceId) {
  std::vector<uint32_t> engines = sdma_anvil::anvil.usableEngineIds();
  if (engines.empty()) {
    // Engine count unknown: fall back to the single engine connect() would use for self.
    const int e = sdma_anvil::anvil.engineFor(deviceId, deviceId);
    if (e < 0) {
      LOG_ERROR("SDMA: no usable engine on device %d", deviceId);
      return false;
    }
    engines.push_back(static_cast<uint32_t>(e));
  }
  const int maxEngines = std::min({static_cast<int>(engines.size()), kMaxPoolEngines,
                                   kMaxPoolQueues});
  const int maxLayers = std::max(1, std::min(numChannels, kMaxPoolQueues / maxEngines));

  poolId = sdma_anvil::anvil.allocPoolId();
  numPoolEngines = 0;
  numPoolLayers = 0;
  for (int layer = 0; layer < maxLayers; ++layer) {
    int created = 0;
    for (int e = 0; e < maxEngines; ++e) {
      if (layer > 0 && e >= numPoolEngines) break;
      if (sdma_anvil::anvil.createPoolQueue(deviceId, poolId, engines[static_cast<size_t>(e)]) ==
          nullptr) {
        break;
      }
      if (layer == 0) poolEngines[e] = static_cast<int>(engines[static_cast<size_t>(e)]);
      ++created;
    }
    if (layer == 0) {
      // Keep however many engines got a first queue; fewer engines is still SDMA.
      numPoolEngines = created;
      if (created == 0) break;
      numPoolLayers = 1;
    } else if (created < numPoolEngines) {
      // Out of queue budget part way through a layer: keep only the complete layers.
      sdma_anvil::anvil.trimPool(poolId, static_cast<size_t>(numPoolLayers * numPoolEngines));
      break;
    } else {
      ++numPoolLayers;
    }
  }

  if (numPoolEngines == 0) {
    LOG_ERROR("SDMA: could not create any queue on device %d", deviceId);
    sdmaReleasePool();
    return false;
  }
  LOG_INFO("SDMA: device %d pool of %d engine(s) x %d queue(s) per engine (%zu usable engines)",
           deviceId, numPoolEngines, numPoolLayers, engines.size());
  return true;
}

__host__ void SdmaImpl::sdmaPublishHandles(int deviceId, bool useQueues,
                                           const std::vector<int>& local_pod_index) {
  const int numSlots = pod_size * numChannels;
  const int poolSize = useQueues ? numPoolEngines * numPoolLayers : 0;

  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> handles_h(numSlots, nullptr);
  std::vector<uint8_t> slotQueue_h(numSlots, 0);
  std::vector<uint64_t> peMask_h(pod_size, 0);
  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> pool_h(std::max(poolSize, 1), nullptr);

  if (useQueues) {
    for (int q = 0; q < poolSize; q++) {
      sdma_anvil::SdmaQueue* queue = sdma_anvil::anvil.getSdmaQueue(deviceId, poolId, q);
      pool_h[q] = queue ? queue->deviceHandle() : nullptr;
    }

    // Starting engine per pod PE. Off-node PEs round-robin over the pool; on-node PEs start on the
    // engine connect() would have picked for that device, so their routing is unchanged.
    std::vector<int> baseEngine(pod_size);
    for (int p = 0; p < pod_size; p++) baseEngine[p] = p % numPoolEngines;
    for (int i = 0; i < static_cast<int>(local_pod_index.size()); i++) {
      const int p = local_pod_index[i];
      if (p < 0 || p >= pod_size) continue;
      const int engine = sdma_anvil::anvil.engineFor(deviceId, i);
      for (int e = 0; e < numPoolEngines; e++) {
        if (poolEngines[e] == engine) {
          baseEngine[p] = e;
          break;
        }
      }
    }

    // Channels of one PE step across engines; PEs that share a starting engine step across that
    // engine's queues, so concurrent peers contend on different write pointers.
    for (int p = 0; p < pod_size; p++) {
      for (int ch = 0; ch < numChannels; ch++) {
        const int e = (baseEngine[p] + ch) % numPoolEngines;
        const int layer = (ch / numPoolEngines + p / numPoolEngines) % numPoolLayers;
        const int q = layer * numPoolEngines + e;
        const int idx = p * numChannels + ch;
        handles_h[idx] = pool_h[q];
        slotQueue_h[idx] = static_cast<uint8_t>(q);
        peMask_h[p] |= 1ULL << q;
      }
    }
  }

  CHECK_HIP(hipMalloc(&deviceHandles_d, numSlots * sizeof(*deviceHandles_d)));
  CHECK_HIP(hipMemcpy(deviceHandles_d, handles_h.data(), numSlots * sizeof(*deviceHandles_d),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMalloc(&slotQueue_d, numSlots * sizeof(*slotQueue_d)));
  CHECK_HIP(hipMemcpy(slotQueue_d, slotQueue_h.data(), numSlots * sizeof(*slotQueue_d),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMalloc(&peQueueMask_d, pod_size * sizeof(*peQueueMask_d)));
  CHECK_HIP(hipMemcpy(peQueueMask_d, peMask_h.data(), pod_size * sizeof(*peQueueMask_d),
                      hipMemcpyHostToDevice));
  CHECK_HIP(hipMalloc(&poolHandles_d, pool_h.size() * sizeof(*poolHandles_d)));
  CHECK_HIP(hipMemcpy(poolHandles_d, pool_h.data(), pool_h.size() * sizeof(*poolHandles_d),
                      hipMemcpyHostToDevice));
}

__host__ void SdmaImpl::sdmaReleasePool() {
  // Only this init's pool: the channel map is process-global, so disconnect() would also destroy
  // queues another user still holds.
  if (poolId != 0) sdma_anvil::anvil.disconnectDevice(poolId);
  poolId = 0;
  numPoolEngines = 0;
  numPoolLayers = 0;
}

__host__ void SdmaImpl::sdmaHostStop() {
  LOG_TRACE("SDMA stop");
  CHECK_HIP(hipFree(deviceHandles_d));
  CHECK_HIP(hipFree(slotQueue_d));
  CHECK_HIP(hipFree(peQueueMask_d));
  CHECK_HIP(hipFree(poolHandles_d));
  deviceHandles_d = nullptr;
  slotQueue_d = nullptr;
  peQueueMask_d = nullptr;
  poolHandles_d = nullptr;
  sdmaReleasePool();
}

#endif  // USE_SDMA

}  // namespace rocshmem
