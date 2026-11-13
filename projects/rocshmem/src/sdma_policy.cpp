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

#include <unistd.h>

#include <cstdint>
#include <unordered_map>
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

namespace {

// What every pod rank publishes about itself: the HIP device it runs on and a hash of its host.
struct SdmaPeerInfo {
  int32_t deviceId;
  int32_t reserved;
  uint64_t hostId;
};
static_assert(sizeof(SdmaPeerInfo) == 16, "exchanged as raw bytes");

// FNV-1a of the hostname. Never 0, so 0 can mean "unknown".
uint64_t sdmaHostId() {
  char name[256] = {};
  gethostname(name, sizeof(name) - 1);
  uint64_t h = 1469598103934665603ull;
  for (const char* c = name; *c != '\0'; ++c) {
    h ^= static_cast<unsigned char>(*c);
    h *= 1099511628211ull;
  }
  return h != 0 ? h : 1;
}

// Names a GPU on another host for Anvil's remote-peer queues, as the RCCL GIN factory does. A
// collision would only make two such peers share a queue.
uint64_t sdmaRemotePeerKey(const SdmaPeerInfo& p) {
  return p.hostId ^ ((static_cast<uint64_t>(p.deviceId) + 1) * 0x9E3779B97F4A7C15ull);
}

// Host-side state of one sdmaHostInit, kept out of SdmaImpl so it stays trivially
// device-constructible. Keyed by SdmaImpl::wiringId.
struct SdmaWiring {
  std::vector<SdmaPeerInfo> peers;           // by pod rank
  std::vector<int> connectedDevs;            // local devices this init connected
  std::vector<uint64_t> connectedRemotes;    // remote-peer keys this init connected
};

std::unordered_map<int, SdmaWiring>& sdmaWirings() {
  static std::unordered_map<int, SdmaWiring> wirings;
  return wirings;
}

int sdmaNextWiringId = 0;

}  // namespace

__host__ void SdmaImpl::sdmaHostInit(int pe, int pod_sz, int prank,
                                     const PodAllGather& allGather) {
  my_pe = pe;
  shm_size = pod_sz;
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

  // Every rank joins the exchange, enabled or not: it is collective over the pod.
  wiringId = ++sdmaNextWiringId;
  SdmaWiring& wiring = sdmaWirings()[wiringId];
  wiring.peers.assign(static_cast<size_t>(shm_size), SdmaPeerInfo{-1, 0, 0});
  wiring.peers[static_cast<size_t>(pod_rank)] = SdmaPeerInfo{deviceId, 0, sdmaHostId()};
  allGather(wiring.peers.data(), static_cast<int>(sizeof(SdmaPeerInfo)));

  const uint64_t myHost = wiring.peers[static_cast<size_t>(pod_rank)].hostId;
  local_size = 0;
  local_rank = 0;
  for (int p = 0; p < shm_size; p++) {
    if (wiring.peers[static_cast<size_t>(p)].hostId != myHost) continue;
    if (p < pod_rank) local_rank++;
    local_size++;
  }
  const int remotePeers = shm_size - local_size;

  uint32_t queuesBefore = 0;
  int status = kSdmaDisabledByEnv;
  if (sdmaEnabled) {
    LOG_INFO("SDMA init with threshold=%zu, channels=%d, pod_size=%d (%d on this host)",
             sdmaThreshold, numChannels, shm_size, local_size);

    // Initialize the Anvil library
    sdma_anvil::anvil.init();
    queuesBefore = sdma_anvil::anvil.queuesUsed();
    status = kSdmaReady;

    int numDevices = 0;
    CHECK_HIP(hipGetDeviceCount(&numDevices));

    // Connect to every pod peer. A peer on this host uses the per-device queues; a peer on another
    // host gets queues of its own and no peer access, because the destination is its fabric-mapped
    // memory. Queues a previous init left behind are reused. A failure is not acted on here; the
    // pod agrees on the result below.
    for (int p = 0; p < shm_size && status == kSdmaReady; p++) {
      const SdmaPeerInfo& peer = wiring.peers[static_cast<size_t>(p)];
      if (peer.hostId == myHost) {
        const int dev = peer.deviceId;
        if (dev < 0 || dev >= numDevices) {
          LOG_ERROR("SDMA: pod rank %d reports device %d, not one of this host's %d", p, dev,
                    numDevices);
          status = kSdmaConnectFailed;
          break;
        }
        if (sdma_anvil::anvil.getSdmaQueue(deviceId, dev, numChannels - 1) != nullptr) continue;
        if (dev != deviceId) sdma_anvil::EnablePeerAccess(deviceId, dev);
        if (!sdma_anvil::anvil.connect(deviceId, dev, numChannels)) {
          LOG_ERROR("SDMA: connect failed from device %d to %d with %d channel(s)", deviceId,
                    dev, numChannels);
          status = kSdmaConnectFailed;
          break;
        }
        wiring.connectedDevs.push_back(dev);
      } else {
        const uint64_t key = sdmaRemotePeerKey(peer);
        if (sdma_anvil::anvil.getRemotePeerQueue(key, numChannels - 1) != nullptr) continue;
        if (!sdma_anvil::anvil.connectRemotePeer(deviceId, key, numChannels)) {
          LOG_ERROR("SDMA: connect failed from device %d to pod rank %d on another host with %d "
                    "channel(s)", deviceId, p, numChannels);
          status = kSdmaConnectFailed;
          break;
        }
        wiring.connectedRemotes.push_back(key);
      }
    }
    if (status == kSdmaReady && remotePeers > 0) {
      LOG_INFO("SDMA: %d of %d pod peers are on other hosts; each gets its own SDMA queue(s)",
               remotePeers, shm_size);
    }
  }

  std::vector<int> statuses(shm_size, kSdmaDisabledByEnv);
  statuses[pod_rank] = applyForcedFailure(status, local_rank, forcedFailureLocalRank());
  if (statuses[pod_rank] != status) {
    LOG_ERROR("SDMA: local rank %d reports a failed connect, forced by "
              "ROCSHMEM_SDMA_FAIL_CONNECT_LOCAL_RANK", local_rank);
  }
  allGather(statuses.data(), static_cast<int>(sizeof(int)));
  const SdmaAgreement agreed = agreeSdmaStatus(statuses.data(), shm_size);

  switch (agreed.decision) {
    case SdmaDecision::kCommit:
      sdmaPublishHandles(deviceId, true);
      return;
    case SdmaDecision::kAllDisabled:
      LOG_INFO("SDMA disabled at runtime (ROCSHMEM_SDMA_ENABLED=0)");
      // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
      sdmaPublishHandles(deviceId, false);
      return;
    case SdmaDecision::kFallbackConnectFailed:
      LOG_ERROR("SDMA: connect failed on pod rank(s) %s of %d; all of them fall back to IPC "
                "memcpy. Lower ROCSHMEM_SDMA_NUM_CHANNELS or use fewer ranks per node to keep "
                "SDMA", agreed.ranks.c_str(), shm_size);
      break;
    case SdmaDecision::kFallbackEnvMismatch:
      LOG_ERROR("SDMA: ROCSHMEM_SDMA_ENABLED=0 on pod rank(s) %s of %d only; all of them fall "
                "back to IPC memcpy. Set it the same on every rank", agreed.ranks.c_str(),
                shm_size);
      break;
  }

  const size_t released = wiring.connectedDevs.size() + wiring.connectedRemotes.size();
  sdmaReleasePeers();
  if (status != kSdmaDisabledByEnv) {
    const uint32_t queuesAfter = sdma_anvil::anvil.queuesUsed();
    if (queuesAfter != queuesBefore) {
      LOG_WARN("SDMA: %u queue(s) held after the fallback, %u before init", queuesAfter,
               queuesBefore);
    } else {
      LOG_INFO("SDMA: fallback released %zu peer(s), %u queue(s) held", released, queuesAfter);
    }
  }
  sdmaEnabled = false;
  // USE_SDMA testers index deviceHandles_d without checking sdmaEnabled.
  sdmaPublishHandles(deviceId, false);
}

__host__ void SdmaImpl::sdmaPublishHandles(int deviceId, bool useQueues) {
  // Total number of handles: shm_size (pod) * numChannels
  // Indexed as: deviceHandles_d[local_pe * numChannels + channel_idx]
  int total_handles = shm_size * numChannels;

  // Allocate device-side array to hold SDMA queue device handles
  CHECK_HIP(hipMalloc(&deviceHandles_d,
                      total_handles * sizeof(sdma_anvil::SdmaQueueDeviceHandle*)));

  // Copy device handles to device memory
  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> handles_h(total_handles, nullptr);
  if (useQueues) {
    const SdmaWiring& wiring = sdmaWirings()[wiringId];
    const uint64_t myHost = wiring.peers[static_cast<size_t>(pod_rank)].hostId;
    for (int i = 0; i < shm_size; i++) {
      const SdmaPeerInfo& peer = wiring.peers[static_cast<size_t>(i)];
      const bool local = peer.hostId == myHost;
      for (int ch = 0; ch < numChannels; ch++) {
        int idx = i * numChannels + ch;
        sdma_anvil::SdmaQueue* queue =
            local ? sdma_anvil::anvil.getSdmaQueue(deviceId, peer.deviceId, ch)
                  : sdma_anvil::anvil.getRemotePeerQueue(sdmaRemotePeerKey(peer), ch);
        handles_h[idx] = queue ? queue->deviceHandle() : nullptr;
      }
    }
  }
  CHECK_HIP(hipMemcpy(deviceHandles_d, handles_h.data(),
                      total_handles * sizeof(sdma_anvil::SdmaQueueDeviceHandle*),
                      hipMemcpyHostToDevice));
}

__host__ void SdmaImpl::sdmaReleasePeers() {
  // Release only the queues this init connected: the channel maps are process-global, so
  // disconnect() would also destroy queues another user still holds.
  auto it = sdmaWirings().find(wiringId);
  if (it == sdmaWirings().end()) return;
  for (int dev : it->second.connectedDevs) sdma_anvil::anvil.disconnectDevice(dev);
  for (uint64_t key : it->second.connectedRemotes) sdma_anvil::anvil.disconnectRemotePeer(key);
  it->second.connectedDevs.clear();
  it->second.connectedRemotes.clear();
}

__host__ void SdmaImpl::sdmaHostStop() {
  LOG_TRACE("SDMA stop");
  if (deviceHandles_d != nullptr) {
    CHECK_HIP(hipFree(deviceHandles_d));
    deviceHandles_d = nullptr;
  }
  sdmaReleasePeers();
  sdmaWirings().erase(wiringId);
  wiringId = 0;
}

#endif  // USE_SDMA

}  // namespace rocshmem
