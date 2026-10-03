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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David, AMD RAD
 */
#ifndef LIBRARY_SRC_SDMA_ANVIL_HPP_
#define LIBRARY_SRC_SDMA_ANVIL_HPP_

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "anvil_device.hpp"
#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"

namespace sdma_anvil {

// True when the doubled OAM-map id cannot be used as-is. A partition (no xGMI
// engines) folds even when the doubled id is in range: same-device peers share
// the map diagonal and would otherwise all land on engine 0. An id past the
// engines this node reports folds too. numSdmaEnginesTotal == 0 never folds,
// so the modulo below is not asked to divide by zero.
inline bool oamMapEngineNeedsFold(uint32_t numSdmaXgmiEngines, uint32_t numSdmaEnginesTotal,
                                 int doubledEngineId) {
  const bool partition = numSdmaXgmiEngines == 0 && numSdmaEnginesTotal > 0;
  const bool outOfRange =
      numSdmaEnginesTotal > 0 && static_cast<uint32_t>(doubledEngineId) >= numSdmaEnginesTotal;
  return partition || outOfRange;
}

// Fold the undoubled OAM-map value and both PCI functions into the engines this
// node reports. numEngines must be > 0. An unreadable function (-1) counts as 0
// and therefore collides with function 0.
inline int foldOamMapEngine(int oamEngine, int srcFn, int dstFn, uint32_t numEngines) {
  const int src = srcFn < 0 ? 0 : srcFn;
  const int dst = dstFn < 0 ? 0 : dstFn;
  return (oamEngine + src + dst) % static_cast<int>(numEngines);
}

// How an engine was chosen for one peer, and the values that explain the choice. The two failure
// logs that report it -- the queue-create error and the missing-xGMI-id error -- need the same
// fields, so they travel as one record: a second copy means the next diagnostic field has to be
// remembered in two places.
struct EngineSelection {
  int engineId{-1};  // -1 when no engine could be mapped for this pair
  int srcDeviceId{-1};
  int dstDeviceId{-1};
  // Result of hsa_amd_memory_get_preferred_copy_engine. queried stays false when the agents were
  // not valid enough to ask, which is different from asking and being refused.
  hsa_status_t preferredStatus{HSA_STATUS_ERROR};
  uint32_t preferredMask{0};
  bool preferredQueried{false};
  bool usedPreferred{false};
  // Engine counts this node reports, which bound every id above.
  uint32_t numSdmaEngines{0};
  uint32_t numSdmaXgmiEngines{0};
  uint32_t numSdmaEnginesTotal{0};
};

class SdmaQueue {
 public:
  SdmaQueue(int localDeviceId, int remoteDeviceId, const hsa_agent_t& localAgent, uint32_t engineId,
            const EngineSelection& selection);
  ~SdmaQueue();

  SdmaQueueDeviceHandle* deviceHandle() const;
  SdmaQueueSingleProducerDeviceHandle* singleProducerDeviceHandle() const;
  void dump(std::ofstream& logFile);
  // False when the constructor could not create an SDMA queue. The constructor releases whatever it
  // acquired before giving up, so an invalid queue must be dropped without being used or destroyed
  // any further.
  bool valid() const;
  // Status of the queue-creation attempt, so the caller can tell an exhausted queue budget
  // (NO_MEMORY) apart from other failures.
  HSAKMT_STATUS createStatus() const;

 private:
  bool valid_{false};
  HSAKMT_STATUS createStatus_{HSAKMT_STATUS_ERROR};
  int remoteDeviceId_{-1};
  uint64_t* cachedWptr_{nullptr};
  uint64_t* committedWptr_{nullptr};
  void* queueBuffer_{nullptr};
  HsaQueueResource queue_{};
  SdmaQueueDeviceHandle* deviceHandle_{nullptr};
  SdmaQueueSingleProducerDeviceHandle* singleProducerDeviceHandle_{nullptr};
};

class AnvilLib {
 private:
  // Make constructor private
  AnvilLib() = default;

 public:
  ~AnvilLib();
  // access to singleton
  static AnvilLib& getInstance();

  AnvilLib(const AnvilLib&) = delete;
  AnvilLib& operator=(const AnvilLib&) = delete;

 public:
  void init();
  bool connect(int srcDeviceId, int dstDeviceId, int numChannels = 1);
  void disconnect();
  SdmaQueue* getSdmaQueue(int srcDeviceId, int dstDeviceId, int channel_idx = 0);
  SdmaQueue* createSdmaQueue(int srcDeviceId, int dstDeviceId, uint32_t engineId,
                             const EngineSelection& selection, int* channelIdx = nullptr);

 private:
  /*
   * OAM MAP
   * src\dst    0  1 2 3 4 5 6 7
   * 0         0 7 6 1 2 4 5 3
   * 1         7 0 1 5 4 2 3 6
   * 2         5 1 0 6 7 3 2 4
   * 3         1 6 5 0 3 7 4 2
   * 4         2 4 7 3 0 5 6 1
   * 5         4 2 3 7 6 0 1 5
   * 6         5 3 2 4 6 1 0 7
   * 7         3 6 4 2 1 5 7 0
   */
  static constexpr size_t kMi300xOamMapDim = 8;
  std::array<std::array<int, kMi300xOamMapDim>, kMi300xOamMapDim> mi300xOamMap = {{
      {0, 7, 6, 1, 2, 4, 5, 3},
      {7, 0, 1, 5, 4, 2, 3, 6},
      {5, 1, 0, 6, 7, 3, 2, 4},
      {1, 6, 5, 0, 3, 7, 4, 2},
      {2, 4, 7, 3, 0, 5, 6, 1},
      {4, 2, 3, 7, 6, 0, 1, 5},
      {5, 3, 2, 4, 6, 1, 0, 7},
      {3, 6, 4, 2, 1, 5, 7, 0}}};

  uint32_t numSdmaEngines_{0};
  uint32_t numSdmaXgmiEngines_{0};
  uint32_t numSdmaEnginesTotal_{0};
  // KFD caps user SDMA queues per engine, so a partition with few engines also has a small total
  // queue budget. Track usage to refuse a mesh that cannot fit before KFD returns NO_MEMORY part
  // way through building it.
  uint32_t numSdmaQueuesPerEngine_{0};
  // Queues already taken by this process, counted across all engines rather than per engine: a
  // rejected engine-pinned create retries as a generic queue and reports engine 0, so on a
  // partition every queue would charge the same key and a per-engine cap would refuse at a
  // fraction of the real budget. connect() compares this against
  // numSdmaEnginesTotal_ * numSdmaQueuesPerEngine_.
  uint32_t queuesUsedTotal_{0};
  HSAKMT_STATUS lastQueueStatus_{HSAKMT_STATUS_SUCCESS};
  // Selection in progress, so getOamId can report why the map was consulted without the caller
  // threading the same values back down.
  EngineSelection selection_;

  void buildGpuAgentMap();
  hsa_agent_t getHipGpuAgent(int hipDeviceId) const;
  void querySdmaEngineCounts();
  int getOamId(int deviceId);
  int getSdmaEngineIdFromOamMap(int srcDeviceId, int dstDeviceId);
  // Chooses an engine for the pair and records why, so the caller can hand the same record to
  // every queue it then creates.
  EngineSelection getSdmaEngineId(int srcDeviceId, int dstDeviceId);

  std::once_flag init_flag;
  std::vector<hsa_agent_t> gpuAgentsByHipDev_;
  std::unordered_map<int, std::vector<std::unique_ptr<SdmaQueue>>> sdma_channels_;
};

extern AnvilLib& anvil;

// Initialize the Anvil subsystem (HSA + KFD). Idempotent.
bool initEndpoint();
// Mark the subsystem inactive. Does not destroy queues or shut down HSA/KFD.
void shutdownEndpoint();

inline void checkHipError(hipError_t err, const char* msg, const char* file, int line) {
  if (err != hipSuccess) {
    std::cerr << "HIP error at " << file << ":" << line << " - " << msg << "\n"
              << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define ANVIL_CHECK_HIP_ERROR(cmd) sdma_anvil::checkHipError((cmd), #cmd, __FILE__, __LINE__)

// Allow access to peerDeviceId from deviceId
inline void EnablePeerAccess(int const deviceId, int const peerDeviceId) {
  int canAccess;
  ANVIL_CHECK_HIP_ERROR(hipDeviceCanAccessPeer(&canAccess, deviceId, peerDeviceId));
  if (!canAccess) {
    std::cerr << "Unable to enable peer access from GPU devices " << deviceId << " to "
              << peerDeviceId << "\n";
  }

  ANVIL_CHECK_HIP_ERROR(hipSetDevice(deviceId));
  hipError_t error = hipDeviceEnablePeerAccess(peerDeviceId, 0);
  if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled) {
    std::cerr << "Unable to enable peer to peer access from " << deviceId << "  to " << peerDeviceId
              << " (" << hipGetErrorString(error) << ")\n";
  }
}


}  // namespace sdma_anvil

#endif  // LIBRARY_SRC_SDMA_ANVIL_HPP_
