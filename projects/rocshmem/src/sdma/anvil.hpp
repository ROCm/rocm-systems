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

namespace rocshmem {
namespace anvil {

class SdmaQueue {
 public:
  SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent, uint32_t engineId);
  ~SdmaQueue();

  SdmaQueueDeviceHandle* deviceHandle() const;
  void dump(std::ofstream& logFile);

 private:
  int remoteDeviceId_;
  uint64_t* cachedWptr_;
  uint64_t* committedWptr_;
  void* queueBuffer_;
  HsaQueueResource queue_;
  SdmaQueueDeviceHandle* deviceHandle_;
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
                             int* channelIdx = nullptr);

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
  std::array<std::array<int, 8>, 8> mi300xOamMap = {{{0, 7, 6, 1, 2, 4, 5, 3},
                                                     {7, 0, 1, 5, 4, 2, 3, 6},
                                                     {5, 1, 0, 6, 7, 3, 2, 4},
                                                     {1, 6, 5, 0, 3, 7, 4, 2},
                                                     {2, 4, 7, 3, 0, 5, 6, 1},
                                                     {4, 2, 3, 7, 6, 0, 1, 5},
                                                     {5, 3, 2, 4, 6, 1, 0, 7},
                                                     {3, 6, 4, 2, 1, 5, 7, 0}}};

  int getOamId(int deviceId);

  int getSdmaEngineId(int srcDeviceId, int dstDeviceId);

  std::once_flag init_flag;
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

#define ANVIL_CHECK_HIP_ERROR(cmd) rocshmem::anvil::checkHipError((cmd), #cmd, __FILE__, __LINE__)

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

}  // namespace anvil
}  // namespace rocshmem

#endif  // LIBRARY_SRC_SDMA_ANVIL_HPP_
