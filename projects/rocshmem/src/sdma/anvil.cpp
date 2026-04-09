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
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "anvil.hpp"

#include <fstream>
#include <cstring>

namespace rocshmem {
namespace anvil {

auto checkHsaError = [](hsa_status_t s, const char* msg, const char* file, int line) {
  if (s != HSA_STATUS_SUCCESS) {
    const char* hsa_err_msg;
    hsa_status_string(s, &hsa_err_msg);
    throw(std::runtime_error{std::string("HSA error at ") + file + std::string(":") +
                             std::to_string(line) + std::string(" - ") + hsa_err_msg});
  }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

#define CHECK_HSAKMT_SUCCESS(call, msg)                                                  \
  do {                                                                                   \
    if ((call) != HSAKMT_STATUS_SUCCESS) {                                               \
      fprintf(stderr, "ERROR code: %d %s (File: %s, Line: %d)\n", call, msg, __FILE__,   \
              __LINE__);                                                                 \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  } while (0)

// HSA agents
std::vector<hsa_agent_t> cpuAgents_;
std::vector<hsa_agent_t> gpuAgents_;

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent, hsa_device_type_t target_device_type,
                                     [[maybe_unused]] void* vector) {
  std::vector<hsa_agent_t>* agents = static_cast<std::vector<hsa_agent_t>*>(vector);
  hsa_device_type_t device_type{};
  hsa_status_t status{hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "Failure to get device type: %#x\n", status);
    return status;
  }
  if (device_type == target_device_type) {
    agents->push_back(agent);
  }
  return status;
}

hsa_status_t rocm_hsa_gpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_GPU, context);
}

hsa_status_t rocm_hsa_cpu_agent_callback(hsa_agent_t agent, [[maybe_unused]] void* context) {
  return rocm_hsa_agent_callback(agent, HSA_DEVICE_TYPE_CPU, context);
}

void SetUpKFD() {
  CHECK_HSAKMT_SUCCESS(hsaKmtOpenKFD(), "hsaKmtOpenKFD() failed!");
  HsaSystemProperties m_SystemProperties;
  memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));
  CHECK_HSAKMT_SUCCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties), "Failed!");
}

void CloseKFD() { CHECK_HSAKMT_SUCCESS(hsaKmtCloseKFD(), "hsaKmtCloseKFD() failed"); }

// Convert a logical deviceId index to the NVML device minor number
static const std::string getBusId(int deviceId) {
  char busIdChar[] = "00000000:00:00.0";
  ANVIL_CHECK_HIP_ERROR(hipDeviceGetPCIBusId(busIdChar, sizeof(busIdChar), deviceId));
  // we need the hex in lower case format
  for (size_t i = 0; i < sizeof(busIdChar); i++) {
    busIdChar[i] = std::tolower(busIdChar[i]);
  }
  return std::string(busIdChar);
}

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent,
                     uint32_t engineId)
    : remoteDeviceId_(remoteDeviceId) {
  int originalDeviceId;

  ANVIL_CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId));  // Save the current device

  uint32_t localNodeId;
  hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);
  if (status != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "Failure to get device info: %#x\n", status);
  }

  // Allocate SDMA queue buffer on device side, requires ExecuteAccess
  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;

  fprintf(stdout, "SDMA: Allocating Queue Buffer for device: %d remote device: %d engineId: %d\n",
          localDeviceId, remoteDeviceId, engineId);

  CHECK_HSAKMT_SUCCESS(hsaKmtAllocMemory(localNodeId, SDMA_QUEUE_SIZE, memFlags, &queueBuffer_),
                       "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtMapMemoryToGPU(queueBuffer_, SDMA_QUEUE_SIZE, NULL), "Failed");

  // Create SDMA Queue
  memset(&queue_, 0, sizeof(HsaQueueResource));

  CHECK_HSAKMT_SUCCESS(hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID,
                                            DEFAULT_QUEUE_PERCENTAGE, DEFAULT_PRIORITY, engineId,
                                            queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_),
                       "hsaKmtCreateQueueExt failed");

  // Populate Device Handle
  ANVIL_CHECK_HIP_ERROR(hipMalloc(&deviceHandle_, sizeof(SdmaQueueDeviceHandle)));
  ANVIL_CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&cachedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));
  ANVIL_CHECK_HIP_ERROR(
      hipExtMallocWithFlags((void**)&committedWptr_, sizeof(uint64_t), hipDeviceMallocUncached));

  uint64_t cachedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  uint64_t committedWptr = (uint64_t)*(queue_.Queue_write_ptr_aql);
  SdmaQueueDeviceHandle handle = {
      .queueBuf = static_cast<uint32_t*>(queueBuffer_),
      .rptr = queue_.Queue_read_ptr_aql,
      .wptr = queue_.Queue_write_ptr_aql,
      .doorbell = queue_.Queue_DoorBell_aql,
      .cachedWptr = cachedWptr_,
      .committedWptr = committedWptr_,
      .cachedHwReadIndex = (uint64_t)*(queue_.Queue_read_ptr_aql),
  };

  ANVIL_CHECK_HIP_ERROR(
      hipMemcpy(deviceHandle_, &handle, sizeof(SdmaQueueDeviceHandle), hipMemcpyHostToDevice));
  ANVIL_CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
  ANVIL_CHECK_HIP_ERROR(
      hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue() {
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
  ANVIL_CHECK_HIP_ERROR(hipFree(deviceHandle_));
  ANVIL_CHECK_HIP_ERROR(hipFree(cachedWptr_));
  ANVIL_CHECK_HIP_ERROR(hipFree(committedWptr_));
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE), "Failed");
}

SdmaQueueDeviceHandle* SdmaQueue::deviceHandle() const { return deviceHandle_; }

AnvilLib::~AnvilLib() {
  for (auto& p : sdma_channels_) {
    p.second.clear();
  }
  CloseKFD();
  hsa_shut_down();
}

void AnvilLib::init() {
  std::call_once(init_flag, []() {
    // HSA
    hsa_status_t status{hsa_init()};
    if (status != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "Failure to open HSA connection: %#x\n", status);
    }
    status = hsa_iterate_agents(&rocm_hsa_gpu_agent_callback, &gpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      fprintf(stderr, "Failure to iterate HSA GPU agents: %#x\n", status);
    }
    status = hsa_iterate_agents(&rocm_hsa_cpu_agent_callback, &cpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      fprintf(stderr, "Failure to iterate HSA CPU agents: %#x\n", status);
    }

    SetUpKFD();
  });
}

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels) {
  uint32_t engineId = getSdmaEngineId(srcDeviceId, dstDeviceId);
  fprintf(stdout, "SDMA: Connect from %d to %d with %d channels using engine %d\n",
          srcDeviceId, dstDeviceId, numChannels, engineId);
  for (int c = 0; c < numChannels; ++c) {
    sdma_channels_[dstDeviceId].emplace_back(
        std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId, gpuAgents_[srcDeviceId], engineId));
  }
  return true;
}

SdmaQueue* AnvilLib::getSdmaQueue(int srcDeviceId, int dstDeviceId, int channel_idx) {
  if (sdma_channels_.find(dstDeviceId) == sdma_channels_.end()) {
    return nullptr;
  }

  if (!(channel_idx < static_cast<int>(sdma_channels_[dstDeviceId].size()))) {
    return nullptr;
  }

  return sdma_channels_[dstDeviceId][channel_idx].get();
}

AnvilLib& AnvilLib::getInstance() {
  static AnvilLib* instance;
  if (instance == nullptr) {
    instance = new AnvilLib();
  }
  return *instance;
}

int AnvilLib::getOamId(int deviceId) {
  std::string busId = getBusId(deviceId);
  std::string file_str = "/sys/bus/pci/devices/" + busId + "/xgmi_physical_id";
  std::ifstream file(file_str);
  int xgmi_physical_id;
  if (file.is_open()) {
    if (!(file >> xgmi_physical_id)) {
      throw std::runtime_error("Failed to read xGMI physical id from file: " + file_str);
    }
  } else {
    throw std::runtime_error("Failed to open file: " + file_str);
  }
  return xgmi_physical_id;
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId) {
  int srcOamId = getOamId(srcDeviceId);
  int dstOamId = getOamId(dstDeviceId);

  // Use even engines only
  return mi300xOamMap[srcOamId][dstOamId] * 2;
}

AnvilLib& anvil = anvil.getInstance();

}  // namespace anvil
}  // namespace rocshmem
