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

#include "anvil.hpp"
#include "log.hpp"

#include <fstream>
#include <cstring>

#include "sdma_pkt_struct.h"
#include "sdma_pkt_struct_mi4.h"

namespace rocshmem {
namespace anvil {

#define CHECK_HSAKMT_SUCCESS(call, msg) do {                                  \
  if ((call) != HSAKMT_STATUS_SUCCESS)                                        \
    LOG_ERROR_EXIT("%s", #call);                                              \
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
    LOG_TRACE("Failure to get device type: %#x", status);
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

// True only after init() has run SetUpKFD. Avoids CloseKFD/hsa_shut_down at
// exit if init was never called (e.g. USE_SDMA not triggered).
static bool s_kfd_opened = false;

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

SdmaQueue::SdmaQueue([[maybe_unused]] int localDeviceId, int remoteDeviceId, hsa_agent_t& localAgent,
                     uint32_t engineId)
    : remoteDeviceId_(remoteDeviceId) {
  int originalDeviceId;

  ANVIL_CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId));  // Save the current device

  uint32_t localNodeId;
  hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);
  if (status != HSA_STATUS_SUCCESS) {
    LOG_TRACE("Failure to get device info: %#x", status);
  }

  // Allocate SDMA queue buffer on device side, requires ExecuteAccess
  HsaMemFlags memFlags = {};
  memFlags.ui32.NonPaged = 1;
  memFlags.ui32.HostAccess = 1;
  memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  memFlags.ui32.NoNUMABind = 1;
  memFlags.ui32.ExecuteAccess = 1;
  memFlags.ui32.Uncached = 1;

  LOG_TRACE("SDMA: Allocating Queue Buffer for device: %d remote device: %d engineId: %d",
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
      .maxWritePtr = (uint64_t)*(queue_.Queue_write_ptr_aql),
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

void SdmaQueue::dump(std::ofstream& logFile) {
  logFile << "Queue -> device " << remoteDeviceId_ << ": "
          << "wptr: " << *deviceHandle_->wptr << ", "
          << "rptr: " << *deviceHandle_->rptr << ", "
          << "doorbell: " << *deviceHandle_->doorbell << ", "
          << "queueBuf: " << deviceHandle_->queueBuf << ", "
          << "committedWptr: " << *deviceHandle_->committedWptr << ", "
          << "cachedWptr: " << *deviceHandle_->cachedWptr << std::endl;

  size_t dw_enqueued =
      std::min(*deviceHandle_->wptr, (uint64_t)SDMA_QUEUE_SIZE) / sizeof(uint32_t);
  uint32_t* dwPtr = deviceHandle_->queueBuf;
  uint64_t wrapped_rptr = *deviceHandle_->rptr % SDMA_QUEUE_SIZE;
  uint64_t wrapped_wptr = *deviceHandle_->wptr % SDMA_QUEUE_SIZE;

  logFile << "valid dw: " << dw_enqueued << "\nwrapped rptr: " << wrapped_rptr
          << " dw rptr: " << wrapped_rptr / sizeof(uint32_t) << "\nwrapped wptr: " << wrapped_wptr
          << " dw wptr: " << wrapped_wptr / sizeof(uint32_t) << std::endl;

  size_t it = 0;
  while (it < dw_enqueued) {
    logFile << "[" << it << "] ";
    uint32_t opcode = *dwPtr & 0xFF;
    uint32_t subop = (*dwPtr >> 8) & 0xFF;
    if (opcode == SDMA_OP_COPY) {
      if (subop == SDMA_SUBOP_COPY_LINEAR_WAIT_SIGNAL_MI4 &&
          sizeof(SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4) / sizeof(uint32_t) <= dw_enqueued - it) {
        auto* ptr = reinterpret_cast<SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4*>(dwPtr);
        logFile << "COPY_WAIT_SIGNAL_MI4 count=" << ptr->COPY_COUNT_UNION.copy_count
                << " wait=" << ptr->HEADER_UNION.wait
                << " signal=" << ptr->HEADER_UNION.signal
                << " src=0x" << std::hex
                << ((uint64_t)ptr->SRC_ADDR_HI_UNION.src_addr_63_32 << 32 |
                    ptr->SRC_ADDR_LO_UNION.src_addr_31_0)
                << " dst=0x"
                << ((uint64_t)ptr->DST_ADDR_HI_UNION.dst_addr_63_32 << 32 |
                    ptr->DST_ADDR_LO_UNION.dst_addr_31_0)
                << std::dec;
        constexpr size_t dw = sizeof(SDMA_PKT_COPY_LINEAR_WAIT_SIGNAL_MI4) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      } else {
        auto* ptr = reinterpret_cast<SDMA_PKT_COPY_LINEAR*>(dwPtr);
        logFile << "COPY count=" << ptr->COUNT_UNION.count
                << " src=0x" << std::hex
                << ((uint64_t)ptr->SRC_ADDR_HI_UNION.src_addr_63_32 << 32 |
                    ptr->SRC_ADDR_LO_UNION.src_addr_31_0)
                << " dst=0x"
                << ((uint64_t)ptr->DST_ADDR_HI_UNION.dst_addr_63_32 << 32 |
                    ptr->DST_ADDR_LO_UNION.dst_addr_31_0)
                << std::dec;
        size_t dw = sizeof(SDMA_PKT_COPY_LINEAR) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      }
    } else if (opcode == SDMA_OP_ATOMIC) {
      auto* ptr = reinterpret_cast<SDMA_PKT_ATOMIC*>(dwPtr);
      logFile << "ATOMIC op=" << ptr->HEADER_UNION.operation
              << " addr=0x" << std::hex
              << ((uint64_t)ptr->ADDR_HI_UNION.addr_63_32 << 32 |
                  ptr->ADDR_LO_UNION.addr_31_0)
              << std::dec;
      size_t dw = sizeof(SDMA_PKT_ATOMIC) / sizeof(uint32_t);
      it += dw;
      dwPtr += dw;
    } else if (opcode == SDMA_OP_FENCE) {
      if (subop == SDMA_SUBOP_FENCE_64B_MI4) {
        auto* ptr = reinterpret_cast<SDMA_PKT_FENCE_64B_MI4*>(dwPtr);
        logFile << "FENCE_64B_MI4"
                << " addr=0x" << std::hex
                << ((uint64_t)ptr->ADDR_HI_UNION.addr_63_32 << 32 |
                    (uint64_t)ptr->ADDR_LO_UNION.addr_31_3 << 3)
                << " data=0x"
                << ((uint64_t)ptr->DATA_HI_UNION.data_63_32 << 32 |
                    ptr->DATA_LO_UNION.data_31_0)
                << std::dec;
        constexpr size_t dw = sizeof(SDMA_PKT_FENCE_64B_MI4) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      } else if (subop == SDMA_SUBOP_FENCE_MI4) {
        auto* ptr = reinterpret_cast<SDMA_PKT_FENCE_MI4*>(dwPtr);
        logFile << "FENCE_MI4 data=" << ptr->DATA_UNION.data
                << " addr=0x" << std::hex
                << ((uint64_t)ptr->ADDR_HI_UNION.fence_addr_hi << 32 |
                    ptr->ADDR_LO_UNION.fence_addr_lo)
                << std::dec;
        constexpr size_t dw = sizeof(SDMA_PKT_FENCE_MI4) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      } else {
        auto* ptr = reinterpret_cast<SDMA_PKT_FENCE*>(dwPtr);
        logFile << "FENCE data=" << ptr->DATA_UNION.data
                << " addr=0x" << std::hex
                << ((uint64_t)ptr->ADDR_HI_UNION.addr_63_32 << 32 |
                    ptr->ADDR_LO_UNION.addr_31_0)
                << std::dec;
        size_t dw = sizeof(SDMA_PKT_FENCE) / sizeof(uint32_t);
        it += dw;
        dwPtr += dw;
      }
    } else {
      logFile << "RAW 0x" << std::hex << *dwPtr << std::dec;
      dwPtr++;
      it++;
    }
    logFile << "\n";
  }
}

AnvilLib::~AnvilLib() {
  for (auto& p : sdma_channels_) {
    p.second.clear();
  }
  if (s_kfd_opened) {
    CloseKFD();
    hsa_shut_down();
  }
}

void AnvilLib::init() {
  std::call_once(init_flag, []() {
    // HSA
    hsa_status_t status{hsa_init()};
    if (status != HSA_STATUS_SUCCESS) {
      LOG_TRACE("Failure to open HSA connection: %#x", status);
    }
    status = hsa_iterate_agents(&rocm_hsa_gpu_agent_callback, &gpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      LOG_TRACE("Failure to iterate HSA GPU agents: %#x", status);
    }
    status = hsa_iterate_agents(&rocm_hsa_cpu_agent_callback, &cpuAgents_);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      LOG_TRACE("Failure to iterate HSA CPU agents: %#x", status);
    }

    SetUpKFD();
    s_kfd_opened = true;
  });
}

SdmaQueue* AnvilLib::createSdmaQueue(int srcDeviceId, int dstDeviceId, uint32_t engineId,
                                     int* channelIdx) {
  auto& vec = sdma_channels_[dstDeviceId];
  vec.emplace_back(
      std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId, gpuAgents_[srcDeviceId], engineId));
  if (channelIdx != nullptr) {
    *channelIdx = static_cast<int>(vec.size() - 1);
  }
  return vec.back().get();
}

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels) {
  uint32_t engineId = getSdmaEngineId(srcDeviceId, dstDeviceId);
  LOG_TRACE("SDMA: Connect from %d to %d with %d channels using engine %d",
            srcDeviceId, dstDeviceId, numChannels, engineId);
  for (int c = 0; c < numChannels; ++c) {
    createSdmaQueue(srcDeviceId, dstDeviceId, engineId);
  }
  return true;
}

void AnvilLib::disconnect() {
  // Destroy all SDMA queues. SdmaQueue destructor calls hsaKmtDestroyQueue.
  sdma_channels_.clear();
  LOG_TRACE("SDMA: Disconnected all queues");
}

SdmaQueue* AnvilLib::getSdmaQueue([[maybe_unused]] int srcDeviceId, int dstDeviceId,
                                  int channel_idx) {
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

// Thin wrappers matching the rocm-xio sdma-ep API style.
// initEndpoint() is idempotent; shutdownEndpoint() only resets the flag,
// it does not destroy queues or shut down HSA/KFD (AnvilLib destructor does
// that at process exit).
bool initEndpoint() {
  try {
    anvil.init();
    return true;
  } catch (const std::exception& e) {
    LOG_WARN("anvil::initEndpoint: %s", e.what());
    return false;
  }
}

void shutdownEndpoint() {
  // no-op: HSA/KFD teardown happens in AnvilLib::~AnvilLib at process exit.
}

}  // namespace anvil
}  // namespace rocshmem
