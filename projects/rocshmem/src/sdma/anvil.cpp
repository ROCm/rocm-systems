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
#include <cstdio>
#include <stdexcept>
#include <cctype>

#include "sdma_pkt_struct.h"
#include "sdma_pkt_struct_mi4.h"

namespace sdma_anvil {


#define CHECK_HSAKMT_SUCCESS(call, msg) do {                                  \
  if ((call) != HSAKMT_STATUS_SUCCESS)                                        \
    LOG_ERROR_EXIT("%s", #call);                                              \
} while (0)

// HSA agents discovered via hsa_iterate_agents (unordered).
std::vector<hsa_agent_t> cpuAgents_;
std::vector<hsa_agent_t> gpuAgents_;

static bool hsaAgentIsValid(const hsa_agent_t& agent) { return agent.handle != 0; }

static std::string hsaAgentBusId(const hsa_agent_t& agent) {
  uint32_t domain = 0;
  uint32_t bdfid = 0;
  if (hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN), &domain) !=
      HSA_STATUS_SUCCESS) {
    return {};
  }
  if (hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdfid) !=
      HSA_STATUS_SUCCESS) {
    return {};
  }
  const unsigned bus = (bdfid >> 8) & 0xff;
  const unsigned dev = (bdfid >> 3) & 0x1f;
  const unsigned fn = bdfid & 0x7;
  char busId[32];
  std::snprintf(busId, sizeof(busId), "%04x:%02x:%02x.%x", domain, bus, dev, fn);
  for (char* p = busId; *p != '\0'; ++p) {
    *p = static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
  }
  return std::string(busId);
}

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

SdmaQueue::SdmaQueue([[maybe_unused]] int localDeviceId, int remoteDeviceId,
                     const hsa_agent_t& localAgent, uint32_t engineId)
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

  ANVIL_CHECK_HIP_ERROR(hipMalloc(&singleProducerDeviceHandle_,
                                  sizeof(SdmaQueueSingleProducerDeviceHandle)));
  ANVIL_CHECK_HIP_ERROR(hipMemcpy(singleProducerDeviceHandle_, &handle,
                                  sizeof(SdmaQueueSingleProducerDeviceHandle),
                                  hipMemcpyHostToDevice));

  ANVIL_CHECK_HIP_ERROR(hipMemcpy(cachedWptr_, &cachedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
  ANVIL_CHECK_HIP_ERROR(
      hipMemcpy(committedWptr_, &committedWptr, sizeof(uint64_t), hipMemcpyHostToDevice));
}

SdmaQueue::~SdmaQueue() {
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
  ANVIL_CHECK_HIP_ERROR(hipFree(deviceHandle_));
  if (singleProducerDeviceHandle_) ANVIL_CHECK_HIP_ERROR(hipFree(singleProducerDeviceHandle_));
  ANVIL_CHECK_HIP_ERROR(hipFree(cachedWptr_));
  ANVIL_CHECK_HIP_ERROR(hipFree(committedWptr_));
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE), "Failed");
}

SdmaQueueDeviceHandle* SdmaQueue::deviceHandle() const { return deviceHandle_; }

SdmaQueueSingleProducerDeviceHandle* SdmaQueue::singleProducerDeviceHandle() const {
  return singleProducerDeviceHandle_;
}

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

void AnvilLib::buildGpuAgentMap() {
  int hipCount = 0;
  ANVIL_CHECK_HIP_ERROR(hipGetDeviceCount(&hipCount));
  gpuAgentsByHipDev_.assign(static_cast<size_t>(hipCount), hsa_agent_t{});

  for (const hsa_agent_t& agent : gpuAgents_) {
    const std::string agentBusId = hsaAgentBusId(agent);
    if (agentBusId.empty()) continue;

    for (int hipDev = 0; hipDev < hipCount; ++hipDev) {
      if (hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(hipDev)])) continue;
      if (getBusId(hipDev) != agentBusId) continue;
      gpuAgentsByHipDev_[static_cast<size_t>(hipDev)] = agent;
      LOG_TRACE("anvil: HIP device %d -> HSA agent (bus %s)", hipDev, agentBusId.c_str());
      break;
    }
  }

  for (int hipDev = 0; hipDev < hipCount; ++hipDev) {
    if (!hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(hipDev)])) {
      LOG_WARN("anvil: no HSA GPU agent for HIP device %d (bus %s)", hipDev,
               getBusId(hipDev).c_str());
    }
  }
}

hsa_agent_t AnvilLib::getHipGpuAgent(int hipDeviceId) const {
  if (hipDeviceId < 0 || hipDeviceId >= static_cast<int>(gpuAgentsByHipDev_.size()) ||
      !hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(hipDeviceId)])) {
    LOG_ERROR_EXIT("anvil: no HSA agent mapped for HIP device %d", hipDeviceId);
  }
  return gpuAgentsByHipDev_[static_cast<size_t>(hipDeviceId)];
}

void AnvilLib::querySdmaEngineCounts() {
  hsa_agent_t agent{};
  for (const hsa_agent_t& hipAgent : gpuAgentsByHipDev_) {
    if (hsaAgentIsValid(hipAgent)) {
      agent = hipAgent;
      break;
    }
  }
  if (!hsaAgentIsValid(agent)) {
    LOG_WARN("anvil: no mapped HIP GPU agents; SDMA engine count unknown");
    return;
  }
  hsa_status_t status = hsa_agent_get_info(
      agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SDMA_ENG), &numSdmaEngines_);
  if (status != HSA_STATUS_SUCCESS) {
    LOG_WARN("anvil: HSA_AMD_AGENT_INFO_NUM_SDMA_ENG query failed: %#x", status);
    numSdmaEngines_ = 0;
  }

  status = hsa_agent_get_info(
      agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG),
      &numSdmaXgmiEngines_);
  if (status != HSA_STATUS_SUCCESS) {
    LOG_WARN("anvil: HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG query failed: %#x", status);
    numSdmaXgmiEngines_ = 0;
  }

  numSdmaEnginesTotal_ = numSdmaEngines_ + numSdmaXgmiEngines_;
  LOG_TRACE("anvil: SDMA engines host=%u xgmi=%u total=%u", numSdmaEngines_, numSdmaXgmiEngines_,
            numSdmaEnginesTotal_);
}

void AnvilLib::init() {
  std::call_once(init_flag, [this]() {
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

    buildGpuAgentMap();
    querySdmaEngineCounts();

    SetUpKFD();
    s_kfd_opened = true;
  });
}

SdmaQueue* AnvilLib::createSdmaQueue(int srcDeviceId, int dstDeviceId, uint32_t engineId,
                                     int* channelIdx) {
  auto& vec = sdma_channels_[dstDeviceId];
  vec.emplace_back(std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId,
                                               getHipGpuAgent(srcDeviceId), engineId));
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
      LOG_ERROR_EXIT("anvil: failed to read xGMI physical id from %s", file_str.c_str());
    }
  } else {
    LOG_ERROR_EXIT("anvil: failed to open file: %s", file_str.c_str());
  }
  return xgmi_physical_id;
}

int AnvilLib::getSdmaEngineIdFromOamMap(int srcDeviceId, int dstDeviceId) {
  auto normalizeOamId = [](int oamId) {
    if (oamId < 0) return 0;
    return oamId % static_cast<int>(AnvilLib::kMi300xOamMapDim);
  };

  const int srcOamId = normalizeOamId(getOamId(srcDeviceId));
  const int dstOamId = normalizeOamId(getOamId(dstDeviceId));

  // Use even engines only (MI300X xGMI SDMA layout).
  int engineId =
      mi300xOamMap[static_cast<size_t>(srcOamId)][static_cast<size_t>(dstOamId)] * 2;

  if (numSdmaEnginesTotal_ > 0 &&
      static_cast<uint32_t>(engineId) >= numSdmaEnginesTotal_) {
    LOG_WARN("anvil: legacy OAM-map engine %d >= total %u", engineId, numSdmaEnginesTotal_);
  }
  return engineId;
}

int AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId) {
  if (srcDeviceId >= 0 && dstDeviceId >= 0 &&
      srcDeviceId < static_cast<int>(gpuAgentsByHipDev_.size()) &&
      dstDeviceId < static_cast<int>(gpuAgentsByHipDev_.size()) &&
      hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(srcDeviceId)]) &&
      hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(dstDeviceId)])) {
    uint32_t engineMask = 0;
    const hsa_agent_t srcAgent = gpuAgentsByHipDev_[static_cast<size_t>(srcDeviceId)];
    const hsa_agent_t dstAgent = gpuAgentsByHipDev_[static_cast<size_t>(dstDeviceId)];
    const hsa_status_t status = hsa_amd_memory_get_preferred_copy_engine(
        dstAgent, srcAgent, &engineMask);
    if (status == HSA_STATUS_SUCCESS && engineMask != 0) {
      const int engineId = __builtin_ctz(engineMask);
      if (engineId >= 0 && (numSdmaEnginesTotal_ == 0 ||
                            static_cast<uint32_t>(engineId) < numSdmaEnginesTotal_)) {
        LOG_TRACE("SDMA: HSA preferred engine %d for %d -> %d (mask=0x%x)", engineId, srcDeviceId,
                  dstDeviceId, engineMask);
        return engineId;
      }
      LOG_WARN("anvil: HSA preferred engine %d out of range (total=%u), using OAM map", engineId,
               numSdmaEnginesTotal_);
    }
  }

  return getSdmaEngineIdFromOamMap(srcDeviceId, dstDeviceId);
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


}  // namespace sdma_anvil
