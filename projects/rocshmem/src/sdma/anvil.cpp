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

static const char* hsakmtStatusName(HSAKMT_STATUS status) {
  switch (status) {
    case HSAKMT_STATUS_SUCCESS:
      return "SUCCESS";
    case HSAKMT_STATUS_ERROR:
      return "ERROR";
    case HSAKMT_STATUS_DRIVER_MISMATCH:
      return "DRIVER_MISMATCH";
    case HSAKMT_STATUS_INVALID_PARAMETER:
      return "INVALID_PARAMETER";
    case HSAKMT_STATUS_INVALID_HANDLE:
      return "INVALID_HANDLE";
    case HSAKMT_STATUS_INVALID_NODE_UNIT:
      return "INVALID_NODE_UNIT";
    case HSAKMT_STATUS_NO_MEMORY:
      return "NO_MEMORY";
    case HSAKMT_STATUS_BUFFER_TOO_SMALL:
      return "BUFFER_TOO_SMALL";
    case HSAKMT_STATUS_NOT_IMPLEMENTED:
      return "NOT_IMPLEMENTED";
    case HSAKMT_STATUS_NOT_SUPPORTED:
      return "NOT_SUPPORTED";
    case HSAKMT_STATUS_UNAVAILABLE:
      return "UNAVAILABLE";
    case HSAKMT_STATUS_OUT_OF_RESOURCES:
      return "OUT_OF_RESOURCES";
    case HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED:
      return "KERNEL_IO_CHANNEL_NOT_OPENED";
    case HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR:
      return "KERNEL_COMMUNICATION_ERROR";
    case HSAKMT_STATUS_KERNEL_ALREADY_OPENED:
      return "KERNEL_ALREADY_OPENED";
    case HSAKMT_STATUS_HSAMMU_UNAVAILABLE:
      return "HSAMMU_UNAVAILABLE";
    case HSAKMT_STATUS_WAIT_FAILURE:
      return "WAIT_FAILURE";
    case HSAKMT_STATUS_WAIT_TIMEOUT:
      return "WAIT_TIMEOUT";
    case HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED:
      return "MEMORY_ALREADY_REGISTERED";
    case HSAKMT_STATUS_MEMORY_NOT_REGISTERED:
      return "MEMORY_NOT_REGISTERED";
    case HSAKMT_STATUS_MEMORY_ALIGNMENT:
      return "MEMORY_ALIGNMENT";
    default:
      return "UNKNOWN";
  }
}

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

SdmaQueue::SdmaQueue(int localDeviceId, int remoteDeviceId, const hsa_agent_t& localAgent,
                     uint32_t engineId, const EngineSelection& selection)
    : remoteDeviceId_(remoteDeviceId) {
  int originalDeviceId;

  ANVIL_CHECK_HIP_ERROR(hipGetDevice(&originalDeviceId));  // Save the current device

  uint32_t localNodeId = 0;
  hsa_status_t status = hsa_agent_get_info(localAgent, HSA_AGENT_INFO_NODE, &localNodeId);
  if (status != HSA_STATUS_SUCCESS) {
    LOG_ERROR("anvil: HSA_AGENT_INFO_NODE failed device=%d status=%#x", localDeviceId, status);
    createStatus_ = HSAKMT_STATUS_ERROR;
    return;
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

  HSAKMT_STATUS pinnedStatus = hsaKmtCreateQueueExt(
      localNodeId, HSA_QUEUE_SDMA_BY_ENG_ID, DEFAULT_QUEUE_PERCENTAGE, DEFAULT_PRIORITY, engineId,
      queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_);
  HSAKMT_STATUS queueStatus = pinnedStatus;

  if (queueStatus != HSAKMT_STATUS_SUCCESS) {
    // An engine-pinned queue needs an engine id the KFD node actually owns. A CPX/DPX partition owns
    // one XCD's SDMA engines and no xGMI engines at all, so an id that is valid on the unpartitioned
    // device is rejected here. A generic HSA_QUEUE_SDMA lets KFD pick an engine it owns: that gives
    // up the per-peer xGMI engine affinity but still reaches the peer over xGMI.
    LOG_WARN("anvil: engine-pinned queue rejected (hsakmt=%d %s node=%u engineId=%u), retrying with "
             "a generic SDMA queue",
             static_cast<int>(queueStatus), hsakmtStatusName(queueStatus), localNodeId, engineId);
    memset(&queue_, 0, sizeof(queue_));
    queueStatus =
        hsaKmtCreateQueueExt(localNodeId, HSA_QUEUE_SDMA, DEFAULT_QUEUE_PERCENTAGE, DEFAULT_PRIORITY,
                             0, queueBuffer_, SDMA_QUEUE_SIZE, nullptr, &queue_);
  }

  // A generic retry can return NO_MEMORY after the pinned create returned INVALID_PARAMETER.
  // Keeping only the pinned status would hide the budget failure from connect().
  if (queueStatus == HSAKMT_STATUS_SUCCESS) {
    createStatus_ = HSAKMT_STATUS_SUCCESS;
  } else if (pinnedStatus == HSAKMT_STATUS_NO_MEMORY || queueStatus == HSAKMT_STATUS_NO_MEMORY) {
    createStatus_ = HSAKMT_STATUS_NO_MEMORY;
  } else {
    createStatus_ = pinnedStatus;
  }

  if (queueStatus != HSAKMT_STATUS_SUCCESS) {
    const std::string srcBus = getBusId(localDeviceId);
    const std::string dstBus = getBusId(remoteDeviceId);
    LOG_ERROR(
        "anvil: hsaKmtCreateQueueExt failed hsakmt=%d (%s) node=%u engineId=%u srcDev=%d (%s) "
        "dstDev=%d (%s) usedPreferred=%d preferredStatus=%#x preferredMask=0x%x hostEng=%u "
        "xgmiEng=%u total=%u",
        static_cast<int>(createStatus_), hsakmtStatusName(createStatus_), localNodeId, engineId,
        localDeviceId, srcBus.c_str(), remoteDeviceId, dstBus.c_str(),
        selection.usedPreferred ? 1 : 0, static_cast<unsigned>(selection.preferredStatus),
        selection.preferredMask, selection.numSdmaEngines, selection.numSdmaXgmiEngines,
        selection.numSdmaEnginesTotal);
    // Leave valid_ false so the caller can drop the Anvil backend instead of killing the job.
    hsaKmtUnmapMemoryToGPU(queueBuffer_);
    hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE);
    queueBuffer_ = nullptr;
    return;
  }

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

  valid_ = true;
}

SdmaQueue::~SdmaQueue() {
  if (!valid_) return;
  CHECK_HSAKMT_SUCCESS(hsaKmtDestroyQueue(queue_.QueueId), "Failed to destroy queue.");
  ANVIL_CHECK_HIP_ERROR(hipFree(deviceHandle_));
  if (singleProducerDeviceHandle_) ANVIL_CHECK_HIP_ERROR(hipFree(singleProducerDeviceHandle_));
  ANVIL_CHECK_HIP_ERROR(hipFree(cachedWptr_));
  ANVIL_CHECK_HIP_ERROR(hipFree(committedWptr_));
  CHECK_HSAKMT_SUCCESS(hsaKmtUnmapMemoryToGPU(queueBuffer_), "Failed");
  CHECK_HSAKMT_SUCCESS(hsaKmtFreeMemory(queueBuffer_, SDMA_QUEUE_SIZE), "Failed");
}

bool SdmaQueue::valid() const { return valid_; }

HSAKMT_STATUS SdmaQueue::createStatus() const { return createStatus_; }

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

  uint32_t node = 0;
  HsaNodeProperties nodeProps{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &node) == HSA_STATUS_SUCCESS &&
      hsaKmtGetNodeProperties(node, &nodeProps) == HSAKMT_STATUS_SUCCESS) {
    numSdmaQueuesPerEngine_ = nodeProps.NumSdmaQueuesPerEngine;
  } else {
    LOG_WARN("anvil: SDMA queues-per-engine unknown for node %u; queue budget unchecked", node);
  }

  LOG_TRACE("anvil: SDMA engines host=%u xgmi=%u total=%u queuesPerEngine=%u", numSdmaEngines_,
            numSdmaXgmiEngines_, numSdmaEnginesTotal_, numSdmaQueuesPerEngine_);
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

    // Before querySdmaEngineCounts: the per-engine queue budget comes from the KFD node properties.
    SetUpKFD();
    s_kfd_opened = true;

    querySdmaEngineCounts();
  });
}

SdmaQueue* AnvilLib::createSdmaQueue(int srcDeviceId, int dstDeviceId, uint32_t engineId,
                                     const EngineSelection& selection, int* channelIdx) {
  auto& vec = sdma_channels_[dstDeviceId];
  auto queue = std::make_unique<SdmaQueue>(srcDeviceId, dstDeviceId, getHipGpuAgent(srcDeviceId),
                                           engineId, selection);
  lastQueueStatus_ = queue->createStatus();
  if (!queue->valid()) return nullptr;
  vec.emplace_back(std::move(queue));
  if (channelIdx != nullptr) {
    *channelIdx = static_cast<int>(vec.size() - 1);
  }
  return vec.back().get();
}

bool AnvilLib::connect(int srcDeviceId, int dstDeviceId, int numChannels) {
  const EngineSelection selection = getSdmaEngineId(srcDeviceId, dstDeviceId);
  if (selection.engineId < 0) {
    LOG_ERROR("anvil: no SDMA engine mapping for %d -> %d", srcDeviceId, dstDeviceId);
    return false;
  }
  const uint32_t engineId = static_cast<uint32_t>(selection.engineId);
  LOG_TRACE("SDMA: Connect from %d to %d with %d channels using engine %d",
            srcDeviceId, dstDeviceId, numChannels, engineId);

  // The queue budget is a property of the partition mode rather than of this peer: a CPX partition
  // owns one XCD's engines, so the mesh a whole MI300X supports does not fit. KFD also does not
  // report how many queues ROCr already holds, so an exhausted budget can only be predicted for
  // gross over-subscription and must additionally be recognised when queue creation fails.
  // The budget is counted across all engines, not per engine. A rejected engine-pinned create
  // retries as a generic queue and reports engine 0, so on a partition every queue charges the
  // same key; a per-engine cap would then refuse at numSdmaQueuesPerEngine_ while the partition
  // really offers numSdmaEnginesTotal_ times that, which is the figure this message quotes. The
  // total is an upper bound, so an uneven distribution is still caught by KFD's own NO_MEMORY.
  const uint32_t queueBudget = numSdmaEnginesTotal_ * numSdmaQueuesPerEngine_;
  auto reportBudget = [&](uint32_t used) {
    LOG_ERROR(
        "anvil: SDMA queue budget exhausted: %u queue(s) taken by this process + %d requested, "
        "limit %u (%u engines: host=%u xgmi=%u, %u queues per engine). ROCm already holds some of "
        "them, so this partition cannot cover this peer count at %d channel(s) per peer. Use fewer "
        "ranks per node, fewer channels, or a coarser partition mode (DPX/SPX).",
        used, numChannels, queueBudget, numSdmaEnginesTotal_, numSdmaEngines_, numSdmaXgmiEngines_,
        numSdmaQueuesPerEngine_, numChannels);
  };

  // The whole request is known up front. Refusing here avoids creating a queue
  // that rollback would destroy immediately.
  const uint32_t used = queuesUsedTotal_;
  if (numChannels > 0 && queueBudget > 0 &&
      used + static_cast<uint32_t>(numChannels) > queueBudget) {
    reportBudget(used);
    return false;
  }

  auto& vec = sdma_channels_[dstDeviceId];
  const size_t already = vec.size();
  auto rollback = [&]() {
    while (vec.size() > already) {
      SdmaQueue* q = vec.back().get();
      if (q != nullptr && q->valid() && queuesUsedTotal_ > 0) queuesUsedTotal_ -= 1;
      vec.pop_back();
    }
  };

  for (int c = 0; c < numChannels; ++c) {
    SdmaQueue* queue = createSdmaQueue(srcDeviceId, dstDeviceId, engineId, selection);
    if (queue == nullptr) {
      if (queueBudget > 0 && lastQueueStatus_ == HSAKMT_STATUS_NO_MEMORY) {
        reportBudget(queuesUsedTotal_);
      }
      rollback();
      return false;
    }
    queuesUsedTotal_ += 1;
  }
  return true;
}

void AnvilLib::disconnect() {
  // Destroy all SDMA queues. SdmaQueue destructor calls hsaKmtDestroyQueue.
  sdma_channels_.clear();
  queuesUsedTotal_ = 0;
  LOG_TRACE("SDMA: Disconnected all queues");
}

void AnvilLib::disconnectDevice(int dstDeviceId) {
  auto it = sdma_channels_.find(dstDeviceId);
  if (it == sdma_channels_.end()) return;
  // Only valid queues were charged to the budget in connect().
  for (const auto& q : it->second) {
    if (q != nullptr && q->valid() && queuesUsedTotal_ > 0) queuesUsedTotal_ -= 1;
  }
  sdma_channels_.erase(it);
  LOG_TRACE("SDMA: Disconnected queues for device %d", dstDeviceId);
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
  // xgmi_physical_id is a property of the physical GPU. A CPX/DPX partition is a HIP alias at PCI
  // function .1-.7 of that GPU and has no sysfs node of its own, so read the physical function
  // whenever the partition's own BDF is absent. Returns -1 when neither is readable.
  const PciFunctionBus loc = pciFunctionBus(getBusId(deviceId));
  const std::string* candidates[2] = {&loc.busId, &loc.physBusId};
  const int nCandidates = loc.physBusId == loc.busId ? 1 : 2;
  for (int i = 0; i < nCandidates; ++i) {
    if (candidates[i]->empty()) continue;
    std::ifstream file("/sys/bus/pci/devices/" + *candidates[i] + "/xgmi_physical_id");
    int xgmi_physical_id = 0;
    if (file.is_open() && (file >> xgmi_physical_id)) return xgmi_physical_id;
  }

  const int preferredEngine =
      selection_.preferredMask == 0 ? -1 : __builtin_ctz(selection_.preferredMask);
  // LOG_ERROR so a default ROCSHMEM_DEBUG_LEVEL=ERROR banff repro still shows the preferred-engine
  // values that explain why the OAM map was needed.
  LOG_ERROR(
      "anvil: no xGMI physical id for %s or %s device=%d pair=%d->%d preferredQueried=%d "
      "preferredStatus=%#x preferredMask=0x%x preferredEngine=%d hostEng=%u xgmiEng=%u total=%u",
      loc.busId.c_str(), loc.physBusId.c_str(), deviceId, selection_.srcDeviceId,
      selection_.dstDeviceId, selection_.preferredQueried ? 1 : 0,
      static_cast<unsigned>(selection_.preferredStatus), selection_.preferredMask, preferredEngine,
      selection_.numSdmaEngines, selection_.numSdmaXgmiEngines, selection_.numSdmaEnginesTotal);
  return -1;
}

int AnvilLib::getSdmaEngineIdFromOamMap(int srcDeviceId, int dstDeviceId) {
  const int srcOamId = getOamId(srcDeviceId);
  const int dstOamId = getOamId(dstDeviceId);
  // Do not clamp an unreadable id to 0: that would make every failed pair select mi300xOamMap[0][0]
  // and pin them all on engine 0. Propagate to connect() instead.
  if (srcOamId < 0 || dstOamId < 0) return -1;

  const int srcIdx = srcOamId % static_cast<int>(AnvilLib::kMi300xOamMapDim);
  const int dstIdx = dstOamId % static_cast<int>(AnvilLib::kMi300xOamMapDim);

  const int oamEngine = mi300xOamMap[static_cast<size_t>(srcIdx)][static_cast<size_t>(dstIdx)];
  // Use even engines only (MI300X xGMI SDMA layout).
  int engineId = oamEngine * 2;

  // The map and the doubling assume the 14 xGMI SDMA engines of an unpartitioned MI300X. A
  // partition owns fewer engines (CPX: 2 host, 0 xGMI). Same-device peers share an OAM id, so they
  // hit the diagonal (oamEngine 0) and engineId 0 is in range. The fold has to run there too, not
  // only when the doubled id is past the engine count.
  //
  // (srcFn + dstFn) rather than dstFn alone: with 2 engines the two split a mixed-parity mesh the
  // same way, but once the node has more than 2 engines the sum still separates pairs that share a
  // destination function. Two engines cannot give 8 partitions distinct ids.
  const bool partition = numSdmaXgmiEngines_ == 0 && numSdmaEnginesTotal_ > 0;
  if (oamMapEngineNeedsFold(numSdmaXgmiEngines_, numSdmaEnginesTotal_, engineId)) {
    const PciFunctionBus srcPci = pciFunctionBus(getBusId(srcDeviceId));
    const PciFunctionBus dstPci = pciFunctionBus(getBusId(dstDeviceId));
    // An unreadable tail adds 0 and therefore collides with function 0. The warning logs the raw
    // BDF so that case is not silent. The info line reports the values the fold actually added.
    const int srcFn = srcPci.function < 0 ? 0 : srcPci.function;
    const int dstFn = dstPci.function < 0 ? 0 : dstPci.function;
    const int folded = foldOamMapEngine(oamEngine, srcFn, dstFn, numSdmaEnginesTotal_);
    if (srcPci.function < 0 || dstPci.function < 0) {
      LOG_WARN("anvil: PCI function unreadable src=%s dst=%s, using engine %d (oam=%d total=%u)",
               srcPci.busId.c_str(), dstPci.busId.c_str(), folded, oamEngine, numSdmaEnginesTotal_);
    } else if (partition) {
      LOG_INFO("anvil: partition engine %d (oam=%d srcFn=%d dstFn=%d total=%u)", folded, oamEngine,
               srcFn, dstFn, numSdmaEnginesTotal_);
    } else {
      LOG_WARN(
          "anvil: legacy OAM-map engine %d >= total %u, using engine %d (oam=%d srcFn=%d dstFn=%d)",
          engineId, numSdmaEnginesTotal_, folded, oamEngine, srcFn, dstFn);
    }
    engineId = folded;
  }
  return engineId;
}

EngineSelection AnvilLib::getSdmaEngineId(int srcDeviceId, int dstDeviceId) {
  // Published on the way in: getOamId reads it to report why the map was needed.
  selection_ = EngineSelection{};
  selection_.srcDeviceId = srcDeviceId;
  selection_.dstDeviceId = dstDeviceId;
  selection_.numSdmaEngines = numSdmaEngines_;
  selection_.numSdmaXgmiEngines = numSdmaXgmiEngines_;
  selection_.numSdmaEnginesTotal = numSdmaEnginesTotal_;

  if (srcDeviceId >= 0 && dstDeviceId >= 0 &&
      srcDeviceId < static_cast<int>(gpuAgentsByHipDev_.size()) &&
      dstDeviceId < static_cast<int>(gpuAgentsByHipDev_.size()) &&
      hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(srcDeviceId)]) &&
      hsaAgentIsValid(gpuAgentsByHipDev_[static_cast<size_t>(dstDeviceId)])) {
    const hsa_agent_t srcAgent = gpuAgentsByHipDev_[static_cast<size_t>(srcDeviceId)];
    const hsa_agent_t dstAgent = gpuAgentsByHipDev_[static_cast<size_t>(dstDeviceId)];
    selection_.preferredStatus =
        hsa_amd_memory_get_preferred_copy_engine(dstAgent, srcAgent, &selection_.preferredMask);
    selection_.preferredQueried = true;
    if (selection_.preferredStatus == HSA_STATUS_SUCCESS && selection_.preferredMask != 0) {
      const int engineId = __builtin_ctz(selection_.preferredMask);
      if (engineId >= 0 && (numSdmaEnginesTotal_ == 0 ||
                            static_cast<uint32_t>(engineId) < numSdmaEnginesTotal_)) {
        selection_.usedPreferred = true;
        selection_.engineId = engineId;
        LOG_TRACE("SDMA: HSA preferred engine %d for %d -> %d (mask=0x%x)", engineId, srcDeviceId,
                  dstDeviceId, selection_.preferredMask);
        return selection_;
      }
      LOG_WARN("anvil: HSA preferred engine %d out of range (total=%u status=%#x mask=0x%x), "
               "using OAM map",
               engineId, numSdmaEnginesTotal_,
               static_cast<unsigned>(selection_.preferredStatus), selection_.preferredMask);
    }
  }

  selection_.engineId = getSdmaEngineIdFromOamMap(srcDeviceId, dstDeviceId);
  return selection_;
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
