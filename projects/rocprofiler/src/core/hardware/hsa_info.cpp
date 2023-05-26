/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hsa_info.h"

#include "src/utils/helper.h"

#define CHECK_STATUS(msg, status)                                                                  \
  do {                                                                                             \
    if ((status) != HSA_STATUS_SUCCESS) {                                                          \
      const char* emsg = 0;                                                                        \
      hsa_status_string(status, &emsg);                                                            \
      throw(ROCPROFILER_STATUS_ERROR_HSA_SUPPORT,                                                  \
            "Error: " << msg << ": " << emsg ? emsg : "<unknown error>");                          \
    }                                                                                              \
  } while (0)

namespace Agent {
// AgentInfo Class

AgentInfo::AgentInfo() {}
AgentInfo::AgentInfo(const hsa_agent_t agent, ::CoreApiTable* table) : handle_(agent.handle) {
  if (table->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type_) != HSA_STATUS_SUCCESS)
    rocmtools::fatal("hsa_agent_get_info failed");

  table->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_NAME, name_);

  const int gfxip_label_len = std::min(strlen(name_) - 2, sizeof(gfxip_) - 1);
  memcpy(gfxip_, name_, gfxip_label_len);
  gfxip_[gfxip_label_len] = '\0';

  if (type_ != HSA_DEVICE_TYPE_GPU) {
    return;
  }

  table->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &max_wave_size_);
  table->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size_);

  table->hsa_agent_get_info_fn(
      agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &cu_num_);

  table->hsa_agent_get_info_fn(
      agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU), &simds_per_cu_);

  table->hsa_agent_get_info_fn(
      agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES), &se_num_);

  if (table->hsa_agent_get_info_fn(agent,
                                   (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
                                   &shader_arrays_per_se_) != HSA_STATUS_SUCCESS ||
      table->hsa_agent_get_info_fn(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                   &waves_per_cu_) != HSA_STATUS_SUCCESS) {
    rocmtools::fatal("hsa_agent_get_info for gfxip hardware configuration failed");
  }

  compute_units_per_sh_ = cu_num_ / (se_num_ * shader_arrays_per_se_);
  wave_slots_per_simd_ = waves_per_cu_ / simds_per_cu_;

  if (table->hsa_agent_get_info_fn(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN,
                                   &pci_domain_) != HSA_STATUS_SUCCESS ||
      table->hsa_agent_get_info_fn(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID,
                                   &pci_location_id_) != HSA_STATUS_SUCCESS) {
    rocmtools::fatal("hsa_agent_get_info for PCI info failed");
  }

  // TODO: (sauverma) use hsa_agent_get_info_fn(HSA_AMD_AGENT_INFO_NUM_XCC)
  // to get xcc_num once hsa headers are updated from rocr/hsa
  std::string gpu_name = std::string(name_).substr(0, 6);
  if (gpu_name == "gfx940")
    xcc_num_ = 6;
  else
    xcc_num_ = 1;
}

int AgentInfo::getIndex() const { return index_; }
hsa_device_type_t AgentInfo::getType() const { return type_; }
uint64_t AgentInfo::getHandle() const { return handle_; }
const std::string_view AgentInfo::getName() const { return name_; }
std::string AgentInfo::getGfxip() const { return std::string(gfxip_); }
uint32_t AgentInfo::getMaxWaveSize() const { return max_wave_size_; }
uint32_t AgentInfo::getMaxQueueSize() const { return max_queue_size_; }
uint32_t AgentInfo::getCUCount() const { return cu_num_; }
uint32_t AgentInfo::getSimdCountPerCU() const { return simds_per_cu_; }
uint32_t AgentInfo::getShaderEngineCount() const { return se_num_; }
uint32_t AgentInfo::getShaderArraysPerSE() const { return shader_arrays_per_se_; }
uint32_t AgentInfo::getMaxWavesPerCU() const { return waves_per_cu_; }
uint32_t AgentInfo::getCUCountPerSH() const { return compute_units_per_sh_; }
uint32_t AgentInfo::getWaveSlotsPerSimd() const { return wave_slots_per_simd_; }
uint32_t AgentInfo::getPCIDomain() const { return pci_domain_; }
uint32_t AgentInfo::getPCILocationID() const { return pci_location_id_; }
uint32_t AgentInfo::getXccCount() const { return xcc_num_; }

void AgentInfo::setIndex(int index) { index_ = index; }
void AgentInfo::setType(hsa_device_type_t type) { type_ = type; }
void AgentInfo::setHandle(uint64_t handle) { handle_ = handle; }
void AgentInfo::setName(const std::string& name) { strcpy(name_, name.c_str()); }

void AgentInfo::setNumaNode(uint32_t numa_node) { numa_node_ = numa_node; }
uint32_t AgentInfo::getNumaNode() { return numa_node_; }

void AgentInfo::setNearCpuAgent(hsa_agent_t near_cpu_agent) { near_cpu_agent_ = near_cpu_agent; }
hsa_agent_t AgentInfo::getNearCpuAgent() { return near_cpu_agent_; }

// CounterHardwareInfo Class

CounterHardwareInfo::CounterHardwareInfo(uint64_t event_id, const char* block_id)
    : event_id_(event_id), block_id_(block_id) {}
int64_t CounterHardwareInfo::getNumInstances() { return num_instances_; }

bool getHardwareInfo(uint64_t event_id, const char* block_id,
                     CounterHardwareInfo* counter_hardware_info) {
  counter_hardware_info = new CounterHardwareInfo(event_id, block_id);
  return true;
}

}  // namespace Agent
