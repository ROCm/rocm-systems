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
#include <fstream>

#include "src/utils/filesystem.hpp"
#include "src/utils/helper.h"

namespace fs = rocprofiler::common::filesystem;

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

char convert(uint32_t version) {
  uint32_t diff = version - 10;
  if (static_cast<char>('a' + diff) >= 'a' && static_cast<char>('a' + diff) <= 'z')
    return static_cast<char>('a' + diff);
  rocprofiler::fatal("Incorrect gpu version");
}

DeviceInfo::DeviceInfo(uint32_t topology_id, uint32_t gpu_id) {
  fs::path sysfs_nodes_path = "/sys/class/kfd/kfd/topology/nodes/";
  fs::directory_entry dirp("/sys/class/kfd/kfd/topology/nodes");
  if (!fs::exists(sysfs_nodes_path))
    rocprofiler::fatal("Could not opendir `%s'", sysfs_nodes_path.c_str());
  // Check the type of the device using gpu id
  if (gpu_id == 0) assert("DeviceInfo does not support CPU");
  numa_node_ = topology_id;

  fs::path node_path = sysfs_nodes_path / std::to_string(topology_id);
  if (!fs::exists(node_path)) rocprofiler::fatal("Could not opendir `%s'", node_path.c_str());
  xcc_num_ = 1;
  gpu_id_ = gpu_id;
  fs::path properties_path = node_path / "properties";
  std::ifstream props_ifs(properties_path);
  uint32_t simd_count = 0, array_count = 0;
  uint32_t max_waves_per_simd = 0, gfx_target_version = 0;
  std::string prop_name, minor_version_str, stepping_str;
  uint64_t prop_value;
  uint32_t major_version = 0, minor_version = 0, stepping = 0;
  std::stringstream hex_minor_version;
  max_wave_size_ = 0;
  simds_per_cu_ = 0;
  shader_arrays_per_se_ = 0;
  se_num_ = 0;
  waves_per_cu_ = 0;
  cu_num_ = 0;
  compute_units_per_sh_ = 0;
  if (!props_ifs.is_open())
    rocprofiler::fatal("Could not open %s/properties", properties_path.c_str());
  while (props_ifs >> prop_name >> prop_value) {
    if (prop_name == "wave_front_size") {
      max_wave_size_ = static_cast<uint32_t>(prop_value);
      if (max_wave_size_ <= 0) rocprofiler::fatal("Invalid max_wave_size_ in the topology file");
    } else if (prop_name == "simd_count") {
      simd_count = static_cast<uint32_t>(prop_value);
      if (simd_count <= 0)
        rocprofiler::fatal("Invalid simd_count in the topology file");
    } else if (prop_name == "array_count") {
      array_count = static_cast<uint32_t>(prop_value);
      if (array_count <= 0) rocprofiler::fatal("Invalid array_count in the topology file");
    } else if (prop_name == "simd_per_cu") {
      simds_per_cu_ = static_cast<uint32_t>(prop_value);
      if (simds_per_cu_ <= 0) rocprofiler::fatal("Invalid simd_per_cu in the topology file");
    } else if (prop_name == "location_id")
      pci_location_id_ = static_cast<uint32_t>(prop_value);
    else if (prop_name == "domain")
      pci_domain_ = static_cast<uint32_t>(prop_value);
    else if (prop_name == "simd_arrays_per_engine") {
      shader_arrays_per_se_ = static_cast<uint32_t>(prop_value);
      if (shader_arrays_per_se_ <= 0)
        rocprofiler::fatal("Invalid simd_arrays_per_engine in the topology file");
    } else if (prop_name == "max_waves_per_simd") {
      max_waves_per_simd = static_cast<uint32_t>(prop_value);
      if (max_waves_per_simd <= 0)
        rocprofiler::fatal("Invalid max_waves_per_simd in the topology file");
    } else if (prop_name == "gfx_target_version")
      gfx_target_version = static_cast<uint32_t>(prop_value);

    else if (prop_name == "unique_id")
      unique_gpu_id_ = static_cast<uint64_t>(prop_value);
    else if (prop_name == "num_xcc")
      xcc_num_ = static_cast<uint32_t>(prop_value);
  }

  se_num_ = array_count / shader_arrays_per_se_;
  waves_per_cu_ = max_waves_per_simd * simds_per_cu_;
  cu_num_ = simd_count / simds_per_cu_;
  major_version = (gfx_target_version / 100) / 100;
  std::string major_version_str = std::to_string(major_version);
  minor_version = (gfx_target_version / 100) % 100;
  if (minor_version > 9)
    minor_version_str = std::string(1, convert(minor_version));
  else
    minor_version_str = std::to_string(minor_version);
  stepping = (gfx_target_version % 100);
  if (stepping > 9)
    stepping_str = std::string(1, convert(stepping));
  else
    stepping_str = std::to_string(stepping);
  std::string gpu_name = "gfx" + major_version_str + minor_version_str + stepping_str;
  strcpy(name_, gpu_name.c_str());
  compute_units_per_sh_ = cu_num_ / (se_num_ * shader_arrays_per_se_);
  wave_slots_per_simd_ = waves_per_cu_ / simds_per_cu_;
  const int gfxip_label_len = std::min(strlen(name_) - 2, sizeof(gfxip_) - 1);
  memcpy(gfxip_, name_, gfxip_label_len);
  gfxip_[gfxip_label_len] = '\0';
}


std::string_view DeviceInfo::getName() const { return name_; }
std::string DeviceInfo::getGfxip() const { return std::string(gfxip_); }
uint32_t DeviceInfo::getMaxWaveSize() const { return max_wave_size_; }
uint32_t DeviceInfo::getMaxQueueSize() const { return max_queue_size_; }
uint32_t DeviceInfo::getCUCount() const { return cu_num_; }
uint32_t DeviceInfo::getSimdCountPerCU() const { return simds_per_cu_; }
uint32_t DeviceInfo::getShaderEngineCount() const { return se_num_; }
uint32_t DeviceInfo::getShaderArraysPerSE() const { return shader_arrays_per_se_; }
uint32_t DeviceInfo::getMaxWavesPerCU() const { return waves_per_cu_; }
uint32_t DeviceInfo::getCUCountPerSH() const { return compute_units_per_sh_; }
uint32_t DeviceInfo::getWaveSlotsPerSimd() const { return wave_slots_per_simd_; }
uint32_t DeviceInfo::getPCIDomain() const { return pci_domain_; }
uint32_t DeviceInfo::getPCILocationID() const { return pci_location_id_; }
uint32_t DeviceInfo::getXccCount() const { return xcc_num_; }
uint64_t DeviceInfo::getUniqueGPUId() const { return unique_gpu_id_; }
uint32_t DeviceInfo::getNumaNode() const { return numa_node_; }
uint64_t DeviceInfo::getGPUId() const { return gpu_id_; }

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
