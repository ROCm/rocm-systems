/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "amd_smi/impl/amd_smi_gpu_device.h"

extern "C" {
#include "ualoe_lib/ualoe_lib.h"
}

#include <dirent.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/fdinfo.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace amd::smi {

// Constant for KFD context directory prefix
static constexpr const char* kContextPrefix = "context_";

AMDSmiGPUDevice::AMDSmiGPUDevice(uint32_t gpu_id, std::string path, amdsmi_bdf_t bdf,
                                 AMDSmiDrm& drm)
    : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
      gpu_id_(gpu_id),
      path_(path),
      bdf_(bdf),
      drm_(drm) {
  populate_ifoe_fabric_bdf_list();
  if (has_ifoe_related_bdf() && device_has_ualink()) {
    auto ifoe_bdf_str = get_ifoe_bdf_string();
    if (auto ualoe_status = ualoe_open(ifoe_bdf_str.c_str(), &ualoe_handle_); ualoe_status != 0) {
      ualoe_handle_ = (-1);
    }
  }
}

AMDSmiGPUDevice::AMDSmiGPUDevice(uint32_t gpu_id, AMDSmiDrm& drm)
    : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU), gpu_id_(gpu_id), drm_(drm) {
  if (check_if_drm_is_supported()) this->get_drm_data();

  populate_ifoe_fabric_bdf_list();
  if (has_ifoe_related_bdf() && device_has_ualink()) {
    auto ifoe_bdf_str = get_ifoe_bdf_string();
    if (auto ualoe_status = ualoe_open(ifoe_bdf_str.c_str(), &ualoe_handle_); ualoe_status != 0) {
      ualoe_handle_ = (-1);
    }
  }
}

AMDSmiGPUDevice::~AMDSmiGPUDevice() {
  if (ualoe_handle_ != -1) {
    ualoe_close(ualoe_handle_);
    ualoe_handle_ = -1;
  }
}

uint32_t AMDSmiGPUDevice::get_gpu_id() const { return gpu_id_; }

uint32_t AMDSmiGPUDevice::get_card_id() {
  // Should never return not_supported, but just in case
  rsmi_status_t ret = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
  uint32_t gpu_index = this->get_gpu_id();
  rsmi_device_identifiers_t identifiers = rsmi_device_identifiers_t{};
  ret = rsmi_dev_device_identifiers_get(gpu_index, &identifiers);
  if (ret != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    this->card_index_ = std::numeric_limits<uint32_t>::max();
  } else {
    this->card_index_ = identifiers.card_index;
  }

  return this->card_index_;
}

uint32_t AMDSmiGPUDevice::get_drm_render_minor() {
  // Should never return not_supported, but just in case
  rsmi_status_t ret = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
  uint32_t gpu_index = this->get_gpu_id();
  rsmi_device_identifiers_t identifiers = rsmi_device_identifiers_t{};
  ret = rsmi_dev_device_identifiers_get(gpu_index, &identifiers);
  if (ret != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    this->drm_render_minor_ = std::numeric_limits<uint32_t>::max();
  } else {
    this->drm_render_minor_ = identifiers.drm_render_minor;
  }

  return this->drm_render_minor_;
}

uint64_t AMDSmiGPUDevice::get_kfd_gpu_id() {
  // Should never return not_supported, but just in case
  rsmi_status_t ret = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
  uint32_t gpu_index = this->get_gpu_id();
  rsmi_device_identifiers_t identifiers = rsmi_device_identifiers_t{};
  ret = rsmi_dev_device_identifiers_get(gpu_index, &identifiers);
  if (ret != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    this->kfd_gpu_id_ = std::numeric_limits<uint64_t>::max();
  } else {
    this->kfd_gpu_id_ = identifiers.kfd_gpu_id;
  }

  return this->kfd_gpu_id_;
}

std::string& AMDSmiGPUDevice::get_gpu_path() { return path_; }
const std::string& AMDSmiGPUDevice::get_gpu_path() const { return path_; }

amdsmi_bdf_t AMDSmiGPUDevice::get_bdf() { return this->bdf_; }

uint32_t AMDSmiGPUDevice::get_vendor_id() { return vendor_id_; }

amdsmi_status_t AMDSmiGPUDevice::get_drm_data() {
  amdsmi_status_t ret;
  std::string path;
  amdsmi_bdf_t bdf;
  ret = drm_.get_drm_path_by_index(gpu_id_, &path);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;
  ret = drm_.get_bdf_by_index(gpu_id_, &bdf);
  if (ret != AMDSMI_STATUS_SUCCESS) return AMDSMI_STATUS_NOT_SUPPORTED;

  bdf_ = bdf, path_ = path;
  vendor_id_ = drm_.get_vendor_id();
  populate_ifoe_fabric_bdf_list();

  return AMDSMI_STATUS_SUCCESS;
}

pthread_mutex_t* AMDSmiGPUDevice::get_mutex() { return amd::smi::GetMutex(gpu_id_); }

amdsmi_status_t AMDSmiGPUDevice::amdgpu_query_cpu_affinity(std::string& cpu_affinity) const {
  char bdf_str[20];
  snprintf(bdf_str, sizeof(bdf_str) - 1, "%04lx:%02x", bdf_.domain_number, bdf_.bus_number);
  std::stringstream domain_bus_sstream;
  domain_bus_sstream << "/sys/class/pci_bus/" << std::string(bdf_str);

  return drm_.amdgpu_query_cpu_affinity(domain_bus_sstream.str(), cpu_affinity);
}

namespace {
// cache the compute process list for the device
struct ComputeProcessCache {
  std::unique_ptr<rsmi_process_info_t[]> list_all_processes_ptr = nullptr;
  std::atomic<std::chrono::steady_clock::time_point> last_compute_process_list_update_time{
      std::chrono::steady_clock::time_point{}};
  std::mutex mtx;
  uint32_t num_running_processes = 0;
};

std::unordered_map<uint32_t, amdsmi_proc_info_t> process_info_cache_map;
std::unordered_map<uint32_t, ComputeProcessCache*> compute_process_cache_map;
std::mutex compute_process_list_mutex;
static const std::chrono::milliseconds kComputeProcessCacheDuration =
    std::chrono::milliseconds(read_env_ms("AMDSMI_PROCESS_INFO_CACHE_MS", 1));

}  // namespace

int32_t AMDSmiGPUDevice::get_compute_process_list_impl(
    GPUComputeProcessList_t& compute_process_list, ComputeProcessListType_t list_type) {
  ComputeProcessCache* cache_ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(compute_process_list_mutex);
    if (compute_process_cache_map.find(gpu_id_) == compute_process_cache_map.end()) {
      compute_process_cache_map[gpu_id_] = new ComputeProcessCache();
    }
    cache_ptr = compute_process_cache_map[gpu_id_];
  }

  /**
   *  The first call to rsmi_compute_process_info_get() to find the number of
   *  rsmi_process_info_t currently running on the system.
   */
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  auto now = std::chrono::steady_clock::now();
  auto last_read_delta = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - cache_ptr->last_compute_process_list_update_time.load());
  // only get new data if cache duration has expired
  if (last_read_delta > kComputeProcessCacheDuration) {
    // double-check locking pattern here
    std::lock_guard<std::mutex> lock(cache_ptr->mtx);
    if (std::chrono::steady_clock::now() -
            cache_ptr->last_compute_process_list_update_time.load() <=
        kComputeProcessCacheDuration) {
      // another thread already updated the data while we were waiting for the lock
      // so just return the existing data
      return rsmi_status_t::RSMI_STATUS_SUCCESS;
    }

    // Clear the process info cache when refreshing
    process_info_cache_map.clear();

    status_code = rsmi_compute_process_info_get(nullptr, &cache_ptr->num_running_processes);
    if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
      return status_code;
    }
    if (cache_ptr->num_running_processes <= 0) {
      compute_process_list.clear();
      cache_ptr->last_compute_process_list_update_time = std::chrono::steady_clock::now();
      return status_code;
    }

    /**
     *  Make a type safe pointer, then
     *
     * second call to rsmi_compute_process_info_get() to get the actual data into
     *  the allocated rsmi_process_info_t array.
     */
    cache_ptr->list_all_processes_ptr =
        std::make_unique<rsmi_process_info_t[]>(cache_ptr->num_running_processes);

    status_code = rsmi_compute_process_info_get(cache_ptr->list_all_processes_ptr.get(),
                                                &cache_ptr->num_running_processes);
    if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
      return status_code;
    }

    if (cache_ptr->num_running_processes <= 0) {
      compute_process_list.clear();
      cache_ptr->last_compute_process_list_update_time = std::chrono::steady_clock::now();
      return rsmi_status_t::RSMI_STATUS_SUCCESS;  // No processes running
    }

    cache_ptr->last_compute_process_list_update_time = std::chrono::steady_clock::now();
  }

  /**
   *  Check that you have devices that are able to be monitored, ie excluding CPUs
   */
  auto num_running_devices = uint32_t(0);
  auto list_device_allocation_size = uint32_t(0);
  status_code = rsmi_num_monitor_devices(&num_running_devices);
  if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || (num_running_devices <= 0)) {
    return status_code;
  }

  /**
   * Populate process information for the given AMDSmiGPUDevice reference.
   * This function retrieves the process information given in rsmi_proc_info_t
   * and populates the amdsmi_proc_info_t structure.
   */
  auto get_process_info = [&](const rsmi_process_info_t& rsmi_proc_info,
                              amdsmi_proc_info_t& amdsmi_proc_info) {
    // amdsmi_proc_info_t gets populated with /proc information from gpuvsmi_get_pid_info()

    auto status_code = gpuvsmi_get_pid_info(get_bdf(), rsmi_proc_info.process_id, amdsmi_proc_info);
    // If we cannot get the info from sysfs, save the minimum info
    if (status_code != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
      amdsmi_proc_info.pid = rsmi_proc_info.process_id;
      amdsmi_proc_info.memory_usage.vram_mem = rsmi_proc_info.vram_usage;
    }

    // Copy the kfd stats from rsmi_process_info_t to amdsmi_proc_info_t
    amdsmi_proc_info.cu_occupancy = rsmi_proc_info.cu_occupancy;
    amdsmi_proc_info.evicted_time = rsmi_proc_info.evicted_time;
    amdsmi_proc_info.sdma_usage = rsmi_proc_info.sdma_usage;

    // Safely handle KFD processes to get total memory_usage of the process
    uint64_t kfd_gpu_id = get_kfd_gpu_id();
    std::string kfd_proc_path =
        "/sys/class/kfd/kfd/proc/" + std::to_string(rsmi_proc_info.process_id);
    std::string kfd_vram_file = "/vram_" + std::to_string(kfd_gpu_id);

    // Helper for safe addition without overflow
    auto safe_add = [](uint64_t a, uint64_t b) -> uint64_t {
      return (a > UINT64_MAX - b) ? UINT64_MAX : a + b;
    };
    // Helper lambda to read VRAM from a path.
    // Returns 0 if file doesn't exist or can't be read (intentional for optional paths).
    // Logs parse errors via LOG_INFO but doesn't propagate them - this is a best-effort
    // aggregation where partial data is better than failing the entire operation.
    auto read_vram_from_path = [&kfd_vram_file](const std::string& base_path) -> uint64_t {
      uint64_t vram_bytes = 0;
      std::string vram_path = base_path + kfd_vram_file;

      // File may not exist for secondary contexts - this is expected, not an error
      if (access(vram_path.c_str(), R_OK) != 0) {
        return 0;  // File doesn't exist or not readable - expected for optional paths
      }

      std::ifstream kfd_file(vram_path);
      if (!kfd_file.is_open()) {
        return 0;  // Couldn't open file - treat as no data available
      }

      std::string line;
      if (std::getline(kfd_file, line)) {
        try {
          vram_bytes = std::stoull(line);
        } catch (const std::exception& e) {
          // Parse error is unexpected - log it for debugging
          std::ostringstream ss;
          ss << __PRETTY_FUNCTION__ << " | Failed to parse VRAM value from KFD: " << e.what();
          LOG_INFO(ss);
          // Return 0 rather than failing - best effort aggregation
        }
      }
      kfd_file.close();
      return vram_bytes;
    };

    // Helper lambda to read VRAM from all contexts in a directory
    auto read_vram_from_all_contexts = [&read_vram_from_path,
                                        &safe_add](const std::string& base_path) -> uint64_t {
      uint64_t total = read_vram_from_path(base_path);

      // Check for secondary contexts (context_xxxx directories)
      DIR* dir = opendir(base_path.c_str());
      if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
          if (strncmp(entry->d_name, kContextPrefix, strlen(kContextPrefix)) == 0) {
            std::string context_path = base_path + "/" + entry->d_name;
            total = safe_add(total, read_vram_from_path(context_path));
          }
        }
        closedir(dir);
      }
      return total;
    };

    // Read VRAM from primary process
    uint64_t total_vram = read_vram_from_all_contexts(kfd_proc_path);

    // Also check for "pid:PID-id:X" format directories at the parent level
    // This is another format used for multi-context processes
    std::string kfd_root = "/sys/class/kfd/kfd/proc/";
    std::string pid_prefix = "pid:" + std::to_string(rsmi_proc_info.process_id) + "-id:";
    DIR* proc_root = opendir(kfd_root.c_str());
    if (proc_root != nullptr) {
      struct dirent* root_entry;
      while ((root_entry = readdir(proc_root)) != nullptr) {
        if (root_entry->d_name[0] == '.') continue;
        std::string entry_name = root_entry->d_name;
        if (entry_name.find(pid_prefix) == 0) {
          std::string alternate_path = kfd_root + entry_name;
          total_vram = safe_add(total_vram, read_vram_from_all_contexts(alternate_path));
        }
      }
      closedir(proc_root);
    }

    if (total_vram > 0) {
      amdsmi_proc_info.mem = total_vram;
    }

    return status_code;
  };

  /**
   *  Devices used by a process.
   */
  auto update_list_by_running_device = [&](rsmi_process_info_t rsmi_proc_info) {
    // Get all devices running this process into list_device_ptr
    auto status_result(true);
    std::unique_ptr<uint32_t[]> list_device_ptr = std::make_unique<uint32_t[]>(num_running_devices);
    list_device_allocation_size = num_running_devices;
    auto status_code = rsmi_compute_process_gpus_get(
        rsmi_proc_info.process_id, list_device_ptr.get(), &list_device_allocation_size);
    if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
      status_result = false;
      return status_result;
    }

    for (auto device_idx = uint32_t(0); device_idx < list_device_allocation_size; ++device_idx) {
      // Is this device running this process?
      if (list_device_ptr[device_idx] == get_gpu_id()) {
        amdsmi_proc_info_t tmp_amdsmi_proc_info{};

        auto cached_amdsmi_proc = process_info_cache_map.find(rsmi_proc_info.process_id);
        if (cached_amdsmi_proc != process_info_cache_map.end()) {
          // Use cached info
          tmp_amdsmi_proc_info = cached_amdsmi_proc->second;
        } else {
          // Need to get new info from system
          std::unordered_set<uint64_t> gpu_set;
          gpu_set.insert(get_kfd_gpu_id());
          GetProcessInfoForPID(rsmi_proc_info.process_id, &rsmi_proc_info, &gpu_set);
          get_process_info(rsmi_proc_info, tmp_amdsmi_proc_info);
          process_info_cache_map[rsmi_proc_info.process_id] = tmp_amdsmi_proc_info;
        }
        compute_process_list.emplace(rsmi_proc_info.process_id, tmp_amdsmi_proc_info);
      }
    }

    return status_result;
  };

  /**
   *  Transfer/Save the ones linked to this device.
   */
  compute_process_list.clear();
  std::lock_guard<std::mutex> lock(cache_ptr->mtx);
  for (auto process_idx = uint32_t(0); process_idx < cache_ptr->num_running_processes;
       ++process_idx) {
    if (list_type == ComputeProcessListType_t::kAllProcesses ||
        list_type == ComputeProcessListType_t::kAllProcessesOnDevice) {
      update_list_by_running_device(cache_ptr->list_all_processes_ptr[process_idx]);
    }
  }

  return status_code;
}

const GPUComputeProcessList_t& AMDSmiGPUDevice::amdgpu_get_compute_process_list(
    ComputeProcessListType_t list_type) {
  auto error_code = get_compute_process_list_impl(compute_process_list_, list_type);
  if (error_code) {
    compute_process_list_.clear();
  }

  return compute_process_list_;
}

// Convert `amdsmi_bdf_t` to a PCI BDF string
std::string AMDSmiGPUDevice::bdf_to_string() const {
  std::ostringstream oss;
  oss << std::setfill('0') << std::hex                                // Use hexadecimal formatting
      << std::setw(4) << bdf_.domain_number << ":"                    // Domain (4 digits)
      << std::setw(2) << static_cast<int>(bdf_.bus_number) << ":"     // Bus (2 digits)
      << std::setw(2) << static_cast<int>(bdf_.device_number) << "."  // Device (2 digits)
      << static_cast<int>(bdf_.function_number);                      // Function (1 digit)
  return oss.str();
}

std::vector<uint64_t> AMDSmiGPUDevice::get_bitmask_from_numa_node(int32_t node_id,
                                                                  uint32_t size) const {
  std::vector<uint64_t> bitmask(size, 0);

  if (node_id < 0) {
    bitmask[0] = std::numeric_limits<int32_t>::max();
    return bitmask;
  }

  std::string path = "/sys/devices/system/node/node" + std::to_string(node_id) + "/cpulist";
  std::ifstream file(path);

  if (file.is_open()) {
    std::string info;
    while (std::getline(file, info)) {
      std::istringstream sstr(info);
      std::string node_cpus;
      while (std::getline(sstr, node_cpus, ',')) {
        size_t hyphen = node_cpus.find('-');
        if (hyphen != std::string::npos) {
          int start = std::stoi(node_cpus.substr(0, hyphen));
          int end = std::stoi(node_cpus.substr(hyphen + 1));
          for (int i = start; i <= end; ++i) {
            bitmask[i / 64] |= (1ULL << (i % 64));
          }
        } else {
          int core = std::stoi(node_cpus);
          bitmask[core / 64] |= (1ULL << (core % 64));
        }
      }
    }
  }
  return bitmask;
}

std::vector<uint64_t> AMDSmiGPUDevice::get_bitmask_from_local_cpulist(uint32_t drm_card,
                                                                      uint32_t size) const {
  std::vector<uint64_t> bitmask(size, 0);

  if (drm_card == std::numeric_limits<uint32_t>::max()) {
    bitmask[0] = std::numeric_limits<int32_t>::max();
    return bitmask;
  }

  std::string path = "/sys/class/drm/card" + std::to_string(drm_card) + "/device/local_cpulist";
  std::ifstream file(path);

  if (file.is_open()) {
    std::string info;
    while (std::getline(file, info)) {
      std::istringstream sstr(info);
      std::string node_cpus;
      while (std::getline(sstr, node_cpus, ',')) {
        size_t hyphen = node_cpus.find('-');
        if (hyphen != std::string::npos) {
          int start = std::stoi(node_cpus.substr(0, hyphen));
          int end = std::stoi(node_cpus.substr(hyphen + 1));
          for (int i = start; i <= end; ++i) {
            bitmask[i / 64] |= (1ULL << (i % 64));
          }
        } else {
          int core = std::stoi(node_cpus);
          bitmask[core / 64] |= (1ULL << (core % 64));
        }
      }
    }
  }
  return bitmask;
}

namespace gpu_device::details {

using UALoeLinkTypeMap_t = std::map<std::string_view, amdsmi_fabric_type_t>;
inline const auto UALoeLinkTypeMap =
    UALoeLinkTypeMap_t{{"ualoe", amdsmi_fabric_type_t::AMDSMI_FABRIC_TYPE_UALOE},
                       {"ualink", amdsmi_fabric_type_t::AMDSMI_FABRIC_TYPE_UALLINK}};

using UALoeAddressModeTypeMap_t = std::map<std::string_view, amdsmi_fabric_npa_address_mode_t>;
inline const auto UALoeAddressModeTypeMap = UALoeAddressModeTypeMap_t{
    {"aliasing", amdsmi_fabric_npa_address_mode_t::AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING},
    {"identification",
     amdsmi_fabric_npa_address_mode_t::AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION}};

using UALoeAcceleratorStateTypeMap_t =
    std::map<std::string_view, amdsmi_fabric_accelerator_vpod_state_t>;
inline const auto UALoeAcceleratorStateTypeMap = UALoeAcceleratorStateTypeMap_t{
    {"unconfigured",
     amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED},
    {"configured",
     amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED},
    {"ready", amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY},
    {"active", amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE},
    {"error", amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR}};

// Default populated values for all link info types, with max values for each field
using EnumDefaultType_t = std::int32_t;
using AcceleratorArrayType_t = std::uint32_t;
using PPodIDType_t = std::uint8_t;

auto log_ualoe_file_info(std::ostringstream&& log_stream_info) -> void {
  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " Log stream: |";
  outstream << log_stream_info.str();
  LOG_DEBUG(outstream);
}

// Set values based on link info type
auto get_fabric_type(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info)
    -> void {
  // Convert the string to lowercase
  auto fabric_type_str = link_value;
  std::transform(fabric_type_str.begin(), fabric_type_str.end(), fabric_type_str.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  const std::string_view fabric_type_sv{fabric_type_str};
  if (auto fabric_type_itr = UALoeLinkTypeMap.find(fabric_type_sv);
      fabric_type_itr != UALoeLinkTypeMap.end()) {
    local_fabric_info.fabric_info.fabric_version.v1.fabric_type = fabric_type_itr->second;
  } else {
    local_fabric_info.fabric_info.fabric_version.v1.fabric_type =
        amdsmi_fabric_type_t::AMDSMI_FABRIC_TYPE_UNKNOWN;
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value << " -> " << fabric_type_str
            << " | fabric_type: " << local_fabric_info.fabric_info.fabric_version.v1.fabric_type
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_accelerator_id(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info)
    -> void {
  auto result = parse_number_from_string<
      decltype(local_fabric_info.fabric_info.fabric_version.v1.accelerator_id)>(link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.accelerator_id =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.accelerator_id)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value << " | accelerator_id: "
            << local_fabric_info.fabric_info.fabric_version.v1.accelerator_id << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_bandwidth(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  auto result =
      parse_number_from_string<decltype(local_fabric_info.fabric_info.fabric_version.v1.bandwidth)>(
          link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.bandwidth =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.bandwidth)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value
            << " | bandwidth: " << local_fabric_info.fabric_info.fabric_version.v1.bandwidth
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_latency(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  auto result =
      parse_number_from_string<decltype(local_fabric_info.fabric_info.fabric_version.v1.latency)>(
          link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.latency =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.latency)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value
            << " | latency: " << local_fabric_info.fabric_info.fabric_version.v1.latency << " |";
  log_ualoe_file_info(std::move(outstream));
}

template <std::size_t MaxElems>
auto get_ppod_id(const std::string& link_value, std::array<PPodIDType_t, MaxElems>& local_ppod_id)
    -> void {
  std::fill(std::begin(local_ppod_id), std::end(local_ppod_id), 0);
  if (link_value.empty()) {
    return;
  }

  /*
   *  -  2 hex characters per byte (0x1234567890abcdef)
   *  - 16 bytes per 128-bit UUID (MaxElems)
   */
  constexpr auto kHEX_CHARS_PER_BYTE = std::size_t(2);
  auto value_view = std::string_view(link_value);
  if ((value_view.size() >= kHEX_CHARS_PER_BYTE) && (value_view[0] == '0') &&
      ((value_view[1] == 'x') || (value_view[1] == 'X'))) {
    value_view.remove_prefix(kHEX_CHARS_PER_BYTE);
  }

  auto byte_idx = std::size_t(0);
  auto char_idx = std::size_t(0);
  while ((byte_idx < MaxElems) && ((char_idx + kHEX_CHARS_PER_BYTE) <= value_view.size())) {
    /*
     *  Convert two hex characters to a single byte
     */
    auto higher_nibble = std::uint8_t(0);
    auto lower_nibble = std::uint8_t(0);
    if (std::isxdigit(static_cast<unsigned char>(value_view[char_idx])) &&
        std::isxdigit(static_cast<unsigned char>(value_view[char_idx + 1]))) {
      auto hex_digit = [](char ch) -> std::uint8_t {
        /*
         *  Convert a hex character to its numeric value
         *  - '0' to '9' -> 0 to 9
         *  - 'a' to 'f' -> 10 to 15
         *  - 'A' to 'F' -> 10 to 15
         */
        if ((ch >= '0') && (ch <= '9')) {
          return static_cast<std::uint8_t>(ch - '0');
        }
        if ((ch >= 'a') && (ch <= 'f')) {
          return static_cast<std::uint8_t>(10 + (ch - 'a'));
        }

        return static_cast<std::uint8_t>(10 + (ch - 'A'));
      };

      higher_nibble = hex_digit(value_view[char_idx]);
      lower_nibble = hex_digit(value_view[char_idx + 1]);
      local_ppod_id[byte_idx++] = static_cast<PPodIDType_t>((higher_nibble << 4) | lower_nibble);
    }

    char_idx += kHEX_CHARS_PER_BYTE;

    /*
     *  Skip single separators (e.g. '-' in UUID) between its bytes
     */
    while ((char_idx < value_view.size()) &&
           ((value_view[char_idx] == '-') ||
            (std::isspace(static_cast<unsigned char>(value_view[char_idx]))))) {
      ++char_idx;
    }
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value << " | local_ppod_id (hex): ";
  for (auto idx = std::size_t(0); idx < byte_idx; ++idx) {
    outstream << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned int>(local_ppod_id[idx]);
  }
  outstream << std::dec;
  outstream << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_ppod_size(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  auto result =
      parse_number_from_string<decltype(local_fabric_info.fabric_info.fabric_version.v1.ppod_size)>(
          link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.ppod_size =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.ppod_size)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value
            << " | ppod_size: " << local_fabric_info.fabric_info.fabric_version.v1.ppod_size
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_vpod_id(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  auto result =
      parse_number_from_string<decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_id)>(
          link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.vpod_id =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_id)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value
            << " | vpod_id: " << local_fabric_info.fabric_info.fabric_version.v1.vpod_id << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_vpod_size(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  auto result =
      parse_number_from_string<decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_size)>(
          link_value);
  if (result.has_value()) {
    local_fabric_info.fabric_info.fabric_version.v1.vpod_size =
        static_cast<decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_size)>(
            result.value());
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value
            << " | vpod_size: " << local_fabric_info.fabric_info.fabric_version.v1.vpod_size
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

/*
 *  Note:   For get_vpod_active_accelerators(), and get_local_accelerators(),
 *          we handle all lines from the file at once, and then fill the array.
 */
template <std::size_t MaxElems>
auto get_vpod_active_accelerators(
    const UALoeLinkInfoLines_t& file_lines,
    std::array<AcceleratorArrayType_t, MaxElems>& local_active_accelerators) -> void {
  std::fill(std::begin(local_active_accelerators), std::end(local_active_accelerators),
            std::numeric_limits<AcceleratorArrayType_t>::max());

  auto idx = std::size_t(0);
  auto accel_count = std::size_t(0);
  for (const auto& line : file_lines) {
    if (idx >= MaxElems) {
      break;
    }

    // We are supposed to only have 1 line per file, and separated by spaces
    auto accel_vector = split_string(line, ' ');

    if (!accel_vector.empty()) {
      accel_count = accel_vector.size();
      for (const auto& accel_value_str : accel_vector) {
        if (idx < MaxElems) {
          auto result = parse_number_from_string<AcceleratorArrayType_t>(accel_value_str);
          if (result.has_value()) {
            local_active_accelerators[idx++] = static_cast<AcceleratorArrayType_t>(result.value());
          }
        }
      }
    }
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | File lines: " << file_lines.size()
            << " | local_active_accelerators_count: " << accel_count
            << "local_active_accelerators: ";
  std::copy(std::begin(local_active_accelerators), std::end(local_active_accelerators),
            amd::smi::make_ostream_joiner(&outstream, ", "));
  outstream << " |";
  log_ualoe_file_info(std::move(outstream));
}

template <std::size_t MaxElems>
auto get_local_accelerators(const UALoeLinkInfoLines_t& file_lines,
                            std::array<AcceleratorArrayType_t, MaxElems>& local_accelerators)
    -> void {
  std::fill(std::begin(local_accelerators), std::end(local_accelerators),
            std::numeric_limits<AcceleratorArrayType_t>::max());

  auto idx = std::size_t(0);
  auto accel_count = std::size_t(0);
  for (const auto& line : file_lines) {
    if (idx >= MaxElems) {
      break;
    }

    // We are supposed to only have 1 line per file, and separated by spaces
    auto accel_vector = split_string(line, ' ');
    if (!accel_vector.empty()) {
      accel_count = accel_vector.size();
      for (const auto& accel_value_str : accel_vector) {
        if (idx < MaxElems) {
          auto result = parse_number_from_string<AcceleratorArrayType_t>(accel_value_str);
          if (result.has_value()) {
            local_accelerators[idx++] = static_cast<AcceleratorArrayType_t>(result.value());
          }
        }
      }
    }
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | File lines: " << file_lines.size()
            << " | local_accelerators_count: " << accel_count << " | local_accelerators: ";
  std::copy(std::begin(local_accelerators), std::end(local_accelerators),
            amd::smi::make_ostream_joiner(&outstream, ", "));
  outstream << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_addr_mode(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info) -> void {
  // Convert the string to lowercase
  auto addr_mode_str = link_value;
  std::transform(addr_mode_str.begin(), addr_mode_str.end(), addr_mode_str.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  const std::string_view addr_mode_sv{addr_mode_str};
  if (auto addr_mode_itr = UALoeAddressModeTypeMap.find(addr_mode_sv);
      addr_mode_itr != UALoeAddressModeTypeMap.end()) {
    local_fabric_info.fabric_info.fabric_version.v1.addr_mode = addr_mode_itr->second;
  } else {
    local_fabric_info.fabric_info.fabric_version.v1.addr_mode =
        amdsmi_fabric_npa_address_mode_t::AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN;
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value << " -> " << addr_mode_str
            << " | addr_mode: " << local_fabric_info.fabric_info.fabric_version.v1.addr_mode
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto get_accel_state(const std::string& link_value, amdsmi_fabric_info_t& local_fabric_info)
    -> void {
  // Convert the string to lowercase
  auto accel_state_str = link_value;
  std::transform(accel_state_str.begin(), accel_state_str.end(), accel_state_str.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  const std::string_view accel_state_sv{accel_state_str};
  if (auto accel_state_itr = UALoeAcceleratorStateTypeMap.find(accel_state_sv);
      accel_state_itr != UALoeAcceleratorStateTypeMap.end()) {
    local_fabric_info.fabric_info.fabric_version.v1.accel_state = accel_state_itr->second;
  } else {
    local_fabric_info.fabric_info.fabric_version.v1.accel_state =
        amdsmi_fabric_accelerator_vpod_state_t::AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN;
  }

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | Link value: " << link_value << " -> " << accel_state_str
            << " | accel_state: " << local_fabric_info.fabric_info.fabric_version.v1.accel_state
            << " |";
  log_ualoe_file_info(std::move(outstream));
}

auto read_fabric_info_file(const std::string& sysfs_file_path,
                           UALoeLinkInfoLines_t& sysfs_file_lines) -> amdsmi_status_t {
  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(outstream);

  auto status_code(amdsmi_status_t::AMDSMI_STATUS_SUCCESS);
  std::ifstream ualoe_file(sysfs_file_path);
  if (!ualoe_file.is_open()) {
    auto rsmi_status = amd::smi::ErrnoToRsmiStatus(errno);
    errno = 0;
    outstream << __PRETTY_FUNCTION__ << " | Failed to read SYSFS file: " << sysfs_file_path
              << " | Error: " << rsmi_status << " (" << amd::smi::getRSMIStatusString(rsmi_status)
              << ")";
    LOG_DEBUG(outstream);
    status_code = amd::smi::rsmi_to_amdsmi_status(rsmi_status);
    return status_code;
  }

  /*
   *  Read all lines from the ualoe file into a vector
   *  For most files we only have a single line, but we have exceptions (e.g. vpod_active_accels)
   */
  auto ualoe_file_line = UALoeLinkInfoLine_t();
  while (std::getline(ualoe_file, ualoe_file_line)) {
    ualoe_file_line = amd::smi::trim(ualoe_file_line);
    if (!ualoe_file_line.empty()) {
      sysfs_file_lines.push_back(ualoe_file_line);
    }
  }

  outstream << __PRETTY_FUNCTION__ << " | Read SYSFS file: " << sysfs_file_path
            << " | Lines: " << sysfs_file_lines.size() << " |";
  LOG_TRACE(outstream);

  if (sysfs_file_lines.empty()) {
    status_code = amdsmi_status_t::AMDSMI_STATUS_NO_DATA;
  }

  return status_code;
}

/*
 *  Scan "/sys/bus/pci/drivers/ifoe/" for PCI device symlink names (BDF form)
 *  For entries where the domain, bus, and device match gpu_bdf (same physical PCIe device / slot;
 * any function), are inserted into fabric_bdf_list with the full BDF including function from sysfs
 */
auto populate_ifoe_fabric_bdf_list(const amdsmi_bdf_t& gpu_bdf, FabricBDFList_t& fabric_bdf_list)
    -> amdsmi_status_t {
  auto status_code = amdsmi_status_t::AMDSMI_STATUS_SUCCESS;

  const auto ifoe_driver_directory = std::filesystem::path{kIFOE_DRIVER_BASE_PATH};
  std::error_code err_code;
  if (!std::filesystem::exists(ifoe_driver_directory, err_code) ||
      !std::filesystem::is_directory(ifoe_driver_directory, err_code)) {
    status_code = amdsmi_status_t::AMDSMI_STATUS_DIRECTORY_NOT_FOUND;
    return status_code;
  }

  fabric_bdf_list.clear();
  const auto dir_itr = std::filesystem::directory_iterator{
      ifoe_driver_directory, std::filesystem::directory_options::skip_permission_denied, err_code};
  if (err_code) {
    status_code = amdsmi_status_t::AMDSMI_STATUS_FILE_ERROR;
    return status_code;
  }

  for (const auto& entry : dir_itr) {
    const auto entry_name = entry.path().filename().string();
    const auto parsed_bdf = from_cstring_to_bdf(entry_name.c_str());
    if (!parsed_bdf.has_value()) {
      continue;
    }

    const auto& parsed_bdf_value = parsed_bdf.value();
    if ((parsed_bdf_value.domain_number == gpu_bdf.domain_number) &&
        (parsed_bdf_value.bus_number == gpu_bdf.bus_number) &&
        (parsed_bdf_value.device_number == gpu_bdf.device_number)) {
      fabric_bdf_list.insert(parsed_bdf_value);
    }
  }

  return status_code;
}

}  // namespace gpu_device::details

auto AMDSmiGPUDevice::populate_ifoe_fabric_bdf_list() -> void {
  static_cast<void>(gpu_device::details::populate_ifoe_fabric_bdf_list(bdf_, fabric_bdf_list_));

  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_DEBUG(outstream);

  if (fabric_bdf_list_.empty()) {
    // No IFoE driver loaded or no matching BDF: UALoE open will be skipped
    // and fabric queries will report NOT_SUPPORTED. Surface the reason at
    // INFO so the contract is visible without enabling debug logging.
    outstream << __PRETTY_FUNCTION__ << " | No IFoE BDF found under " << kIFOE_DRIVER_BASE_PATH
              << " matching GPU BDF " << stringify_bdf(bdf_)
              << " | UALoE access disabled for this device";
    LOG_INFO(outstream);
    return;
  }

  outstream << __PRETTY_FUNCTION__ << " | fabric_bdf_list_size: " << fabric_bdf_list_.size()
            << " |";
  auto first_bdf = true;
  for (const auto& fabric_bdf : fabric_bdf_list_) {
    if (!first_bdf) {
      outstream << ", ";
    }
    first_bdf = false;
    outstream << stringify_bdf(fabric_bdf);
  }

  outstream << " |";
  LOG_DEBUG(outstream);
}

auto AMDSmiGPUDevice::has_ifoe_related_bdf() const -> bool { return (!fabric_bdf_list_.empty()); }

auto AMDSmiGPUDevice::get_ifoe_bdf_string() const -> std::string {
  if (has_ifoe_related_bdf()) {
    auto ifoe_bdf = *fabric_bdf_list_.begin();
    return stringify_bdf(ifoe_bdf);
  }

  return {};
}

// Check if this GPU device has ualink sysfs directory (required for UALOE)
// Uses the same path construction as get_fabric_info_from_ualoe()
bool AMDSmiGPUDevice::device_has_ualink() const {
  if (!has_ifoe_related_bdf()) {
    return false;
  }

  // Use the same ualink path logic as get_fabric_info_from_ualoe()
  const auto ualink_directory =
      (std::string(kUALOE_BASE_PATH) + get_gpu_path() + std::string(kUALOE_UALINK_DIRECTORY));
  const auto ualink_directory_path = std::filesystem::path(ualink_directory);

  return std::filesystem::is_directory(ualink_directory_path);
}

auto AMDSmiGPUDevice::get_fabric_info_from_ualoe(amdsmi_fabric_info_t& fabric_info,
                                                 UALoeLinkInfo_t link_info_type) const
    -> amdsmi_status_t {
  std::ostringstream outstream;
  outstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(outstream);

  auto local_fabric_info = amdsmi_fabric_info_t{};
  auto local_ppod_id = std::array<std::uint8_t, AMDSMI_MAX_UUID_ELEMENTS>{};
  auto local_active_accelerators = std::array<gpu_device::details::AcceleratorArrayType_t,
                                              AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE>{};
  auto local_accelerators =
      std::array<gpu_device::details::AcceleratorArrayType_t, AMDSMI_FABRIC_MAX_LOCAL_GPUS>{};

  if (has_ifoe_related_bdf()) {
    local_fabric_info.bdf = *fabric_bdf_list_.begin();
  } else {
    local_fabric_info.bdf = bdf_;
  }

  /*
   *  TODO: We need to define a which offset (kUALOE_BDF_OFFSET) will be used for the function
   * number (in BDF).
   *
   *  - PCIe BDF format:
   *      - Bits 31-24: Bus number (8 bits)
   *      - Bits 23-16: Device number (5 bits, but stored in 8 bits)
   *      - Bits  15-8: Function number (3 bits, but stored in 8 bits)
   *      - Bits   7-0: Other info
   *
   *  - 0x07 is the mask for the function number (extracts only the lower 3 bits)
   *      - 00000111 (last 3 bits are 1, all others 0)
   */
  // local_fabric_info.bdf.function_number = ((bdf_.function_number + kUALOE_BDF_OFFSET) & 0x07);
  local_fabric_info.fabric_info.version =
      std::numeric_limits<decltype(local_fabric_info.fabric_info.version)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.fabric_type = static_cast<amdsmi_fabric_type_t>(
      std::numeric_limits<
          decltype(local_fabric_info.fabric_info.fabric_version.v1.fabric_type)>::max());

  local_fabric_info.fabric_info.fabric_version.v1.accelerator_id = std::numeric_limits<
      decltype(local_fabric_info.fabric_info.fabric_version.v1.accelerator_id)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.bandwidth = std::numeric_limits<
      decltype(local_fabric_info.fabric_info.fabric_version.v1.bandwidth)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.latency =
      std::numeric_limits<decltype(local_fabric_info.fabric_info.fabric_version.v1.latency)>::max();

  // Sentinel when sysfs provides no ppod_id: UUID 99999999-9999-9999-9999-999999999999 (16 × 0x99)
  std::fill(std::begin(local_fabric_info.fabric_info.fabric_version.v1.ppod_id),
            std::end(local_fabric_info.fabric_info.fabric_version.v1.ppod_id),
            static_cast<std::uint8_t>(0x99));

  local_fabric_info.fabric_info.fabric_version.v1.ppod_size = std::numeric_limits<
      decltype(local_fabric_info.fabric_info.fabric_version.v1.ppod_size)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.vpod_id =
      std::numeric_limits<decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_id)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.vpod_size = std::numeric_limits<
      decltype(local_fabric_info.fabric_info.fabric_version.v1.vpod_size)>::max();

  local_fabric_info.fabric_info.fabric_version.v1.addr_mode =
      static_cast<amdsmi_fabric_npa_address_mode_t>(
          std::numeric_limits<
              decltype(local_fabric_info.fabric_info.fabric_version.v1.addr_mode)>::max());

  local_fabric_info.fabric_info.fabric_version.v1.accel_state =
      static_cast<amdsmi_fabric_accelerator_vpod_state_t>(
          std::numeric_limits<
              decltype(local_fabric_info.fabric_info.fabric_version.v1.accel_state)>::max());

  std::fill(std::begin(local_active_accelerators), std::end(local_active_accelerators),
            std::numeric_limits<gpu_device::details::AcceleratorArrayType_t>::max());
  std::fill(std::begin(local_accelerators), std::end(local_accelerators),
            std::numeric_limits<gpu_device::details::AcceleratorArrayType_t>::max());
  std::fill(std::begin(local_ppod_id), std::end(local_ppod_id), static_cast<std::uint8_t>(0));

  /**
   * Check if the 'ualink' directory exists in the sysfs path, if not, return
   * AMDSMI_STATUS_NOT_SUPPORTED. Use get_gpu_path() (the DRM card directory name resolved via
   * libdrm), matching the convention used elsewhere in the codebase (see amd_smi_utils.cc). gpu_id_
   * is the AMDSMI processor index and is not guaranteed to equal the DRM card index when devices
   * are gapped or filtered.
   */
  const auto ualink_directory =
      (std::string(kUALOE_BASE_PATH) + get_gpu_path() + std::string(kUALOE_UALINK_DIRECTORY));
  const auto ualink_directory_path = std::filesystem::path(ualink_directory);

  outstream << __PRETTY_FUNCTION__ << " | gpu_path: " << get_gpu_path()
            << " | ualink_directory_path: " << ualink_directory_path
            << " | ualink_directory: " << ualink_directory << " |";
  LOG_DEBUG(outstream);

  if (!std::filesystem::is_directory(ualink_directory_path)) {
    outstream << __PRETTY_FUNCTION__
              << " | UALOE sysfs not present: " << ualink_directory_path.string()
              << " | returning AMDSMI_STATUS_NOT_SUPPORTED";
    LOG_DEBUG(outstream);
    fabric_info = local_fabric_info;
    return amdsmi_status_t::AMDSMI_STATUS_NOT_SUPPORTED;
  }

  auto link_info_files_in_scope = std::size_t(0);
  auto link_info_files_with_usable_content = std::size_t(0);
  for (const auto& [link_info_type_key, link_info_file_value] : UALoeLinkInfoMap) {
    // If not all, apply filter based on the link info type
    if ((link_info_type != UALoeLinkInfo_t::ALL_LINK_INFO) &&
        (link_info_type_key != link_info_type)) {
      continue;
    }

    ++link_info_files_in_scope;
    const auto sysfs_file_path = (ualink_directory + "/" + std::string(link_info_file_value));
    outstream << __PRETTY_FUNCTION__ << " | sysfs_file_path: " << sysfs_file_path << " |";
    LOG_DEBUG(outstream);

    auto ualoe_file_lines = UALoeLinkInfoLines_t{};
    const auto read_status =
        gpu_device::details::read_fabric_info_file(sysfs_file_path, ualoe_file_lines);
    if (ualoe_file_lines.empty()) {
      outstream << __PRETTY_FUNCTION__
                << " | Skipping sysfs file (no usable content): " << sysfs_file_path
                << " | read_status: " << read_status << " |";
      LOG_DEBUG(outstream);
      continue;
    }

    ++link_info_files_with_usable_content;
    outstream << __PRETTY_FUNCTION__ << " | UALOE File: " << sysfs_file_path
              << " | Lines: " << ualoe_file_lines.size()
              << " | Link Info Type: " << static_cast<UALoeLinkInfoType_t>(link_info_type_key)
              << " |";
    LOG_DEBUG(outstream);

    /*
     *  List-based fields: one value per line; pass all lines at once
     *      - VPOD_ACTIVE_ACCELS
     *      - LOCAL_ACCELS
     */
    if (link_info_type_key == UALoeLinkInfo_t::VPOD_ACTIVE_ACCELS) {
      gpu_device::details::get_vpod_active_accelerators(ualoe_file_lines,
                                                        local_active_accelerators);
      std::copy(local_active_accelerators.data(),
                (local_active_accelerators.data() + local_active_accelerators.size()),
                local_fabric_info.fabric_info.fabric_version.v1.vpod_active_accelerators);
      continue;
    }
    if (link_info_type_key == UALoeLinkInfo_t::LOCAL_ACCELS) {
      gpu_device::details::get_local_accelerators(ualoe_file_lines, local_accelerators);
      std::copy(local_accelerators.data(), (local_accelerators.data() + local_accelerators.size()),
                local_fabric_info.fabric_info.fabric_version.v1.local_accelerators);
      continue;
    }

    for (const auto& link_value : ualoe_file_lines) {
      switch (link_info_type_key) {
        case UALoeLinkInfo_t::LINK_TYPE:
          gpu_device::details::get_fabric_type(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::ACCEL_ID:
          gpu_device::details::get_accelerator_id(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::BANDWIDTH:
          gpu_device::details::get_bandwidth(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::LATENCY:
          gpu_device::details::get_latency(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::PPOD_ID:
          gpu_device::details::get_ppod_id(link_value, local_ppod_id);
          std::copy(local_ppod_id.data(), (local_ppod_id.data() + local_ppod_id.size()),
                    local_fabric_info.fabric_info.fabric_version.v1.ppod_id);
          break;

        case UALoeLinkInfo_t::PPOD_SIZE:
          gpu_device::details::get_ppod_size(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::VPOD_ID:
          gpu_device::details::get_vpod_id(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::VPOD_SIZE:
          gpu_device::details::get_vpod_size(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::ADDR_MODE:
          gpu_device::details::get_addr_mode(link_value, local_fabric_info);
          break;

        case UALoeLinkInfo_t::ACCEL_STATE:
          gpu_device::details::get_accel_state(link_value, local_fabric_info);
          break;

        default:
          break;
      }
    }
  }

  /**
   * For cases where the 'ualink' directory exists in the sysfs path, but we can read any usable
   * content, return AMDSMI_STATUS_NO_DATA.
   */
  fabric_info = local_fabric_info;
  if (link_info_files_in_scope == 0) {
    return amdsmi_status_t::AMDSMI_STATUS_NO_DATA;
  }

  /**
   * For cases where the 'ualink' directory exists in the sysfs path, and we can read all files with
   * usable content, return AMDSMI_STATUS_SUCCESS. Otherwise, return AMDSMI_STATUS_UNEXPECTED_DATA
   * (some files are missing or empty).
   */
  return (link_info_files_with_usable_content == link_info_files_in_scope)
             ? amdsmi_status_t::AMDSMI_STATUS_SUCCESS
             : amdsmi_status_t::AMDSMI_STATUS_UNEXPECTED_DATA;
}

}  // namespace amd::smi
