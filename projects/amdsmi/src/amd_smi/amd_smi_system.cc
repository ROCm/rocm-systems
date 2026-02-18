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

#include <cassert>
#include <dirent.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cerrno>
#include <cstring>
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#ifdef BRCM_NIC
#include "amd_smi/impl/nic/amd_smi_nic_device.h"
#include "amd_smi/impl/nic/amd_smi_switch_device.h"
#endif//BRCM_NIC
#include "amd_smi/impl/amd_smi_utils.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_logger.h"
#include <map>
#include <algorithm>
#include <regex>


namespace amd::smi {


#define  AMD_SMI_INIT_FLAG_RESRV_TEST1 0x800000000000000  //!< Reserved for test

AMDSmiSystem& AMDSmiSystem::getInstance() {
    static AMDSmiSystem instance;
    return instance;
}

const std::map<int, std::string> smi_nic_status_str = {
	{SMI_NIC_STATUS_SUCCESS, "API completed successfully"},
	{SMI_NIC_STATUS_ERROR, "Generic error"},
	{SMI_NIC_STATUS_WRONG_PARAM, "Wrong parameter provided"},
	{SMI_NIC_STATUS_NOT_FOUND, "NIC not found"},
	{SMI_NIC_STATUS_NO_RESOURCE, "Memory allocation failed"},
	{SMI_NIC_STATUS_NOT_SUPPORTED, "API not supported"},
	{SMI_NIC_STATUS_NOT_INIT, "Not initialized"},
	{SMI_NIC_STATUS_NO_DATA, "Requested data not found"},
	{SMI_NIC_STATUS_DRIVER_NOT_LOADED, "Required driver not loaded"},
};


#define CHK_AMDNIC_RET(status) \
    if (status != SMI_NIC_STATUS_SUCCESS) { \
        std::ostringstream ss; \
        ss << __PRETTY_FUNCTION__ \
            <<  "[" << __FILE__ << ":" << __LINE__ << "] smi_nic_status_t: " << status << ":" << smi_nic_status_str.at(status) << std::endl; \
        LOG_INFO(ss); \
        return amd::smi::ainic_to_amdsmi_status(status); \
    }

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t AMDSmiSystem::get_cpu_family(uint32_t *cpu_family) {
    amdsmi_status_t ret;
    ret = static_cast<amdsmi_status_t>(esmi_cpu_family_get(cpu_family));

    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "Failed to get cpu family, Err["<<ret<<"]" << std::endl;
        return ret;
    }
    return AMDSMI_STATUS_SUCCESS;
 }


amdsmi_status_t AMDSmiSystem::get_cpu_model(uint32_t *cpu_model) {
    amdsmi_status_t ret;
    ret = static_cast<amdsmi_status_t>(esmi_cpu_model_get(cpu_model));

    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "Failed to get cpu model, Err["<<ret<<"]" << std::endl;
        return ret;
    }
    return AMDSMI_STATUS_SUCCESS;
}

static amdsmi_status_t get_nr_cpu_cores(uint32_t *num_cpus) {
    amdsmi_status_t ret;
    ret = static_cast<amdsmi_status_t>(esmi_number_of_cpus_get(num_cpus));

    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "Failed to get number of cpus, Err["<<ret<<"]" << std::endl;
        return ret;
    }
    return AMDSMI_STATUS_SUCCESS;
}

static amdsmi_status_t get_nr_threads_per_core(uint32_t *threads_per_core) {
    amdsmi_status_t ret;
    ret = static_cast<amdsmi_status_t>(esmi_threads_per_core_get(threads_per_core));

    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "Failed to get threads per core, Err["<<ret<<"]" << std::endl;
        return ret;
    }
    return AMDSMI_STATUS_SUCCESS;
}

static amdsmi_status_t get_nr_cpu_sockets(uint32_t *num_socks) {
    amdsmi_status_t ret;
    ret = static_cast<amdsmi_status_t>(esmi_number_of_sockets_get(num_socks));

    if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "Failed to get number of sockets, Err["<<ret<<"]" << std::endl;
        return ret;
    }
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSystem::get_cpu_model_name(uint32_t socket_id, std::string *model_name) {
    std::ifstream cpu_info("/proc/cpuinfo");
    std::string info;
    std::map<uint32_t, std::string> socket_model_map;

    if (!cpu_info.is_open()) {
      std::cerr << "Failed to open /proc/cpuinfo:" << strerror(errno) << std::endl;
      return AMDSMI_STATUS_FILE_ERROR;
    } else {
      int current_socket_id = -1;
      while (std::getline(cpu_info, info)) {
        if (info.find("processor") != std::string::npos) {
          current_socket_id = std::stoi(info.substr(info.find(':') + 1));
        }
        if (info.find("model name") != std::string::npos) {
          *model_name = info.substr(info.find(':') + 2);
          if (current_socket_id != -1) {
            socket_model_map[current_socket_id] = *model_name;
          }
        }
      }
      cpu_info.close();
    }

    if (socket_model_map.find(socket_id) != socket_model_map.end()) {
      *model_name = socket_model_map[socket_id];
    } else {
      return AMDSMI_STATUS_NO_DATA;
    }
    return AMDSMI_STATUS_SUCCESS;
}

#endif

amdsmi_status_t  AMDSmiSystem::get_sys_cpu_cores_per_socket(uint32_t *core_num) {
    std::map<uint32_t, uint32_t> socket_core_count;
    std::string base_path = "/sys/devices/system/cpu/";

    DIR* dir = opendir(base_path.c_str());
    if (dir == nullptr) {
        return AMDSMI_STATUS_FILE_ERROR;
    }

    uint32_t physical_id, core_id;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string file_name = entry->d_name;
        if (file_name.find("cpu") == 0) {
            std::string cpu_path = base_path + file_name;
            std::ifstream package_id_file(cpu_path + "/topology/physical_package_id");
            std::ifstream core_id_file(cpu_path + "/topology/core_id");

            if (package_id_file.is_open() && core_id_file.is_open()) {
                package_id_file >> physical_id;
                core_id_file >> core_id;

                socket_core_count[physical_id]++;
            }
        }
    }

    closedir(dir);

    if (socket_core_count.find(physical_id) != socket_core_count.end()) {
      *core_num = socket_core_count[physical_id];
    } else {
      return AMDSMI_STATUS_NO_DATA;
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSystem::get_sys_num_of_cpu_sockets(uint32_t *sock_num) {
    std::map<uint32_t, uint32_t> socket_count_map;
    std::string base_path = "/sys/devices/system/cpu/";

    DIR* dir = opendir(base_path.c_str());
    if (dir == nullptr) {
        return AMDSMI_STATUS_API_FAILED;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string file_name = entry->d_name;
        if (file_name.find("cpu") == 0) {
            std::string path = base_path + file_name;
            std::ifstream package_id_file(path + "/topology/physical_package_id");

            if (package_id_file.is_open()) {
                uint32_t physical_id;
                package_id_file >> physical_id;

                socket_count_map[physical_id]++;
            }
        }
    }

    closedir(dir);

    *sock_num = static_cast<uint32_t>(socket_count_map.size());

    return AMDSMI_STATUS_SUCCESS;
}

std::vector<uint32_t> AMDSmiSystem::get_cpu_sockets_from_numa_node(int32_t numa_node) {
    std::vector<uint32_t> sockets;
    if (numa_node < 0) {
        sockets[0] = std::numeric_limits<int32_t>::max();
        return sockets;
    }
    std::ifstream node_info("/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist");
    std::string info;

    if (node_info.is_open()) {
        std::getline(node_info, info);
        std::istringstream iss(info);
        uint32_t index;
        while (iss >> index) {
            std::ifstream cpu_info("/sys/devices/system/cpu/cpu" + std::to_string(index) + "/topology/physical_package_id");
            if (cpu_info.is_open()) {
                uint32_t socket;
                cpu_info >> socket;
                sockets.push_back(socket);
            }
        }
    }

    // Discarding duplicate socket entries
    std::sort(sockets.begin(), sockets.end());
    sockets.erase(std::unique(sockets.begin(), sockets.end()), sockets.end());

    return sockets;
}

amdsmi_status_t AMDSmiSystem::init(uint64_t flags) {
    init_flag_ = flags;
    amdsmi_status_t amd_smi_status;

    // populate GPU sockets and processors
    if (flags & AMDSMI_INIT_AMD_GPUS) {
        amd_smi_status = populate_amd_gpu_devices();
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS)
            return amd_smi_status;
    }
#ifdef ENABLE_ESMI_LIB
    // populate CPU sockets and processors
    if (flags & AMDSMI_INIT_AMD_CPUS) {
        amd_smi_status = populate_amd_cpus();
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS)
            return amd_smi_status;
    }
#endif
#ifdef BRCM_NIC
    if (flags & AMDSMI_INIT_AMD_NICS) {
        amd_smi_status = populate_brcm_nic_devices();
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS)
            return amd_smi_status;
        amd_smi_status = populate_brcm_switch_devices();
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS)
            return amd_smi_status;
        amd_smi_status = populate_amd_ainic_devices();
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS)
            return amd_smi_status;
    }
#endif
    return AMDSMI_STATUS_SUCCESS;
}

#ifdef ENABLE_ESMI_LIB
amdsmi_status_t AMDSmiSystem::populate_amd_cpus() {
    uint32_t sockets, cpus, threads;
    amdsmi_status_t amd_smi_status;

    /* esmi is for AMD cpus, if its not AMD CPU, we are not going to initialise esmi */
    amd_smi_status = static_cast<amdsmi_status_t>(esmi_init());
    if (amd_smi_status != AMDSMI_STATUS_SUCCESS){
        std::cout<<"\tESMI Not initialized, drivers not found " << std::endl;
        return amd_smi_status;
    }

    amd_smi_status = get_nr_cpu_sockets(&sockets);
    amd_smi_status = get_nr_cpu_cores(&cpus);
    amd_smi_status = get_nr_threads_per_core(&threads);

    for(uint32_t i = 0; i < sockets; i++) {
        std::string cpu_socket_id = std::to_string(i);
        // Multiple cores may share the same socket
        AMDSmiSocket* socket = nullptr;
        for (uint32_t j = 0; j < sockets_.size(); j++) {
            if (sockets_[j]->get_socket_id() == cpu_socket_id) {
                socket = sockets_[j];
                break;
            }
        }
        if (socket == nullptr) {
            socket = new AMDSmiSocket(cpu_socket_id);
            sockets_.push_back(socket);
        }
        AMDSmiProcessor* cpusocket = new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_CPU, i);
        socket->add_processor(cpusocket);
        processors_.insert(cpusocket);

       for (uint32_t k = 0; k < (cpus/threads)/sockets; k++) {
            AMDSmiProcessor* core = new AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, k);
            socket->add_processor(core);
            processors_.insert(core);
       }
    }

    return AMDSMI_STATUS_SUCCESS;
}
#endif

amdsmi_status_t AMDSmiSystem::populate_amd_gpu_devices() {
    AMDSmiSystem::cleanup();
    // init rsmi
    rsmi_driver_state_t state;
    rsmi_status_t ret = rsmi_init(0);
    if (ret != RSMI_STATUS_SUCCESS) {
        if (rsmi_driver_status(&state) == RSMI_STATUS_SUCCESS &&
                state != RSMI_DRIVER_MODULE_STATE_LIVE) {
            return AMDSMI_STATUS_DRIVER_NOT_LOADED;
        }
        return amd::smi::rsmi_to_amdsmi_status(ret);
    }

    // The init of libdrm depends on rsmi_init
    // libdrm is optional, ignore the error even if init fail.
    amdsmi_status_t amd_smi_status = drm_.init();

    uint32_t device_count = 0;
    ret = rsmi_num_monitor_devices(&device_count);
    if (ret != RSMI_STATUS_SUCCESS) {
        return amd::smi::rsmi_to_amdsmi_status(ret);
    }

    for (uint32_t i=0; i < device_count; i++) {
        // GPU device uses the bdf as the socket id
        std::string socket_id;
        amd_smi_status = get_gpu_socket_id(i, socket_id);
        if (amd_smi_status != AMDSMI_STATUS_SUCCESS) {
            return amd_smi_status;
        }

        // Multiple devices may share the same socket
        AMDSmiSocket* socket = nullptr;
        for (unsigned int j=0; j < sockets_.size(); j++) {
            if (sockets_[j]->get_socket_id() == socket_id) {
                socket = sockets_[j];
                break;
            }
        }
        if (socket == nullptr) {
            socket = new AMDSmiSocket(socket_id);
            sockets_.push_back(socket);
        }

        AMDSmiProcessor* device = new AMDSmiGPUDevice(i, drm_);
        socket->add_processor(device);
        processors_.insert(device);
    }
    return AMDSMI_STATUS_SUCCESS;
}

static amdsmi_status_t populate_amd_ainic_device(const smi_nic_ctx_t &ctx, uint64_t bdf_int, AMDSmiAINICDevice::AINICInfo &ai_nic_info) {
    static_assert(sizeof(smi_nic_bus_info_t) == sizeof(ai_nic_info.bus));
    smi_nic_status_t status = smi_get_nic_bus_info(ctx, bdf_int, reinterpret_cast<smi_nic_bus_info_t*>(&ai_nic_info.bus));
    CHK_AMDNIC_RET(status)

    static_assert(sizeof(smi_nic_driver_info_t) == sizeof(ai_nic_info.driver));
    status = smi_get_nic_driver_info(ctx, bdf_int, reinterpret_cast<smi_nic_driver_info_t*>(&ai_nic_info.driver));
    CHK_AMDNIC_RET(status)

    static_assert(sizeof(smi_nic_asic_info_t) == sizeof(ai_nic_info.asic));
    status = smi_get_nic_asic_info(ctx, bdf_int, reinterpret_cast<smi_nic_asic_info_t*>(&ai_nic_info.asic));
    CHK_AMDNIC_RET(status)

    static_assert(sizeof(smi_nic_numa_info_t) == sizeof(ai_nic_info.numa));
    status = smi_get_nic_numa_info(ctx, bdf_int, reinterpret_cast<smi_nic_numa_info_t*>(&ai_nic_info.numa));
    CHK_AMDNIC_RET(status)

    static_assert(sizeof(smi_nic_port_info_t) == sizeof(ai_nic_info.port));
    status = smi_get_nic_port_info(ctx, bdf_int, reinterpret_cast<smi_nic_port_info_t*>(&ai_nic_info.port));
    CHK_AMDNIC_RET(status);

    static_assert(sizeof(smi_nic_rdma_devices_info_t) == sizeof(ai_nic_info.rdma_dev));
    status = smi_get_nic_rdma_dev_info(ctx, bdf_int, reinterpret_cast<smi_nic_rdma_devices_info_t*>(&ai_nic_info.rdma_dev));
    CHK_AMDNIC_RET(status)

    return AMDSMI_STATUS_SUCCESS;
}

std::tuple<uint64_t, amdsmi_bdf_t> bdf_to_int(const std::string &bdf) {
    std::regex pattern("([0-9a-fA-F]{1,12}):([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\\.([0-9a-fA-F]{1,2})");
    std::smatch matches;
    amdsmi_bdf_t bdf_info = {};
    if (std::regex_search(bdf, matches, pattern)) {
        bdf_info.domain_number = std::stoul(matches[1], nullptr, 16) & 0xffffffffffff;
        bdf_info.bus_number = std::stoul(matches[2], nullptr, 16) & 0xff;
        bdf_info.device_number = std::stoul(matches[3], nullptr, 16) & 0x1f;
        bdf_info.function_number = std::stoul(matches[4], nullptr, 16) & 0x7;
        return {(bdf_info.domain_number << 16) | (bdf_info.bus_number << 8) | (bdf_info.device_number << 3) | (bdf_info.function_number << 0), bdf_info};
    }
    return {0, bdf_info};
}

amdsmi_status_t AMDSmiSystem::populate_amd_ainic_devices() {
    smi_nic_status_t status = smi_nic_create_context(&ainic_ctx_);
    CHK_AMDNIC_RET(status);

    smi_nic_discovery_t discovery = {};
    status = smi_discover_nics(ainic_ctx_, &discovery);
    CHK_AMDNIC_RET(status);

    for(uint32_t nic_idx = 0; nic_idx < discovery.count; ++nic_idx) {
        const char *bdf_str = discovery.devices[nic_idx].bdf;
        auto [bdfid, bdf_info] = bdf_to_int(bdf_str);
        AMDSmiAINICDevice::AINICInfo ai_nic_info = {};
        amdsmi_status_t status = populate_amd_ainic_device(ainic_ctx_, bdfid, ai_nic_info);
        if (status != AMDSMI_STATUS_SUCCESS) {
            return status;
        }
        ai_nic_info_.emplace_back(ai_nic_info);

        auto [domain, bus, device_id, function] = parse_bdfid(bdfid);

        // The BD part of the BDF is used as the socket id as it
        // represents a physical device.
        std::stringstream ss;
        ss << std::setfill('0') << std::uppercase << std::hex << std::setw(4) << domain << ":"
            << std::setw(2) << bus << ":" << std::setw(2) << device_id;
        std::string socket_id = ss.str();

        // Multiple devices may share the same socket
        AMDSmiSocket* socket = nullptr;
        for (unsigned int j = 0; j < sockets_.size(); j++) {
            if (sockets_[j]->get_socket_id() == socket_id) {
            socket = sockets_[j];
            break;
            }
        }
        if (socket == nullptr) {
            socket = new AMDSmiSocket(socket_id);
            sockets_.push_back(socket);
        }

        auto device = std::make_unique<AMDSmiAINICDevice>(nic_idx, bdf_info, ai_nic_info);
        socket->add_processor(device.get());
        ainic_processors_.insert(device.get());
        device.release();
    }
    return AMDSMI_STATUS_SUCCESS;
}
const auto &AMDSmiSystem::get_ai_nic_info() const {
    return ai_nic_info_;
}

amdsmi_status_t AMDSmiSystem::populate_brcm_nic_devices() {
#ifdef BRCM_NIC
    uint32_t device_count = 0;
    amdsmi_status_t amd_smi_status = no_drm_nic.init();
  
    rsmi_status_t ret = rsmi_num_nic_monitor_devices(&device_count);
    if (ret != RSMI_STATUS_SUCCESS) {
      return amd::smi::rsmi_to_amdsmi_status(ret);
    }

    for (uint32_t i = 0; i < device_count; i++) {
      // NIC device uses the bdf as the socket id
      std::string socket_id;
      uint64_t bdfid = 0;
      rsmi_status_t ret = rsmi_nic_dev_pci_id_get(i, &bdfid);
      if (ret != RSMI_STATUS_SUCCESS) {
        continue;
      }

    auto [domain, bus, device_id, function] = parse_bdfid(bdfid);

      // The BD part of the BDF is used as the socket id as it
      // represents a physical device.
      std::stringstream ss;
      ss << std::setfill('0') << std::uppercase << std::hex << std::setw(4) << domain << ":"
          << std::setw(2) << bus << ":" << std::setw(2) << device_id;
      socket_id = ss.str();

      // Multiple devices may share the same socket
      AMDSmiSocket* socket = nullptr;
      for (unsigned int j = 0; j < sockets_.size(); j++) {
        if (sockets_[j]->get_socket_id() == socket_id) {
          socket = sockets_[j];
          break;
        }
      }
      if (socket == nullptr) {
        socket = new AMDSmiSocket(socket_id);
        sockets_.push_back(socket);
      }

      auto [domain_number, bus_number, device_number, function_number] = parse_bdfid(bdfid);
      amdsmi_bdf_t bdf = {
        .function_number = function_number,
        .device_number = device_number,
        .bus_number = bus_number,
        .domain_number = domain_number
      };

      auto device = std::make_unique<AMDSmiNICDevice>(i, bdf, no_drm_nic);

      std::string nicPath;
      if ( (no_drm_nic.get_device_path_by_index(i, &nicPath)) != AMDSMI_STATUS_SUCCESS) continue;
      std::string driverPath = nicPath + "/driver";
      std::string command = "readlink " + driverPath;
      std::string getData;
      if (smi_brcm_execute_cmd_get_data(command, &getData) != AMDSMI_STATUS_SUCCESS) continue;
      if (getData.find("bnxt_en") == std::string::npos) continue;

      socket->add_processor(device.get());
      nic_processors_.insert(deviceget());
      device.release();
    }
#endif//BRCM_NIC
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSystem::populate_brcm_switch_devices() {
#ifdef BRCM_NIC
  uint32_t device_count = 0;
  amdsmi_status_t amd_smi_status = no_drm_switch.init();
  rsmi_status_t ret = rsmi_num_switch_monitor_devices(&device_count);
  if (ret != RSMI_STATUS_SUCCESS) {
    return amd::smi::rsmi_to_amdsmi_status(ret);
  }

  for (uint32_t i = 0; i < device_count; i++) {
    // NIC device uses the bdf as the socket id
    std::string socket_id;
    uint64_t bdfid = 0;
    rsmi_status_t ret = rsmi_switch_dev_pci_id_get(i, &bdfid);
    if (ret != RSMI_STATUS_SUCCESS) {
      // return amd::smi::rsmi_to_amdsmi_status(ret);
      // device might be removed; continue with next device;
      continue;
    }

    auto [domain, bus, device_id, function] = parse_bdfid(bdfid);

    // The BD part of the BDF is used as the socket id as it
    // represents a physical device.
    std::stringstream ss;
    ss << std::setfill('0') << std::uppercase << std::hex << std::setw(4) << domain << ":"
       << std::setw(2) << bus << ":" << std::setw(2) << device_id;
    socket_id = ss.str();

    // Multiple devices may share the same socket
    AMDSmiSocket* socket = nullptr;
    for (unsigned int j = 0; j < sockets_.size(); j++) {
      if (sockets_[j]->get_socket_id() == socket_id) {
        socket = sockets_[j];
        break;
      }
    }
    if (socket == nullptr) {
      socket = new AMDSmiSocket(socket_id);
      sockets_.push_back(socket);
    }

    amdsmi_bdf_t bdf = {};
    bdf.function_number = bdfid & 0x7;
    bdf.device_number = (bdfid >> 3) & 0x1f;
    bdf.bus_number = (bdfid >> 8) & 0xff;
    bdf.domain_number = (bdfid >> 32) & 0xffffffff;

    AMDSmiProcessor* device = new AMDSmiSWITCHDevice(i, bdf, no_drm_switch);
    socket->add_processor(device);
    switch_processors_.insert(device);
  }
#endif
  return AMDSMI_STATUS_SUCCESS;
}
amdsmi_status_t AMDSmiSystem::get_gpu_socket_id(uint32_t index,
            std::string& socket_id) {
    uint64_t bdfid = 0;
    rsmi_status_t ret = rsmi_dev_pci_id_get(index, &bdfid);
    if (ret != RSMI_STATUS_SUCCESS) {
        return amd::smi::rsmi_to_amdsmi_status(ret);
    }

/**
*  | Name         | Field   | KFD property       KFD -> PCIe ID (uint64_t)
*  -------------- | ------- | ---------------- | ---------------------------- |
*  | Domain       | [63:32] | "domain"         | (DOMAIN & 0xFFFFFFFF) << 32  |
*  | Partition id | [31:28] | "location id"    | (LOCATION & 0xF0000000)      |
*  | Reserved     | [27:16] | "location id"    | N/A                          |
*  | Bus          | [15: 8] | "location id"    | (LOCATION & 0xFF00)          |
*  | Device       | [ 7: 3] | "location id"    | (LOCATION & 0xF8)            |
*  | Function     | [ 2: 0] | "location id"    | (LOCATION & 0x7)             |
*/

    auto [domain, bus, device_id, function] = parse_bdfid(bdfid);

    // The BD part of the BDF is used as the socket id as it
    // represents a physical device.
    std::stringstream ss;
    ss << std::setfill('0') << std::uppercase << std::hex
       << std::setw(4) << domain << ":" << std::setw(2) << bus << ":"
       << std::setw(2) << device_id;
    socket_id = ss.str();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSystem::cleanup() {
#ifdef ENABLE_ESMI_LIB
    if (init_flag_ & AMDSMI_INIT_AMD_CPUS) {
        for (uint32_t i = 0; i < sockets_.size(); i++) {
            delete sockets_[i];
        }
        processors_.clear();
        sockets_.clear();
        esmi_exit();
    }
#endif
    if (init_flag_ & AMDSMI_INIT_AMD_GPUS) {
        // we do not need to delete the processors, deleting sockets takes care of this
        if (!processors_.empty()) {processors_.clear();}
        for (uint32_t i = 0; i < sockets_.size(); i++) {
            delete sockets_[i];
        }
        if (!sockets_.empty()) {sockets_.clear();}
        drm_.cleanup();
        rsmi_status_t ret = rsmi_shut_down();
        if (ret != RSMI_STATUS_SUCCESS) {
            return amd::smi::rsmi_to_amdsmi_status(ret);
        }
    }
#ifdef BRCM_NIC
    if (init_flag_ & AMDSMI_INIT_AMD_NICS) {
        smi_nic_destroy_context(ainic_ctx_);
    }
#endif
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiSystem::handle_to_socket(
            amdsmi_socket_handle socket_handle,
            AMDSmiSocket** socket) {
    if (socket_handle == nullptr || socket == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    *socket = static_cast<AMDSmiSocket*>(socket_handle);

    // double check handlers is here
    if (std::find(sockets_.begin(), sockets_.end(), *socket)
                != sockets_.end()) {
        return AMDSMI_STATUS_SUCCESS;
    }
    return AMDSMI_STATUS_INVAL;
    }

amdsmi_status_t AMDSmiSystem::handle_to_processor(
            amdsmi_processor_handle processor_handle,
            AMDSmiProcessor** processor) {
    if (processor_handle == nullptr || processor == nullptr) {
        return AMDSMI_STATUS_INVAL;
    }
    *processor = static_cast<AMDSmiProcessor*>(processor_handle);

    // double check handlers is here
    if (std::find(processors_.begin(), processors_.end(), *processor)
            != processors_.end()) {
        return AMDSMI_STATUS_SUCCESS;
    }
    if (std::find(nic_processors_.begin(), nic_processors_.end(), *processor)
            != nic_processors_.end()) {
        return AMDSMI_STATUS_SUCCESS;
    }
    if (std::find(switch_processors_.begin(), switch_processors_.end(), *processor) !=
        switch_processors_.end()) {
        return AMDSMI_STATUS_SUCCESS;
    }
    if (std::find(ainic_processors_.begin(), ainic_processors_.end(), *processor)
            != ainic_processors_.end()) {
        return AMDSMI_STATUS_SUCCESS;
    }
    return AMDSMI_STATUS_NOT_FOUND;
}

amdsmi_status_t AMDSmiSystem::gpu_index_to_handle(uint32_t gpu_index,
                    amdsmi_processor_handle* processor_handle) {
    if (processor_handle == nullptr)
        return AMDSMI_STATUS_INVAL;

    auto iter = processors_.begin();
    for (; iter != processors_.end(); iter++) {
        auto cur_device = (*iter);
        if (cur_device->get_processor_type() != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
            continue;
        amd::smi::AMDSmiGPUDevice* gpu_device =
                static_cast<amd::smi::AMDSmiGPUDevice*>(cur_device);
        uint32_t cur_gpu_index = gpu_device->get_gpu_id();
        if (gpu_index == cur_gpu_index) {
            *processor_handle = cur_device;
            return AMDSMI_STATUS_SUCCESS;
        }
    }
    return AMDSMI_STATUS_INVAL;
}


} // namespace amd::smi

