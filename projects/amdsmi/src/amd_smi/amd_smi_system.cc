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

#include <dirent.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cerrno>
#include <cstring>
#include "amd_smi/impl/amd_smi_system.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_common.h"
#include "rocm_smi/rocm_smi.h"
#include <map>
#include <algorithm>

namespace amd::smi {


#define  AMD_SMI_INIT_FLAG_RESRV_TEST1 0x800000000000000  //!< Reserved for test

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

    uint64_t domain = (bdfid >> 32) & 0xffffffff;
    /* May need later
    // may need to identify with partition_id in the future as well... TBD
    uint64_t partition_id = (bdfid >> 28) & 0xf;
    */
    uint64_t bus = (bdfid >> 8) & 0xff;
    uint64_t device_id = (bdfid >> 3) & 0x1f;
    /* May need later
    uint64_t function = bdfid & 0x7;
    */

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

