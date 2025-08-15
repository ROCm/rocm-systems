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

#include <memory>
#include <unordered_set>
#include <dirent.h>
#include <sys/types.h>

#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/fdinfo.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

uint32_t AMDSmiGPUDevice::get_gpu_id() const {
    return gpu_id_;
}

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

std::string& AMDSmiGPUDevice::get_gpu_path() {
    return path_;
}

amdsmi_bdf_t AMDSmiGPUDevice::get_bdf() {
    return this->bdf_;
}

uint32_t AMDSmiGPUDevice::get_vendor_id() {
    return vendor_id_;
}

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

    return AMDSMI_STATUS_SUCCESS;
}

pthread_mutex_t* AMDSmiGPUDevice::get_mutex() {
    return amd::smi::GetMutex(gpu_id_);
}

int32_t AMDSmiGPUDevice::get_compute_process_list_impl(GPUComputeProcessList_t& compute_process_list,
                                                       ComputeProcessListType_t list_type)
{

    /**
     *  Clear the compute_process_list before starting.
     */
    compute_process_list.clear();

    /**
     *  The first call to rsmi_compute_process_info_get() to find the number of
     *  rsmi_process_info_t currently running on the system.
     */
    auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
    auto num_running_processes = uint32_t(0);

    status_code = rsmi_compute_process_info_get(nullptr, &num_running_processes);
    if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || (num_running_processes <= 0)) {
        return status_code;
    }

    /**
     *  Make a type safe pointer, then
     *
     * second call to rsmi_compute_process_info_get() g
     *  the allocated rsmi_process_info_t array.
     */
    using RsmiDeviceList_t = uint32_t[];
    using RsmiProcessList_t = rsmi_process_info_t[];
    std::unique_ptr<RsmiProcessList_t> list_all_processes_ptr = std::make_unique<RsmiProcessList_t>(num_running_processes);

    status_code = rsmi_compute_process_info_get(list_all_processes_ptr.get(), &num_running_processes);
    if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
        return status_code;
    }

    if (num_running_processes <= 0) {
        return rsmi_status_t::RSMI_STATUS_SUCCESS; // No processes running
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
    auto get_process_info = [&](const rsmi_process_info_t& rsmi_proc_info, amdsmi_proc_info_t& amdsmi_proc_info) {
        // amdsmi_proc_info_t gets populated with /proc information from gpuvsmi_get_pid_info()

        auto status_code = gpuvsmi_get_pid_info(get_bdf(), rsmi_proc_info.process_id, amdsmi_proc_info);
        // If we cannot get the info from sysfs, save the minimum info
        if (status_code != amdsmi_status_t::AMDSMI_STATUS_SUCCESS) {
            amdsmi_proc_info.pid = rsmi_proc_info.process_id;
            amdsmi_proc_info.memory_usage.vram_mem = rsmi_proc_info.vram_usage;
        }

        // Copy the cu occupancy from rsmi_process_info_t to amdsmi_proc_info_t
        amdsmi_proc_info.cu_occupancy = rsmi_proc_info.cu_occupancy;

        // Safely handle KFD processes to get total memory_usage of the process
        uint64_t kfd_gpu_id = get_kfd_gpu_id();
        std::string kfd_path = "/sys/class/kfd/kfd/proc/" +
                            std::to_string(rsmi_proc_info.process_id) +
                            "/vram_" + std::to_string(kfd_gpu_id);

        // Check if the file exists before attempting to open it
        if (access(kfd_path.c_str(), R_OK) == 0) {
            std::ifstream kfd_file(kfd_path.c_str());
            if (kfd_file.is_open()) {
                std::string line;
                if (std::getline(kfd_file, line)) {
                    try {
                        uint64_t vram_bytes = std::stoull(line);
                        amdsmi_proc_info.mem = vram_bytes; // Already in bytes
                    } catch (const std::exception& e) {
                        // Handle conversion error gracefully
                        std::ostringstream ss;
                        ss << __PRETTY_FUNCTION__ << " | Failed to parse VRAM value from KFD: " << e.what();
                        LOG_DEBUG(ss);
                    }
                }
                kfd_file.close();
            }
        }

        return status_code;
    };

    /**
     *  Devices used by a process.
     */
    auto update_list_by_running_device = [&](rsmi_process_info_t rsmi_proc_info) {
        // Get all devices running this process into list_device_ptr
        auto status_result(true);
        std::unique_ptr<RsmiDeviceList_t> list_device_ptr = std::make_unique<RsmiDeviceList_t>(num_running_devices);
        list_device_allocation_size = num_running_devices;
        auto status_code = rsmi_compute_process_gpus_get(rsmi_proc_info.process_id, list_device_ptr.get(), &list_device_allocation_size);
        if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
            status_result = false;
            return status_result;
        }

        for (auto device_idx = uint32_t(0); device_idx < list_device_allocation_size; ++device_idx) {
            // Is this device running this process?
            if (list_device_ptr[device_idx] == get_gpu_id()) {
                std::unordered_set<uint64_t> gpu_set;
                gpu_set.insert(get_kfd_gpu_id());
                GetProcessInfoForPID(rsmi_proc_info.process_id, &rsmi_proc_info, &gpu_set);
                amdsmi_proc_info_t tmp_amdsmi_proc_info{};
                get_process_info(rsmi_proc_info, tmp_amdsmi_proc_info);
                compute_process_list.emplace(rsmi_proc_info.process_id, tmp_amdsmi_proc_info);
           }
        }

        return status_result;
    };


    /**
     *  Transfer/Save the ones linked to this device.
     */
    compute_process_list.clear();
    for (auto process_idx = uint32_t(0); process_idx < num_running_processes; ++process_idx) {
        if (list_type == ComputeProcessListType_t::kAllProcesses ||
            list_type == ComputeProcessListType_t::kAllProcessesOnDevice) {
            if (update_list_by_running_device(list_all_processes_ptr[process_idx])) {
            }
        }
    }

    return status_code;
}

const GPUComputeProcessList_t& AMDSmiGPUDevice::amdgpu_get_compute_process_list(ComputeProcessListType_t list_type)
{
    auto error_code = get_compute_process_list_impl(compute_process_list_, list_type);
    if (error_code) {
        compute_process_list_.clear();
    }

    return compute_process_list_;
}

// Convert `amdsmi_bdf_t` to a PCI BDF string
std::string AMDSmiGPUDevice::bdf_to_string() const {
    std::ostringstream oss;
    oss << std::setfill('0') << std::hex      // Use hexadecimal formatting
        << std::setw(4) << bdf_.domain_number << ":"  // Domain (4 digits)
        << std::setw(2) << static_cast<int>(bdf_.bus_number) << ":"  // Bus (2 digits)
        << std::setw(2) << static_cast<int>(bdf_.device_number) << "."  // Device (2 digits)
        << static_cast<int>(bdf_.function_number);  // Function (1 digit)
    return oss.str();
}

std::vector<uint64_t> AMDSmiGPUDevice::get_bitmask_from_numa_node(int32_t node_id, uint32_t size) const {
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

} // namespace amd::smi
