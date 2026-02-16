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

#include <sys/types.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <regex>
#include "config/amd_smi_config.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "impl/scoped_fd.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"

namespace amd::smi {


std::string AMDSmiDrm::find_file_in_folder(const std::string& folder,
               const std::string& regex) {
    std::string file_name;
    DIR *drm_dir;
    struct dirent *dir;
    std::regex file_regex(regex);
    drm_dir = opendir(folder.c_str());
    if (drm_dir == nullptr) return file_name;
    std::cmatch m;
    while ((dir = readdir(drm_dir)) != nullptr) {
      if (std::regex_search(dir->d_name, m, file_regex)) {
        file_name = dir->d_name;
        break;
      }
    }
    closedir(drm_dir);
    return file_name;
}

amdsmi_status_t AMDSmiDrm::init() {
    amdsmi_status_t status = lib_loader_.load(LIBDRM_AMDGPU_SONAME);
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    typedef int (*drmCommandWrite_t)(int fd, unsigned long drmCommandIndex,
                                    void *data, unsigned long size);
    drmCommandWrite_t drmCommandWrite = nullptr;

    // load symbol from libdrm
    status = lib_loader_.load_symbol(reinterpret_cast<drmCommandWrite_t *>(&drmCommandWrite),
                                     "drmCommandWrite");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    typedef int (*drmGetDevice_t)(int fd, drmDevicePtr *device);  // drmGetDevice
    typedef void (*drmFreeDevice_t)(drmDevicePtr *device);  // drmFreeDevice

    drmGetDevice_t drm_get_device = nullptr;
    drmFreeDevice_t drm_free_device = nullptr;

    // Define a function pointer for drmGetVersion
    typedef struct _drmVersion* (*drmGetVersion_t)(int fd);  // drmGetVersion
    drmGetVersion_t drm_get_version = nullptr;
    typedef void (*drmFreeVersion_t)(drmVersionPtr version);  // drmFreeVersion
    drmFreeVersion_t drm_free_version = nullptr;

    status = lib_loader_.load_symbol(
        reinterpret_cast<drmGetVersion_t *>(&drm_get_version), "drmGetVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    status = lib_loader_.load_symbol(
        reinterpret_cast<drmGetVersion_t *>(&drm_free_version), "drmFreeVersion");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    status = lib_loader_.load_symbol(
        reinterpret_cast<drmGetDevice_t *>(&drm_get_device), "drmGetDevice");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }
    status = lib_loader_.load_symbol(
        reinterpret_cast<drmFreeDevice_t *>(&drm_free_device), "drmFreeDevice");
    if (status != AMDSMI_STATUS_SUCCESS) {
        return status;
    }

    /* Need to map the /dev/dri/render* file to /sys/class/drm/card*
       The former is for drm fd and the latter is used for rocm-smi gpu index.
       Here it will search the /sys/class/drm/card0/../renderD128
    */
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
    auto devices = smi.devices();

    for (uint32_t i=0; i < devices.size(); i++) {
        auto rocm_smi_device = devices[i];
        drmDevicePtr device;

        const std::string regex("renderD([0-9]+)");
        const std::string renderD_folder = "/sys/class/drm/card"
                    + std::to_string(rocm_smi_device->index()) + "/../";

        // looking for /sys/class/drm/card0/../renderD*
        std::string render_name = find_file_in_folder(renderD_folder, regex);
        drm_paths_.push_back(render_name);

        std::string name = "/dev/dri/" + render_name;
        ScopedFD fd(name.c_str(), O_RDWR | O_CLOEXEC);

        amdsmi_bdf_t bdf;
        if (fd.valid()) {
            auto version = drm_get_version(fd);
            if (drm_get_device(fd, &device) == 0) {
                vendor_id = device->deviceinfo.pci->vendor_id;
                drm_free_device(&device);
            } else {
                drm_free_device(&device);
            }
            drm_free_version(version);
        }

        uint64_t bdfid = 0;
        rsmi_status_t rsmi_ret = rsmi_dev_pci_id_get(i, &bdfid);
        auto [domain, bus, device_id, function] = parse_bdfid(bdfid);
        if (rsmi_ret != RSMI_STATUS_SUCCESS) {
            // Set empty values on error
            bdf = {}; // zero-initialize
        } else {
            bdf.function_number = function & 0x7;
            bdf.device_number = device_id & 0x1f;
            bdf.bus_number = bus & 0xff;
            bdf.domain_number = domain & 0xffffffffffff;
        }
        drm_bdfs_.push_back(bdf);
    }

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::cleanup() {
    if (!drm_paths_.empty()) {drm_paths_.clear();}
    if (!drm_bdfs_.empty()) {drm_bdfs_.clear();}
    lib_loader_.unload();
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::get_bdf_by_index(uint32_t gpu_index, amdsmi_bdf_t *bdf_info) const {
    if (gpu_index + 1 > drm_bdfs_.size()) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
    }
    *bdf_info = drm_bdfs_[gpu_index];
    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::amdgpu_query_cpu_affinity(const std::string &device_path, std::string &cpu_affinity) {
  std::string cpuAffFile = "cpulistaffinity";
  cpu_affinity = smi_brcm_get_value_string(device_path, cpuAffFile);
  
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t AMDSmiDrm::get_drm_path_by_index(uint32_t gpu_index, std::string *drm_path) const {
    if (gpu_index + 1 > drm_paths_.size()) return AMDSMI_STATUS_NOT_SUPPORTED;
    *drm_path = drm_paths_[gpu_index];
    return AMDSMI_STATUS_SUCCESS;
}

std::vector<std::string>& AMDSmiDrm::get_drm_paths() {
    return drm_paths_;
}

bool AMDSmiDrm::check_if_drm_is_supported() {
    return drm_bdfs_.size() > 0;
}

std::vector<amdsmi_bdf_t> AMDSmiDrm::get_bdfs() {
    return drm_bdfs_;
}

uint32_t AMDSmiDrm::get_vendor_id() {
    return vendor_id;
}

} // namespace amd::smi
